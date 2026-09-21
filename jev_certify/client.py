"""Client for Jev (TypeSafe System One) over OpenRouter's Decisions API.

Jev is not a chat model: it does not generate text. You send a ``state`` plus a map
of *typed questions* and get back typed answers with calibrated probabilities, plus a
token/cost record. It is reached through OpenRouter's Decisions endpoint
(``POST https://openrouter.ai/api/alpha/decisions``), *not* the OpenAI-compatible
chat endpoint -- chat SDKs will not work against it.

Two properties of this client matter for the rest of the package:

* **Every call is journalled** to an append-only JSONL file. Calibration, evaluation
  and reporting all read that journal, so an experiment can be re-analysed forever
  without re-billing the API.
* **Every call is content-addressed** by ``sha256(model, state, questions)``. A key
  already present in the journal is served from it, which makes re-runs free and
  makes the CLI's ``--offline`` flag possible.

Zero third-party dependencies (stdlib only) so the client can be dropped into any
codebase.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "typesafe/jev-1.13"
DEFAULT_REFERER = "https://github.com/nikko/jev-certify"
DEFAULT_TITLE = "jev-certify"


class JevError(RuntimeError):
    """Raised when the Decisions API cannot be reached or returns an error."""


class MissingApiKey(JevError):
    """Raised when no OpenRouter key can be found."""


def load_api_key(explicit: str | None = None, env_var: str = "OPENROUTER_API_KEY") -> str:
    """Find an OpenRouter key: argument, environment, then a ``.env`` file.

    The key is never logged or stored by this package; it is only put into the
    ``Authorization`` header of the request.
    """
    if explicit:
        return explicit.strip()
    if os.environ.get(env_var):
        return os.environ[env_var].strip()
    for parent in [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent.parent]:
        candidate = parent / ".env"
        if candidate.is_file():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                if name.strip() == env_var:
                    return value.strip().strip('"').strip("'")
    raise MissingApiKey(
        f"No {env_var} found. Export it, or put `{env_var}=sk-or-...` in a .env file "
        "in the project root. Create a key at https://openrouter.ai/keys"
    )


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def decision_key(model: str, state: Any, questions: Any) -> str:
    """Stable content hash of a decision request -- the journal's primary key."""
    payload = _canonical({"model": model, "state": state, "questions": questions})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclass
class Decision:
    """One recorded Jev request/response pair."""

    key: str
    model: str
    state: Any
    questions: dict[str, Any]
    answers: dict[str, Any]
    usage: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    fetched_at: str = ""
    served_from: str = "api"  # "api" | "journal"

    @property
    def input_tokens(self) -> int:
        return int(self.usage.get("input_tokens", 0) or 0)

    @property
    def output_tokens(self) -> int:
        return int(self.usage.get("output_tokens", 0) or 0)

    @property
    def cost_usd(self) -> float:
        """Cost as reported by the provider (0.0 when replayed from the journal)."""
        return float(self.usage.get("cost", 0.0) or 0.0)

    def to_record(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "model": self.model,
            "state": self.state,
            "questions": self.questions,
            "answers": self.answers,
            "usage": self.usage,
            "latency_s": round(self.latency_s, 4),
            "fetched_at": self.fetched_at,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Decision":
        # `state` and `questions` are optional: a slim journal (see `pack-journal`) keeps the
        # answer, usage and latency but not the request payload, because the caller rebuilds
        # the payload from the plan and checks its hash against `key` before using it.
        return cls(
            key=record["key"],
            model=record["model"],
            state=record.get("state", ""),
            questions=record.get("questions", {}),
            answers=record["answers"],
            usage=record.get("usage", {}),
            latency_s=record.get("latency_s", 0.0),
            fetched_at=record.get("fetched_at", ""),
            served_from="journal",
        )


class DecisionJournal:
    """Append-only JSONL store of every Jev decision (content-addressed)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._index: dict[str, dict[str, Any]] | None = None

    def load(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            if self._index is None:
                index: dict[str, dict[str, Any]] = {}
                if self.path.is_file():
                    for line in self.path.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue  # tolerate a torn tail line
                        index[record["key"]] = record
                self._index = index
            return self._index

    def get(self, key: str) -> dict[str, Any] | None:
        return self.load().get(key)

    def append(self, decision: Decision) -> None:
        record = decision.to_record()
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(_canonical(record) + "\n")
            if self._index is not None:
                self._index[record["key"]] = record

    def __len__(self) -> int:
        return len(self.load())

    def total_cost_usd(self) -> float:
        return sum(float(r.get("usage", {}).get("cost", 0.0) or 0.0) for r in self.load().values())


class JevClient:
    """Thread-safe Jev client with journalling, content-addressed caching and retries."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        journal: DecisionJournal | str | Path | None = None,
        timeout: float = 90.0,
        max_retries: int = 4,
        referer: str = DEFAULT_REFERER,
        title: str = DEFAULT_TITLE,
        offline: bool = False,
        fresh: bool = False,
    ):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.referer = referer
        self.title = title
        self.offline = offline
        self.fresh = fresh
        self._api_key = None if offline else load_api_key(api_key)
        if journal is None:
            journal = Path(__file__).resolve().parent.parent / "results" / "journal.jsonl"
        self.journal = journal if isinstance(journal, DecisionJournal) else DecisionJournal(journal)

        self._mem: dict[str, Decision] = {}
        self._stats_lock = threading.Lock()
        self.n_api_calls = 0
        self.n_cache_hits = 0
        self.cost_usd = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.latencies: list[float] = []

    # ------------------------------------------------------------------ internals

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = _canonical(payload).encode("utf-8")
        request = urllib.request.Request(
            DECISIONS_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.referer,
                "X-Title": self.title,
                "User-Agent": "jev-certify/0.1",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:  # noqa: PERF203
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")[:400]
                except Exception:  # pragma: no cover - body may be unreadable
                    pass
                last_error = JevError(f"HTTP {exc.code} from Decisions API: {detail}")
                if exc.code not in (408, 409, 425, 429, 500, 502, 503, 504):
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = JevError(f"transport error: {exc!r}")
            if attempt < self.max_retries:
                time.sleep(min(8.0, 0.5 * (2**attempt)) + 0.15 * attempt)
        raise last_error if last_error else JevError("unknown failure")

    # ------------------------------------------------------------------- public API

    def decide(self, state: Any, questions: dict[str, Any]) -> Decision:
        """Ask one Jev request (state + typed questions). Cached by content hash."""
        key = decision_key(self.model, state, questions)
        if not self.fresh:
            if key in self._mem:
                with self._stats_lock:
                    self.n_cache_hits += 1
                hit = self._mem[key]
                hit.served_from = "journal"
                return hit
            record = self.journal.get(key)
            if record is not None:
                decision = Decision.from_record(record)
                decision.state = state  # keep the caller's object identity
                self._mem[key] = decision
                with self._stats_lock:
                    self.n_cache_hits += 1
                return decision
        if self.offline:
            raise JevError(
                "offline mode: this request is not in the journal "
                f"(key={key[:12]}…). Drop --offline to spend a real call."
            )

        payload = {"model": self.model, "state": state, "questions": questions}
        started = time.perf_counter()
        raw = self._post(payload)
        latency = time.perf_counter() - started

        decision = Decision(
            key=key,
            model=raw.get("model", self.model),
            state=state,
            questions=questions,
            answers=raw.get("answers", {}),
            usage=raw.get("usage", {}) or {},
            latency_s=latency,
            fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self.journal.append(decision)
        self._mem[key] = decision
        with self._stats_lock:
            self.n_api_calls += 1
            self.cost_usd += decision.cost_usd
            self.input_tokens += decision.input_tokens
            self.output_tokens += decision.output_tokens
            self.latencies.append(latency)
        return decision

    def decide_many(
        self,
        items: Sequence[tuple[Any, dict[str, Any]]],
        workers: int = 8,
        progress: Callable[[int, int], None] | None = None,
        strict: bool = False,
    ) -> tuple[list[Decision], list[tuple[int, str]]]:
        """Run many decisions concurrently.

        Returns ``(decisions, errors)`` where ``decisions`` preserves the order of
        ``items``. A long collection run should not die because one request out of two
        thousand returned a 520, so by default failures are collected and reported
        instead of raised (``strict=True`` restores raising). Every request that
        succeeded is already in the journal, so a resumed run pays only for the gaps.
        """
        results: list[Decision | None] = [None] * len(items)
        errors: list[tuple[int, str]] = []
        done = 0
        done_lock = threading.Lock()

        def work(index: int, state: Any, questions: dict[str, Any]) -> None:
            nonlocal done
            try:
                results[index] = self.decide(state, questions)
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                if strict:
                    raise
                with done_lock:
                    errors.append((index, f"{type(exc).__name__}: {exc}"))
            finally:
                with done_lock:
                    done += 1
                    if progress is not None:
                        progress(done, len(items))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(work, i, s, q) for i, (s, q) in enumerate(items)]:
                future.result()
        return [r for r in results if r is not None], errors

    def warm_from_journal(self) -> int:
        """Preload every journalled decision into memory. Returns the entry count."""
        for key, record in self.journal.load().items():
            self._mem.setdefault(key, Decision.from_record(record))
        return len(self._mem)

    def stats(self) -> dict[str, Any]:
        lat = sorted(self.latencies)
        def pct(p: float) -> float | None:
            if not lat:
                return None
            idx = min(len(lat) - 1, max(0, int(round(p * (len(lat) - 1)))))
            return round(lat[idx], 4)

        return {
            "api_calls": self.n_api_calls,
            "journal_hits": self.n_cache_hits,
            "cost_usd": round(self.cost_usd, 6),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_p50_s": pct(0.50),
            "latency_p95_s": pct(0.95),
            "journal_entries": len(self.journal),
            "journal_total_cost_usd": round(self.journal.total_cost_usd(), 6),
        }


def answers_of(decisions: Iterable[Decision]) -> list[dict[str, Any]]:
    return [d.answers for d in decisions]
