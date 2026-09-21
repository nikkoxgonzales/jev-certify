# jev-certify

**Turning Jev's calibrated probabilities into claims that survive an audit: a finite-sample bound on how many queries get routed wrong, plus a way to check the bound in production with a few hundred hand labels instead of all of them.**

## Headline numbers

- With 460 calibration queries and a target of **5% silently misrouted queries per incoming request**, conformal risk control picks a confidence threshold of **0.831** and certifies an expected per-query loss of **0.0499**.
- On 400 held-out human-labelled queries that threshold auto-routes **84.75%** of traffic, of which **2.65%** are wrong (measured per-query loss 0.0225 — the certificate held: True).
- The scope guardrail (`noul` ≥ 0.671) certifies at most **0.0065** out-of-scope queries accepted per incoming request: it passes **82.00%** of in-scope traffic while accepting **1.67%** of the human out-of-scope test set. The looser targets below are prevalence-sensitive — see §3, which is the section that matters most here.
- Auditing the live policy with **25 hand labels** out of 550 decisions: PPI++ gives a 95% interval of width **0.1017** against **0.2696** from labels alone (62.27% narrower).
- Cost of a decision, measured from the provider's own usage records: **$0.1455 per 1,000 queries**, p50 latency **0.37s**, p95 **0.744s**.

## Setup

| Item | Value |
|---|---|
| Model | typesafe/jev-1.13 (OpenRouter Decisions API) |
| Task data | CLINC150 (Larson et al., EMNLP 2019) human test + out-of-scope splits |
| Deployed intents | 150 |
| Calibration / evaluation queries | 460 / 400 |
| Target risk (alpha) | 5% |
| Jev top-1 routing accuracy (held out) | 93.00% |
| …on in-scope calibration queries | 91.50% |
| …on the calibration mix as sampled (includes the gate's out-of-scope rows, which no router can get right) | 79.57% |
| Decisions journalled | 1310 |

## 1. Certified routing threshold

Conformal risk control (Angelopoulos et al., 2022) on the loss `1{auto-routed} x 1{wrong}`. The bound is on **expected silently misrouted queries per incoming request** — a per-query rate, not an accuracy on the queries that happened to be routed.

| Target | Threshold | Certified loss/query | Coverage | Error when routed | Measured loss/query | Held? |
|---|---|---|---|---|---|---|
| 1% | infeasible | n/a | n/a | n/a | n/a | no |
| 2% | 0.971 | 0.0195 | 73.00% | 0.68% | 0.0050 | yes |
| 5% | 0.831 | 0.0499 | 84.75% | 2.65% | 0.0225 | yes |
| 10% | 0.657 | 0.0998 | 92.50% | 3.78% | 0.0350 | yes |

**A target of 1% is not attainable at this sample size.** The tightest certificate 460 calibration queries support is 0.0195 per query, because two things set this floor: 1/(n+1) with n = 460 is 0.0022, and the model reports a top probability of 1.0 on 234 calibration queries, of which the wrong ones keep the loss above zero no matter how strict the threshold gets -- collect more traffic and re-check the rounding.
This is the resolution floor of the certificate, and it is a property of the data, not of the method: the bound cannot be tightened by choosing a cleverer threshold, only by collecting more calibration traffic (or by the model not returning exactly 1.0 on queries it gets wrong).

## 2. What Jev is choosing between (conformal prediction sets)

A `choice` answer is a distribution, and its top label is only one number. Conformal prediction sets turn that distribution into a set that provably contains the truth with probability ≥ 95.00% (Romano et al., 2020 for APS; Sadinle et al., 2019 for the margin score). Set size is the honest measure of how much ambiguity is left.

| Score | Calibrated threshold | Coverage | Mean set size | 95% CI on coverage | Top-1 fallbacks |
|---|---|---|---|---|---|
| APS | 1.0000 | 99.75% | 148.54 | [98.60%, 99.96%] | 0 |
| LAC | 1.0000 | 100.00% | 150.00 | [99.05%, 100.00%] | 0 |

**Set-valued routing is not usable on this task.** LAC sets average 150.0 labels at 100.00% coverage: with a 150-label taxonomy and a peaked distribution, guaranteeing the truth is in the set means admitting most of the taxonomy, which is not a routing decision. Set-valued prediction earns its keep on small label spaces or over retrieval-style candidate shortlists; on a 150-way router the threshold certificate above is the part that buys something. Kept in the report because a method that is measured and then dropped is worth more than one that was never tried.

## 3. Scope guardrail

A second, independent question rides along in the same call: *does the deployed taxonomy cover this query at all?* Its loss is `1{accepted} x 1{not covered}`, so the certificate bounds the share of traffic the gate silently lets through.

| Target | Threshold on noul | Certified false-accepts/query | In-scope passed | OOS accepted | Error among accepted | Held? |
|---|---|---|---|---|---|---|
| 1% | 0.671 | 0.0065 | 82.00% | 1.67% | 1.50% | yes |
| 2% | 0.221 | 0.0195 | 97.00% | 18.33% | 12.42% | no |
| 5% | 0.111 | 0.0456 | 99.25% | 38.67% | 22.61% | no |
| 10% | 0.071 | 0.0998 | 100.00% | 77.33% | 36.71% | no |

The gate's per-query loss counts one failure per *incoming query*, so it scales with how much of the traffic the taxonomy does not cover. The calibration mix was 13.04% uncovered; the evaluation mix was 42.86%. Re-scoring the same thresholds on a mix whose uncovered share matches the calibration mix isolates the effect:

| Target | Certified | Loss/query (as-sampled mix) | Held? | Loss/query (matched mix) | Held? |
|---|---|---|---|---|---|
| 1% | 0.0065 | 0.0071 | yes | 0.0043 | yes |
| 2% | 0.0195 | 0.0786 | **no** | 0.0217 | **no** |
| 5% | 0.0456 | 0.1657 | **no** | 0.0522 | **no** |
| 10% | 0.0998 | 0.3314 | **no** | 0.1109 | **no** |

**Prevalence is a first-class input to a per-query certificate, not a detail.** The as-sampled mix carries a factor of 3.3x more uncovered traffic than the calibration mix, and the per-query loss moves with it. Two fixes exist and both are the caller's job: calibrate on a mix that matches production, or certify the *conditional* rate on accepted queries and multiply by the prevalence your monitoring measures.

One caveat on the matched-mix column: the promise is on the *expected* per-query loss, so a single held-out draw may land above it. The loose target (10%) sits 0.8 standard errors above its certified value — treat that as sampling noise, not as a broken certificate. The as-sampled column is the systematic failure: it misses by a factor, not by a fraction of a standard error.

## 4. Calibration of the probabilities themselves

The whole stack leans on Jev's numbers meaning something. Measured on the calibration split, against human labels:

| Quantity | Brier | ECE (10 bins) |
|---|---|---|
| Scope question (noul) | 0.0836 | 0.1688 |
| Routing confidence (max probability vs. correctness) | 0.1061 | 0.0994 |

## 4b. The resolution floor is the model's, not the statistics'

A per-query risk certificate cannot separate queries the model scores identically. Measured across the calibration and evaluation answers:

| Measurement | Value |
|---|---|
| Distinct top-probabilities returned | 94 |
| …of which are multiples of 0.01 | 60/94 |
| Smallest non-zero top-probability | 0.1500 |
| Share of answers at exactly 1.0 | 56.40% |
| Error rate among those answers | 1.86% |
| Answers at 1.0 that were wrong | 9 |

Shares at 1.0 with errors behind them are the reason a strict threshold cannot drive the measured risk to zero, and therefore the reason a very small alpha is infeasible at a given sample size.

## 5. Auditing the live policy with few labels

Prediction-powered inference (Angelopoulos et al., *Science*, 2023). Each repeat draws a random hand-labelling budget from a pool of 550 decisions; `f` is Jev's own estimate that the query was answered correctly by the certified policy, `y` is the human label.

| Hand labels | PPI / classical CI coverage | PPI width | Classical width | Narrower by | Fewer labels for same width |
|---|---|---|---|---|---|
| 25 | 0.966 / 0.894 | 0.1017 | 0.2696 | 62.27% | 7.025x |
| 50 | 0.996 / 0.944 | 0.0880 | 0.1931 | 54.40% | 4.81x |
| 100 | 0.994 / 0.946 | 0.0792 | 0.1397 | 43.27% | 3.108x |
| 200 | 0.998 / 0.988 | 0.0781 | 0.0992 | 21.33% | 1.616x |

Truth for the pool (finite-population mean of the policy's per-query success): **0.8509**, with **86.91%** of traffic auto-routed at the certified threshold.

**How many labels to reach a tighter audit?**

| 95% interval half-width | Labels needed (labels only) | Labels needed (PPI++) | Tuned lambda |
|---|---|---|---|
| ±0.05 | 254 | 40 | 0.997 |
| ±0.02 | inf | inf | 0.997 |
| ±0.01 | inf | inf | 0.997 |

labels needed to reach a given 95% interval half-width. Infinite means no amount of hand labelling gets there: the term lambda^2 Var(f)/N is set by how much traffic the policy has served, so the pool has to grow, not the labelling budget.

## 6. Where the guarantee breaks

Conformal guarantees rest on exchangeability: calibration and production traffic must be draws from the same distribution. The table below applies the *in-distribution* certificate to traffic it was never calibrated on. This is the most important table in the report, because it is the one that says what the certificate does not buy you.

| Traffic | n | Intent accuracy | Auto-routed | Loss/query | Certificate held? | Gate false-accepts/query |
|---|---|---|---|---|---|---|
| evaluation exchangeable | 400 | 93.00% | 84.75% | 0.0225 | yes | 0.00% |
| in taxonomy exchangeable | 150 | 95.33% | 92.67% | 0.0067 | yes | 0.00% |
| out of scope traffic | 300 | 0.00% | 23.00% | 0.2300 | **no** | 38.67% |

### Same procedure, shifted traffic

A second deployment sends only **20 intents**, calibrated on that taxonomy, and then receives real traffic containing the other 130 intents. Same model, same code, same α — only the traffic changes.

| Slice | n | Threshold | Loss/query | Certificate held? |
|---|---|---|---|---|
| calibration (in taxonomy) | 400 | 0.000 | 0.0175 | — |
| in taxonomy exchangeable | 200 | 0.000 | 0.0350 | yes |
| unlisted intents traffic | 500 | 0.000 | 1.0000 | **no** |

The restricted deployment's scope gate could not be calibrated at all: its calibration pool contained no traffic outside the 20 intents, so the gate has never seen a query it should reject. That is a deployment hazard the certificate cannot fix — sampling your traffic mix is a prerequisite for auditing it.

## 7. Cost

| Configuration | Threshold | Auto-routed | Loss/query | Certified | Cost / 1k |
|---|---|---|---|---|---|
| certified risk <= 5% | 0.831 | 84.75% | 0.0225 | 0.0499 | $0.79 |
| saturated threshold 0.999 (not certified) | 0.999 | 62.75% | 0.0025 | — | $1.71 |
| hand-picked threshold 0.5 | 0.500 | 98.50% | 0.0600 | — | $0.21 |
| hand-picked threshold 0.7 | 0.700 | 91.25% | 0.0350 | — | $0.51 |
| hand-picked threshold 0.9 | 0.900 | 82.25% | 0.0175 | — | $0.89 |
| hand-picked threshold 0.99 | 0.990 | 70.50% | 0.0050 | — | $1.38 |
| always LLM (no router) | — | 100.00% | n/a | — | $4.20 |

Assumptions beyond Jev's measured cost ($0.1455 / 1k): escalation 4.2 USD / 1k, human review 0.0 USD / 1k.

## Examples from the evaluation set

**Confident and right**

- `can you tell me why my card got declined` → **card_declined** (p = 1.0)
- `what's the exhange rate between mxn and gbp` → **exchange_rate** (p = 1.0)
- `i want to freeze my bank account` → **freeze_account** (p = 1.0)
- `i would like to know the timezone for britain` → **timezone** (p = 1.0)
- `what should i do if i need to report my card lost` → **report_lost_card** (p = 1.0)

**Confident and wrong** (what the certificate is paying for)

- `what will tomorrow be on the calendar` → **calendar** (p = 1.0) but truth is **date** (runner-up accept_reservations @ 0.000)
- `did someone who is unauthorized try to get into my bank account` → **report_fraud** (p = 0.99) but truth is **account_blocked** (runner-up account_blocked @ 0.010)
- `i don't want to forget to call mom` → **reminder** (p = 0.97) but truth is **reminder_update** (runner-up todo_list @ 0.020)
- `hush` → **whisper_mode** (p = 0.96) but truth is **cancel** (runner-up change_volume @ 0.030)
- `can i hurt my credit rating if i open several credit cards in a short time frame` → **credit_score** (p = 0.92) but truth is **improve_credit_score** (runner-up improve_credit_score @ 0.060)

## Limits

- **Exchangeability is the assumption, and it is not free.** The drift section measures it breaking. A certificate printed for one traffic distribution says nothing about another; re-calibrate on a fresh sample whenever the mix moves.
- **The routing certificate is per-query, not per-routed-query.** It bounds silently misrouted traffic, not the error rate among routed queries; both columns are reported because teams ask for different ones.
- **Calibration split size sets the resolution.** With n calibration points the smallest certifiable bound is about 1/(n+1); more traffic buys a tighter certificate, not a better model.
- **The cost model contains one assumption** (the price of the escalated fallback). Jev's own cost, latency and token counts are measured, not assumed.
- **CLINC150 is a research test set**, not your traffic. The method transfers; the numbers are this dataset's.
- **Two questions rode in every call**, so the choice and the gate share a request. They are evaluated independently, but the gate's accuracy is not independent evidence about the router's.

## Reproduce

```bash
export OPENROUTER_API_KEY=sk-or-...        # https://openrouter.ai/keys
python -m experiments.collect              # ~2,400 Jev decisions, journals everything
python -m experiments.analyse              # writes results/results.json + this report
python -m pytest tests -q                  # math checks, no API calls
```

## References

- Angelopoulos, Bates, Fisch, Lei & Schuster. *Conformal Risk Control.* ICLR 2023 (arXiv:2208.02814).
- Angelopoulos, Bates, Fannjiang, Jordan & Zrnic. *Prediction-Powered Inference.* Science 382:669-674, 2023.
- Angelopoulos, Bates, Fannjiang, Jordan & Zrnic. *PPI++: Efficient Prediction-Powered Inference.* NeurIPS 2023 (arXiv:2311.01453).
- Romano, Sesia & Candès. *Classification with Valid and Adaptive Coverage.* NeurIPS 2020 (arXiv:2006.02544).
- Sadinle, Lei & Wasserman. *Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.* JASA 2019.
- Geifman & El-Yaniv. *Selective Classification for Deep Neural Networks.* NeurIPS 2017.
- Guo, Pleiss, Sun & Weinberger. *On Calibration of Modern Neural Networks.* ICML 2017.
- Larson et al. *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction.* EMNLP 2019.
- TypeSafe AI. *System One* and *Confidence* documentation, and the *Intent routing* pattern (docs.typesafe.ai).
