"""jev-certify: certified abstention and prediction-powered audits for Jev.

Jev (TypeSafe AI's System One model) answers typed questions with calibrated
probabilities. Calibration is not a guarantee, and a threshold picked by taste has no
idea what its own error rate is. This package turns Jev's answers into claims that
survive scrutiny:

* :mod:`jev_certify.conformal` -- split conformal prediction sets and conformal risk
  control: a finite-sample bound on the share of traffic auto-routed to the wrong
  destination.
* :mod:`jev_certify.ppi` -- prediction-powered inference: audit the deployed policy
  with a small random sample of hand labels instead of all of it.
* :mod:`jev_certify.cascade` -- price the certified operating point against
  always-escalate and against a hand-picked threshold.
* :mod:`jev_certify.client` -- journal-backed client for the OpenRouter Decisions API
  (content-addressed, so re-analysis never re-bills).
* :mod:`jev_certify.tasks` -- the CLINC150 router/guardrail task with human labels.
"""

from .cascade import CostAssumptions, build_operating_points
from .client import Decision, DecisionJournal, JevClient, JevError, MissingApiKey, decision_key
from .conformal import (
    CertifiedThreshold,
    certified_threshold,
    conformal_quantile,
    coverage_of_sets,
    prediction_set,
    risk_coverage_curve,
    selective_report,
)
from .ppi import audit_simulation, expected_calibration_error, ppi_estimate, reliability_bins

__version__ = "0.1.0"

__all__ = [
    "CostAssumptions",
    "CertifiedThreshold",
    "Decision",
    "DecisionJournal",
    "JevClient",
    "JevError",
    "MissingApiKey",
    "audit_simulation",
    "build_operating_points",
    "certified_threshold",
    "conformal_quantile",
    "coverage_of_sets",
    "decision_key",
    "expected_calibration_error",
    "prediction_set",
    "ppi_estimate",
    "reliability_bins",
    "risk_coverage_curve",
    "selective_report",
]
