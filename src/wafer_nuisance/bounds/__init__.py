"""The only bounds this project uses, each with its formula and the standard result behind it.

``hoeffding``: estimation-error interval for every population quantity estimated from a pool.
``result_1``: threshold ERM rates. ``result_2``: VC and Rademacher bounds for the linear filter.
``result_3``: asymptotic learning curve of logistic regression. ``result_4``: certified false-alarm
rate.
"""

from wafer_nuisance.bounds.hoeffding import hoeffding_halfwidth, hoeffding_interval
from wafer_nuisance.bounds.result_1 import agnostic_bound, realizable_bound
from wafer_nuisance.bounds.result_2 import (
    clipped_logistic_loss,
    rademacher_bound,
    rademacher_bound_clipped,
    rademacher_complexity_bound,
    vc_bound,
)
from wafer_nuisance.bounds.result_3 import (
    misspecification_ratio,
    predicted_excess,
    predicted_improvement,
    sandwich,
    trace_HinvG,
)
from wafer_nuisance.bounds.result_4 import (
    certification_margin,
    certified_far,
    select_threshold,
    violation_rate_bound,
)

__all__ = ["agnostic_bound", "certification_margin", "certified_far", "clipped_logistic_loss",
           "hoeffding_halfwidth", "hoeffding_interval", "misspecification_ratio",
           "predicted_excess", "predicted_improvement", "rademacher_bound",
           "rademacher_bound_clipped", "rademacher_complexity_bound", "realizable_bound",
           "sandwich", "select_threshold", "trace_HinvG", "vc_bound", "violation_rate_bound"]
