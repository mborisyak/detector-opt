"""Design trainers: a shared base (:mod:`.common`) plus three strategies.

* :class:`DesignTrainer` -- per-design, data-growing convergence;
* :class:`ContinualTrainer` -- one persistent network across designs, with replay;
* :class:`FullBudgetTrainer` -- one design, full budget up front, fixed epochs.

All are **design-conditioned**: ``combine`` always sees each event's true scaled
design (no design scramble).
"""

from .common import TrainResult, Trainer
from .continual import ContinualTrainer
from .design import DesignTrainer
from .full_budget import FullBudgetTrainer

__all__ = ["Trainer", "DesignTrainer", "ContinualTrainer", "FullBudgetTrainer", "TrainResult"]
