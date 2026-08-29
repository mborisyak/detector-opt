"""Design trainers: a shared base (:mod:`.common`) plus four strategies.

* :class:`DesignTrainer` -- per-design, data-growing convergence;
* :class:`ContinualTrainer` -- one persistent network across designs, with replay;
* :class:`ContinualRatioTrainer` -- the same, at a configurable current:replay batch ratio;
* :class:`FullBudgetTrainer` -- one design, full budget up front, fixed epochs.

``combine`` always sees each event's TRUE design in every strategy (no design
scramble). Whether the NETWORK is told it is a separate, per-strategy answer --
:meth:`Trainer.reveals_design` -- and the strategies disagree: the continual pair
reveal it, because one network spans many designs and replay mixes them in a
batch; the per-design pair withhold it, because a fresh network sees one design at
a time and the design is constant across its whole batch. The regressor is built
for whichever layout its strategy will feed it.
"""

from .common import REVEAL, TrainResult, Trainer
from .continual import ContinualReinitTrainer, ContinualTrainer
from .continual_random import ContinualRandomFrozenTrainer, ContinualRandomOnlineTrainer
from .continual_ratio import ContinualRatioTrainer
from .design import DesignTrainer
from .fixed_window import FixedWindowTrainer
from .full_budget import FullBudgetTrainer

__all__ = [
  'REVEAL', "Trainer", "DesignTrainer", "ContinualTrainer", "ContinualReinitTrainer", "ContinualRatioTrainer",
  "ContinualRandomFrozenTrainer", "ContinualRandomOnlineTrainer", "FullBudgetTrainer", "FixedWindowTrainer", "TrainResult"
]
