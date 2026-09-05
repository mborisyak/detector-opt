"""The derived-sigma shrink-and-perturb (`param_noise = sqrt(1 - shrink^2)`) is retired: a `shrink`
below 1 without an explicit `param_noise` is refused at construction, while the independent-sigma
form and the no-op default still construct."""

import pytest

from analytic import analytic_detector
from detopt.nn.trainer import DesignTrainer


def _trainer(**retention):
  import optax

  return DesignTrainer(
    analytic_detector(), regressor_config={"set-regressor": {
      "features": [[16, 16]],
      "p_dropout": 0.1
    }}, optimizer=optax.adamw(learning_rate=1e-3,
                              weight_decay=1e-3), batch=64, n0=256, n_increment=128, iteration_limit=512, warmup_epochs=2,
    patience=3, loss_precision=0.5, budget=20_000, val_fraction=0.25, eval_batch=128, device=None, seed=0, **retention,
  )


def test_shrink_without_param_noise_is_refused():
  with pytest.raises(ValueError, match="retired"):
    _trainer(shrink=0.3)


def test_shrink_with_explicit_param_noise_constructs():
  trainer = _trainer(shrink=0.3, param_noise=0.01)
  assert trainer.shrink == 0.3
  assert trainer.param_noise == 0.01


def test_no_shrink_constructs_without_param_noise():
  trainer = _trainer()
  assert trainer.shrink == 1.0
  assert trainer.param_noise == 0.0


def test_param_noise_without_shrink_is_refused():
  with pytest.raises(ValueError, match="no effect at shrink = 1.0"):
    _trainer(param_noise=0.01)
