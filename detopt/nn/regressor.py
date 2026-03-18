import inspect
import math
from multiprocessing import Event
from typing import Sequence

import jax
import jax.nn as jnn
import jax.numpy as jnp
from flax import nnx
from numpy._core.umath import spacing

from ..detector import Detector
from .common import Block, LeakyReLU, LeakyTanh, Model, SiLU, bayes_aggregate

__all__ = [
    "Regressor",
    "MLP",
    "AlphaResNet",
    "CNN",
    "HyperResNet",
    "DeepSet",
    "BayesDeepSet",
]


class Regressor(Model):
    def __call__(self, X: jax.Array, design: jax.Array, *, deterministic: bool = True):
        raise NotImplementedError()


class MLP(Regressor):
    def __init__(
        self,
        detector: Detector,
        features: Sequence[int],
        *,
        layer_norm: bool = False,
        p_dropout: float | None = None,
        rngs: nnx.Rngs,
    ):
        super().__init__(detector, rngs=rngs)
        input_dim, design_dim = (
            math.prod(self.input_shape),
            math.prod(self.design_shape),
        )
        target_dim = math.prod(self.target_shape)

        self.units = (input_dim + design_dim, *features, target_dim)
        self.layer_norm = layer_norm

        self.layers: list[nnx.Module] = list()
        for i, (n_in, n_out) in enumerate(zip(self.units[:-2], self.units[1:-1])):
            if layer_norm:
                self.layers.append(nnx.LayerNorm(n_in, rngs=rngs))
            if p_dropout is not None:
                self.layers.append(nnx.Dropout(rate=p_dropout, rngs=rngs))
            self.layers.append(nnx.Linear(n_in, n_out, rngs=rngs))
            self.layers.append(
                LeakyTanh(
                    n_out,
                )
            )

        *_, n_in, n_out = self.units
        self.layers.append(nnx.Linear(n_in, n_out, rngs=rngs))

    def __call__(self, X: jax.Array, design: jax.Array, deterministic: bool = True):
        n, *_ = X.shape
        X = jnp.reshape(X, shape=(n, -1))
        design = jnp.reshape(design, shape=(n, -1))
        result = jnp.concatenate([X, design], axis=-1)

        for layer in self.layers:
            result = layer(result)

        result = jnp.reshape(result, shape=(result.shape[0], *self.target_shape))
        return result


class AlphaResNet(Regressor):
    def __init__(
        self,
        detector: Detector,
        n_hidden: int,
        depth: int,
        p_dropout: float | None = 0.2,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(detector, rngs=rngs)
        input_dim, design_dim = (
            math.prod(self.input_shape),
            math.prod(self.design_shape),
        )
        target_dim = math.prod(self.target_shape)

        n_in = input_dim + design_dim
        self.embedding = nnx.Linear(n_in, n_hidden, rngs=rngs)

        self.hidden: list[list[nnx.Module]] = list()
        self.alphas: list[nnx.Param[jax.Array]] = list()

        for i in range(depth):
            block: list[nnx.Module] = list()

            block.append(SiLU())
            if p_dropout:
                block.append(nnx.Dropout(n_hidden, rngs=rngs))
            block.append(nnx.Linear(n_hidden, n_hidden, rngs=rngs))
            self.alphas.append(
                nnx.Param(
                    jnp.zeros(shape=(n_hidden,)),
                )
            )

        self.output: list[nnx.Module] = [
            SiLU(),
            nnx.Linear(n_hidden, target_dim, rngs=rngs),
        ]

    def __call__(self, X: jax.Array, design: jax.Array, *, deterministic: bool = True):
        n, *_ = X.shape
        X = jnp.reshape(X, shape=(n, -1))
        design = jnp.reshape(design, shape=(n, -1))

        result = jnp.concatenate([X, design], axis=-1)
        result = self.embedding(result)

        for block, alpha in zip(self.hidden, self.alphas):
            hidden = result
            for layer in block:
                if hasattr(layer, "deterministic"):
                    hidden = layer(hidden, deterministic=deterministic)
                else:
                    hidden = layer(hidden)

            result = result + alpha.value * hidden

        for layer in self.output:
            result = layer(jax.nn.celu(result))

        return jnp.reshape(result, shape=(result.shape[0], *self.target_shape))


class CNN(Regressor):
    def __init__(
        self, detector: Detector, features, p_dropout: float = 0.1, *, rngs: nnx.Rngs
    ):
        super().__init__(detector, rngs=rngs)
        input_dim, design_dim = (
            math.prod(self.input_shape),
            math.prod(self.design_shape),
        )
        target_dim = math.prod(self.target_shape)

        n_d = 4

        n_layers, n_straws = self.input_shape

        self.blocks: list[tuple[Block, Block]] = list()
        n_features = 0
        for n_f in features[:-1]:
            self.blocks.append(
                (
                    Block(
                        nnx.Conv(
                            2 * n_features + n_d,
                            n_f,
                            kernel_size=(1, 1),
                            padding="VALID",
                            rngs=rngs,
                        ),
                        nnx.Dropout(rate=p_dropout, rngs=rngs),
                        nnx.Conv(
                            n_f,
                            n_f,
                            kernel_size=(1, n_straws),
                            feature_group_count=n_f,
                            padding="VALID",
                            rngs=rngs,
                        ),
                        SiLU(),
                    ),
                    Block(
                        nnx.Conv(
                            n_f, n_f, kernel_size=(1, 1), padding="VALID", rngs=rngs
                        ),
                        nnx.Dropout(rate=p_dropout, rngs=rngs),
                        nnx.Conv(
                            n_f,
                            n_f,
                            kernel_size=(n_layers, 1),
                            feature_group_count=n_f,
                            padding="VALID",
                            rngs=rngs,
                        ),
                        SiLU(),
                    ),
                )
            )
            n_features = n_f

        n_f = features[-1]

        self.final_block = Block(
            nnx.Conv(
                2 * n_features + n_d,
                n_f,
                kernel_size=(1, 1),
                padding="VALID",
                rngs=rngs,
            ),
            nnx.Dropout(rate=p_dropout, rngs=rngs),
            nnx.Conv(
                n_f,
                n_f,
                kernel_size=(1, n_straws),
                feature_group_count=n_f,
                padding="VALID",
                rngs=rngs,
            ),
            SiLU(),
            nnx.Conv(n_f, n_f, kernel_size=(1, 1), padding="VALID", rngs=rngs),
            nnx.Dropout(rate=p_dropout, rngs=rngs),
            nnx.Conv(
                n_f,
                n_f,
                kernel_size=(n_layers, 1),
                feature_group_count=n_f,
                padding="VALID",
                rngs=rngs,
            ),
            SiLU(),
        )

        self.output = Block(nnx.Linear(n_f, target_dim, rngs=rngs))

    def convolve(self, X, design, deterministic: bool = True):
        n_b, n_l, n_s = X.shape
        _, n_d = design.shape

        X = jnp.reshape(X, shape=(n_b, n_l, n_s, 1))
        positions, angles, magnetic_strength = (
            design[:, :n_l],
            design[:, n_l : 2 * n_l],
            design[:, -1],
        )
        positions = jnp.broadcast_to(
            positions[:, :, None, None], shape=(n_b, n_l, n_s, 1)
        )
        angles = jnp.broadcast_to(angles[:, :, None, None], shape=(n_b, n_l, n_s, 1))
        magnetic_strength = jnp.broadcast_to(
            magnetic_strength[:, None, None, None], shape=(n_b, n_l, n_s, 1)
        )

        X = jnp.concatenate([X, positions, angles, magnetic_strength], axis=-1)
        hidden = X

        for block_straw_wise, block_layer_wise in self.blocks:
            hidden_straw_wise = block_straw_wise(hidden, deterministic=deterministic)
            hidden_layer_wise = block_layer_wise(
                hidden_straw_wise, deterministic=deterministic
            )

            *_, h_sw_c = hidden_straw_wise.shape
            hidden_straw_wise = jnp.broadcast_to(
                hidden_straw_wise, (n_b, n_l, n_s, h_sw_c)
            )
            *_, h_lw_c = hidden_layer_wise.shape
            hidden_layer_wise = jnp.broadcast_to(
                hidden_layer_wise, (n_b, n_l, n_s, h_lw_c)
            )
            hidden = jnp.concatenate([X, hidden_straw_wise, hidden_layer_wise], axis=-1)

        hidden = self.final_block(hidden, deterministic=deterministic)
        _, n_x, n_y, _ = hidden.shape
        assert n_x == n_y == 1

        hidden = jnp.mean(hidden, axis=(1, 2))

        return hidden

    def __call__(self, X: jax.Array, design: jax.Array, *, deterministic: bool = True):
        result = self.convolve(X, design, deterministic=deterministic)
        result = self.output(result)

        return jnp.reshape(result, shape=(result.shape[0], *self.target_shape))


class HyperResNet(Regressor):
    def __init__(
        self,
        detector: Detector,
        n_hidden: int,
        depth: int,
        p_dropout: float | None = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(detector, rngs=rngs)
        input_dim, design_dim = (
            math.prod(self.input_shape),
            math.prod(self.design_shape),
        )
        target_dim = math.prod(self.target_shape)

        dropout = lambda: (
            [] if p_dropout is None else [nnx.Dropout(rate=p_dropout, rngs=rngs)]
        )
        self.initial_embeddings = (
            nnx.Linear(input_dim, n_hidden, rngs=rngs),
            nnx.Linear(design_dim, n_hidden, rngs=rngs),
        )
        self.initial_block = Block(
            SiLU(),
            *dropout(),
            nnx.Linear(n_hidden, n_hidden, rngs=rngs),
        )

        self.embeddings = list()
        self.blocks = list()
        self.alphas = list()

        for i in range(depth):
            self.embeddings.append(
                (
                    nnx.Linear(input_dim, n_hidden, rngs=rngs),
                    nnx.Linear(design_dim, n_hidden, rngs=rngs),
                    nnx.Linear(n_hidden, n_hidden, rngs=rngs),
                )
            )
            self.blocks.append(
                Block(
                    SiLU(),
                    *dropout(),
                    nnx.Linear(n_hidden, n_hidden, rngs=rngs),
                )
            )
            self.alphas.append(
                nnx.Param(
                    jnp.zeros(shape=()),
                )
            )

        self.output = nnx.Linear(n_hidden, target_dim, rngs=rngs)

    def __call__(self, X: jax.Array, design: jax.Array, *, deterministic: bool = True):
        n, *_ = X.shape
        X = jnp.reshape(X, shape=(n, -1))
        design = jnp.reshape(design, shape=(n, -1))

        initial_emb_input, initial_emb_design = self.initial_embeddings
        X_emb = initial_emb_input(X)
        design_emb = initial_emb_design(design)

        latent = self.initial_block(X_emb + design_emb, deterministic=deterministic)

        for embds, block, alpha in zip(self.embeddings, self.blocks, self.alphas):
            emb_input, emb_design, emb_latent = embds
            delta = block(
                emb_input(X) + emb_design(design) + emb_latent(latent),
                deterministic=deterministic,
            )
            latent = latent + alpha * delta

        result = self.output(latent)
        return jnp.reshape(result, shape=(result.shape[0], *self.target_shape))


class DeepSet(Regressor):
    def __init__(
        self,
        detector: Detector,
        features: Sequence[Sequence[int]],
        p_dropout: float = 0.1,
        n_max_hits: int = 200,  # ADDED: Maximum number of hits per event for padding
        *,
        rngs: nnx.Rngs,
    ):

        self.input_shape = None  # Sparse data has variable length, no fixed input shape
        self.design_shape = detector.design_shape()
        self.target_shape = detector.target_shape()
        self.ground_truth_shape = detector.ground_truth_shape()
        self.rngs = rngs

        print(f"DeepSet input_shape: {self.input_shape} (variable - sparse data)")
        print(f"DeepSet design_shape: {self.design_shape}")
        print(f"DeepSet target_shape: {self.target_shape}")
        target_dim = math.prod(self.target_shape)

        # Input feature normalization statistics (estimated from data)
        # Feature order: [station, view, layer, straw, tdc_value, position, angle, B]
        self.tdc_mean = 440.0  # ns (typical TDC time)
        self.tdc_std = 80.0  # ns (typical spread)
        self.position_mean = 8950.0  # cm (middle of detector)
        self.position_std = 500.0  # cm (layer spread)
        self.angle_mean = 0.0  # radians
        self.angle_std = 0.5  # radians (max stereo angle)
        self.B_mean = 5  # T (typical field)
        self.B_std = 10  # T (field range)

        # ADDED: Store detector geometry - use detector's parameters instead of hardcoding
        self.n_max_hits = n_max_hits
        self.n_stations = detector.n_stations  # From StrawDetector.n_stations
        self.n_views_per_station = (
            detector.n_views_per_station
        )  # From StrawDetector.n_views_per_station
        self.n_layers_per_view = (
            detector.n_layers_per_view
        )  # From StrawDetector.n_layers_per_view
        self.n_straws = detector.n_straws  # From StrawDetector.n_straws

        # CHANGED: Feature dimension is now 8 per hit (4 indices + TDC + position + angle + B)
        n_hit_features = 8
        # Build blocks using nnx.List to properly track modules in Flax NNX
        all_blocks = []

        n_features = n_hit_features  # Start with 8 features per hit
        for block_def in features:
            units = (n_features, *block_def)
            layers = []
            # Build intermediate layers with activation
            for n_in, n_out in zip(units[:-2], units[1:-1]):
                if p_dropout is not None and p_dropout > 0:
                    layers.append(nnx.Dropout(rate=p_dropout, rngs=rngs))
                layers.append(nnx.Linear(n_in, n_out, rngs=rngs))
                layers.append(LeakyTanh(n_out))
            # Final layer without activation
            if p_dropout is not None and p_dropout > 0:
                layers.append(nnx.Dropout(rate=p_dropout, rngs=rngs))
            layers.append(nnx.Linear(units[-2], units[-1], rngs=rngs))

            all_blocks.append(nnx.List(layers))  # Wrap each block in nnx.List
            n_features = 2 * units[-1]  # After aggregation: [local, global]

        self.blocks = nnx.List(all_blocks)  # Wrap outer list in nnx.List too

        *_, last = features
        *_, n_latent = last

        self.output = nnx.Linear(n_latent, target_dim, rngs=rngs)

    def combine(self, info, design):
        events, layers, straws, times, mask = info

        mask_bool = jnp.asarray(mask, dtype=jnp.bool_)
        hit_mask = mask_bool.astype(jnp.float32)[:, None]  # (n_hits, 1)

        n_batch = design.shape[0]
        n_layers_total = (
            self.n_stations * self.n_views_per_station * self.n_layers_per_view
        )

        # Make padded entries safe for indexing/segment ops
        events = jnp.asarray(events, dtype=jnp.int32)
        layers = jnp.asarray(layers, dtype=jnp.int32)
        straws = jnp.asarray(straws, dtype=jnp.int32)
        values = jnp.asarray(times, dtype=jnp.float32)

        event_indices = jnp.where(mask_bool, events, 0)
        layer_indices = jnp.where(mask_bool, layers, 0)
        straw_indices = jnp.where(mask_bool, straws, 0)

        # Extra safety in case upstream padding uses bad sentinel values
        event_indices = jnp.clip(event_indices, 0, n_batch - 1)
        layer_indices = jnp.clip(layer_indices, 0, n_layers_total - 1)
        straw_indices = jnp.clip(straw_indices, 0, self.n_straws - 1)

        # Decode global layer index -> station/view/layer
        station = layer_indices // (self.n_views_per_station * self.n_layers_per_view)
        remainder = layer_indices % (self.n_views_per_station * self.n_layers_per_view)
        view = remainder // self.n_layers_per_view
        layer = remainder % self.n_layers_per_view

        # Normalize categorical indices
        station_norm = station.astype(jnp.float32) / max(self.n_stations - 1, 1)
        view_norm = view.astype(jnp.float32) / max(self.n_views_per_station - 1, 1)
        layer_norm = layer.astype(jnp.float32) / max(self.n_layers_per_view - 1, 1)
        straw_norm = straw_indices.astype(jnp.float32) / max(self.n_straws - 1, 1)

        # Extract design parameters
        positions = design[:, :n_layers_total]  # (batch, n_layers)
        angles = design[:, n_layers_total : 2 * n_layers_total]  # (batch, n_layers)
        magnetic_strength = design[:, -1]  # (batch,)

        # Per-hit design lookup
        layer_positions = positions[event_indices, layer_indices]
        layer_angles = angles[event_indices, layer_indices]
        magnetic_strength_hits = magnetic_strength[event_indices]

        # Normalize continuous features
        values_norm = (values - self.tdc_mean) / self.tdc_std

        # Keep this consistent with whatever you want to do for position scaling.
        # If you want normalized positions, use the commented line below instead.
        # positions_norm = (layer_positions - self.position_mean) / self.position_std
        positions_norm = layer_positions

        angles_norm = (layer_angles - self.angle_mean) / self.angle_std
        B_norm = (magnetic_strength_hits - self.B_mean) / self.B_std

        hit_features = jnp.stack(
            [
                station_norm,
                view_norm,
                layer_norm,
                straw_norm,
                values_norm,
                positions_norm,
                angles_norm,
                B_norm,
            ],
            axis=-1,
        )

        # Zero padded hits at input
        hit_features = hit_features * hit_mask

        return hit_features, event_indices, hit_mask

    def __call__(self, X: jax.Array, design: jax.Array, *, deterministic: bool = True):
        hit_features, event_indices, hit_mask = self.combine(X, design)
        n_batch = design.shape[0]

        def masked_segment_mean(values):
            # values: (n_hits, d)
            values = values * hit_mask
            sum_per_event = jax.ops.segment_sum(
                values, event_indices, num_segments=n_batch
            )
            count_per_event = jax.ops.segment_sum(
                hit_mask, event_indices, num_segments=n_batch
            )  # (n_batch, 1)
            return sum_per_event / jnp.clip(count_per_event, min=1.0)

        result = hit_features
        *rest, last = self.blocks

        # Repeated DeepSet blocks
        for block_layers in rest:
            for layer in block_layers:
                if isinstance(layer, nnx.Dropout):
                    result = layer(result, deterministic=deterministic)
                else:
                    result = layer(result)

            # Critical fix: kill padded-hit activations after the block
            result = result * hit_mask

            # Aggregate valid hits only
            aggregated = masked_segment_mean(result)

            # Broadcast event embedding back to hit level
            aggregated_per_hit = aggregated[event_indices]
            result = jnp.concatenate([result, aggregated_per_hit], axis=-1)

            # Keep padded rows zero after concat too
            result = result * hit_mask

        # Final block
        for layer in last:
            if isinstance(layer, nnx.Dropout):
                result = layer(result, deterministic=deterministic)
            else:
                result = layer(result)

        # Final mask before event pooling
        result = result * hit_mask

        aggregated = masked_segment_mean(result)
        result = self.output(aggregated)

        return jnp.reshape(result, shape=(result.shape[0], *self.target_shape))


class SparseBayesBlock(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        block_def: Sequence[int],
        p_dropout: float | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        if len(block_def) < 1:
            raise ValueError(
                "Each block_def must contain at least one output dimension."
            )

        hidden_dims = tuple(block_def[:-1])
        out_dim = block_def[-1]

        shared_layers = []
        prev_dim = in_dim

        for hidden_dim in hidden_dims:
            if p_dropout is not None and p_dropout > 0:
                shared_layers.append(nnx.Dropout(rate=p_dropout, rngs=rngs))
            shared_layers.append(nnx.Linear(prev_dim, hidden_dim, rngs=rngs))
            shared_layers.append(LeakyTanh(hidden_dim))
            prev_dim = hidden_dim

        self.shared = nnx.List(shared_layers)

        self.mu_head = nnx.Linear(prev_dim, out_dim, rngs=rngs)
        self.log_sigma_head = nnx.Linear(prev_dim, out_dim, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool = True):
        h = x
        for layer in self.shared:
            if isinstance(layer, nnx.Dropout):
                h = layer(h, deterministic=deterministic)
            else:
                h = layer(h)

        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h)

        return mu, log_sigma


def masked_bayes_segment_aggregate(
    mu: jax.Array,
    log_sigma: jax.Array,
    event_indices: jax.Array,
    hit_mask: jax.Array,
    n_batch: int,
    eps: float = 1e-4,
):

    mu = mu * hit_mask

    log_sigma = jnp.clip(log_sigma, -3.0, 3.0)
    sigma2 = jnp.exp(2.0 * log_sigma)
    precision = 1.0 / (sigma2 + eps)

    # Remove padded hits from aggregation
    precision = precision * hit_mask
    weighted_mu = precision * mu

    precision_sum = jax.ops.segment_sum(precision, event_indices, num_segments=n_batch)
    weighted_mu_sum = jax.ops.segment_sum(
        weighted_mu, event_indices, num_segments=n_batch
    )

    mu_event = weighted_mu_sum / jnp.clip(precision_sum, min=eps)
    sigma_event = jnp.sqrt(1.0 / jnp.clip(precision_sum, min=eps))

    # Clip sigma_event to prevent extreme values
    sigma_event = jnp.clip(sigma_event, 0.01, 10.0)

    return mu_event, sigma_event


class BayesDeepSet(Regressor):
    def __init__(
        self,
        detector: Detector,
        features: Sequence[Sequence[int]],
        p_dropout: float | None = None,
        n_max_hits: int = 200,
        *,
        rngs: nnx.Rngs,
    ):
        self.input_shape = None  # Sparse data has variable length
        self.design_shape = detector.design_shape()
        self.target_shape = detector.target_shape()
        self.ground_truth_shape = detector.ground_truth_shape()
        self.rngs = rngs

        print(f"BayesDeepSet input_shape: {self.input_shape} (variable - sparse data)")
        print(f"BayesDeepSet design_shape: {self.design_shape}")
        print(f"BayesDeepSet target_shape: {self.target_shape}")

        target_dim = math.prod(self.target_shape)

        self.tdc_mean = 440.0
        self.tdc_std = 80.0
        self.position_mean = 8950.0
        self.position_std = 500.0
        self.angle_mean = 0.0
        self.angle_std = 0.5
        self.B_mean = 5.0
        self.B_std = 10.0

        # Detector geometry
        self.n_max_hits = n_max_hits
        self.n_stations = detector.n_stations
        self.n_views_per_station = detector.n_views_per_station
        self.n_layers_per_view = detector.n_layers_per_view
        self.n_straws = detector.n_straws

        # Per-hit sparse feature dimension:
        # station, view, layer, straw, tdc, position, angle, B
        n_hit_features = 8

        all_blocks = []
        n_features = n_hit_features

        for block_def in features:
            block = SparseBayesBlock(
                in_dim=n_features,
                block_def=block_def,
                p_dropout=p_dropout,
                rngs=rngs,
            )
            all_blocks.append(block)

            latent_dim = block_def[-1]
            n_features = 3 * latent_dim  # [mu_hit, mu_event, sigma_event]

        self.blocks = nnx.List(all_blocks)

        *_, last = features
        n_latent = last[-1]

        self.output = nnx.Linear(2 * n_latent, target_dim, rngs=rngs)

    def combine(self, info, design):
        """
        info = (events, layers, straws, times, mask)

        Returns:
            hit_features:  (n_hits, 8)
            event_indices: (n_hits,)
            hit_mask:      (n_hits, 1)
        """
        events, layers, straws, times, mask = info

        mask_bool = jnp.asarray(mask, dtype=jnp.bool_)
        hit_mask = mask_bool.astype(jnp.float32)[:, None]  # (n_hits, 1)

        n_batch = design.shape[0]
        n_layers_total = (
            self.n_stations * self.n_views_per_station * self.n_layers_per_view
        )

        # Convert to arrays
        events = jnp.asarray(events, dtype=jnp.int32)
        layers = jnp.asarray(layers, dtype=jnp.int32)
        straws = jnp.asarray(straws, dtype=jnp.int32)
        values = jnp.asarray(times, dtype=jnp.float32)

        # Safe dummy indices for padded rows
        event_indices = jnp.where(mask_bool, events, 0)
        layer_indices = jnp.where(mask_bool, layers, 0)
        straw_indices = jnp.where(mask_bool, straws, 0)

        # Extra safety against bad sentinel values
        event_indices = jnp.clip(event_indices, 0, n_batch - 1)
        layer_indices = jnp.clip(layer_indices, 0, n_layers_total - 1)
        straw_indices = jnp.clip(straw_indices, 0, self.n_straws - 1)

        # Decode global layer index -> station/view/layer
        station = layer_indices // (self.n_views_per_station * self.n_layers_per_view)
        remainder = layer_indices % (self.n_views_per_station * self.n_layers_per_view)
        view = remainder // self.n_layers_per_view
        layer = remainder % self.n_layers_per_view

        # Normalize discrete indices to ~[0, 1]
        station_norm = station.astype(jnp.float32) / max(self.n_stations - 1, 1)
        view_norm = view.astype(jnp.float32) / max(self.n_views_per_station - 1, 1)
        layer_norm = layer.astype(jnp.float32) / max(self.n_layers_per_view - 1, 1)
        straw_norm = straw_indices.astype(jnp.float32) / max(self.n_straws - 1, 1)

        # Extract design = [positions(n_layers), angles(n_layers), B]
        positions = design[:, :n_layers_total]
        angles = design[:, n_layers_total : 2 * n_layers_total]
        magnetic_strength = design[:, -1]

        # Per-hit lookup
        layer_positions = positions[event_indices, layer_indices]
        layer_angles = angles[event_indices, layer_indices]
        magnetic_strength_hits = magnetic_strength[event_indices]

        # Normalize continuous features
        values_norm = (values - self.tdc_mean) / self.tdc_std

        # Use whichever you prefer here
        positions_norm = (layer_positions - self.position_mean) / self.position_std
        # positions_norm = layer_positions

        angles_norm = (layer_angles - self.angle_mean) / self.angle_std
        B_norm = (magnetic_strength_hits - self.B_mean) / self.B_std

        hit_features = jnp.stack(
            [
                station_norm,
                view_norm,
                layer_norm,
                straw_norm,
                values_norm,
                positions_norm,
                angles_norm,
                B_norm,
            ],
            axis=-1,
        )

        # Zero padded hits at input
        hit_features = hit_features * hit_mask

        return hit_features, event_indices, hit_mask

    def __call__(self, X: jax.Array, design: jax.Array, *, deterministic: bool = True):
        hit_features, event_indices, hit_mask = self.combine(X, design)
        n_batch = design.shape[0]

        result = hit_features
        *rest, last = self.blocks

        for block in rest:
            mu_hit, log_sigma_hit = block(result, deterministic=deterministic)

            # Keep padded rows dead
            mu_hit = mu_hit * hit_mask
            log_sigma_hit = log_sigma_hit * hit_mask

            mu_event, sigma_event = masked_bayes_segment_aggregate(
                mu_hit,
                log_sigma_hit,
                event_indices,
                hit_mask,
                n_batch,
            )

            mu_event_per_hit = mu_event[event_indices]
            sigma_event_per_hit = sigma_event[event_indices]
            result = jnp.concatenate(
                [mu_hit, mu_event_per_hit, sigma_event_per_hit],
                axis=-1,
            )
            result = result * hit_mask

        mu_hit, log_sigma_hit = last(result, deterministic=deterministic)
        mu_hit = mu_hit * hit_mask
        log_sigma_hit = log_sigma_hit * hit_mask

        mu_event, sigma_event = masked_bayes_segment_aggregate(
            mu_hit,
            log_sigma_hit,
            event_indices,
            hit_mask,
            n_batch,
        )

        event_repr = jnp.concatenate([mu_event, sigma_event], axis=-1)

        result = self.output(event_repr)

        return jnp.reshape(result, shape=(result.shape[0], *self.target_shape))
