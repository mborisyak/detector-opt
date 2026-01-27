import inspect
import math
from typing import Sequence

import jax
import jax.nn as jnn
import jax.numpy as jnp
from flax import nnx

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

        dropout = (
            lambda: []
            if p_dropout is None
            else [nnx.Dropout(rate=p_dropout, rngs=rngs)]
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
        # FIXED: Don't call super().__init__ because detector.output_shape() needs batch_size
        # Instead, manually set the shapes we need for sparse data
        self.input_shape = None  # Sparse data has variable length, no fixed input shape
        self.design_shape = detector.design_shape()
        self.target_shape = detector.target_shape()
        self.ground_truth_shape = detector.ground_truth_shape()
        self.rngs = rngs

        print(f"DeepSet input_shape: {self.input_shape} (variable - sparse data)")
        print(f"DeepSet design_shape: {self.design_shape}")
        print(f"DeepSet target_shape: {self.target_shape}")
        target_dim = math.prod(self.target_shape)

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

    # ADDED: proper indentation and class method
    def combine(self, sparse_dict, design):
        # Convert sparse hit dictionary to dense padded arrays with features
        # Fully vectorized using JAX segment operations (no Python loops)
        events = jnp.array(sparse_dict["events"])  # (n_hits,) event index per hit
        layers = jnp.array(sparse_dict["layers"])  # (n_hits,) global layer index (0-31)
        straws = jnp.array(sparse_dict["straws"])  # (n_hits,) straw index (0-199)
        values = jnp.array(sparse_dict["values"])  # (n_hits,) TDC values

        n_batch = design.shape[0]
        n_layers_total = (
            self.n_stations * self.n_views_per_station * self.n_layers_per_view
        )  # Total number of layers (typically 32)

        # Compute station, view, layer from global layer index (vectorized)
        station = layers // (self.n_views_per_station * self.n_layers_per_view)
        remainder = layers % (self.n_views_per_station * self.n_layers_per_view)
        view = remainder // self.n_layers_per_view
        layer = remainder % self.n_layers_per_view

        # Normalize indices to [0, 1]
        station_norm = station.astype(jnp.float32) / self.n_stations
        view_norm = view.astype(jnp.float32) / self.n_views_per_station
        layer_norm = layer.astype(jnp.float32) / self.n_layers_per_view
        straw_norm = straws.astype(jnp.float32) / self.n_straws

        # Extract design parameters: [positions (n_layers), angles (n_layers), B (1)]
        positions = design[:, :n_layers_total]  # (batch, n_layers)
        angles = design[:, n_layers_total : 2 * n_layers_total]  # (batch, n_layers)
        magnetic_strength = design[:, -1]  # (batch,)

        # Get design parameters for each hit using its layer index (vectorized indexing)
        batch_indices = events  # Which batch each hit belongs to
        layer_positions = positions[batch_indices, layers]  # (n_hits,)
        layer_angles = angles[batch_indices, layers]  # (n_hits,)
        magnetic_strength_hits = magnetic_strength[batch_indices]  # (n_hits,)

        # Stack all features per hit: (n_hits, 8)
        hit_features = jnp.stack(
            [
                station_norm,
                view_norm,
                layer_norm,
                straw_norm,
                values,
                layer_positions,
                layer_angles,
                magnetic_strength_hits,
            ],
            axis=-1,
        )

        # Vectorized slot assignment using cumsum within segments
        # Sort by event to group hits together
        sort_indices = jnp.argsort(events)
        sorted_events = events[sort_indices]
        sorted_features = hit_features[sort_indices]

        # Compute slot index for each hit within its event (vectorized)
        # By comparing each event with the previous, we can detect boundaries
        event_changes = jnp.concatenate(
            [jnp.array([True]), sorted_events[1:] != sorted_events[:-1]]
        )
        # Cumsum of changes gives us segment IDs, subtract to get within-segment index
        segment_ids = jnp.cumsum(event_changes) - 1
        # Create a counter that resets at each segment boundary
        hit_slots = jnp.arange(len(sorted_events)) - jnp.maximum.accumulate(
            jnp.where(event_changes, jnp.arange(len(sorted_events)), 0)
        )

        # Create output arrays
        combined = jnp.zeros((n_batch, self.n_max_hits, 8), dtype=jnp.float32)
        mask = jnp.zeros((n_batch, self.n_max_hits), dtype=jnp.bool_)

        # Only keep hits that fit within n_max_hits - use where to mask values
        valid_mask = hit_slots < self.n_max_hits

        # Use where to select valid indices, then clip for safety
        safe_events = jnp.where(valid_mask, sorted_events, 0)
        safe_slots = jnp.where(valid_mask, hit_slots, 0)

        # Scatter all hits, but only those with valid_mask will have correct positions
        # Invalid hits go to position 0, which we'll overwrite or mask out
        combined = combined.at[safe_events, safe_slots].set(
            jnp.where(valid_mask[:, None], sorted_features, 0.0)
        )
        mask = mask.at[safe_events, safe_slots].set(valid_mask)

        return combined, mask

    # FIXED: Correct signature (removed extra mask parameter)
    def __call__(self, X: jax.Array, design: jax.Array, *, deterministic: bool = True):
        # Handle input format
        print(
            f"DEBUG: X type = {type(X)}, X = {X if not isinstance(X, dict) else 'dict with keys: ' + str(X.keys())}"
        )
        if isinstance(X, dict):
            # Sparse format: convert to dense
            combined, mask = self.combine(X, design)
        else:
            # Dense format (for compatibility)
            combined = X
            mask = jnp.any(X != 0, axis=-1)

        result = combined
        *rest, last = self.blocks

        # DeepSet pattern with masked aggregation
        for block_layers in rest:
            # Apply all layers in this block sequentially
            hit_features = result
            for layer in block_layers:
                if isinstance(layer, nnx.Dropout):
                    hit_features = layer(hit_features, deterministic=deterministic)
                else:
                    hit_features = layer(hit_features)

            # Masked mean aggregation
            masked_features = mask[..., None] * hit_features
            sum_features = jnp.sum(masked_features, axis=1, keepdims=True)
            count = jnp.sum(mask, axis=1, keepdims=True)[..., None]
            aggregated = sum_features / jnp.clip(
                count, min=1.0
            )  # Avoid division by zero

            # Broadcast and concatenate
            aggregated = jnp.broadcast_to(aggregated, shape=hit_features.shape)
            result = jnp.concatenate([hit_features, aggregated], axis=-1)

        # Final aggregation - apply last block layers
        hit_features = result
        for layer in last:
            if isinstance(layer, nnx.Dropout):
                hit_features = layer(hit_features, deterministic=deterministic)
            else:
                hit_features = layer(hit_features)
        masked_features = mask[..., None] * hit_features
        sum_features = jnp.sum(masked_features, axis=1)
        count = jnp.sum(mask, axis=1)[..., None]
        result = sum_features / jnp.clip(count, min=1.0)

        result = self.output(result)
        return jnp.reshape(result, shape=(result.shape[0], *self.target_shape))


class BayesDeepSet(Regressor):
    def __init__(
        self,
        detector: Detector,
        features: Sequence[Sequence[int]],
        p_dropout: float | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(detector, rngs=rngs)
        input_dim, design_dim = (
            math.prod(self.input_shape),
            math.prod(self.design_shape),
        )
        target_dim = math.prod(self.target_shape)
        ### position + angle + B
        n_design = 3

        n_layers, n_straws = self.input_shape
        self.blocks: list[Block] = []

        n_features = n_design + n_straws
        for block_def in features:
            units = (n_features, *block_def)
            self.blocks.append(
                Block(
                    *(
                        Block(
                            nnx.Linear(n_in, n_out, rngs=rngs),
                            LeakyTanh(
                                n_out,
                            ),
                        )
                        for n_in, n_out in zip(units[:-2], units[1:-1])
                    ),
                    [
                        nnx.Linear(units[-2], units[-1], rngs=rngs),
                        nnx.Linear(units[-2], units[-1], rngs=rngs),
                    ],
                )
            )
            n_features = 3 * units[-1]

        *_, last = features
        *_, n_latent = last

        self.output = nnx.Linear(2 * n_latent, target_dim, rngs=rngs)

    def combine(self, X, design):
        n_b, n_l, n_s = X.shape
        _, n_d = design.shape

        X = jnp.reshape(X, shape=(n_b, n_l, n_s))
        positions, angles, magnetic_strength = (
            design[:, :n_l],
            design[:, n_l : 2 * n_l],
            design[:, -1],
        )
        positions = jnp.broadcast_to(positions[:, :, None], shape=(n_b, n_l, 1))
        angles = jnp.broadcast_to(angles[:, :, None], shape=(n_b, n_l, 1))
        magnetic_strength = jnp.broadcast_to(
            magnetic_strength[:, None, None], shape=(n_b, n_l, 1)
        )

        return jnp.concatenate([X, positions, angles, magnetic_strength], axis=-1)

    def __call__(self, X: jax.Array, design: jax.Array, *, deterministic: bool = True):
        result = self.combine(X, design)

        *rest, last = self.blocks

        for block in rest:
            mus, log_sigmas = block(result)
            mu, sigma = bayes_aggregate(mus, log_sigmas, axis=1, keepdims=True)
            mu = jnp.broadcast_to(mu, shape=mus.shape)
            sigma = jnp.broadcast_to(sigma, shape=mus.shape)
            result = jnp.concatenate([mus, mu, sigma], axis=-1)

        mus, log_sigmas = last(result)
        mu, sigma = bayes_aggregate(mus, log_sigmas, axis=1, keepdims=False)
        result = jnp.concatenate([mu, sigma], axis=-1)

        result = self.output(result)

        return jnp.reshape(result, shape=(result.shape[0], *self.target_shape))
