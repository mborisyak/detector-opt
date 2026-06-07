import glob
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm


class PrecomputedDataLoader:
    """
    Loads ALL precomputed detector responses into memory.

    The detector saves fixed-size arrays with masks. This loader simply
    concatenates them across chunks and returns batches directly.

    max_hits is calculated dynamically per batch as: 2 * batch_size * max_particles * n_layers
    (matching the detector's array allocation formula)
    """

    def __init__(
        self,
        data_dir: str,
        file_pattern: str = "*_chunk_*.npz",
        max_particles: int = 2,
        n_layers: int = 32,
    ):
        """
        Initialize data loader - loads ALL precomputed data into memory.

        Args:
            data_dir: Directory containing chunk_*.npz files
            file_pattern: Glob pattern to match chunk files
            max_particles: Maximum particles per event (from detector config)
            n_layers: Number of detector layers (from detector config)
        """
        self.data_dir = Path(data_dir)
        self.max_particles = max_particles
        self.n_layers = n_layers

        # Find all chunk files
        files = sorted(glob.glob(str(self.data_dir / file_pattern)))

        if len(files) == 0:
            raise FileNotFoundError(f"No chunk files found in {data_dir} matching pattern {file_pattern}")

        print(f"PrecomputedDataLoader: Loading {len(files)} chunk files into memory...")

        # Pre-allocate lists
        all_events = []
        all_layers = []
        all_straws = []
        all_times = []
        all_masks = []
        all_targets = []

        # Load ALL chunk files and concatenate
        event_offset = 0
        for file_path in tqdm(files, desc="Loading chunks"):
            data = np.load(file_path, allow_pickle=True)

            events = data["events"]
            layers = data["layers"]
            straws = data["straws"]
            times = data["times"]
            mask = data["mask"]
            targets = data["targets"]

            # Adjust event indices to be sequential across chunks
            # Only adjust where mask is True (valid hits)
            adjusted_events = events.copy()
            valid_mask = mask.astype(bool)
            if valid_mask.any():
                adjusted_events[valid_mask] += event_offset
                event_offset = int(adjusted_events[valid_mask].max()) + 1

            all_events.append(adjusted_events)
            all_layers.append(layers)
            all_straws.append(straws)
            all_times.append(times)
            all_masks.append(mask)
            all_targets.append(targets)

        # Concatenate all arrays
        events_concat = np.concatenate(all_events).astype(np.int32)
        layers_concat = np.concatenate(all_layers).astype(np.int32)
        straws_concat = np.concatenate(all_straws).astype(np.int32)
        times_concat = np.concatenate(all_times).astype(np.float32)
        mask_concat = np.concatenate(all_masks).astype(np.int32)
        self.targets = np.concatenate(all_targets, axis=0).astype(np.float32)

        self.n_events = len(self.targets)

        # Sort all arrays by event ID to ensure contiguous storage
        # This is critical for fast O(1) indexing
        print("Sorting arrays by event ID for contiguous storage...")
        sort_indices = np.argsort(events_concat, kind="stable")
        self.events = events_concat[sort_indices]
        self.layers = layers_concat[sort_indices]
        self.straws = straws_concat[sort_indices]
        self.times = times_concat[sort_indices]
        self.mask = mask_concat[sort_indices]

        # Extract only valid hits (where mask > 0) into compact arrays
        # This removes all padding/invalid entries
        print("Extracting valid hits into compact arrays...")
        valid_mask = self.mask > 0
        self.events = self.events[valid_mask]
        self.layers = self.layers[valid_mask]
        self.straws = self.straws[valid_mask]
        self.times = self.times[valid_mask]
        self.mask = self.mask[valid_mask]

        # Build event index array for O(1) batch extraction
        # Now all entries are valid, so we just find event boundaries
        print("Building event index array for fast sampling...")
        self.event_start_idx = np.zeros(self.n_events + 1, dtype=np.int32)

        # Find where each event starts (events are sorted and contiguous)
        if len(self.events) > 0:
            # Find boundaries where event ID changes
            change_points = np.concatenate([[0], np.where(np.diff(self.events) != 0)[0] + 1, [len(self.events)]])

            # Map each unique event to its start/end index
            for i in range(len(change_points) - 1):
                event_id = self.events[change_points[i]]
                start_idx = change_points[i]
                end_idx = change_points[i + 1]
                self.event_start_idx[event_id] = start_idx
                self.event_start_idx[event_id + 1] = end_idx

        # Debug: print first few event indices
        print("\nDebug: First 10 events:")
        for event_id in range(min(10, self.n_events)):
            start = self.event_start_idx[event_id]
            end = self.event_start_idx[event_id + 1]
            n_hits = end - start
            print(f"  Event {event_id}: start={start}, end={end}, n_hits={n_hits}")

        print(f"\n✓ Loaded {self.n_events} precomputed events into memory")
        print(f"  max_particles: {self.max_particles}, n_layers: {self.n_layers}")
        print(f"  Valid hits in dataset: {self.mask.sum()}")
        print(f"  Avg hits per event: {self.mask.sum() / self.n_events:.1f}")

    def get_batch(
        self,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple:
        """
        Get a batch of precomputed events with random sampling.

        Returns measurements as tuple (events, layers, straws, times, mask)
        where each is a 1D array padded to max_hits = 2 * batch_size * max_particles * n_layers

        Args:
            batch_size: Number of events to return
            rng: Random number generator for sampling

        Returns:
            measurements: tuple (events, layers, straws, times, mask)
            targets: (batch_size, 6) array of targets
        """
        if rng is None:
            rng = np.random.default_rng()

        # Random sampling with replacement
        event_indices = rng.choice(self.n_events, size=batch_size, replace=True)

        # Use internal method
        return self._get_batch_internal(event_indices)

    def get_sequential_batch(
        self,
        start_idx: int,
        end_idx: int,
    ) -> Tuple:
        """
        Get a sequential batch of events by index range.

        Args:
            start_idx: Starting index (inclusive)
            end_idx: Ending index (exclusive)

        Returns:
            measurements: tuple (events, layers, straws, times, mask)
            targets: (batch_size, 6) array of targets
        """
        # Validate indices
        start_idx = max(0, start_idx)
        end_idx = min(self.n_events, end_idx)

        if start_idx >= end_idx:
            raise ValueError(f"Invalid range: start_idx={start_idx}, end_idx={end_idx}")

        # Sequential slice
        event_indices = np.arange(start_idx, end_idx)
        batch_size = len(event_indices)

        # Calculate max_hits for this batch size
        max_hits = 2 * batch_size * self.max_particles * self.n_layers

        # Get targets
        targets = self.targets[event_indices]

        # Pre-allocate output arrays for maximum speed
        padded_events = np.zeros(max_hits, dtype=np.int32)
        padded_layers = np.zeros(max_hits, dtype=np.int32)
        padded_straws = np.zeros(max_hits, dtype=np.int32)
        padded_times = np.zeros(max_hits, dtype=np.float32)
        padded_mask = np.zeros(max_hits, dtype=np.int32)

        # Fast extraction using pre-computed start indices
        write_pos = 0
        for new_idx, old_idx in enumerate(event_indices):
            start = self.event_start_idx[old_idx]
            end = self.event_start_idx[old_idx + 1]
            n_event_hits = end - start

            if n_event_hits > 0:
                # Copy hits directly into output arrays
                padded_events[write_pos : write_pos + n_event_hits] = new_idx
                padded_layers[write_pos : write_pos + n_event_hits] = self.layers[start:end]
                padded_straws[write_pos : write_pos + n_event_hits] = self.straws[start:end]
                padded_times[write_pos : write_pos + n_event_hits] = self.times[start:end]
                padded_mask[write_pos : write_pos + n_event_hits] = self.mask[start:end]
                write_pos += n_event_hits

        n_hits = write_pos

        # Check if we exceed max_hits
        if n_hits > max_hits:
            raise ValueError(
                f"Too many hits ({n_hits}) for batch_size={batch_size}! "
                f"Expected max_hits={max_hits} (2 * {batch_size} * {self.max_particles} * {self.n_layers}). "
                f"This indicates a mismatch between saved data and loader configuration."
            )

        measurements = (
            padded_events,
            padded_layers,
            padded_straws,
            padded_times,
            padded_mask,
        )

        return measurements, targets

    def get_train_val_split(self, val_fraction: float = 0.2, seed: int = 42):
        """
        Get train/validation split indices.

        Args:
            val_fraction: Fraction of data to use for validation (0.0-1.0)
            seed: Random seed for reproducible splits

        Returns:
            train_indices: Array of event indices for training
            val_indices: Array of event indices for validation
        """
        rng = np.random.default_rng(seed)
        all_indices = np.arange(self.n_events)
        rng.shuffle(all_indices)

        n_val = int(self.n_events * val_fraction)
        val_indices = all_indices[:n_val]
        train_indices = all_indices[n_val:]

        return train_indices, val_indices

    def get_batch_from_indices(
        self,
        indices: np.ndarray,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple:
        """
        Get a batch by sampling from specified event indices.

        Args:
            indices: Array of event indices to sample from
            batch_size: Number of events to return
            rng: Random number generator for sampling

        Returns:
            measurements: tuple (events, layers, straws, times, mask)
            targets: (batch_size, 6) array of targets
        """
        if rng is None:
            rng = np.random.default_rng()

        # Sample from the provided indices
        sampled_indices = rng.choice(indices, size=batch_size, replace=True)

        # Use the regular get_batch logic with these specific indices
        return self._get_batch_internal(sampled_indices)

    def _get_batch_internal(self, event_indices: np.ndarray) -> Tuple:
        """Internal method to get batch for given event indices."""
        batch_size = len(event_indices)
        max_hits = 2 * batch_size * self.max_particles * self.n_layers

        # Get targets
        targets = self.targets[event_indices]

        # Pre-allocate output arrays for maximum speed
        padded_events = np.zeros(max_hits, dtype=np.int32)
        padded_layers = np.zeros(max_hits, dtype=np.int32)
        padded_straws = np.zeros(max_hits, dtype=np.int32)
        padded_times = np.zeros(max_hits, dtype=np.float32)
        padded_mask = np.zeros(max_hits, dtype=np.int32)

        # Fast extraction using pre-computed start indices
        write_pos = 0
        for new_idx, old_idx in enumerate(event_indices):
            start = self.event_start_idx[old_idx]
            end = self.event_start_idx[old_idx + 1]
            n_event_hits = end - start

            if n_event_hits > 0:
                # Copy hits directly into output arrays
                padded_events[write_pos : write_pos + n_event_hits] = new_idx
                padded_layers[write_pos : write_pos + n_event_hits] = self.layers[start:end]
                padded_straws[write_pos : write_pos + n_event_hits] = self.straws[start:end]
                padded_times[write_pos : write_pos + n_event_hits] = self.times[start:end]
                padded_mask[write_pos : write_pos + n_event_hits] = self.mask[start:end]
                write_pos += n_event_hits

        n_hits = write_pos

        # Check if we exceed max_hits
        if n_hits > max_hits:
            raise ValueError(
                f"Too many hits ({n_hits}) for batch_size={batch_size}! "
                f"Expected max_hits={max_hits} (2 * {batch_size} * {self.max_particles} * {self.n_layers}). "
                f"This indicates a mismatch between saved data and loader configuration."
            )

        measurements = (
            padded_events,
            padded_layers,
            padded_straws,
            padded_times,
            padded_mask,
        )

        return measurements, targets

    def get_statistics(self) -> dict:
        """Compute dataset statistics from loaded data."""
        total_hits = self.mask.sum()
        avg_hits = total_hits / self.n_events if self.n_events > 0 else 0

        return {
            "n_events": self.n_events,
            "max_particles": self.max_particles,
            "n_layers": self.n_layers,
            "avg_hits_per_event": float(avg_hits),
            "target_stats": {
                "mean": self.targets.mean(axis=0).tolist(),
                "std": self.targets.std(axis=0).tolist(),
                "min": self.targets.min(axis=0).tolist(),
                "max": self.targets.max(axis=0).tolist(),
            },
        }


# Test the loader
if __name__ == "__main__":
    print("=" * 70)
    print("Testing Precomputed Data Loader")
    print("=" * 70)

    try:
        # Try loading from a precomputed directory
        # Use detector config values: max_particles=4, n_layers=32
        loader = PrecomputedDataLoader("precomputed_data3", max_particles=2, n_layers=32)

        print("\nDataset Statistics:")
        stats = loader.get_statistics()
        print(f"  Total events: {stats['n_events']}")
        print(f"  max_particles: {stats['max_particles']}, n_layers: {stats['n_layers']}")
        print(f"  Avg hits per event: {stats['avg_hits_per_event']:.1f}")
        print(f"\n  Target statistics:")
        print(f"    Mean: {stats['target_stats']['mean']}")
        print(f"    Std:  {stats['target_stats']['std']}")

        for val in loader.events[:2500]:
            print(val, end=" ")
        print("\n\n")
        prev = 0
        for val in loader.event_start_idx[1:2500]:
            print(val, loader.events[val - 1], loader.events[val])
            prev = val

        print(len(loader.events))

        print("\n\nLoading test batch (should be instant)...")
        import time

        start = time.time()
        measurements, targets = loader.get_batch(batch_size=128, rng=np.random.default_rng(42))
        elapsed = time.time() - start

        print(f"✓ Loaded 128 events in {elapsed * 1000:.2f} ms")
        print(f"  Measurements type: tuple of 5 arrays (events, layers, straws, times, mask)")
        events, layers, straws, times, mask = measurements
        print(f"  Events shape: {events.shape}")
        print(f"  Layers shape: {layers.shape}")
        print(f"  Straws shape: {straws.shape}")
        print(f"  Times shape: {times.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Total valid hits: {mask.sum()}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Max hits for batch_size=128: {2 * 128 * 4 * 32}")

        print("\n" + "=" * 70)
        print("✅ Precomputed data loader working!")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("\nTo use this loader:")
        print("  1. Run save_all_data.py to generate precomputed chunks")
        print("  2. Specify the correct data directory")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
