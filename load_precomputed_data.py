import glob
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

from detopt.detector.straw import SparseHits


class PrecomputedDataLoader:
    """
    Loads ALL precomputed detector responses into memory.

    Fast sampling with no detector simulation during training.
    """

    def __init__(self, data_dir: str, file_pattern: str = "*_chunk_*.npz"):
        """
        Initialize data loader - loads ALL precomputed data into memory.

        Args:
            data_dir: Directory containing chunk_*.npz files
            file_pattern: Glob pattern to match chunk files
        """
        self.data_dir = Path(data_dir)

        # Find all chunk files
        files = sorted(glob.glob(str(self.data_dir / file_pattern)))

        if len(files) == 0:
            raise FileNotFoundError(
                f"No chunk files found in {data_dir} matching pattern {file_pattern}"
            )

        print(f"PrecomputedDataLoader: Loading {len(files)} chunk files into memory...")

        # Pre-allocate lists
        all_measurements = []
        all_targets = []

        # Load ALL chunk files
        for file_path in tqdm(files, desc="Loading chunks"):
            data = np.load(file_path, allow_pickle=True)

            events = data["events"]
            layers = data["layers"]
            straws = data["straws"]
            times = data["times"]
            targets = data["targets"]

            # Store as tuple for this chunk
            all_measurements.append((events, layers, straws, times))
            all_targets.append(targets)

        # Concatenate all targets
        self.targets = np.concatenate(all_targets, axis=0).astype(np.float32)

        # For measurements, check format
        # New format: tuple of (events, layers, straws, times)
        if len(all_measurements) > 0:
            self.measurements = self._concatenate_measurement_tuples(all_measurements)
            self.is_sparse = True
        else:
            raise ValueError(f"Unknown measurement type: {type(all_measurements[0])}")

        self.n_events = len(self.targets)

        print(f"✓ Loaded {self.n_events} precomputed events into memory")
        print(f"  Memory usage: ~{self._estimate_memory_mb():.1f} MB")
        print(f"  Data format: tuple of (events, layers, straws, times)")

    def _concatenate_measurement_tuples(self, measurement_tuples):
        """Concatenate multiple measurement tuples (events, layers, straws, times)."""
        all_events = []
        all_layers = []
        all_straws = []
        all_times = []

        event_offset = 0

        for events, layers, straws, times in measurement_tuples:
            # Adjust event indices to be sequential across chunks
            adjusted_events = events + event_offset
            all_events.append(adjusted_events)
            all_layers.append(layers)
            all_straws.append(straws)
            all_times.append(times)

            # Update offset for next chunk
            if len(events) > 0:
                event_offset = int(adjusted_events.max()) + 1

        # Concatenate all arrays
        combined = (
            np.concatenate(all_events),
            np.concatenate(all_layers),
            np.concatenate(all_straws),
            np.concatenate(all_times),
        )

        return combined

    def get_batch(
        self,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple:
        """
        Get a batch of precomputed events with random sampling.

        Args:
            batch_size: Number of events to return
            rng: Random number generator for sampling

        Returns:
            measurements: SparseHits object or dense array for the batch
            targets: (batch_size, 6) array of targets
        """
        if rng is None:
            rng = np.random.default_rng()

        # Random sampling with replacement
        indices = rng.choice(self.n_events, size=batch_size, replace=True)

        # Get targets
        targets = self.targets[indices]

        # Get measurements
        measurements = self._extract_tuple_batch(indices)

        return measurements, targets

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
            measurements: SparseHits object or dense array for the batch
            targets: (batch_size, 6) array of targets
        """
        # Validate indices
        start_idx = max(0, start_idx)
        end_idx = min(self.n_events, end_idx)

        if start_idx >= end_idx:
            raise ValueError(f"Invalid range: start_idx={start_idx}, end_idx={end_idx}")

        # Sequential slice
        indices = np.arange(start_idx, end_idx)

        # Get targets
        targets = self.targets[indices]

        # Get measurements
        measurements = self._extract_tuple_batch(indices)

        return measurements, targets

    def _extract_tuple_batch(self, event_indices):
        """Extract a batch of events from tuple format measurements."""
        events, layers, straws, times = self.measurements

        # Create a mask for hits belonging to selected events
        mask = np.isin(events, event_indices)

        # Extract hits
        batch_events = events[mask]
        batch_layers = layers[mask]
        batch_straws = straws[mask]
        batch_times = times[mask]

        # Remap event indices to 0, 1, 2, ...
        event_mapping = {
            old_idx: new_idx for new_idx, old_idx in enumerate(event_indices)
        }
        remapped_events = np.array([event_mapping[e] for e in batch_events])

        return (remapped_events, batch_layers, batch_straws, batch_times)

    def get_statistics(self) -> dict:
        """Compute dataset statistics from loaded data."""
        return {
            "n_events": self.n_events,
            "is_sparse": self.is_sparse,
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
        loader = PrecomputedDataLoader("precomputed_data")

        print("\nDataset Statistics:")
        stats = loader.get_statistics()
        print(f"  Total events: {stats['n_events']}")
        print(f"  Sparse format: {stats['is_sparse']}")
        print(f"\n  Target statistics:")
        print(f"    Mean: {stats['target_stats']['mean']}")
        print(f"    Std:  {stats['target_stats']['std']}")

        print("\n\nLoading test batch (should be instant)...")
        import time

        start = time.time()
        measurements, targets = loader.get_batch(
            batch_size=128, rng=np.random.default_rng(42)
        )
        elapsed = time.time() - start

        print(f"✓ Loaded 128 events in {elapsed * 1000:.2f} ms")
        print(f"  Measurements type: {type(measurements)}")
        if isinstance(measurements, SparseHits):
            print(f"  Total hits: {len(measurements.events)}")
        else:
            print(f"  Measurements shape: {measurements.shape}")
        print(f"  Targets shape: {targets.shape}")

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
