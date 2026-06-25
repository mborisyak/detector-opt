import os

import numpy as np

import detopt

# NOTE: NOT YET PORTED to the new Detector contract (detector-spec.md).
# This script monkey-patches the loader and unpacks the old ragged measurements
# tuple (events, layers, straws, times, mask); the new detector returns a padded
# (B, M, 5) tensor via sample_events. It will not run end-to-end until reworked
# for the padded layout.


def save_all_data(seed, output, chunk_size=1000, **config):

    detector = detopt.detector.from_config(config["detector"])
    enc = detector.encode_design(config["design"])  # design ALWAYS from config (detector holds none)
    enc = enc.reshape(1, -1)

    # Load data to get total number of events
    if detector._data_loader is None:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from detopt.data import HNLDataLoader

        detector._data_loader = HNLDataLoader(detector.data_dir, max_particles=detector.max_particles)

    n_events = detector._data_loader.n_events
    n_chunks = (n_events + chunk_size - 1) // chunk_size

    base_dir = os.path.dirname(output) or "."
    base_name = os.path.splitext(os.path.basename(output))[0]
    os.makedirs(base_dir, exist_ok=True)

    print(f"Processing {n_events} events in {n_chunks} chunks of {chunk_size}")

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_events)
        chunk_n_events = end_idx - start_idx

        print(f"\nChunk {chunk_idx + 1}/{n_chunks}: events {start_idx}-{end_idx - 1}")

        # Load sequential batch from data loader (no random sampling)
        # This ensures we process ALL events exactly once, in order
        daughter_data, hnl_targets = detector._data_loader.get_sequential_batch(start_idx, end_idx)

        design = np.tile(enc, (chunk_n_events, 1))
        seed_chunk = hash((seed, chunk_idx)) & 0xFFFFFFFF

        # Temporarily override get_batch to return our pre-loaded sequential data
        # The detector's simulate() method internally calls get_batch(), so we
        # intercept it to provide the exact sequential slice we want
        original_get_batch = detector._data_loader.get_batch

        def sequential_get_batch(batch_size, rng=None):
            # Return the pre-loaded sequential chunk (ignores batch_size and rng)
            return daughter_data, hnl_targets

        detector._data_loader.get_batch = sequential_get_batch

        # Run detector simulation on this sequential chunk
        _ev = detector.sample_events(seed_chunk, design)
        measurements, target = _ev["X"], _ev["targets"]

        # Restore original get_batch method for next iteration
        detector._data_loader.get_batch = original_get_batch

        chunk_file = os.path.join(base_dir, f"{base_name}_chunk_{chunk_idx:04d}.npz")
        print(f"  Saving to {chunk_file}")

        # measurements is a tuple: (events, layers, straws, times)
        # Save as separate arrays for proper loading
        events, layers, straws, times, mask = measurements

        print(len(events))

        np.savez_compressed(
            chunk_file,
            events=events,
            layers=layers,
            straws=straws,
            times=times,
            mask=mask,
            targets=np.array(target),
        )

        file_size_mb = os.path.getsize(chunk_file) / (1024 * 1024)
        print(f"  ✓ Saved {chunk_n_events} events: {file_size_mb:.2f} MB")

    print(f"\n✓ Complete: {n_chunks} files saved")


if __name__ == "__main__":
    import gearup

    gearup.gearup(save=save_all_data).with_config("config/regression.yaml")()
