"""
HNL Data Loader for Real Experimental Data

This loader reads daughter particle data from combined_all/*.npz files
and returns it in a format suitable for detector simulation via solve_sparse.

Data flow:
1. Load daughter particles (prestraw_x, y, z, px, py, pz, pdg) from NPZ
2. Pass to straw_detector.solve_sparse() to simulate detector response
3. Get fdigi_times (TDC hits) as simulation output
4. Use hits as input to neural network
5. Target is HNL decay vertex (prestraw_hnl_dx, dy, dz, px, py, pz)
"""

import glob
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class HNLDataLoader:
    """
    Loads daughter particle data from HNL decay events.

    The data is formatted for use with straw_detector.solve_sparse()
    to simulate detector response and generate TDC hits.
    """

    def __init__(self, data_dir: str = "combined_all"):
        """
        Initialize data loader.

        Args:
            data_dir: Directory containing combined_data_*.npz files
        """
        self.data_dir = Path(data_dir)
        # Try both naming conventions
        self.files = sorted(glob.glob(str(self.data_dir / "combined_data_*.npz")))
        if len(self.files) == 0:
            # Try alternative naming pattern
            self.files = sorted(
                glob.glob(str(self.data_dir / "geom_*_combined_data.npz"))
            )

        if len(self.files) == 0:
            raise FileNotFoundError(f"No NPZ files found in {data_dir}")

        print(f"HNLDataLoader: Found {len(self.files)} data files")

        # Cache for loaded files
        self._cache = {}
        self._current_file_idx = 0

    def _load_file(self, file_idx: int) -> dict:
        """Load a single NPZ file into cache."""
        if file_idx in self._cache:
            return self._cache[file_idx]

        data = np.load(self.files[file_idx], allow_pickle=True)

        # Extract relevant fields
        result = {
            "ev": data["prestraw_ev"],  # Event number for grouping
            "x": data["prestraw_x"].astype(np.float32),
            "y": data["prestraw_y"].astype(np.float32),
            "z": data["prestraw_z"].astype(np.float32),
            "px": data["prestraw_px"].astype(np.float32),
            "py": data["prestraw_py"].astype(np.float32),
            "pz": data["prestraw_pz"].astype(np.float32),
            "pdg": data["prestraw_pdg"],
            # HNL decay vertex (TARGET)
            "hnl_dx": data["prestraw_hnl_dx"].astype(np.float32),
            "hnl_dy": data["prestraw_hnl_dy"].astype(np.float32),
            "hnl_dz": data["prestraw_hnl_dz"].astype(np.float32),
            "hnl_px": data["prestraw_hnl_px"].astype(np.float32),
            "hnl_py": data["prestraw_hnl_py"].astype(np.float32),
            "hnl_pz": data["prestraw_hnl_pz"].astype(np.float32),
        }

        # Keep limited cache (10 files)
        if len(self._cache) > 10:
            oldest_key = min(self._cache.keys())
            del self._cache[oldest_key]

        self._cache[file_idx] = result
        return result

    def get_batch(
        self,
        batch_size: int,
        max_particles: int = 50,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[dict, np.ndarray]:
        """
        Get a batch of events with daughter particle data.

        Args:
            batch_size: Number of events to return
            max_particles: Maximum particles per event (for padding)
            rng: Random number generator (optional)

        Returns:
            daughter_data: Dict with daughter particle info
                {
                    'masses': (batch, max_particles) - particle masses in MeV
                    'charges': (batch, max_particles) - particle charges
                    'positions': (batch, max_particles, 3) - initial positions [x,y,z] in mm
                    'momenta': (batch, max_particles, 3) - initial momenta [px,py,pz] in GeV/c
                    'n_particles': (batch,) - actual number of particles per event
                }
            targets: (batch_size, 6) array of [dx, dy, dz, px, py, pz]
                - dx, dy, dz: HNL decay vertex position in mm
                - px, py, pz: HNL momentum in GeV/c
        """
        if rng is None:
            rng = np.random.default_rng()

        # Initialize padded arrays
        masses = np.zeros((batch_size, max_particles), dtype=np.float32)
        charges = np.zeros((batch_size, max_particles), dtype=np.float32)
        positions = np.zeros((batch_size, max_particles, 3), dtype=np.float32)
        momenta = np.zeros((batch_size, max_particles, 3), dtype=np.float32)
        n_particles = np.zeros(batch_size, dtype=np.int32)
        targets = np.zeros((batch_size, 6), dtype=np.float32)

        event_count = 0

        while event_count < batch_size:
            # Load current file
            data = self._load_file(self._current_file_idx)

            # Get unique events in this file
            unique_events = np.unique(data["ev"])

            # Sample events from this file
            events_needed = min(batch_size - event_count, len(unique_events))
            sampled_events = rng.choice(
                unique_events, size=events_needed, replace=False
            )

            for global_ev_id in sampled_events:
                if event_count >= batch_size:
                    break

                # Get daughter particles for this event
                mask = data["ev"] == global_ev_id
                n_parts = min(np.sum(mask), max_particles)

                if n_parts == 0:
                    continue

                # Extract particle data
                positions[event_count, :n_parts, 0] = data["x"][mask][:n_parts]
                positions[event_count, :n_parts, 1] = data["y"][mask][:n_parts]
                positions[event_count, :n_parts, 2] = data["z"][mask][:n_parts]

                momenta[event_count, :n_parts, 0] = data["px"][mask][:n_parts]
                momenta[event_count, :n_parts, 1] = data["py"][mask][:n_parts]
                momenta[event_count, :n_parts, 2] = data["pz"][mask][:n_parts]

                pdg_codes = data["pdg"][mask][:n_parts]

                # Map PDG codes to masses and charges
                for i, pdg in enumerate(pdg_codes):
                    masses[event_count, i], charges[event_count, i] = (
                        self._pdg_to_mass_charge(int(pdg))
                    )

                n_particles[event_count] = n_parts

                # Extract HNL target (same for all particles in event)
                targets[event_count, 0] = data["hnl_dx"][mask][0]  # dx
                targets[event_count, 1] = data["hnl_dy"][mask][0]  # dy
                targets[event_count, 2] = data["hnl_dz"][mask][0]  # dz
                targets[event_count, 3] = data["hnl_px"][mask][0]  # px
                targets[event_count, 4] = data["hnl_py"][mask][0]  # py
                targets[event_count, 5] = data["hnl_pz"][mask][0]  # pz

                event_count += 1

            # Move to next file
            self._current_file_idx = (self._current_file_idx + 1) % len(self.files)

        daughter_data = {
            "masses": masses,
            "charges": charges,
            "positions": positions,
            "momenta": momenta,
            "n_particles": n_particles,
        }

        return daughter_data, targets

    def _pdg_to_mass_charge(self, pdg: int) -> Tuple[float, float]:
        """
        Convert PDG code to mass (MeV) and charge.

        Args:
            pdg: Particle PDG code

        Returns:
            (mass, charge) tuple
        """
        # Common particles in HNL decays
        pdg_mass_map = {
            13: (105.66, -1.0),  # muon-
            -13: (105.66, 1.0),  # muon+
            11: (0.511, -1.0),  # electron
            -11: (0.511, 1.0),  # positron
            211: (139.57, 1.0),  # pi+
            -211: (139.57, -1.0),  # pi-
            321: (493.68, 1.0),  # K+
            -321: (493.68, -1.0),  # K-
            2212: (938.27, 1.0),  # proton
            -2212: (938.27, -1.0),  # antiproton
            22: (0.0, 0.0),  # photon
            111: (134.98, 0.0),  # pi0
            130: (497.61, 0.0),  # K_L
            310: (497.61, 0.0),  # K_S
            12: (0.0, 0.0),  # nu_e
            -12: (0.0, 0.0),  # nu_e_bar
            14: (0.0, 0.0),  # nu_mu
            -14: (0.0, 0.0),  # nu_mu_bar
        }

        if pdg in pdg_mass_map:
            return pdg_mass_map[pdg]
        else:
            # Default to pion mass for unknown particles
            charge = 1.0 if pdg > 0 else -1.0 if pdg < 0 else 0.0
            return (139.57, charge)

    def get_statistics(self, n_files: int = 10) -> dict:
        """
        Compute dataset statistics.

        Args:
            n_files: Number of files to sample

        Returns:
            Dictionary of statistics
        """
        all_n_particles = []
        all_targets = []

        for i in range(min(n_files, len(self.files))):
            data = self._load_file(i)
            unique_events = np.unique(data["ev"])

            for ev in unique_events:
                mask = data["ev"] == ev
                n_parts = np.sum(mask)
                all_n_particles.append(n_parts)

                target = np.array(
                    [
                        data["hnl_dx"][mask][0],
                        data["hnl_dy"][mask][0],
                        data["hnl_dz"][mask][0],
                        data["hnl_px"][mask][0],
                        data["hnl_py"][mask][0],
                        data["hnl_pz"][mask][0],
                    ]
                )
                all_targets.append(target)

        all_n_particles = np.array(all_n_particles)
        all_targets = np.array(all_targets)

        return {
            "n_events_sampled": len(all_n_particles),
            "n_files": len(self.files),
            "particles_per_event": {
                "min": int(all_n_particles.min()),
                "max": int(all_n_particles.max()),
                "mean": float(all_n_particles.mean()),
                "median": float(np.median(all_n_particles)),
            },
            "target_ranges": {
                "dx": (float(all_targets[:, 0].min()), float(all_targets[:, 0].max())),
                "dy": (float(all_targets[:, 1].min()), float(all_targets[:, 1].max())),
                "dz": (float(all_targets[:, 2].min()), float(all_targets[:, 2].max())),
                "px": (float(all_targets[:, 3].min()), float(all_targets[:, 3].max())),
                "py": (float(all_targets[:, 4].min()), float(all_targets[:, 4].max())),
                "pz": (float(all_targets[:, 5].min()), float(all_targets[:, 5].max())),
            },
        }


# Test the loader
if __name__ == "__main__":
    print("=" * 70)
    print("Testing HNL Data Loader")
    print("=" * 70)

    try:
        loader = HNLDataLoader("combined_all")

        print("\nComputing statistics...")
        stats = loader.get_statistics(n_files=5)
        print(f"\nDataset Statistics (from 5 files):")
        print(f"  Total files: {stats['n_files']}")
        print(f"  Events sampled: {stats['n_events_sampled']}")
        print(f"  Particles per event: {stats['particles_per_event']}")
        print(f"\n  Target ranges:")
        for key, (vmin, vmax) in stats["target_ranges"].items():
            print(f"    {key}: [{vmin:8.2f}, {vmax:8.2f}]")

        print("\n\nLoading test batch...")
        daughter_data, targets = loader.get_batch(
            batch_size=4, max_particles=50, rng=np.random.default_rng(42)
        )

        print(f"\nBatch loaded:")
        print(f"  Batch size: 4 events")
        print(f"  Masses shape: {daughter_data['masses'].shape}")
        print(f"  Charges shape: {daughter_data['charges'].shape}")
        print(f"  Positions shape: {daughter_data['positions'].shape}")
        print(f"  Momenta shape: {daughter_data['momenta'].shape}")
        print(f"  N particles: {daughter_data['n_particles']}")
        print(f"  Targets shape: {targets.shape}")
        print(f"\nFirst event:")
        print(f"  N particles: {daughter_data['n_particles'][0]}")
        print(f"  First particle position: {daughter_data['positions'][0, 0]}")
        print(f"  First particle momentum: {daughter_data['momenta'][0, 0]}")
        print(f"  HNL target: {targets[0]}")

        print("\n" + "=" * 70)
        print("✅ Data loader working!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
