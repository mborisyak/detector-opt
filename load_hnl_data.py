"""
HNL Data Loader for Real Experimental Data - Memory-Optimized Version

This loader reads ALL daughter particle data from combined_all/*.npz files
into memory once at initialization for fast sampling during training.

Data flow:
1. Load ALL daughter particles at initialization
2. Sample from memory during training (fast!)
3. Pass to straw_detector.solve_sparse() to simulate detector response
4. Use hits as input to neural network
5. Target is HNL decay vertex (prestraw_hnl_dx, dy, dz, px, py, pz)
"""

import glob
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class HNLDataLoader:
    """
    Loads ALL daughter particle data from HNL decay events into memory.

    Fast sampling with no file I/O during training.
    """

    def __init__(self, data_dir: str = "clean_data", max_particles: int = 50):
        """
        Initialize data loader - loads ALL data into memory.

        Args:
            data_dir: Directory containing combined_data_*.npz files
            max_particles: Maximum particles per event (for padding)
        """
        self.data_dir = Path(data_dir)
        self.max_particles = max_particles

        # Try both naming conventions
        files = sorted(glob.glob(str(self.data_dir / "*.npz")))
        if len(files) == 0:
            files = sorted(glob.glob(str(self.data_dir / "geom_*_combined_data.npz")))

        if len(files) == 0:
            raise FileNotFoundError(f"No NPZ files found in {data_dir}")

        print(f"HNLDataLoader: Loading {len(files)} files into memory...")

        # Pre-allocate lists
        all_masses = []
        all_charges = []
        all_positions = []
        all_momenta = []
        all_times = []
        all_n_particles = []
        all_targets = []
        all_pdgs = []

        cnt = 0
        cnt2 = 0
        cnt3 = 0
        cnt_ev = 0
        # Load ALL files
        for file_path in tqdm(files, desc="Loading data"):
            data = np.load(file_path, allow_pickle=True)
            # cnt3 += 1
            # Get unique events in this file
            unique_events = np.unique(data["prestraw_ev"])

            for event_id in unique_events:
                mask = data["prestraw_ev"] == event_id
                n_parts = min(np.sum(mask), max_particles)
                # if cnt_ev > 15:
                #     break
                if n_parts == 0:
                    continue
                if n_parts != 2:
                    continue
                # if event_id >= 1:
                #     continue
                # cnt_ev += 1

                # Initialize padded arrays for this event
                masses = np.zeros(max_particles, dtype=np.float32)
                charges = np.zeros(max_particles, dtype=np.float32)
                positions = np.zeros((max_particles, 3), dtype=np.float32)
                momenta = np.zeros((max_particles, 3), dtype=np.float32)
                times = np.zeros(max_particles, dtype=np.float32)

                # Extract particle data
                positions[:n_parts, 0] = data["prestraw_x"][mask][:n_parts]
                positions[:n_parts, 1] = data["prestraw_y"][mask][:n_parts]
                positions[:n_parts, 2] = data["prestraw_z"][mask][:n_parts]

                # Convert momentum from GeV/c to MeV/c (multiply by 1000)
                momenta[:n_parts, 0] = data["prestraw_px"][mask][:n_parts] * 1000.0
                momenta[:n_parts, 1] = data["prestraw_py"][mask][:n_parts] * 1000.0
                momenta[:n_parts, 2] = data["prestraw_pz"][mask][:n_parts] * 1000.0

                # Extract time of flight (TOF) in nanoseconds
                if "prestraw_tof" in data:
                    times[:n_parts] = data["prestraw_tof"][mask][:n_parts]

                pdg_codes = data["prestraw_pdg"][mask][:n_parts]
                pdgs = np.zeros(max_particles, dtype=np.int32)
                pdgs[:n_parts] = pdg_codes.astype(np.int32)

                # Map PDG codes to masses and charges
                for i, pdg in enumerate(pdg_codes):
                    masses[i], charges[i] = self._pdg_to_mass_charge(int(pdg))

                # Extract HNL target
                # Target: positions in cm, momenta in GeV/c (keep as GeV for targets)
                target = np.array(
                    [
                        data["prestraw_hnl_dx"][mask][0],
                        data["prestraw_hnl_dy"][mask][0],
                        data["prestraw_hnl_dz"][mask][0],
                        data["prestraw_hnl_px"][mask][0],
                        data["prestraw_hnl_py"][mask][0],
                        data["prestraw_hnl_pz"][mask][0],
                    ],
                    dtype=np.float32,
                )

                all_masses.append(masses)
                all_charges.append(charges)
                all_positions.append(positions)
                all_momenta.append(momenta)
                all_times.append(times)
                all_n_particles.append(n_parts)
                all_targets.append(target)
                all_pdgs.append(pdgs)

                cnt += -n_parts + max_particles
                cnt2 += 1
                print(-n_parts + max_particles)
                print(cnt3)
            if cnt3 == 3:
                break
            if cnt_ev == 15:
                break
        # Convert to arrays
        self.masses = np.array(all_masses, dtype=np.float32)
        self.charges = np.array(all_charges, dtype=np.float32)
        self.positions = np.array(all_positions, dtype=np.float32)
        self.momenta = np.array(all_momenta, dtype=np.float32)
        self.times = np.array(all_times, dtype=np.float32)
        self.n_particles = np.array(all_n_particles, dtype=np.int32)
        self.targets = np.array(all_targets, dtype=np.float32)
        self.pdgs = np.array(all_pdgs, dtype=np.int32)

        self.n_events = len(self.n_particles)
        print(cnt / cnt2)
        print(self.n_events)
        input("wait")

        # print(f"✓ Loaded {self.n_events} events into memory")
        # print(f"  Memory usage: ~{self._estimate_memory_mb():.1f} MB")
        # print(
        #     f"  Particles per event: min={self.n_particles.min()}, "
        #     f"max={self.n_particles.max()}, mean={self.n_particles.mean():.1f}"
        # )

    def _estimate_memory_mb(self) -> float:
        """Estimate memory usage in MB."""
        total_bytes = (
            self.masses.nbytes
            + self.charges.nbytes
            + self.positions.nbytes
            + self.momenta.nbytes
            + self.times.nbytes
            + self.n_particles.nbytes
            + self.targets.nbytes
        )
        return total_bytes / (1024 * 1024)

    def get_batch(
        self,
        batch_size: int,
        max_particles: int = 50,  # Kept for API compatibility
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[dict, np.ndarray]:
        """
        Get a batch of events - returns ALL events sequentially (no random sampling).

        Args:
            batch_size: Number of events to return (must equal n_events for full dataset)
            max_particles: Ignored (uses self.max_particles from init)
            rng: Ignored (no random sampling)

        Returns:
            daughter_data: Dict with daughter particle info
                - masses: MeV
                - charges: e
                - positions: cm
                - momenta: MeV/c
                - times: ns (time of flight)
            targets: (batch_size, 6) array of [dx, dy, dz in cm, px, py, pz in GeV/c]
        """
        # Return all events sequentially (no sampling)
        indices = np.arange(min(batch_size, self.n_events))
        # indices = rng.choice(self.n_events, size=batch_size, replace=True)
        # Return pre-loaded data (just indexing - super fast!)
        daughter_data = {
            "masses": self.masses[indices],
            "charges": self.charges[indices],
            "positions": self.positions[indices],
            "momenta": self.momenta[indices],
            "times": self.times[indices],
            "n_particles": self.n_particles[indices],
        }

        targets = self.targets[indices]

        return daughter_data, targets

    def _pdg_to_mass_charge(self, pdg: int) -> Tuple[float, float]:
        """
        Convert PDG code to mass (MeV) and charge.

        Note: Masses are in MeV to match C code expectations.
        Momenta are converted to MeV/c in get_batch().

        Args:
            pdg: Particle PDG code

        Returns:
            (mass in MeV, charge) tuple
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
            2112: (939.57, 0.0),  # neutron
            -2112: (939.57, 0.0),  # antineutron
        }

        if pdg in pdg_mass_map:
            return pdg_mass_map[pdg]
        else:
            # Default to pion mass for unknown particles
            charge = 1.0 if pdg > 0 else -1.0 if pdg < 0 else 0.0
            return (139.57, charge)

    def get_statistics(self) -> dict:
        """Compute dataset statistics from loaded data."""
        return {
            "stds": {"targets": self.targets.std(axis=0)},
            "means": {"targets": self.targets.mean(axis=0)},
            "n_events": self.n_events,
            "particles_per_event": {
                "min": int(self.n_particles.min()),
                "max": int(self.n_particles.max()),
                "mean": float(self.n_particles.mean()),
                "median": float(np.median(self.n_particles)),
            },
            "target_ranges": {
                "dx": (
                    float(self.targets[:, 0].min()),
                    float(self.targets[:, 0].max()),
                ),
                "dy": (
                    float(self.targets[:, 1].min()),
                    float(self.targets[:, 1].max()),
                ),
                "dz": (
                    float(self.targets[:, 2].min()),
                    float(self.targets[:, 2].max()),
                ),
                "px": (
                    float(self.targets[:, 3].min()),
                    float(self.targets[:, 3].max()),
                ),
                "py": (
                    float(self.targets[:, 4].min()),
                    float(self.targets[:, 4].max()),
                ),
                "pz": (
                    float(self.targets[:, 5].min()),
                    float(self.targets[:, 5].max()),
                ),
            },
        }


# Test the loader
if __name__ == "__main__":
    print("=" * 70)
    print("Testing HNL Data Loader (Memory-Optimized)")
    print("=" * 70)

    try:
        loader = HNLDataLoader("selected_data")  # "combined_tof"

        print("\nDataset Statistics:")
        stats = loader.get_statistics()
        print(stats["stds"]["targets"])
        print(stats["means"]["targets"])
        print(f"  Total events: {stats['n_events']}")
        print(f"  Particles per event: {stats['particles_per_event']}")
        print(f"\n  Target ranges:")
        for key, (vmin, vmax) in stats["target_ranges"].items():
            print(f"    {key}: [{vmin:8.2f}, {vmax:8.2f}]")

        print("\n\nLoading test batch (should be instant)...")
        import time

        start = time.time()
        daughter_data, targets = loader.get_batch(
            batch_size=128, rng=np.random.default_rng(42)
        )
        elapsed = time.time() - start

        print(f"✓ Loaded 128 events in {elapsed * 1000:.2f} ms")
        print(f"  Masses shape: {daughter_data['masses'].shape}")
        print(f"  Positions shape: {daughter_data['positions'].shape}")
        print(f"  Targets shape: {targets.shape}")

        print("\n" + "=" * 70)
        print("✅ Memory-optimized data loader working!")
        print("=" * 70)

        P = loader.max_particles
        mask = np.arange(P)[None, :] < loader.n_particles[:, None]
        codes = loader.pdgs[mask]  # all real PDG entries across all events
        u, c = np.unique(codes, return_counts=True)
        mean_per_event = c / loader.n_events  # <-- "mean entries in events"
        # sort by most frequent
        order = np.argsort(mean_per_event)[::-1]
        u = u[order]
        mean_per_event = mean_per_event[order]
        # print top few
        print("\nTop PDG codes by mean entries/event:")
        for pdg, m in zip(u[:20], mean_per_event[:20]):
            print(f"  PDG {int(pdg):6d}: {m:.4f} per event")

        # plot top N as a bar-histogram
        topN = 30
        N = min(topN, len(u))

        plt.figure(figsize=(12, 6))
        plt.bar(np.arange(N), mean_per_event[:N])
        plt.xticks(np.arange(N), [str(int(x)) for x in u[:N]], rotation=45, ha="right")
        plt.ylabel("Mean entries per event")
        plt.title("PDG code frequency (mean per event)")
        plt.tight_layout()
        plt.savefig("output/pdg_mean_per_event.png", dpi=150)
        plt.close()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
