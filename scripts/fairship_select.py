"""Build a CLEAN dataset with FairShip's reconstruction/candidate selection, then redo the track fit.

Our sparse pool keeps any event with two reconstructable daughters (weak selection -- see
`scripts/track_fit.py`, >=4 hits). FairShip's ShipAna applies a much stricter track-quality +
HNL-candidate selection before it quotes a resolution; comparing our wire fit to FairShip on OUR
loose set is unfair. This script re-runs the same Boris MAP fit (`track_fit.run_pipeline`) but with
FairShip's cuts, on the events that pass them -- the apples-to-apples "clean" comparison.

FairShip selection (FairShip/macro/Ship2NumPy.py + macro/ShipAna.py; fiducialCut = False):
  * MEAS_CUT = 25 : >= 25 straw measurements (Ndf) per daughter track,
  * (>= 3 of the 4 stations crossed -- shipDigiReco's `len(stationCrossed) >= 3`),
  * CHI2_CUT = 4.0 : fitted chi2 / Ndf < 4 per track (Ndf = nhits - 5),
  * DOCA_CUT = 2.0 : distance of closest approach between the two daughter tracks <= 2 cm.

The first two are pre-fit (hit-count) cuts; chi2/Ndf and DoCA are applied post-fit.

    python scripts/fairship_select.py seed=0 n_events=8192 coef=0.1
"""

import detopt

import track_fit  # the fit machinery lives here; we only change the selection

# FairShip ShipAna / Ship2NumPy constants.
MEAS_CUT = 25
MIN_STATIONS = 3
CHI2_CUT = 4.0
DOCA_CUT = 2.0


def select(seed=0, n_events=8192, dt=0.4, n_steps=160, n_iters=1500, lr=0.05, coef=0.1, out="fairship_clean.png", **config):
    detector = detopt.detector.from_config(config["detector"])
    return track_fit.run_pipeline(
        detector,
        config,
        seed=seed,
        n_events=n_events,
        dt=dt,
        n_steps=n_steps,
        n_iters=n_iters,
        lr=lr,
        coef=coef,
        min_hits=MEAS_CUT,
        min_stations=MIN_STATIONS,
        chi2_cut=CHI2_CUT,
        doca_cut=DOCA_CUT,
        out=out,
        label="Boris track fit (FairShip clean selection)",
    )


if __name__ == "__main__":
    import gearup

    gearup.gearup(select).with_config("config/regression.yaml")()
