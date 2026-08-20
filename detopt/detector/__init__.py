from typing import Any

from .common import Detector

from . import straw_detector
from . import straw

from .. import utils
from .enzyme import EnzymeDetector
from .enzyme_inhibitor import EnzymeInhibitorDetector
from .enzyme_mm import EnzymeMMDetector
from .enzyme_depletion import EnzymeDepletionDetector
from .enzyme_depletion_bi import EnzymeDepletionBiDetector
from .growth import GrowthDetector
from .linear import LinearDetector
from .mnist import MNISTDetector
from .erasure import ErasureDetector
from .pixel_density import PixelDensityDetector
from .straw import StrawDetector  # abstract base (no design scheme)
from .free_straw import FreeStrawDetector, free_design_array
from .stereo_straw import StereoStrawDetector
from .stereo_tracking import Stereo4Feature, StereoTrackerTruth
from .stereo_tracking_layerwise import StereoLayerWise
from .stereo_tracking_image import StereoImage
from .stereo_tracking_hits import StereoHits

__detectors__: dict[str, type[Detector]] = {
    # GEOMETRY bases (StrawDetector / StereoStrawDetector) are abstract-combine and NOT registered as
    # leaves: every entry below is a concrete combine LEAF. FairShip REPLAY is no longer a separate class
    # -- it is the combine leaf with `engine: relay` + `data_dir` (a special engine-less source handled in
    # StrawDetector, loads one digi file), so `ship_relay_*` are those leaves; their configs set the engine.
    # Not a particle detector at all: the enzymatic-reaction single-batch design of experiments
    # (the proposal's bioprocess work package) under the same contract.
    "enzyme": EnzymeDetector,
    # The same chemistry plus an inhibitor whose concentration is a design coordinate; the target is
    # the compound's MECHANISM CLASS (3-way), scored by cross-entropy / ln 3.
    "enzyme_inhib": EnzymeInhibitorDetector,
    # Same chemistry, different QUESTION: the target is the enzyme's own Michaelis-Menten coefficients
    # rather than its melting point, the substrate concentrations are design coordinates, and the
    # temperature box sits below the melt.
    "enzyme_mm": EnzymeMMDetector,
    # The SIMPLEST enzymatic task here and the only one with no nuisance at all: single-substrate
    # Michaelis-Menten depletion, dA/dt = -q A / (A + K). The design is the initial concentrations and
    # nothing else, the target is the variant's own (ln q, ln K), and the objective is scored by an
    # ANALYTIC instrument -- a grid posterior mean -- rather than by a network.
    "enzyme_depletion": EnzymeDepletionDetector,
    # The same depletion question with TWO substrates, A + B -> C + D by the ping-pong bi-bi law:
    # three parameters (q, K_A, K_B), the design is the pair of initial concentrations per
    # experiment, and on the diagonal A0 = B0 the two constants are EXACTLY degenerate, so a batch
    # has to break the A/B symmetry in both directions. Scored by the same kind of ANALYTIC
    # instrument -- a grid posterior mean -- rather than by a network.
    "enzyme_depletion_bi": EnzymeDepletionBiDetector,
    # Also not a particle detector: bacterial growth, one batch of cultures per design (CTMI cardinal
    # temperatures + Monod batch kinetics), the strain's optimal growth temperature as the target.
    "growth": GrowthDetector,
    # The DEBUG task: probe positions on a linear response, no physics, no data, milliseconds on a
    # CPU -- and the only detector here whose optimal design and achievable loss are known in closed
    # form, so a driver can be checked against the answer rather than against its own output.
    "linear": LinearDetector,
    # Not a particle detector either: MNIST digit classification through a rectangular visible
    # WINDOW, the design being where that window sits and how big it is. A dense image task -- the
    # only one here whose combine emits a 2-D image with a channel axis rather than a set of
    # elements -- and the one that prices its design (`design_penalty` charges the visible area).
    "mnist": MNISTDetector,
    # The same digits and the same ln(10) price, but the design is HOW MUCH OF THE FRAME SURVIVES:
    # one keep probability, each pixel independently erased to 0. An erased pixel and a background
    # pixel are indistinguishable, so the design costs a random share of the ink with no way to tell
    # which share -- a subsampling problem, not a signal-to-noise one.
    "erasure": ErasureDetector,
    # The same digits again, but the frame is a CONTINUOUS field (a bicubic interpolant) read out on
    # an n_grid x n_grid grid whose point DENSITY is the design: x_i = Phi(sigma_x * z_i) over fixed
    # normal quantiles z, so sigma = 1 is the uniform grid, below it the points crowd the centre and
    # above it the borders. The sample budget is fixed by the grid, so the trade-off is intrinsic and
    # `design_penalty` returns None -- the only image task here that prices nothing.
    "pixel_density": PixelDensityDetector,
    "straw": FreeStrawDetector,  # free per-layer geometry + 4-feature combine
    "stereo_tracking": Stereo4Feature,  # stereo geometry + 4-feature combine (the default stereo detector)
    "stereo_tracker_truth": StereoTrackerTruth,  # + per-hit (x,y)/drift_r/tdc TRUTH for tracker experiments
    "stereo_layerwise": StereoLayerWise,  # set element = layer (TDC grid per layer)
    "stereo_image": StereoImage,  # (n_layers, n_straws, 6) image for the CNN regressor
    "stereo_hits": StereoHits,  # per-hit (M, 7) features for the continuous-conv regressor
    "ship_relay_4feat": Stereo4Feature,  # replay FairShip digi hits (engine: relay): 4-feature combine
    "ship_relay_layerwise": StereoLayerWise,  # replay FairShip digi hits (engine: relay): layer-wise combine
}


def from_config(config: dict[str, Any]):
    clazz, arguments = utils.config.extract(config, library=__detectors__)
    return clazz.from_config(config=arguments)
