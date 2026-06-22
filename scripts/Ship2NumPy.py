#!/usr/bin/env python
# coding: utf-8

"""Convert FairShip simulation ROOT files into flat float32 numpy arrays.

For every *accepted* event the script writes a ground-truth row describing the
HNL and its two decay products, and a variable number of *particle* rows ---
one snapshot per MC particle that crosses the upstream border of the first
straw station (SST).  An index array attributes each particle row to its event.

Acceptance / selection
-----------------------
* The boundary plane z is configurable (``--boundary``); the default is the
  downstream end of the decay vessel.  Presets: ``decay-vessel-end``,
  ``sst-front`` (``TrackStation1.z - station_length``), ``sst-back``
  (``TrackStation4.z + station_length``); or an explicit z in cm.
* The acceptance region on that plane is the straw aperture enlarged by
  APERTURE_SCALE (=1.2, i.e. +20%) in x and y: ``|x| < 1.2*aperture_width``,
  ``|y| < 1.2*aperture_height``.
* An event is kept only if *both* HNL products, linearly extrapolated along
  their production momentum, fall inside that region (they intersect the
  region, not merely the infinite plane).
* With ``--require-all4`` an event is additionally required to have *both*
  daughters leave straw MC hits in all 4 tracking stations.  Hits from a
  daughter's descendants (e.g. a track created when the daughter scatters)
  count as the original daughter's.
* A particle is snapshotted if it was born upstream of the boundary and is
  still alive past it ("born before, decayed after").  Its decay z is taken
  from the production vertex of its daughters (``+inf`` if it has none).
  There is no magnetic field before the SST, so trajectories are straight and
  the crossing point / momentum are obtained by linear interpolation.
* Only charged particles and photons are kept (neutrinos and neutral hadrons
  are dropped), and the crossing point must fall inside the acceptance region.

Output (single ``.npz``)
------------------------
``truth``        (N_events, 15) float32
    [hnl_mass, hnl_px, hnl_py, hnl_pz, vx, vy, vz,
     p1_mass, p1_px, p1_py, p1_pz, p2_mass, p2_px, p2_py, p2_pz]
    ``v`` is the HNL decay vertex.
``particles``    (M, 8) float32
    [charge, mass, px, py, pz, x, y, t0]
    ``charge`` signed electric charge in units of e (0 for photons);
    ``px,py,pz`` production momentum (constant, no field); ``x,y`` the crossing
    point on the boundary plane (z is the boundary, constant); ``t0`` the
    crossing time [ns]. Within each event the two HNL decay products are placed
    first (rows 0 and 1), in the order they appear in ``truth`` (p1, p2).
``event_index``  (M,) int32
    Row in ``truth`` that each particle belongs to.
``boundary_z``     scalar float32   z of the boundary plane [cm].
``boundary_half``  (2,) float32     region half-sizes [half_width, half_height] [cm].

With ``--reco`` an additional array, keyed to the ``truth`` event rows (one row
per accepted event), is added.  ``--reco`` requires the matching ``*_rec.root``
file.

``reco``  (N_events, 19) float32
    [hnl_mass, hnl_px, hnl_py, hnl_pz, vx, vy, vz,
     p1_px, p1_py, p1_pz, o1x, o1y, o1z,
     p2_px, p2_py, p2_pz, o2x, o2y, o2z]
    Reconstructed kinematics of the HNL candidate matched to the true HNL
    (ShipAna selection: both daughters Ndf >= measCut, same-HNL MC match,
    Doca <= DOCA_CUT). ``hnl_mass`` is the reconstructed invariant mass and ``v``
    the candidate's fitted (common) decay vertex. Each daughter's ``(p, o)`` is
    its fitted track genfit-extrapolated to the candidate's vertex z-plane --
    exactly the per-track (momentum, position) the reconstruction feeds into the
    DOCA / vertex fit (shipVertex.TwoTrackVertex), through the field so the weak
    decay-volume bending is included. ``o1`` and ``o2`` therefore share that z
    but differ in x,y by the track-to-track distance of closest approach (``v``
    is their chi2-optimal common fit); the pair is ordered to match p1/p2 in
    ``truth``. Reproducing these points re-runs genfit extrapolation, so
    ``--reco`` loads the geometry and field map. The row is all-NaN for an
    accepted event with no matched candidate.

With ``--full-mc`` the following are added (still keyed to the ``truth`` event
rows, i.e. only accepted events).  ``--full-mc`` requires the matching
``*_rec.root`` file, which supplies the digitized straw response (TDC).

``mc_particles``    (K, 8) float32   [charge, mass, px, py, pz, vx, vy, vz]
    ``charge`` signed electric charge in units of e (NaN if the species is
    unknown to the PDG database); ``px,py,pz`` production momentum; ``vx,vy,vz``
    the production (origin) vertex [cm].
``mc_info``         (K, 4) int32     [id, parent_id, pdg, proc_id]
    ``id`` is the MCTrack index within its event; ``parent_id`` the mother's
    ``id`` (-1 for primaries); both refer to that event's block. ``proc_id`` is
    the Geant4 production process (TMCProcess enum) that created the particle.
``mc_event_index``  (K,) int32       row in ``truth`` for each MC particle.
``hits``            (H, 3) float32   [x, y, z]  straw MC points [cm].
``hit_track``       (H,) int32       MCTrack ``id`` that produced the hit.
``hit_event_index`` (H,) int32       row in ``truth`` for each hit.
``digi_tdc``        (D,) float32     measured straw TDC [ns] (fdigi).
``digi_straw``      (D, 4) int32     [station, view, layer, straw] of the fired
    straw (decoded detector ID; 1-based station/straw, 0-based view/layer).
``digi_invalid``    (D,) bool        True for an invalid hit, i.e. a same-straw
    duplicate with a larger TDC (``not strawtubesHit::isValid()``).
``digi_event_index`` (D,) int32      row in ``truth`` for each digitized hit.
"""

import math
import os
from argparse import ArgumentParser

import numpy as np
import ROOT

import shipRoot_conf
import ShipGeoConfig

shipRoot_conf.configure()
PDG = ROOT.TDatabasePDG.Instance()

HNL_code = 9900015

# Speed of light in FairShip units (cm / ns).
C_CM_PER_NS = 29.9792458

# The acceptance region on the boundary plane is the straw aperture enlarged
# by this factor in x and y (aperture + 20%).
APERTURE_SCALE = 1.2

# Boundary-plane presets (resolved against ShipGeo); see resolve_boundary_z.
BOUNDARY_PRESETS = ("decay-vessel-end", "sst-front", "sst-back")

# Minimum straw measurements per daughter for the reco-quality count
# (ShipAna.py measCutFK; pattern recognition uses measCutPR = 22).
MEAS_CUT = 25
MEAS_CUT_PR = 22
CHI2_CUT = 4.0  # ShipAna.py chi2CutOff on chi2/Ndf
DOCA_CUT = 2.0  # ShipAna.py docaCut [cm] on the HNL candidate

# Max genfit extrapolation step [cm] (shipVertex.Task._stepwise_extrapolate_to_z).
MAX_EXTRAP_STEP = 500.0


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-f", "--inputFile", dest="input_files", nargs="+", required=True,
        help="Input simulation ROOT file(s). Shells expand globs; quote a "
             "pattern if you want python to expand it.",
    )
    parser.add_argument(
        "-g", "--geo_file", dest="geo_file", default=None,
        help="ROOT geofile. If omitted, derived from each input file "
             "(sim_*.root -> geo_*.root).",
    )
    parser.add_argument(
        "-o", "--output", dest="output", default="ship2numpy.npz",
        help="Output .npz file (default: ship2numpy.npz).",
    )
    parser.add_argument(
        "--boundary", dest="boundary", default="decay-vessel-end",
        help="Boundary plane: a preset name "
             f"({', '.join(BOUNDARY_PRESETS)}) or an explicit z in cm "
             "(default: decay-vessel-end).",
    )
    parser.add_argument(
        "--require-all4", dest="require_all4", action="store_true",
        help="Keep only events in which both HNL daughters leave straw MC hits "
             "in all 4 tracking stations. A daughter's scattered descendants "
             "count as the original daughter. Requires the strawtubesPoint "
             "branch in the input file.",
    )
    parser.add_argument(
        "--full-mc", dest="full_mc", action="store_true",
        help="Additionally dump, per accepted event, every MC particle "
             "(mc_particles/mc_info/mc_event_index), every straw MC point "
             "(hits/hit_track/hit_event_index) and every digitized straw hit "
             "with its TDC (digi_tdc/digi_straw/digi_event_index). Requires the "
             "matching *_rec.root file.",
    )
    parser.add_argument(
        "--rec", dest="rec", action="store_true",
        help="Friend-add the matching *_rec.root file and report the true "
             "reconstruction efficiency (both daughters reconstructed with "
             "ShipAna quality cuts).",
    )
    parser.add_argument(
        "--reco", dest="reco", action="store_true",
        help="Additionally dump, per accepted event, the reconstructed HNL "
             "candidate kinematics (reco array: mass, momentum, decay vertex and "
             "each daughter's momentum and own reconstructed origin -- the points "
             "the DOCA/vertex fit uses), matched to the true HNL with the ShipAna "
             "selection. Requires the matching *_rec.root file and loads the "
             "geometry and field map to re-extrapolate FitTracks.",
    )
    return parser.parse_args()


def geo_for(input_file, override):
    if override is not None:
        return override
    root_dir, file_name = os.path.split(input_file)
    if not file_name.startswith("sim_"):
        raise ValueError(f"cannot derive geo file from {file_name!r}; pass --geo_file")
    return os.path.join(root_dir, f"geo_{file_name[4:]}")


def resolve_boundary_z(ShipGeo, spec):
    """Resolve the boundary plane z [cm] from a preset name or an explicit value."""
    straw = ShipGeo.strawtubes_geo
    presets = {
        "decay-vessel-end": ShipGeo.decayVolume.z0 + ShipGeo.decayVolume.length,
        "sst-front": ShipGeo.TrackStation1.z - straw.station_length,
        "sst-back": ShipGeo.TrackStation4.z + straw.station_length,
    }
    if spec in presets:
        return presets[spec]
    try:
        return float(spec)
    except (TypeError, ValueError):
        raise ValueError(
            f"unknown boundary {spec!r}; use one of {list(presets)} or a z in cm"
        )


def load_geometry(geo_file, boundary_spec):
    """Return (z_boundary, half_width, half_height) for the chosen boundary plane.

    The acceptance region is the straw aperture enlarged by APERTURE_SCALE.
    """
    fgeo = ROOT.TFile(geo_file)
    ShipGeo = ShipGeoConfig.load_from_root_file(fgeo)

    z_boundary = resolve_boundary_z(ShipGeo, boundary_spec)
    half_width = APERTURE_SCALE * ShipGeo.strawtubes_geo.width    # aperture half-width in x
    half_height = APERTURE_SCALE * ShipGeo.strawtubes_geo.height  # aperture half-height in y
    fgeo.Close()
    return z_boundary, half_width, half_height


# Keeps the FairRunSim / field map alive for the process; the genfit
# FieldManager and MaterialEffects are singletons initialised exactly once.
_genfit_keepalive = None


def setup_genfit_extrapolation(geo_file):
    """Initialise geometry, field and genfit material so FitTracks can be
    extrapolated exactly as the reconstruction does (needed by --reco to
    reproduce the per-daughter DOCA/vertex points). Runs once; subsequent calls
    are no-ops. Mirrors the setup in macro/ShipAna.py.
    """
    global _genfit_keepalive
    if _genfit_keepalive is not None:
        return
    import geomGeant4
    import shipDet_conf

    fgeo = ROOT.TFile.Open(geo_file)
    ShipGeo = ShipGeoConfig.load_from_root_file(fgeo)
    if not hasattr(ShipGeo.Bfield, "fieldMap"):
        raise RuntimeError(
            f"{geo_file} has no field map; --reco cannot extrapolate FitTracks"
        )

    run = ROOT.FairRunSim()
    run.SetName("TGeant4")  # transport not used, only needed to build the field
    run.SetSink(ROOT.FairRootFileSink(ROOT.TMemFile("output", "recreate")))
    run.SetUserConfig("g4Config_basic.C")
    run.GetRuntimeDb()
    modules = shipDet_conf.configure(run, ShipGeo)
    fieldMaker = geomGeant4.addVMCFields(ShipGeo, "", True, withVirtualMC=False)
    fgeo.Get("FAIRGeom")  # load the TGeoManager used for material lookups

    geoMat = ROOT.genfit.TGeoMaterialInterface()
    ROOT.genfit.MaterialEffects.getInstance().init(geoMat)
    bfield = ROOT.genfit.FairShipFields()
    bfield.setField(fieldMaker.getGlobalField())
    ROOT.genfit.FieldManager.getInstance().init(bfield)

    # Hold references so ROOT/genfit don't lose the field and geometry.
    _genfit_keepalive = (run, modules, fieldMaker, geoMat, bfield, fgeo)


def find_hnl_and_daughters(mc_tracks):
    """Return (hnl_id, [daughter_ids]) or (None, []) if no HNL in the event."""
    hnl_id = None
    daughters = []
    for track_id, track in enumerate(mc_tracks):
        mother_id = track.GetMotherId()
        if mother_id < 0:
            continue
        if mc_tracks[mother_id].GetPdgCode() == HNL_code:
            if hnl_id is None:
                hnl_id = mother_id
            elif hnl_id != mother_id:
                raise AssertionError(f"two HNLs? {mother_id}, {hnl_id}")
            daughters.append(track_id)
    return hnl_id, daughters


def daughter_owner(mc_tracks, daughters):
    """Map every track id to the daughter (one of `daughters`) it descends from.

    Walks each track's mother chain up to the first daughter ancestor. A particle
    that scatters is recorded by Geant4 as a fresh MCTrack whose chain leads back
    to the original; treating such a descendant as the original daughter equates
    the scattered particle with the one it came from (the same mother-chain
    matching used by reconstructed_mc_ids / match_candidate_to_hnl). Tracks that
    do not descend from either daughter are absent from the returned mapping.
    """
    daughter_set = set(daughters)
    owner = {}
    for track_id in range(len(mc_tracks)):
        cur = track_id
        while cur >= 0:
            if cur in daughter_set:
                owner[track_id] = cur
                break
            cur = mc_tracks[cur].GetMotherId()
    return owner


def daughter_hit_summary(straw_points, owner, d1_id, d2_id):
    """Per-daughter (n_distinct_stations, n_straw_hits) from the true MC hits.

    Uses the true MC hits (so it reflects the real trajectory through the field,
    energy loss and decays), not a straight-line extrapolation. Hits left by a
    daughter's descendants (e.g. after a scatter) are attributed to that daughter
    via `owner` (see daughter_owner).
    """
    s1, s2 = set(), set()
    n1 = n2 = 0
    for hit in straw_points:
        owner_id = owner.get(hit.GetTrackID())
        if owner_id == d1_id:
            s1.add(hit.GetDetectorID() // 1_000_000)
            n1 += 1
        elif owner_id == d2_id:
            s2.add(hit.GetDetectorID() // 1_000_000)
            n2 += 1
    return (len(s1), n1), (len(s2), n2)


def passes_quality(fit_track, meas_cut, chi2_cut):
    """ShipAna track quality: fit converged, Ndf >= meas_cut, chi2/Ndf < chi2_cut."""
    status = fit_track.getFitStatus()
    if not status.isFitConverged():
        return False
    ndf = status.getNdf()
    if ndf < meas_cut:
        return False
    return status.getChi2() / ndf < chi2_cut


def reconstructed_mc_ids(mc_tracks, fit_tracks, fit2mc, meas_cut, chi2_cut):
    """MC track ids reconstructed by a quality-passing fitted track.

    A fitted track is attributed to its matched MC track and, walking up the
    mother chain, to that track's ancestors (as in ship.py) -- so a daughter
    whose reco track matches a descendant still counts.
    """
    matched = {}
    for i, track in enumerate(fit_tracks):
        if not passes_quality(track, meas_cut, chi2_cut):
            continue
        mc_id = fit2mc[i]
        if mc_id >= 0:
            matched.setdefault(mc_id, i)
    front = dict(matched)
    while front:
        front = {mc_tracks[tid].GetMotherId(): i for tid, i in front.items()}
        front = {tid: i for tid, i in front.items() if tid > 0}
        matched.update(front)
    return set(matched)


def match_candidate_to_hnl(mc_tracks, fit2mc, t1, t2):
    """Both candidate tracks trace up to the *same* HNL (ShipAna match2HNL)."""
    keys = []
    for t in (t1, t2):
        mcp = fit2mc[t]
        while mcp > -0.5:
            if mcp >= len(mc_tracks):
                break
            mo = mc_tracks[mcp]
            if abs(mo.GetPdgCode()) == HNL_code:
                keys.append(mcp)
                break
            mcp = mo.GetMotherId()
    return len(keys) == 2 and keys[0] == keys[1]


def has_hnl_candidate(particles, fit_tracks, fit2mc, mc_tracks, meas_cut, doca_cut):
    """True if any reconstructed HNL candidate passes the ShipAna selection.

    With ShipAna's default fiducialCut = False the fiducial-volume cuts are
    no-ops, so the candidate selection reduces to: both daughter tracks with
    Ndf >= meas_cut, MC-matched to the same HNL, and Doca <= doca_cut.
    """
    for hnl in particles:
        t1, t2 = hnl.GetDaughter(0), hnl.GetDaughter(1)
        if fit_tracks[t1].getFitStatus().getNdf() < meas_cut:
            continue
        if fit_tracks[t2].getFitStatus().getNdf() < meas_cut:
            continue
        if not match_candidate_to_hnl(mc_tracks, fit2mc, t1, t2):
            continue
        if hnl.GetDoca() > doca_cut:
            continue
        return True
    return False


def trace_to_daughter(mc_tracks, fit2mc, fit_track_id, daughter_set):
    """The true daughter (one of `daughter_set`) a fitted track descends from, or -1.

    Walks the matched MC track's mother chain up to the first daughter ancestor,
    matching the attribution used by match_candidate_to_hnl / reconstructed_mc_ids.
    """
    mcp = fit2mc[fit_track_id]
    while mcp > -0.5:
        if mcp >= len(mc_tracks):
            break
        if mcp in daughter_set:
            return mcp
        mcp = mc_tracks[mcp].GetMotherId()
    return -1


def _stepwise_extrapolate_to_z(extrapolate_fn, current_z, target_z):
    """Genfit-extrapolate to target_z in pieces no larger than MAX_EXTRAP_STEP.

    A single long extrapolation through the field can make the Runge-Kutta
    stepper diverge; splitting it mirrors shipVertex.Task._stepwise_extrapolate_to_z
    so we reproduce the reconstruction's trajectory exactly.
    """
    dz_total = target_z - current_z
    n_steps = max(1, math.ceil(abs(dz_total) / MAX_EXTRAP_STEP))
    for istep in range(n_steps):
        step_z = current_z + dz_total * (istep + 1) / n_steps
        plane = ROOT.genfit.DetPlane(
            ROOT.TVector3(0, 0, step_z),
            ROOT.TVector3(1, 0, 0),
            ROOT.TVector3(0, 1, 0),
        )
        extrapolate_fn(ROOT.genfit.SharedPlanePtr(plane))


def daughter_state_at_z(fit_track, z_target):
    """(pos, mom) of a FitTrack genfit-extrapolated to the z_target plane.

    Reproduces exactly the per-track point the reconstruction feeds into the
    DOCA / vertex fit (shipVertex.TwoTrackVertex): the fitted state at the track's
    first measurement, RK-extrapolated *through the field* to the candidate's
    fitted vertex z-plane -- so the weak bending from residual spectrometer-field
    leakage in the decay volume (which the shipVertex DOCA explicitly accounts
    for) is included, not approximated by a straight line. Returns
    (TVector3 pos, TVector3 mom), or None if the extrapolation fails. Requires
    setup_genfit_extrapolation() to have initialised the field and material.
    """
    state = fit_track.getFittedState()
    try:
        _stepwise_extrapolate_to_z(state.extrapolateToPlane, state.getPos().Z(), z_target)
    except Exception:
        return None
    return state.getPos(), state.getMom()


def reco_hnl_for_event(particles, fit_tracks, fit2mc, mc_tracks, daughters,
                       meas_cut, doca_cut):
    """Reconstructed kinematics of the HNL candidate matched to the true HNL.

    Returns ``[mass, px, py, pz, vx, vy, vz, p1x, p1y, p1z, o1x, o1y, o1z,
    p2x, p2y, p2z, o2x, o2y, o2z]`` for the first candidate passing the ShipAna
    selection (both daughters with Ndf >= meas_cut, MC-matched to the same HNL,
    Doca <= doca_cut), or None if none passes (or a daughter extrapolation
    fails). ``mass`` is the reconstructed invariant mass and ``v`` the
    candidate's fitted (common) decay vertex. Each daughter's ``(p, o)`` is its
    fitted track genfit-extrapolated to the candidate's vertex z-plane -- the
    same per-track (momentum, position) the reconstruction feeds into the DOCA /
    vertex fit (shipVertex.TwoTrackVertex) -- ordered to match p1/p2 in ``truth``
    (by tracing each track to its true daughter). ``o1`` and ``o2`` share z = vz
    but differ in x,y by the track-to-track distance of closest approach.
    Requires setup_genfit_extrapolation() to have initialised the field.
    """
    daughter_set = set(daughters)
    for hnl in particles:
        t1, t2 = hnl.GetDaughter(0), hnl.GetDaughter(1)
        if fit_tracks[t1].getFitStatus().getNdf() < meas_cut:
            continue
        if fit_tracks[t2].getFitStatus().getNdf() < meas_cut:
            continue
        if not match_candidate_to_hnl(mc_tracks, fit2mc, t1, t2):
            continue
        if hnl.GetDoca() > doca_cut:
            continue

        # Order the daughter tracks to match p1/p2 in truth: t1 should descend
        # from daughters[0]. If the tracing is ambiguous (shouldn't happen for a
        # matched candidate) the candidate order is kept.
        if trace_to_daughter(mc_tracks, fit2mc, t1, daughter_set) == daughters[1]:
            t1, t2 = t2, t1

        pos = ROOT.TLorentzVector()
        hnl.ProductionVertex(pos)
        mom = ROOT.TLorentzVector()
        hnl.Momentum(mom)
        # Each daughter's own (pos, mom) at the candidate's fitted vertex
        # z-plane -- exactly the per-track points the reconstruction feeds into
        # the DOCA / vertex fit (shipVertex). o1/o2 share z = vz but differ in
        # x,y by the track-to-track DOCA; m1/m2 are the momenta at that plane.
        s1 = daughter_state_at_z(fit_tracks[t1], pos.Z())
        s2 = daughter_state_at_z(fit_tracks[t2], pos.Z())
        if s1 is None or s2 is None:
            return None
        o1, m1 = s1
        o2, m2 = s2
        return [
            mom.M(), mom.X(), mom.Y(), mom.Z(),
            pos.X(), pos.Y(), pos.Z(),
            m1.X(), m1.Y(), m1.Z(), o1.X(), o1.Y(), o1.Z(),
            m2.X(), m2.Y(), m2.Z(), o2.X(), o2.Y(), o2.Z(),
        ]
    return None


def crosses_finite_tracker(track, z_boundary, half_width, half_height):
    """True if `track` is born upstream of the plane and crosses inside the region."""
    pz = track.GetPz()
    dz = z_boundary - track.GetStartZ()
    if pz <= 0 or dz <= 0:  # must be born upstream and travel forward to cross
        return False
    x = track.GetStartX() + dz * track.GetPx() / pz
    y = track.GetStartY() + dz * track.GetPy() / pz
    return abs(x) < half_width and abs(y) < half_height


def particle_charge(pdg):
    """Signed electric charge in units of e, or None if the species is unknown."""
    if pdg == 22:  # photon
        return 0.0
    particle = PDG.GetParticle(pdg)
    if particle:
        return particle.Charge() / 3.0  # ROOT stores charge in units of |e|/3
    if abs(pdg) > 1_000_000_000:  # nucleus / ion: Z is encoded in the code
        z = (abs(pdg) // 10000) % 1000
        return float(z) if pdg > 0 else -float(z)
    return None  # unknown to the PDG database


def is_tracked(pdg, charge):
    """Keep photons and charged particles; drop neutrinos / neutral hadrons."""
    return pdg == 22 or (charge is not None and charge != 0)


def decay_z_map(mc_tracks):
    """z of the decay vertex for each track id (= min daughter StartZ, else +inf)."""
    decay_z = {}
    for track in mc_tracks:
        mother_id = track.GetMotherId()
        if mother_id < 0:
            continue
        z = track.GetStartZ()
        if mother_id not in decay_z or z < decay_z[mother_id]:
            decay_z[mother_id] = z
    return decay_z


def snapshot_at_boundary(track, z_boundary):
    """Crossing (x, y, t0) for a straight track reaching the boundary plane."""
    px, py, pz = track.GetPx(), track.GetPy(), track.GetPz()
    z0 = track.GetStartZ()
    dz = z_boundary - z0
    x = track.GetStartX() + dz * px / pz
    y = track.GetStartY() + dz * py / pz

    p2 = px * px + py * py + pz * pz
    mass = track.GetMass()
    energy = math.sqrt(p2 + mass * mass)
    # dt = path_length / (beta c) = dz * E / (pz * c)
    t0 = track.GetStartT() + dz * energy / (pz * C_CM_PER_NS)
    return x, y, t0, mass, px, py, pz


def process_file(input_file, geo_override, boundary_spec, event_offset, store,
                 full_mc, rec, reco, require_all4):
    """Append rows for one ROOT file; return the number of accepted events."""
    truth = store["truth"]
    particles = store["particles"]
    event_index = store["event_index"]

    z_boundary, half_width, half_height = load_geometry(
        geo_for(input_file, geo_override), boundary_spec
    )
    store["boundary"] = (z_boundary, half_width, half_height)

    f = ROOT.TFile(input_file)
    sTree = f.Get("cbmsim")
    n_entries = sTree.GetEntries()
    print(f"{input_file}: {n_entries} events, boundary z = {z_boundary:.2f} cm "
          f"(region +-{half_width:.0f} x +-{half_height:.0f} cm)")

    has_straw = bool(sTree.GetBranch("strawtubesPoint"))
    has_hits = full_mc and has_straw
    if full_mc and not has_hits:
        print("  warning: no strawtubesPoint branch; hits will be empty")
    if require_all4 and not has_straw:
        raise RuntimeError(
            f"--require-all4 needs the strawtubesPoint branch, absent in {input_file}"
        )

    # Reconstruction friend (*_rec.root): needed for the --rec efficiency and,
    # with --full-mc, for the digitized straw hits (TDC). It is mandatory for
    # --full-mc but only opportunistic for --rec.
    rec_ok = False       # report reco efficiency (--rec)
    reco_dump = False     # dump reconstructed HNL kinematics (--reco)
    has_digi = False
    if rec or full_mc or reco:
        rec_file = input_file[:-5] + "_rec.root"  # ".root" -> "_rec.root"
        if not os.path.exists(rec_file):
            if full_mc or reco:
                raise FileNotFoundError(
                    f"--full-mc/--reco require the reconstruction file {rec_file}, "
                    "but it was not found"
                )
            print(f"  warning: no {rec_file}; reco efficiency not counted")
        else:
            sTree.AddFriend("ship_reco_sim", rec_file)
            has_digi = bool(sTree.GetBranch("Digi_strawtubesHits"))
            if full_mc and not has_digi:
                raise RuntimeError(
                    f"{rec_file} has no Digi_strawtubesHits branch; cannot dump "
                    "straw TDC for --full-mc"
                )
            if rec or reco:
                use_pr = bool(sTree.GetBranch("FitTracks_PR"))
                meas_cut = MEAS_CUT_PR if use_pr else MEAS_CUT
                fit_name = "FitTracks_PR" if use_pr else "FitTracks"
                f2mc_name = "fitTrack2MC_PR" if use_pr else "fitTrack2MC"
                part_name = "Particles_PR" if use_pr else "Particles"
                rec_ok = rec
                reco_dump = reco
                if rec:
                    store["rec_available"] = True
                if reco:
                    # Needed to genfit-extrapolate FitTracks to the vertex plane.
                    setup_genfit_extrapolation(geo_for(input_file, geo_override))
                print(f"  rec friend: {rec_file} (measCut={meas_cut}, "
                      f"{'PR' if use_pr else 'truth-matched'})")

    accepted = 0
    for k in range(n_entries):
        sTree.GetEntry(k)
        mc_tracks = sTree.MCTrack

        hnl_id, daughters = find_hnl_and_daughters(mc_tracks)
        if hnl_id is None or len(daughters) != 2:
            continue

        # diagnostics from true MC hits (counts only; no filtering)
        store["n_signal"] += 1
        both_all4 = False
        if has_straw:
            owner = daughter_owner(mc_tracks, daughters)
            (ns1, nh1), (ns2, nh2) = daughter_hit_summary(
                sTree.strawtubesPoint, owner, daughters[0], daughters[1]
            )
            both_all4 = ns1 >= 4 and ns2 >= 4              # both hit all 4 stations
            if both_all4:
                store["both_all4"] += 1
            if nh1 >= MEAS_CUT and nh2 >= MEAS_CUT:         # ShipAna measurement cut
                store["both_recoqual"] += 1

        if rec_ok:
            reco_ids = reconstructed_mc_ids(
                mc_tracks, getattr(sTree, fit_name), getattr(sTree, f2mc_name),
                meas_cut, CHI2_CUT,
            )
            if daughters[0] in reco_ids and daughters[1] in reco_ids:
                store["both_reco"] += 1

            if has_hnl_candidate(
                getattr(sTree, part_name), getattr(sTree, fit_name),
                getattr(sTree, f2mc_name), mc_tracks, meas_cut, DOCA_CUT,
            ):
                store["both_candidate"] += 1

        if require_all4 and not both_all4:
            continue

        d1, d2 = (mc_tracks[i] for i in daughters)
        if not (crosses_finite_tracker(d1, z_boundary, half_width, half_height)
                and crosses_finite_tracker(d2, z_boundary, half_width, half_height)):
            continue

        # --- ground truth row -------------------------------------------------
        hnl = mc_tracks[hnl_id]
        v1 = (d1.GetStartX(), d1.GetStartY(), d1.GetStartZ())
        v2 = (d2.GetStartX(), d2.GetStartY(), d2.GetStartZ())
        assert all(
            abs(a - b) <= 8 * np.spacing(max(abs(a), abs(b), 1.0))
            for a, b in zip(v1, v2)
        ), f"daughter decay vertices differ: {v1} vs {v2}"
        vx = 0.5 * (v1[0] + v2[0])
        vy = 0.5 * (v1[1] + v2[1])
        vz = 0.5 * (v1[2] + v2[2])
        truth.append([
            hnl.GetMass(), hnl.GetPx(), hnl.GetPy(), hnl.GetPz(),
            vx, vy, vz,
            d1.GetMass(), d1.GetPx(), d1.GetPy(), d1.GetPz(),
            d2.GetMass(), d2.GetPx(), d2.GetPy(), d2.GetPz(),
        ])

        # --- reconstructed HNL candidate row (optional) -----------------------
        # One row per accepted event, aligned with `truth`; all-NaN when no
        # matched candidate passes the ShipAna selection.
        if reco_dump:
            reco_row = reco_hnl_for_event(
                getattr(sTree, part_name), getattr(sTree, fit_name),
                getattr(sTree, f2mc_name), mc_tracks, daughters,
                meas_cut, DOCA_CUT,
            )
            store["reco"].append(reco_row if reco_row is not None else [math.nan] * 19)

        # --- particle snapshots at the boundary -------------------------------
        this_event = event_offset + accepted
        decay_z = decay_z_map(mc_tracks)
        block_start = len(particles)
        snap_ids = []   # MCTrack id of each particle row appended for this event
        for track_id, track in enumerate(mc_tracks):
            if track.GetStartZ() >= z_boundary:      # born after the boundary
                continue
            if track.GetPz() <= 0:                   # not moving toward the boundary
                continue
            if decay_z.get(track_id, math.inf) <= z_boundary:  # decayed before it
                continue
            charge = particle_charge(track.GetPdgCode())
            if not is_tracked(track.GetPdgCode(), charge):     # neutrino / neutral hadron
                continue
            x, y, t0, mass, px, py, pz = snapshot_at_boundary(track, z_boundary)
            if abs(x) >= half_width or abs(y) >= half_height:  # misses the tracker
                continue
            particles.append([charge, mass, px, py, pz, x, y, t0])
            event_index.append(this_event)
            snap_ids.append(track_id)

        # The HNL daughters must lead this event's block: locate each (in p1, p2
        # order) and swap it to the front. `slot` only advances for daughters
        # that actually snapshotted, so if one is missing (neutral / decayed, or
        # only a scattered descendant crossed) the other still lands in row 0 and
        # we never index past the block (event_index is the same for the whole
        # block, so it is unaffected by the swap).
        slot = 0
        for daughter_id in daughters:
            try:
                cur = snap_ids.index(daughter_id)
            except ValueError:
                continue  # daughter did not produce a snapshot (neutral / decayed)
            a, b = block_start + slot, block_start + cur
            particles[a], particles[b] = particles[b], particles[a]
            snap_ids[slot], snap_ids[cur] = snap_ids[cur], snap_ids[slot]
            slot += 1

        # --- full MC dump (optional) -----------------------------------------
        if full_mc:
            for track_id, track in enumerate(mc_tracks):
                charge = particle_charge(track.GetPdgCode())
                store["mc_particles"].append(
                    [math.nan if charge is None else charge, track.GetMass(),
                     track.GetPx(), track.GetPy(), track.GetPz(),
                     track.GetStartX(), track.GetStartY(), track.GetStartZ()]
                )
                store["mc_info"].append(
                    [track_id, track.GetMotherId(), track.GetPdgCode(), track.GetProcID()]
                )
                store["mc_event_index"].append(this_event)
            if has_hits:
                for hit in sTree.strawtubesPoint:
                    store["hits"].append([hit.GetX(), hit.GetY(), hit.GetZ()])
                    store["hit_track"].append(hit.GetTrackID())
                    store["hit_event_index"].append(this_event)
            if has_digi:
                for digi in sTree.Digi_strawtubesHits:
                    store["digi_tdc"].append(digi.GetTDC())
                    store["digi_straw"].append([
                        digi.GetStationNumber(), digi.GetViewNumber(),
                        digi.GetLayerNumber(), digi.GetStrawNumber(),
                    ])
                    store["digi_invalid"].append(not digi.isValid())
                    store["digi_event_index"].append(this_event)

        accepted += 1

    f.Close()
    print(f"  accepted {accepted}/{n_entries} events")
    return accepted


def main():
    options = parse_args()

    store = {"truth": [], "particles": [], "event_index": [],
             "n_signal": 0, "both_all4": 0, "both_recoqual": 0,
             "both_reco": 0, "both_candidate": 0, "rec_available": False}
    if options.reco:
        store["reco"] = []
    if options.full_mc:
        store.update({
            "mc_particles": [], "mc_info": [], "mc_event_index": [],
            "hits": [], "hit_track": [], "hit_event_index": [],
            "digi_tdc": [], "digi_straw": [], "digi_invalid": [],
            "digi_event_index": [],
        })

    event_offset = 0
    for input_file in options.input_files:
        event_offset += process_file(
            input_file, options.geo_file, options.boundary,
            event_offset, store, options.full_mc, options.rec,
            options.reco, options.require_all4,
        )

    z_boundary, half_width, half_height = store["boundary"]
    arrays = {
        "truth": np.asarray(store["truth"], dtype=np.float32).reshape(-1, 15),
        "particles": np.asarray(store["particles"], dtype=np.float32).reshape(-1, 8),
        "event_index": np.asarray(store["event_index"], dtype=np.int32),
        "boundary_z": np.asarray(z_boundary, dtype=np.float32),
        "boundary_half": np.asarray([half_width, half_height], dtype=np.float32),
    }
    if options.reco:
        arrays["reco"] = np.asarray(store["reco"], dtype=np.float32).reshape(-1, 19)
    if options.full_mc:
        arrays.update({
            "mc_particles": np.asarray(store["mc_particles"], dtype=np.float32).reshape(-1, 8),
            "mc_info": np.asarray(store["mc_info"], dtype=np.int32).reshape(-1, 4),
            "mc_event_index": np.asarray(store["mc_event_index"], dtype=np.int32),
            "hits": np.asarray(store["hits"], dtype=np.float32).reshape(-1, 3),
            "hit_track": np.asarray(store["hit_track"], dtype=np.int32),
            "hit_event_index": np.asarray(store["hit_event_index"], dtype=np.int32),
            "digi_tdc": np.asarray(store["digi_tdc"], dtype=np.float32),
            "digi_straw": np.asarray(store["digi_straw"], dtype=np.int32).reshape(-1, 4),
            "digi_invalid": np.asarray(store["digi_invalid"], dtype=bool),
            "digi_event_index": np.asarray(store["digi_event_index"], dtype=np.int32),
        })

    np.savez(options.output, **arrays)
    summary = (f"{arrays['truth'].shape[0]} events, "
               f"{arrays['particles'].shape[0]} particles")
    if options.reco:
        n_reco = int(np.isfinite(arrays["reco"][:, 0]).sum())
        summary += f", {n_reco}/{arrays['reco'].shape[0]} events with a reco HNL"
    if options.full_mc:
        summary += (f", {arrays['mc_particles'].shape[0]} mc particles, "
                    f"{arrays['hits'].shape[0]} hits, "
                    f"{arrays['digi_tdc'].shape[0]} digi hits")
    print(f"wrote {options.output}: {summary}")
    print(f"events with both daughters hitting all 4 stations: "
          f"{store['both_all4']}/{store['n_signal']} signal events")
    print(f"events with both daughters >= {MEAS_CUT} straw hits "
          f"(ShipAna measurement cut): {store['both_recoqual']}/{store['n_signal']}")
    if options.rec:
        if store["rec_available"]:
            print(f"events with both daughters reconstructed "
                  f"(ShipAna quality): {store['both_reco']}/{store['n_signal']}")
            print(f"events with an accepted HNL candidate "
                  f"(+match +Doca<={DOCA_CUT:g}): {store['both_candidate']}/{store['n_signal']}")
        else:
            print("reconstruction efficiency: no *_rec.root file found")


if __name__ == "__main__":
    main()
