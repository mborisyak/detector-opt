import numpy as np

# ---------- Parameters ----------
W = 26.0  # eV per ion pair
Fano = 0.2

vd = 0.0033  # drift velocity [mm/ns]
DL = 0.012   # longitudinal diffusion [sqrt(mm)] -> ns
DT = 0.012   # transverse diffusion [sqrt(mm)]

G_mean = 2e4   # mean gas gain
theta = 0.5    # Polya parameter

tau_r = 2.0    # ns, rise time
tau_f = 20.0   # ns, fall time

def primary_ionisation(Edep_mev, rng=np.random):
    """Return number of primary electrons with Fano smearing."""
    N_mean = Edep_mev * 1e6 / W
    N = rng.normal(N_mean, np.sqrt(Fano*N_mean))
    return max(int(N), 0)

def drift_time(r_mm, rng=np.random):
    """Drift time + diffusion jitter."""
    t_mean = r_mm / vd
    sigma_t = DL * np.sqrt(r_mm) / vd
    return rng.normal(t_mean, sigma_t)

def avalanche_gain(n_electrons, rng=np.random):
    """Gas gain sampled from Polya distribution."""
    if n_electrons == 0:
        return 0.0
    # Polya ~ Gamma(k=1+θ, scale=G_mean/(1+θ))
    gains = rng.gamma(1+theta, G_mean/(1+theta), size=n_electrons)
    return np.sum(gains)

def pulse_shape(t, Q):
    """Electronics shaping function for charge Q (Coulombs)."""
    return Q * (1 - np.exp(-t/tau_r)) * np.exp(-t/tau_f) * (t>0)

def straw_response(Edep_mev, r_mm, t0=0.0, rng=np.random):
    """
    Full straw response to one step: returns waveform arrays.
    Edep_mev: energy deposit in MeV
    r_mm: distance from wire in mm
    t0: reference time (ns)
    rng: numpy random generator
    Returns: t (ns), s (Coulombs)
    """
    Ne = primary_ionisation(Edep_mev, rng)
    t_d = drift_time(r_mm, rng)
    total_gain = avalanche_gain(Ne, rng)
    print("IN   ", Ne, t_d, total_gain)
    Q = 1.6e-19 * total_gain  # Coulombs
    t = np.linspace(0, 2000, 20000)  # ns
    s = pulse_shape(t - (t0 + t_d), Q)

    return t, s

def fairship_fdigi(
    t0_event, t_MC, r_mm, x_hit, x_readout, sigma_spatial=0.012, v_drift=0.0033, c=29.9792, rng=np.random
):
    """
    Compute FairShip-style TDC time (fdigi) for a straw hit.
    Args:
        t0_event: trigger/reference time (ns)
        t_MC: MC time at hit (ns)
        r_mm: true distance to wire (mm)
        x_hit: hit position along wire (cm)
        x_readout: readout end position along wire (cm)
        sigma_spatial: spatial resolution (mm)
        v_drift: drift velocity (mm/ns)
        c: signal propagation speed (cm/ns)
        rng: numpy random generator
    Returns:
        fdigi: TDC time (ns)
    """
    t_drift = abs(rng.normal(r_mm, sigma_spatial)) / v_drift
    L_prop = abs(x_readout - x_hit)  # in cm
    t_wire = L_prop / c
    return t0_event + t_MC + t_drift + t_wire

def compute_tdc_times(waveforms, t0_arr, r_mm, straw_length=400.0, v_wire=0.2, t0_event=0.0, rng=np.random):
    """
    Compute FairShip-style TDC times (fdigi) for each hit.
    Args:
        waveforms: dict of {(event, particle, layer, straw): (t, s)}
        t0_arr: array of MC hit times (ns), shape (events, particles, layers, straws)
        r_mm: array of drift distances (mm), shape (events, particles, layers, straws)
        straw_length: straw length in mm
        v_wire: signal propagation speed along wire (mm/ns)
        t0_event: event reference time (ns)
        rng: numpy random generator
    Returns:
        tdc_times: dict {(event, particle, layer, straw): fdigi}
    """
    tdc_times = {}
    for key in waveforms:
        event, particle, layer, straw = key
        t_MC = t0_arr[event, particle, layer, straw]
        r_mm_val = r_mm[event, particle, layer, straw]
        drift = drift_time(r_mm_val, rng)
        # For demo: randomize hit position along wire
        along_wire_distance = rng.uniform(-straw_length/2, straw_length/2)
        t_wire = abs(along_wire_distance) / v_wire
        fdigi = t0_event + t_MC + drift + t_wire
        tdc_times[key] = fdigi
    return tdc_times
