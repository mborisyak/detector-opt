import numpy as np

f32_array = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
i32_array = np.ndarray[tuple[int, ...], np.dtype[np.int32]]

# Init-once solver objects (opaque); built in straw.py and passed straight back to solve().
class SimParams: ...
class Layout: ...
class InputEvents: ...
class Scratch: ...

class DebugBuffers:
    # (process_ids (n, M) i32 | None, tree_int (cap,4) i32 | None, tree_float (cap,7) f32 | None,
    #  tree_event (cap,) i32 | None, tree_count (1,) i32 | None)
    def __init__(self, process_ids, tree_int, tree_float, tree_event, tree_count) -> None: ...

def solve(
    sim_params: SimParams,
    layout: Layout,
    input_events: InputEvents,
    scratch: Scratch,
    seeds: np.ndarray,  # (n,) uint32
    boundaries: i32_array,  # (n, 2)
    layers: f32_array,  # (n, n_layers)
    angles: f32_array,  # (n, n_layers)
    B: f32_array,  # (n,)
    X: f32_array,  # (n, M, 5) out
    mask: i32_array,  # (n, M) out
    z_planes: f32_array | None,  # (m,) reference z's (= max crossings per track slot)
    traj: f32_array | None,  # (n, n_tracks, m, 3) out: ordered (x, y, z) plane crossings per slot
    n_cross: i32_array | None,  # (n, n_tracks) out: number of crossings recorded per slot
    part_idx: i32_array | None,  # (n, n_tracks) out: input-particle index per slot (-1 if secondary)
    primaries: int,  # 1 -> record only primaries (slot = input index); 0 -> all particles
    debug: DebugBuffers | None,
) -> int: ...
