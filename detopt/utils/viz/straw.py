import numpy as np
import pyvista as pv

__all__ = ["show"]

try:
    from detopt.detector.straw import SparseHits, sparse_to_dense
except ImportError:
    SparseHits = None
    sparse_to_dense = None


def show(
    layers,
    angles,
    width,
    height,
    response,
    trajectories=None,
    signal=None,
    threshold=1.0,
    mask=None,
    n_particles=None,
    n_straws=None,
):
    """
    Visualize detector response and trajectories.
    
    Args:
        response: Can be:
            - Dense array: (n_particles, n_layers, n_straws) or (n_layers, n_straws)
            - SparseHits object: sparse hit representation
            - Dict with sparse arrays: {'events', 'particles', 'layers', 'straws', 'values'}
        n_particles: Required if response is SparseHits or sparse dict
        n_straws: Required if response is SparseHits or sparse dict
    """
    plotter = pv.Plotter(off_screen=False)
    plotter.camera_position = [
        (-100, 0, 50),
        (0, 0, 0),
        (0, 1, 0),
    ]

    # Handle sparse representation
    if SparseHits is not None and isinstance(response, SparseHits):
        if n_particles is None or n_straws is None:
            raise ValueError("n_particles and n_straws required when response is SparseHits")
        n_layers = len(layers)
        # Convert sparse to dense for visualization
        response_dense, _, _, _, _ = response.to_dense(1, n_particles, n_layers, n_straws)
        # Sum over particles
        combined_response = np.sum(response_dense[0], axis=0)  # (n_layers, n_straws)
    elif isinstance(response, dict) and 'values' in response:
        # Sparse arrays as dict
        if n_particles is None or n_straws is None:
            raise ValueError("n_particles and n_straws required when response is sparse dict")
        n_layers = len(layers)
        events = response['events']
        particles = response['particles']
        layers_arr = response['layers']
        straws = response['straws']
        values = response['values']
        
        # Create dense array from sparse data
        combined_response = np.zeros((n_layers, n_straws), dtype=np.float32)
        for i in range(len(events)):
            layer_idx = int(layers_arr[i])
            straw_idx = int(straws[i])
            if layer_idx < n_layers and straw_idx < n_straws:
                combined_response[layer_idx, straw_idx] += values[i]
    elif isinstance(response, np.ndarray):
        # Dense array - original logic
        if response.ndim == 3:
            _, n_layers, n_straws = response.shape
            combined_response = np.sum(response, axis=0)
        elif response.ndim == 2:
            n_layers, n_straws = response.shape
            combined_response = response
        else:
            raise ValueError(
                "response must be either 3D (per-particle) or 2D (combined) array, SparseHits, or sparse dict"
            )
    else:
        raise ValueError(
            "response must be numpy array, SparseHits object, or dict with sparse arrays"
        )

    # draw detector frames
    max_layers = len(layers)
    max_straws = 10  # just for quick visualization
    for i, l_z in enumerate(layers[:max_layers]):
        A = np.array(
            [
                [np.cos(angles[i]), np.sin(angles[i]), 0],
                [-np.sin(angles[i]), np.cos(angles[i]), 0],
                [0, 0, 1],
            ]
        )

        h, w = height[i], width[i]
        r = h / max_straws

        skew = h * np.tan(angles[i])
        verts = np.array(
            [
                [-w - skew, -h, l_z],
                [-w + skew, h, l_z],
                [w + skew, h, l_z],
                [w - skew, -h, l_z],
            ]
        )
        verts = np.dot(verts, A)
        faces = np.array([[4, 0, 1, 2, 3]])
        mesh = pv.PolyData(verts, faces=faces)
        plotter.add_mesh(
            mesh,
            color="black",
            style="wireframe",
            opacity=0.5,
            line_width=1.0,
        )

        # straws (limited)
        for k in range(max_straws):
            verts = np.array(
                [
                    [-w, 2 * r * k - h + r, l_z],
                    [w, 2 * r * k - h + r, l_z],
                ]
            )
            verts = np.dot(verts, A)
            mesh = pv.lines_from_points(verts)
            plotter.add_mesh(
                mesh,
                color=(1.0, 0.0, 0.0),
                show_edges=False,
                opacity=1.0,
            )

    # normalize mask into shape (n_particles,)
    mask_vec = None
    if mask is not None:
        mask = np.asarray(mask)
        if mask.ndim == 2:
            # assume (batch, n_particles) -> take first batch
            mask_vec = mask[0]
        elif mask.ndim == 1:
            mask_vec = mask
        else:
            raise ValueError("mask must be (n_particles,) or (batch, n_particles)")
    # --- trajectories ---
    if trajectories is not None:
        # trajectories: (n_particles, n_steps, 3)
        n_particles, n_steps, _ = trajectories.shape

        if signal is None:
            signal = 1.0

        for i in range(n_particles):
            traj = trajectories[i]  # (n_steps, 3)

            # skip all-zero trajectories
            norm = np.linalg.norm(traj, axis=1)
            nonzero_idx = np.where(norm > 1e-6)[0]
            if nonzero_idx.size == 0:
                continue

            start = nonzero_idx[0]
            sub_traj = traj[start:]  # from first non-zero point to the end

            # downsample
            n_sub = sub_traj.shape[0]
            n_samples = min(512, n_sub)
            idx = np.linspace(0, n_sub - 1, n_samples).astype(int)
            sub_traj_ds = sub_traj[idx]

            # pick color: primary vs secondary
            if mask_vec is not None and i < mask_vec.shape[0]:
                is_secondary = mask_vec[i] > 0.5
            else:
                is_secondary = False

            color = "red" if not is_secondary else "dodgerblue"

            spline = pv.Spline(sub_traj_ds)
            plotter.add_mesh(
                spline,
                color=color,
                line_width=4,
                opacity=0.6,
            )

    plotter.show_grid()
    plotter.reset_camera()
    plotter.show(screenshot="straw.png")
