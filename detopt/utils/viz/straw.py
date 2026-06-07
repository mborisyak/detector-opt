import numpy as np
import pyvista as pv

__all__ = ["show"]

# Import SparseHits lazily to avoid circular import


def show(
    layers,
    angles,
    width,
    height,
    hits,
    masses,
    trajectories=None,
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
        n_particles: Required if response is SparseHits
        n_straws: Required if response is SparseHits
    """
    plotter = pv.Plotter(off_screen=False)
    plotter.camera_position = [
        (-100, 0, 50),
        (0, 0, 0),
        (0, 1, 0),
    ]

    if True:
        if n_particles is None or n_straws is None:
            raise ValueError("n_particles and n_straws required when response is SparseHits")
        n_layers = len(layers)
        # Convert sparse to dense for visualization
        # response_dense, _, _, _, _ = response.to_dense(
        #     1, n_particles, n_layers, n_straws
        # )
        # # Sum over particles
        # combined_response = np.sum(response_dense[0], axis=0)  # (n_layers, n_straws)

    # draw detector frames
    max_layers = len(layers)
    max_straws = 10  # just for quick visualization

    print(hits)
    print(len(hits))

    hit_map = [[] for _ in range(len(layers))]

    for key in hits.keys():  # hits is your dict
        layer_idx = int(key[2])
        straw = int(key[3])
        hit_map[layer_idx].append(straw)
    print(trajectories.shape)
    input("wa")

    # if masses.ndim == 2:
    #     # assume (batch, n_particles)
    #     mass_vec = masses[0]
    # elif masses.ndim == 1:
    #     # assume (n_particles,)
    #     mass_vec = masses
    # else:
    #     raise ValueError("masses must be (n_particles,) or (batch, n_particles)")

    # # round so nearly-equal float masses collapse to one particle type
    # mass_keys = np.round(mass_vec.astype(float), 6)
    # unique_masses = np.unique(mass_keys)

    # palette = [
    #     "red",
    #     "dodgerblue",
    #     "limegreen",
    #     "gold",
    #     "magenta",
    #     "cyan",
    #     "orange",
    #     "white",
    # ]

    # mass_to_color = {
    #     m: palette[j % len(palette)]
    #     for j, m in enumerate(unique_masses)
    # }

    # optional: dedupe + sort
    hit_map = [sorted(set(straws)) for straws in hit_map]
    print(hit_map)
    input("wait hit")
    for i, l_z in enumerate(layers[:max_layers]):
        print(i, l_z)
        A = np.array(
            [
                [np.cos(angles[i]), np.sin(angles[i]), 0],
                [-np.sin(angles[i]), np.cos(angles[i]), 0],
                [0, 0, 1],
            ]
        )

        h, w = height[i], width[i]
        r = h / max_straws
        r_hit = h / n_straws

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
        if i < len(hit_map):
            for k_hit in hit_map[i]:
                if 0 <= k_hit < n_straws:
                    hit_pts = np.array(
                        [
                            [-w, 2 * r_hit * k_hit - h + r_hit, l_z],
                            [w, 2 * r_hit * k_hit - h + r_hit, l_z],
                        ]
                    )
                    hit_pts = np.dot(hit_pts, A)

                    hit_mesh = pv.lines_from_points(hit_pts).tube(
                        radius=10 * r_hit,  # tune this
                        n_sides=16,
                    )
                    plotter.add_mesh(
                        hit_mesh,
                        color="yellow",
                        opacity=1.0,
                        show_edges=False,
                    )

    # normalize mask into shape (n_particles,)
    mask_vec = None
    mask = np.asarray(mask)
    masses = np.asarray(masses)
    all_mass_keys = np.round(masses.astype(float).ravel(), 6)
    unique_masses = np.unique(all_mass_keys)

    palette = [
        "red",
        "dodgerblue",
        "limegreen",
        "gold",
        "magenta",
        "cyan",
        "orange",
        "white",
    ]

    mass_to_color = {m: palette[j % len(palette)] for j, m in enumerate(unique_masses)}
    print("mass_to_color:")
    for m, c in mass_to_color.items():
        print(f"  mass={m:.6f} -> {c}")

    # --- trajectories ---
    if trajectories is not None:
        # trajectories: (n_particles, n_steps, 3)
        print(trajectories.shape)
        _, n_particles, n_steps, _ = trajectories.shape

        # if signal is None:
        #     signal = 1.0
        for ev in range(trajectories.shape[0]):
            # mask_vec = mask[i]
            # mass_vec = masses[i]
            for i in range(n_particles):
                traj = trajectories[ev][i]  # (n_steps, 3)
                print(traj.shape)
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
                # if mask_vec is not None and i < mask_vec.shape[0]:
                #     is_secondary = mask_vec[i] > 0.5
                # else:
                #     is_secondary = False

                m = masses[ev, i]
                m_key = np.round(float(m), 6)
                color = mass_to_color.get(m_key, "white")

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
