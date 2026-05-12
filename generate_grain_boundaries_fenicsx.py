from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    mtri = None
    HAS_MATPLOTLIB = False
import numpy as np
import ufl
from basix.ufl import element, mixed_element
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, io
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import gmsh


# Gmsh physical tags, must match battery_prlsi25_regions_coarse.msh.
OMEGA1 = 1  # Li metal / solid electrolyte phase-field region
OMEGA2 = 2  # SE-carbon mixture region
OMEGA3 = 3  # Cathode particle region

DEFAULT_MSH_FILE = "battery_prlsi25_regions_coarse.msh"


@dataclass(frozen=True)
class GrainParams:
    # Number of solid electrolyte grains represented by order parameters eta_i.
    n_grains: int = 10

    # Geometry scale. Mesh coordinates are nondimensionalized as x_hat = x / L.
    length_scale: float = 50.0e-6  # [m]

    # Allen-Cahn annealing parameters from the COMSOL/PRL setup.
    L_phi: float = 1.5e-8  # [m*s/kg], grain phase-field mobility
    W_phi: float = 1.025e7  # [J/m^3], multiwell barrier height
    k_phi: float = 12.8e-7  # [J/m], gradient coefficient

    # Time integration. These are physical seconds before nondimensionalization.
    time_scale: float = 1.0  # [s]
    dt: float = 10.0  # [s], matches COMSOL range(0,10,3600)
    t_end: float = 4000.0  # [s], full COMSOL-like grain annealing time

    # Initial columnar grain seeds.
    seed: int = 7
    initializer: str = "comsol_random"  # "comsol_random" or "columnar"
    grain_excluded_top_thickness: float = 5.0e-6  # [m], initial Li cap above SE grains.
    interface_width: float = 0.015  # nondimensional tanh transition width
    boundary_wiggle: float = 0.025  # nondimensional horizontal waviness
    boundary_drift: float = 0.035  # nondimensional linear tilt amplitude
    comsol_random_grid: int = 16  # coarse random lattice used to mimic COMSOL rn_i

    # Weakly constrain eta fields outside Omega1 because this debug script uses
    # full-domain mixed spaces while the grain equations only live in Omega1.
    inactive_penalty: float = 1.0e-8
    clip_eta_after_solve: bool = True

    # Clamp only for post-processing B. The PDE itself remains smooth.
    b_clip: float = 0.2
    b_scale: float = 5.0


def resolve_msh_file(msh_file: str | Path) -> Path:
    path = Path(msh_file)
    if path.is_absolute() or path.exists():
        return path
    return Path(__file__).resolve().parent / path


def read_mesh(msh_file: str | Path, p: GrainParams):
    mesh_data = gmsh.read_from_msh(
        resolve_msh_file(msh_file), MPI.COMM_WORLD, rank=0, gdim=2
    )
    msh = mesh_data.mesh
    msh.geometry.x[:, : msh.geometry.dim] /= p.length_scale
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim - 1)
    msh.topology.create_connectivity(msh.topology.dim, 0)
    return msh, cell_tags, facet_tags


def print_tag_report(comm, cell_tags, facet_tags):
    if comm.rank != 0:
        return
    print("Cell tags in mesh:", sorted(set(cell_tags.values.tolist())))
    print("Facet tags in mesh:", sorted(set(facet_tags.values.tolist())))


def tagged_cell_bbox(msh, cell_tags, tag: int):
    """Bounding box of one physical cell region."""
    tdim = msh.topology.dim
    c_to_v = msh.topology.connectivity(tdim, 0)
    cells = cell_tags.find(tag)
    if len(cells) == 0:
        raise ValueError(f"No cells found for physical tag {tag}.")

    vertices = np.unique(np.hstack([c_to_v.links(int(cell)) for cell in cells]))
    coords = msh.geometry.x[vertices, :2]
    return coords.min(axis=0), coords.max(axis=0)


def grain_y_cut(msh, p: GrainParams):
    """Top of the grain-generation window, excluding the added Li cap."""
    return float(msh.geometry.x[:, 1].max()) - p.grain_excluded_top_thickness / p.length_scale


def build_columnar_grain_initializers(
    msh, cell_tags, p: GrainParams, tag: int = OMEGA1, y_cut: float | None = None
):
    """Create COMSOL-like columnar eta_i initializers.

    The earlier Voronoi initializer creates radial/fan-shaped domains. The PRL
    figure and the COMSOL-exported setup are closer to nearly vertical grains
    with slightly curved boundaries, so each eta_i is initialized as a smooth
    stripe between two perturbed vertical boundary curves.
    """
    (x_min, y_min), (x_max, y_max) = tagged_cell_bbox(msh, cell_tags, tag)
    if y_cut is not None:
        y_max = min(y_max, y_cut)
    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span <= 0.0 or y_span <= 0.0:
        raise ValueError("Mesh bounding box is degenerate; cannot seed grains.")

    rng = np.random.default_rng(p.seed)
    base = np.linspace(x_min, x_max, p.n_grains + 1)
    spacing = x_span / p.n_grains

    internal_jitter = rng.uniform(-0.18 * spacing, 0.18 * spacing, p.n_grains - 1)
    base[1:-1] += internal_jitter
    base[1:-1] = np.sort(base[1:-1])

    phases = rng.uniform(0.0, 2.0 * np.pi, p.n_grains + 1)
    freqs = rng.uniform(1.2, 3.0, p.n_grains + 1)
    amps = rng.uniform(0.45, 1.0, p.n_grains + 1) * p.boundary_wiggle * x_span
    drifts = rng.uniform(-1.0, 1.0, p.n_grains + 1) * p.boundary_drift * x_span
    width = max(p.interface_width * x_span, 1.0e-8)

    def boundary_curve(k: int, y):
        if k == 0:
            return np.full_like(y, x_min - 4.0 * width)
        if k == p.n_grains:
            return np.full_like(y, x_max + 4.0 * width)

        yn = (y - y_min) / y_span
        curve = (
            base[k]
            + amps[k] * np.sin(2.0 * np.pi * freqs[k] * yn + phases[k])
            + drifts[k] * (yn - 0.5)
        )
        left_limit = base[k - 1] + 0.25 * spacing
        right_limit = base[k + 1] - 0.25 * spacing
        return np.clip(curve, left_limit, right_limit)

    def smooth_step(s):
        return 0.5 * (1.0 + np.tanh(s / width))

    def make_expr(i: int):
        def expr(X):
            x = X[0]
            y = X[1]
            left = boundary_curve(i, y)
            right = boundary_curve(i + 1, y)
            value = smooth_step(x - left) * smooth_step(right - x)
            if y_cut is not None:
                value = np.where(y <= y_cut, value, 1.0 if i == 0 else 0.0)
            return value.astype(PETSc.ScalarType)

        return expr

    return [make_expr(i) for i in range(p.n_grains)]


def build_comsol_random_initializers(
    msh, cell_tags, p: GrainParams, tag: int = OMEGA1, y_cut: float | None = None
):
    """Approximate the rn/an random initializer exported from COMSOL.

    COMSOL defines random functions rn1...rn9 and analytic functions an1...an10:
      eta_i(x,y) = 1 - an_i(x/W, y/H)
    The an_i definitions allocate each point to the first successful random
    bin, and an10 receives the leftover points. A coarse random lattice is used
    here so the initial grains are nuclei/patches rather than node-scale noise.
    """
    (x_min, y_min), (x_max, y_max) = tagged_cell_bbox(msh, cell_tags, tag)
    if y_cut is not None:
        y_max = min(y_max, y_cut)
    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span <= 0.0 or y_span <= 0.0:
        raise ValueError("Mesh bounding box is degenerate; cannot seed grains.")

    rng = np.random.default_rng(p.seed)
    ngrid = p.comsol_random_grid

    # COMSOL's rn_i are independent seeded random functions with mean 1 and
    # uniformrange = 10, 9, ..., 2. We use U(1-range/2, 1+range/2), which gives
    # the same threshold probabilities implied by floor(min(rn_i + offset_i, 1)).
    random_fields = [
        rng.uniform(1.0 - r / 2.0, 1.0 + r / 2.0, size=(ngrid, ngrid))
        for r in range(10, 1, -1)
    ]

    def nearest_random(field, x, y):
        xi = np.clip(((x - x_min) / x_span * (ngrid - 1)).astype(np.int64), 0, ngrid - 1)
        yi = np.clip(((y - y_min) / y_span * (ngrid - 1)).astype(np.int64), 0, ngrid - 1)
        return field[xi, yi]

    def labels_at_points(x, y):
        assigned = np.zeros_like(x, dtype=bool)
        labels = np.full_like(x, p.n_grains - 1, dtype=np.int64)
        offsets = np.array([4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0])

        for i, offset in enumerate(offsets):
            rn = nearest_random(random_fields[i], x, y)
            # floor(min(rn + offset, 1)) is 0 only when rn + offset < 1.
            eta_i_is_one = (~assigned) & (rn + offset < 1.0)
            labels[eta_i_is_one] = i
            assigned |= eta_i_is_one

        return labels

    def make_expr(i: int):
        def expr(X):
            labels = labels_at_points(X[0], X[1])
            value = np.where(labels == i, 1.0, 0.0)
            if y_cut is not None:
                value = np.where(X[1] <= y_cut, value, 1.0 if i == 0 else 0.0)
            return value.astype(PETSc.ScalarType)

        return expr

    return [make_expr(i) for i in range(p.n_grains)]


def build_grain_initializers(
    msh, cell_tags, p: GrainParams, tag: int = OMEGA1, y_cut: float | None = None
):
    """Select the initial grain generator."""
    if p.initializer == "comsol_random":
        return build_comsol_random_initializers(msh, cell_tags, p, tag=tag, y_cut=y_cut)
    if p.initializer == "columnar":
        return build_columnar_grain_initializers(msh, cell_tags, p, tag=tag, y_cut=y_cut)
    raise ValueError(
        f"Unknown initializer {p.initializer!r}; use 'comsol_random' or 'columnar'."
    )


def normalize_eta_components(w, ME, n_grains: int):
    """Normalize nodal eta values so sum_i eta_i is approximately one."""
    collapsed = []
    maps = []
    for i in range(n_grains):
        _, submap = ME.sub(i).collapse()
        collapsed.append(w.x.array[submap].copy())
        maps.append(submap)

    total = np.sum(np.vstack(collapsed), axis=0)
    total[total < 1.0e-12] = 1.0
    for values, submap in zip(collapsed, maps):
        w.x.array[submap] = values / total


def clip_eta_components(w):
    """Keep eta values in the physical range used by COMSOL's bounded solve."""
    w.x.array[:] = np.clip(w.x.array.real, 0.0, 1.0).astype(w.x.array.dtype)
    w.x.scatter_forward()


def set_top_cap_single_grain(w, ME, p: GrainParams, y_cut: float):
    """Keep the added top cap out of the grain-boundary model."""
    for i in range(p.n_grains):
        Vc, submap = ME.sub(i).collapse()
        coords = Vc.tabulate_dof_coordinates()[:, :2]
        cap = coords[:, 1] > y_cut
        if np.any(cap):
            values = w.x.array[submap].copy()
            values[cap] = 1.0 if i == 0 else 0.0
            w.x.array[submap] = values
    w.x.scatter_forward()


def assign_component_from_expression(w, ME, component: int, expr):
    """Interpolate an initializer into one component of a mixed function."""
    Vc, submap = ME.sub(component).collapse()
    f = fem.Function(Vc)
    f.interpolate(expr)
    w.x.array[submap] = f.x.array


def split_to_scalar_functions(w, V_out, names):
    """Project mixed eta components to one common scalar output space.

    Do not copy collapsed-subspace arrays directly into the plotting/output
    space. Different collapsed spaces can have different dof orderings, which
    makes a correct columnar field look like a fan-shaped artifact in plots.
    """
    scalar_outputs = []
    points = V_out.element.interpolation_points
    if callable(points):
        points = points()
    components = ufl.split(w)
    for i, name in enumerate(names):
        out = fem.Function(V_out, name=name)
        expr = fem.Expression(components[i], points)
        out.interpolate(expr)
        out.x.scatter_forward()
        scalar_outputs.append(out)
    return scalar_outputs


def update_scalar_functions(w, V_out, scalar_outputs):
    """Refresh scalar eta outputs by interpolation from the mixed solution."""
    points = V_out.element.interpolation_points
    if callable(points):
        points = points()
    components = ufl.split(w)
    for i, out in enumerate(scalar_outputs):
        expr = fem.Expression(components[i], points)
        out.interpolate(expr)
        out.x.scatter_forward()


def clamp_scalar_outputs(functions):
    """Clamp post-processing copies so saved eta_i stay in [0, 1]."""
    for fun in functions:
        fun.x.array[:] = np.clip(fun.x.array.real, 0.0, 1.0).astype(fun.x.array.dtype)
        fun.x.scatter_forward()


def set_top_cap_scalar_outputs(eta_outputs, p: GrainParams, y_cut: float):
    for i, fun in enumerate(eta_outputs):
        coords = fun.function_space.tabulate_dof_coordinates()[:, :2]
        cap = coords[:, 1] > y_cut
        if np.any(cap):
            fun.x.array[cap] = 1.0 if i == 0 else 0.0
            fun.x.scatter_forward()


def interpolate_grain_boundary_indicator(msh, etas, V_out, p: GrainParams, y_cut=None):
    """Compute B = 5 * min(0.2, max(0, sum_{i<j} eta_i eta_j))."""
    b_raw = sum(
        etas[i] * etas[j]
        for i in range(p.n_grains)
        for j in range(i + 1, p.n_grains)
    )
    b_expr = p.b_scale * ufl.min_value(p.b_clip, ufl.max_value(0.0, b_raw))
    b_fun = fem.Function(V_out, name="grain_boundary_B")
    points = V_out.element.interpolation_points
    if callable(points):
        points = points()
    expr = fem.Expression(b_expr, points)
    b_fun.interpolate(expr)
    if y_cut is not None:
        coords = V_out.tabulate_dof_coordinates()[:, :2]
        b_fun.x.array[coords[:, 1] > y_cut] = 0.0
    b_fun.x.scatter_forward()
    return b_fun


def make_triangulation(V, cell_tags=None, tag=None):
    """Build a triangulation whose node numbering matches Function.x.array."""
    cells, points, mask = cells_and_dof_points(V, cell_tags=cell_tags, tag=tag)
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], cells)
    if mask is not None:
        triangulation.set_mask(mask)
    return triangulation


def save_field_png(V, field, filename, title, cmap="viridis", cell_tags=None, tag=None):
    """Save one scalar finite-element field as a PNG preview."""
    if V.mesh.comm.size != 1:
        if V.mesh.comm.rank == 0:
            print(f"Skip {filename}: matplotlib preview expects one MPI rank.")
        return

    if not HAS_MATPLOTLIB:
        svg_name = str(Path(filename).with_suffix(".svg"))
        save_scalar_svg(
            V,
            field.x.array.real,
            svg_name,
            title,
            palette=cmap,
            cell_tags=cell_tags,
            tag=tag,
        )
        return

    triangulation = make_triangulation(V, cell_tags=cell_tags, tag=tag)
    values = field.x.array.real

    fig, ax = plt.subplots(figsize=(7.2, 8.4), dpi=180)
    color = ax.tripcolor(triangulation, values, shading="gouraud", cmap=cmap)
    ax.set_aspect("equal")
    ax.set_xlabel("x / L")
    ax.set_ylabel("y / L")
    ax.set_title(title)
    fig.colorbar(color, ax=ax)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def save_grain_label_png(
    V, eta_outputs, filename="grain_labels.png", cell_tags=None, tag=None
):
    """Save argmax_i eta_i as a quick grain-domain preview."""
    if V.mesh.comm.size != 1:
        if V.mesh.comm.rank == 0:
            print(f"Skip {filename}: matplotlib preview expects one MPI rank.")
        return

    eta_values = np.vstack([eta_i.x.array.real for eta_i in eta_outputs])
    labels = np.argmax(eta_values, axis=0)

    if not HAS_MATPLOTLIB:
        svg_name = str(Path(filename).with_suffix(".svg"))
        save_label_svg(
            V,
            labels,
            svg_name,
            "Grain labels: argmax eta_i",
            cell_tags=cell_tags,
            tag=tag,
        )
        return

    triangulation = make_triangulation(V, cell_tags=cell_tags, tag=tag)

    fig, ax = plt.subplots(figsize=(7.2, 8.4), dpi=180)
    color = ax.tripcolor(triangulation, labels, shading="flat", cmap="tab20")
    ax.set_aspect("equal")
    ax.set_xlabel("x / L")
    ax.set_ylabel("y / L")
    ax.set_title("Grain labels: argmax eta_i")
    fig.colorbar(color, ax=ax, ticks=range(len(eta_outputs)))
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def save_boundary_preview_png(
    V, eta_outputs, filename="grain_boundary_preview.png", cell_tags=None, tag=None
):
    """Save an intentionally high-contrast grain-boundary preview.

    COMSOL's physical B uses overlap eta_i eta_j, which can be very small on a
    coarse mesh. For visual checking, 1 - max_i(eta_i) highlights transition
    bands even when the physical B image looks dark.
    """
    if V.mesh.comm.size != 1:
        if V.mesh.comm.rank == 0:
            print(f"Skip {filename}: matplotlib preview expects one MPI rank.")
        return

    eta_values = np.vstack([eta_i.x.array.real for eta_i in eta_outputs])
    preview = 1.0 - np.max(eta_values, axis=0)
    if np.max(preview) > 0.0:
        preview = preview / np.max(preview)

    if not HAS_MATPLOTLIB:
        svg_name = str(Path(filename).with_suffix(".svg"))
        save_scalar_svg(
            V,
            preview,
            svg_name,
            "Visual grain-boundary preview: 1 - max eta_i",
            palette="Reds",
            cell_tags=cell_tags,
            tag=tag,
        )
        return

    triangulation = make_triangulation(V, cell_tags=cell_tags, tag=tag)

    fig, ax = plt.subplots(figsize=(7.2, 8.4), dpi=180)
    color = ax.tripcolor(triangulation, preview, shading="gouraud", cmap="Reds")
    ax.set_aspect("equal")
    ax.set_xlabel("x / L")
    ax.set_ylabel("y / L")
    ax.set_title("Visual grain-boundary preview: 1 - max eta_i")
    fig.colorbar(color, ax=ax)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def cells_and_dof_points(V, cell_tags=None, tag=None):
    """Return cell dofs and coordinates using the same numbering as field values.

    Matplotlib's tripcolor expects the triangle connectivity to index the same
    array used for scalar values. Using plot.vtk_mesh(V) can reorder points for
    visualization, while Function.x.array is ordered by finite-element dofs.
    That mismatch produces the fan/radiating artifact seen in the preview.
    """
    msh = V.mesh
    tdim = msh.topology.dim
    index_map = msh.topology.index_map(tdim)
    num_cells = index_map.size_local + index_map.num_ghosts
    cells = np.array(
        [V.dofmap.cell_dofs(cell) for cell in range(num_cells)],
        dtype=np.int32,
    )
    points = V.tabulate_dof_coordinates()[:, :2]

    # Basix dof order on quadrilateral cells is not guaranteed to be geometric
    # counter-clockwise order. Sort each cell's dofs by angle before plotting;
    # otherwise the preview can contain artificial crossed triangles.
    cell_points = points[cells]
    centers = np.mean(cell_points, axis=1)
    angles = np.arctan2(
        cell_points[:, :, 1] - centers[:, None, 1],
        cell_points[:, :, 0] - centers[:, None, 0],
    )
    order = np.argsort(angles, axis=1)
    cells = np.take_along_axis(cells, order, axis=1)

    cell_mask = None
    if cell_tags is not None and tag is not None:
        selected = np.zeros(num_cells, dtype=bool)
        tagged_cells = cell_tags.find(tag)
        tagged_cells = tagged_cells[tagged_cells < num_cells]
        selected[tagged_cells] = True
        cell_mask = ~selected

    if cells.shape[1] == 3:
        triangles = cells
        triangle_mask = cell_mask
    elif cells.shape[1] == 4:
        # Matplotlib only supports triangular cells. The gmsh battery mesh uses
        # bilinear quads, so split each quad into two triangles for previewing.
        triangles = np.vstack(
            (
                cells[:, [0, 1, 2]],
                cells[:, [0, 2, 3]],
            )
        ).astype(np.int32)
        triangle_mask = (
            np.concatenate((cell_mask, cell_mask)) if cell_mask is not None else None
        )
    else:
        raise ValueError(
            "Preview plotting only supports P1 triangles or Q1 quadrilaterals; "
            f"got {cells.shape[1]} dofs per cell."
        )

    return triangles, points, triangle_mask


def svg_project(points, width=900.0, margin=36.0):
    """Project nondimensional mesh coordinates into an SVG viewport."""
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    span_x = max(x_max - x_min, 1.0e-12)
    span_y = max(y_max - y_min, 1.0e-12)
    scale = (width - 2.0 * margin) / span_x
    height = span_y * scale + 2.0 * margin

    xy = np.empty_like(points)
    xy[:, 0] = margin + (points[:, 0] - x_min) * scale
    xy[:, 1] = height - margin - (points[:, 1] - y_min) * scale
    return xy, width, height


def scalar_color(value, vmin, vmax, palette):
    """Small dependency-free colormaps used when matplotlib is unavailable."""
    if vmax <= vmin:
        s = 0.0
    else:
        s = float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))

    if palette == "Reds":
        r = 255
        g = int(248 - 210 * s)
        b = int(240 - 225 * s)
    elif palette == "inferno":
        r = int(12 + 240 * s)
        g = int(7 + 120 * s**1.8)
        b = int(60 * (1.0 - s) + 15 * s)
    else:
        r = int(45 + 30 * s)
        g = int(75 + 155 * s)
        b = int(120 + 60 * (1.0 - s))
    return f"rgb({r},{g},{b})"


def save_scalar_svg(
    V, values, filename, title, palette="viridis", cell_tags=None, tag=None
):
    """Dependency-free SVG scalar preview for environments without matplotlib."""
    cells, points, mask = cells_and_dof_points(V, cell_tags=cell_tags, tag=tag)
    if mask is not None:
        cells = cells[~mask]
    xy, width, height = svg_project(points)
    cell_values = np.mean(values[cells], axis=1)
    vmin = float(np.nanmin(cell_values))
    vmax = float(np.nanmax(cell_values))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
        f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="24" font-size="18" font-family="sans-serif">{title}</text>',
    ]
    for tri, value in zip(cells, cell_values):
        coords = " ".join(f"{xy[k, 0]:.2f},{xy[k, 1]:.2f}" for k in tri)
        color = scalar_color(value, vmin, vmax, palette)
        parts.append(f'<polygon points="{coords}" fill="{color}" stroke="{color}" stroke-width="0.2"/>')
    parts.append("</svg>")
    Path(filename).write_text("\n".join(parts), encoding="utf-8")


def save_label_svg(V, labels, filename, title, cell_tags=None, tag=None):
    """Dependency-free SVG grain-label preview."""
    cells, points, mask = cells_and_dof_points(V, cell_tags=cell_tags, tag=tag)
    if mask is not None:
        cells = cells[~mask]
    xy, width, height = svg_project(points)
    palette = [
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#59A14F",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC",
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
        f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="24" font-size="18" font-family="sans-serif">{title}</text>',
    ]
    for tri in cells:
        label = int(np.bincount(labels[tri]).argmax())
        coords = " ".join(f"{xy[k, 0]:.2f},{xy[k, 1]:.2f}" for k in tri)
        color = palette[label % len(palette)]
        parts.append(f'<polygon points="{coords}" fill="{color}" stroke="{color}" stroke-width="0.2"/>')
    parts.append("</svg>")
    Path(filename).write_text("\n".join(parts), encoding="utf-8")


def save_final_npz(filename, V_out, eta_outputs, b_fun, p: GrainParams, t: float):
    """Save final grain fields in a compact format for the next FEniCSx solver.

    This assumes the next script uses the same mesh and the same first-order
    scalar function space. The arrays can then be copied directly into matching
    dolfinx Function.x.array buffers.
    """
    if V_out.mesh.comm.rank != 0:
        return

    eta_values = np.vstack([eta_i.x.array.real for eta_i in eta_outputs])
    np.savez(
        filename,
        t=np.array(t, dtype=np.float64),
        length_scale=np.array(p.length_scale, dtype=np.float64),
        dof_coordinates=V_out.tabulate_dof_coordinates()[:, :2],
        eta=eta_values,
        B=b_fun.x.array.real.copy(),
        names=np.array([eta_i.name for eta_i in eta_outputs] + [b_fun.name]),
    )


def main(
    msh_file: str = DEFAULT_MSH_FILE,
    out_file: str = "grain_boundaries.bp",
):
    p = GrainParams()
    msh, cell_tags, facet_tags = read_mesh(msh_file, p)
    print_tag_report(msh.comm, cell_tags, facet_tags)

    dx = ufl.Measure("dx", domain=msh, subdomain_data=cell_tags)
    y_cut = grain_y_cut(msh, p)
    y = ufl.SpatialCoordinate(msh)[1]
    active_grain_window = ufl.conditional(ufl.le(y, y_cut), 1.0, 0.0)

    # Mixed space for eta1...eta10. These fields describe solid electrolyte
    # grains; their overlap region after annealing is treated as grain boundary.
    P1 = element("Lagrange", msh.basix_cell(), 1)
    ME = fem.functionspace(msh, mixed_element([P1] * p.n_grains))

    eta = fem.Function(ME, name="eta")
    eta_n = fem.Function(ME, name="eta_old")
    etas = ufl.split(eta)
    etas_n = ufl.split(eta_n)
    tests = ufl.TestFunctions(ME)
    deta = ufl.TrialFunction(ME)

    # Grain seeds. "comsol_random" follows the rn/an initializer in text.m;
    # "columnar" is an artificial visual helper for nearly vertical grains.
    with eta.x.petsc_vec.localForm() as loc:
        loc.set(0.0)
    for i, initializer in enumerate(
        build_grain_initializers(msh, cell_tags, p, tag=OMEGA1, y_cut=y_cut)
    ):
        assign_component_from_expression(eta, ME, i, initializer)
    normalize_eta_components(eta, ME, p.n_grains)
    set_top_cap_single_grain(eta, ME, p, y_cut)
    eta.x.scatter_forward()
    if p.clip_eta_after_solve:
        clip_eta_components(eta)
        set_top_cap_single_grain(eta, ME, p, y_cut)
    eta_n.x.array[:] = eta.x.array

    dt_hat = fem.Constant(msh, PETSc.ScalarType(p.dt / p.time_scale))

    # Nondimensional coefficients after x_hat = x/L and t_hat = t/t0.
    ac_grad = p.time_scale * p.L_phi * p.k_phi / (p.length_scale**2)
    ac_chem = p.time_scale * p.L_phi * p.W_phi

    # COMSOL equation form for each eta_i:
    #   d eta_i / dt =
    #       L_phi*k_phi*Delta(eta_i)
    #       - L_phi*(-W_phi*eta_i + W_phi*eta_i^3
    #                + 2 W_phi eta_i sum_{j != i} eta_j^2)
    #
    # Weak backward-Euler residual on Omega1:
    #   int (eta_i-eta_i_n)/dt * v_i dx
    # + int ac_grad grad(eta_i).grad(v_i) dx
    # + int ac_chem*(-eta_i + eta_i^3 + 2 eta_i sum eta_j^2) * v_i dx = 0
    #
    # Mechanical coupling and xi coupling are omitted in this preprocessing
    # version. The goal here is only to generate a stable grain-boundary field.
    F_eta = 0
    for i in range(p.n_grains):
        eta_i = etas[i]
        eta_i_n = etas_n[i]
        v_i = tests[i]
        cross = sum(etas[j] ** 2 for j in range(p.n_grains) if j != i)
        dF_deta_i = -eta_i + eta_i**3 + 2.0 * eta_i * cross
        F_eta += (
            (eta_i - eta_i_n) / dt_hat * v_i * active_grain_window * dx(OMEGA1)
            + ac_grad * ufl.dot(ufl.grad(eta_i), ufl.grad(v_i)) * active_grain_window * dx(OMEGA1)
            + ac_chem * dF_deta_i * v_i * active_grain_window * dx(OMEGA1)
        )

    # Weakly pin eta outside the grain-generation window to avoid singular rows
    # from full-domain DOFs and to keep the added top cap grain-free.
    eps = fem.Constant(msh, PETSc.ScalarType(p.inactive_penalty))
    F_inactive = sum(
        eps * etas[i] * tests[i] * (dx(OMEGA2) + dx(OMEGA3))
        + eps
        * (etas[i] - (1.0 if i == 0 else 0.0))
        * tests[i]
        * (1.0 - active_grain_window)
        * dx(OMEGA1)
        for i in range(p.n_grains)
    )

    F_total = F_eta + F_inactive
    J = ufl.derivative(F_total, eta, deta)

    problem = NonlinearProblem(
        F_total,
        eta,
        bcs=[],
        J=J,
        petsc_options_prefix="grain_anneal_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1.0e-7,
            "snes_atol": 1.0e-9,
            "snes_max_it": 30,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )

    V_out = fem.functionspace(msh, ("Lagrange", 1))
    eta_outputs = split_to_scalar_functions(
        eta, V_out, [f"eta{i + 1}" for i in range(p.n_grains)]
    )
    if p.clip_eta_after_solve:
        clamp_scalar_outputs(eta_outputs)
    set_top_cap_scalar_outputs(eta_outputs, p, y_cut)
    b_fun = interpolate_grain_boundary_indicator(msh, etas, V_out, p, y_cut=y_cut)
    outputs = eta_outputs + [b_fun]

    writer = io.VTXWriter(msh.comm, out_file, outputs, engine="BP4")
    t = 0.0
    writer.write(t)

    # Save the initial map separately. This isolates seeding/plotting mistakes
    # from possible Allen-Cahn evolution issues.
    save_grain_label_png(
        V_out,
        eta_outputs,
        filename="initial_grain_labels.png",
        cell_tags=cell_tags,
        tag=OMEGA1,
    )
    save_boundary_preview_png(
        V_out,
        eta_outputs,
        filename="initial_grain_boundary_preview.png",
        cell_tags=cell_tags,
        tag=OMEGA1,
    )

    step = 0
    while t < p.t_end - 1.0e-14:
        problem.solve()
        reason = problem.solver.getConvergedReason()
        its = problem.solver.getIterationNumber()
        if reason <= 0:
            raise RuntimeError(
                f"Grain annealing SNES failed at t={t:g}, "
                f"iterations={its}, reason={reason}"
            )

        eta.x.scatter_forward()
        if p.clip_eta_after_solve:
            clip_eta_components(eta)
            set_top_cap_single_grain(eta, ME, p, y_cut)
        eta_n.x.array[:] = eta.x.array
        update_scalar_functions(eta, V_out, eta_outputs)
        if p.clip_eta_after_solve:
            clamp_scalar_outputs(eta_outputs)
        set_top_cap_scalar_outputs(eta_outputs, p, y_cut)
        b_fun = interpolate_grain_boundary_indicator(msh, etas, V_out, p, y_cut=y_cut)
        outputs[-1].x.array[:] = b_fun.x.array
        outputs[-1].x.scatter_forward()

        t += p.dt
        step += 1
        writer.write(t)

        if msh.comm.rank == 0:
            b_int = fem.assemble_scalar(fem.form(outputs[-1] * dx(OMEGA1)))
            print(
                f"step={step}, t={t:.4e} s, SNES iterations={its}, "
                f"reason={reason}, int_B={b_int:.6e}"
            )

    writer.close()
    save_final_npz("grain_boundaries_final.npz", V_out, eta_outputs, outputs[-1], p, t)

    # PNG previews for quick verification without ParaView.
    save_field_png(
        V_out,
        outputs[-1],
        "grain_boundary_B.png",
        "Grain-boundary indicator B",
        cmap="inferno",
        cell_tags=cell_tags,
        tag=OMEGA1,
    )
    save_grain_label_png(V_out, eta_outputs, cell_tags=cell_tags, tag=OMEGA1)
    save_boundary_preview_png(V_out, eta_outputs, cell_tags=cell_tags, tag=OMEGA1)
    if msh.comm.rank == 0:
        print(
            "Saved previews: grain_boundary_B.png, grain_labels.png, "
            "grain_boundary_preview.png"
        )


if __name__ == "__main__":
    main()
