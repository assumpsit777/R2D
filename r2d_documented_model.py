from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import csv
import argparse
import os
import tempfile

import numpy as np
import ufl
from basix.ufl import element, mixed_element
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import gmsh, XDMFFile

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.collections import PolyCollection

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    mtri = None
    PolyCollection = None
    HAS_MATPLOTLIB = False


# Gmsh 物理区域编号，与 battery_prlsi25_regions_coarse.geo/.msh 保持一致。
OMEGA1 = 1  # Li-SE/金属锂与固态电解质复合区域：求解 xi、phil、eta_i 和力学位移。
OMEGA2 = 2  # SE-carbon 区域：求解 phil、phis 和力学位移。
OMEGA3 = 3  # NMC 正极颗粒区域：求解归一化锂浓度 c 和力学位移；不在这里求解 phis。

GAMMA_A = 11  # 顶部锂侧边界；当前用于探针和初始锂层位置，不作为 phil 的 Dirichlet 边界。
GAMMA_B = 12  # 外侧绝缘边界标签。
GAMMA_C = 13  # 底部集流体边界；恒流电流从这里进入 phis 方程。
GAMMA_L = 14  # 侧边界标签。
GAMMA_S = 15  # Ω2-Ω3 正极颗粒界面；Butler-Volmer 反应通量 i_s 在这里耦合 phil、phis、c。

DEFAULT_MSH_FILE = "battery_prlsi25_regions_coarse.msh"
DEFAULT_GB_FILE = "grain_boundaries_final.npz"


#   eta_c = phis - phil - Eeq(theta) - V_Li+,m*sigma_h/F
E_EQ_TABLE = np.array(
    [
        [0.2228930343076256, 4.256817954840526],
        [0.23718770939025557, 4.2212385803217725],
        [0.2503742701253948, 4.198216215024365],
        [0.2635608308605341, 4.184354581468334],
        [0.2767473915956734, 4.175555457558853],
        [0.28993395233081265, 4.169287588472648],
        [0.3031205130659519, 4.163501863162304],
        [0.3163070738010912, 4.156631314356272],
        [0.3294936345362305, 4.145300935623516],
        [0.3426801952713697, 4.130836622347658],
        [0.355866756006509, 4.113841054248525],
        [0.3690533167416483, 4.09395262349422],
        [0.38223987747678756, 4.0746668724597415],
        [0.3954264382119268, 4.056104337089057],
        [0.4086129989470661, 4.037903409550268],
        [0.4217995596822054, 4.021944450569238],
        [0.43498612041734463, 4.007287279783036],
        [0.44817268115248393, 3.9945104697226936],
        [0.4613592418876232, 3.9798050845589046],
        [0.47454580262276247, 3.96497916345115],
        [0.4877323633579017, 3.9507559220632222],
        [0.500918924093041, 3.9348451774597786],
        [0.5141054848281803, 3.918090681248576],
        [0.5272920455633195, 3.901215649093408],
        [0.5404786062984588, 3.884581688826171],
        [0.5536651670335981, 3.8661396893994517],
        [0.5668517277687374, 3.850108408852042],
        [0.5800382885038766, 3.834920879912391],
        [0.593224849239016, 3.819612815028774],
        [0.6064114099741551, 3.806233325248605],
        [0.6195979707092945, 3.795023482459815],
        [0.6327845314444338, 3.7852600709986106],
        [0.6459710921795729, 3.77646094708913],
        [0.6591576529147123, 3.7660948559080984],
        [0.6723442136498515, 3.7569341241667216],
        [0.6855307743849908, 3.748376072145172],
        [0.6987173351201301, 3.7407823076753464],
        [0.7119038958552694, 3.7321037197098312],
        [0.7250904565904086, 3.724148347408109],
        [0.738277017325548, 3.7154697594425943],
        [0.7514635780606872, 3.7046215244857006],
        [0.7646501387958264, 3.69582240057622],
        [0.7778366995309657, 3.686541132890878],
        [0.791023260266105, 3.6770187933176044],
        [0.8042098210012443, 3.6672553818564],
        [0.8173963817363835, 3.6556839312357132],
        [0.8305829424715228, 3.643871408727096],
        [0.8437695032066621, 3.633505317546064],
        [0.8569560639418013, 3.6226570825891704],
        [0.8701426246769406, 3.6130142070719313],
        [0.8833291854120799, 3.603009723722796],
        [0.8965157461472192, 3.592643632541764],
        [0.9097023068823584, 3.585049868071939],
        [0.9228888676174977, 3.5790230708736646],
        [0.9351335311572699, 3.5724538619275457],
        [0.9505178520149323, 3.5661257248693574],
        [0.9624485498229155, 3.559857855783152],
        [0.9756351105580549, 3.5525051632012574],
        [0.9888216712931941, 3.541536392300398],
        [0.998240643246865, 3.488540755603573],
        [0.9994965061740211, 3.3640592684056174],
        [1.0, 3.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class Params:
    length_scale: float = 50.0e-6  # [m]
    dt: float = 0.05  # [s]
    charge_time: float = 1.0
    discharge_time: float = 0.0
    dt_min: float = 1.0e-6  # [s]
    dt_max: float = 1  # [s], conservative cap to avoid repeated failed adaptive trials.
    dt_growth: float = 1.2
    dt_shrink: float = 0.7
    dt_retry_growth: float = 1.0  # Do not grow the next step after a retry-recovered solve.
    max_retries_per_step: int = 5
    preview_dir: str = "r2d_documented_model_preview"
    diagnostics_file: str = "r2d_documented_model_preview/diagnostics.csv"
    final_npz_file: str = "r2d_documented_model_preview/final_state.npz"
    final_xdmf_file: str = "r2d_documented_model_preview/final_fields.xdmf"

    #
    li_layer_thickness: float = 5.0e-6  # [m]
    xi_interface_width: float = 1.0e-6  # [m]
    enforce_top_xi_bc: bool = False

    R: float = 8.314462618  # [J/mol/K]
    T: float = 303.15  # [K]
    F: float = 96485.33212  # [C/mol]
    alpha: float = 0.5

    current_abs: float = 10.0  # [A/m^2]
    charge_current_sign: float = 1.0
    li_side_potential: float = -0.1  # [V], initial phil from COMSOL text.m init1.phil; not a Dirichlet BC.

    sigmae: float = 1.0e7
    sigmal: float = 2.2
    sse: float = 1.85
    sigma_cathode: float = 0.17
    sigma_phi: float = 0.10

    #
    E_li: float = 7.8e9  # [Pa]
    nu_li: float = 0.381
    E_se: float = 20.0e9  # [Pa]
    nu_se: float = 0.257
    E_mix: float = 20.0e9  # [Pa]
    nu_mix: float = 0.257
    E_cathode: float = 177.5e9  # [Pa]
    nu_cathode: float = 0.253
    #
    #   eps_Li = beta_li * h(xi) * I.
    #   13.08e-6 [m^3/mol] * 76.4e3 [mol/m^3] / 3 ~= 0.333.
    beta_li: float = 0.333056
    omega_cathode: float = 3.5e-6  # [m^3/mol], V_Li+,m = 3.5 cm^3/mol.
    mechanics_residual_scale: float = 1.0e-10

    gamma: float = 0.6  # [J/m^2]
    interface_dx: float = 8.9e-7  # [m]
    omega_li_m: float = 13.08e-6
    c_li_metal: float = 76.4e3
    k_xi: float = 3.2e-6
    W_xi: float = 4.0e6
    M_li: float = 7.0e-3
    rho_li: float = 535.0
    i0_ref_li: float = 7.81  # [A/m^2]
    e_eq_li: float = 0.0  # [V], Li/Li+ anode equilibrium potential E_a^Theta.
    xi_mobility_scale: float = 1.0
    gb_xi_source_scale: float = 0.0

    #
    #   n_cathode [mol/m^2] = int_Omega3 c_li_max * c * dV.
    #
    c_li_max: float = 50.06e3  # [mol/m^3]
    c_li_ref: float = 29.1e3  # [mol/m^3]
    c_li_init: float = 50.06e3  # [mol/m^3]
    D_li: float = 5.0e-13  # [m^2/s]
    i0_c_ref: float = 7.4  # [A/m^2]
    soc_min: float = 0.222
    soc_max: float = 0.942
    soc_init: float = 0.942
    initial_cathode_overpotential: float = 0.0
    #   phis_init = li_side_potential + Eeq(soc_init) + initial_cathode_overpotential
    phis_init: float | None = 3.56  # [V], initial phis from COMSOL text.m init1.phis/ec1.phis0init.
    cathode_reaction_mode: str = "interface"

    L_gb: float = 1.5e-10
    W_gb: float = 1.0e7
    W_gb_xi: float = 2.0e7
    k_gb: float = 12.8e-7
    kappa_gb_cross: float = 0.0
    eta_mobility_scale: float = 1.0
    eta_clip_after_solve: bool = True
    b_clip: float = 0.2
    b_scale: float = 5.0

    rho: float = 0.2

    inactive_penalty: float = 1.0e-8
    exp_clip: float = 20.0
    reaction_clip: float = 1.0e4
    snes_rtol: float = 1.0e-12
    snes_atol: float = 1.0e-3
    snes_stol: float = 1.0e-12  # SNES step-size tolerance; PETSc default is commonly 1e-8.
    snes_max_it: int = 10
    snes_monitor: bool = True

    @property
    def omega_li(self) -> float:
        return self.omega_li_m

    @property
    def kappa0(self) -> float:
        return self.k_xi

    @property
    def W_b(self) -> float:
        return self.W_xi

    @property
    def L_eta(self) -> float:
        return self.i0_ref_li * self.omega_li / (6.0 * self.interface_dx * self.F)

    @property
    def L_sigma(self) -> float:
        return self.omega_li * self.L_eta / (self.R * self.T)

    @property
    def Vt(self) -> float:
        return self.R * self.T / self.F


def resolve_here(filename: str | Path) -> Path:
    path = Path(filename)
    if path.is_absolute() or path.exists():
        return path
    return Path(__file__).resolve().parent / path


# 读取 Gmsh 网格以及体/边界物理标签。
def read_mesh(msh_file: str | Path, p: Params):
    mesh_data = gmsh.read_from_msh(
        resolve_here(msh_file), MPI.COMM_WORLD, rank=0, gdim=2
    )
    msh = mesh_data.mesh
    msh.geometry.x[:, : msh.geometry.dim] /= p.length_scale
    msh.topology.create_connectivity(msh.topology.dim, 0)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim - 1)
    return msh, mesh_data.cell_tags, mesh_data.facet_tags


def h(z):
    return z**3 * (6.0 * z**2 - 15.0 * z + 10.0)


def hp(z):
    return 30.0 * z**2 * (1.0 - z) ** 2


def lame_lambda(E, nu):
    return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def lame_mu(E, nu):
    return E / (2.0 * (1.0 + nu))


def strain(u):
    return ufl.sym(ufl.grad(u))


def plane_strain_hydrostatic_stress(lam, mu, eps_tensor, eigenstrain_tensor):
    eps_eff = eps_tensor - eigenstrain_tensor
    tr_eff = ufl.tr(eps_eff)
    sigma_xx = 2.0 * mu * eps_eff[0, 0] + lam * tr_eff
    sigma_yy = 2.0 * mu * eps_eff[1, 1] + lam * tr_eff
    sigma_zz = lam * tr_eff
    return (sigma_xx + sigma_yy + sigma_zz) / 3.0


def smooth_li_profile(y, y_interface, width):
    return 0.5 * (1.0 + np.tanh((y - y_interface) / width))


def safe_exp(z, p: Params):
    return ufl.exp(ufl.max_value(-p.exp_clip, ufl.min_value(p.exp_clip, z)))


def safe_reaction(z, p: Params):
    return ufl.max_value(-p.reaction_clip, ufl.min_value(p.reaction_clip, z))


def piecewise_linear_ufl(x, table: np.ndarray):
    x0, y0 = table[0]
    x1, y1 = table[1]
    result = y0 + (y1 - y0) / (x1 - x0) * (x - x0)
    for i in range(len(table) - 1, 0, -1):
        xa, ya = table[i - 1]
        xb, yb = table[i]
        value = ya + (yb - ya) / (xb - xa) * (x - xa)
        result = ufl.conditional(ufl.le(x, xb), value, result)
    return result


def pchip_slopes(table: np.ndarray) -> np.ndarray:
    x = table[:, 0]
    y = table[:, 1]
    h_seg = np.diff(x)
    delta = np.diff(y) / h_seg
    slopes = np.zeros_like(y)

    for i in range(1, len(y) - 1):
        if delta[i - 1] == 0.0 or delta[i] == 0.0 or np.sign(delta[i - 1]) != np.sign(delta[i]):
            slopes[i] = 0.0
        else:
            w1 = 2.0 * h_seg[i] + h_seg[i - 1]
            w2 = h_seg[i] + 2.0 * h_seg[i - 1]
            slopes[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    slopes[0] = ((2.0 * h_seg[0] + h_seg[1]) * delta[0] - h_seg[0] * delta[1]) / (
        h_seg[0] + h_seg[1]
    )
    if np.sign(slopes[0]) != np.sign(delta[0]):
        slopes[0] = 0.0
    elif np.sign(delta[0]) != np.sign(delta[1]) and abs(slopes[0]) > abs(3.0 * delta[0]):
        slopes[0] = 3.0 * delta[0]

    slopes[-1] = ((2.0 * h_seg[-1] + h_seg[-2]) * delta[-1] - h_seg[-1] * delta[-2]) / (
        h_seg[-1] + h_seg[-2]
    )
    if np.sign(slopes[-1]) != np.sign(delta[-1]):
        slopes[-1] = 0.0
    elif np.sign(delta[-1]) != np.sign(delta[-2]) and abs(slopes[-1]) > abs(3.0 * delta[-1]):
        slopes[-1] = 3.0 * delta[-1]

    return slopes


E_EQ_SLOPES = pchip_slopes(E_EQ_TABLE)


def smooth_table_ufl(x, table: np.ndarray, slopes: np.ndarray):
    xa, ya = table[0]
    xb, yb = table[1]
    ma = slopes[0]
    mb = slopes[1]
    h_seg = xb - xa
    t = (x - xa) / h_seg
    result = (
        (2.0 * t**3 - 3.0 * t**2 + 1.0) * ya
        + (t**3 - 2.0 * t**2 + t) * h_seg * ma
        + (-2.0 * t**3 + 3.0 * t**2) * yb
        + (t**3 - t**2) * h_seg * mb
    )
    for i in range(len(table) - 1, 0, -1):
        xa, ya = table[i - 1]
        xb, yb = table[i]
        ma = slopes[i - 1]
        mb = slopes[i]
        h_seg = xb - xa
        t = (x - xa) / h_seg
        value = (
            (2.0 * t**3 - 3.0 * t**2 + 1.0) * ya
            + (t**3 - 2.0 * t**2 + t) * h_seg * ma
            + (-2.0 * t**3 + 3.0 * t**2) * yb
            + (t**3 - t**2) * h_seg * mb
        )
        result = ufl.conditional(ufl.le(x, xb), value, result)
    return result


def smooth_table_numpy(theta: float | np.ndarray, table: np.ndarray, slopes: np.ndarray):
    x = table[:, 0]
    y = table[:, 1]
    theta_arr = np.asarray(theta, dtype=np.float64)
    theta_clamped = np.clip(theta_arr, x[0], x[-1])
    idx = np.searchsorted(x, theta_clamped, side="right") - 1
    idx = np.clip(idx, 0, len(x) - 2)
    xa = x[idx]
    xb = x[idx + 1]
    ya = y[idx]
    yb = y[idx + 1]
    ma = slopes[idx]
    mb = slopes[idx + 1]
    h_seg = xb - xa
    t = (theta_clamped - xa) / h_seg
    value = (
        (2.0 * t**3 - 3.0 * t**2 + 1.0) * ya
        + (t**3 - 2.0 * t**2 + t) * h_seg * ma
        + (-2.0 * t**3 + 3.0 * t**2) * yb
        + (t**3 - t**2) * h_seg * mb
    )
    return float(value) if np.isscalar(theta) else value


def eeq_from_soc(theta):
    return smooth_table_ufl(theta, E_EQ_TABLE, E_EQ_SLOPES)


def eeq_from_soc_numpy(theta: float) -> float:
    return float(smooth_table_numpy(theta, E_EQ_TABLE, E_EQ_SLOPES))


def soc_from_concentration(c, p: Params):
    return c / p.c_li_max


def nearest_old_values(old_coords, old_values, new_coords, default_values, tol=1.0e-10):
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(old_coords)
        dist, idx = tree.query(new_coords, k=1)
        out = old_values[..., idx]
        far = dist > tol
    except ModuleNotFoundError:
        idx = np.empty(new_coords.shape[0], dtype=np.int64)
        dist = np.empty(new_coords.shape[0], dtype=np.float64)
        chunk = 512
        for start in range(0, new_coords.shape[0], chunk):
            stop = min(start + chunk, new_coords.shape[0])
            diff = new_coords[start:stop, None, :] - old_coords[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            local_idx = np.argmin(d2, axis=1)
            idx[start:stop] = local_idx
            dist[start:stop] = np.sqrt(d2[np.arange(stop - start), local_idx])
        out = old_values[..., idx]
        far = dist > tol
    if np.any(far):
        out = np.array(out, copy=True)
        out[..., far] = default_values[..., None]
    return out


def load_frozen_grain_fields(V, gb_file: str | Path):
    data = np.load(resolve_here(gb_file), allow_pickle=True)
    eta_values = data["eta"]
    b_values = data["B"]

    ndofs = V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts
    if b_values.shape[0] != ndofs:
        if "dof_coordinates" not in data:
            raise ValueError(
                f"GB npz has {b_values.shape[0]} dofs, but scalar space has {ndofs}, "
                "and the npz has no coordinates for remapping."
            )
        old_coords = np.asarray(data["dof_coordinates"], dtype=np.float64)
        new_coords = V.tabulate_dof_coordinates()[:, :2]
        old_ymax = float(np.max(old_coords[:, 1]))
        new_top = new_coords[:, 1] > old_ymax + 1.0e-8
        default_eta = np.zeros(eta_values.shape[0], dtype=np.float64)
        default_eta[0] = 1.0
        default_B = np.array(0.0, dtype=np.float64)
        eta_values = nearest_old_values(old_coords, eta_values, new_coords, default_eta)
        b_values = nearest_old_values(old_coords, b_values, new_coords, default_B)
        eta_values[:, new_top] = default_eta[:, None]
        b_values[new_top] = 0.0
        if V.mesh.comm.rank == 0:
            print(
                "Remapped frozen GB fields from "
                f"{old_coords.shape[0]} old dofs to {ndofs} new dofs; "
                f"{int(np.count_nonzero(new_top))} top-cap dofs set to B=0."
            )

    etas = []
    for i in range(eta_values.shape[0]):
        eta_i = fem.Function(V, name=f"eta{i + 1}")
        eta_i.x.array[:] = eta_values[i].astype(eta_i.x.array.dtype)
        eta_i.x.scatter_forward()
        etas.append(eta_i)

    B = fem.Function(V, name="B")
    B.x.array[:] = b_values.astype(B.x.array.dtype)
    B.x.scatter_forward()
    return etas, B


def load_grain_arrays(V, gb_file: str | Path):
    """Load eta_i arrays, remapping old GB files onto a slightly changed mesh."""
    etas, B = load_frozen_grain_fields(V, gb_file)
    eta_values = np.vstack([eta_i.x.array.real.copy() for eta_i in etas])
    return eta_values, B.x.array.real.copy()


def grain_boundary_indicator_expr(etas, p: Params):
    gb_overlap = sum(
        etas[i] * etas[j]
        for i in range(len(etas))
        for j in range(i + 1, len(etas))
    )
    return p.b_scale * ufl.min_value(p.b_clip, ufl.max_value(0.0, gb_overlap))


def grain_boundary_window_expr(etas, p: Params):
    return sum(
        (1.0 - eta_i)
        * ufl.conditional(ufl.lt(abs(eta_i - 0.5), p.rho), 1.0, 0.0)
        for eta_i in etas
    )


def copy_component_from_array(w, ME, component: int, values):
    _, submap = ME.sub(component).collapse()
    if values.shape[0] != submap.shape[0]:
        raise ValueError(
            f"Component {component} has {submap.shape[0]} dofs, but eta input has {values.shape[0]}."
        )
    w.x.array[submap] = values.astype(w.x.array.dtype)


def assign_component_from_expression(w, ME, component: int, expr):
    V_sub, submap = ME.sub(component).collapse()
    fun = fem.Function(V_sub)
    fun.interpolate(expr)
    fun.x.scatter_forward()
    w.x.array[submap] = fun.x.array


def update_scalar_outputs(w, scalar_outputs, p: Params | None = None):
    for i, out in enumerate(scalar_outputs):
        values = w.sub(i).collapse().x.array
        out.x.array[:] = values
        out.x.scatter_forward()


def update_interpolated(out, expr):
    points = out.function_space.element.interpolation_points
    if callable(points):
        points = points()
    out.interpolate(fem.Expression(expr, points))
    out.x.scatter_forward()


def cell_marker_array(msh, cell_tags):
    tdim = msh.topology.dim
    num_cells = msh.topology.index_map(tdim).size_local + msh.topology.index_map(tdim).num_ghosts
    markers = np.full(num_cells, -1, dtype=np.int32)
    markers[np.asarray(cell_tags.indices, dtype=np.int32)] = np.asarray(cell_tags.values, dtype=np.int32)
    return markers


def region_dofs_from_markers(V, markers, region_ids):
    region_ids = set(region_ids)
    tdim = V.mesh.topology.dim
    num_cells = V.mesh.topology.index_map(tdim).size_local + V.mesh.topology.index_map(tdim).num_ghosts
    dofmap_list = V.dofmap.list
    dofmap_array = dofmap_list.array if hasattr(dofmap_list, "array") else np.asarray(dofmap_list)
    cells = dofmap_array.reshape((num_cells, -1)).astype(np.int32)
    chosen_cells = np.flatnonzero(np.isin(markers[:num_cells], list(region_ids)))
    if chosen_cells.size == 0:
        return np.empty(0, dtype=np.int32)
    return np.unique(cells[chosen_cells].reshape(-1)).astype(np.int32)



def boundary_dofs_from_tag(V, facet_tags, tag, fallback_locator=None):
    """Return scalar-space dofs on a tagged boundary for COMSOL-like boundary probes.

    This is diagnostics only. It does not impose a Dirichlet boundary condition.
    """
    fdim = V.mesh.topology.dim - 1
    facets = facet_tags.find(tag)
    if len(facets) == 0 and fallback_locator is not None:
        facets = mesh.locate_entities_boundary(V.mesh, fdim, fallback_locator)
    if len(facets) == 0:
        return np.empty(0, dtype=np.int32)
    return fem.locate_dofs_topological(V, fdim, facets).astype(np.int32)

def cells_and_points(V, markers=None):
    msh = V.mesh
    tdim = msh.topology.dim
    num_cells = (
        msh.topology.index_map(tdim).size_local
        + msh.topology.index_map(tdim).num_ghosts
    )
    dofmap_list = V.dofmap.list
    dofmap_array = dofmap_list.array if hasattr(dofmap_list, "array") else np.asarray(dofmap_list)
    cells = dofmap_array.reshape((num_cells, -1)).astype(np.int32)
    points = V.tabulate_dof_coordinates()[:, :2]
    if cells.shape[1] not in (3, 4):
        raise ValueError(f"Unsupported plotting cell dof count {cells.shape[1]}.")
    cell_markers_out = markers[:num_cells] if markers is not None else None
    return cells, points, cell_markers_out


def restricted_cells(V, cell_markers=None, valid_regions=None):
    cells, points, markers = cells_and_points(V, cell_markers)
    if valid_regions is None or markers is None:
        return cells, points
    if isinstance(valid_regions, (int, np.integer)):
        valid_regions = (int(valid_regions),)
    keep = np.isin(markers, list(valid_regions))
    return cells[keep], points


def save_field_png(
    V,
    field,
    filename,
    title,
    cmap="viridis",
    symmetric=False,
    vmin=None,
    vmax=None,
    overlay=None,
    cell_markers=None,
    valid_regions=None,
    fill_outside=None,
):
    if V.mesh.comm.size != 1 or not HAS_MATPLOTLIB:
        return
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    values = field.x.array.real.copy()
    clim_values = values.copy()
    if fill_outside is None:
        cells, points = restricted_cells(V, cell_markers, valid_regions)
    else:
        cells, points, markers = cells_and_points(V, cell_markers)
        if valid_regions is not None and markers is not None:
            regions = (valid_regions,) if isinstance(valid_regions, (int, np.integer)) else tuple(valid_regions)
            keep_cells = np.isin(markers, regions)
            valid_dofs = np.unique(cells[keep_cells].reshape(-1)) if np.any(keep_cells) else np.empty(0, dtype=np.int32)
            invalid = np.ones(values.shape[0], dtype=bool)
            invalid[valid_dofs] = False
            values[invalid] = float(fill_outside)
            if fill_outside == 0.0:
                clim_values[invalid] = 0.0
            else:
                clim_values[invalid] = np.nan
    if cells.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7.0, 8.0), dpi=180)
    if symmetric:
        finite_values = clim_values[np.isfinite(clim_values)]
        vmax = max(float(np.max(np.abs(finite_values))) if finite_values.size else 0.0, 1.0e-30)
        vmin = -vmax
    triangles = cells if cells.shape[1] == 3 else np.vstack(
        (cells[:, [0, 1, 2]], cells[:, [0, 2, 3]])
    )
    tri = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    finite_values = clim_values[np.isfinite(clim_values)]
    try:
        if finite_values.size == 0:
            raise ValueError("no finite values to plot")
        if vmin is None:
            vmin_plot = float(np.min(finite_values))
        else:
            vmin_plot = float(vmin)
        if vmax is None:
            vmax_plot = float(np.max(finite_values))
        else:
            vmax_plot = float(vmax)
        if np.isclose(vmin_plot, vmax_plot):
            pad = max(abs(vmin_plot), 1.0) * 1.0e-9
            vmin_plot -= pad
            vmax_plot += pad
        levels = np.linspace(vmin_plot, vmax_plot, 96)
        color = ax.tricontourf(
            tri,
            values,
            levels=levels,
            cmap=cmap,
            vmin=vmin_plot,
            vmax=vmax_plot,
            extend="both",
            antialiased=False,
        )
        ax.set_xlim(float(points[:, 0].min()), float(points[:, 0].max()))
        ax.set_ylim(float(points[:, 1].min()), float(points[:, 1].max()))
    except Exception:
        color = ax.tripcolor(
            tri,
            values,
            shading="gouraud",
            cmap=cmap,
            edgecolors="none",
            linewidth=0.0,
            antialiased=False,
        )
        if vmin is not None and vmax is not None:
            color.set_clim(vmin, vmax)
    if overlay is not None:
        overlay_cells, overlay_points = restricted_cells(V, cell_markers, valid_regions)
        overlay_triangles = overlay_cells if overlay_cells.shape[1] == 3 else np.vstack(
            (overlay_cells[:, [0, 1, 2]], overlay_cells[:, [0, 2, 3]])
        )
        overlay_tri = mtri.Triangulation(overlay_points[:, 0], overlay_points[:, 1], overlay_triangles)
        ax.tricontour(
            overlay_tri,
            overlay.x.array.real,
            levels=[0.35],
            colors="black",
            linewidths=0.95,
            alpha=0.90,
        )
    ax.set_aspect("equal")
    ax.set_xlabel("x / L")
    ax.set_ylabel("y / L")
    ax.set_title(title)
    fig.colorbar(color, ax=ax)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


# 保存主要预览图。PNG 仅用于快速查看，定量后处理建议使用 XDMF/ParaView。
def save_outputs(out_dir, prefix, V, scalar_outputs, derived_outputs, B, cell_markers):
    xi_fun, phil_fun, phis_fun, c_fun, ux_fun, uy_fun = scalar_outputs[:6]
    (
        ce_fun,
        gb_window_fun,
        hxi_fun,
        sigma_fun,
        reaction_li_fun,
        deposition_drive_fun,
        xi_source_weight_fun,
        xit_fun,
        cathode_reaction_fun,
        overp_li_fun,
        overp_c_fun,
        eeq_fun,
        hydro_fun,
        overp_mech_fun,
        u_mag_fun,
        liion_source_fun,
        dfmechdxi_fun,
        eeq_eff_fun,
        hydro_cathode_fun,
        overp_mech_c_fun,
        stress_flux_drive_fun,
    ) = derived_outputs
    out_dir = Path(out_dir)
    reg_xi = (OMEGA1,)
    reg_phil = (OMEGA1, OMEGA2)
    reg_phis = (OMEGA2,)
    reg_cathode = (OMEGA3,)
    save_field_png(V, xi_fun, out_dir / f"{prefix}_xi.png", f"{prefix}: phase field xi", vmin=0.0, vmax=1.0, overlay=B, cell_markers=cell_markers, valid_regions=reg_xi )
    save_field_png(V, phil_fun, out_dir / f"{prefix}_phil.png", f"{prefix}: phil [V]", cell_markers=cell_markers, valid_regions=reg_phil )
    save_field_png(V, phis_fun, out_dir / f"{prefix}_phis.png", f"{prefix}: phis in Omega2 [V]", cell_markers=cell_markers, valid_regions=reg_phis )
    save_field_png(V, c_fun, out_dir / f"{prefix}_c_soc.png", f"{prefix}: particle Li fraction c", cell_markers=cell_markers, valid_regions=reg_cathode )
    save_field_png(V, c_fun, out_dir / f"{prefix}_c_soc_full.png", f"{prefix}: normalized Li concentration c, full domain", vmin=0.0, vmax=1.0, cell_markers=cell_markers, valid_regions=reg_cathode, fill_outside=0.0)
    save_field_png(V, hydro_fun, out_dir / f"{prefix}_hydrostatic_stress_omega1.png", f"{prefix}: Omega1 hydrostatic stress [Pa]", cmap="coolwarm", symmetric=True, overlay=B, cell_markers=cell_markers, valid_regions=reg_xi )
    save_field_png(V, hydro_cathode_fun, out_dir / f"{prefix}_hydrostatic_stress_cathode.png", f"{prefix}: cathode hydrostatic stress [Pa]", cmap="coolwarm", symmetric=True, cell_markers=cell_markers, valid_regions=reg_cathode )
    if prefix == "initial":
        save_field_png(V, B, out_dir / "initial_B.png", "initial: GB indicator B", cmap="magma", vmin=0.0, vmax=1.0, overlay=B, cell_markers=cell_markers, valid_regions=reg_xi )
    elif prefix in ("final", "after_charge", "after_discharge"):
        save_field_png(V, B, out_dir / f"{prefix}_B.png", f"{prefix}: GB indicator B", cmap="magma", vmin=0.0, vmax=1.0, overlay=B, cell_markers=cell_markers, valid_regions=reg_xi )


def save_delta_outputs(out_dir, V, initial_outputs, final_outputs, B, cell_markers):
    names = tuple(fun.name for fun in final_outputs)
    valid_regions = {
        "xi": (OMEGA1,),
        "phil": (OMEGA1, OMEGA2),
        "phis": (OMEGA2,),
        "c": (OMEGA3,),
        "ux": (OMEGA1, OMEGA2, OMEGA3),
        "uy": (OMEGA1, OMEGA2, OMEGA3),
    }
    for name, initial, final in zip(names, initial_outputs, final_outputs):
        if name.startswith("eta"):
            continue
        delta = fem.Function(final.function_space, name=f"delta_{name}")
        delta.x.array[:] = final.x.array.real - initial.x.array.real
        delta.x.scatter_forward()
        save_field_png(
            V,
            delta,
            Path(out_dir) / f"delta_{name}.png",
            f"final - initial: {name}",
            cmap="coolwarm",
            symmetric=True,
            overlay=B if name == "xi" else None,
            cell_markers=cell_markers,
            valid_regions=valid_regions.get(name),
            fill_outside=0.0,
        )


def append_csv(path, row, write_header=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def field_stats(fun, dofs=None):
    values = fun.x.array.real if dofs is None else fun.x.array.real[dofs]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(values.min()), float(values.max()), float(values.mean())


def constant_scalar_value(constant):
    value = constant.value
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def assemble_total(expr):
    return float(fem.assemble_scalar(fem.form(expr)))


def save_final_npz(path, V, scalar_outputs, derived_outputs, params):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for fun in list(scalar_outputs) + list(derived_outputs):
        arrays[fun.name] = fun.x.array.real.copy()
    arrays["dof_coordinates"] = V.tabulate_dof_coordinates()[:, :2]
    arrays["params"] = np.array(str(asdict(params)))
    np.savez(path, **arrays)


def save_final_xdmf(path, msh, scalar_outputs, derived_outputs, B, cell_tags=None, facet_tags=None):
    """保存最终场到 XDMF，供 ParaView 后处理。

    cell_region 是体区域标签，可用于 Threshold 提取 Ω1/Ω2/Ω3。
    facet_region 是边界标签，可用于检查 Γa、Γc、Γs 等边界。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [B] + list(scalar_outputs) + list(derived_outputs)
    with XDMFFile(msh.comm, str(path), "w") as xdmf:
        xdmf.write_mesh(msh)
        for fun in fields:
            xdmf.write_function(fun, 0.0)


def assign_constant(constant, value: float):
    try:
        constant.value = PETSc.ScalarType(value)
    except TypeError:
        constant.value[...] = PETSc.ScalarType(value)


def nonlinear_solver_status(problem):
    solver = getattr(problem, "solver", None)
    if solver is None:
        return "unknown", "unknown", float("nan")
    try:
        reason = solver.getConvergedReason()
    except Exception:
        reason = "unknown"
    try:
        iterations = solver.getIterationNumber()
    except Exception:
        iterations = "unknown"
    try:
        residual = float(solver.getFunctionNorm())
    except Exception:
        residual = float("nan")
    return reason, iterations, residual


def nonlinear_solver_converged(reason):
    try:
        return int(reason) > 0
    except Exception:
        return False


def cathode_interface_flux_terms(i_s, v_l, v_s, v_c, dS, dt_expr, p: Params):
    """Gamma_s 上的界面反应弱形式项。

    i_s follows the paper convention:
        i_s = i0,c*(exp(-alpha*eta/Vt) - exp((1-alpha)*eta/Vt)).
    The normal n on Gamma_s points from Omega2 to Omega3. Therefore i_s > 0
    means Li enters the cathode particle, while charge has i_s < 0.
    """
    scale_c = 1.0 / (p.F * p.c_li_max * p.length_scale)
    phil_term = -i_s * ufl.avg(v_l) * dS(GAMMA_S)
    phis_term = i_s * ufl.avg(v_s) * dS(GAMMA_S)
    c_term = -scale_c * i_s * ufl.avg(v_c) * dS(GAMMA_S)
    return phil_term, phis_term, c_term


def main(
    msh_file: str = DEFAULT_MSH_FILE,
    gb_file: str = DEFAULT_GB_FILE,
    charge_time: float | None = None,
    discharge_time: float | None = None,
    dt: float | None = None,
    cathode_reaction_mode: str = "interface",
    li_side_potential: float | None = None,
    soc_init: float | None = None,
    initial_cathode_overpotential: float | None = None,
    phis_init: float | None = None,
    charge_current_sign: float | None = None,
    current_abs: float | None = None,
    dt_max: float | None = None,
    dt_growth: float | None = None,
    dt_shrink: float | None = None,
    snes_rtol: float | None = None,
    snes_atol: float | None = None,
    snes_stol: float | None = None,
    snes_max_it: int | None = None,
    snes_monitor: bool | None = None,
):
    p = Params(cathode_reaction_mode="interface")
    if charge_time is not None:
        p = replace(p, charge_time=float(charge_time))
    if discharge_time is not None:
        p = replace(p, discharge_time=float(discharge_time))
    if dt is not None:
        p = replace(p, dt=float(dt))
    if li_side_potential is not None:
        p = replace(p, li_side_potential=float(li_side_potential))
    if soc_init is not None:
        p = replace(p, soc_init=float(np.clip(soc_init, p.soc_min, p.soc_max)))
    if initial_cathode_overpotential is not None:
        p = replace(p, initial_cathode_overpotential=float(initial_cathode_overpotential))
    if phis_init is not None:
        p = replace(p, phis_init=float(phis_init))
    if charge_current_sign is not None:
        sign = 1.0 if float(charge_current_sign) >= 0.0 else -1.0
        p = replace(p, charge_current_sign=sign)
    if current_abs is not None:
        p = replace(p, current_abs=abs(float(current_abs)))
    if dt_max is not None:
        p = replace(p, dt_max=float(dt_max))
    if dt_growth is not None:
        p = replace(p, dt_growth=float(dt_growth))
    if dt_shrink is not None:
        p = replace(p, dt_shrink=float(dt_shrink))
    if snes_rtol is not None:
        p = replace(p, snes_rtol=float(snes_rtol))
    if snes_atol is not None:
        p = replace(p, snes_atol=float(snes_atol))
    if snes_stol is not None:
        p = replace(p, snes_stol=float(snes_stol))
    if snes_max_it is not None:
        p = replace(p, snes_max_it=int(snes_max_it))
    if snes_monitor is not None:
        p = replace(p, snes_monitor=bool(snes_monitor))

    out_dir = Path(p.preview_dir)
    if MPI.COMM_WORLD.rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        diagnostics = Path(p.diagnostics_file)
        if diagnostics.exists():
            diagnostics.unlink()

    msh, cell_tags, facet_tags = read_mesh(msh_file, p)
    dx = ufl.Measure("dx", domain=msh, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    dS = ufl.Measure("dS", domain=msh, subdomain_data=facet_tags)

    V_scalar = fem.functionspace(msh, ("Lagrange", 1))
    domain_markers = cell_marker_array(msh, cell_tags)
    stats_dofs = {
        "xi": region_dofs_from_markers(V_scalar, domain_markers, (OMEGA1,)),
        "phil": region_dofs_from_markers(V_scalar, domain_markers, (OMEGA1, OMEGA2)),
        "phis": region_dofs_from_markers(V_scalar, domain_markers, (OMEGA2,)),
        "c": region_dofs_from_markers(V_scalar, domain_markers, (OMEGA3,)),
        "u": region_dofs_from_markers(V_scalar, domain_markers, (OMEGA1, OMEGA2, OMEGA3)),
    }
    boundary_probe_dofs = {
        "phil_gamma_a": boundary_dofs_from_tag(
            V_scalar,
            facet_tags,
            GAMMA_A,
            lambda X: np.isclose(X[1], float(msh.geometry.x[:, 1].max())),
        ),
        "phis_gamma_c": boundary_dofs_from_tag(
            V_scalar,
            facet_tags,
            GAMMA_C,
            lambda X: np.isclose(X[1], float(msh.geometry.x[:, 1].min())),
        ),
    }
    # Boundary probe values are diagnostics only, matching COMSOL Boundary Probe style.
    # They do not impose Dirichlet conditions on phil or phis.

    eta_initial_values, _ = load_grain_arrays(V_scalar, gb_file)
    n_grains = int(eta_initial_values.shape[0])

    P1 = element("Lagrange", msh.basix_cell(), 1)
    ME = fem.functionspace(msh, mixed_element([P1] * (6 + n_grains)))
    w = fem.Function(ME, name="state_xi_phil_phis_c_u_eta")
    w_n = fem.Function(ME, name="state_old")
    components = ufl.split(w)
    components_n = ufl.split(w_n)
    xi, phil, phis, c, ux, uy = components[:6]
    xi_n, phil_n, phis_n, c_n, ux_n, uy_n = components_n[:6]
    etas = components[6:]
    etas_n = components_n[6:]
    u_vec = ufl.as_vector((ux, uy))
    tests = ufl.TestFunctions(ME)
    v_xi, v_l, v_s, v_c, v_ux, v_uy = tests[:6]
    v_etas = tests[6:]
    v_u = ufl.as_vector((v_ux, v_uy))
    dw = ufl.TrialFunction(ME)

    # ------------------------------------------------------------------
    #
    #
    # ------------------------------------------------------------------
    y_top = float(msh.geometry.x[:, 1].max())
    thickness_hat = p.li_layer_thickness / p.length_scale
    width_hat = max(p.xi_interface_width / p.length_scale, 1.0e-4)
    e_eq_init = eeq_from_soc_numpy(p.soc_init)
    phis_init_value = (
        p.li_side_potential + e_eq_init + p.initial_cathode_overpotential
        if p.phis_init is None
        else float(p.phis_init)
    )
    eta_c_init = phis_init_value - p.li_side_potential - e_eq_init
    if msh.comm.rank == 0:
        print(
            f"initial phis = {phis_init_value:.6g} V "
            f"(phil_ref={p.li_side_potential:.6g} V, "
            f"Eeq(soc_init)={e_eq_init:.6g} V, "
            f"initial eta_c={eta_c_init:.6g} V)"
        )
    assign_component_from_expression(
        w,
        ME,
        0,
        lambda X: smooth_li_profile(X[1], y_top - thickness_hat, width_hat).astype(
            PETSc.ScalarType
        ),
    )
    assign_component_from_expression(
        w,
        ME,
        1,
        lambda X: np.full(X.shape[1], p.li_side_potential, dtype=PETSc.ScalarType),
    )
    assign_component_from_expression(
        w, ME, 2, lambda X: np.full(X.shape[1], phis_init_value, dtype=PETSc.ScalarType)
    )
    # c 只在 Omega3 正极颗粒中有物理意义；Omega1/Omega2 中设为 0，
    c_initial_values = np.zeros(
        V_scalar.dofmap.index_map.size_local + V_scalar.dofmap.index_map.num_ghosts,
        dtype=PETSc.ScalarType,
    )
    c_initial_values[stats_dofs["c"]] = PETSc.ScalarType(p.soc_init)
    copy_component_from_array(w, ME, 3, c_initial_values)
    assign_component_from_expression(
        w, ME, 4, lambda X: np.zeros(X.shape[1], dtype=PETSc.ScalarType)
    )
    assign_component_from_expression(
        w, ME, 5, lambda X: np.zeros(X.shape[1], dtype=PETSc.ScalarType)
    )
    for i in range(n_grains):
        copy_component_from_array(w, ME, 6 + i, eta_initial_values[i])
    w.x.scatter_forward()
    w_n.x.array[:] = w.x.array
    w_n.x.scatter_forward()

    dt_const = fem.Constant(msh, PETSc.ScalarType(p.dt))
    current_density = fem.Constant(msh, PETSc.ScalarType(-p.current_abs))
    i_app = fem.Constant(msh, PETSc.ScalarType(-p.current_abs))

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    Xi = ufl.variable(xi)
    f_well = Xi**2 * (1.0 - Xi) ** 2
    df_dxi = ufl.diff(f_well, Xi)
    hxi = h(xi)
    hXi = h(Xi)
    B_expr = grain_boundary_indicator_expr(etas, p)
    gb_window = grain_boundary_window_expr(etas, p)
    sum_eta_sq = sum(eta_i * eta_i for eta_i in etas)
    gb_pair_sum = sum(
        etas[i] * etas[j]
        for i in range(n_grains)
        for j in range(n_grains)
        if j != i
    )
    ce_expr = xi + (1.0 - xi) ** 2 * gb_window

    #   sigma_eff = h(xi)*sigma_Li
    #             + sum_i h(eta_i)*sigma_SE
    #             + sigma_GB*sum_{p!=q} eta_p eta_q.
    sigma_eff = (
        hxi * p.sigmae
        + sum(h(eta_i) for eta_i in etas) * p.sigmal
        + p.sigma_phi * gb_pair_sum
    )

    I2 = ufl.Identity(2)
    eps_u = strain(u_vec)
    E1 = p.E_li * hxi + p.E_se * (1.0 - hxi)
    nu1 = p.nu_li * hxi + p.nu_se * (1.0 - hxi)
    lam1 = lame_lambda(E1, nu1)
    mu1 = lame_mu(E1, nu1)
    lam2 = lame_lambda(p.E_mix, p.nu_mix)
    mu2 = lame_mu(p.E_mix, p.nu_mix)
    lam3 = lame_lambda(p.E_cathode, p.nu_cathode)
    mu3 = lame_mu(p.E_cathode, p.nu_cathode)
    #
    #   eps_eig,1 = beta_li*h(xi)*I.
    #
    #   eps_eig,3 = V_Li+,m * c_li_max * (c-c0) / 3 * I.
    eig1 = p.beta_li * hxi * I2
    eig2 = 0.0 * I2
    eig3 = (p.omega_cathode * p.c_li_max * (c - p.soc_init) / 3.0) * I2
    E1_var = p.E_li * hXi + p.E_se * (1.0 - hXi)
    nu1_var = p.nu_li * hXi + p.nu_se * (1.0 - hXi)
    lam1_var = lame_lambda(E1_var, nu1_var)
    mu1_var = lame_mu(E1_var, nu1_var)
    eig1_var = p.beta_li * hXi * I2
    eps_eff1 = eps_u - eig1
    eps_eff2 = eps_u - eig2
    eps_eff3 = eps_u - eig3
    eps_eff1_var = eps_u - eig1_var
    sigma1 = 2.0 * mu1 * eps_eff1 + lam1 * ufl.tr(eps_eff1) * I2
    sigma2 = 2.0 * mu2 * eps_eff2 + lam2 * ufl.tr(eps_eff2) * I2
    sigma3 = 2.0 * mu3 * eps_eff3 + lam3 * ufl.tr(eps_eff3) * I2
    sigma1_var = 2.0 * mu1_var * eps_eff1_var + lam1_var * ufl.tr(eps_eff1_var) * I2
    fmech1 = 0.5 * ufl.inner(sigma1_var, eps_eff1_var)
    dfmech_dxi = ufl.diff(fmech1, Xi)
    hydro1 = plane_strain_hydrostatic_stress(lam1, mu1, eps_u, eig1)
    hydro3 = plane_strain_hydrostatic_stress(lam3, mu3, eps_u, eig3)
    overp_mech_volts = hydro1 * p.omega_li / p.F
    overp_mech_c_volts = hydro3 * p.omega_cathode / p.F

    #
    #   (2 theta)^alpha [2(1-theta)]^(1-alpha).
    theta_eps = 1.0e-4
    theta = ufl.max_value(theta_eps, ufl.min_value(1.0 - theta_eps, c))
    e_eq = eeq_from_soc(theta)
    e_eq_eff = overp_mech_c_volts
    i0_c = p.i0_c_ref * (2.0 * theta) ** p.alpha * (2.0 * (1.0 - theta)) ** (
        1.0 - p.alpha
    )
    overp_c_volts = phis - phil - e_eq - e_eq_eff
    overp_c = overp_c_volts / p.Vt
    i_cathode = i0_c * (
        safe_exp(-p.alpha * overp_c, p) - safe_exp((1.0 - p.alpha) * overp_c, p)
    )
    i_cathode_hat = safe_reaction(i_cathode / p.i0_c_ref, p)
    theta_s = ufl.max_value(
        theta_eps,
        ufl.min_value(1.0 - theta_eps, ufl.avg(c)),
    )
    overp_c_volts_s = (
        ufl.avg(phis)
        - ufl.avg(phil)
        - ufl.avg(e_eq)
        - ufl.avg(overp_mech_c_volts)
    )
    overp_c_s = overp_c_volts_s / p.Vt
    i0_c_s = p.i0_c_ref * (2.0 * theta_s) ** p.alpha * (2.0 * (1.0 - theta_s)) ** (
        1.0 - p.alpha
    )
    i_cathode_s = i0_c_s * (
        safe_exp(-p.alpha * overp_c_s, p) - safe_exp((1.0 - p.alpha) * overp_c_s, p)
    )
    i_cathode_s = p.i0_c_ref * safe_reaction(i_cathode_s / p.i0_c_ref, p)

    #
    # overp_li_volts [V] = phil - E_a^Theta - V_Li,m*sigma_h/F.
    # For Li/Li+ reference, E_a^Theta defaults to 0 V.
    overp_li_volts = phil - p.e_eq_li - overp_mech_volts
    overp_li = overp_li_volts / p.Vt
    reaction_li_raw = safe_exp((1.0 - p.alpha) * overp_li, p) - ufl.min_value(
        ce_expr, 1.0
    ) * safe_exp(-p.alpha * overp_li, p)
    reaction_li = safe_reaction(reaction_li_raw, p)
    # Paper S13 uses -L_eta*(exp((1-alpha)*eta/Vt) - ce*exp(-alpha*eta/Vt));
    # therefore negative anode overpotential during charge should increase xi.
    deposition_drive = -reaction_li
    xi_source_weight = hp(xi) + p.gb_xi_source_scale * B_expr * (1.0 - hxi)
    localized_reaction_li = xi_source_weight * deposition_drive
    xit_expr = (xi - xi_n) / dt_const
    liion_source_expr = p.F * (1.0 / p.omega_li) * xit_expr
    cathode_phil_term, cathode_phis_term, cathode_c_term = cathode_interface_flux_terms(
        i_cathode_s, v_l, v_s, v_c, dS, dt_const, p
    )

    # ------------------------------------------------------------------
    #
    # ------------------------------------------------------------------

    #
    #   (xi^{n+1}-xi^n)/dt
    #   = -L_xi^sigma * deltaG/dxi
    #     - L_eta*h'(xi)*BV_anode.
    #
    F_xi = (
        (xi - xi_n) / dt_const / p.L_sigma * v_xi * dx(OMEGA1)
        + (p.kappa0 / p.length_scale**2)
        * ufl.dot(ufl.grad(xi), ufl.grad(v_xi))
        * dx(OMEGA1)
        + p.W_b * df_dxi * v_xi * dx(OMEGA1)
        + dfmech_dxi * v_xi * dx(OMEGA1)
        + p.W_gb_xi * xi * sum_eta_sq * v_xi * dx(OMEGA1)
        - p.xi_mobility_scale
        * (p.L_eta / p.L_sigma)
        * localized_reaction_li
        * v_xi
        * dx(OMEGA1)
    )

    F_phil = (
        (sigma_eff / p.length_scale) * ufl.dot(ufl.grad(phil), ufl.grad(v_l)) * dx(OMEGA1)
        + (p.sse / p.length_scale) * ufl.dot(ufl.grad(phil), ufl.grad(v_l)) * dx(OMEGA2)
        - p.length_scale * liion_source_expr * v_l * dx(OMEGA1)
        + cathode_phil_term
    )

    #
    #   div(-sigma_s grad(phi_s)) = 0.
    #
    F_phis = (
        (p.sigma_cathode / p.length_scale) * ufl.dot(ufl.grad(phis), ufl.grad(v_s)) * dx(OMEGA2)
        + cathode_phis_term
        + i_app * v_s * ds(GAMMA_C)
    )

    #
    #   dc/dt = div(D grad c)
    #           - div(D*c*V_Li+,m/(RT) grad(sigma_h))
    #
    F_c = (
        (c - c_n) / dt_const * v_c * dx(OMEGA3)
        + (p.D_li / p.length_scale**2)
        * ufl.dot(ufl.grad(c), ufl.grad(v_c))
        * dx(OMEGA3)
        - (p.D_li * p.omega_cathode / (p.R * p.T * p.length_scale**2))
        * c
        * ufl.dot(ufl.grad(hydro3), ufl.grad(v_c))
        * dx(OMEGA3)
        + cathode_c_term
    )

    #
    #   deta_i/dt = -L_phi^sigma * deltaG/deta_i.
    #
    eta_grad_coeff = p.eta_mobility_scale * p.L_gb * p.k_gb / (p.length_scale**2)
    eta_chem_coeff = p.eta_mobility_scale * p.L_gb * p.W_gb
    eta_active = 1.0 - hxi
    F_eta = 0
    for i in range(n_grains):
        eta_i = etas[i]
        eta_i_n = etas_n[i]
        v_eta_i = v_etas[i]
        cross = sum(etas[j] ** 2 for j in range(n_grains) if j != i)
        dF_deta_i = (
            2.0 * eta_i * (1.0 - eta_i) * (1.0 - 2.0 * eta_i)
            + 6.0 * eta_i * cross
        )
        F_eta += (
            eta_active * (eta_i - eta_i_n) / dt_const * v_eta_i * dx(OMEGA1)
            + eta_grad_coeff * eta_active * ufl.dot(ufl.grad(eta_i), ufl.grad(v_eta_i)) * dx(OMEGA1)
            - p.eta_mobility_scale
            * p.L_gb
            * p.kappa_gb_cross
            / (p.length_scale**2)
            * 2.0
            * eta_active
            * sum(
                ufl.dot(ufl.grad(etas[j]), ufl.grad(v_eta_i))
                for j in range(n_grains)
                if j != i
            )
            * dx(OMEGA1)
            + eta_chem_coeff * eta_active * dF_deta_i * v_eta_i * dx(OMEGA1)
            + p.eta_mobility_scale
            * p.L_gb
            * p.W_gb_xi
            * eta_active
            * xi**2
            * eta_i
            * v_eta_i
            * dx(OMEGA1)
        )

    #
    #   div(sigma)=0.
    #
    F_mech = p.mechanics_residual_scale * (
        ufl.inner(sigma1, strain(v_u)) * dx(OMEGA1)
        + ufl.inner(sigma2, strain(v_u)) * dx(OMEGA2)
        + ufl.inner(sigma3, strain(v_u)) * dx(OMEGA3)
    )

    #
    eps = fem.Constant(msh, PETSc.ScalarType(p.inactive_penalty))
    eta_inactive = sum(etas[i] * v_etas[i] for i in range(n_grains))
    F_inactive = eps * (
        (xi * v_xi + phil * v_l + phis * v_s + c * v_c) * dx(OMEGA1)
        + (xi * v_xi + c * v_c + eta_inactive) * dx(OMEGA2)
        + (xi * v_xi + phil * v_l + phis * v_s + eta_inactive) * dx(OMEGA3)
    )
    F_total = F_xi + F_phil + F_phis + F_c + F_eta + F_mech + F_inactive
    J = ufl.derivative(F_total, w, dw)

    bcs = []
    if p.enforce_top_xi_bc:
        top_facets = facet_tags.find(GAMMA_A)
        if len(top_facets) == 0:
            top_facets = mesh.locate_entities_boundary(
                msh, msh.topology.dim - 1, lambda X: np.isclose(X[1], y_top)
            )
        V_xi, _ = ME.sub(0).collapse()
        dofs_xi_top = fem.locate_dofs_topological(
            (ME.sub(0), V_xi), msh.topology.dim - 1, top_facets
        )
        xi_top_fun = fem.Function(V_xi)
        xi_top_fun.x.array[:] = 1.0
        xi_top_fun.x.scatter_forward()
        bcs.append(fem.dirichletbc(xi_top_fun, dofs_xi_top, ME.sub(0)))

    x_min = float(msh.geometry.x[:, 0].min())
    left_facets = facet_tags.find(GAMMA_L)
    if len(left_facets) == 0:
        left_facets = mesh.locate_entities_boundary(
            msh, msh.topology.dim - 1, lambda X: np.isclose(X[0], x_min)
        )
    # No Dirichlet/gauge BC is imposed for phil.
    # In text.m, init1.phil = -0.1 is an initial value only. The physical outer
    # boundaries of the phil equation are natural Neumann boundaries; Gamma_s
    # reaction flux is already included through cathode_interface_flux_terms().
    # In the fully coupled system, the common potential level is closed by the
    # xi equation, Jcell=(F/Omega_Li)*(xi^{n+1}-xi^n)/dt, and the phis current BC.


    y_min = float(msh.geometry.x[:, 1].min())
    bottom_facets = facet_tags.find(GAMMA_C)
    if len(bottom_facets) == 0:
        bottom_facets = mesh.locate_entities_boundary(
            msh, msh.topology.dim - 1, lambda X: np.isclose(X[1], y_min)
        )

    V_ux, _ = ME.sub(4).collapse()
    dofs_ux_left = fem.locate_dofs_topological(
        (ME.sub(4), V_ux), msh.topology.dim - 1, left_facets
    )
    ux_zero = fem.Function(V_ux)
    ux_zero.x.array[:] = 0.0
    ux_zero.x.scatter_forward()
    bcs.append(fem.dirichletbc(ux_zero, dofs_ux_left, ME.sub(4)))

    V_uy, _ = ME.sub(5).collapse()
    dofs_uy_bottom = fem.locate_dofs_topological(
        (ME.sub(5), V_uy), msh.topology.dim - 1, bottom_facets
    )
    uy_zero = fem.Function(V_uy)
    uy_zero.x.array[:] = 0.0
    uy_zero.x.scatter_forward()
    bcs.append(fem.dirichletbc(uy_zero, dofs_uy_bottom, ME.sub(5)))

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": p.snes_rtol,
        "snes_atol": p.snes_atol,
        "snes_stol": p.snes_stol,
        "snes_max_it": p.snes_max_it,
        "snes_converged_reason": None,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    if p.snes_monitor:
        petsc_options["snes_monitor"] = None

    jit_cache_dir = Path(tempfile.gettempdir()) / f"r2d_documented_fenics_jit_{os.getpid()}"
    jit_cache_dir.mkdir(parents=True, exist_ok=True)
    jit_options = {"cache_dir": str(jit_cache_dir), "timeout": 120}

    problem = NonlinearProblem(
        F_total,
        w,
        bcs=bcs,
        J=J,
        petsc_options_prefix="charge_discharge_full_",
        petsc_options=petsc_options,
        jit_options=jit_options,
    )

    B = fem.Function(V_scalar, name="B")
    scalar_outputs = []
    scalar_names = ["xi", "phil", "phis", "c", "ux", "uy"] + [
        f"eta{i + 1}" for i in range(n_grains)
    ]
    for i, name in enumerate(scalar_names):
        Vi, _ = ME.sub(i).collapse()
        scalar_outputs.append(fem.Function(Vi, name=name))
    update_scalar_outputs(w, scalar_outputs, p)
    initial_outputs = []
    for fun in scalar_outputs:
        copy_fun = fem.Function(fun.function_space, name=f"initial_{fun.name}")
        copy_fun.x.array[:] = fun.x.array.real
        copy_fun.x.scatter_forward()
        initial_outputs.append(copy_fun)

    derived_outputs = tuple(
        fem.Function(V_scalar, name=name)
        for name in (
            "ce",
            "gb_window",
            "hxi",
            "sigma_eff",
            "reaction_li",
            "deposition_drive",
            "xi_source_weight",
            "xit",
            "cathode_reaction",
            "overp_li",
            "overp_c",
            "eeq",
            "hydrostatic_stress",
            "overp_mech",
            "u_magnitude",
            "liion_source",
            "dfmechdxi",
            "eeq_eff",
            "hydrostatic_stress_cathode",
            "overp_mech_cathode",
            "stress_flux_drive",
        )
    )

    def update_derived():
        update_interpolated(B, B_expr)
        update_interpolated(derived_outputs[0], ce_expr)
        update_interpolated(derived_outputs[1], gb_window)
        update_interpolated(derived_outputs[2], hxi)
        update_interpolated(derived_outputs[3], sigma_eff)
        update_interpolated(derived_outputs[4], reaction_li)
        update_interpolated(derived_outputs[5], deposition_drive)
        update_interpolated(derived_outputs[6], xi_source_weight)
        update_interpolated(derived_outputs[7], xit_expr)
        update_interpolated(derived_outputs[8], i_cathode_hat)
        update_interpolated(derived_outputs[9], overp_li_volts)
        update_interpolated(derived_outputs[10], overp_c_volts)
        update_interpolated(derived_outputs[11], e_eq)
        update_interpolated(derived_outputs[12], hydro1)
        update_interpolated(derived_outputs[13], overp_mech_volts)
        update_interpolated(derived_outputs[14], ufl.sqrt(ux * ux + uy * uy))
        update_interpolated(derived_outputs[15], liion_source_expr)
        update_interpolated(derived_outputs[16], dfmech_dxi)
        update_interpolated(derived_outputs[17], e_eq_eff)
        update_interpolated(derived_outputs[18], hydro3)
        update_interpolated(derived_outputs[19], overp_mech_c_volts)
        update_interpolated(
            derived_outputs[20],
            (p.D_li * p.omega_cathode / (p.R * p.T * p.length_scale**2))
            * c
            * ufl.sqrt(ufl.dot(ufl.grad(hydro3), ufl.grad(hydro3))),
        )

    update_derived()
    save_outputs(out_dir, "initial", V_scalar, scalar_outputs, derived_outputs, B, domain_markers)
    xi_initial_mol_m2 = assemble_total(
        (xi / p.omega_li) * p.length_scale**2 * dx(OMEGA1)
    )
    c_initial_mol_m2 = assemble_total(
        p.c_li_max * c * p.length_scale**2 * dx(OMEGA3)
    )
    _, _, phil_initial_mean = field_stats(scalar_outputs[1], stats_dofs["phil"])
    _, _, phis_initial_mean = field_stats(scalar_outputs[2], stats_dofs["phis"])
    cell_voltage_initial = phis_initial_mean - phil_initial_mean
    _, _, phil_gamma_a_initial_mean = field_stats(
        scalar_outputs[1], boundary_probe_dofs["phil_gamma_a"]
    )
    _, _, phis_gamma_c_initial_mean = field_stats(
        scalar_outputs[2], boundary_probe_dofs["phis_gamma_c"]
    )
    boundary_voltage_initial = phis_gamma_c_initial_mean - phil_gamma_a_initial_mean

    def write_diag(step, time_s, phase, dt_used):
        xi_min, xi_max, xi_mean = field_stats(scalar_outputs[0], stats_dofs["xi"])
        c_min, c_max, c_mean = field_stats(scalar_outputs[3], stats_dofs["c"])
        phil_min, phil_max, phil_mean = field_stats(scalar_outputs[1], stats_dofs["phil"])
        phis_min, phis_max, phis_mean = field_stats(scalar_outputs[2], stats_dofs["phis"])
        phil_gamma_a_min, phil_gamma_a_max, phil_gamma_a_mean = field_stats(
            scalar_outputs[1], boundary_probe_dofs["phil_gamma_a"]
        )
        phis_gamma_c_min, phis_gamma_c_max, phis_gamma_c_mean = field_stats(
            scalar_outputs[2], boundary_probe_dofs["phis_gamma_c"]
        )
        boundary_voltage = phis_gamma_c_mean - phil_gamma_a_mean
        overp_li_min, overp_li_max, overp_li_mean = field_stats(
            derived_outputs[9], stats_dofs["xi"]
        )
        eeq_min, eeq_max, eeq_mean = field_stats(derived_outputs[11], stats_dofs["c"])
        overp_c_min, overp_c_max, overp_c_mean = field_stats(derived_outputs[10], stats_dofs["c"])
        eta_c_mean = phis_mean - phil_mean - eeq_mean
        terminal_voltage_proxy = phis_max - phil_min
        li_metal_mol_m2 = assemble_total(
            (xi / p.omega_li) * p.length_scale**2 * dx(OMEGA1)
        )
        cathode_li_mol_m2 = assemble_total(
            p.c_li_max * c * p.length_scale**2 * dx(OMEGA3)
        )
        li_metal_gain_mol_m2 = li_metal_mol_m2 - xi_initial_mol_m2
        cathode_li_delta_mol_m2 = cathode_li_mol_m2 - c_initial_mol_m2
        cathode_li_loss_mol_m2 = -cathode_li_delta_mol_m2
        li_balance_error_mol_m2 = li_metal_gain_mol_m2 + cathode_li_delta_mol_m2
        denom = max(abs(li_metal_gain_mol_m2), abs(cathode_li_delta_mol_m2), 1.0e-30)
        loss_denom = max(abs(cathode_li_loss_mol_m2), 1.0e-30)
        # 即时守恒检测：这些量直接来自当前时间步方程中的通量/源项。
        # 网格坐标使用 x/L，所以线积分需要乘 L，面积积分需要乘 L^2，
        # 才能得到二维单位厚度下的物理量。
        gamma_s_current_A_m = assemble_total(
            i_cathode_s * p.length_scale * dS(GAMMA_S)
        )
        gamma_c_current_A_m = assemble_total(
            i_app * p.length_scale * ds(GAMMA_C)
        )
        omega1_xi_current_A_m = assemble_total(
            (p.F / p.omega_li) * xit_expr * p.length_scale**2 * dx(OMEGA1)
        )
        gamma_s_li_rate_mol_m_s = gamma_s_current_A_m / p.F
        gamma_c_li_rate_mol_m_s = gamma_c_current_A_m / p.F
        omega1_xi_li_rate_mol_m_s = omega1_xi_current_A_m / p.F
        gamma_s_plus_gamma_c_A_m = gamma_s_current_A_m + gamma_c_current_A_m
        xi_plus_gamma_s_A_m = omega1_xi_current_A_m + gamma_s_current_A_m
        current_balance_scale = max(
            abs(gamma_s_current_A_m),
            abs(gamma_c_current_A_m),
            abs(omega1_xi_current_A_m),
            1.0e-30,
        )

        row = {
            "step": step,
            "time_s": time_s,
            "dt_s": dt_used,
            "phase": phase,
            "current_density_A_m2": constant_scalar_value(current_density),
            "cell_voltage_est_V": phis_mean - phil_mean,
            "cell_voltage_delta_V": (phis_mean - phil_mean) - cell_voltage_initial,
            "terminal_voltage_proxy_V": terminal_voltage_proxy,
            "boundary_voltage_probe_V": boundary_voltage,
            "boundary_voltage_probe_delta_V": boundary_voltage - boundary_voltage_initial,
            "phil_gamma_a_min_V": phil_gamma_a_min,
            "phil_gamma_a_max_V": phil_gamma_a_max,
            "phil_gamma_a_mean_V": phil_gamma_a_mean,
            "phil_gamma_a_mean_delta_V": phil_gamma_a_mean - phil_gamma_a_initial_mean,
            "phis_gamma_c_min_V": phis_gamma_c_min,
            "phis_gamma_c_max_V": phis_gamma_c_max,
            "phis_gamma_c_mean_V": phis_gamma_c_mean,
            "phis_gamma_c_mean_delta_V": phis_gamma_c_mean - phis_gamma_c_initial_mean,
            "eeq_min_V": eeq_min,
            "eeq_max_V": eeq_max,
            "eeq_mean_V": eeq_mean,
            "eta_c_mean_no_mech_V": eta_c_mean,
            "eta_a_min_V": overp_li_min,
            "eta_a_max_V": overp_li_max,
            "eta_a_mean_V": overp_li_mean,
            "overp_c_min_V": overp_c_min,
            "overp_c_max_V": overp_c_max,
            "overp_c_mean_V": overp_c_mean,
            "xi_min": xi_min,
            "xi_max": xi_max,
            "xi_mean": xi_mean,
            "c_min": c_min,
            "c_max": c_max,
            "c_mean": c_mean,
            "phil_min": phil_min,
            "phil_max": phil_max,
            "phil_mean": phil_mean,
            "phil_mean_delta_V": phil_mean - phil_initial_mean,
            "phis_min": phis_min,
            "phis_max": phis_max,
            "phis_mean": phis_mean,
            "phis_mean_delta_V": phis_mean - phis_initial_mean,
            "li_metal_mol_m2": li_metal_mol_m2,
            "cathode_li_mol_m2": cathode_li_mol_m2,
            "li_metal_gain_mol_m2": li_metal_gain_mol_m2,
            "cathode_li_delta_mol_m2": cathode_li_delta_mol_m2,
            "cathode_li_loss_mol_m2": cathode_li_loss_mol_m2,
            "li_gain_to_cathode_loss_ratio": li_metal_gain_mol_m2 / loss_denom,
            "li_balance_error_mol_m2": li_balance_error_mol_m2,
            "li_balance_rel_error": li_balance_error_mol_m2 / denom,
            "gamma_s_current_A_m": gamma_s_current_A_m,
            "gamma_c_current_A_m": gamma_c_current_A_m,
            "omega1_xi_current_A_m": omega1_xi_current_A_m,
            "gamma_s_li_rate_mol_m_s": gamma_s_li_rate_mol_m_s,
            "gamma_c_li_rate_mol_m_s": gamma_c_li_rate_mol_m_s,
            "omega1_xi_li_rate_mol_m_s": omega1_xi_li_rate_mol_m_s,
            "gamma_s_plus_gamma_c_A_m": gamma_s_plus_gamma_c_A_m,
            "xi_plus_gamma_s_A_m": xi_plus_gamma_s_A_m,
            "gamma_s_plus_gamma_c_rel": gamma_s_plus_gamma_c_A_m / current_balance_scale,
            "xi_plus_gamma_s_rel": xi_plus_gamma_s_A_m / current_balance_scale,
        }
        if msh.comm.rank == 0:
            append_csv(p.diagnostics_file, row, write_header=(step == 0))
        return row

    write_diag(0, 0.0, "initial", p.dt)

    phases = []
    if p.charge_time > 0.0:
        phases.append(("charge", p.charge_time, p.charge_current_sign * p.current_abs))
    if p.discharge_time > 0.0:
        phases.append(("discharge", p.discharge_time, p.current_abs))
    total_step = 0
    time_s = 0.0
    _, xi_map = ME.sub(0).collapse()
    _, c_map = ME.sub(3).collapse()
    eta_maps = [ME.sub(6 + i).collapse()[1] for i in range(n_grains)]

    for phase_name, duration, current_value in phases:
        if duration <= 0.0:
            continue
        assign_constant(current_density, current_value)
        assign_constant(i_app, current_value)

        phase_elapsed = 0.0
        accepted_steps = 0
        dt_trial = min(max(p.dt, p.dt_min), p.dt_max)
        while phase_elapsed < duration - 1.0e-15:
            dt_used = min(dt_trial, duration - phase_elapsed)
            assign_constant(dt_const, dt_used)
            trial_success = False

            retries_used = 0
            for _retry in range(p.max_retries_per_step):
                w.x.array[:] = w_n.x.array
                w.x.scatter_forward()
                try:
                    problem.solve()
                    w.x.scatter_forward()
                    snes_reason, snes_its, snes_residual = nonlinear_solver_status(problem)
                    if isinstance(snes_reason, (int, np.integer)) and snes_reason <= 0:
                        raise RuntimeError(
                            f"SNES did not converge, reason={snes_reason}, "
                            f"its={snes_its}, fnorm={snes_residual:.3e}"
                        )
                    trial_success = True
                    break
                except Exception as exc:
                    if msh.comm.rank == 0:
                        print(
                            f"retry at t={time_s:.3e}s with dt={dt_used:.3e}s "
                            f"after: {exc}"
                        )
                    dt_used *= p.dt_shrink
                    if dt_used < p.dt_min:
                        raise RuntimeError(
                            f"{phase_name} phase failed: dt dropped below dt_min={p.dt_min:g} s"
                        )
                    assign_constant(dt_const, dt_used)

            if not trial_success:
                raise RuntimeError(f"{phase_name} phase could not find a converged adaptive step.")

            # 不再对 xi 和 c 做求解后的硬裁剪。
            # 硬裁剪不是弱形式的一部分，会改变 xi/c 的总量，直接破坏
            # “正极 Li 损失 = Omega1 Li 金属增加”的积分守恒检测。
            if p.eta_clip_after_solve:
                for i, eta_map in enumerate(eta_maps):
                    eta_sub = w.sub(6 + i).collapse()
                    eta_sub.x.array[:] = np.clip(eta_sub.x.array.real, 0.0, 1.0)
                    w.x.array[eta_map] = eta_sub.x.array
            w.x.scatter_forward()

            accepted_steps += 1
            total_step += 1
            phase_elapsed += dt_used
            time_s += dt_used
            next_growth = p.dt_retry_growth if retries_used > 0 else p.dt_growth
            dt_trial = min(max(dt_used * next_growth, p.dt_min), p.dt_max)

            update_scalar_outputs(w, scalar_outputs, p)
            update_derived()
            diag_row = write_diag(total_step, time_s, phase_name, dt_used)
            if msh.comm.rank == 0:
                print(
                    f"{phase_name}: accepted_step={accepted_steps}, "
                    f"phase_t={phase_elapsed:.3f}/{duration:.3f} s, "
                    f"global_t={time_s:.3f} s, dt={dt_used:.4f} s, "
                    f"I={current_value:.3g} A/m^2, "
                    f"SNES reason={snes_reason}, its={snes_its}, "
                    f"fnorm={snes_residual:.3e}, "
                    f"Gs={diag_row['gamma_s_current_A_m']:.3e} A/m, "
                    f"Gc={diag_row['gamma_c_current_A_m']:.3e} A/m, "
                    f"Xi={diag_row['omega1_xi_current_A_m']:.3e} A/m, "
                    f"Xi+Gs={diag_row['xi_plus_gamma_s_A_m']:.3e} A/m"
                )

            # 诊断写完后再更新旧解；否则 (xi-xi_n)/dt 会被错误地算成 0。
            w_n.x.array[:] = w.x.array
            w_n.x.scatter_forward()

        update_scalar_outputs(w, scalar_outputs, p)
        update_derived()
        save_outputs(out_dir, f"after_{phase_name}", V_scalar, scalar_outputs, derived_outputs, B, domain_markers)

    update_scalar_outputs(w, scalar_outputs, p)
    update_derived()
    save_outputs(out_dir, "final", V_scalar, scalar_outputs, derived_outputs, B, domain_markers)
    save_delta_outputs(out_dir, V_scalar, initial_outputs, scalar_outputs, B, domain_markers)
    save_final_npz(p.final_npz_file, V_scalar, scalar_outputs, derived_outputs, p)
    save_final_xdmf(p.final_xdmf_file, msh, scalar_outputs, derived_outputs, B, cell_tags, facet_tags)

    if msh.comm.rank == 0:
        print(f"已保存预览图到: {out_dir}")
        print(f"已保存诊断 CSV: {p.diagnostics_file}")
        print(f"已保存最终状态 NPZ: {p.final_npz_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="R2D documented split model with evolving grain boundaries and interface fluxes."
    )
    parser.add_argument("--msh", default=DEFAULT_MSH_FILE)
    parser.add_argument("--gb", default=DEFAULT_GB_FILE)
    parser.add_argument("--t-end", type=float, default=None, help="Charge time in seconds.")
    parser.add_argument("--discharge-time", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--li-side-potential", type=float, default=None)
    parser.add_argument("--soc-init", type=float, default=None)
    parser.add_argument("--initial-cathode-overpotential", type=float, default=None)
    parser.add_argument("--phis-init", type=float, default=None)
    parser.add_argument(
        "--charge-current-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=None,
        help="Sign for charge current; default keeps Params. Use +1 for sign checks.",
    )
    parser.add_argument("--current-density", type=float, default=None, help="Absolute applied current density [A/m^2].")
    parser.add_argument("--dt-max", type=float, default=None, help="Maximum adaptive time step [s].")
    parser.add_argument("--dt-growth", type=float, default=None, help="Growth factor after a clean converged step.")
    parser.add_argument("--dt-shrink", type=float, default=None, help="Shrink factor after a failed nonlinear trial.")
    parser.add_argument("--snes-rtol", type=float, default=None)
    parser.add_argument("--snes-atol", type=float, default=None)
    parser.add_argument("--snes-stol", type=float, default=None)
    parser.add_argument("--snes-max-it", type=int, default=None)
    parser.add_argument("--no-snes-monitor", action="store_true")
    args = parser.parse_args()
    main(
        msh_file=args.msh,
        gb_file=args.gb,
        charge_time=args.t_end,
        discharge_time=args.discharge_time,
        dt=args.dt,
        li_side_potential=args.li_side_potential,
        soc_init=args.soc_init,
        initial_cathode_overpotential=args.initial_cathode_overpotential,
        phis_init=args.phis_init,
        charge_current_sign=args.charge_current_sign,
        current_abs=args.current_density,
        dt_max=args.dt_max,
        dt_growth=args.dt_growth,
        dt_shrink=args.dt_shrink,
        snes_rtol=args.snes_rtol,
        snes_atol=args.snes_atol,
        snes_stol=args.snes_stol,
        snes_max_it=args.snes_max_it,
        snes_monitor=False if args.no_snes_monitor else None,
    )
















