"""Unit-consistency tests for the grid free energy (GFE) bulk-density term.

The box volume reported by GROMACS/OpenMM-style statistics is in nm^3, while the
histogram voxel is gridsize (in Angstrom) cubed. The bulk reference density must
convert the box volume to A^3 before dividing by the voxel volume; otherwise every
GFE is offset by +kT ln(1000) (about +4.118 kcal/mol at 300 K).

Route used: `cosolvkit.analysis` cannot be imported in this environment (its eager
imports need pymol, MDAnalysis, gridData and openff, none of which are installed in
the naurmalade uv or pixi envs). So the function under test is extracted from the
source file with `ast` and executed with numpy only. This exercises the real source
text of `_grid_free_energy`, not a re-implementation.
"""
import ast
import math
import os

import numpy as np

ANALYSIS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cosolvkit",
    "analysis.py",
)

BOLTZMANN_KCAL = 0.0019872041  # kcal/(mol K), matches cosolvkit.analysis.BOLTZMANN_CONSTANT_KB
KT_300_KCAL = BOLTZMANN_KCAL * 300.0
OFFSET_KCAL = KT_300_KCAL * math.log(1000.0)  # ~ 4.118 kcal/mol


def _load_grid_free_energy():
    with open(ANALYSIS_PATH, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=ANALYSIS_PATH)
    namespace = {"np": np}
    wanted_constants = {"BOLTZMANN_CONSTANT_KB", "NM3_TO_A3"}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in wanted_constants for t in node.targets
        ):
            exec(compile(ast.Module([node], []), ANALYSIS_PATH, "exec"), namespace)
        elif isinstance(node, ast.FunctionDef) and node.name == "_grid_free_energy":
            exec(compile(ast.Module([node], []), ANALYSIS_PATH, "exec"), namespace)
            return namespace["_grid_free_energy"]
    raise RuntimeError("_grid_free_energy not found in analysis.py")


# Parameters from the orchestrator spike: 50 A cube, 0.5 A grid, 1000 atoms, 100 frames.
BOX_VOLUME_A3 = 50.0 ** 3          # 125000 A^3
BOX_VOLUME_NM3 = BOX_VOLUME_A3 / 1000.0  # 125 nm^3
GRIDSIZE_A = 0.5
N_ATOMS = 1000
N_FRAMES = 100
N_VOXEL = BOX_VOLUME_A3 / GRIDSIZE_A ** 3  # 1e6 voxels


def _uniform_hist():
    # Uniform occupancy exactly matching the bulk reference density:
    # N = hist / n_frames must equal n_atoms / n_voxel.
    return np.full((20, 20, 20), N_FRAMES * N_ATOMS / N_VOXEL, dtype=float)


def test_uniform_histogram_gives_zero_gfe_with_box_volume_in_nm3():
    grid_free_energy = _load_grid_free_energy()
    gfe = grid_free_energy(
        _uniform_hist(), BOX_VOLUME_NM3, GRIDSIZE_A, N_ATOMS, N_FRAMES, 300.0
    )
    # The histogram is 1e-20-regularised, so allow a tiny tolerance.
    assert np.allclose(gfe, 0.0, atol=1e-9), f"mean GFE = {np.mean(gfe)!r}, expected 0"


def test_wrong_units_are_detected_negative_control():
    # Passing the box volume already converted to A^3 as if it were nm^3 inflates the
    # voxel count by 1e6 instead of 1e3 and must shift the GFE by -kT ln(1000).
    grid_free_energy = _load_grid_free_energy()
    gfe = grid_free_energy(
        _uniform_hist(), BOX_VOLUME_A3, GRIDSIZE_A, N_ATOMS, N_FRAMES, 300.0
    )
    assert np.allclose(np.mean(gfe), -OFFSET_KCAL, atol=1e-6), (
        f"mean GFE = {np.mean(gfe)!r}, expected {-OFFSET_KCAL!r}"
    )
