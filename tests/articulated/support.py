"""Shared helpers for the articulated (per-vertex) TemporalBurst tests.

Importing this module applies the Dr.Jit workaround needed for depth-3 tracing on some GPU / driver
combinations (JitFlag.VCallRecord = False, set before Sionna is imported) and enables TensorFlow GPU
memory growth. Test outputs are written to tests/articulated/_out (ignored by git)."""

import json
import sys
from pathlib import Path

import numpy as np
import drjit as dr

dr.set_flag(dr.JitFlag.VCallRecord, False)
import tensorflow as tf  # noqa: E402

for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from waveverse_sim.spatial import configure_scene  # noqa: E402
from waveverse_sim.temporal_burst import TemporalBurst  # noqa: E402

OUT = Path(__file__).resolve().parent / "_out"
OUT.mkdir(exist_ok=True)
CFG = json.loads((ROOT / "configs/temporal.json").read_text())
C0 = 299792458.0
FC = 77e9
CRI = 60.01e-6  # PanoRadar chirp repetition interval (s)
F_EFF = 78990967285.1562  # effective centre frequency of the official FMCW chirp (official temporal metadata)

# Acceptance thresholds (frozen before the tests were first run).
THRESHOLDS = {
    "A_zero_displacement_rel_l2": 1e-6,
    "B_barycentric_path_length_m": 2.5e-7,
    "B_barycentric_phase_deg": 0.05,
    "B_signal_rel_l2": 1e-5,
    "B_range_doppler_rel_peak": 1e-5,
    "C_dtau_vs_analytic_m": 5e-6,
    "C_phase_deg": 1.0,
    "C_doppler_peak_bins": 0.05,
    "D_dtau_vs_analytic_m": 5e-6,
    "D_phase_deg": 1.0,
    "D_doppler_peak_bins": 0.1,
    "D_static_peak_bins": 0.05,
    "D_min_separation_bins": 4,
    "batching_signal_rel_l2": 1e-5,
}


def load(xml, sensor, materials=None, assign=None):
    scene, spec, tx, rx, look, _ = configure_scene(
        Path(xml), np.asarray(sensor, np.float32), materials if materials is not None else CFG["materials"], assign or {}
    )
    return scene, spec, np.asarray(tx, np.float32), np.asarray(rx, np.float32), np.asarray(look, np.float32)


def trace(scene, n=100000, depth=None):
    return scene.trace_paths(
        max_depth=depth or CFG["max_depth"], num_samples=n, scattering=True, reflection=True, los=False,
        diffraction=False, ris=False, scat_keep_prob=1.0,
    )


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / max(np.linalg.norm(np.asarray(b)), 1e-300))


class Observer:
    """Passive sys.setprofile observer of grouped_path_batches locals at each yield (read-only)."""

    def __init__(self):
        self.B = []

    def __call__(self, frame, event, arg):
        if event == "return" and frame.f_code.co_name == "grouped_path_batches" and arg is not None:
            L = frame.f_locals
            try:
                self.B.append(dict(
                    parents=np.asarray(L["parents"]).copy(), child=np.asarray(L["child"]).copy(), depth=np.asarray(L["depth"]).copy(),
                    local_groups=np.asarray(L["local_groups"]).copy(), expanded=np.asarray(L["expanded"]).copy(), idx=np.asarray(L["idx"]).copy(),
                    n_out=len(L["weights"]), vertex0=np.asarray(L["vertex0"]).copy() if "vertex0" in L else None,
                ))
            except Exception as e:  # recorded, the identity test then fails visibly
                self.B.append(dict(error=repr(e)))
