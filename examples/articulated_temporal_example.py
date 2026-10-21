"""Minimal example: per-vertex (articulated) intra-burst motion with TemporalBurst.signals_articulated.

One reference ray trace of the official laundry room + official pose; the right forearm and hand then swing
about the elbow during a 32-chirp burst (no HMG, single person). Run from the repository root:

    python examples/articulated_temporal_example.py

On RTX 6000 Ada / driver 570 the official depth-3 trace needs the documented Dr.Jit workaround
(JitFlag.VCallRecord=False), applied below before Sionna is imported.
"""

import sys
from pathlib import Path

import drjit as dr

dr.set_flag(dr.JitFlag.VCallRecord, False)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402

import numpy as np  # noqa: E402

from waveverse_sim.grouped_scattering import HumanVertexGroups  # noqa: E402
from waveverse_sim.spatial import configure_scene  # noqa: E402
from waveverse_sim.temporal import prepare_scene  # noqa: E402
from waveverse_sim.temporal_burst import TemporalBurst  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "articulated_example"
CFG = json.loads((ROOT / "configs/temporal.json").read_text())
SENSOR = np.asarray(CFG["sensor"], np.float32)
CHIRPS, CRI = 32, 60.01e-6  # PanoRadar chirp repetition interval

xml, _ = prepare_scene(
    ROOT / "data/laundry_room/laundry_room.xml",
    ROOT / "data/person/pose.ply",
    OUT,
    SENSOR,
    False,
    CFG["person_distance"],
)
groups = HumanVertexGroups.from_correspondence(
    ROOT / "data/person/pose.ply",
    ROOT / "data/person/correspondence.pkl",
    OUT / "assets/human_reference.ply",
)
scene, spec, tx, rx, look, _ = configure_scene(xml, SENSOR, CFG["materials"], {})
traced = scene.trace_paths(
    max_depth=CFG["max_depth"],
    num_samples=CFG["num_samples"],
    scattering=True,
    reflection=True,
    los=False,
    diffraction=False,
    ris=False,
    scat_keep_prob=1.0,
)
burst = TemporalBurst(
    scene, traced, spec, tx, rx, look, "temporal_human", groups=groups, group_displacement=[0, 0, 0]
)
print("paths", burst.G["P"], "audit", json.dumps(burst.articulation_audit()["interactions"]))

# Forearm + hand rotate about the elbow (lateral y axis) at 3 rad/s.
names = groups.names
moving = np.isin(groups.vertex_group, [names.index(n) for n in ("rightForeArm", "rightHand", "rightHandIndex1")])
upper = np.unique(groups.faces[groups.face_group == names.index("rightArm")])
fore = np.unique(groups.faces[groups.face_group == names.index("rightForeArm")])
elbow = burst.reference_vertices[np.intersect1d(upper, fore)].mean(0)
t = np.arange(CHIRPS) * CRI
displacements = np.zeros((CHIRPS, burst.n_vertices, 3), np.float32)
r = burst.reference_vertices[moving] - elbow
for k, angle in enumerate(3.0 * t):
    c, s = np.cos(angle), np.sin(angle)
    rotated = r @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]).T
    displacements[k, moving] = rotated - r

signal = burst.signals_articulated(
    vertex_displacements=displacements, chunk=2, unknown_interactions="barycentric"
)
range_fft = np.fft.fft(signal * np.hanning(signal.shape[-1]), axis=-1)
doppler = np.abs(np.fft.fftshift(np.fft.fft(range_fft * np.hanning(CHIRPS)[:, None, None], axis=0), axes=0)) ** 2
spectrum = doppler.reshape(CHIRPS, -1).sum(1)
bins = np.arange(CHIRPS) - CHIRPS // 2
print("signal", signal.shape, "Doppler centroid [bins, + receding]", float((bins * spectrum).sum() / spectrum.sum()))
