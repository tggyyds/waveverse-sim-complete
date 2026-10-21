# WaveVerse-Sim: Scalable RF Simulation in Generative 4D Worlds

<a href="https://arxiv.org/abs/2508.12176"><img src="https://img.shields.io/badge/arXiv-2508.12176-b31b1b.svg" alt="arXiv"></a>
<a href="https://waves.seas.upenn.edu/projects/waveverse/"><img src="https://img.shields.io/badge/Project-Website-green" alt="Project Page"></a>

WaveVerse-Sim is the **GPU-accelerated RF simulator** accompanying
*Scalable RF Simulation in Generative 4D Worlds*. It models **spatial and temporal
phase coherence** and generates FMCW radar signals from indoor scenes and moving
human meshes using GPU ray tracing, CUDA signal synthesis, and grouped human
scattering.

> **This repository extends the official [WaveVerse-Sim](https://github.com/penn-waves-lab/WaveVerse-Sim) release (commit `771df9a`)
> with articulated, per-vertex micro-Doppler** (`TemporalBurst.signals_articulated`). Everything else (spatial / temporal
> workflows, static scattering, materials, radar model) is the unchanged upstream code; see
> [Articulated micro-Doppler](#-articulated-micro-doppler) below.


## 🛠️ Installation

Simulation requires an NVIDIA GPU. The examples below have been validated on an RTX 3090 Ti with 24 GB VRAM and CUDA 12.2.

```bash
git clone git@github.com:penn-waves-lab/WaveVerse-Sim.git
cd WaveVerse-Sim
conda env create -f environment.yml
conda activate waveverse-sim
```


## 📦 Example data

Both examples use the bundled laundry room. To make temporal phase coherence
easy to understand, the temporal example adds a posed human mesh and moves it
back and forth at runtime. This is a simple translation of the entire mesh,
not a recorded human-motion sequence or an articulated walking animation.

```text
data/
├── laundry_room/
│   ├── laundry_room.xml
│   ├── meshes/
│   └── scene.json
└── person/
    ├── pose.ply
    └── correspondence.pkl
```

## 🚀 Spatial phase coherence

Simulate panoramic high-resolution imaging as described in our paper, and compare
the results with and without spatial phase coherence:

```bash
python examples/spatial_coherence.py --config configs/spatial.json
```

With the bundled config, signals and range images are saved to `results/spatial/`.
Open `results/spatial/spatial_comparison.png` to compare imaging with and without
spatial phase coherence.

## 🚀 Temporal phase coherence

Compare Doppler with and without temporal phase coherence over 0.1–0.8 m:

```bash
python examples/temporal_coherence.py --config configs/temporal.json
```

With the bundled config, per-frame signals are saved to `results/temporal/`.
Open `results/temporal/doppler_comparison.png` to compare Doppler with and without
temporal phase coherence over 0.1–0.8 m.


To plot full-range 0–9.6 m Doppler from the saved signals:

```bash
python examples/plot_doppler.py results/temporal --full-range
```

The full-range figure is saved to `results/temporal/doppler_full_range.png`.

## 🚀 Articulated micro-Doppler

The official temporal example moves the whole human mesh rigidly. `TemporalBurst.signals_articulated` adds
**per-vertex (articulated) motion inside a chirp burst**: every human scattering path follows the mesh vertex it was
traced from, so limbs that move differently produce their own Doppler components (micro-Doppler). One reference ray
trace per burst is reused for all chirps, exactly as the rigid `signals()` does.

```python
burst = TemporalBurst(scene, traced, spec, tx, rx, look, "temporal_human",
                      groups=groups, group_displacement=[0, 0, 0])      # grouped burst (as in the temporal example)
signal = burst.signals_articulated(
    vertex_displacements=displacements,   # (chirps, burst.n_vertices, 3) world-frame offsets of every mesh vertex
    unknown_interactions="barycentric",   # paths on the human without a vertex identity move with their triangle
)                                         # -> (chirps, RX*TX, ADC), same convention as signals()
print(burst.articulation_audit())         # counts / power of known-vertex, static and barycentric interactions
```

Displacements are relative to `burst.reference_vertices`. The path set, visibility, last-interaction normals and
prefix Fresnel terms stay frozen for the burst, as in `signals()`; use one burst per radar frame for longer motion.
`unknown_interactions` can also be `"exclude"` (drop such paths) or `None` (raise if they exist).

**Example** (bundled laundry room and person; the right forearm swings about the elbow during a 32-chirp burst):

```bash
python examples/articulated_temporal_example.py
```

**Validation** (identity, signal-level and end-to-end checks; a few minutes on one GPU):

```bash
bash tests/articulated/run_all.sh
```

| Check | Criterion |
| --- | --- |
| Every expanded child keeps its source vertex id; official ownership; TX visibility (independent ray test); batching invariance | exact |
| Zero displacement = `signals(zeros)` | rel. L2 ≤ 1e-6 (measured 0) |
| Same displacement for every vertex = rigid `signals()` | per-path outputs bitwise, FMCW rel. L2 ≤ 1e-5 (measured 0) |
| One moving part vs independent NumPy path length / phase / Doppler | ≤ 5 µm, ≤ 1°, ≤ 0.05 bins |
| Two parts at different velocities + a static part | ≤ 5 µm, ≤ 1°, ≤ 0.1 bins; static part at 0 bins |

The rigid API is unchanged: on the same scene, `signals()` of this repository and of the official release agree to the
official code's own run-to-run variation (≤ 2e-7 relative, identical path counts). On some GPU / driver combinations
depth-3 tracing needs `drjit.set_flag(drjit.JitFlag.VCallRecord, False)` before Sionna is imported; the example and
the tests set it.

**Scope.** The extension validates the chain from mesh motion to radar Doppler (vertex identity, path-length and phase
evolution, mixer, FFT, sign, scaling). It is not a calibration against measured human radar data, and the sparse human
scattering of the default ray budget can miss small, fast body parts.

## 🙏 Acknowledgements

The codebase is built on [Sionna](https://github.com/NVlabs/sionna),
[Mitsuba](https://github.com/mitsuba-renderer/mitsuba3),
[Dr.Jit](https://github.com/mitsuba-renderer/drjit), TensorFlow and CuPy.

## 📜 Citation

```bibtex
@inproceedings{zheng2026scalable,
  title={Scalable RF Simulation in Generative 4D Worlds},
  author={Zhiwei Zheng and Dongyin Hu and Mingmin Zhao},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
}
```

## License

Code is released under [Apache-2.0](LICENSE). Bundled third-party software retains its notices.
