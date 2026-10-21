"""Phase-coherent Doppler bursts with rigidly moving scattering points.

Each burst traces at its first chirp, keeps those material points attached to
the moving object, and updates geometric delay, spreading, directive gain and
array phase across chirps.
Visibility, intermediate Fresnel/polarization terms and antenna gain are frozen
within the short burst. A fresh trace is needed for each frame.
"""

import time

import numpy as np
import tensorflow as tf
from sionna.rt import Paths
from sionna.rt.solver_paths import PathsTmpData
from sionna.rt.scattering_pattern import ScatteringPattern

from sionna.rt.utils import rotation_matrix
from . import cuda_mixer
from .grouped_scattering import grouped_path_batches

C0 = 299792458.0
_CORRECTORS = {}


def _checked_interactions(part, inter):
    """Validate that grouped-expansion interaction metadata is aligned with a part's paths.

    Exact checks: path vertices equal the expansion points for every interaction
    of the path, depths agree, and the official moving flag equals the
    known-vertex/unknown-moving kinds. Any mismatch raises.
    """
    start = time.perf_counter()
    pts = np.asarray(part.G["pts0"])
    depth = np.asarray(part.G["depth"])
    vidx = np.asarray(part.G["vidx"])
    used = np.arange(pts.shape[0])[:, None] < depth[None]
    if inter["kind"].shape != vidx.shape or not np.array_equal(depth, inter["depth"] + 1):
        raise AssertionError("Interaction metadata/path depth alignment broken")
    if not np.array_equal(pts[used], inter["points"][used]):
        raise AssertionError("Interaction metadata/path vertex alignment broken")
    moving = (inter["kind"] == 0) | (inter["kind"] == 2)
    if not np.array_equal(moving[used], (vidx == 0)[used]):
        raise AssertionError("Interaction metadata/moving-flag alignment broken")
    out = {k: inter[k] for k in ("kind", "vertex", "face")}
    out["seconds"] = inter["seconds"] + time.perf_counter() - start
    return out


def _shared_corrector(burst, articulated=False):
    """One graph per radar/depth, with dynamic path and chirp dimensions.

    Capturing only the immutable radar equations avoids retaining an old scene
    or its path tensors in the graph cache. New frame path counts do not retrace.

    articulated=True (extension): the geometry additionally carries per-interaction
    vertex tables "avid"/"aw" and the displacement is (K, V + 1, 3); the
    correction equations are the same _correct_impl.
    """
    key = (
        ("articulated",) if articulated else (),
        burst.G["D"],
        float(burst.spec.fc),
        float(burst.spec.wavelength),
        burst.tx.tobytes(),
        burst.rx.tobytes(),
        burst.look_at.tobytes(),
        np.asarray(burst.pos_rx).tobytes(),
        np.asarray(burst.pos_tx).tobytes(),
    )
    if key not in _CORRECTORS:
        equations = TemporalBurst.__new__(TemporalBurst)
        for name in ("spec", "tx", "rx", "look_at", "pos_rx", "pos_tx"):
            setattr(equations, name, getattr(burst, name))
        equations.spec = burst.spec.copy()
        depth = burst.G["D"]
        signature = {}
        for name, value in burst.G.items():
            if isinstance(value, tf.Tensor):
                shape = value.shape.as_list()
                shape[1 if name in ("pts0", "vidx") else 0] = None
                signature[name] = tf.TensorSpec(shape, value.dtype)
        if articulated:
            signature["avid"] = tf.TensorSpec([depth, None, 3], tf.int32)
            signature["aw"] = tf.TensorSpec([depth, None, 3], tf.float32)
        rows = None if articulated else 2

        @tf.function(
            input_signature=[signature, tf.TensorSpec([None, rows, 3], tf.float32)]
        )
        def correction(geometry, displacement):
            geometry = dict(geometry, D=depth, P=tf.shape(geometry["depth"])[0])
            if articulated:
                return equations._correct_articulated_impl(geometry, displacement)
            return equations._correct_impl(geometry, displacement)

        if len(_CORRECTORS) >= 4:
            _CORRECTORS.clear()
        _CORRECTORS[key] = correction
    return _CORRECTORS[key]


class TemporalBurst:
    def _lengths(self, pts, depth, D):
        """Total path length TX -> v1 -> ... -> v_depth -> RX; pts (K, D, P, 3)."""
        tx = tf.constant(self.tx)
        L = tf.norm(pts[:, 0] - tx, axis=-1)
        for d in range(D - 1):
            seg = tf.norm(pts[:, d + 1] - pts[:, d], axis=-1)
            L += tf.where((depth > d + 1)[None], seg, 0.0)
        last = tf.reduce_sum(
            tf.one_hot(depth - 1, D, dtype=tf.float32)[None, :, :, None]
            * tf.transpose(pts, [0, 2, 1, 3]),
            2,
        )
        prev = tf.reduce_sum(
            tf.one_hot(depth - 2, D, dtype=tf.float32)[None, :, :, None]
            * tf.transpose(pts, [0, 2, 1, 3]),
            2,
        )
        prev = tf.where(
            (depth == 1)[None, :, None],
            tf.broadcast_to(tx[None, None], tf.shape(prev)),
            prev,
        )
        r_last = tf.norm(tf.constant(self.rx)[None, None] - last, axis=-1)
        return L + r_last, last, prev, r_last

    def _correct_impl(self, G, disp, move=None):
        """disp: (K, n_verts+1, 3) vertex displacements per chirp (last row zero).

        move (extension): optional precomputed per-interaction displacement
        (K, D, P, 3); when None the official rigid gather is used unchanged.
        """
        K = tf.shape(disp)[0]
        D, P = G["D"], G["P"]
        if move is None:
            move = tf.gather(disp, G["vidx"], axis=1)  # (K, D, P, 3)
        pts = G["pts0"][None] + move
        L, last, prev, r_last = self._lengths(pts, G["depth"], D)
        L0, _, _, r_last0 = self._lengths(G["pts0"][None], G["depth"], D)
        L0, r_last0 = L0[0], r_last0[0]  # (P,)
        dtau = (L - L0[None]) / C0  # (K, P)
        tx = tf.constant(self.tx)
        k_s = (tf.constant(self.rx)[None, None] - last) / r_last[..., None]
        k_i = last - prev
        k_i = k_i / tf.norm(k_i, axis=-1, keepdims=True)
        nn = tf.broadcast_to(G["n"][None], tf.shape(k_s))
        rep = lambda x: tf.reshape(tf.broadcast_to(x[None], [K, P]), [-1])
        fsk = tf.reshape(
            ScatteringPattern.pattern(
                tf.reshape(k_i, [-1, 3]),
                tf.reshape(k_s, [-1, 3]),
                tf.reshape(nn, [-1, 3]),
                rep(G["alpha_r"]),
                rep(G["alpha_i"]),
                rep(G["lambda_"]),
            ),
            [K, P],
        )
        amp = (r_last0[None] / r_last) * tf.sqrt(
            fsk / tf.maximum(G["fs0"][None], 1e-30)
        )
        same_side = tf.reduce_sum(nn * k_s, -1) > 1e-6
        ph = -2 * np.pi * self.spec.fc * dtau
        scale = tf.complex(amp * tf.cos(ph), amp * tf.sin(ph))
        # synthetic-array phase change from the moved directions (radar static)
        yaw = np.arctan2(self.look_at[1] - self.tx[1], self.look_at[0] - self.tx[0])
        R = rotation_matrix(tf.constant([yaw, 0.0, 0.0], tf.float32))
        k_rx = -k_s
        k_tx = pts[:, 0] - tx
        k_tx = k_tx / tf.norm(k_tx, axis=-1, keepdims=True)
        k_rx0 = -G["k_s0"][None]
        k_tx0 = G["pts0"][0][None] - tx
        k_tx0 = k_tx0 / tf.norm(k_tx0, axis=-1, keepdims=True)
        two_pi_l = 2 * np.pi / self.spec.wavelength
        d_rx = two_pi_l * tf.einsum(
            "ei,kpi->kep", tf.linalg.matvec(R[None], self.pos_rx), k_rx - k_rx0
        )
        d_tx = two_pi_l * tf.einsum(
            "ei,kpi->kep", tf.linalg.matvec(R[None], self.pos_tx), k_tx - k_tx0
        )
        return scale, dtau, same_side, last, k_s, r_last, d_rx, d_tx

    def _correct_articulated_impl(self, G, disp):
        """disp: (K, V + 1, 3) per-vertex displacements, row V zero (static sentinel).

        Each interaction moves by sum_i aw[..., i] * disp[avid[..., i]]: a known
        vertex has (v, V, V) / (1, 0, 0); static and unused interactions gather
        the zero row. The official correction equations are reused unchanged.
        """
        move = G["aw"][None, :, :, 0, None] * tf.gather(disp, G["avid"][:, :, 0], axis=1)
        for i in (1, 2):
            move += G["aw"][None, :, :, i, None] * tf.gather(disp, G["avid"][:, :, i], axis=1)
        return self._correct_impl(G, disp, move)

    def __init__(
        self,
        scene,
        traced,
        spec,
        tx_position,
        rx_position,
        look_at,
        moving_object,
        mixer_split=64,
        groups=None,
        group_displacement=None,
        group_batch_candidates=100000,
    ):
        self.spec = spec
        self.tx = np.asarray(tx_position, np.float32)
        self.rx = np.asarray(rx_position, np.float32)
        self.look_at = np.asarray(look_at, np.float32)
        self.pos_rx = tf.constant(np.asarray(spec.rx_pos), tf.float32)
        self.pos_tx = tf.constant(np.asarray(spec.tx_pos), tf.float32)
        self.mixer_split = mixer_split
        self.group_audit = {"enabled": False}
        self.interactions = None
        if groups is not None:
            parts, audits, interactions = [], [], []
            offset = np.zeros(3) if group_displacement is None else group_displacement
            for paths, data, weights, audit, inter in grouped_path_batches(
                scene,
                traced,
                moving_object,
                groups,
                offset,
                group_batch_candidates,
                return_interactions=True,
            ):
                audits.append(audit)
                if not len(weights):
                    continue
                chunk_trace = list(traced)
                chunk_trace[2], chunk_trace[6] = paths, data
                part = TemporalBurst(
                    scene,
                    chunk_trace,
                    spec,
                    tx_position,
                    rx_position,
                    look_at,
                    moving_object,
                    mixer_split,
                )
                if part.G["P"] != len(weights):
                    raise AssertionError("Field/path normalization alignment changed")
                part.a0 *= tf.cast(weights[None, None], part.a0.dtype)
                parts.append(part)
                interactions.append(_checked_interactions(part, inter))
            if not parts:
                raise ValueError(
                    "No scattered paths survived grouped visibility checks"
                )
            self.G = dict(parts[0].G)
            self.G["P"] = sum(p.G["P"] for p in parts)
            for key, value in self.G.items():
                if isinstance(value, tf.Tensor):
                    self.G[key] = tf.concat(
                        [p.G[key] for p in parts],
                        axis=1 if key in ("pts0", "vidx") else 0,
                    )
            self.a0 = tf.concat([p.a0 for p in parts], axis=2)
            self.tau0 = tf.concat([p.tau0 for p in parts], axis=0)
            self.moving = np.concatenate([p.moving for p in parts])
            self._correct = _shared_corrector(self)
            # Articulated extension: per-interaction vertex identity, aligned with G.
            self.interactions = {
                key: np.concatenate([i[key] for i in interactions], axis=1)
                for key in ("kind", "vertex", "face")
            }
            self.vertex_id_seconds = sum(i["seconds"] for i in interactions)
            self.n_vertices = len(groups.vertices)
            self.reference_vertices = np.asarray(
                groups.vertices + np.asarray(offset, np.float32), np.float32
            )
            self.reference_faces = np.asarray(groups.faces, np.int32)
            self._articulated = {}
            self.group_audit = dict(
                enabled=True,
                normalization="field: 1/sqrt(N_valid), per parent/prefix",
                groups=len(groups.groups),
                vertices=len(groups.vertices),
                batches=len(audits),
                max_split_power_error=max(a["max_split_power_error"] for a in audits),
            )
            for key in (
                "input_candidates",
                "expanded_candidates",
                "expanded_human_parents",
                "tx_blocked_children",
                "next_segment_blocked_children",
                "output_paths",
                "valid_parent_prefix_families",
            ):
                self.group_audit[key] = sum(a[key] for a in audits)
            self.group_audit["human_first_paths_by_depth"] = np.sum(
                [a["human_first_paths_by_depth"] for a in audits], axis=0
            ).tolist()
            return
        source, target = traced[2].sources, traced[2].targets

        def empty(kind):
            return (
                Paths(sources=source, targets=target, scene=scene, types=kind),
                PathsTmpData(source, target, scene._solver_paths._dtype),
            )

        sp, st = empty(Paths.SPECULAR)
        dp, dt = empty(Paths.DIFFRACTED)
        rp, rt = empty(Paths.RIS)
        paths = scene.compute_fields(
            spec_paths=sp,
            spec_paths_tmp=st,
            diff_paths=dp,
            diff_paths_tmp=dt,
            scat_paths=traced[2],
            scat_paths_tmp=traced[6],
            ris_paths=rp,
            ris_paths_tmp=rt,
            check_scene=False,
            scat_random_phases=False,
            testing=True,
        )
        paths.normalize_delays = False
        a, tau = paths.cir(los=False)
        self.a0 = tf.constant(np.asarray(a)[0, 0, :, 0, :, :, 0])
        self.tau0 = tf.constant(np.asarray(tau).reshape(-1))
        objects = np.asarray(paths.objects)[:, 0, 0]
        points = np.asarray(paths.vertices)[:, 0, 0]
        depth = (objects != -1).sum(0).astype(np.int32)
        D, P = objects.shape
        if not P:
            raise ValueError("No scattered paths in this scene")
        moving = objects == scene.get(moving_object).object_id
        self.moving = moving.any(0)
        f32 = lambda x: tf.constant(x, tf.float32)
        s = paths.scat_tmp
        ki = np.asarray(s.scat_last_k_i).reshape(-1, 3)
        normals = np.asarray(s.scat_last_normals).reshape(-1, 3)
        normals = normals * -np.sign((ki * normals).sum(-1, keepdims=True))
        obj = np.maximum(np.asarray(s.scat_last_objects).reshape(-1), 0)
        _, _, _, ar, ai, lam, _ = (
            scene._solver_paths._build_scene_object_properties_tensors()
        )
        ar, ai, lam = (tf.gather(x, obj) for x in (ar, ai, lam))
        ks = f32(np.asarray(s.scat_k_s).reshape(-1, 3))
        self.G = dict(
            P=P,
            D=D,
            pts0=f32(points),
            vidx=tf.constant(np.where(moving, 0, 1), tf.int32),
            depth=tf.constant(depth),
            n=f32(normals),
            k_s0=ks,
            alpha_r=ar,
            alpha_i=ai,
            lambda_=lam,
            fs0=ScatteringPattern.pattern(f32(ki), ks, f32(normals), ar, ai, lam),
        )
        self._correct = _shared_corrector(self)

    def _signals(
        self, G, a0, tau0, displacement, phases=None, chunk=16, correct=None, path_mask=None
    ):
        """correct/path_mask (extension): articulated corrector taking (k, V + 1, 3)
        displacements, and an optional per-path inclusion mask; None = official path."""
        outputs = []
        nrx, ntx = a0.shape[:2]
        for first in range(0, len(displacement), chunk):
            xyz = displacement[first : first + chunk]
            geometry = {k: v for k, v in G.items() if isinstance(v, tf.Tensor)}
            if correct is None:
                disp = tf.constant(np.stack((xyz, np.zeros_like(xyz)), axis=1), tf.float32)
                scale, dtau, valid, _, _, _, pr, pt = self._correct(geometry, disp)
            else:
                disp = tf.constant(xyz, tf.float32)
                scale, dtau, valid, _, _, _, pr, pt = correct(geometry, disp)
            if path_mask is not None:
                valid = tf.logical_and(valid, tf.constant(path_mask)[None])
            phase = pr[:, :, None] + pt[:, None, :]
            a = a0[None] * (scale * tf.cast(valid, tf.complex64))[:, None, None]
            a *= tf.complex(tf.cos(phase), tf.sin(phase))
            if phases is not None:
                a *= tf.constant(
                    phases[first : first + len(xyz), None, None], tf.complex64
                )
            signal = cuda_mixer.mix(
                tf.reshape(a, [len(xyz), nrx * ntx, G["P"]]),
                tau0[None] + dtau,
                self.spec,
                split=self.mixer_split,
            )
            outputs.append(np.asarray(signal))
        return np.concatenate(outputs, axis=0)

    def signals(self, displacement, mode="coherent", seed=0, chunk=16):
        """Evaluate all paths for every chirp; return (chirps, RX*TX, ADC).

        incoherent randomizes moving-path phase without randomizing the room.
        """
        if mode not in ("coherent", "incoherent"):
            raise ValueError("Unknown temporal control")
        displacement = np.asarray(displacement, np.float32)
        if (
            displacement.ndim != 2
            or displacement.shape[1] != 3
            or not len(displacement)
        ):
            raise ValueError(
                "displacement must have shape (chirps, 3) with at least one chirp"
            )
        if chunk < 1 or not np.all(np.isfinite(displacement)):
            raise ValueError("chunk must be positive and displacements must be finite")
        phases = self._control_phases(len(displacement), mode, seed)
        start = time.perf_counter()
        out = self._signals(self.G, self.a0, self.tau0, displacement, phases, chunk)
        self.last_seconds = time.perf_counter() - start
        return out

    def _control_phases(self, chirps, mode, seed):
        if mode != "incoherent":
            return None
        rng = np.random.default_rng(seed)
        phase = rng.uniform(-np.pi, np.pi, (chirps, self.moving.sum()))
        phases = np.ones((chirps, self.G["P"]), np.complex64)
        phases[:, self.moving] = np.exp(1j * phase).astype(np.complex64)
        return phases

    # ------------------------------------------------------------------
    # Articulated extension (feature/articulated-temporal-coherence)
    # ------------------------------------------------------------------
    UNKNOWN_MODES = ("exclude", "barycentric")

    def articulation_audit(self):
        """Counts and |a0|^2 power of interaction kinds, by path depth (grouped bursts only)."""
        if self.interactions is None:
            raise ValueError("Articulated metadata requires a grouped burst (groups=...)")
        kind = self.interactions["kind"]
        depth = np.asarray(self.G["depth"])
        power = np.sum(np.abs(np.asarray(self.a0)) ** 2, axis=(0, 1))
        unknown = (kind == 2).any(0)
        out = dict(
            paths=int(len(depth)),
            vertices=int(self.n_vertices),
            interactions={
                name: int((kind == code).sum())
                for code, name in enumerate(("known_vertex", "static", "unknown_moving"))
            },
            paths_with_unknown=int(unknown.sum()),
            power_total=float(power.sum()),
            power_moving=float(power[self.moving].sum()),
            power_with_unknown=float(power[unknown].sum()),
            by_depth={},
        )
        for d in np.unique(depth):
            m = depth == d
            out["by_depth"][int(d)] = dict(
                paths=int(m.sum()),
                moving_paths=int((m & self.moving).sum()),
                paths_with_unknown=int((m & unknown).sum()),
                power=float(power[m].sum()),
                power_with_unknown=float(power[m & unknown].sum()),
            )
        return out

    def _articulated_tables(self, unknown_interactions):
        """avid/aw (D, P, 3) and the path mask for one unknown-interaction mode (cached)."""
        if unknown_interactions in self._articulated:
            return self._articulated[unknown_interactions]
        kind, vertex = self.interactions["kind"], self.interactions["vertex"]
        D, P = kind.shape
        V = self.n_vertices
        avid = np.full((D, P, 3), V, np.int32)
        aw = np.zeros((D, P, 3), np.float32)
        aw[..., 0] = 1.0
        known = kind == 0
        avid[known, 0] = vertex[known]
        unknown = kind == 2
        path_mask = None
        if unknown.any():
            if unknown_interactions == "exclude":
                path_mask = ~unknown.any(0)
            elif unknown_interactions == "barycentric":
                face = self.interactions["face"][unknown]
                tri = self.reference_vertices[self.reference_faces[face]].astype(np.float64)
                p = np.asarray(self.G["pts0"]).astype(np.float64)[unknown]
                e1, e2, r = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], p - tri[:, 0]
                d11, d12, d22 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1)
                r1, r2 = (r * e1).sum(1), (r * e2).sum(1)
                den = d11 * d22 - d12 * d12
                w1 = (d22 * r1 - d12 * r2) / den
                w2 = (d11 * r2 - d12 * r1) / den
                w = np.stack((1 - w1 - w2, w1, w2), 1)
                avid[unknown] = self.reference_faces[face]
                aw[unknown] = w.astype(np.float32)
                residual = np.linalg.norm((w[:, :, None] * tri).sum(1) - p, axis=1)
                self.barycentric_check = dict(
                    interactions=int(unknown.sum()),
                    max_reconstruction_residual_m=float(residual.max()),
                    min_weight=float(w.min()),
                    max_weight=float(w.max()),
                )
            else:
                raise ValueError(
                    f"{int(unknown.any(0).sum())} paths contain human interactions without a "
                    "vertex identity (later human hits of expanded parents, or human hits after a "
                    "static first hit); choose unknown_interactions in "
                    f"{self.UNKNOWN_MODES} (see articulation_audit())"
                )
        tables = (tf.constant(avid), tf.constant(aw), path_mask)
        self._articulated[unknown_interactions] = tables
        return tables

    def signals_articulated(
        self,
        vertex_displacements,
        mode="coherent",
        seed=0,
        chunk=16,
        unknown_interactions=None,
    ):
        """Per-vertex intra-burst motion; return (chirps, RX*TX, ADC) like signals().

        vertex_displacements: (K, V, 3) world-frame displacement of every human mesh
        vertex at every chirp, relative to burst.reference_vertices (the mesh used for
        this burst's reference trace). Requires a grouped burst. Path set, visibility,
        last-interaction normals and prefix Fresnel terms stay frozen as in signals().
        unknown_interactions: None (raise if such paths exist), "exclude" (mask paths
        with human interactions lacking a vertex identity) or "barycentric" (move such
        a surface point with its reference triangle's vertex displacements).
        """
        if self.interactions is None:
            raise ValueError("signals_articulated requires a grouped burst (groups=...)")
        if mode not in ("coherent", "incoherent"):
            raise ValueError("Unknown temporal control")
        vertex_displacements = np.asarray(vertex_displacements, np.float32)
        if (
            vertex_displacements.ndim != 3
            or vertex_displacements.shape[1:] != (self.n_vertices, 3)
            or not len(vertex_displacements)
        ):
            raise ValueError(
                f"vertex_displacements must have shape (chirps, {self.n_vertices}, 3)"
            )
        if chunk < 1 or not np.all(np.isfinite(vertex_displacements)):
            raise ValueError("chunk must be positive and displacements must be finite")
        avid, aw, path_mask = self._articulated_tables(unknown_interactions)
        correct = _shared_corrector(self, articulated=True)
        G = dict(self.G, avid=avid, aw=aw)
        displacement = np.concatenate(
            (vertex_displacements, np.zeros((len(vertex_displacements), 1, 3), np.float32)),
            axis=1,
        )
        phases = self._control_phases(len(displacement), mode, seed)
        start = time.perf_counter()
        out = self._signals(
            G, self.a0, self.tau0, displacement, phases, chunk, correct, path_mask
        )
        self.last_seconds = time.perf_counter() - start
        return out
