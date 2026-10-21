"""Articulated TemporalBurst: signal-level tests on a controlled scene (floor, metal wall, three separated boxes as the
'temporal_human', one vertex group per box).

    python tests/articulated/test_articulated_burst.py        (or: pytest tests/articulated)

A  zero vertex displacement reproduces TemporalBurst.signals(zeros)
B  identical displacement of every vertex reproduces the rigid TemporalBurst.signals (per-path outputs and FMCW)
C  one moving part vs independent NumPy path length, phase and Doppler
D  two parts with different velocities plus a static part
The Doppler references use the effective centre frequency of the official FMCW chirp (F_EFF); + = receding."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness  # noqa: E402
import fixtures as FX  # noqa: E402
from support import *  # noqa: E402,F401,F403
from waveverse_sim.temporal_burst import _shared_corrector  # noqa: E402

T = THRESHOLDS
_C = {}


def setup():
    if "b" not in _C:
        xml, groups, _ = FX.write_scene(OUT / "fixture_burst", FX.TWO_PARTS)
        scene, spec, tx, rx, look = load(xml, FX.SENSOR, assign=FX.ASSIGN)
        tr = trace(scene, 30000, 3)
        b = TemporalBurst(scene, tr, spec, tx, rx, look, "temporal_human", groups=groups, group_displacement=[0, 0, 0])
        _C.update(b=b, groups=groups, tx=tx.astype(float), rx=rx.astype(float), spec=spec)
    return _C


def corr(b, vdisp, mode="barycentric"):
    """Per-path outputs of the articulated corrector (same call path as signals_articulated)."""
    avid, aw, _ = b._articulated_tables(mode)
    b._articulated_avid, b._articulated_aw = avid, aw
    c = _shared_corrector(b, articulated=True)
    G = {k: v for k, v in dict(b.G, avid=avid, aw=aw).items() if isinstance(v, tf.Tensor)}
    d = np.concatenate([vdisp, np.zeros((len(vdisp), 1, 3), np.float32)], 1)
    return [np.asarray(x) for x in c(G, tf.constant(d, tf.float32))]


def corr_rigid(b, disp):
    G = {k: v for k, v in b.G.items() if isinstance(v, tf.Tensor)}
    d = np.stack([disp, np.zeros_like(disp)], 1)
    return [np.asarray(x) for x in b._correct(G, tf.constant(d, tf.float32))]


def rd(sig):
    R = np.fft.fft(sig * np.hanning(sig.shape[-1]), axis=-1)
    return np.fft.fftshift(np.fft.fft(R * np.hanning(sig.shape[0])[:, None, None], axis=0), axes=0)


def test_A_zero_displacement():
    b = setup()["b"]
    K, V = 16, b.n_vertices
    s0 = b.signals(np.zeros((K, 3), np.float32), chunk=4)
    s1 = b.signals_articulated(np.zeros((K, V, 3), np.float32), chunk=4, unknown_interactions="barycentric")
    e = rel(s1, s0)
    print("   A: articulated(0) vs signals(0) rel L2", e)
    assert e <= T["A_zero_displacement_rel_l2"]


def test_B_rigid_equivalence():
    b = setup()["b"]
    K, V = 32, b.n_vertices
    t = np.arange(K) * CRI
    disp = np.stack([0.8 * t, -0.3 * t, 0.2 * t], 1).astype(np.float32)  # rigid velocity (0.8, -0.3, 0.2) m/s
    vd = np.broadcast_to(disp[:, None], (K, V, 3)).copy()
    r, a = corr_rigid(b, disp), corr(b, vd)
    uk = (b.interactions["kind"] == 2).any(0)  # paths with interactions that have no vertex identity (barycentric)

    def known(x):
        if x.shape[-1] == len(uk):
            return x[..., ~uk]
        if x.ndim >= 2 and x.shape[1] == len(uk):
            return x[:, ~uk]
        return x

    names = ["scale", "dtau", "valid", "last", "k_s", "r_last", "d_rx", "d_tx"]
    bit = {n: bool(np.array_equal(known(x), known(y))) for n, x, y in zip(names, r, a)}
    dl = float(np.abs((a[1] - r[1])[:, uk]).max() * C0) if uk.any() else 0.0
    ph = float(np.rad2deg(np.abs(np.angle(a[0][:, uk] * np.conj(r[0][:, uk])))).max()) if uk.any() else 0.0
    s0 = b.signals(disp, chunk=4)
    s1 = b.signals_articulated(vd, chunk=4, unknown_interactions="barycentric")
    e = rel(s1, s0)
    R0, R1 = rd(s0), rd(s1)
    er = float(np.abs(R1 - R0).max() / np.abs(R0).max())
    print("   B: bitwise", bit, "barycentric dL", dl, "phase", ph, "signal rel", e, "RD rel", er)
    assert all(bit.values())
    assert dl <= T["B_barycentric_path_length_m"] and ph <= T["B_barycentric_phase_deg"]
    assert e <= T["B_signal_rel_l2"] and er <= T["B_range_doppler_rel_peak"]


def isolate(b, group):
    """Paths whose only moving interaction is a known vertex of `group` at depth 1 (direct TX -> vertex -> RX)."""
    k, v, d = b.interactions["kind"], b.interactions["vertex"], np.asarray(b.G["depth"])
    g = _C["groups"].vertex_group
    return (d == 1) & (k[0] == 0) & (g[np.minimum(v[0], len(g) - 1)] == group)


def doppler_peak(y, Z=16):
    """Signed peak bin of a chirp sequence (zero-padded, Hann), receding positive (beat-signal convention)."""
    Kc = len(y)
    Y = np.abs(np.fft.fft(y * np.hanning(Kc), Kc * Z))
    i = int(np.argmax(Y))
    M = len(Y)
    d = 0.5 * (Y[(i - 1) % M] - Y[(i + 1) % M]) / (Y[(i - 1) % M] - 2 * Y[i] + Y[(i + 1) % M])
    f = (i + d) / Z
    return f - Kc if f >= Kc / 2 else f


def part_signal(b, vd, mask):
    avid, aw, _ = b._articulated_tables("barycentric")
    b._articulated_avid, b._articulated_aw = avid, aw
    d = np.concatenate([vd, np.zeros((len(vd), 1, 3), np.float32)], 1)
    return b._signals(dict(b.G, avid=avid, aw=aw), b.a0, b.tau0, d, None, 4, _shared_corrector(b, articulated=True), mask)


def analytic(c, P0, vel, t):
    P = P0[None] + vel[None, None] * t[:, None, None]
    return np.linalg.norm(P - c["tx"], axis=-1) + np.linalg.norm(P - c["rx"], axis=-1)


def check_part(c, b, vd, vel, group, t):
    m = isolate(b, group)
    pts = np.asarray(b.G["pts0"])[0, m].astype(float)
    out = corr(b, vd)
    dtau, scale = out[1][:, m], out[0][:, m]
    L = analytic(c, pts, vel, t)
    dL = L - L[0:1]
    e_d = float(np.abs(dtau * C0 - dL).max())
    ph = np.angle(scale * np.conj(scale[0:1]))
    e_p = float(np.rad2deg(np.abs(np.angle(np.exp(1j * (ph + 2 * np.pi * FC * dL / C0))))).max())
    sig = part_signal(b, vd, m)
    R = np.fft.fft(sig * np.hanning(sig.shape[-1]), axis=-1)
    kb = int(np.argmax(np.abs(R).sum((0, 1))))
    fd = doppler_peak(R[:, :, kb].sum(1))
    rates = (L[-1] - L[0]) / (t[-1] - t[0])
    w = np.sum(np.abs(np.asarray(b.a0)[:, :, m]) ** 2, axis=(0, 1))
    to_bins = lambda r: F_EFF * r / C0 * (len(t) * CRI)  # noqa: E731
    rate_w = float(np.sum(w * rates) / w.sum())
    # an extended part has a spread of path rates: the whole-part peak must lie inside it; a single vertex is compared exactly
    vid = b.interactions["vertex"][0, m]
    vs = vid[np.argmax(w)]
    ms = m.copy()
    ms[np.flatnonzero(m)[vid != vs]] = False
    sig1 = part_signal(b, vd, ms)
    R1 = np.fft.fft(sig1 * np.hanning(sig1.shape[-1]), axis=-1)
    kb1 = int(np.argmax(np.abs(R1).sum((0, 1))))
    fd1 = doppler_peak(R1[:, :, kb1].sum(1))
    exp1 = to_bins(float(np.mean(rates[vid == vs])))
    lo, hi = sorted((to_bins(float(rates.min())), to_bins(float(rates.max()))))
    return dict(paths=int(m.sum()), c_dtau_err_m=e_d, phase_err_deg=e_p, doppler_bins=float(fd), expected_bins=to_bins(rate_w),
                inside_rate_spread=bool(lo <= fd <= hi) if hi > lo else bool(abs(fd) < 1e-9), single_vertex_err_bins=abs(float(fd1) - exp1),
                sign_ok=bool(np.sign(fd) == np.sign(to_bins(rate_w))))


def test_C_one_moving_part():
    c = setup()
    b, g = c["b"], c["groups"]
    K = 128
    t = np.arange(K) * CRI
    vel = np.array([-2.0, 0.0, 0.0])  # part 0 approaches the radar at 2 m/s
    vd = np.zeros((K, b.n_vertices, 3), np.float32)
    vd[:, g.groups[0]] = (vel[None] * t[:, None])[:, None]
    r = check_part(c, b, vd, vel, 0, t)
    print("   C:", r)
    assert r["c_dtau_err_m"] <= T["C_dtau_vs_analytic_m"] and r["phase_err_deg"] <= T["C_phase_deg"]
    assert r["single_vertex_err_bins"] <= T["C_doppler_peak_bins"] and r["inside_rate_spread"] and r["sign_ok"]


def test_D_two_parts_and_static():
    c = setup()
    b, g = c["b"], c["groups"]
    K = 128
    t = np.arange(K) * CRI
    v0, v1 = np.array([1.5, 0, 0]), np.array([-2.5, 0, 0])
    vd = np.zeros((K, b.n_vertices, 3), np.float32)
    vd[:, g.groups[0]] = (v0[None] * t[:, None])[:, None]
    vd[:, g.groups[1]] = (v1[None] * t[:, None])[:, None]
    r0, r1, r2 = check_part(c, b, vd, v0, 0, t), check_part(c, b, vd, v1, 1, t), check_part(c, b, vd, np.zeros(3), 2, t)
    sep = min(abs(r0["expected_bins"] - r1["expected_bins"]), abs(r0["expected_bins"] - r2["expected_bins"]), abs(r1["expected_bins"] - r2["expected_bins"]))
    print("   D:", r0, r1, r2, "separation", sep)
    assert all(x["c_dtau_err_m"] <= T["D_dtau_vs_analytic_m"] and x["phase_err_deg"] <= T["D_phase_deg"] for x in (r0, r1, r2))
    assert max(r0["single_vertex_err_bins"], r1["single_vertex_err_bins"]) <= T["D_doppler_peak_bins"]
    assert r0["inside_rate_spread"] and r1["inside_rate_spread"] and abs(r2["doppler_bins"]) <= T["D_static_peak_bins"]
    assert sep >= T["D_min_separation_bins"] and r0["sign_ok"] and r1["sign_ok"]


def test_unknown_mode_required():
    b = setup()["b"]
    if not (b.interactions["kind"] == 2).any():
        return
    try:
        b.signals_articulated(np.zeros((2, b.n_vertices, 3), np.float32))
        raise AssertionError("no error raised")
    except ValueError as e:
        assert "unknown_interactions" in str(e)


if __name__ == "__main__":
    _harness.run(globals(), OUT / "test_articulated_burst.json")
