"""Articulated TemporalBurst: every retained expanded child keeps the identity of its source human vertex.

    python tests/articulated/test_vertex_identity.py        (or: pytest tests/articulated)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness  # noqa: E402
import fixtures as FX  # noqa: E402
import trimesh  # noqa: E402
from support import *  # noqa: E402,F401,F403

_C = {}


def burst(batch=100000):
    if batch in _C:
        return _C[batch]
    if "scene" not in _C:
        xml, groups, _ = FX.write_scene(OUT / "fixture_identity", FX.TWO_PARTS)
        scene, spec, tx, rx, look = load(xml, FX.SENSOR, assign=FX.ASSIGN)
        _C.update(scene=scene, spec=spec, tx=tx, rx=rx, look=look, groups=groups, tr=trace(scene, 20000, 3))
    c = _C
    obs = Observer()
    sys.setprofile(obs)
    try:
        b = TemporalBurst(c["scene"], c["tr"], c["spec"], c["tx"], c["rx"], c["look"], "temporal_human", groups=c["groups"],
                          group_displacement=[0, 0, 0], group_batch_candidates=batch)
    finally:
        sys.setprofile(None)
    _C[batch] = (b, obs)
    return _C[batch]


def test_identity_exact():
    b, _ = burst()
    I = b.interactions
    k0 = I["kind"] == 0
    pts = np.asarray(b.G["pts0"])
    assert k0.any() and np.array_equal(pts[k0], b.reference_vertices[I["vertex"][k0]])


def _blocked(tri, o, e, eps=1e-4):
    """Independent NumPy Moller-Trumbore segment test against all scene triangles."""
    V0 = tri[:, 0]
    E1, E2 = tri[:, 1] - V0, tri[:, 2] - V0
    d = e - o
    L = np.linalg.norm(d, axis=1)
    d = d / L[:, None]
    out = np.zeros(len(o), bool)
    for i in range(0, len(o), 64):
        oo, dd, LL = o[i:i + 64], d[i:i + 64], L[i:i + 64]
        pv = np.cross(dd[:, None], E2[None])
        det = np.einsum("rtk,tk->rt", pv, E1)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1 / np.where(np.abs(det) < 1e-12, np.nan, det)
            tv = oo[:, None] - V0[None]
            u = np.einsum("rtk,rtk->rt", tv, pv) * inv
            qv = np.cross(tv, E1[None])
            v = np.einsum("rk,rtk->rt", dd, qv) * inv
            t = np.einsum("tk,rtk->rt", E2, qv) * inv
            out[i:i + 64] = np.nan_to_num((u >= 0) & (v >= 0) & (u + v <= 1) & (t > eps) & (t < LL[:, None] - eps)).any(1)
    return out


def test_ownership_and_visibility_vs_observer():
    b, obs = burst()
    I, g = b.interactions, _C["groups"]
    tri = [np.asarray(m.vertices)[np.asarray(m.faces)] for m in (trimesh.load(OUT / "fixture_identity" / f"{k}.ply", process=False) for k in ("floor", "metal_wall"))]
    tri = np.concatenate(tri + [g.vertices[g.faces].astype(float)])
    vis = ~_blocked(tri, np.broadcast_to(_C["tx"].astype(float), g.vertices.shape).copy(), g.vertices.astype(float))
    off = 0
    for B in obs.B:
        assert "error" not in B, B
        n = B["n_out"]
        if not n:
            continue
        sl = slice(off, off + n)
        kk = I["kind"][0, sl] == 0
        child = B["child"]
        par = B["parents"][child]
        vid = I["vertex"][0, sl]
        assert np.array_equal(kk, B["expanded"][child])
        assert np.array_equal(vid[kk], B["vertex0"][child][kk])  # same id as the generator's own source table
        assert np.all(g.vertex_group[vid[kk]] == B["local_groups"][par][kk])  # official ownership convention
        assert np.all(B["idx"][0, 0, child][kk] >= 0)  # no TX-blocked child retained
        assert np.mean(vis[vid[kk]]) >= 0.99  # independently TX-visible (grazing tolerance)
        for p_ in np.unique(par[kk]):  # children of a parent: subset of its group, no duplicates per depth
            m = kk & (par == p_)
            for d in np.unique(B["depth"][m]):
                ids = vid[m & (B["depth"] == d)]
                assert len(ids) == len(np.unique(ids)) and set(ids) <= set(g.groups[B["local_groups"][p_]])
        off += n
    assert off == b.G["P"]


def test_static_sentinel_and_unknown():
    b, _ = burst()
    I, V, vidx = b.interactions, b.n_vertices, np.asarray(b.G["vidx"])
    st, uk = I["kind"] == 1, I["kind"] == 2
    assert np.all(I["vertex"][st] == V) and np.all(vidx[st] == 1) and np.all(I["face"][st] == -1)
    assert np.all(I["vertex"][uk] == V) and np.all(I["face"][uk] >= 0) and np.all(vidx[uk] == 0)
    a = b.articulation_audit()
    print("   audit", a["interactions"], a["paths_with_unknown"], {k: v["paths"] for k, v in a["by_depth"].items()})
    assert set(a["by_depth"]) >= {1, 2, 3}


def test_depth_alignment():
    b, _ = burst()
    d, k = np.asarray(b.G["depth"]), b.interactions["kind"]
    used = np.arange(k.shape[0])[:, None] < d[None]
    assert np.all(k[~used] == 3) and not np.any(k[used] == 3)


def test_batching_invariance():
    b1, _ = burst(100000)
    b2, _ = burst(37)

    def keys(b):
        I = b.interactions
        p, d = np.asarray(b.G["pts0"]), np.asarray(b.G["depth"])
        K = np.concatenate([I["kind"].T, I["vertex"].T, d[:, None], np.round(p.transpose(1, 0, 2).reshape(len(d), -1) * 1e6)], 1).astype(np.int64)
        return K[np.lexsort(K.T[::-1])]

    assert b2.group_audit["batches"] > b1.group_audit["batches"]
    assert np.array_equal(keys(b1), keys(b2))
    disp = np.zeros((4, 3), np.float32)
    disp[:, 0] = np.arange(4) * 1e-4
    assert rel(b2.signals(disp, chunk=2), b1.signals(disp, chunk=2)) <= THRESHOLDS["batching_signal_rel_l2"]


if __name__ == "__main__":
    _harness.run(globals(), OUT / "test_vertex_identity.json")
