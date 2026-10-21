"""Synthetic controlled scenes for the articulated TemporalBurst tests (written to tests/articulated/_out; no bundled asset is touched).

Scene: floor (plasterboard) + metal back wall, both with flat normals (face_normals=true, so reflections are
specular on the flat faces), and a 'temporal_human' mesh made of separated subdivided boxes, one vertex group per box.
Groups are built directly with the official HumanVertexGroups constructor (face_group per box; each vertex's
representative face = the first face containing it)."""
from pathlib import Path
import numpy as np
import trimesh
from waveverse_sim.grouped_scattering import HumanVertexGroups

SENSOR = np.array([0.0, 0.0, 1.0], np.float32)


def box(center, size, subdiv=2):
    m = trimesh.creation.box(extents=size)
    for _ in range(subdiv):
        m = m.subdivide()
    m.apply_translation(center)
    return m


def human_parts(parts):
    """parts: list of (center, size). Returns trimesh (process=False), face_group, vertex_face, names."""
    meshes = [box(c, s) for c, s in parts]
    V, F, G = [], [], []
    off = 0
    for g, m in enumerate(meshes):
        V.append(np.asarray(m.vertices))
        F.append(np.asarray(m.faces) + off)
        G.append(np.full(len(m.faces), g))
        off += len(m.vertices)
    V, F, G = np.concatenate(V), np.concatenate(F), np.concatenate(G)
    vertex_face = np.full(len(V), -1, np.int64)
    for f in range(len(F) - 1, -1, -1):
        vertex_face[F[f]] = f
    assert (vertex_face >= 0).all()
    return trimesh.Trimesh(V, F, process=False), G, vertex_face, [f"part{g}" for g in range(len(parts))]


def write_scene(out, parts, wall=True):
    """Write the scene XML and meshes; return (xml path, HumanVertexGroups, mesh)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    floor = trimesh.creation.box(extents=[8, 8, 0.02])
    floor.apply_translation([2, 0, -0.01])
    objs = {"floor": floor}
    if wall:
        w = trimesh.creation.box(extents=[0.02, 4, 2.5])
        w.apply_translation([4.0, 0, 1.25])
        objs["metal_wall"] = w
    human, face_group, vertex_face, names = human_parts(parts)
    xml = ['<scene version="2.1.0">', '<bsdf type="diffuse" id="mat-itu_concrete"><rgb name="reflectance" value="0.5 0.5 0.5"/></bsdf>']
    for k, m in objs.items():
        m.export(out / f"{k}.ply")
        xml.append(f'<shape type="ply" id="{k}"><string name="filename" value="{(out / (k + ".ply")).resolve()}"/><boolean name="face_normals" value="true"/><ref id="mat-itu_concrete" name="bsdf"/></shape>')
    hp = (out / "human_reference.ply").resolve()
    human.export(hp)
    xml.append(f'<shape type="ply" id="temporal_human"><string name="filename" value="{hp}"/><ref id="mat-itu_concrete" name="bsdf"/></shape>')
    (out / "scene.xml").write_text("\n".join(xml + ["</scene>"]) + "\n")
    placed = np.asarray(__import__("trimesh").load(hp, process=False).vertices)
    groups = HumanVertexGroups(placed, human.faces, face_group, vertex_face, names)
    return out / "scene.xml", groups, human


ASSIGN = {"metal_wall": "itu_metal_ex", "floor": "itu_plasterboard_ex", "temporal_human": "itu_fabric"}
TWO_PARTS = [((2.0, -0.6, 1.0), (0.25, 0.25, 0.25)), ((2.0, 0.6, 1.0), (0.25, 0.25, 0.25)), ((2.2, 0.0, 1.0), (0.25, 0.25, 0.25))]
