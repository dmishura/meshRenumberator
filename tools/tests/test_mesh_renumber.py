"""Focused command-line tests for mesh_renumber.py."""

from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
CLI = REPOSITORY / "mesh_renumber.py"


def foam_header(class_name: str, object_name: str, fmt: str) -> bytes:
    return (
        "FoamFile\n{\n"
        f"    format {fmt};\n"
        f"    class {class_name};\n"
        '    arch "LSB;label=32;scalar=64";\n'
        f"    object {object_name};\n"
        "}\n"
    ).encode("ascii")


def binary_list(path: Path, class_name: str, object_name: str,
                values: list[int], *, compact: bool = False) -> None:
    header = foam_header(class_name, object_name, "binary")
    if compact:
        offsets = list(range(0, len(values) + 1, 3))
        body = (f"{len(offsets)}\n(".encode()
                + struct.pack(f"<{len(offsets)}i", *offsets)
                + b"\n)\n"
                + f"{len(values)}\n(".encode()
                + struct.pack(f"<{len(values)}i", *values)
                + b"\n)\n")
    else:
        body = (f"{len(values)}\n(".encode()
                + struct.pack(f"<{len(values)}i", *values)
                + b"\n)\n")
    path.write_bytes(header + body)


def write_fixture(root: Path) -> Path:
    mesh = root / "source" / "constant" / "polyMesh"
    mesh.mkdir(parents=True)
    faces = [0, 1, 2, 0, 3, 1, 1, 3, 2, 2, 3, 0,
             0, 1, 4, 1, 2, 4, 2, 0, 4]
    binary_list(mesh / "faces", "faceCompactList", "faces", faces, compact=True)
    points = [(0., 0., 0.), (1., 0., 0.), (0., 1., 0.),
              (0., 0., 1.), (0., 0., -1.)]
    (mesh / "points").write_bytes(
        foam_header("vectorField", "points", "binary")
        + f"{len(points)}\n(".encode()
        + b"".join(struct.pack("<ddd", *point) for point in points)
        + b"\n)\n"
    )

    def label_file(name: str, vals: list[int]) -> None:
        data = "\n".join(map(str, vals))
        (mesh / name).write_text(
            "FoamFile\n{\n    format ascii;\n    class labelList;\n"
            '    note "nCells:2 nFaces:7 nPoints:5";\n'
            f"    object {name};\n}}\n{len(vals)}\n(\n{data}\n)\n",
            encoding="ascii",
        )
    label_file("owner", [0, 0, 0, 0, 1, 1, 1])
    label_file("neighbour", [1])
    (mesh / "boundary").write_text(
        "FoamFile\n{ format ascii; class polyBoundaryMesh; object boundary; }\n"
        "1\n(\nwall\n{ type wall; nFaces 6; startFace 1; }\n)\n",
        encoding="ascii",
    )
    return mesh


class MeshRenumberCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.mesh = write_fixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(REPOSITORY), env.get("PYTHONPATH", "")]
        )
        return subprocess.run(
            [sys.executable, str(CLI), *args], cwd=REPOSITORY, env=env,
            text=True, capture_output=True, check=False,
        )

    def renumber(self, algorithm: str) -> Path:
        experiment = self.root / algorithm
        result = self.run_cli("renumber", "--mesh", str(self.mesh),
                              "--output", str(experiment), "--algorithm", algorithm)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return experiment

    def test_identity_writes_expected_artifacts_and_validates(self) -> None:
        import numpy as np
        experiment = self.renumber("identity")
        self.assertTrue((experiment / "polyMesh" / "faces").is_file())
        with np.load(experiment / "permutation.npz") as data:
            self.assertEqual(set(data.files), {
                "old_to_new", "new_to_old", "new_face_to_old", "face_flipped"
            })
            self.assertEqual(data["old_to_new"].tolist(), [0, 1])
            self.assertEqual(data["new_to_old"].tolist(), [0, 1])
            self.assertEqual(data["new_face_to_old"].tolist(), list(range(7)))
            self.assertEqual(data["face_flipped"].tolist(), [False] * 7)
        report = json.loads((experiment / "validation.json").read_text())
        self.assertTrue(report.get("valid"))
        result = self.run_cli("validate", "--experiment", str(experiment))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_rcm_roundtrip_preserves_face_orientation(self) -> None:
        import numpy as np
        experiment = self.renumber("rcm")
        with np.load(experiment / "permutation.npz") as data:
            old_to_new = data["old_to_new"]
            new_to_old = data["new_to_old"]
            self.assertEqual(sorted(old_to_new.tolist()), [0, 1])
            self.assertEqual(old_to_new[new_to_old].tolist(), [0, 1])
            self.assertEqual(len(data["new_face_to_old"]), 7)
            self.assertEqual(data["face_flipped"].tolist(), [True] + [False] * 6)
        result = self.run_cli("validate", "--experiment", str(experiment))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_unknown_metadata_is_rejected(self) -> None:
        (self.mesh / "boundary").write_text(
            "FoamFile\n{ format ascii; class polyBoundaryMesh; object boundary; }\n"
            "1\n(\nwall\n{ type wall; nFaces 6; startFace 1; customData 4; }\n)\n",
            encoding="ascii",
        )
        result = self.run_cli("renumber", "--mesh", str(self.mesh),
                              "--output", str(self.root / "bad"), "--algorithm", "identity")
        self.assertNotEqual(result.returncode, 0)

    def test_existing_output_is_protected(self) -> None:
        experiment = self.renumber("identity")
        expected = (experiment / "config.json").read_bytes()
        result = self.run_cli("renumber", "--mesh", str(self.mesh),
                              "--output", str(experiment), "--algorithm", "rcm")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((experiment / "config.json").read_bytes(), expected)

    def test_corrupt_output_fails_validation(self) -> None:
        experiment = self.renumber("identity")
        (experiment / "polyMesh" / "owner").write_text("corrupt\n", encoding="ascii")
        result = self.run_cli("validate", "--experiment", str(experiment))
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
