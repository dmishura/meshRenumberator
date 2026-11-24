"""Normalize only geometry encoding, preserving the Original numbering."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from mesh_renumber import foam_object, load_mesh, read_poly_mesh


def normalize(source, output, converter):
    reference = read_poly_mesh(source)
    source = reference.poly_mesh_dir
    if output.exists() or output.is_symlink() or source in output.resolve().parents:
        raise ValueError(f"unsafe output path: {output}")
    with foam_object(source / "faces", {"faceCompactList", "faceList"}) as (_, faces, face_binary):
        if face_binary and faces != "faceCompactList":
            raise ValueError("unsupported binary non-compact faces")
    with foam_object(source / "points", {"vectorField"}) as (_, _, point_binary):
        pass
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".geometry-") as temporary:
        mesh = Path(temporary) / "polyMesh"
        shutil.copytree(source, mesh, ignore=shutil.ignore_patterns("faces", "points"))
        if faces == "faceCompactList" and face_binary and point_binary:
            for name in ("faces", "points"):
                shutil.copyfile(source / name, mesh / name)
        else:
            subprocess.run([str(converter.resolve()), str(source), str(mesh)], check=True)
        normalized, _, _ = load_mesh(mesh)
        if normalized.topology_checksum() != reference.topology_checksum():
            raise ValueError("topology changed during encoding conversion")
        output.mkdir()
        mesh.replace(output)
    print(json.dumps({"original_polyMesh": str(source), "prepared_polyMesh": str(output.resolve()),
                      "encoding_converted": not (faces == "faceCompactList" and face_binary and point_binary),
                      "cells": reference.n_cells, "topology_sha256": reference.topology_checksum()},
                     indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    args = parser.parse_args()
    normalize(args.mesh, args.output, args.converter)
