#!/usr/bin/env python3
"""Generate and verify physically renumbered serial OpenFOAM meshes."""
import argparse
from collections import deque
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from subprocess import CalledProcessError
import sys
import tempfile

import numpy as np

WFGS = Path(os.environ.get("WFGS_ROOT", Path(__file__).resolve().parents[2] /
                           "WavefrontGaussSeidel")).resolve()
sys.path.insert(0, str(WFGS / "tools"))
from openfoam_mesh_reader import (
    _read_header, _read_unsigned, _skip_space_and_comments, read_poly_mesh,
)

I32 = np.dtype("<i4")
ZONES = {"cellZones", "faceZones", "pointZones"}
BASE = {"owner", "neighbour", "faces", "points", "boundary"}
EXTRAS = ZONES | {"cellLevel", "pointLevel", "surfaceIndex", "level0Edge", "sets"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(a, b, label):
    require(np.array_equal(a, b), f"{label} mismatch")


def comments_removed(data):
    return re.sub(rb"//[^\n]*|/\*.*?\*/", b" ", data, flags=re.S)


@contextmanager
def foam_object(path, classes):
    with path.open("rb") as stream:
        header = _read_header(stream, path)
        kind = re.search(rb"\bclass\s+(\w+)\s*;", header)
        fmt = re.search(rb"\bformat\s+(ascii|binary)\s*;", header)
        obj = re.search(rb"\bobject\s+(\w+)\s*;", header)
        require(kind and kind[1].decode() in classes and fmt and obj
                and obj[1].decode() == path.name, f"unsupported header: {path}")
        arch = re.search(rb'\barch\s+"([^"]+)"\s*;', header)
        require(not arch or (b"LSB" in arch[1] and b"label=32" in arch[1]
                             and b"scalar=64" in arch[1]),
                f"unsupported architecture: {path}")
        yield stream, kind[1].decode(), fmt[1] == b"binary"


def foam_list(stream, path, binary, dtype=I32, width=1):
    _skip_space_and_comments(stream, path)
    count = _read_unsigned(stream, path, "list size")
    _skip_space_and_comments(stream, path)
    if count == 0 and stream.peek(1)[:1] != b"(":
        return np.empty(0, dtype=dtype)
    require(stream.read(1) == b"(", f"missing list opening: {path}")
    if binary:
        payload = stream.read(count * width * np.dtype(dtype).itemsize)
        require(len(payload) == count * width * np.dtype(dtype).itemsize, f"truncated binary list: {path}")
        values = np.frombuffer(payload, dtype=dtype)
    else:
        data = bytearray()
        while (byte := stream.read(1)) not in (b")", b""):
            data.extend(byte)
        if byte:
            stream.seek(-1, 1)
        payload = comments_removed(bytes(data))
        require(re.fullmatch(rb"\s*(?:[+-]?\d+\s*)*", payload),
                f"invalid ASCII labels: {path}")
        values = np.fromstring(payload.strip(), dtype=dtype, sep=" ")
    _skip_space_and_comments(stream, path)
    require(len(values) == count * width and stream.read(1) == b")",
            f"invalid list length or closing: {path}")
    return values


def read_arrays(path, classes, lists=1, width=1, dtype=I32):
    with foam_object(path, classes) as (stream, kind, binary):
        require(binary or width == 1, f"expected binary geometry: {path}")
        arrays = [foam_list(stream, path, binary and not kind.endswith("Set"), dtype, width) for _ in range(lists)]
        require(comments_removed(stream.read()).strip() in (b"", b";"),
                f"unexpected trailing data: {path}")
        return kind, arrays


def write_arrays(path, kind, arrays, mesh=None, location="constant/polyMesh"):
    note = ""
    if mesh is not None:
        note = (f'    note "nPoints:{mesh.n_points} nCells:{mesh.n_cells} '
                f'nFaces:{mesh.n_faces} nInternalFaces:{mesh.n_internal_faces}";\n')
    if kind.endswith("Set"):
        values, = arrays
        with path.open("wb") as stream:
            stream.write((f'FoamFile\n{{ format ascii; class {kind}; '
                          f'location "{location}"; object {path.name}; }}\n'
                          f'{values.size}\n(\n').encode())
            np.savetxt(stream, values, fmt="%d")
            stream.write(b")\n")
        return
    with path.open("wb") as stream:
        stream.write((f'FoamFile\n{{\n    version 2.0;\n    format binary;\n'
                      f'    arch "LSB;label=32;scalar=64";\n    class {kind};\n'
                      f'    location "{location}";\n    object {path.name};\n'
                      f'{note}}}\n').encode())
        for values in arrays:
            stream.write(f"{values.size}\n".encode())
            if values.size or path.name in BASE:
                stream.write(b"(")
                values.astype(I32, copy=False).tofile(stream)
                stream.write(b")")
            stream.write(b"\n")


def validate_boundary(path, mesh):
    with foam_object(path, {"polyBoundaryMesh"}) as (stream, _, _):
        data = comments_removed(stream.read()).strip()
    outer = re.fullmatch(rb"(\d+)\s*\((.*)\)\s*;?", data, re.S)
    require(outer, f"invalid boundary: {path}")
    pattern = rb'[\w.%:+-]+\s*\{([^{}]*)\}'
    patches = re.findall(pattern, outer[2])
    require(len(patches) == int(outer[1]) and not re.sub(
        pattern, b"", outer[2]).strip(), f"unsupported boundary syntax: {path}")
    next_face = mesh.n_internal_faces
    for patch in patches:
        kind = re.search(rb"\btype\s+(\w+)\s*;", patch)
        start = re.search(rb"\bstartFace\s+(\d+)\s*;", patch)
        count = re.search(rb"\bnFaces\s+(\d+)\s*;", patch)
        require(kind and kind[1] in {b"wall", b"patch", b"empty", b"symmetry",
                                    b"symmetryPlane", b"wedge"}
                and start and count and int(start[1]) == next_face
                and set(re.findall(rb"(?:^|;)\s*(\w+)\s", patch))
                <= {b"type", b"nFaces", b"startFace", b"inGroups"}
                and not re.search(rb"[#$]", patch), f"unsupported patch: {patch!r}")
        next_face += int(count[1])
    require(next_face == mesh.n_faces, "boundary does not cover boundary faces")


def load_mesh(path):
    mesh = read_poly_mesh(path)
    require(mesh.poly_mesh_dir == path.resolve(), "pass a polyMesh directory")
    entries = {item.name for item in path.iterdir()}
    require(BASE <= entries and entries <= BASE | EXTRAS,
            f"missing mesh files or unsupported metadata: {sorted((BASE - entries) | (entries - BASE - EXTRAS))}")
    require(all(not item.is_symlink() and
                (item.is_dir() if item.name == "sets" else item.is_file())
                for item in path.iterdir()), "unsupported mesh directory entry")
    _, (offsets, vertices) = read_arrays(path / "faces", {"faceCompactList"}, 2)
    _, (points,) = read_arrays(path / "points", {"vectorField"}, width=3, dtype="<f8")
    require(points.size == 3 * mesh.n_points and np.isfinite(points).all(),
            "invalid points or nPoints")
    require(offsets.size == mesh.n_faces + 1 and offsets[0] == 0
            and offsets[-1] == vertices.size and np.all(np.diff(offsets.astype("i8")) >= 3)
            and np.all((vertices >= 0) & (vertices < mesh.n_points)),
            "invalid compact face addressing")
    validate_boundary(path / "boundary", mesh)
    return mesh, offsets, vertices


def rcm(mesh):
    owner = np.asarray(mesh.internal_owner)
    neighbour = np.asarray(mesh.neighbour)
    rows = np.concatenate((owner, neighbour))
    cols = np.concatenate((neighbour, owner))
    degree = np.bincount(rows, minlength=mesh.n_cells)
    cols = cols[np.lexsort((cols, degree[cols], rows))]
    starts = np.r_[0, np.cumsum(degree)]
    visited = bytearray(mesh.n_cells)
    order = []
    for root in np.argsort(degree, kind="stable").tolist():
        if visited[root]:
            continue
        queue = deque([root])
        visited[root] = 1
        component = []
        while queue:
            cell = queue.popleft()
            component.append(cell)
            for other in cols[starts[cell]:starts[cell + 1]].tolist():
                if not visited[other]:
                    visited[other] = 1
                    queue.append(other)
        order.extend(reversed(component))
    return np.asarray(order, dtype=I32)


def inverse(order, size):
    require(order.dtype == I32 and order.shape == (size,)
            and np.all((order >= 0) & (order < size)), "invalid permutation labels")
    result = np.full(size, -1, dtype=I32)
    result[order] = np.arange(size, dtype=I32)
    require(np.all(result >= 0), "permutation is not bijective")
    return result


def remap(mesh, old_to_new, identity):
    owner = old_to_new[np.asarray(mesh.owner)]
    neighbour = old_to_new[np.asarray(mesh.neighbour)]
    n = neighbour.size
    flipped = owner[:n] > neighbour
    low = np.minimum(owner[:n], neighbour)
    high = np.maximum(owner[:n], neighbour)
    internal_order = np.arange(n) if identity else np.lexsort((high, low))
    order = np.concatenate((internal_order, np.arange(n, owner.size))).astype(I32)
    face_flipped = np.zeros(owner.size, dtype=bool)
    face_flipped[:n] = flipped[internal_order]
    owner[:n] = low
    return owner[order], high[internal_order], order, face_flipped


def remap_faces(offsets, vertices, order, flipped):
    lengths = np.diff(offsets)[order]
    new_offsets = np.r_[0, np.cumsum(lengths)].astype(I32)
    relative = np.arange(vertices.size, dtype=I32) - np.repeat(new_offsets[:-1], lengths)
    reverse = np.repeat(flipped, lengths) & (relative != 0)
    relative[reverse] = np.repeat(lengths, lengths)[reverse] - relative[reverse]
    return new_offsets, vertices[np.repeat(offsets[order], lengths) + relative]


def read_cell_zones(path, n_cells):
    """Read standard cellZone entries, including a FoamFile meta names list."""
    def token(stream, expected):
        _skip_space_and_comments(stream, path)
        require(stream.read(len(expected)) == expected, f"expected {expected!r} in {path}")

    zones = {}
    with foam_object(path, {"regIOobject", "cellZoneMesh"}) as (stream, _, binary):
        _skip_space_and_comments(stream, path)
        if stream.peek(1)[:1] == b"}":
            stream.read(1)  # Outer FoamFile close after the nested meta dictionary.
        _skip_space_and_comments(stream, path)
        count = _read_unsigned(stream, path, "zone count")
        token(stream, b"(")
        names = re.compile(rb"[A-Za-z0-9_.+-]+")
        for _ in range(count):
            _skip_space_and_comments(stream, path)
            name = names.match(stream.peek())
            require(name is not None, f"invalid zone name: {path}")
            raw = stream.read(len(name[0]))
            require(raw not in zones, f"duplicate zone {raw!r} in {path}")
            for item in (b"{", b"type", b"cellZone", b";", b"cellLabels", b"List<label>"):
                token(stream, item)
            labels = foam_list(stream, path, binary)
            _skip_space_and_comments(stream, path)
            if stream.peek(1)[:1] == b";":
                stream.read(1)
            token(stream, b"}")
            require(np.all((labels >= 0) & (labels < n_cells)), f"invalid cell label: {path}")
            zones[raw] = labels
        token(stream, b")")
        require(comments_removed(stream.read()).strip() in (b"", b";"),
                f"unexpected zone text: {path}")
    return zones


def write_cell_zones(path, zones):
    with path.open("wb") as stream:
        stream.write(b'FoamFile\n{ version 2.0; format binary; arch "LSB;label=32;scalar=64"; '
                     b'class regIOobject; location "constant/polyMesh"; object cellZones;\n')
        stream.write(b"meta { names ( " + b" ".join(zones) + b" ); }\n}\n")
        stream.write(f"{len(zones)}\n(\n".encode())
        for name, labels in zones.items():
            stream.write(name + b"\n{ type cellZone; cellLabels List<label>\n")
            stream.write(f"{labels.size}\n(".encode())
            labels.astype(I32, copy=False).tofile(stream)
            stream.write(b")\n; }\n")
        stream.write(b")\n")


def metadata(source, mesh, new_to_old, old_to_new, order):
    """Yield copies or remapped arrays for supported mesh metadata."""
    for name in sorted(EXTRAS - {"sets"}):
        path = source / name
        if not path.exists():
            continue
        if name == "cellZones":
            zones = read_cell_zones(path, mesh.n_cells)
            if zones:
                yield Path(name), "cellZones", {zone: old_to_new[labels]
                                               for zone, labels in zones.items()}
                continue
        elif name in ZONES:
            with foam_object(path, {"regIOobject", "cellZoneMesh", "faceZoneMesh",
                                    "pointZoneMesh"}) as (stream, _, _):
                body = comments_removed(stream.read())
                empty = re.fullmatch(rb"\s*0\s*\(\s*\)\s*;?\s*", body)
                # OpenFOAM can retain an empty named point zone in a nested
                # FoamFile meta dictionary. Point labels are never renumbered.
                if name == "pointZones" and not empty:
                    empty = re.fullmatch(
                        rb"\s*\}\s*1\s*\(\s*[\w.-]+\s*\{\s*"
                        rb"type\s+pointZone\s*;\s*pointLabels\s+List<label>\s+0"
                        rb"\s*(?:\(\s*\))?\s*;"
                        rb"\s*\}\s*\)\s*;?\s*", body)
                require(empty, f"nonempty or unsupported zone file: {path}")
        elif name in {"cellLevel", "surfaceIndex", "pointLevel"}:
            kind, (values,) = read_arrays(path, {"labelList"})
            size = {"cellLevel": mesh.n_cells, "surfaceIndex": mesh.n_faces,
                    "pointLevel": mesh.n_points}[name]
            require(values.size == size, f"incorrect metadata size: {path}")
            if name != "pointLevel":
                yield Path(name), kind, values[new_to_old if name == "cellLevel" else order]
                continue
        else:
            with foam_object(path, {"uniformDimensionedScalarField"}):
                pass
        yield Path(name), None, None

    sets = source / "sets"
    if sets.exists():
        old_face_to_new = inverse(order, mesh.n_faces)
        for path in sorted(sets.iterdir()):
            require(path.is_file() and not path.is_symlink(), f"unsupported set: {path}")
            kind, (values,) = read_arrays(path, {"cellSet", "faceSet", "pointSet"})
            size = {"cellSet": mesh.n_cells, "faceSet": mesh.n_faces,
                    "pointSet": mesh.n_points}[kind]
            require(np.all((values >= 0) & (values < size)), f"out of range set: {path}")
            mapping = {"cellSet": old_to_new, "faceSet": old_face_to_new,
                       "pointSet": None}[kind]
            yield Path("sets") / path.name, kind, values if mapping is None else mapping[values]


def digest(path):
    with path.open("rb") as stream:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def json_write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def validate(experiment, source=None):
    config = json.loads((experiment / "config.json").read_text())
    source = source or Path(config["source"])
    output = experiment / "polyMesh"
    old, old_offsets, old_vertices = load_mesh(source)
    new, offsets, vertices = load_mesh(output)
    require((old.n_cells, old.n_faces, old.n_internal_faces, old.n_points) ==
            (new.n_cells, new.n_faces, new.n_internal_faces, new.n_points),
            "mesh sizes changed")
    with np.load(experiment / "permutation.npz") as maps:
        new_to_old, old_to_new, order, flipped = (
            maps[key] for key in ("new_to_old", "old_to_new", "new_face_to_old", "face_flipped"))
    equal(inverse(new_to_old, old.n_cells), old_to_new, "inverse cell permutation")
    inverse(order, old.n_faces)
    n = old.n_internal_faces
    require(flipped.dtype == np.bool_ and flipped.shape == order.shape
            and np.all(order[:n] < n) and not np.any(flipped[n:]),
            "invalid internal face map")
    equal(order[n:], np.arange(n, old.n_faces), "boundary face numbering")
    restored_owner = new_to_old[np.asarray(new.owner)]
    restored_neighbour = new_to_old[np.asarray(new.neighbour)]
    equal(np.where(flipped[:n], restored_neighbour, restored_owner[:n]),
          np.asarray(old.owner)[order[:n]], "inverse internal owner")
    equal(np.where(flipped[:n], restored_owner[:n], restored_neighbour),
          np.asarray(old.neighbour)[order[:n]], "inverse neighbour")
    equal(restored_owner[n:], np.asarray(old.owner)[n:], "inverse boundary owner")
    expected_offsets, expected_vertices = remap_faces(old_offsets, old_vertices, order, flipped)
    equal(offsets, expected_offsets, "face offsets")
    equal(vertices, expected_vertices, "oriented face vertices")
    expected_files = set(BASE)
    for name in ("points", "boundary"):
        require(digest(source / name) == digest(output / name), f"{name} changed")
    for relative, kind, values in metadata(source, old, new_to_old, old_to_new, order):
        expected_files.add(str(relative))
        if kind is None:
            require(digest(source / relative) == digest(output / relative),
                    f"{relative} changed")
        elif kind == "cellZones":
            actual = read_cell_zones(output / relative, new.n_cells)
            require(list(actual) == list(values), "cell zone names changed")
            for name, labels in values.items():
                equal(actual[name], labels, f"cell zone {name!r}")
        else:
            _, (actual,) = read_arrays(output / relative, {kind})
            equal(actual, values, str(relative))
    require({str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
            == expected_files, "unexpected output metadata")
    return {"valid": True, "cells": new.n_cells, "faces": new.n_faces, "internal_faces": n,
            "source_topology_sha256": old.topology_checksum(),
            "output_topology_sha256": new.topology_checksum(),
            "permutation_sha256": digest(experiment / "permutation.npz"),
            "flipped_faces": int(flipped.sum()),
            "validated": ["permutations", "LDU", "inverse topology", "oriented vertices",
                          "points", "boundary", "mesh metadata"]}


def renumber(args):
    source = args.mesh.resolve()
    destination = args.output.resolve()
    require(not args.output.is_symlink() and not destination.exists(),
            f"output already exists: {destination}")
    require(source not in destination.parents, "output cannot be inside source mesh")
    mesh, offsets, vertices = load_mesh(source)
    algorithm = "original" if args.algorithm == "identity" else args.algorithm
    optimization = None
    partition_result = None
    if algorithm == "existing":
        require(args.initial_permutation is not None, "existing requires --initial-permutation")
        with np.load(args.initial_permutation) as maps:
            new_to_old = maps["new_to_old"]
            equal(inverse(new_to_old, mesh.n_cells), maps["old_to_new"], "imported inverse")
    elif algorithm == "locality":
        from locality import optimize

        require(args.initial_permutation is None, "locality always starts from Original numbering")
        new_to_old, optimization = optimize(mesh, args)
    elif algorithm == "cache-traversal":
        from cache_traversal import run

        require(args.initial_permutation is None, "cache-traversal starts from Original numbering")
        provenance = source.parent / "config.json"
        if provenance.is_file():
            prior = json.loads(provenance.read_text())
            require(not isinstance(prior, dict) or prior.get("algorithm") not in
                    {"rcm", "locality", "cache-partition", "cache-traversal", "existing"},
                    "pass the Original mesh to cache-traversal")
        run(source, destination, mesh, offsets, vertices, args)
        print(destination)
        return
    elif algorithm == "cache-partition":
        from cache_partition import choose_partition

        require(args.initial_permutation is None, "cache-partition starts from Original numbering")
        provenance = source.parent / "config.json"
        if provenance.is_file():
            prior = json.loads(provenance.read_text())
            require(prior.get("algorithm") not in {"rcm", "locality", "cache-partition", "existing"},
                    "pass the Original mesh to cache-partition")
        new_to_old, report, partition_arrays = choose_partition(mesh, args)
        partition_result = (report, partition_arrays)
    else:
        require(args.initial_permutation is None, "permutation requires --algorithm existing")
        new_to_old = (rcm(mesh) if args.algorithm == "rcm"
                      else np.arange(mesh.n_cells, dtype=I32))
    write_numbered_experiment(source, destination, mesh, offsets, vertices, algorithm,
                              new_to_old, args, optimization, partition_result)
    print(destination)


def write_numbered_experiment(source, destination, mesh, offsets, vertices, algorithm,
                              new_to_old, args, optimization=None, partition_result=None):
    old_to_new = inverse(new_to_old, mesh.n_cells)
    owner, neighbour, order, flipped = remap(mesh, old_to_new, algorithm == "original")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".renumber-") as temp:
        experiment = Path(temp)
        output = experiment / "polyMesh"
        output.mkdir()
        for name in ("points", "boundary"):
            shutil.copyfile(source / name, output / name)
        for name, array in (("owner", owner), ("neighbour", neighbour)):
            write_arrays(output / name, "labelList", [array], mesh)
        write_arrays(output / "faces", "faceCompactList",
                     remap_faces(offsets, vertices, order, flipped))
        for relative, kind, values in metadata(source, mesh, new_to_old, old_to_new, order):
            target = output / relative
            target.parent.mkdir(exist_ok=True)
            if kind is None:
                shutil.copyfile(source / relative, target)
            elif kind == "cellZones":
                write_cell_zones(target, values)
            else:
                location = "constant/polyMesh"
                if relative.parent != Path("."):
                    location += f"/{relative.parent}"
                write_arrays(target, kind, [values], location=location)
        np.savez(experiment / "permutation.npz", new_to_old=new_to_old,
                 old_to_new=old_to_new, new_face_to_old=order, face_flipped=flipped)
        config = {"source": str(source), "algorithm": algorithm,
                  "source_topology_sha256": mesh.topology_checksum()}
        if optimization is not None:
            from locality import match_metrics

            match_metrics(read_poly_mesh(output), optimization["final"])
            require(optimization["final"]["coefficient_transitions"]
                    <= optimization["coefficient_limit"], "coefficient locality limit exceeded")
            optimization["settings"]["optimizer_sha256"] = digest(args.optimizer)
            config["optimization"] = optimization["settings"]
            json_write(experiment / "optimization.json", optimization)
        if partition_result is not None:
            report, partition_arrays = partition_result
            report["settings"]["partition_backend_sha256"] = digest(args.partition_backend)
            report["settings"]["working_set_backend_sha256"] = digest(args.partition_ws_backend)
            config["partition"] = report["settings"]
            json_write(experiment / "partition.json", report)
            np.savez(experiment / "partitions.npz", **partition_arrays)
        json_write(experiment / "config.json", config)
        json_write(experiment / "validation.json", validate(experiment))
        # Reserve the path before moving files, so even an empty existing output is protected.
        destination.mkdir()
        experiment.replace(destination)


def main():
    from experiment_stages import analyze, benchmark, compare
    from working_set import DEFAULT_CAPACITIES, DEFAULT_WINDOWS, analyze_locality

    def positive_list(value):
        try:
            values = tuple(map(int, value.split(",")))
        except ValueError as error:
            raise argparse.ArgumentTypeError("expected comma separated integers") from error
        if any(x <= 0 for x in values) or len(set(values)) != len(values):
            raise argparse.ArgumentTypeError("expected distinct positive integers")
        return values

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("renumber")
    generate.add_argument("--mesh", type=Path, required=True)
    generate.add_argument("--output", "--output-dir", type=Path, required=True)
    generate.add_argument("--algorithm", metavar="{original,rcm,locality,cache-partition,cache-traversal,existing}",
                          choices=("original", "rcm", "locality", "cache-partition", "cache-traversal", "existing", "identity"),
                          required=True)
    generate.add_argument("--initial", choices=("original",), default="original")
    generate.add_argument("--objective", choices=("total", "balanced"), default="total")
    generate.add_argument("--balance-weight", "--lambda", type=float, default=0.5)
    generate.add_argument("--coeff-tolerance", type=float, default=0.0)
    generate.add_argument("--candidates", type=int, default=1_000_000)
    generate.add_argument("--window", type=int, default=16)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--patience", type=int, default=100_000)
    generate.add_argument("--verify-every", type=int, default=100_000)
    generate.add_argument("--optimizer", type=Path, default=Path(
        __file__).resolve().parents[1] / "experiments/build/locality_optimizer")
    generate.add_argument("--initial-permutation", type=Path)
    generate.add_argument("--partition-budget-kib", type=int, default=64)
    generate.add_argument("--partition-seed", type=int, default=42)
    generate.add_argument("--partition-local-order", choices=("auto", "traversal", "rcm"), default="auto")
    generate.add_argument("--partition-coarse-order", choices=("auto", "breadth", "depth"), default="auto")
    generate.add_argument("--partition-coeff-tolerance", type=float, default=0.0)
    generate.add_argument("--partition-lru", action="store_true")
    generate.add_argument("--partition-cache-kib", type=positive_list, default=(32, 64, 256, 1024))
    build_dir = Path(__file__).resolve().parents[1] / "experiments/build"
    generate.add_argument("--partition-backend", type=Path, default=build_dir / "cache_partition_backend")
    generate.add_argument("--partition-ws-backend", type=Path, default=build_dir / "working_set_backend")
    def weights(value):
        try:
            values = tuple(float(x) for x in value.split(","))
        except ValueError as error:
            raise argparse.ArgumentTypeError("expected four comma separated weights") from error
        if len(values) != 4:
            raise argparse.ArgumentTypeError("expected weights NEW,REUSE,SPAN,FRONT")
        return values

    generate.add_argument("--traversal-profile", choices=("auto", "breadth", "reuse", "balanced"),
                          default="auto")
    generate.add_argument("--traversal-weights", type=weights, metavar="NEW,REUSE,SPAN,FRONT")
    generate.add_argument("--traversal-frontier-scan", type=int)
    generate.add_argument("--traversal-recent-rows", type=int, default=128)
    generate.add_argument("--traversal-root", choices=("seed", "peripheral", "centre"))
    generate.add_argument("--traversal-seed", type=int, default=42)
    generate.add_argument("--traversal-verify-every", type=int, default=0)
    generate.add_argument("--traversal-coeff-tolerance", type=float, default=0.0)
    generate.add_argument("--traversal-schedule-width", type=int, default=1024)
    generate.add_argument("--traversal-cache-kib", type=positive_list, default=(32, 64, 256, 1024))
    generate.add_argument("--max-wave-ratio", type=float, default=1.5)
    generate.add_argument("--min-median-wave-width", type=int, default=512)
    generate.add_argument("--traversal-backend", type=Path, default=build_dir / "cache_traversal_backend")
    generate.add_argument("--traversal-ws-backend", type=Path, default=build_dir / "working_set_backend")
    generate.add_argument("--traversal-layout-probe", type=Path, default=build_dir / "gs_layout")
    check = commands.add_parser("validate")
    check.add_argument("--experiment", type=Path, required=True)
    check.add_argument("--mesh", type=Path, help="override the original source path")
    analysis = commands.add_parser("analyze")
    analysis.add_argument("--experiment", type=Path, required=True)
    analysis.add_argument("--centres-export", type=Path)
    analysis.add_argument("--layout-probe", type=Path)
    analysis.add_argument("--layout-width", type=int, default=1024)
    timing = commands.add_parser("benchmark")
    timing.add_argument("--experiment", type=Path, required=True)
    timing.add_argument("--probe", type=Path, default=Path(
        __file__).resolve().parents[1] / "experiments/build/gs_probe")
    timing.add_argument("--threads", type=int, default=4)
    timing.add_argument("--sweeps", type=int, default=64)
    timing.add_argument("--repetitions", type=int, default=7)
    timing.add_argument("--widths", nargs="+", type=int, default=[1024])
    comparison = commands.add_parser("compare")
    comparison.add_argument("runs", type=Path, nargs="+")
    comparison.add_argument("--tomography", type=Path)
    comparison.add_argument("--centres-export", type=Path)
    diagnostic = commands.add_parser("analyze-locality")
    diagnostic.add_argument("--mesh", type=Path, required=True)
    diagnostic.add_argument("--output-dir", "--output", type=Path, required=True)
    diagnostic.add_argument("--window-sizes", type=positive_list, default=DEFAULT_WINDOWS)
    diagnostic.add_argument("--stride", type=int, help="override the stride for every window size")
    diagnostic.add_argument("--reuse-distance", action="store_true")
    diagnostic.add_argument("--cache-kib", type=positive_list, default=DEFAULT_CAPACITIES)
    diagnostic.add_argument("--face-permutation", type=Path)
    diagnostic.add_argument("--backend", type=Path, default=Path(
        __file__).resolve().parents[1] / "experiments/build/working_set_backend")
    args = parser.parse_args()
    try:
        if args.command == "renumber":
            renumber(args)
        elif args.command == "validate":
            print(json.dumps(validate(args.experiment, args.mesh), indent=2))
        else:
            {"analyze": analyze, "benchmark": benchmark, "compare": compare,
             "analyze-locality": analyze_locality}[args.command](args)
    except (RuntimeError, OSError, ValueError, KeyError, CalledProcessError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
