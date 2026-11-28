#!/usr/bin/env python3
"""GS-specific structural and cache-locality analysis for archived numberings.

This is an offline analyzer. It never writes meshes or runs solver benchmarks.
Large per-access reuse-distance calculations use working_set_backend.C.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import sys
import tempfile
import time

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "../WavefrontGaussSeidel/tools"))
sys.path.insert(0, str(ROOT / "tools"))
from openfoam_mesh_reader import read_poly_mesh
from mesh_renumber import remap
from working_set import summarize_reuse, distribution, graph_metrics

EXP = ROOT / "experiments/m6-mesh-applicability-20261125/experiments"
SOURCE = {
    "MotorBike": "../WavefrontGaussSeidel/analysis-results/face-structural/source/MotorBike/polyMesh",
    "AHBody": "../WavefrontGaussSeidel/analysis-results/face-structural/source/AHBody_original/polyMesh",
    "Cavity3d": "/home/dmishura/OpenFOAM/OFM-v2026/rundir/Cavity3d_1791730800_mesh/constant/polyMesh",
    "ConicalDiffuser": "/home/dmishura/OpenFOAM/OFM-v2026/rundir/ConicalDiffuser3M_1791709835_mesh/constant/polyMesh",
    "MTBHPC": "/home/dmishura/OpenFOAM/OFM-v2026/rundir/MTBHPC_Small_1789294406/constant/polyMesh",
    "WD_DamBreak": "/home/dmishura/OpenFOAM/OFM-v2026/rundir/WD_DamBreak_1790076487_mesh/constant/polyMesh",
}
EXISTING = {
    "AHBody": "experiments/locality-20261123T231847/AHBody/existing_renumbered/polyMesh",
    "Cavity3d": "/home/dmishura/OpenFOAM/OFM-v2026/rundir/Cavity3d_1791731776_mesh/constant/polyMesh",
    "ConicalDiffuser": "/home/dmishura/OpenFOAM/OFM-v2026/rundir/ConicalDiffuser3M_1791710694_mesh/constant/polyMesh",
    "MTBHPC": "/home/dmishura/OpenFOAM/OFM-v2026/rundir/MTBHPC_Small_rn_1789333137/constant/polyMesh",
    "WD_DamBreak": "/home/dmishura/OpenFOAM/OFM-v2026/rundir/WD_DamBreak_1790066877_mesh_renumbered/constant/polyMesh",
}
ORDERINGS = {
    "MotorBike": ("original", "rcm", "C", "D"),
    "AHBody": ("original", "existing_renumbered", "rcm", "C", "D"),
    "Cavity3d": ("original", "existing_renumbered", "rcm", "C", "D"),
    "ConicalDiffuser": ("original", "existing_renumbered", "rcm", "C", "D"),
    "MTBHPC": ("original", "existing_renumbered", "rcm", "C", "D"),
    "WD_DamBreak": ("original", "existing_renumbered", "rcm", "C", "D"),
}
LINE_WINDOWS = (4, 8, 16, 32, 64, 128)


def quantiles(a):
    a = np.asarray(a)
    if not a.size:
        return {"mean": 0., "median": 0., "p50": 0., "p90": 0., "p95": 0.,
                "p99": 0., "p999": 0., "max": 0.}
    q = np.percentile(a, [50, 90, 95, 99, 99.9])
    return {"mean": float(np.mean(a)), "median": float(q[0]), "p50": float(q[0]),
            "p90": float(q[1]), "p95": float(q[2]), "p99": float(q[3]),
            "p999": float(q[4]), "max": int(np.max(a))}


def histogram(values):
    h = np.bincount(np.asarray(values, dtype=np.int64))
    return {str(i): int(v) for i, v in enumerate(h) if v}


@njit(cache=True)
def chunk_locality(starts, upper, width, terminal_row):
    # Main kernels issue width indirect psi loads per chunk. Terminal suffixes
    # use the M6 helper (width four), where inactive index lanes become psi[0].
    nch = 0
    for c in range(starts.size - 1):
        d = starts[c + 1] - starts[c]
        w = width if c < terminal_row else 4
        if d:
            nch += (d + w - 1) // w
    lines_n = np.empty(nch, np.int8)
    pages4_n = np.empty(nch, np.int8)
    pages2_n = np.empty(nch, np.int8)
    maxdist = np.empty(nch, np.int32)
    pairmean = np.empty(nch, np.float64)
    consecutive = np.empty(max(0, nch * max(0, width - 1)), np.int32)
    # The active chunk geometry is kept distinct from executed tail loads.
    active_n = 0
    same_line_next = 0
    same_xmm = 0
    full_one_line = 0
    counts = np.zeros(width + 1, np.int64)
    align_sum = np.zeros(8, np.float64)
    align_samples = 0
    ci = 0
    for c in range(starts.size - 1):
        begin, end = starts[c], starts[c + 1]
        d = end - begin
        if d <= 0:
            continue
        w = width if c < terminal_row else 4
        face = begin
        while face < end:
            n = min(w, end - face)
            lo = upper[face]
            hi = lo
            unique_lines = np.zeros(width, np.int64)
            unique4 = np.zeros(width, np.int64)
            unique2 = np.zeros(width, np.int64)
            for k in range(n):
                v = upper[face + k]
                if v < lo: lo = v
                if v > hi: hi = v
                l = v // 8
                p4 = v // 512
                p2 = v // 262144
                found = False
                for z in range(k):
                    if unique_lines[z] == l:
                        found = True; break
                if not found: unique_lines[k] = l
                else: unique_lines[k] = -1
                found = False
                for z in range(k):
                    if unique4[z] == p4:
                        found = True; break
                if not found: unique4[k] = p4
                else: unique4[k] = -1
                found = False
                for z in range(k):
                    if unique2[z] == p2:
                        found = True; break
                if not found: unique2[k] = p2
                else: unique2[k] = -1
            ul = 0; u4 = 0; u2 = 0
            for k in range(n):
                if unique_lines[k] >= 0: ul += 1
                if unique4[k] >= 0: u4 += 1
                if unique2[k] >= 0: u2 += 1
            lines_n[ci] = ul; pages4_n[ci] = u4; pages2_n[ci] = u2
            md = 0; total = 0; pairs = 0
            for a in range(n):
                va = upper[face + a]
                for b in range(a + 1, n):
                    vb = upper[face + b]
                    dd = abs(va - vb)
                    if dd > md: md = dd
                    total += dd; pairs += 1
                if a + 1 < n:
                    consecutive[active_n] = abs(va - upper[face + a + 1]); active_n += 1
                    if va // 8 == upper[face + a + 1] // 8: same_line_next += 1
            maxdist[ci] = md
            pairmean[ci] = total / pairs if pairs else 0.
            counts[n] += 1
            if n == width and ul == 1: full_one_line += 1
            if n >= 2:
                for a in range(0, n - 1, 2):
                    if a + 1 < n and upper[face+a] // 8 == upper[face+a+1] // 8:
                        same_xmm += 1
            if ci % 1024 == 0:
                for off in range(8):
                    un=0
                    for a in range(n):
                        va=(upper[face+a]*8+off*8)//64
                        seen=False
                        for b in range(a):
                            if (upper[face+b]*8+off*8)//64==va:
                                seen=True; break
                        if not seen: un+=1
                    align_sum[off]+=un
                align_samples+=1
            ci += 1
            face += w
    return lines_n, pages4_n, pages2_n, maxdist, pairmean, consecutive[:active_n], counts, same_line_next, same_xmm, full_one_line, align_sum/max(1,align_samples)


def build_layout(mesh, case, numbering):
    exp = EXP / case / numbering
    if numbering == "existing_renumbered":
        path = (ROOT / EXISTING[case]).resolve() if not Path(EXISTING[case]).is_absolute() else Path(EXISTING[case])
        m = read_poly_mesh(path)
        # Existing meshes were previously validated as exact topology-preserving
        # outputs; check their archived validation and OpenFOAM LDU invariants.
        val = json.loads((exp / "validation.json").read_text())
        if not val.get("valid") or val.get("output_topology_sha256") != val.get("source_topology_sha256"):
            raise ValueError(f"archived topology validation failed for {case}/{numbering}")
        if m.topology_checksum() != val["output_topology_sha256"]:
            raise ValueError(f"existing-renumbered source checksum mismatch for {case}")
        return np.asarray(m.internal_owner, dtype=np.int32), np.asarray(m.neighbour, dtype=np.int32), m.n_cells, "archived validated existing-renumbered mesh"
    base = Path(SOURCE[case]); base = base if base.is_absolute() else (ROOT / base).resolve()
    src = read_poly_mesh(base)
    cfg=json.loads((exp/"config.json").read_text())
    if src.topology_checksum()!=cfg["source_topology_sha256"]:
        raise ValueError(f"original source checksum mismatch for {case}/{numbering}")
    owner0 = np.asarray(src.internal_owner, dtype=np.int32)
    neigh0 = np.asarray(src.neighbour, dtype=np.int32)
    if numbering == "original":
        return owner0.copy(), neigh0.copy(), src.n_cells, src.topology_checksum()
    z = np.load(exp / "permutation.npz")
    new_to_old = z["new_to_old"].astype(np.int32, copy=False)
    old_to_new = z["old_to_new"].astype(np.int32, copy=False)
    if len(new_to_old) != src.n_cells or not np.array_equal(new_to_old[old_to_new], np.arange(src.n_cells, dtype=np.int32)):
        raise ValueError(f"invalid cell permutation {case}/{numbering}")
    actual_owner, actual_neigh, order, flipped = remap(src, old_to_new, numbering == "original")
    stored_order = z["new_face_to_old"]
    if not np.array_equal(order, stored_order):
        raise ValueError(f"face order mismatch {case}/{numbering}")
    # Exact per-face graph identity after mapping new labels back to source IDs.
    back_a = new_to_old[actual_owner[:len(neigh0)]]
    back_b = new_to_old[actual_neigh]
    got_lo = np.minimum(back_a, back_b); got_hi = np.maximum(back_a, back_b)
    old_a = owner0[order[:len(neigh0)]]; old_b = neigh0[order[:len(neigh0)]]
    exp_lo = np.minimum(old_a, old_b); exp_hi = np.maximum(old_a, old_b)
    if not np.array_equal(got_lo, exp_lo) or not np.array_equal(got_hi, exp_hi):
        raise ValueError(f"graph connectivity changed {case}/{numbering}")
    # LDU requires low-label owner and face order grouped by owner.
    if np.any(actual_owner[:len(neigh0)] >= actual_neigh) or np.any(actual_owner[1:len(neigh0)] < actual_owner[:len(neigh0)-1]):
        raise ValueError(f"invalid LDU owner/neighbour layout {case}/{numbering}")
    return actual_owner[:len(neigh0)].copy(), actual_neigh.copy(), src.n_cells, src.topology_checksum()


def owner_arrays(n, owner, neighbour):
    od = np.bincount(owner, minlength=n).astype(np.int32)
    nd = np.bincount(neighbour, minlength=n).astype(np.int32)
    starts = np.empty(n + 1, np.int64); starts[0] = 0
    np.cumsum(od, out=starts[1:])
    if starts[-1] != len(neighbour): raise ValueError("owner CSR mismatch")
    # The OpenFOAM LDU internal face order is already grouped by owner.
    upper = neighbour
    return od, nd, starts, upper


@njit(cache=True)
def adjacent_swap_deltas(n, owner, neighbour, degree, rowstart, adjacency, max_degree):
    """Exact owner-row chunk-count change for swapping each connected pair."""
    ecount=owner.size
    hist=np.zeros(4*max_degree+1,np.int64)
    markers=np.zeros(n,np.int32); delta=np.zeros(n,np.int32)
    touched=np.empty(n,np.int32); token=0
    improve=neutral=worse=0
    for e in range(ecount):
        a=owner[e]; b=neighbour[e]; token+=1; nt=0
        # Process each incident edge exactly once; the a-b candidate edge is
        # skipped in b's adjacency list because it was already visited at a.
        for side in range(2):
            x=a if side==0 else b
            for p in range(rowstart[x],rowstart[x+1]):
                y=adjacency[p]
                if side==1 and y==a: continue
                nx=b if x==a else (a if x==b else x)
                ny=b if y==a else (a if y==b else y)
                old=min(x,y); new=min(nx,ny)
                if old==new: continue
                if markers[old]!=token:
                    markers[old]=token; delta[old]=0; touched[nt]=old; nt+=1
                if markers[new]!=token:
                    markers[new]=token; delta[new]=0; touched[nt]=new; nt+=1
                delta[old]-=1; delta[new]+=1
        change=0
        for i in range(nt):
            v=touched[i]
            d0=degree[v]; d1=d0+delta[v]
            if d1<0: return improve,neutral,worse,hist
            change+=(d1+3)//4-(d0+3)//4
        hist[change+2*max_degree]+=1
        if change<0: improve+=1
        elif change==0: neutral+=1
        else: worse+=1
    return improve,neutral,worse,hist


def adjacency_csr(n, owner, neighbour):
    degree=np.bincount(np.r_[owner,neighbour],minlength=n).astype(np.int32)
    starts=np.empty(n+1,np.int64); starts[0]=0; np.cumsum(degree,out=starts[1:])
    rows=np.concatenate((owner,neighbour)); cols=np.concatenate((neighbour,owner))
    order=np.argsort(rows,kind="stable")
    adj=cols[order].astype(np.int32,copy=False)
    return starts,adj


def unique_edge_pairs(owner, neighbour):
    order=np.lexsort((neighbour,owner))
    so=owner[order]; sn=neighbour[order]
    keep=np.ones(len(order),dtype=bool)
    if len(order)>1: keep[1:]=(so[1:]!=so[:-1])|(sn[1:]!=sn[:-1])
    return so[keep].astype(np.int32,copy=False),sn[keep].astype(np.int32,copy=False)


def owner_summary(deg):
    d = np.asarray(deg)
    out = {"cells": int(d.size), "mean": float(d.mean()), "median": float(np.median(d)),
           "std": float(d.std()), "max": int(d.max()), "p95": float(np.percentile(d,95)),
           "p99": float(np.percentile(d,99)), "histogram": histogram(d)}
    bins = ((0,0),(1,1),(2,2),(3,3),(4,4),(5,8),(9,12),(13,2**31-1))
    for lo,hi in bins:
        name = str(lo) if lo == hi else ("gt12" if lo == 13 else f"{lo}-{hi}")
        count = int(np.count_nonzero((d>=lo)&(d<=hi)))
        out[f"count_{name}"] = count; out[f"percent_{name}"] = 100*count/d.size
    out["cells_more_than_one_chunk_w4"] = int(np.count_nonzero(d > 4))
    out["percent_more_than_one_chunk_w4"] = 100*out["cells_more_than_one_chunk_w4"]/d.size
    return out


def chunk_model(deg, width):
    chunks = (deg.astype(np.int64) + width - 1) // width
    chunks[deg == 0] = 0
    n = int(chunks.sum()); slots = n*width; active = int(deg.sum())
    full = int(np.sum(deg // width)); partial = n-full
    active_hist = np.zeros(width+1, np.int64)
    active_hist[width] = int(np.sum(deg // width, dtype=np.int64))
    rem = deg % width
    if np.any(rem):
        active_hist += np.bincount(rem[rem > 0], minlength=width+1)
    return {"width": width, "chunks": n, "fully_occupied_chunks": full,
            "partially_occupied_chunks": partial, "total_slots": slots,
            "active_slots": active, "inactive_slots": slots-active,
            "utilization": active/slots if slots else 0,
            "average_chunks_per_cell": n/len(deg),
            "active_elements_per_chunk_histogram": {str(i):int(v) for i,v in enumerate(active_hist) if v}}


@njit(cache=True)
def _make_execution_chunks(starts, upper, width, terminal_row, nfaces):
    nrows=starts.size-1
    count=0; slots=0
    for row in range(nrows):
        b,e=starts[row],starts[row+1]
        if e>b:
            w=width if row<terminal_row else 4
            chunks=(e-b+w-1)//w
            count+=chunks; slots+=chunks*w
    offsets=np.empty(count+1,np.uint64); active_offsets=np.empty(count+1,np.uint64)
    row_chunks=np.empty(nrows+1,np.int64)
    active=np.empty(count,np.uint8)
    owners=np.empty(count,np.int32); widths=np.empty(count,np.uint8)
    stream=np.empty(slots,np.uint64)
    active_stream=np.empty(nfaces,np.uint64)
    offsets[0]=0; active_offsets[0]=0; ci=0; pos=0; apos=0
    for row in range(nrows):
        row_chunks[row]=ci
        b,e=starts[row],starts[row+1]
        if e<=b: continue
        w=width if row<terminal_row else 4
        face=b
        while face<e:
            n=min(w,e-face)
            if row<terminal_row and face+w>nfaces:
                raise ValueError("main gather chunk would overread the physical LDU face arrays")
            for lane in range(w):
                if row<terminal_row:
                    # Main path unconditionally issues scalar indirect loads.
                    stream[pos]=upper[face+lane]
                elif lane<n:
                    stream[pos]=upper[face+lane]
                else:
                    # Terminal masked index lanes are zero; psi[0] is read.
                    stream[pos]=0
                pos+=1
            for lane in range(n):
                active_stream[apos]=upper[face+lane]; apos+=1
            active[ci]=n; owners[ci]=row; widths[ci]=w
            ci+=1; offsets[ci]=pos; active_offsets[ci]=apos; face+=w
    row_chunks[nrows]=ci
    return offsets,stream,active_offsets,active_stream,active,owners,widths,row_chunks


@njit(cache=True)
def stream_transition_stats(offsets, stream, owners, row_chunks, n_cells):
    total=len(stream); prev_chunk_hits=0; prev_cell_hits=0; own_line_hits=0
    cold_chunk=0; cold_cell=0
    for c in range(len(owners)):
        lo,hi=offsets[c],offsets[c+1]
        row=owners[c]
        p0,p1=(offsets[c-1],offsets[c]) if c else (0,0)
        r0,r1=(row_chunks[row-1],row_chunks[row]) if row>0 else (0,0)
        for p in range(lo,hi):
            line=stream[p]//8; found=False
            for q in range(p0,p1):
                if stream[q]//8==line: found=True; break
            if found: prev_chunk_hits+=1
            else: cold_chunk+=1
            found=False
            for cc in range(r0,r1):
                for q in range(offsets[cc],offsets[cc+1]):
                    if stream[q]//8==line: found=True; break
                if found: break
            if found: prev_cell_hits+=1
            else: cold_cell+=1
            if line==(row//8): own_line_hits+=1
    return (100*prev_chunk_hits/max(1,total),100*prev_cell_hits/max(1,total),
            100*own_line_hits/max(1,total),cold_chunk,cold_cell,total)


def make_execution_chunks(starts, upper, width, nfaces, nrows, kernel):
    # Actual kernel routing. The final physical suffix uses the existing safe
    # M6 terminal helper (four lanes), even for the 8-wide kernels.
    if width == 4:
        terminal_row = int(np.searchsorted(starts[1:], max(0,nfaces-3), side="right"))
        terminal_row = min(nrows, terminal_row)
        cutoff = terminal_row
    else:
        terminal_row = int(np.searchsorted(starts[1:], max(0,nfaces-7), side="right"))
        terminal_row = min(nrows, terminal_row)
        cutoff = terminal_row
    offsets,stream,active_offsets,active_stream,active,owners,widths,row_chunks=_make_execution_chunks(starts,upper,width,terminal_row,nfaces)
    return offsets,stream,active_offsets,active_stream,active,owners,widths,row_chunks,terminal_row


def backend_metrics(offsets, indices, outdir, name, windows, backend, reuse=True,
                    one_window_stride=None):
    # working_set.trace_metrics accepts uint indices and emits exact stack
    # distances. Windows are measured in chunks, not rows/cells.
    data={}; hist=None
    with tempfile.TemporaryDirectory(prefix="gs-numbering-locality-") as td:
        path=Path(td)/"trace"; prefix=Path(td)/"result"
        with path.open("wb") as stream:
            np.asarray(offsets,dtype=np.uint64).tofile(stream)
            np.asarray(indices,dtype=np.uint64).tofile(stream)
        # One process and one trace write per stream. Summarize each large
        # window file immediately, avoiding simultaneous multi-GB arrays.
        options=[]
        for k in windows:
            stride=max(1,k//4)
            options.append(f"{k}:{stride}")
        if not options:
            options=[f"1:{one_window_stride or max(1,len(offsets)-1)}"]
        cmd=[str(Path(backend).resolve()),str(path),str(prefix),str(len(offsets)-1),
             str(len(indices)),"8",str(int(reuse)),*options]
        run=subprocess.run(cmd,capture_output=True,text=True)
        if run.returncode: raise RuntimeError(run.stderr.strip() or "working set backend failed")
        for k in windows:
            f=Path(f"{prefix}-{k}")
            a=np.fromfile(f,dtype=np.float64).reshape(-1,8)
            data[str(k)]={"windows":int(len(a)),"unique_lines":quantiles(a[:,0]),
                          "unique_pages":quantiles(a[:,1]),"span_elements":quantiles(a[:,2]),
                          "span_bytes":quantiles(a[:,3]),"line_density":quantiles(a[:,4]),
                          "line_touches":quantiles(a[:,5]),"reuse_factor":quantiles(a[:,7])}
            del a
        if reuse:
            hist=np.fromfile(f"{prefix}-reuse",dtype=np.uint64)
    rep=summarize_reuse(hist,(32,64,256,1024,4096))
    return data,rep


def load_timing_files():
    files=[ROOT/"experiments/M6.4d/amdr16-two-round-20261126/timing_summary.json",
           ROOT/"experiments/avx512-perf-amdR16/other-meshes/timing/timing_summary.json",
           ROOT/"experiments/avx512-perf-amdR16/ahmedbody/timing/timing_summary.json",
           ROOT/"experiments/avx512-perf-amdR16/timing-corrected/timing_summary.json",
           ROOT/"experiments/avx512-perf-amdR16/m8wide/results/initial/timing_summary.json"]
    return [(str(f.relative_to(ROOT)),json.loads(f.read_text())) for f in files if f.exists()]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir",type=Path,default=ROOT/"experiments/gs-specific-numbering-analysis-20261128")
    ap.add_argument("--backend",type=Path,default=ROOT/"experiments/build/working_set_backend")
    ap.add_argument("--skip-exact-reuse",action="store_true")
    ap.add_argument("--meshes",nargs="*",choices=tuple(ORDERINGS),default=list(ORDERINGS))
    ap.add_argument("--numberings",nargs="*",default=None)
    args=ap.parse_args(); out=args.output_dir
    if out.exists(): raise SystemExit(f"output exists: {out}")
    out.mkdir(parents=True)
    if not args.backend.exists():
        subprocess=None
        import subprocess as sp
        sp.run(["bash",str(ROOT/"tools/build_working_set.sh"),str(args.backend)],check=True)
    started=time.time(); cases=[]; all_details=[]
    chunk_rows=[]; degree_rows=[]; window_rows=[]; reuse_rows=[]; scatter_rows=[]; swap_rows=[]
    selected={m:tuple(n for n in ORDERINGS[m] if args.numberings is None or n in args.numberings)
              for m in args.meshes}
    for mesh_name,numbers in selected.items():
        base=Path(SOURCE[mesh_name]); base=base if base.is_absolute() else (ROOT/base).resolve()
        src=read_poly_mesh(base)
        for numbering in numbers:
            print(f"analyzing {mesh_name}/{numbering}",flush=True)
            owner,neigh,n,topology=build_layout(src,mesh_name,numbering)
            od,nd,starts,upper=owner_arrays(n,owner,neigh); nf=len(neigh)
            structural, rowmetrics = graph_metrics(n, owner, neigh)
            if not np.array_equal(np.diff(starts),od): raise ValueError("owner row offsets mismatch")
            degree_rows.extend([{"mesh":mesh_name,"numbering":numbering,"side":"owner",**owner_summary(od)},
                                {"mesh":mesh_name,"numbering":numbering,"side":"neighbour",**owner_summary(nd)}])
            models={w:chunk_model(od,w) for w in (2,4,8,16)}
            n4,n8=models[4]["chunks"],models[8]["chunks"]
            case={"mesh":mesh_name,"numbering":numbering,"cells":n,"internal_faces":nf,
                  "topology_reference":topology,"owner_degree":owner_summary(od),"neighbour_degree":owner_summary(nd),
                  "graph_structure":structural,"row_neighbour_metrics":rowmetrics,
                  "chunk_models":models,"n4_n8_ratio":n4/n8 if n8 else 0,
                  "break_even_m8_42":n4/n8>42/24 if n8 else False,
                  "break_even_m8_43":n4/n8>43/24 if n8 else False,
                  "lower_bounds":{},"gs_actual_execution":{}}
            for w,ip in ((4,24),(8,42)):
                m=models[w]; lb=math.ceil(nf/w)
                case["lower_bounds"][str(w)]={"minimum_chunks":lb,"actual_chunks":m["chunks"],
                  "relative_excess":m["chunks"]/lb-1 if lb else 0,
                  "max_chunk_instruction_reduction":(m["chunks"]-lb)*ip}
                case[f"gather_estimate_w{w}"]={"total_chunk_instructions":m["chunks"]*ip,
                  "instructions_per_cell":m["chunks"]*ip/n,
                  "instructions_per_internal_face":m["chunks"]*ip/nf,
                  "inactive_slot_equivalent_instructions":m["inactive_slots"]*ip/w}
            for w,ip in ((4,24),(8,42),(8,43)):
                if w==4: continue
                case[f"gather_estimate_w8_{ip}"]={"total_chunk_instructions":models[8]["chunks"]*ip,
                  "instructions_per_cell":models[8]["chunks"]*ip/n,
                  "instructions_per_internal_face":models[8]["chunks"]*ip/nf,
                  "inactive_slot_equivalent_instructions":models[8]["inactive_slots"]*ip/w}
            for w in (4,8):
                offsets,stream,active_offsets,active_stream,active,owners,widths,row_chunks,terminal=make_execution_chunks(starts,upper,w,nf,n,"M6" if w==4 else "M8")
                # Chunk-local stats on active reads; the full stream includes
                # architecturally issued inactive accesses for global metrics.
                loc=chunk_locality(starts,upper,w,terminal)
                lines,p4,p2,md,pair,consec,hist,same,nextpairs,one,alignmeans=loc
                chunk_count=int(len(active))
                exe_chunks={"chunks":chunk_count,"terminal_row":terminal,"executed_psi_reads":int(len(stream)),
                            "inactive_executed_psi_reads":int(len(stream)-nf),
                            "active_chunk_lines":quantiles(lines),"active_chunk_pages_4k":quantiles(p4),
                            "active_chunk_pages_2m":quantiles(p2),"max_neighbor_distance":quantiles(md),
                            "mean_pairwise_distance":quantiles(pair),"consecutive_distance":quantiles(consec),
                            "same_line_consecutive_percent":100*same/max(1,len(consec)),
                            "same_line_xmm_pair_percent":100*nextpairs/max(1,sum(int(x)//2 for x in active)),
                            "full_chunks_one_line_percent":100*one/max(1,int(np.sum(active==w))),
                            "active_unique_line_count_histogram":histogram(lines)}
                exe_chunks["mean_unique_lines_by_psi_base_offset_0_to_56_bytes"]={str(o*8):float(v) for o,v in enumerate(alignmeans)}
                arch_trans=stream_transition_stats(offsets,stream,owners,row_chunks,n)
                active_trans=stream_transition_stats(active_offsets,active_stream,owners,row_chunks,n)
                exe_chunks["architectural_stream_transitions"]={"reuse_from_previous_chunk_percent":arch_trans[0],
                    "reuse_from_previous_processed_cell_percent":arch_trans[1],
                    "accesses_to_current_owner_cell_line_percent":arch_trans[2],
                    "cold_vs_previous_chunk":arch_trans[3],"cold_vs_previous_cell":arch_trans[4],"accesses":arch_trans[5]}
                exe_chunks["active_stream_transitions"]={"reuse_from_previous_chunk_percent":active_trans[0],
                    "reuse_from_previous_processed_cell_percent":active_trans[1],
                    "accesses_to_current_owner_cell_line_percent":active_trans[2],
                    "cold_vs_previous_chunk":active_trans[3],"cold_vs_previous_cell":active_trans[4],"accesses":active_trans[5]}
                case["gs_actual_execution"][str(w)]=exe_chunks
                # Detailed chunk rows indexed by metric for auditable result.
                chunk_rows.append({"mesh":mesh_name,"numbering":numbering,"width":w,
                    "active_unique_lines":json.dumps(quantiles(lines)),"active_unique_4k_pages":json.dumps(quantiles(p4)),
                    "active_unique_2m_pages":json.dumps(quantiles(p2)),"max_distance":json.dumps(quantiles(md)),
                    "pairwise_distance":json.dumps(quantiles(pair)),"consecutive_distance":json.dumps(quantiles(consec)),
                    "same_line_consecutive_percent":exe_chunks["same_line_consecutive_percent"],
                    "same_line_xmm_pair_percent":exe_chunks["same_line_xmm_pair_percent"],
                    "one_line_full_chunk_percent":exe_chunks["full_chunks_one_line_percent"],
                    "active_line_histogram":json.dumps(histogram(lines))})
                if not args.skip_exact_reuse:
                    ws,rep=backend_metrics(offsets,stream,out, f"{mesh_name}-{numbering}-w{w}", LINE_WINDOWS,args.backend,one_window_stride=max(1,len(active)))
                    active_ws,active_rep=backend_metrics(active_offsets,active_stream,out,
                        f"{mesh_name}-{numbering}-w{w}-active",LINE_WINDOWS,args.backend,
                        one_window_stride=max(1,len(active)))
                    case["gs_actual_execution"][str(w)]["working_set_chunks"]=ws
                    case["gs_actual_execution"][str(w)]["active_working_set_chunks"]=active_ws
                    case["gs_actual_execution"][str(w)]["active_exact_line_reuse_distance"]=active_rep
                    case["gs_actual_execution"][str(w)]["exact_line_reuse_distance"]=rep
                    pageids=(stream//512)*8
                    _,page_rep=backend_metrics(offsets,pageids,out,f"{mesh_name}-{numbering}-w{w}-pages",(),args.backend,
                                               one_window_stride=max(1,len(offsets)-1))
                    case["gs_actual_execution"][str(w)]["exact_page_reuse_distance"]={
                        k:v for k,v in page_rep.items() if k in ("cold_accesses","total_accesses","repeat_distance_quantiles","definition")}
                    reuse_rows.append({"mesh":mesh_name,"numbering":numbering,"width":w,"scope":"psi_line",
                                       "cold":rep["cold_accesses"],"accesses":rep["total_accesses"],
                                       "p50":rep["repeat_distance_quantiles"]["p50"],
                                       "p90":rep["repeat_distance_quantiles"]["p90"],
                                       "p99":rep["repeat_distance_quantiles"]["p99"]})
                    reuse_rows.append({"mesh":mesh_name,"numbering":numbering,"width":w,"scope":"psi_page",
                                       "cold":page_rep["cold_accesses"],"accesses":page_rep["total_accesses"],
                                       "p50":page_rep["repeat_distance_quantiles"]["p50"],
                                       "p90":page_rep["repeat_distance_quantiles"]["p90"],
                                       "p99":page_rep["repeat_distance_quantiles"]["p99"]})
            # Scatter destination addresses in exact face order, row order.
            # Owner-grouped LDU means this is precisely upperAddr[0:F].
            if not args.skip_exact_reuse:
                scatter_offsets=starts.astype(np.uint64)
                sdata,sreuse=backend_metrics(scatter_offsets,neigh.astype(np.uint64),out,
                    f"{mesh_name}-{numbering}-scatter",LINE_WINDOWS,args.backend)
                case["scatter"]={"unique_line_windows_in_face_rows":sdata,
                    "exact_line_reuse_distance":sreuse,
                    "destination_distance_consecutive":quantiles(np.abs(np.diff(neigh.astype(np.int64)))),
                    "same_line_consecutive_percent":100*np.count_nonzero(neigh[:-1]//8==neigh[1:]//8)/max(1,nf-1)}
                scatter_rows.append({"mesh":mesh_name,"numbering":numbering,
                    "destination_distance":json.dumps(case["scatter"]["destination_distance_consecutive"]),
                    "same_line_consecutive_percent":case["scatter"]["same_line_consecutive_percent"],
                    "reuse_p90":sreuse["repeat_distance_quantiles"]["p90"],
                    "reuse_p99":sreuse["repeat_distance_quantiles"]["p99"],
                    "window32_lines":sdata["32"]["unique_lines"]["mean"] if "32" in sdata else None})
            # Timing match from the archived two-round M6.4d campaign.
            case["timings"]={}
            for file,td in load_timing_files():
                key=f"{mesh_name}/{numbering}"
                if key in td:
                    case["timings"][file]=td[key]
                elif mesh_name=="MotorBike" and numbering in ("C","D") and numbering in td:
                    case["timings"][file]=td[numbering]
            cases.append(case)
            all_details.append(case)
            # Exact single-edge swap deltas are local: only owner status of
            # faces incident to either swapped cell can change.
            csr_starts,csr_adj=adjacency_csr(n,owner,neigh)
            swap_owner,swap_neigh=unique_edge_pairs(owner,neigh)
            si,sn,sh,shist=adjacent_swap_deltas(n,swap_owner,swap_neigh,od,csr_starts,csr_adj,int(od.max()))
            swap_rows.append({"mesh":mesh_name,"numbering":numbering,"candidate_connected_swaps":len(swap_owner),
                "reducing":int(si),"neutral":int(sn),"increasing":int(sh),
                "delta_histogram":json.dumps({str(i-2*int(od.max())):int(v) for i,v in enumerate(shist) if v})})
            case["adjacent_swap_w4"]={"candidate_edges":len(swap_owner),"reducing":int(si),"neutral":int(sn),"increasing":int(sh),
                "delta_histogram":{str(i-2*int(od.max())):int(v) for i,v in enumerate(shist) if v}}
    (out/"analysis.json").write_text(json.dumps({"analysis":"GS numbering diagnostics","created":"2026-11-28",
      "machine":"amdR16 archived M6.4d timings only; no new benchmarks executed",
      "definitions":{"owner_degree":"number of internal LDU faces stored in each owner row",
        "neighbour_degree":"number of internal LDU faces whose upperAddr equals the cell",
        "cache_line":"psi is FP64; 64-byte aligned primary base; line=floor(cell/8)",
        "page":"4 KiB page=floor(cell/512); 2 MiB page=floor(cell/262144)",
        "actual_chunk_execution":"main M6 width4 or M8 width8; terminal suffix uses safe M6 width4 helper; inactive terminal lanes use psi[0]",
        "estimated_chunk_instructions":"chunk instruction body only; outer loops/scatter excluded; inactive-slot equivalent is proportional allocation, not retired instruction count"},
      "cases":cases,"analysis_seconds":time.time()-started},indent=2))
    tracked=[ROOT/"tools/analyze_gs_numberings.py",ROOT/"tools/summarize_gs_numberings.py",
             ROOT/"tools/working_set_backend.C",ROOT/"tools/gs_m74d.H",ROOT/"tools/gs_minimal_opt.H",ROOT/"tools/gs_m8wide.H"]
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    (out/"config.json").write_text(json.dumps({"created":"2026-11-28","cases":len(cases),
      "sources":{k:str(v) for k,v in SOURCE.items()},"existing_meshes":{k:str(v) for k,v in EXISTING.items()},
      "numberings":selected,"per_case_experiment_root":str(EXP),
      "timing_campaigns":["experiments/M6.4d/amdr16-two-round-20261126/timing_summary.json",
        "experiments/avx512-perf-amdR16/other-meshes/timing/timing_summary.json",
        "experiments/avx512-perf-amdR16/ahmedbody/timing/timing_summary.json",
        "experiments/avx512-perf-amdR16/timing-corrected/timing_summary.json",
        "experiments/avx512-perf-amdR16/m8wide/results/initial/timing_summary.json"],
      "address_assumptions":{"psi_element_bytes":8,"cache_line_bytes":64,"base_alignment_bytes":64,
        "page_bytes":[4096,2097152],"alignment_sensitivity_sample_stride_chunks":1024},
      "chunk_windows":list(LINE_WINDOWS),"window_stride":"max(1,floor(K/4))",
      "no_benchmark_runs":True,"source_hashes":{str(p.relative_to(ROOT)):sha(p) for p in tracked}},indent=2))
    with (out/"summary.csv").open("w",newline="") as f:
        cols=["mesh","numbering","cells","internal_faces","chunks_w4","util_w4","chunks_w8","util_w8","n4_n8","threshold_42","instructions_face_w4","instructions_face_w8_42","actual_lines_chunk_w4","same_line_xmm_pairs_w4_percent","M6_4d_ms"]
        wr=csv.DictWriter(f,fieldnames=cols); wr.writeheader()
        for c in cases:
            timing=c["timings"].get("experiments/M6.4d/amdr16-two-round-20261126/timing_summary.json",{}).get("M6.4d",{})
            wr.writerow({"mesh":c["mesh"],"numbering":c["numbering"],"cells":c["cells"],"internal_faces":c["internal_faces"],
              "chunks_w4":c["chunk_models"][4]["chunks"],"util_w4":c["chunk_models"][4]["utilization"],
              "chunks_w8":c["chunk_models"][8]["chunks"],"util_w8":c["chunk_models"][8]["utilization"],
              "n4_n8":c["n4_n8_ratio"],"threshold_42":c["break_even_m8_42"],
              "instructions_face_w4":c["gather_estimate_w4"]["instructions_per_internal_face"],
              "instructions_face_w8_42":c["gather_estimate_w8_42"]["instructions_per_internal_face"],
              "actual_lines_chunk_w4":c["gs_actual_execution"]["4"]["active_chunk_lines"]["mean"],
              "same_line_xmm_pairs_w4_percent":c["gs_actual_execution"]["4"]["same_line_xmm_pair_percent"],
              "M6_4d_ms":timing.get("median_ms")})
    def dump_csv(name,rows):
        if rows:
            fields=list(dict.fromkeys(k for r in rows for k in r))
            with (out/name).open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    dump_csv("chunk_locality.csv",chunk_rows); dump_csv("degree_statistics.csv",degree_rows)
    dump_csv("reuse_distance.csv",reuse_rows); dump_csv("scatter_locality.csv",scatter_rows)
    dump_csv("adjacent_swap_candidates.csv",swap_rows)
    print(f"completed {len(cases)} cases in {time.time()-started:.1f}s -> {out}")


if __name__ == "__main__":
    main()
