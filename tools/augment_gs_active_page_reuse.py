#!/usr/bin/env python3
"""Calculate exact page stack distance for active GS psi reads only."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from analyze_gs_numberings import (ORDERINGS,SOURCE,build_layout,owner_arrays,backend_metrics)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("analysis_dir",type=Path);a=ap.parse_args()
    p=a.analysis_dir/"analysis.json";doc=json.loads(p.read_text())
    by={(c["mesh"],c["numbering"]):c for c in doc["cases"]}
    for mesh,numbers in ORDERINGS.items():
        base=Path(SOURCE[mesh]);base=base if base.is_absolute() else (ROOT/base).resolve()
        source=__import__("openfoam_mesh_reader").read_poly_mesh(base)
        for numbering in numbers:
            print(f"active page reuse {mesh}/{numbering}",flush=True)
            owner,neigh,n,_=build_layout(source,mesh,numbering)
            _,_,starts,upper=owner_arrays(n,owner,neigh)
            # page_id is encoded as element-index page_id*8, so the backend's
            # 64-byte line mapping gives exactly one synthetic line per page.
            page_ids=(upper.astype("uint64")//512)*8
            _,rep=backend_metrics(starts,page_ids,a.analysis_dir,
                f"{mesh}-{numbering}-active-pages",(),ROOT/"experiments/build/working_set_backend",
                True,one_window_stride=max(1,n))
            slim={k:v for k,v in rep.items() if k in ("cold_accesses","total_accesses","repeat_distance_quantiles","definition")}
            for w in (4,8): by[(mesh,numbering)]["gs_actual_execution"][str(w)]["active_exact_page_reuse_distance"]=slim
    p.write_text(json.dumps(doc,indent=2))
    rows=[]
    for c in doc["cases"]:
        for width in (4,8):
            x=c["gs_actual_execution"][str(width)]
            for scope,key in (("active_line", "active_exact_line_reuse_distance"),
                              ("active_page", "active_exact_page_reuse_distance"),
                              ("executed_line", "exact_line_reuse_distance"),
                              ("executed_page", "exact_page_reuse_distance")):
                if key not in x: continue
                r=x[key]["repeat_distance_quantiles"]
                rows.append({"mesh":c["mesh"],"numbering":c["numbering"],"width":width,"scope":scope,
                    "cold_accesses":x[key]["cold_accesses"],"accesses":x[key]["total_accesses"],
                    "p50":r["p50"],"p90":r["p90"],"p99":r["p99"]})
    import csv
    with (a.analysis_dir/"reuse_distance.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print("updated",p)


if __name__=="__main__":main()
