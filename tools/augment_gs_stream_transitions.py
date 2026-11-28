#!/usr/bin/env python3
"""Add previous-chunk/cell reuse percentages to an existing GS analysis."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from analyze_gs_numberings import (EXP, ORDERINGS, SOURCE, EXISTING, build_layout,
    owner_arrays, make_execution_chunks, stream_transition_stats, backend_metrics, LINE_WINDOWS)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("analysis_dir",type=Path);a=ap.parse_args()
    p=a.analysis_dir/"analysis.json";doc=json.loads(p.read_text())
    by={(c["mesh"],c["numbering"]):c for c in doc["cases"]}
    for mesh,numbers in ORDERINGS.items():
        base=Path(SOURCE[mesh]);base=base if base.is_absolute() else (ROOT/base).resolve()
        source=__import__("openfoam_mesh_reader").read_poly_mesh(base)
        for numbering in numbers:
            print(f"transition metrics {mesh}/{numbering}",flush=True)
            owner,neigh,n,_=build_layout(source,mesh,numbering)
            _,_,starts,upper=owner_arrays(n,owner,neigh);nf=len(neigh)
            case=by[(mesh,numbering)]
            for width in (4,8):
                offsets,stream,aoffs,astream,active,owners,widths,row_chunks,terminal=make_execution_chunks(
                    starts,upper,width,nf,n,"M6" if width==4 else "M8")
                arch=stream_transition_stats(offsets,stream,owners,row_chunks,n)
                logical=stream_transition_stats(aoffs,astream,owners,row_chunks,n)
                same_arch=100*np.count_nonzero(stream[:-1]//8==stream[1:]//8)/max(1,len(stream)-1)
                same_active=100*np.count_nonzero(astream[:-1]//8==astream[1:]//8)/max(1,len(astream)-1)
                rec=case["gs_actual_execution"][str(width)]
                rec["architectural_stream_transitions"]={"reuse_from_previous_chunk_percent":arch[0],
                    "reuse_from_previous_processed_cell_percent":arch[1],"accesses_to_current_owner_cell_line_percent":arch[2],
                    "cold_vs_previous_chunk":arch[3],"cold_vs_previous_cell":arch[4],"accesses":arch[5],
                    "consecutive_same_line_percent":float(same_arch)}
                rec["active_stream_transitions"]={"reuse_from_previous_chunk_percent":logical[0],
                    "reuse_from_previous_processed_cell_percent":logical[1],"accesses_to_current_owner_cell_line_percent":logical[2],
                    "cold_vs_previous_chunk":logical[3],"cold_vs_previous_cell":logical[4],"accesses":logical[5],
                    "consecutive_same_line_percent":float(same_active)}
                if width==4:
                    active_pages=(astream//512)*8
                    _,active_page_reuse=backend_metrics(aoffs,active_pages,a.analysis_dir,
                        f"{mesh}-{numbering}-active-pages",(),
                        ROOT/"experiments/build/working_set_backend",True,
                        one_window_stride=max(1,len(active)))
                    for w in (4,8):
                        case["gs_actual_execution"][str(w)]["active_exact_page_reuse_distance"]={
                            k:v for k,v in active_page_reuse.items() if k in ("cold_accesses","total_accesses","repeat_distance_quantiles","definition")}
                del offsets,stream,aoffs,astream,active,owners,widths,row_chunks
            del owner,neigh,starts,upper
    p.write_text(json.dumps(doc,indent=2))
    rows=[]
    for c in doc["cases"]:
        for width in (4,8):
            for scope,key in (("active","active_stream_transitions"),("architectural","architectural_stream_transitions")):
                x=c["gs_actual_execution"][str(width)][key]
                rows.append({"mesh":c["mesh"],"numbering":c["numbering"],"width":width,"scope":scope,
                    "accesses":x["accesses"],"same_line_consecutive_percent":x["consecutive_same_line_percent"],
                    "reuse_previous_chunk_percent":x["reuse_from_previous_chunk_percent"],
                    "reuse_previous_cell_percent":x["reuse_from_previous_processed_cell_percent"],
                    "current_owner_line_percent":x["accesses_to_current_owner_cell_line_percent"],
                    "cold_vs_previous_chunk":x["cold_vs_previous_chunk"],"cold_vs_previous_cell":x["cold_vs_previous_cell"]})
    with (a.analysis_dir/"stream_transitions.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print("updated",p)


if __name__=="__main__":main()
