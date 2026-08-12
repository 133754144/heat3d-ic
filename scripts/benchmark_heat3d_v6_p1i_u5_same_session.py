#!/usr/bin/env python3
"""Run the three U5 routes sequentially in one persistent Python/JAX session."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

import run_heat3d_v6_p1i_p5r_resolution_cell as p5r
import benchmark_heat3d_v6_p1i_u2_asymmetric_runtime as u2

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()

def invoke(function, arguments: list[str]) -> None:
    saved=sys.argv; sys.argv=[saved[0],*arguments]
    try: function()
    finally: sys.argv=saved

def main() -> int:
    p=argparse.ArgumentParser()
    for name in ("protocol","p5r_protocol","binding","artifact_root","dataset_root","manifest","full_fields","run_dir","native_padding","e16384_padding","e240825_padding","u240825_padding","output_dir","output"):
        p.add_argument(f"--{name.replace('_','-')}",dest=name,type=Path,required=True)
    p.add_argument("--checkpoint-sha256",required=True); a=p.parse_args(); q=json.loads(a.protocol.read_text())
    if q["status"]!="preregistered_before_execution" or not q["same_session"]: raise RuntimeError("U5 protocol")
    a.output_dir.mkdir(parents=True,exist_ok=True); route_files={}; started=time.perf_counter()
    common=["--binding",str(a.binding),"--artifact-root",str(a.artifact_root),"--dataset-root",str(a.dataset_root),"--manifest",str(a.manifest),"--full-fields",str(a.full_fields),"--run-dir",str(a.run_dir),"--checkpoint-sha256",a.checkpoint_sha256,"--sample-count","32"]
    for route,padding in (("E16384_reconstruction",a.e16384_padding),("E240825_direct",a.e240825_padding)):
        output=a.output_dir/f"{route}.json"; invoke(p5r.main,["--protocol",str(a.p5r_protocol),*common,"--padding-result",str(padding),"--native-padding-result",str(a.native_padding),"--route",route,"--output",str(output)]); route_files[route]=output
    output=a.output_dir/"U_direct240825.json"; invoke(u2.main,["--protocol",str(a.protocol),*common,"--native-padding-result",str(a.native_padding),"--query-padding-result",str(a.u240825_padding),"--resolution","240825","--repeats",str(q["repeats"]),"--batch-sizes","1","--output",str(output)]); route_files["U_direct240825"]=output
    result={"schema_version":"heat3d_v6_p1i_u5_same_session_v1","status":"passed","same_session":True,"wall_seconds":time.perf_counter()-started,"routes":{},"lean_output_query":{},"protocol_sha256":sha(a.protocol),"execution_commit":__import__("subprocess").check_output(["git","rev-parse","HEAD"],text=True).strip(),"role_contract":q["role_contract"]}
    for route,path in route_files.items():
        payload=json.loads(path.read_text()); result["routes"][route]={"path":str(path),"sha256":sha(path),"accuracy":payload["accuracy"]["full_field"],"timing":payload["runtime"]["fresh_sample"] if route.startswith("U_") else payload["timing"],"peak_vram_bytes":payload.get("memory",{}).get("peak_bytes_in_use",payload.get("peak_vram_bytes"))}
    u=json.loads(route_files["U_direct240825"].read_text()); result["lean_output_query"]={"mode":u["packing_optimization"]["mode"],"full_output_group_never_constructed_in_production_path":u["packing_optimization"]["full_output_group_never_constructed_in_production_path"],"cpu_prediction_bitwise_exact":u["packing_optimization"]["prediction_bitwise_exact_vs_U3"]}
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps({"status":"passed","wall_seconds":result["wall_seconds"]})); return 0
if __name__=="__main__": raise SystemExit(main())
