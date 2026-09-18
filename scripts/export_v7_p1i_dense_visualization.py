#!/usr/bin/env python3
"""Replay the frozen G1 Full direct240825 route and read one valid FVM row."""
import argparse, hashlib, json, os, sys, subprocess
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--subset',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--formal-receipt',type=Path,required=True);p.add_argument('--route-config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');sys.path.insert(0,str(a.repo))
    import jax
    from scripts import closeout_heat3d_v7_g1_h2 as h
    from rigno.heat3d_runtime.high_n import FullFieldGeometry
    from rigno.heat3d_runtime.u_split import UHighNRuntime
    from rigno.heat3d_v6_p1i_anchor_query import HIGH_N_SELECTION_SEED
    frozen=json.loads(a.route_config.read_text());route=frozen['route']
    assert route['route_id']=='U_v2_direct240825'
    assert sha(a.checkpoint)==frozen['source_best_checkpoint_sha256']
    archive_sha=sha(a.archive)
    assert archive_sha=='49023ac1205b8e7cf7c5bf782b89fcdb34997704b3f9aa2fb2d46cf1a59163cb'
    run=dict(run_id='Full_seed0',variant='Full',seed=0)
    manifest=a.repo/'configs/heat3d_v6_p1i/v6_p1i_formal1024_v1_manifest.json'
    train,valid,contexts,_=h._load_examples(repo=a.repo,subset=a.subset,manifest=manifest,full_fields=a.archive,variant='Full',seed=0)
    ex=valid[0];assert ex.sample_id=='v6p1if1_0003'
    geometry=FullFieldGeometry.load(a.archive)
    truth=geometry.valid_truth([ex.sample_id])[ex.sample_id]
    session,_,_,_=h._build_session(repo=a.repo,run=run,raw_receipt=json.loads(a.formal_receipt.read_text()),raw_checkpoint_path=a.checkpoint,source_run_config=h._source_run_config(a.repo,'Full_seed0'),train_examples=train,context_rows_by_id=contexts,route=route)
    runtime=UHighNRuntime.from_session(session,geometry,graph_builder_fingerprint=sha(a.repo/'rigno/graphBuilder_Heat3D.py'))
    fixture=h._prepare_fixture(example=ex,geometry=geometry,resolution=240825,selection_seed=HIGH_N_SELECTION_SEED)
    case=runtime.build_case(ex,240825,support=fixture['support'],native_edge_targets=route['fixed_edge_targets']['native'],query_edge_targets=route['fixed_edge_targets']['query'])
    result=runtime.apply(case);jax.block_until_ready(result['raw_temperature'])
    query=np.asarray(result['raw_temperature'],dtype=float).reshape(-1)-300.
    pred=np.empty(240825);pred[fixture['selected_indices']]=query
    assert np.isfinite(pred).all() and truth.shape==pred.shape
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'P1i_FVM.npz',coords=geometry.coords,truth=truth,layer_id=geometry.layer_id)
    np.savez_compressed(a.output/'Full_dense.npz',coords=geometry.coords,truth=truth,prediction=pred,layer_id=geometry.layer_id)
    receipt=dict(sample_id=ex.sample_id,split='valid_iid',model='Full',seed=0,checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint),checkpoint_metadata={'epoch':156},route=route,route_config_sha256=sha(a.route_config),full_field_archive=str(a.archive),archive_sha256=archive_sha,truth_sha256=sha(a.output/'P1i_FVM.npz'),output_sha256=sha(a.output/'Full_dense.npz'),inference_nodes=240825,point_relative_L2_pct=float(100*np.linalg.norm(pred-truth)/np.linalg.norm(truth)),raw_point_RMSE_K=float(np.sqrt(np.mean((pred-truth)**2))),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip(),script_sha256=sha(__file__),command=sys.argv,training=False,test_sealed_access=False)
    (a.output/'Full_dense.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
