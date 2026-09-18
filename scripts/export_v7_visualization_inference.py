#!/usr/bin/env python3
"""Inference-only, seed-0 validation exports for six V7 diagnostic panels.

Run with the registered model-specific environment. Never trains or selects a
checkpoint. The first manifest validation case is fixed before prediction.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(os.environ.get('HEAT3D_REPO', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=['Full','vanilla_RIGNO','GINO','Transolver','Heat3D-DeepOHeat','DeepOHeat'], required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--p1i-root', type=Path)
    ap.add_argument('--upstream', type=Path)
    ap.add_argument('--labels-root', type=Path)
    ap.add_argument('--fs-train', type=Path)
    a = ap.parse_args()
    if a.output.exists() or a.output.with_suffix('.json').exists(): raise FileExistsError(a.output)
    start = time.time()
    def guard(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            name = os.fsdecode(args[0]).lower()
            if any(t in name for t in ['fs_test_volume', 'u_test_volume', '/test_iid/', '/sealed/']):
                raise PermissionError(name)
    sys.addaudithook(guard)
    meta = dict(model=a.model, seed=0, checkpoint=str(a.checkpoint.resolve()), checkpoint_sha256=sha(a.checkpoint),
                selection='existing checkpoint; no reselection', sample_selection='first validation row in frozen manifest, before inference',
                code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                script_sha256=sha(__file__), command=sys.argv, training=False, test_or_sealed_access=False)
    config_path = ROOT/'configs/heat3d_v7/v7_g1_full_p1i.json'
    config = json.loads(config_path.read_text())
    if a.model in ['Full','vanilla_RIGNO','GINO','Transolver']:
        manifest = ROOT/'configs/heat3d_v6_p1i/v6_p1i_formal1024_v1_manifest.json'
        rows = [r for r in json.loads(manifest.read_text())['samples'] if r['split_role']=='valid_iid']
        sid = rows[0]['sample_id']; sample = a.p1i_root/'samples'/sid
        coords=np.load(sample/'coords.npy'); truth=np.load(sample/'deltaT.npy').reshape(-1)
        layer=np.load(sample/'layer_id.npy').reshape(-1)
        meta.update(dataset='P1i', split='valid_iid', sample_id=sid, manifest_sha256=sha(manifest),
                    sample_files={p.name:sha(p) for p in sample.iterdir() if p.is_file()}, inference_nodes=1024)
        if a.model in ['GINO','Transolver']:
            import torch
            from scripts.run_v7_g2_p1i_external_formal import P1iRoleDataset, load_stats, normalize, predict, build_gino, build_transolver, latent_queries
            sys.path.insert(0,str(a.upstream))
            device=torch.device('cuda'); stats=load_stats(ROOT/'docs/v7_g2_p3_p1i_train_statistics.json')
            row=P1iRoleDataset(a.p1i_root,manifest,'valid_iid')[0]
            model=(build_gino(.15,.033,use_open3d=True,use_torch_scatter=True) if a.model=='GINO' else build_transolver(a.upstream)).to(device)
            checkpoint=torch.load(a.checkpoint,map_location=device,weights_only=False)
            model.load_state_dict(checkpoint['model']); model.eval()
            c,f,_,_,local=normalize(row,stats,device)
            with torch.no_grad(): pred=(predict(a.model,model,c,f,latent_queries(32).to(device) if a.model=='GINO' else None)*local['target_std']+local['target_mean']).cpu().numpy().reshape(-1)
            meta.update(epoch=checkpoint.get('epoch'), hardware=torch.cuda.get_device_name(), normalization_sha256=sha(ROOT/'docs/v7_g2_p3_p1i_train_statistics.json'))
        else:
            import jax
            from rigno.heat3d_training.p1i import load_selected_p1i_examples, legacy_train_only_stats, COORD_POLICY_TRAIN_MINMAX_UNIT_BOX, build_p1i_batches, attach_input_contexts, attach_native_physics, attach_qk_features, model_apply_full, model_apply_vanilla, prediction_to_raw_delta
            from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
            from rigno.models.rigno import RIGNO
            from scripts.run_heat3d_v7_formal_p1i_training import _variant_model_config, _resolve_model_config
            from scripts.evaluate_v7_g2_p14_1_heat3d_u_v2 import checkpoint_params
            loaded=load_selected_p1i_examples(a.p1i_root,manifest)
            stats=legacy_train_only_stats(loaded['train'],coord_policy=COORD_POLICY_TRAIN_MINMAX_UNIT_BOX)
            valid=loaded['valid_iid'][:32]
            batches=build_p1i_batches(valid,stats,Heat3DGraphBuilder(**config['graph']),label='valid_iid',batch_size=32,graph_seed=0)
            mc=_variant_model_config(config['model'],a.model)
            context=attach_input_contexts(batches,loaded['train'],loaded['train']+valid,mc)
            byid={r.sample_id:r for r in loaded['train']+valid}
            attach_native_physics(batches,byid,context_by_id=context['raw_context_by_id'])
            attach_qk_features(batches,byid,feature_version=config['model']['qk_region_feature_version'])
            model=RIGNO(**_resolve_model_config(mc,tuple(stats['feature_names'])))
            params,cm=checkpoint_params(a.checkpoint)
            apply=model_apply_full if a.model=='Full' else model_apply_vanilla
            result=jax.jit(lambda p:apply(model,p,batches[0],None))(params)
            pred=prediction_to_raw_delta(result[0],variant=a.model,stats=stats)[0,0,:,0]
            assert batches[0].sample_ids[0]==sid
            meta.update(checkpoint_metadata=cm, hardware=str(jax.devices()), graph_seed=0, graph_batch_size=32, config_sha256=sha(config_path))
    else:
        import jax
        import jax.numpy as jnp
        from scripts import load_v7_g2_p6_deepoheat_v1_compact as loader
        from scripts import prepare_v7_g2_p5_deepoheat_v1_support as support
        valid=loader.CompactDeepOHeatV1Dataset(fs_train=a.fs_train,labels_root=a.labels_root,role='valid')
        valid.rows=valid.rows[:1]; row=valid[0]; mesh=support.mesh_arrays()
        coords=mesh['coords']; layer=mesh['layer_id']; truth=row['target_full_571256'].reshape(-1)
        meta.update(dataset='DeepOHeat-v1 volumetric',split='valid',sample_id=row['sample_id'],source_index=int(row['source_index']),inference_nodes=len(coords),
                    label_receipt_sha256=sha(a.labels_root/'label_generation_receipt.json'),fs_train_sha256=sha(a.fs_train),hardware=str(jax.devices()))
        power=np.asarray(valid.fs_train[row['source_index']],dtype=np.float32)
        if a.model=='DeepOHeat':
            import equinox as eqx
            sys.path.insert(0,str(a.upstream)); from models import DeepOHeat_v1
            _,key=jax.random.split(jax.random.PRNGKey(0),2)
            model=DeepOHeat_v1(dim=3,branch_dim=101**2,field_dim=1,branch_depth=8,branch_hidden=256,trunk_depth=3,trunk_hidden=64,rank=128,key=key)
            model=eqx.tree_deserialise_leaves(a.checkpoint,model)
            axes=tuple(jnp.asarray(x.reshape(-1,1),dtype=jnp.float32) for x in [np.linspace(0,1,101),np.linspace(0,1,101),np.linspace(0,.55,56)])
            output=eqx.filter_jit(lambda m:m((axes,jnp.asarray(power.reshape(1,-1)))))(model)
            pred=25*(np.asarray(output,dtype=np.float64).reshape(-1)-.2)
            meta['output_conversion']='DeltaT_K=25*(u-0.2)'
        else:
            from scripts import evaluate_v7_g2_p14_1_heat3d_u_v2 as h
            from dataclasses import replace
            from types import SimpleNamespace
            from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
            from rigno.heat3d_training.p1i import build_p1i_batches,attach_input_contexts
            from rigno.models.rigno import RIGNO
            from rigno.heat3d_runtime.u_split import UHighNRuntime
            from rigno.heat3d_runtime.high_n import FullFieldGeometry
            evaluator=h.load_script('evaluate_v7_g2_p14_heat3d_v1_fullfield.py'); converter=h.load_script('convert_v7_g2_semiconductor_case.py')
            train=loader.CompactDeepOHeatV1Dataset(fs_train=a.fs_train,labels_root=a.labels_root,role='train',verify_source_file=False)
            kwargs=dict(full_coords=coords,full_cv=mesh['control_volume'],layer_id=layer,converter=converter)
            print('Building frozen train-only context',flush=True)
            tr=evaluator.build_examples(dataset=train,role='train',**kwargs)
            va=evaluator.build_examples(dataset=valid,role='valid',**kwargs)
            stats=h.load_stats(a.labels_root/'train_only_normalization.json')
            batches=build_p1i_batches(va,stats,Heat3DGraphBuilder(**config['graph']),label='visual_valid',batch_size=32,graph_seed=0)
            context=attach_input_contexts(batches,tr,tr+va,config['model'])
            model=RIGNO(**h.resolve_model_config(config['model'],tuple(stats['feature_names'])))
            params,cm=h.checkpoint_params(a.checkpoint)
            session=h.build_session(model=model,params=params,stats=stats,config=config,context_standardizer=context['standardizer'],seed=0)
            runtime=UHighNRuntime.from_session(session,FullFieldGeometry(path=Path('<DeepOHeat mesh>'),coords=coords,control_volume=mesh['control_volume'],layer_id=layer,sample_ids=(),split_roles=()))
            runtime.geometry=SimpleNamespace(coords=coords,control_volume=mesh['control_volume'],layer_id=layer)
            anchor=replace(va[0],condition=replace(va[0].condition,coords=coords[row['support_indices']]),meta={**va[0].meta,'top_h_W_m2K':.1/2,'bottom_h_W_m2K':.1/40})
            full=h.full_support_artifact(power=power,converter=converter,mesh=mesh)
            print('Running U-v2 dense query',flush=True)
            case=runtime.build_case(anchor,len(coords),support=full,native_edge_targets=None,query_edge_targets=None)
            pred=np.asarray(runtime.apply(case)['raw_temperature'],dtype=np.float64)[0,0,:,0]-298.15
            meta.update(checkpoint_metadata=cm,route='U-v2 direct-query dense',config_sha256=sha(config_path),normalization_sha256=sha(a.labels_root/'train_only_normalization.json'))
    pred=np.asarray(pred).reshape(-1); truth=np.asarray(truth).reshape(-1)
    assert pred.shape==truth.shape and np.isfinite(pred).all() and np.isfinite(truth).all()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,coords=coords,truth=truth,prediction=pred,layer_id=layer)
    meta.update(output_sha256=sha(a.output), elapsed_seconds=time.time()-start, raw_point_RMSE_K=float(np.sqrt(np.mean((pred-truth)**2))),point_relative_L2_pct=float(100*np.linalg.norm(pred-truth)/np.linalg.norm(truth)))
    a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2,default=str)+'\n')
    print(json.dumps(meta,indent=2,default=str),flush=True)

if __name__=='__main__': main()
