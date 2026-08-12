#!/usr/bin/env python3
"""P7 fresh distinct-case CPU-preprocess to batched-GPU E16384 benchmark."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ROOT=Path(os.environ.get('HEAT3D_REPO_ROOT',Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT,ROOT/'scripts'):
    if str(value) not in sys.path:sys.path.insert(0,str(value))

import benchmark_heat3d_v6_inference_qualification as qualification
import run_heat3d_v6_p1i_anchor_high_n_development as highn
import run_heat3d_v6_p1i_p5r_resolution_cell as p5r
import run_heat3d_v6_p1i_graph_scale_candidate as candidate
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_v6_full_field import build_reconstruction_map,prepare_reconstruction_domain_partition
from rigno.heat3d_v6_p1i_anchor_query import conservative_selected_control_volume,deterministic_nested_query_prefix,prepare_nested_query_geometry_cache
from rigno.models.rigno import RIGNO as GraphNeuralOperator

def sha256(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def block(tree:Any)->None:jax.tree_util.tree_map(lambda x:x.block_until_ready() if hasattr(x,'block_until_ready') else x,tree)
def host_tree(tree:Any)->Any:return jax.tree_util.tree_map(lambda x:np.asarray(jax.device_get(x)),tree)
def stack(trees:list[Any])->Any:return jax.tree_util.tree_map(lambda *xs:np.concatenate([np.asarray(x) for x in xs],axis=0),*trees)

def parse()->argparse.Namespace:
    p=argparse.ArgumentParser()
    for name in ('protocol','binding','artifact_root','dataset_root','manifest','full_fields','run_dir','native_padding_result','query_padding_result','output'):
        p.add_argument(f'--{name.replace("_","-")}',dest=name,type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--sample-count',type=int,choices=[1,2,32],required=True)
    p.add_argument('--batch-sizes',default='1,2,4,8,16,32');return p.parse_args()

def main()->int:
    args=parse();protocol=json.loads(args.protocol.read_text())
    if protocol['status']!='preregistered_before_execution':raise RuntimeError('protocol not frozen')
    if jax.devices()[0].platform!='gpu':raise RuntimeError('P7 requires GPU')
    runtime=p5r._runtime(args);binding=json.loads(args.binding.read_text());dataset=highn._dataset(args)
    all_anchors=highn._valid_examples(dataset,binding)
    if len(all_anchors)!=32:raise RuntimeError('frozen valid32 required')
    preflight=json.loads((args.artifact_root/'actual_data_preflight.json').read_text())
    if preflight['sample_ids']!=[x.sample_id for x in all_anchors]:raise RuntimeError('valid32 order drift')
    anchors=all_anchors[:args.sample_count];physics_rows={r['sample_id']:r for r in preflight['samples']}
    full,_=highn._full_shared(args);coords=np.asarray(full['coords'],dtype=np.float64);cv=np.asarray(full['cv'],dtype=np.float64);layer=np.asarray(full['layer'],dtype=np.int32)
    boundaries=highn._boundaries(anchors[0],float(np.min(coords[:,2])))
    geometry=prepare_nested_query_geometry_cache(full_coords=coords,full_control_volume=cv,full_layer_id=layer,layer_boundaries_m=boundaries)
    partition=prepare_reconstruction_domain_partition(coords=coords,layer_id=layer,boundaries=boundaries)
    graph_key=highn.runner._metadata_key(int(runtime['run_config']['graph_seed']))
    anchor_targets=p5r._edge_targets(args.native_padding_result);query_targets=p5r._edge_targets(args.query_padding_result)
    graph_config=dict(runtime['graph_config']);graph_config.update(subsample_factor=64.0,discrete_graph_backend='sparse_kdtree_v1',reuse_exact_p2r_for_r2p=True)
    anchor_config=dict(runtime['graph_config']);anchor_config.update(subsample_factor=4.0,discrete_graph_backend='sparse_kdtree_v1',reuse_exact_p2r_for_r2p=True)
    model=GraphNeuralOperator(**runtime['model_config']);params=highn.runner._device_params(runtime['checkpoint']['params']);gpu=jax.devices('gpu')[0];cpu=jax.devices('cpu')[0]

    @jax.jit
    def forward(params:Any,anchor:Any,query:Any,weights:Any,indices:Any,map_weights:Any)->Any:
        a=highn.runner._model_apply(model,params,anchor);raw=a['raw_temperature'][:,0,:,0];scale=a['s_hat'].reshape((raw.shape[0],-1))[:,0]
        q=highn.runner._model_apply(model,params,query)['raw_temperature'][:,0,:,0]-highn.REFERENCE_K
        normalized=weights/jnp.sum(weights,axis=1,keepdims=True);qscale=jnp.sqrt(jnp.sum(normalized*q*q,axis=1))
        support=q/qscale[:,None]*scale[:,None]
        gathered=support[jnp.arange(support.shape[0])[:,None,None],indices]
        return jnp.sum(gathered*map_weights.astype(support.dtype),axis=2)

    # Qualification-only full valid32 edge envelope; outside every measured span.
    frozen={r['sample_id']:r for r in preflight['supports']['16384']}
    for anchor in all_anchors:
        with jax.default_device(cpu):
            for targets,builder,example in (
                (anchor_targets,Heat3DGraphBuilder(**anchor_config),anchor),
                (query_targets,Heat3DGraphBuilder(**graph_config),highn._query_example(anchor,highn._load_support(Path(frozen[anchor.sample_id]['support_file'])),coords)),
            ):
                metadata=builder.build_metadata(highn.runner._graph_coords_for_example(example,runtime['stats']),key=graph_key);block(metadata)
                for field in qualification.EDGE_FIELDS:
                    value=getattr(metadata,field)
                    if value is not None:targets[field]=max(int(targets.get(field) or 0),int(np.asarray(value).shape[1]))

    def prepare(anchor:Any)->dict[str,Any]:
        stages={};start=time.perf_counter()
        with np.load(physics_rows[anchor.sample_id]['physics_cache_file'],allow_pickle=False) as physics:
            full_k=np.asarray(physics['k_xyz'],dtype=np.float64);full_q=np.asarray(physics['q_W_m3'],dtype=np.float64)
        anchor_indices,distance=highn._anchor_indices(anchor,coords,float(binding['numeric_tolerances']['anchor_to_solver_coordinate_max_distance_m']))
        if distance!=0.0:raise RuntimeError('anchor drift')
        phase=time.perf_counter();selected,_=deterministic_nested_query_prefix(sample_id=anchor.sample_id,anchor_indices=anchor_indices,full_q=full_q,target_count=16384,geometry_cache=geometry)
        selected_cv,_=conservative_selected_control_volume(full_coords=coords,full_control_volume=cv,full_layer_id=layer,selected_indices=selected,query_workers=1);stages['support_plus_cv']=time.perf_counter()-phase
        anchor_support={'selected_indices':anchor_indices,'operator_control_volume':np.asarray(anchor.operator_point_weights,dtype=np.float64),'k_xyz':np.asarray(anchor.condition.condition_features[:,:3],dtype=np.float64),'q_W_m3':np.asarray(anchor.condition.condition_features[:,3],dtype=np.float64),'layer_id':layer[anchor_indices]}
        query_support={'selected_indices':selected,'operator_control_volume':selected_cv,'k_xyz':full_k[selected],'q_W_m3':full_q[selected],'layer_id':layer[selected]}
        anchor_example=highn._query_example(anchor,anchor_support,coords);query_example=highn._query_example(anchor,query_support,coords)
        ab=Heat3DGraphBuilder(**anchor_config);qb=Heat3DGraphBuilder(**graph_config)
        with jax.default_device(cpu):
            phase=time.perf_counter();am=ab.build_metadata(highn.runner._graph_coords_for_example(anchor_example,runtime['stats']),key=graph_key);block(am);stages['anchor_graph']=time.perf_counter()-phase
            phase=time.perf_counter();qm=qb.build_metadata(highn.runner._graph_coords_for_example(query_example,runtime['stats']),key=graph_key);block(qm);stages['query_graph']=time.perf_counter()-phase
            phase=time.perf_counter();ah=host_tree(highn._model_group(highn._prepare_group(example=anchor_example,anchor=anchor,runtime=runtime,builder=ab,metadata=am,edge_targets=p5r._compatible_targets(anchor_targets,am))));stages['anchor_group_pack']=time.perf_counter()-phase
            phase=time.perf_counter();qh=host_tree(highn._model_group(highn._prepare_group(example=query_example,anchor=anchor,runtime=runtime,builder=qb,metadata=qm,edge_targets=p5r._compatible_targets(query_targets,qm))));stages['query_group_pack']=time.perf_counter()-phase
        phase=time.perf_counter();mapping,_=build_reconstruction_map(coords=coords,layer_id=layer,boundaries=boundaries,support_indices=selected,empty_domain_fallback='same_layer',prepared_partition=partition,query_workers=1);stages['reconstruction_map']=time.perf_counter()-phase
        return {'sample_id':anchor.sample_id,'anchor':ah,'query':qh,'weights':np.asarray(selected_cv,dtype=np.float32)[None,:],
                'indices':np.asarray(mapping.neighbor_local_indices,dtype=np.int32)[None,:,:],'map_weights':np.asarray(mapping.neighbor_weights,dtype=np.float64)[None,:,:],
                'stages':stages,'cpu_preprocessing_total':time.perf_counter()-start}

    # Compile outside timing with one prepared sample.
    warm=prepare(anchors[0]);wd=jax.device_put((warm['anchor'],warm['query'],warm['weights'],warm['indices'],warm['map_weights']),gpu);block(wd);block(forward(params,*wd))
    batch_sizes=[int(x) for x in args.batch_sizes.split(',') if int(x)<=args.sample_count];rows=[];t1=None
    workers=int(protocol['fresh_batch_contract']['cpu_preprocess_workers'])
    for batch_size in batch_sizes:
        selected_anchors=anchors[:batch_size];total_start=time.perf_counter();phase=time.perf_counter()
        if batch_size==1:prepared=[prepare(selected_anchors[0])]
        else:
            with ThreadPoolExecutor(max_workers=min(workers,batch_size)) as executor:prepared=list(executor.map(prepare,selected_anchors))
        cpu_wall=time.perf_counter()-phase
        phase=time.perf_counter();host=(stack([r['anchor'] for r in prepared]),stack([r['query'] for r in prepared]),np.concatenate([r['weights'] for r in prepared]),np.concatenate([r['indices'] for r in prepared]),np.concatenate([r['map_weights'] for r in prepared]));pack_wall=time.perf_counter()-phase
        phase=time.perf_counter();device=jax.device_put(host,gpu);enqueue=time.perf_counter()-phase;phase=time.perf_counter();block(device);sync=time.perf_counter()-phase
        phase=time.perf_counter();prediction=forward(params,*device);block(prediction);inference=time.perf_counter()-phase;total=time.perf_counter()-total_start
        per_stage={key:sum(r['stages'][key] for r in prepared) for key in prepared[0]['stages']}
        t1=total if t1 is None else t1
        rows.append({'batch_size':batch_size,'status':'passed','sample_ids':[r['sample_id'] for r in prepared],'total_wall_seconds':total,'samples_per_second':batch_size/total,'average_per_case_seconds':total/batch_size,
                     'marginal_per_case_seconds':None if batch_size==1 else (total-t1)/(batch_size-1),'cpu_preprocessing_wall_seconds':cpu_wall,'batch_host_pack_seconds':pack_wall,
                     'h2d_enqueue_seconds':enqueue,'h2d_sync_seconds':sync,'gpu_forward_and_reconstruction_seconds':inference,'summed_case_stage_seconds':per_stage,
                     'stage_utilization_fraction':{'cpu_preprocessing_wall':cpu_wall/total,'batch_host_pack':pack_wall/total,'h2d':(enqueue+sync)/total,'gpu_forward_and_reconstruction':inference/total},
                     'peak_vram_bytes':int(candidate.publication._device_memory().get('peak_bytes_in_use',0)),'prediction_finite':bool(np.all(np.isfinite(np.asarray(prediction))))})
    result={'schema_version':'heat3d_v6_p1i_p7_fresh_batch_v1','status':'passed' if args.sample_count==32 else 'passed_smoke','sample_count':args.sample_count,'route':'E16384','checkpoint_sha256':args.checkpoint_sha256,
            'protocol_sha256':sha256(args.protocol),'fresh_batch':rows,'cpu_preprocess_workers':workers,'kdtree_workers_per_case':1,'thread_oversubscription_guarded':True,
            'memory':candidate.publication._device_memory(),'role_contract':protocol['role_contract']}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':result['status'],'max_batch':max(batch_sizes),'throughput':rows[-1]['samples_per_second']}));return 0

if __name__=='__main__':raise SystemExit(main())
