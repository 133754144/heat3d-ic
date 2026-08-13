#!/usr/bin/env python3
"""P8 exact CPU preprocessing backends and persistent-pool FVM throughput."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
import multiprocessing as mp
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
import benchmark_heat3d_v6_p1i_p7_fresh_batch as p7
import run_heat3d_v6_p1i_anchor_high_n_development as highn
import run_heat3d_v6_p1i_p5r_resolution_cell as p5r
import run_heat3d_v6_p1i_graph_scale_candidate as candidate
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_v6_full_field import build_reconstruction_map,prepare_reconstruction_domain_partition
from rigno.heat3d_v6_p1i_anchor_query import conservative_selected_control_volume,deterministic_nested_query_prefix,prepare_nested_query_geometry_cache
from rigno.models.rigno import RIGNO as GraphNeuralOperator


_PREPARE_STATE:dict[str,Any]={}
_FVM_STATE:dict[str,Any]={}

def sha256(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def bytes_sha(array:Any)->str:return hashlib.sha256(np.ascontiguousarray(np.asarray(array)).tobytes()).hexdigest()
def block(tree:Any)->None:jax.tree_util.tree_map(lambda x:x.block_until_ready() if hasattr(x,'block_until_ready') else x,tree)
def host_tree(tree:Any)->Any:return jax.tree_util.tree_map(lambda x:np.asarray(jax.device_get(x)),tree)
def stack(trees:list[Any])->Any:return jax.tree_util.tree_map(lambda *xs:np.concatenate([np.asarray(x) for x in xs],axis=0),*trees)

def graph_semantic_sha(tree:Any)->str:
    digest=hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree['graphs']):
        array=np.ascontiguousarray(np.asarray(leaf));digest.update(str(array.dtype).encode());digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()

def prepared_hash(payload:dict[str,Any])->str:
    digest=hashlib.sha256()
    for key in ('selected','selected_cv','indices','map_weights'):
        digest.update(bytes.fromhex(bytes_sha(payload[key])))
    digest.update(bytes.fromhex(graph_semantic_sha(payload['anchor'])));digest.update(bytes.fromhex(graph_semantic_sha(payload['query'])))
    return digest.hexdigest()

def worker_ready(delay:float=1.0)->int:
    time.sleep(delay);return os.getpid()

def parse()->argparse.Namespace:
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('neural','fvm'),required=True)
    for name in ('protocol','binding','artifact_root','dataset_root','manifest','full_fields','run_dir','native_padding_result','query_padding_result','output'):
        p.add_argument(f'--{name.replace("_","-")}',dest=name,type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--checkpoint-epoch',type=int,default=559)
    p.add_argument('--sample-count',type=int,choices=(1,2,32),required=True);p.add_argument('--batch-sizes',default='1,4,8,16,32');p.add_argument('--process-counts',default='1,2,4,8');return p.parse_args()

def runtime_state(args:argparse.Namespace)->dict[str,Any]:
    runtime=p5r._runtime(args);binding=json.loads(args.binding.read_text());dataset=highn._dataset(args);anchors=highn._valid_examples(dataset,binding)
    if len(anchors)!=32:raise RuntimeError('frozen valid32 required')
    preflight=json.loads((args.artifact_root/'actual_data_preflight.json').read_text());
    if preflight['sample_ids']!=[x.sample_id for x in anchors]:raise RuntimeError('valid32 order drift')
    full,_=highn._full_shared(args);coords=np.asarray(full['coords'],dtype=np.float64);cv=np.asarray(full['cv'],dtype=np.float64);layer=np.asarray(full['layer'],dtype=np.int32)
    boundaries=highn._boundaries(anchors[0],float(np.min(coords[:,2])));geometry=prepare_nested_query_geometry_cache(full_coords=coords,full_control_volume=cv,full_layer_id=layer,layer_boundaries_m=boundaries);partition=prepare_reconstruction_domain_partition(coords=coords,layer_id=layer,boundaries=boundaries)
    graph_key=highn.runner._metadata_key(int(runtime['run_config']['graph_seed']));anchor_targets=p5r._edge_targets(args.native_padding_result);query_targets=p5r._edge_targets(args.query_padding_result)
    query_config=dict(runtime['graph_config']);query_config.update(subsample_factor=64.0,discrete_graph_backend='sparse_kdtree_v1',reuse_exact_p2r_for_r2p=True)
    anchor_config=dict(runtime['graph_config']);anchor_config.update(subsample_factor=4.0,discrete_graph_backend='sparse_kdtree_v1',reuse_exact_p2r_for_r2p=True)
    physics={r['sample_id']:r for r in preflight['samples']};frozen={r['sample_id']:r for r in preflight['supports']['16384']}
    return locals()

def ensure_edge_envelope(state:dict[str,Any])->None:
    for anchor in state['anchors']:
        with jax.default_device(jax.devices('cpu')[0]):
            for targets,builder,example in (
                (state['anchor_targets'],Heat3DGraphBuilder(**state['anchor_config']),anchor),
                (state['query_targets'],Heat3DGraphBuilder(**state['query_config']),highn._query_example(anchor,highn._load_support(Path(state['frozen'][anchor.sample_id]['support_file'])),state['coords'])),
            ):
                metadata=builder.build_metadata(highn.runner._graph_coords_for_example(example,state['runtime']['stats']),key=state['graph_key']);block(metadata)
                for field in qualification.EDGE_FIELDS:
                    value=getattr(metadata,field)
                    if value is not None:targets[field]=max(int(targets.get(field) or 0),int(np.asarray(value).shape[1]))

def prepare_case(state:dict[str,Any],index:int)->dict[str,Any]:
    anchor=state['anchors'][index];start=time.perf_counter();stages={}
    with np.load(state['physics'][anchor.sample_id]['physics_cache_file'],allow_pickle=False) as physics:
        full_k=np.asarray(physics['k_xyz'],dtype=np.float64);full_q=np.asarray(physics['q_W_m3'],dtype=np.float64)
    anchor_indices,distance=highn._anchor_indices(anchor,state['coords'],float(state['binding']['numeric_tolerances']['anchor_to_solver_coordinate_max_distance_m']))
    if distance!=0.0:raise RuntimeError('anchor drift')
    phase=time.perf_counter();selected,_=deterministic_nested_query_prefix(sample_id=anchor.sample_id,anchor_indices=anchor_indices,full_q=full_q,target_count=16384,geometry_cache=state['geometry']);selected_cv,_=conservative_selected_control_volume(full_coords=state['coords'],full_control_volume=state['cv'],full_layer_id=state['layer'],selected_indices=selected,query_workers=1);stages['support_plus_cv']=time.perf_counter()-phase
    anchor_support={'selected_indices':anchor_indices,'operator_control_volume':np.asarray(anchor.operator_point_weights,dtype=np.float64),'k_xyz':np.asarray(anchor.condition.condition_features[:,:3],dtype=np.float64),'q_W_m3':np.asarray(anchor.condition.condition_features[:,3],dtype=np.float64),'layer_id':state['layer'][anchor_indices]}
    query_support={'selected_indices':selected,'operator_control_volume':selected_cv,'k_xyz':full_k[selected],'q_W_m3':full_q[selected],'layer_id':state['layer'][selected]}
    anchor_example=highn._query_example(anchor,anchor_support,state['coords']);query_example=highn._query_example(anchor,query_support,state['coords']);ab=Heat3DGraphBuilder(**state['anchor_config']);qb=Heat3DGraphBuilder(**state['query_config'])
    with jax.default_device(jax.devices('cpu')[0]):
        phase=time.perf_counter();am=ab.build_metadata(highn.runner._graph_coords_for_example(anchor_example,state['runtime']['stats']),key=state['graph_key']);block(am);stages['anchor_graph']=time.perf_counter()-phase
        phase=time.perf_counter();qm=qb.build_metadata(highn.runner._graph_coords_for_example(query_example,state['runtime']['stats']),key=state['graph_key']);block(qm);stages['query_graph']=time.perf_counter()-phase
        phase=time.perf_counter();ah=host_tree(highn._model_group(highn._prepare_group(example=anchor_example,anchor=anchor,runtime=state['runtime'],builder=ab,metadata=am,edge_targets=p5r._compatible_targets(state['anchor_targets'],am))));stages['anchor_group_pack']=time.perf_counter()-phase
        phase=time.perf_counter();qh=host_tree(highn._model_group(highn._prepare_group(example=query_example,anchor=anchor,runtime=state['runtime'],builder=qb,metadata=qm,edge_targets=p5r._compatible_targets(state['query_targets'],qm))));stages['query_group_pack']=time.perf_counter()-phase
    phase=time.perf_counter();mapping,_=build_reconstruction_map(coords=state['coords'],layer_id=state['layer'],boundaries=state['boundaries'],support_indices=selected,empty_domain_fallback='same_layer',prepared_partition=state['partition'],query_workers=1);stages['reconstruction_map']=time.perf_counter()-phase
    payload={'sample_id':anchor.sample_id,'selected':selected,'selected_cv':selected_cv,'anchor':ah,'query':qh,'weights':np.asarray(selected_cv,dtype=np.float32)[None,:],'indices':np.asarray(mapping.neighbor_local_indices,dtype=np.int32)[None,:,:],'map_weights':np.asarray(mapping.neighbor_weights,dtype=np.float64)[None,:,:],'stages':stages,'wall_seconds':time.perf_counter()-start}
    payload['prepared_payload_sha256']=prepared_hash(payload);return payload

def init_prepare_worker(serialized:dict[str,str])->None:
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='')
    ns=argparse.Namespace(**{key:(Path(value) if key not in {'checkpoint_sha256','checkpoint_epoch','sample_count'} else value) for key,value in serialized.items()});ns.checkpoint_epoch=int(ns.checkpoint_epoch);ns.sample_count=int(ns.sample_count)
    global _PREPARE_STATE;_PREPARE_STATE=runtime_state(ns)
    _PREPARE_STATE['anchor_targets']=json.loads(serialized['_anchor_targets_json'])
    _PREPARE_STATE['query_targets']=json.loads(serialized['_query_targets_json'])

def prepare_worker(index:int)->dict[str,Any]:return prepare_case(_PREPARE_STATE,index)

def run_backend(state:dict[str,Any],backend:str,count:int)->tuple[list[dict[str,Any]],float,float]:
    indices=list(range(count));startup=0.0;start=time.perf_counter()
    if backend=='serial':rows=[prepare_case(state,index) for index in indices]
    elif backend.startswith('thread'):
        workers=int(backend[6:]);
        with ThreadPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(lambda index:prepare_case(state,index),indices))
    else:
        workers=int(backend[7:]);ctx=mp.get_context('spawn');serialized={key:str(getattr(state['args'],key)) for key in ('protocol','binding','artifact_root','dataset_root','manifest','full_fields','run_dir','native_padding_result','query_padding_result','checkpoint_sha256','checkpoint_epoch','sample_count')};serialized['_anchor_targets_json']=json.dumps(state['anchor_targets']);serialized['_query_targets_json']=json.dumps(state['query_targets'])
        os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        pool_start=time.perf_counter();pool=ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=init_prepare_worker,initargs=(serialized,));ready={future.result() for future in [pool.submit(worker_ready) for _ in range(workers*4)]};
        if len(ready)!=workers:raise RuntimeError(f'{backend}: persistent workers did not all initialize')
        startup=time.perf_counter()-pool_start
        steady=time.perf_counter();rows=list(pool.map(prepare_worker,indices));wall=time.perf_counter()-steady;pool.shutdown();return rows,startup,wall
    return rows,startup,time.perf_counter()-start

def create_prepare_pool(state:dict[str,Any],backend:str)->tuple[ProcessPoolExecutor,float]:
    workers=int(backend[7:]);ctx=mp.get_context('spawn');serialized={key:str(getattr(state['args'],key)) for key in ('protocol','binding','artifact_root','dataset_root','manifest','full_fields','run_dir','native_padding_result','query_padding_result','checkpoint_sha256','checkpoint_epoch','sample_count')};serialized['_anchor_targets_json']=json.dumps(state['anchor_targets']);serialized['_query_targets_json']=json.dumps(state['query_targets']);os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1');started=time.perf_counter();pool=ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=init_prepare_worker,initargs=(serialized,));ready={future.result() for future in [pool.submit(worker_ready) for _ in range(workers*4)]}
    if len(ready)!=workers:raise RuntimeError(f'{backend}: batch pool initialization failed')
    return pool,time.perf_counter()-started

def neural(args:argparse.Namespace)->int:
    protocol=json.loads(args.protocol.read_text());state=runtime_state(args);state['args']=args;ensure_edge_envelope(state);count=args.sample_count
    backends=['serial','thread2','thread4','thread8','process2','process4','process8'];reference=None;comparisons=[];prepared_by_backend={}
    for name in backends:
        rows,startup,wall=run_backend(state,name,count);hashes=[row['prepared_payload_sha256'] for row in rows];reference=hashes if reference is None else reference;exact=hashes==reference;comparisons.append({'backend':name,'status':'passed' if exact else 'failed_exact','worker_count':1 if name=='serial' else int(''.join(filter(str.isdigit,name))),'startup_seconds':startup,'steady_wall_seconds':wall,'samples_per_second':count/wall,'average_per_case_seconds':wall/count,'prepared_payload_sha256':hashes,'exact_vs_serial':exact,'summed_stage_seconds':{key:sum(row['stages'][key] for row in rows) for key in rows[0]['stages']}});prepared_by_backend[name]=rows
        if not exact:raise RuntimeError(f'{name}: exact gate failed')
    winner=max(comparisons,key=lambda row:row['samples_per_second'])['backend'];batch_sizes=[int(x) for x in args.batch_sizes.split(',') if int(x)<=count]
    if count<32:
        result={'schema_version':'heat3d_v6_p1i_p8_neural_v1','status':'passed_smoke','sample_count':count,'backend_comparison':comparisons,'winner':winner,'all_backends_exact':True,'fresh_batch':[],'role_contract':protocol['role_contract']};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'passed_smoke','winner':winner}));return 0
    runtime=state['runtime'];model=GraphNeuralOperator(**runtime['model_config']);params=highn.runner._device_params(runtime['checkpoint']['params']);gpu=jax.devices('gpu')[0]
    @jax.jit
    def forward(params,anchor,query,weights,indices,map_weights):
        a=highn.runner._model_apply(model,params,anchor);raw=a['raw_temperature'][:,0,:,0];scale=a['s_hat'].reshape((raw.shape[0],-1))[:,0];q=highn.runner._model_apply(model,params,query)['raw_temperature'][:,0,:,0]-highn.REFERENCE_K;normalized=weights/jnp.sum(weights,axis=1,keepdims=True);qscale=jnp.sqrt(jnp.sum(normalized*q*q,axis=1));support=q/qscale[:,None]*scale[:,None];gathered=support[jnp.arange(support.shape[0])[:,None,None],indices];return jnp.sum(gathered*map_weights.astype(support.dtype),axis=2)
    def host_batch(rows):return (stack([r['anchor'] for r in rows]),stack([r['query'] for r in rows]),np.concatenate([r['weights'] for r in rows]),np.concatenate([r['indices'] for r in rows]),np.concatenate([r['map_weights'] for r in rows]))
    warm=prepared_by_backend[winner][0];warm_host=host_batch([warm]);warm_device=jax.device_put(warm_host,gpu);block(warm_device);block(forward(params,*warm_device));batch_rows=[];t1=None;batch_pool=None;batch_pool_startup=0.0
    if winner.startswith('process'):batch_pool,batch_pool_startup=create_prepare_pool(state,winner)
    for batch_size in batch_sizes:
        compile_host=host_batch([warm]*batch_size);compile_device=jax.device_put(compile_host,gpu);block(compile_device);block(forward(params,*compile_device));total_start=time.perf_counter();phase=time.perf_counter()
        if batch_pool is not None:rows=list(batch_pool.map(prepare_worker,range(batch_size)));startup=batch_pool_startup
        else:rows,startup,_=run_backend(state,winner,batch_size)
        prep_wall=time.perf_counter()-phase;phase=time.perf_counter();host=host_batch(rows);pack=time.perf_counter()-phase;phase=time.perf_counter();device=jax.device_put(host,gpu);enqueue=time.perf_counter()-phase;phase=time.perf_counter();block(device);sync=time.perf_counter()-phase;phase=time.perf_counter();prediction=forward(params,*device);block(prediction);infer=time.perf_counter()-phase;total=time.perf_counter()-total_start;t1=total if t1 is None else t1;batch_rows.append({'batch_size':batch_size,'status':'passed','backend':winner,'backend_startup_seconds_outside_span':startup,'total_wall_seconds':total,'samples_per_second':batch_size/total,'average_per_case_seconds':total/batch_size,'marginal_per_case_seconds':None if batch_size==1 else (total-t1)/(batch_size-1),'cpu_preprocessing_wall_seconds':prep_wall,'host_pack_seconds':pack,'h2d_enqueue_seconds':enqueue,'h2d_sync_seconds':sync,'gpu_forward_and_reconstruction_seconds':infer,'prediction_finite':bool(np.all(np.isfinite(np.asarray(prediction)))),'peak_vram_bytes':int(candidate.publication._device_memory().get('peak_bytes_in_use',0))})
    if batch_pool is not None:batch_pool.shutdown()
    result={'schema_version':'heat3d_v6_p1i_p8_neural_v1','status':'passed','sample_count':32,'backend_comparison':comparisons,'winner':winner,'all_backends_exact':True,'fresh_batch':batch_rows,'kdtree_workers_per_case':1,'protocol_sha256':sha256(args.protocol),'role_contract':protocol['role_contract']};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'passed','winner':winner,'best_samples_per_s':max(r['samples_per_second'] for r in batch_rows)}));return 0

def init_fvm_worker(serialized:dict[str,str])->None:
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='')
    args=argparse.Namespace(**{key:(Path(value) if key in {'dataset_root','manifest','full_fields'} else value) for key,value in serialized.items()});data=qualification.FamilyData(family='p1i',dataset_root=args.dataset_root,manifest_path=args.manifest,full_fields_path=args.full_fields,randomblock_config=None);rows=data.selected_rows(32);physics=data.physics(rows[0]);mesh=qualification.prior.core.build_mesh(physics);shared=data.full_shared()
    if not np.array_equal(mesh['coords'],shared['coords']):raise RuntimeError('FVM mesh drift')
    global _FVM_STATE;_FVM_STATE={'data':data,'rows':rows,'mesh':mesh,'prepared':{}}
    if serialized.get('prepare_all')=='true':
        for index in range(len(rows)):
            row=rows[index];example,_=data.load_example(row);k,q=data.full_kq(row);top_h=float(example.condition.condition_features[0,8]);bottom_h=float(example.condition.condition_features[0,9]);_FVM_STATE['prepared'][index]=qualification.prior._assemble(mesh,k,q,top_h,bottom_h)

def fvm_worker(index:int)->dict[str,Any]:
    state=_FVM_STATE;row=state['rows'][index];start=time.perf_counter();example,_=state['data'].load_example(row);k,q=state['data'].full_kq(row);data_s=time.perf_counter()-start;phase=time.perf_counter();top_h=float(example.condition.condition_features[0,8]);bottom_h=float(example.condition.condition_features[0,9]);system=qualification.prior._assemble(state['mesh'],k,q,top_h,bottom_h);assembly=time.perf_counter()-phase;phase=time.perf_counter();temperature=qualification.prior._solve(*system);solve=time.perf_counter()-phase
    if not np.all(np.isfinite(temperature)):raise RuntimeError('nonfinite FVM')
    return {'sample_id':row['sample_id'],'data_seconds':data_s,'assembly_seconds':assembly,'linear_solve_seconds':solve,'continuous_compute_seconds':data_s+assembly+solve}

def fvm_prepare_worker(index:int)->dict[str,Any]:
    state=_FVM_STATE;row=state['rows'][index];example,_=state['data'].load_example(row);k,q=state['data'].full_kq(row);top_h=float(example.condition.condition_features[0,8]);bottom_h=float(example.condition.condition_features[0,9]);system=qualification.prior._assemble(state['mesh'],k,q,top_h,bottom_h);return {'sample_id':row['sample_id'],'system':system}

def fvm_solve_prepared(system:Any)->np.ndarray:
    temperature=qualification.prior._solve(*system)
    if not np.all(np.isfinite(temperature)):raise RuntimeError('nonfinite prepared FVM')
    return temperature

def fvm_solve_cached_worker(index:int)->dict[str,Any]:
    started=time.perf_counter();temperature=fvm_solve_prepared(_FVM_STATE['prepared'][index]);return {'sample_id':_FVM_STATE['rows'][index]['sample_id'],'solve_seconds':time.perf_counter()-started,'finite':bool(np.all(np.isfinite(temperature)))}

def fvm(args:argparse.Namespace)->int:
    protocol=json.loads(args.protocol.read_text());counts=[int(x) for x in args.process_counts.split(',')];serialized={key:str(getattr(args,key)) for key in ('dataset_root','manifest','full_fields')};ctx=mp.get_context('spawn');rows=[]
    for count in counts:
        os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        startup_start=time.perf_counter();pool=ProcessPoolExecutor(max_workers=count,mp_context=ctx,initializer=init_fvm_worker,initargs=(serialized,));ready={future.result() for future in [pool.submit(worker_ready) for _ in range(count*4)]};
        if len(ready)!=count:raise RuntimeError(f'FVM p{count}: persistent workers did not all initialize')
        startup=time.perf_counter()-startup_start;steady_start=time.perf_counter();measurements=list(pool.map(fvm_worker,range(args.sample_count)));steady=time.perf_counter()-steady_start;shutdown_start=time.perf_counter();pool.shutdown();shutdown=time.perf_counter()-shutdown_start;rows.append({'process_count':count,'threads_per_process':1,'persistent_worker_pool':True,'worker_pids':sorted(ready),'sample_count':args.sample_count,'status':'passed','startup_seconds':startup,'steady_wall_seconds':steady,'shutdown_seconds_outside_steady':shutdown,'samples_per_second':args.sample_count/steady,'average_per_case_seconds':steady/args.sample_count,'measurements':measurements})
    saturation=max(rows,key=lambda row:row['samples_per_second']);result={'schema_version':'heat3d_v6_p1i_p8_persistent_fvm_v1','status':'passed','sample_count':args.sample_count,'rows':rows,'saturation':saturation,'process_counts':counts,'threads_per_process':1,'startup_separate_from_steady':True,'protocol_sha256':sha256(args.protocol),'role_contract':{'accessed_roles':['valid_iid'],'training':False,'test':False,'sealed':False,'checkpoint_modified':False,'dataset_modified':False,'graph_semantics_modified':False}};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'passed','saturation_processes':saturation['process_count'],'samples_per_s':saturation['samples_per_second']}));return 0

def main()->int:
    args=parse();return neural(args) if args.mode=='neural' else fvm(args)
if __name__=='__main__':raise SystemExit(main())
