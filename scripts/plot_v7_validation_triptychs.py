#!/usr/bin/env python3
"""Plot verified V7 exports with shared temperature/error scales and provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/heat3d-v7-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def draw(root, names, basename, title):
    data=[dict(np.load(root/(n+'.npz'))) for n in names]
    meta=[json.loads((root/(n+'.json')).read_text()) for n in names]
    first=data[0]; c=first['coords']; t=first['truth']; peak=int(np.argmax(t)); z=float(c[peak,2]); layer=int(first['layer_id'][peak])
    for d,m,n in zip(data,meta,names):
        assert m['output_sha256']==sha(root/(n+'.npz'))
        assert m['sample_id']==meta[0]['sample_id'] and np.array_equal(d['coords'],c) and np.array_equal(d['truth'],t)
    dense=len(c)>1024
    if dense:
        sel=np.isclose(c[:,2],z,atol=1e-12,rtol=0); xs=np.unique(c[sel,0]); ys=np.unique(c[sel,1])
        ix=np.searchsorted(xs,c[sel,0]); iy=np.searchsorted(ys,c[sel,1])
        def project(v):
            out=np.empty((len(ys),len(xs))); out[iy,ix]=v[sel]; return out
        mode=f'Direct dense inference: {len(c):,} nodes; exact {len(xs)} x {len(ys)} slice; no spatial smoothing'
        fallback=0.
    else:
        sel=first['layer_id']==layer; source=c[sel]; origin=c.min(0); length=np.ptp(c,axis=0).max()
        xs=np.linspace(c[:,0].min(),c[:,0].max(),256); ys=np.linspace(c[:,1].min(),c[:,1].max(),256)
        xx,yy=np.meshgrid(xs,ys); query=np.column_stack([xx.ravel(),yy.ravel(),np.full(xx.size,z)])
        points=(source-origin)/length; query=(query-origin)/length
        from scipy.interpolate import RBFInterpolator
        def project(v):
            return RBFInterpolator(points,v[sel],kernel='thin_plate_spline',smoothing=0)(query).reshape(256,256)
        from scipy.spatial import Delaunay
        fallback=float(np.mean(Delaunay(points).find_simplex(query)<0))
        mode=f'Native 1,024 nodes; same-layer thin-plate RBF ({len(source)} nodes) to 256 x 256; display only; {fallback:.1%} outside support hull'
    truth=project(t); predictions=[project(d['prediction']) for d in data]; errors=[p-truth for p in predictions]
    assert all(np.isfinite(v).all() for v in [truth,*predictions,*errors])
    lo=min(float(truth.min()),*(float(p.min()) for p in predictions)); hi=max(float(truth.max()),*(float(p.max()) for p in predictions))
    errmax=max(float(np.max(np.abs(e))) for e in errors)
    extent=[xs.min(),xs.max(),ys.min(),ys.max()]
    if not dense: extent=[v*1000 for v in extent]
    unit='benchmark coordinate' if dense else 'mm'
    source_boxes=[]
    if not dense:
        physical=json.loads((root/'P1i_sample_meta.json').read_text())
        assert sha(root/'P1i_sample_meta.json')==meta[0]['sample_files']['sample_meta.json']
        layer_name=physical['physics']['layers_bottom_to_top'][layer]['id']
        source_boxes=[b for b in physical['q_blocks'] if b['layer']==layer_name]
    fig=plt.figure(figsize=(14,12 if dense else 3.35*len(names)+2.25),facecolor='white')
    gs=fig.add_gridspec(len(names),3,left=.11,right=.95,top=.84 if dense else .875,bottom=.21 if dense else .12,wspace=.36,hspace=.65 if dense else .48)
    labels={'Full':'Heat3D (G1 Full)','RIGNO':'RIGNO (G1 vanilla)','GINO':'GINO','Transolver':'Transolver','Heat3D-DeepOHeat':'Heat3D (G2 e600)','DeepOHeat':'DeepOHeat (matched 768)'}
    for i,(name,p,e,m) in enumerate(zip(names,predictions,errors,meta)):
        pair_lo=min(float(truth.min()),float(p.min())); pair_hi=max(float(truth.max()),float(p.max()))
        for j,v in enumerate([truth,p,e]):
            ax=fig.add_subplot(gs[i,j]); im=ax.imshow(v,origin='lower',extent=extent,aspect='equal',interpolation='bilinear',cmap='turbo' if j<2 else 'RdBu_r',norm=Normalize(pair_lo,pair_hi) if j<2 else Normalize(-errmax,errmax))
            ax.set_xlabel(f'x ({unit})',fontsize=9); ax.set_ylabel(f'y ({unit})',fontsize=9)
            ax.tick_params(labelsize=8)
            if i==0: ax.set_title([r'Reference $\Delta T$ (K)',r'Prediction $\Delta T$ (K)','Prediction − reference (K)'][j],fontsize=12,pad=14)
            if j==0:
                ax.text(-.42,.5,labels[name],rotation=90,transform=ax.transAxes,va='center',ha='center',fontsize=12,fontweight='bold')
            cb=fig.colorbar(im,ax=ax,fraction=.047,pad=.035); cb.ax.tick_params(labelsize=8); cb.set_label('K',fontsize=8)
            from matplotlib.patches import Rectangle
            import matplotlib.patheffects as pe
            for b in source_boxes:
                x0,x1,y0,y1=b['bbox_fraction_xy']; x0,x1=x0*10,x1*10; y0,y1=y0*10,y1*10
                patch=Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='white',linewidth=1.05,linestyle='--')
                patch.set_path_effects([pe.Stroke(linewidth=2,foreground='#222222'),pe.Normal()]); ax.add_patch(patch)
                if j==0: ax.text((x0+x1)/2,y1+.13,'Q'+b['block_id'][-2:],ha='center',fontsize=7,color='white',path_effects=[pe.withStroke(linewidth=1.7,foreground='#222222')])
        pos=gs[i,0].get_position(fig)
        epoch=m.get('epoch',m.get('checkpoint_metadata',{}).get('epoch','best'))
        fig.text(.12,pos.y0-(.055 if dense else .035),f"seed 0 | epoch/iteration: {epoch} | ckpt SHA256: {m['checkpoint_sha256'][:16]}… | native/full point relative L2: {m['point_relative_L2_pct']:.3f}%",fontsize=8.3,color='#39424c')
        ckpt=('/'.join(Path(m['checkpoint']).parts[-3:]) if name not in ['Full','RIGNO'] else ('formal_21_runs/'+('Full_seed0' if name=='Full' else 'vanilla_RIGNO_seed0')+'/params_best_sample_first.pkl'))
        fig.text(.12,pos.y0-(.075 if dense else .050),'Checkpoint: '+ckpt,fontsize=7.7,color='#39424c')
    fig.suptitle(title,fontsize=19,fontweight='bold',y=.981)
    fig.text(.5,.945,f"{meta[0]['sample_id']} | split: {meta[0]['split']} | z = {z:.7g} | layer = {layer} | slice at reference peak",ha='center',fontsize=10)
    fig.text(.5,.921,mode,ha='center',fontsize=9,color='#44515d')
    foot='Each row: identical temperature limits/colors for reference and prediction; each panel has its own colorbar.\nDashed boxes: actual heat-source footprints in the displayed layer. Shared symmetric error limits across models.\nFull checkpoint paths, hashes, sample provenance and commands: '+basename+'_manifest.json + README.md'
    if dense: foot=foot.replace('Dashed boxes: actual heat-source footprints in the displayed layer. ', '')
    fig.text(.12,.025,foot,fontsize=8.5,linespacing=1.6,color='#39424c')
    fig.savefig(root/(basename+'.png'),dpi=220); fig.savefig(root/(basename+'.pdf')); plt.close(fig)
    manifest=dict(title=title,sample=meta[0]['sample_id'],z=z,layer=layer,interpolation=mode,outside_hull_fraction=fallback,temperature_limits_by_row=[[min(float(truth.min()),float(p.min())),max(float(truth.max()),float(p.max()))] for p in predictions],source_boxes=source_boxes,error_limits=[-errmax,errmax],rows=meta,plot_script_sha256=sha(__file__))
    (root/(basename+'_manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    print(basename, 'PASS', manifest['temperature_limits_by_row'],manifest['error_limits'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--group',choices=['p1i','deepoheat','all'],default='all');a=p.parse_args()
    if a.group in ['p1i','all']: draw(a.artifacts,['Full','RIGNO','GINO','Transolver'],'V7_P1i_comparison','V7 G1 / G2 · P1i validation case')
    if a.group in ['deepoheat','all']: draw(a.artifacts,['Heat3D-DeepOHeat','DeepOHeat'],'V7_DeepOHeat_comparison','V7 G2 · DeepOHeat validation case')
