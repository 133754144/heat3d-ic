#!/usr/bin/env python3
"""Full FVM reference; registered dense/native model outputs on one full slice."""
import argparse,hashlib,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/heat3d-v7-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay,cKDTree

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);a=p.parse_args();r=a.artifacts
 truthfile=r/'P1i_FVM.npz';full=dict(np.load(truthfile));densemeta=json.loads((r/'Full_dense.json').read_text());assert sha(truthfile)==densemeta['truth_sha256']
 c=full['coords'];t=full['truth'];layer=full['layer_id'];assert len(c)==240825
 peak=int(t.argmax());z=float(c[peak,2]);lid=int(layer[peak]);sel=np.isclose(c[:,2],z,atol=1e-12,rtol=0)
 xs=np.unique(c[sel,0]);ys=np.unique(c[sel,1]);assert len(xs)==len(ys)==65
 ix=np.searchsorted(xs,c[sel,0]);iy=np.searchsorted(ys,c[sel,1])
 def grid(v):
  out=np.empty((65,65));out[iy,ix]=v[sel];return out
 reference=grid(t);xx,yy=np.meshgrid(xs,ys);queries=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,z)]
 names=['Full_dense','RIGNO','GINO','Transolver'];labels=['Heat3D (G1 Full)','RIGNO (G1 vanilla)','GINO','Transolver'];preds=[];metadata=[];routes=[]
 for name in names:
  d=dict(np.load(r/(name+'.npz')));m=json.loads((r/(name+'.json')).read_text());assert sha(r/(name+'.npz'))==m['output_sha256'];assert m['sample_id']==densemeta['sample_id']
  if name=='Full_dense':
   assert np.array_equal(d['coords'],c) and np.array_equal(d['truth'],t);prediction=grid(d['prediction']);route='240,825 direct U-v2 queries → exact 65 × 65 slice';extra=0.
  else:
   distance,ids=cKDTree(c).query(d['coords']);assert distance.max()<1e-14
   assert np.allclose(d['truth'],t[ids],atol=1e-5,rtol=1e-6), 'native labels differ from full FVM at matching nodes'
   mask=d['layer_id']==lid;origin=c.min(0);scale=np.ptp(c,axis=0).max();points=(d['coords'][mask]-origin)/scale;q=(queries-origin)/scale
   prediction=RBFInterpolator(points,d['prediction'][mask],kernel='thin_plate_spline',smoothing=0)(q).reshape(65,65)
   extra=float(np.mean(Delaunay(points).find_simplex(q)<0));route=f'1,024 native queries → same-layer RBF slice; extrapolation {extra:.1%}'
  assert np.isfinite(prediction).all();preds.append(prediction);metadata.append(m);routes.append(dict(description=route,outside_hull_fraction=extra))
 lo=min(reference.min(),*(v.min() for v in preds));hi=max(reference.max(),*(v.max() for v in preds));errors=[v-reference for v in preds];limit=max(abs(v).max() for v in errors)
 norm=Normalize(lo,hi);cmap=plt.get_cmap('turbo');rgba=cmap(norm(reference));checks=[]
 physical=json.loads((r/'P1i_sample_meta.json').read_text());layername=physical['physics']['layers_bottom_to_top'][lid]['id'];boxes=[b for b in physical['q_blocks'] if b['layer']==layername]
 fig=plt.figure(figsize=(14,18));gs=fig.add_gridspec(4,3,left=.115,right=.95,top=.86,bottom=.16,wspace=.36,hspace=.54)
 extent=[xs.min()*1000,xs.max()*1000,ys.min()*1000,ys.max()*1000]
 for i,(name,label,pred,error,m,route) in enumerate(zip(names,labels,preds,errors,metadata,routes)):
  for j,v in enumerate([reference,pred,error]):
   ax=fig.add_subplot(gs[i,j]);im=ax.imshow(v,origin='lower',extent=extent,aspect='equal',interpolation='bilinear',cmap='turbo' if j<2 else 'RdBu_r',norm=norm if j<2 else Normalize(-limit,limit))
   cb=fig.colorbar(im,ax=ax,fraction=.047,pad=.035);cb.ax.tick_params(labelsize=8);cb.set_label('K',fontsize=8)
   ax.set_xlabel('x (mm)',fontsize=9);ax.set_ylabel('y (mm)',fontsize=9);ax.tick_params(labelsize=8)
   if i==0:ax.set_title(['FVM reference ΔT (K)','Prediction ΔT (K)','Prediction − FVM (K)'][j],fontsize=12,pad=15)
   if j==0:
    assert np.array_equal(im.to_rgba(im.get_array()),rgba);checks.append(hashlib.sha256(rgba.tobytes()).hexdigest());ax.text(-.42,.5,label,rotation=90,transform=ax.transAxes,ha='center',va='center',fontsize=12,fontweight='bold')
   for b in boxes:
    x0,x1,y0,y1=b['bbox_fraction_xy'];x0,x1=[extent[0]+x*(extent[1]-extent[0]) for x in (x0,x1)];y0,y1=[extent[2]+y*(extent[3]-extent[2]) for y in (y0,y1)]
    patch=Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color='white',linewidth=1,linestyle='--');patch.set_path_effects([pe.Stroke(linewidth=1.8,foreground='#333'),pe.Normal()]);ax.add_patch(patch)
  pos=gs[i,0].get_position(fig);epoch=m.get('epoch',m.get('checkpoint_metadata',{}).get('epoch'))
  fig.text(.115,pos.y0-.032,f"seed 0 | epoch {epoch} | {route['description']}",fontsize=8.5)
  ckpt='/'.join(Path(m['checkpoint']).parts[-3:]) if name not in ['Full_dense','RIGNO'] else f"{'Full' if name=='Full_dense' else 'vanilla_RIGNO'}_seed0/params_best_sample_first.pkl"
  fig.text(.115,pos.y0-.049,f"Checkpoint: {ckpt} | SHA256: {m['checkpoint_sha256'][:16]}…",fontsize=7.7,color='#39424c')
 fig.suptitle('V7 G1 / G2 · P1i full-resolution FVM reference',fontsize=19,fontweight='bold',y=.98)
 fig.text(.5,.944,f"{densemeta['sample_id']} | valid_iid | z = {z*1000:.6g} mm | {layername} | slice at full-FVM peak",ha='center',fontsize=10)
 fig.text(.5,.92,'Reference: 65 × 65 × 57 = 240,825 FVM nodes; identical exact 65 × 65 slice in every row',ha='center',fontsize=10)
 fig.text(.5,.899,'Complete rectangular fields · shared temperature range and colors · one colorbar per panel',ha='center',fontsize=9)
 fig.text(.115,.032,'Reference is read directly from the full FVM archive, never interpolated from sparse labels.\nHeat3D: registered direct-query route. Other models: native outputs interpolated/extrapolated for display.\nDashed boxes: actual source footprints. Diagnostic illustration; mixed output resolutions, not a new benchmark ranking.\nReproduction: V7_P1i_comparison_manifest.json and README.md',fontsize=8.5,linespacing=1.6,color='#39424c')
 fig.savefig(r/'V7_P1i_comparison.png',dpi=220);fig.savefig(r/'V7_P1i_comparison.pdf');plt.close(fig)
 manifest=dict(sample_id=densemeta['sample_id'],reference='exact dense FVM',reference_nodes=240825,slice_shape=[65,65],z=z,layer=lid,temperature_limits=[float(lo),float(hi)],error_limits=[-float(limit),float(limit)],reference_rgba_sha256_by_row=checks,reference_rgba_identical=len(set(checks))==1,rows=metadata,routes=routes,plot_script_sha256=sha(__file__),masked_pixels=0)
 (r/'V7_P1i_comparison_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 np.savez_compressed(r/'P1i_display_arrays.npz',reference=reference,predictions=np.stack(preds),errors=np.stack(errors),x=xs,y=ys)
 print('PASS: dense FVM references identical; full rectangular slice; finite values; native/full labels aligned')
if __name__=='__main__':main()
