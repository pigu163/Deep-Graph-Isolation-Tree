from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, polars as pl
ROOT=Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT/'script'):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from script.common.metrics import compute_metrics
from script.common.results import run_dir, save_scores, write_json
from script.common.seeds import set_seed
from script.common.timing import Timer
from script.common.validation import tabular_interface_check
from script.tabular_pipeline.datasets import load_tabular_dataset
from script.model.SILIF.silif import SILIFDetector
MODEL='SILIF'
def args():
 p=argparse.ArgumentParser(description='SilIF tabular baseline')
 p.add_argument('--dataset',required=True,choices=['midas_darpa','ibm_hi_small','ibm_li_small','paysim','elliptic_pp'])
 p.add_argument('--protocol',default='chronological',choices=['chronological','transductive']); p.add_argument('--seed',type=int,default=42)
 p.add_argument('--device',default='cpu',choices=['cpu','cuda']); p.add_argument('--overwrite',action='store_true'); p.add_argument('--num-workers',type=int,default=0); p.add_argument('--batch-size',type=int,default=0); p.add_argument('--max-samples',type=int,default=None); p.add_argument('--quick',action='store_true')
 p.add_argument('--n-estimators',type=int,default=100); p.add_argument('--n-clusters',type=int,default=8); p.add_argument('--alpha',type=float,default=.5); p.add_argument('--tree-max-samples',type=int,default=256)
 return p.parse_args()
def splits(n):
 t=int(n*.7); return {'train':(0,t),'test':(t,n)}
def main():
 a=args();
 if a.device != 'cpu': print('SILIF is CPU-only; using CPU.')
 out=run_dir(MODEL,a.dataset,a.protocol,a.seed)
 if out.exists() and not a.overwrite: print(f'skip (exists, no --overwrite): {out}'); return
 out.mkdir(parents=True,exist_ok=True); set_seed(a.seed); data=load_tabular_dataset(a.dataset,'all')
 if a.max_samples:
  n=min(a.max_samples,len(data.X)); data.X,data.y,data.label_available,data.sample_ids,data.timestamps=(z[:n] for z in (data.X,data.y,data.label_available,data.sample_ids,data.timestamps))
 sp=splits(len(data.X)); fit_x=data.X if a.protocol=='transductive' else data.X[:sp['train'][1]]
 ft=Timer();
 with ft: model=SILIFDetector(a.n_estimators,a.n_clusters,a.alpha,a.tree_max_samples,a.seed,-1).fit(fit_x)
 st=Timer();
 with st: score=model.score(data.X)
 metrics={}; warnings=[]
 for name,(s,e) in sp.items():
  m=compute_metrics(data.y[s:e],score[s:e]); warnings+=m.pop('warnings',[]); metrics[name]=m
 metrics.update(status='ok',paper_fidelity='source_algorithm_temporal_adaptation',timing={'fit_seconds':round(ft.seconds,6),'score_seconds':round(st.seconds,6),'throughput':round(len(score)/max(st.seconds,1e-9),3)})
 cfg={'model':MODEL,'dataset':a.dataset,'protocol':a.protocol,'seed':a.seed,'device':'cpu','requested_device':a.device,'num_workers':a.num_workers,'batch_size':a.batch_size,'max_samples_used':a.max_samples,'is_quick_run':a.quick,'score_direction':'larger=more anomalous','paper_fidelity':'source_algorithm_temporal_adaptation',**model.config()}
 write_json(out/'config.json',cfg); write_json(out/'metrics.json',metrics); write_json(out/'timing.json',{'fit_seconds':ft.seconds,'score_seconds':st.seconds,'throughput':len(score)/max(st.seconds,1e-9)}); write_json(out/'profile.json',model.profile_); write_json(out/'warnings.json',{'warnings':warnings})
 ss=np.empty(len(score),dtype=object)
 for name,(s,e) in sp.items(): ss[s:e]=name
 save_scores(out/'scores.parquet',pl.DataFrame({'sample_id':data.sample_ids,'timestamp_or_step':data.timestamps,'label':data.y,'label_available':data.label_available,'anomaly_score':score,'split':ss}))
 write_json(out/'interface_check.json',tabular_interface_check(data.X,data.y,data.label_available,data.sample_ids,data.feature_names,score,fitted_on='all' if a.protocol=='transductive' else 'train'))
 (out/'run.log').write_text(json.dumps({'shape':list(data.X.shape),'timing':metrics['timing']},indent=2)); print(f'done: {out}')
if __name__=='__main__': main()
