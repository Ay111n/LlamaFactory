"""One original-CE checkpoint restoration; use the existing experiment layout."""
import datetime,json,os,subprocess,time,traceback
from pathlib import Path
E=Path(__file__).resolve().parent;C=json.loads((E/'restore_checkpoint6000_config.json').read_text());P=json.loads((E/'paths.json').read_text());A=Path(P['analysis_root']);M=Path(P['model_root'])
STATE={'status':'running','stage':'waiting_for_verified_upload','step':6000,'training':False,'TEST':False,'same_experiment_as_other_CE_checkpoints':True}
def save():
 STATE['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();temp=E/'restore_checkpoint6000_status.tmp';temp.write_text(json.dumps(STATE,indent=2));temp.replace(E/'restore_checkpoint6000_status.json')
def gpu():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True);c=[]
 for line in raw.splitlines():
  i,n,total,used,util=[x.strip() for x in line.split(',')]
  if n=='NVIDIA L20' and int(used)<=4096 and int(total)-int(used)>=40960:c.append((int(util),int(used),int(i)))
 return min(c)[2] if c else None
def run(stage,script,args,python,card):
 STATE['stage']=stage;save();folder=E/'logs' if stage=='merge' else A/'logs';folder.mkdir(exist_ok=True)
 with (folder/('restore_checkpoint6000_'+stage+'.log')).open('x') as log:
  subprocess.run(['nice','-n','10',python,'-u',str(script),*args],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(card),SFT_EXPERIMENT_ROOT=str(E),SFT_EXPERIMENT_PATHS=str(E/'paths.json'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
def main():
 save();marker=M/'adapters/checkpoint-6000/transfer_verified.json'
 while not marker.exists():time.sleep(10)
 assert json.loads(marker.read_text())['passed'];STATE['stage']='waiting_for_low_occupancy_GPU';save()
 card=gpu()
 while card is None:time.sleep(15);card=gpu()
 STATE['gpu']=card
 assert not (M/'merged/checkpoint-6000').exists(),'Refuse to replace any existing merged model'
 run('merge',Path(C['export_script']),['--arm','checkpoint-6000'],P['training_python'],card)
 gate=json.loads((E/'export_checkpoint-6000.json').read_text());assert gate['bf16_save_reload']['passed']
 run('DEV_inference',A/'evaluate_restored_ce6000.py',[],P['inference_python'],card)
 STATE.update(stage='inference_completed_awaiting_catalog_sync',status='inference_completed',raw_result=json.loads((Path(P['results_root'])/'evaluation_checkpoint-6000.json').read_text())['run_dir']);save()
 (A/'restored_checkpoint6000_provenance/remote_inference_completed.json').write_text(json.dumps(STATE,indent=2))
if __name__=='__main__':
 try:main()
 except Exception:
  STATE.update(status='failed',stage='halted');(E/'restore_checkpoint6000_error.txt').write_text(traceback.format_exc());save();raise
