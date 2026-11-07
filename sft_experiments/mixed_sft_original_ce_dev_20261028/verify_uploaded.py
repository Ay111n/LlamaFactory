"""Verify explicit local-to-L20 adapter transfer once and mark checkpoint ready."""
import argparse,hashlib,json,datetime
from pathlib import Path
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--step',type=int,required=True);a=p.parse_args();assert a.step in [1500,3000,4500,6000,7500,9000]
paths=json.loads((R/'paths.json').read_text());cp=Path(paths['model_root'])/'adapters'/f'checkpoint-{a.step}'
m=json.loads((cp/'transfer_manifest.json').read_text(encoding='utf-8-sig'))
for x in m['files']:
 f=cp/x['name'];assert f.is_file() and f.stat().st_size==x['bytes'],f
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
 assert h.hexdigest()==x['sha256'].lower(),f'Hash mismatch: {f}'
assert json.loads((cp/'trainer_state.json').read_text())['global_step']==a.step
cfg=json.loads((cp/'adapter_config.json').read_text());assert cfg['r']==64 and cfg['lora_alpha']==128 and cfg['lora_dropout']==.05
assert cfg['base_model_name_or_path']==paths['base_model'];assert 'lm_head' in cfg['modules_to_save']
result={'passed':True,'step':a.step,'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':m['files'],'local_source':m['local_source'],'optimizer_uploaded':False}
(cp/'transfer_verified.json').write_text(json.dumps(result,indent=2));print('TRANSFER_VERIFIED',a.step,flush=True)
