import json,re,sys,time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
from peft import PeftModel
here=Path(__file__).resolve().parent
root=here.parents[2]
runtime=here/'runtime'
cfg=json.loads((runtime/'config.json').read_text())
ckpt=json.loads((runtime/'checkpoint.json').read_text())['checkpoint']
base=cfg['model_name_or_path']
tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
model=AutoModelForImageTextToText.from_pretrained(base,local_files_only=True,torch_dtype=torch.bfloat16,
                                                device_map={'':'cuda:0'},attn_implementation='sdpa')
model=PeftModel.from_pretrained(model,ckpt).eval()
items=[json.loads(l) for l in (root/'benchmark/eval/tibetan_formateval_validation/mcqa_4way.jsonl').open()]
unique={}
for r in sorted(items,key=lambda x:len(x['question'])+len(str(x['choices']))):
    unique.setdefault(r['source_dataset'],r)
items=list(unique.values())[:4]
assert len(items)>=3
items.insert(0, {
    'id': 'language_smoke_arithmetic',
    'source_dataset': 'synthetic',
    'question': 'གཉིས་དང་གསུམ་བསྡོམས་ན་ག་ཚོད་ཡིན།',
    'choices': ['བཞི།', 'ལྔ།', 'དྲུག', 'བདུན།'],
})
extended = '--extended' in sys.argv
if extended:
    arithmetic = {
        'id': 'language_smoke_arithmetic_open',
        'source_dataset': 'synthetic',
        'question': 'གཉིས་དང་གསུམ་བསྡོམས་ན་ག་ཚོད་ཡིན། ཚིག་གཅིག་གིས་ལན་རྒྱོབ།',
        'choices': [],
    }
    items = [arithmetic, next(row for row in items if row['id'] == 'tfe-val-0000')]
def script(s):
    bo=len(re.findall(r'[\u0f40-\u0f6c]',s))
    en=len(re.findall(r'[A-Za-z]',s))
    return dict(tibetan_letters=bo,latin_letters=en,dominant='bo' if bo>2*en else 'en' if en>2*bo else 'mixed_or_short')
filename = 'predictions_extended.jsonl' if extended else 'predictions.jsonl'
with (runtime/filename).open('w') as out,torch.inference_mode():
 for r in items:
    opts=r['choices']
    if isinstance(opts,list):
        assert all(isinstance(v,str) for v in opts)
    choices='\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(opts)) if isinstance(opts,list) else '\n'.join(f'{k}. {v}' for k,v in opts.items())
    question='གཤམ་གྱི་དྲི་བར་ལན་ཐོབ།\n'+r['question']+'\n'+choices
    prompt=tok.apply_chat_template([{'role':'user','content':question}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
    assert prompt.rstrip().endswith('<think>')
    inputs=tok(prompt,return_tensors='pt',add_special_tokens=False).to('cuda')
    start=time.monotonic()
    limit = 6144 if extended and r['source_dataset'] != 'synthetic' else 2048
    ids=model.generate(**inputs,max_new_tokens=limit,do_sample=False,pad_token_id=tok.eos_token_id)[0,inputs.input_ids.shape[1]:]
    output=tok.decode(ids,skip_special_tokens=False).replace('<|im_end|>','').replace('<|endoftext|>','').strip()
    think,sep,answer=output.partition('</think>')
    think=think.removeprefix('<think>').strip(); answer=answer.strip()
    result=dict(id=r['id'],source=r['source_dataset'],thinking=think,answer=answer,thinking_closed=bool(sep),
                reasoning_script=script(think),answer_script=script(answer),generated_tokens=len(ids),seconds=time.monotonic()-start)
    out.write(json.dumps(result,ensure_ascii=False)+'\n');out.flush()
    print(r['id'],result['reasoning_script'],result['answer_script'],len(ids),bool(sep),flush=True)
