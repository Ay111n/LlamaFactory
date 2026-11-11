"""Resume only the failed Qwen follow-up; retain the live sft-new2 worker."""
import fcntl,json,os,subprocess,time,traceback
from pathlib import Path
import followup as original
R=original.RUNTIME

def main():
    lock=(R/'recovery_coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (R/'recovery_launch.json').exists()
    for path,digest in original.read(R/'launch.json')['asset_hashes'].items():
        assert original.sha(Path(path))==digest,path
    with (R/'qwen35_worker_attempt2.log').open('wb') as log:
        child=subprocess.Popen([original.PYTHON,str(Path(original.__file__)),'--arm','qwen35'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        original.write(R/'recovery_launch.json',dict(pid=os.getpid(),qwen_worker_pid=child.pid,retained_sft_worker=original.read(R/'sft_new2_status.json')['pid']))
        deadline=time.time()+72*3600
        while True:
            states={a:original.read(R/(a+'_status.json')) for a in original.ARMS}
            if states['qwen35']['pid']!=child.pid:
                if child.poll() is not None:raise RuntimeError('Qwen worker exited before status initialization')
                time.sleep(1);continue
            if any(s['stage']=='failed' for s in states.values()):raise RuntimeError(json.dumps(states))
            if all(s['stage']=='generation_completed' for s in states.values()):break
            if time.time()>deadline:raise TimeoutError('Follow-up deadline reached')
            time.sleep(30)
        assert child.wait()==0
    original.call([original.PYTHON,str(original.EVAL),'compare'],R/'recovery_comparison.log',dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    original.write(R/'completion.json',dict(status='generation_complete_review_pending',recovered_qwen_worker=True))

if __name__=='__main__':
    try:main()
    except BaseException:
        original.write(R/'recovery_failure.json',dict(status='failed',traceback=traceback.format_exc()))
        raise
