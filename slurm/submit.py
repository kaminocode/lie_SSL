#!/usr/bin/env python3
"""Create a source snapshot and submit the LUMI experiment dependency chain."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

SIF = '/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20260929_104918/lumi-multitorch-full-u24r72f21m50t211-20260929_104918.sif'
DATA = '/scratch/project_465003463/rahul/unsupervised_g_iclr_2026_my_implementation/data/rottrans_mnist'
BASE = '/scratch/project_465003463/checkpoints/baburaja/lie_SSL'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('smoke','production'))
    parser.add_argument('--smoke-root',type=Path)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--account',default='project_465003463')
    parser.add_argument('--partition',choices=('standard-g','small-g'),default='standard-g')
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    stamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    root = args.root or Path(BASE)/(args.mode+'_'+stamp)
    if args.mode=='production':
        if not args.smoke_root:
            parser.error('Production requires --smoke-root with a completed passing audit')
        audit = json.loads((args.smoke_root/'comparison/audit.json').read_text())
        if not audit.get('passed') or not audit.get('smoke') or audit.get('probe_fits') != 32:
            raise RuntimeError('The smoke audit did not pass completely')
        prior = json.loads((args.smoke_root/'source_hashes.json').read_text())
        for name,digest in prior.items():
            if hashlib.sha256((repo/name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Source changed after smoke: '+name)
    if root.exists():
        raise FileExistsError('Refusing to overwrite run root '+str(root))
    if not Path(SIF).is_file() or not (Path(DATA)/'MNIST/raw/train-images-idx3-ubyte').is_file():
        raise FileNotFoundError('Pinned SIF or MNIST data are missing')
    if args.dry_run:
        print(json.dumps(dict(root=str(root),account=args.account,mode=args.mode,backbones=8,
                              probe_fits=32 if args.mode=='smoke' else 160,
                              partition='dev-g' if args.mode=='smoke' else args.partition,
                              concurrency=2 if args.mode=='smoke' else (8 if args.partition=='standard-g' else 4)),indent=2))
        return
    root.mkdir(parents=True)
    snapshot = root/'source'
    snapshot.mkdir()
    for name in ('lie_ssl','tests','slurm'):
        shutil.copytree(str(repo/name),str(snapshot/name),ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('README.md','PATCHES.md','source_provenance.json'):
        shutil.copy2(str(repo/name),str(snapshot/name))
    hashes = {str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob('*')) if p.is_file()}
    (root/'source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    for path in snapshot.rglob('*'):
        if path.is_file():
            path.chmod(0o440)
    (root/'environment.sh').write_text('export SIF='+shlex.quote(SIF)+'\nexport DATA_ROOT='+shlex.quote(DATA)+'\n')
    environment = dict(sif=SIF,sif_bytes=Path(SIF).stat().st_size,data_root=DATA,account=args.account,
                       partition='dev-g' if args.mode=='smoke' else args.partition,
                       wandb_mode='offline',smoke_root=str(args.smoke_root) if args.smoke_root else None)
    (root/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    variants = ('baseline','block_skew','dense_circle','block_skew_circle')
    manifest = 'index\tdataset\tvariant\tseed\tepochs\n'
    for index in range(8):
        manifest += '{}\t{}\t{}\t42\t{}\n'.format(index,('unblocked','blocked_per_source')[index//4],variants[index%4],2 if args.mode=='smoke' else 300)
    (root/'manifest.tsv').write_text(manifest)
    logs = repo/'logs'/root.name
    logs.mkdir(parents=True,exist_ok=True)
    jobs = []
    def submit(stage,script,extra,script_args):
        command = ['sbatch','--parsable','--account='+args.account,'--job-name=lie_'+stage,
                   '--chdir='+str(snapshot),'--output='+str(logs/(stage+'_%A_%a.out')),
                   '--error='+str(logs/(stage+'_%A_%a.err'))]+extra+[str(snapshot/'slurm'/script),str(root)]+script_args
        job = subprocess.check_output(command,universal_newlines=True).strip().split(';')[0]
        jobs.append(dict(stage=stage,job_id=job,command=command))
        (root/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
        print(stage,job,flush=True)
        return job
    if args.mode=='smoke':
        first = submit('smoke','job.sh',['--partition=dev-g','--time=00:10:00','--array=0-1%2'],['smoke'])
        submit('audit','aggregate.sh',['--dependency=afterok:'+first,'--kill-on-invalid-dep=yes'],['smoke'])
    elif args.partition=='standard-g':
        extra_args = [str(snapshot/'slurm'),str(logs)]
        first = submit('train','full_node.sh',['--time=05:00:00'],['train']+extra_args)
        render = submit('evaluate','full_node.sh',['--time=01:00:00','--dependency=afterok:'+first,'--kill-on-invalid-dep=yes'],['evaluate']+extra_args)
        probes = submit('probe','full_node.sh',['--array=0-19%1','--time=24:00:00','--dependency=afterok:'+render,'--kill-on-invalid-dep=yes'],['probe']+extra_args)
        submit('aggregate','aggregate.sh',['--dependency=afterok:'+probes,'--kill-on-invalid-dep=yes'],[])
    else:
        first = submit('train','job.sh',['--array=0-7%4','--time=05:00:00'],['train'])
        render = submit('evaluate','job.sh',['--array=0-7%4','--time=01:00:00','--dependency=afterok:'+first,'--kill-on-invalid-dep=yes'],['evaluate'])
        probes = submit('probe','job.sh',['--array=0-159%4','--time=24:00:00','--dependency=afterok:'+render,'--kill-on-invalid-dep=yes'],['probe'])
        submit('aggregate','aggregate.sh',['--dependency=afterok:'+probes,'--kill-on-invalid-dep=yes'],[])
    (repo/('latest_'+args.mode+'.json')).write_text(json.dumps(dict(root=str(root),jobs=jobs),indent=2)+'\n')
    print('RUN_ROOT='+str(root),flush=True)

if __name__=='__main__':
    main()
