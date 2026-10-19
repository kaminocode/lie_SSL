"""Single entrypoint used by snapshot-based Slurm jobs."""
import argparse
from dataclasses import replace
from pathlib import Path
import torch
from .config import setting
from .runtime import device,load,write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('train','evaluate','probe','smoke','aggregate'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    if args.stage=='aggregate':
        from .aggregate import aggregate
        aggregate(args.root,args.data_root,args.smoke)
        return
    target_device = device()
    print('PyTorch',torch.__version__,'HIP',torch.version.hip,'GPU',torch.cuda.get_device_name(0),flush=True)
    if args.stage=='probe':
        from .probe import fit
        if not 0 <= args.index < 160:
            raise ValueError('Probe index must be 0..159')
        config = setting(args.index//20,args.data_root,args.root)
        local = args.index%20
        task = ('classification','rotation')[local//10]
        representation = ('canonical','pose')[(local%10)//5]
        fit(config.run_dir/'best.pt',task,representation,42+local%5,target_device=target_device)
        return
    config = setting(args.index,args.data_root,args.root,args.smoke or args.stage=='smoke')
    if args.stage=='train':
        from .train import train
        train(config,target_device)
    elif args.stage=='evaluate':
        from .evaluate import evaluate
        evaluate(config.run_dir/'best.pt',target_device)
    else:
        from .train import train
        from .evaluate import evaluate
        from .probe import fit
        # Exercise entrypoint resume with the same planned epoch budget.
        train(config,target_device,stop_after=1)
        checkpoint = train(config,target_device)
        evaluate(checkpoint,target_device)
        for task in ('classification','rotation'):
            for representation in ('canonical','pose'):
                fit(checkpoint,task,representation,42,epochs=2,smoke=True,stop_after=1,target_device=target_device)
                fit(checkpoint,task,representation,42,epochs=2,smoke=True,target_device=target_device)
        if args.index in (0,3):
            twin = replace(config,output_root=str(args.root/'resume_reference'),wandb_mode='disabled')
            train(twin,target_device)
            actual,expected = load(config.run_dir/'last.pt'),load(twin.run_dir/'last.pt')
            for key in actual['model']:
                torch.testing.assert_close(actual['model'][key],expected['model'][key],rtol=0,atol=0)
            for a,b in zip(actual['history'],expected['history']):
                for key in a:
                    if key != 'seconds' and a[key] != b[key]:
                        raise AssertionError('Full-entrypoint resume changed training history')
        write_json(config.run_dir/'smoke_passed.json',dict(passed=True,variant=config.variant))

if __name__=='__main__':
    main()
