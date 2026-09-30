"""Auditable MK-UNet screening experiments; official architecture stays unchanged."""
import argparse, sys, json, time, random, hashlib, platform, csv
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).parent/'MK-UNet'))
from mkunet_network import MK_UNet

BASE=Path(__file__).parent
MEAN=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
STD=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_data(dataset,split):
    xs=[];ys=[];names=[];native=[];manifest=[]
    if dataset=='clinic':
        root=BASE/'data/ClinicDB'/split
        pairs=[(p,root/'masks'/p.name) for p in sorted((root/'images').glob('*.png'))]
        size=352
    else:
        root=BASE/'data/Massachusetts'/('valid' if split=='val' else split)
        masks={p.stem:p for p in (root/'map').glob('*.tif*')}
        pairs=[(p,masks[p.stem]) for p in sorted((root/'sat').glob('*.tif*'))]
        size=512
    if not pairs:raise RuntimeError(f'No data: {root}')
    for p,q in pairs:
        a=Image.open(p).convert('RGB');m=Image.open(q).convert('L')
        assert a.size==m.size,(p,a.size,m.size)
        native.append(np.array(m)>127)
        x=np.array(a.resize((size,size),Image.Resampling.BILINEAR))
        y=(np.array(m.resize((size,size),Image.Resampling.NEAREST))>127).astype(np.uint8)
        manifest.append(dict(image=str(p.relative_to(BASE)),mask=str(q.relative_to(BASE)),image_sha256=digest(p),mask_sha256=digest(q),native_size=a.size))
        if dataset=='clinic':
            xs.append(x);ys.append(y);names.append(p.name)
        else:
            for row in range(2):
                for col in range(2):
                    xs.append(x[row*256:(row+1)*256,col*256:(col+1)*256]);ys.append(y[row*256:(row+1)*256,col*256:(col+1)*256]);names.append(f'{p.name}:{row},{col}')
    x=torch.from_numpy(np.stack(xs).transpose(0,3,1,2).copy()).cuda().float()/255
    x=(x-MEAN)/STD
    y=torch.from_numpy(np.stack(ys)[:,None].copy()).cuda().float()
    return dict(x=x,y=y,names=names,native=native,manifest=manifest,tile_names=[p.name for p,q in pairs])

def structure_loss(pred,mask):
    w=1+5*torch.abs(F.avg_pool2d(mask,31,1,15)-mask)
    b=(w*F.binary_cross_entropy_with_logits(pred,mask,reduction='none')).sum((2,3))/w.sum((2,3))
    p=pred.sigmoid();inter=(p*mask*w).sum((2,3));union=((p+mask)*w).sum((2,3))
    return (b+1-(inter+1)/(union-inter+1)).mean()

def scores(p,y):
    p=p.bool();y=y.bool();i=(p&y).sum().item();a=p.sum().item();b=y.sum().item()
    return dict(dice=(2*i+1e-6)/(a+b+1e-6),iou=(i+1e-6)/(a+b-i+1e-6),intersection=i,predicted=a,foreground=b,pixels=y.numel())

@torch.inference_mode()
def logits_for(model,data,bs=16):
    model.eval();out=[]
    for i in range(0,len(data['x']),bs):out.append(model(data['x'][i:i+bs])[0].float())
    return torch.cat(out)

@torch.inference_mode()
def evaluate(model,data,dataset,normalized=False):
    z=logits_for(model,data);records=[]
    for i,y in enumerate(data['native']):
        if dataset=='clinic':p=z[i:i+1]
        else:
            v=z[i*4:(i+1)*4]
            p=torch.cat([torch.cat([v[0:1],v[1:2]],3),torch.cat([v[2:3],v[3:4]],3)],2)
        p=F.interpolate(p,size=y.shape,mode='bilinear',align_corners=False).sigmoid()[0,0]
        if normalized:p=(p-p.min())/(p.max()-p.min()+1e-8)
        s=scores(p>=.5,torch.from_numpy(y).cuda());s['name']=data['tile_names'][i];records.append(s)
    total_i=sum(x['intersection'] for x in records);total_a=sum(x['predicted'] for x in records);total_b=sum(x['foreground'] for x in records)
    return dict(mean_dice=float(np.mean([x['dice'] for x in records])),mean_iou=float(np.mean([x['iou'] for x in records])),global_iou=(total_i+1e-6)/(total_a+total_b-total_i+1e-6),per_image=records)

def main():
    a=argparse.ArgumentParser();a.add_argument('--dataset',choices=['clinic','mass'],required=True);a.add_argument('--epochs',type=int,default=200);a.add_argument('--seed',type=int,default=42);a.add_argument('--batch',type=int,default=16);a.add_argument('--amp',action='store_true');a.add_argument('--smoke',action='store_true');opt=a.parse_args()
    torch.set_num_threads(4);random.seed(opt.seed);np.random.seed(opt.seed);torch.manual_seed(opt.seed);torch.cuda.manual_seed_all(opt.seed);torch.backends.cudnn.benchmark=True
    run=BASE/'runs'/f'{opt.dataset}_seed{opt.seed}';run.mkdir(parents=True,exist_ok=True)
    config=vars(opt)|dict(started_utc=datetime.now(timezone.utc).isoformat(),torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),python=sys.version,platform=platform.platform(),architecture_commit='2fa8b230a057602539af7655203ad434bf56a5b6',architecture_sha256=digest(BASE/'MK-UNet/mkunet_network.py'),lr=.0001,weight_decay=.0001,channels=[16,32,64,96,160],kernels=[1,3,5],augmentation=False,scales=[.75,1.,1.25] if opt.dataset=='clinic' else [1.],selection='maximum validation mean native-resolution min-max-normalized Dice' if opt.dataset=='clinic' else 'maximum validation mean native-resolution raw-sigmoid Dice',test_access='once after checkpoint selection',bitwise_determinism=False)
    (run/'config.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(config),flush=True)
    train=load_data(opt.dataset,'train');val=load_data(opt.dataset,'val')
    manifest={'train':train['manifest'],'val':val['manifest']}
    train_ids={x['image_sha256'] for x in manifest['train']};val_ids={x['image_sha256'] for x in manifest['val']};assert train_ids.isdisjoint(val_ids),'Duplicate train/val images'
    print('Loaded',len(train['x']),len(val['x']),'training/validation samples',flush=True)
    model=MK_UNet().cuda();print('Parameters',sum(p.numel() for p in model.parameters()),flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
    scaler=torch.amp.GradScaler('cuda',enabled=opt.amp)
    best=-1;best_epoch=0;start=time.perf_counter();history=[]
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,opt.epochs+1):
        t=time.perf_counter();model.train();order=torch.randperm(len(train['x']),device='cuda');losses=[]
        for step in range(0,len(order),opt.batch):
            idx=order[step:step+opt.batch];x=train['x'][idx];y=train['y'][idx]
            for scale in config['scales']:
                size=int(round(x.shape[-1]*scale/32)*32)
                xx=x if scale==1 else F.interpolate(x,size=(size,size),mode='bilinear',align_corners=True)
                yy=y if scale==1 else F.interpolate(y,size=(size,size),mode='nearest')
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',enabled=opt.amp):pred=model(xx)[0]
                loss=structure_loss(pred.float(),yy)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is not None:p.grad.clamp_(-.5,.5)
                scaler.step(optimizer);scaler.update()
                if scale==1:losses.append(loss.item())
        metric=evaluate(model,val,opt.dataset,normalized=opt.dataset=='clinic')
        if metric['mean_dice']>best:
            best=metric['mean_dice'];best_epoch=epoch
            torch.save(dict(model=model.state_dict(),epoch=epoch,validation=metric,config=config),run/'best.pt')
        row=dict(epoch=epoch,loss=float(np.mean(losses)),val_dice=metric['mean_dice'],val_iou=metric['mean_iou'],seconds=time.perf_counter()-t,best_epoch=best_epoch)
        history.append(row)
        with (run/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if opt.smoke:break
    torch.save(dict(model=model.state_dict(),epoch=epoch,optimizer=optimizer.state_dict(),config=config),run/'last.pt')
    checkpoint=torch.load(run/'best.pt',weights_only=False);model.load_state_dict(checkpoint['model'])
    if opt.smoke:return
    test=load_data(opt.dataset,'test');manifest['test']=test['manifest'];test_ids={x['image_sha256'] for x in manifest['test']}
    assert test_ids.isdisjoint(train_ids|val_ids),'Duplicate held-out images'
    (run/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    result=dict(best_epoch=best_epoch,best_val_dice=best,training_wall_seconds=time.perf_counter()-start,peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20,parameter_count=sum(p.numel() for p in model.parameters()),checkpoint_sha256=digest(run/'best.pt'),native_raw=evaluate(model,test,opt.dataset),native_minmax=evaluate(model,test,opt.dataset,normalized=True),completed_utc=datetime.now(timezone.utc).isoformat())
    (run/'results.json').write_text(json.dumps(result,indent=2));print('FINAL',json.dumps({k:v for k,v in result.items() if k not in ['native_raw','native_minmax']}),flush=True)

if __name__=='__main__':main()
