"""Predeclared stress test and one-step episodic adaptation pilot, not full TENT/EATA."""
import json,sys,time,copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import train_screening as s

PROTOCOL=dict(seed=20260930,conditions=['clean','brightness_0.65','gamma_1.8','gaussian_noise_0.08','gaussian_blur_1.5'],lr=.001,steps=1,accept_js_improvement=1e-6,max_area_change=.05,harm_dice_threshold=.01,episodic_reset=True,update='BatchNorm affine only; source running statistics frozen',labels='used only for final scoring, never for updates or gating',status='fixed before seeing held-out stress-test results')

def bernoulli_js(p,q):
    p=p.clamp(1e-6,1-1e-6);q=q.clamp(1e-6,1-1e-6);m=(p+q)/2
    def kl(a,b):return a*(a/b).log()+(1-a)*((1-a)/(1-b)).log()
    return ((kl(p,m)+kl(q,m))/2).mean()

def entropy(p,base):
    p=p.clamp(1e-6,1-1e-6);h=-(p*p.log()+(1-p)*(1-p).log())
    mask=base>=.5
    parts=[h[mask].mean()] if mask.any() else []
    if (~mask).any():parts.append(h[~mask].mean())
    return torch.stack(parts).mean()

def corruption(x,name,index):
    rgb=(x*s.STD+s.MEAN).clamp(0,1)
    if name=='brightness_0.65':rgb=rgb*.65
    if name=='gamma_1.8':rgb=rgb**1.8
    if name=='gaussian_noise_0.08':
        gen=torch.Generator(device='cuda').manual_seed(PROTOCOL['seed']+index)
        rgb=(rgb+.08*torch.randn(rgb.shape,device='cuda',generator=gen)).clamp(0,1)
    if name=='gaussian_blur_1.5':
        a=torch.arange(-5,6,device='cuda').float();g=torch.exp(-a*a/(2*1.5**2));g=g/g.sum();k=(g[:,None]*g[None,:])[None,None].repeat(3,1,1,1)
        rgb=F.conv2d(F.pad(rgb,(5,5,5,5),mode='reflect'),k,groups=3)
    return (rgb-s.MEAN)/s.STD

def native_logits(z,dataset,shape):
    if dataset=='mass':
        z=torch.cat([torch.cat([z[0:1],z[1:2]],3),torch.cat([z[2:3],z[3:4]],3)],2)
    return F.interpolate(z.float(),size=shape,mode='bilinear',align_corners=False).sigmoid()[0,0]

def bootstrap_delta(a,b):
    delta=np.asarray(a)-np.asarray(b);rng=np.random.default_rng(PROTOCOL['seed']);means=delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(1)
    return dict(mean=float(delta.mean()),low=float(np.quantile(means,.025)),high=float(np.quantile(means,.975)),unit='case/tile; source video dependence not modeled')

def main(dataset):
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    run=s.BASE/'runs'/f'{dataset}_seed42';out=run/'pilot';out.mkdir(exist_ok=True)
    (out/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
    model=s.MK_UNet().cuda();checkpoint=torch.load(run/'best.pt',weights_only=False);model.load_state_dict(checkpoint['model']);model.eval()
    (out/'provenance.json').write_text(json.dumps(dict(checkpoint_sha256=s.digest(run/'best.pt'),script_sha256=s.digest(Path(__file__)),protocol=PROTOCOL),indent=2))
    for p in model.parameters():p.requires_grad_(False)
    affine=[]
    for module in model.modules():
        if isinstance(module,torch.nn.BatchNorm2d):
            module.weight.requires_grad_(True);module.bias.requires_grad_(True);affine.extend([module.weight,module.bias])
    print(dataset,'adaptable parameters',sum(p.numel() for p in affine),flush=True)
    data=s.load_data(dataset,'test');summary=[];all_records=[]
    for condition in PROTOCOL['conditions']:
        cases=[]
        warm=data['x'][:1] if dataset=='clinic' else data['x'][:4]
        with torch.no_grad():
            for _ in range(3):model(warm)
        torch.cuda.synchronize()
        for i,y in enumerate(data['native']):
            model.load_state_dict(checkpoint['model']);model.eval()
            x=data['x'][i:i+1] if dataset=='clinic' else data['x'][i*4:(i+1)*4]
            x=corruption(x,condition,i)
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.no_grad():
                z0=model(x)[0];p0=z0.sigmoid();zh0=model(x.flip(3))[0].flip(3);js0=bernoulli_js(p0,zh0.sigmoid()).item()
            torch.cuda.synchronize();baseline_seconds=time.perf_counter()-start
            start=time.perf_counter();optimizer=torch.optim.SGD(affine,lr=PROTOCOL['lr']);optimizer.zero_grad()
            p=model(x)[0].sigmoid();loss=entropy(p,p0);loss.backward();optimizer.step()
            with torch.no_grad():
                z1=model(x)[0];p1=z1.sigmoid();zh1=model(x.flip(3))[0].flip(3);js1=bernoulli_js(p1,zh1.sigmoid()).item()
            area_change=((p1>=.5).float().mean()-(p0>=.5).float().mean()).abs().item()
            accept=js1<=js0-PROTOCOL['accept_js_improvement'] and area_change<=PROTOCOL['max_area_change']
            torch.cuda.synchronize();extra_seconds=time.perf_counter()-start
            target=torch.from_numpy(y).cuda();base=s.scores(native_logits(z0,dataset,y.shape)>=.5,target);candidate=s.scores(native_logits(z1,dataset,y.shape)>=.5,target);gated=candidate if accept else base
            row=dict(condition=condition,name=data['tile_names'][i],baseline=base,candidate=candidate,gated=gated,accepted=accept,js_before=js0,js_after=js1,area_change=area_change,baseline_two_forward_seconds=baseline_seconds,adapt_extra_seconds=extra_seconds)
            cases.append(row);all_records.append(row)
        agg=dict(condition=condition,n=len(cases),baseline_dice=float(np.mean([c['baseline']['dice'] for c in cases])),candidate_dice=float(np.mean([c['candidate']['dice'] for c in cases])),gated_dice=float(np.mean([c['gated']['dice'] for c in cases])),accepted=sum(c['accepted'] for c in cases),harmful_accepted=sum(c['accepted'] and c['candidate']['dice']<c['baseline']['dice']-PROTOCOL['harm_dice_threshold'] for c in cases),harmful_accepted_any=sum(c['accepted'] and c['candidate']['dice']<c['baseline']['dice']-1e-4 for c in cases),harmful_candidates=sum(c['candidate']['dice']<c['baseline']['dice']-PROTOCOL['harm_dice_threshold'] for c in cases),mean_extra_seconds=float(np.mean([c['adapt_extra_seconds'] for c in cases])),delta_ci=bootstrap_delta([c['gated']['dice'] for c in cases],[c['baseline']['dice'] for c in cases]))
        summary.append(agg);print(json.dumps(agg),flush=True)
    (out/'per_case.json').write_text(json.dumps(all_records,indent=2));(out/'summary.json').write_text(json.dumps(dict(protocol=PROTOCOL,trainable_parameters=sum(p.numel() for p in affine),conditions=summary),indent=2))

if __name__=='__main__':main(sys.argv[1])
