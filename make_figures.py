import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
import train_screening as s

def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    out=s.BASE/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(9,2.7),layout='constrained')
    for ax,d in zip(axes,['clinic','mass']):
        run=s.BASE/'runs'/f'{d}_seed42';history=[json.loads(x) for x in (run/'history.jsonl').read_text().splitlines()]
        ax.plot([x['epoch'] for x in history],[100*x['val_dice'] for x in history],color='#127D85',lw=1.4)
        result=json.loads((run/'results.json').read_text());ax.axvline(result['best_epoch'],color='#DF854A',ls='--',lw=1)
        ax.set(xlabel='Epoch',ylabel='Validation Dice (%)',title=('ClinicDB (normalized)' if d=='clinic' else 'Buildings (fixed sigmoid)'));ax.grid(alpha=.15)
    fig.savefig(out/'learning_curves.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(9,4.8),layout='constrained');checks={}
    for row,d in enumerate(['clinic','mass']):
        run=s.BASE/'runs'/f'{d}_seed42';result=json.loads((run/'results.json').read_text())
        record=min(result['native_raw']['per_image'],key=lambda x:x['dice']);name=record['name']
        if d=='clinic':
            p=s.BASE/'data/ClinicDB/test/images'/name;q=s.BASE/'data/ClinicDB/test/masks'/name;size=352
        else:
            p=s.BASE/'data/Massachusetts/test/sat'/name
            q=next((s.BASE/'data/Massachusetts/test/map').glob(p.stem+'.tif*'));size=512
        image=Image.open(p).convert('RGB')
        data=s.load_data(d,'test');index=data['tile_names'].index(name);mask=data['native'][index]
        model=s.MK_UNet().cuda();model.load_state_dict(torch.load(run/'best.pt',weights_only=False)['model']);model.eval()
        all_z=s.logits_for(model,data)
        z=all_z[index:index+1] if d=='clinic' else all_z[index*4:(index+1)*4]
        if d=='mass':z=torch.cat([torch.cat([z[0:1],z[1:2]],3),torch.cat([z[2:3],z[3:4]],3)],2)
        pmap=F.interpolate(z,size=mask.shape,mode='bilinear',align_corners=False).sigmoid()[0,0].detach().cpu().numpy();pred=pmap>=.5
        check=s.scores(torch.from_numpy(pred),torch.from_numpy(mask))
        assert abs(check['dice']-record['dice'])<1e-3,(check['dice'],record['dice'])
        checks[d]=dict(name=name,recorded=record,regenerated=check,absolute_dice_difference=abs(check['dice']-record['dice']),note='GPU inference is not bitwise deterministic; masks can differ at rounding-sensitive threshold pixels')
        for ax,data,title in zip(axes[row],[np.asarray(image),mask,pred],['Input','Native reference','Prediction (sigmoid >= 0.5)']):
            ax.imshow(data,cmap='gray' if data.ndim==2 else None);ax.set_title(title,fontsize=9);ax.axis('off')
        axes[row,0].text(0,-.06,('ClinicDB' if d=='clinic' else 'Buildings')+f' lowest-Dice case: {name}; regenerated Dice {100*check["dice"]:.2f}%',transform=axes[row,0].transAxes,fontsize=8)
        del model,data,all_z,z;torch.cuda.empty_cache()
    fig.savefig(out/'qualitative_failures.png',dpi=200);plt.close(fig)
    (out/'qualitative_verification.json').write_text(json.dumps(checks,indent=2))
    print('Figures saved; selected-case metrics independently reproduced')

if __name__=='__main__':main()
