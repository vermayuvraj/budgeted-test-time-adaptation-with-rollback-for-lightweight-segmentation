from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

work=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(9,2.7),layout='constrained')
for ax,d,label in zip(axes,['clinic','mass'],['ClinicDB','Massachusetts Buildings']):
    run=work/'runs'/f'{d}_seed42'
    history=[json.loads(x) for x in (run/'history.jsonl').read_text().splitlines()]
    result=json.loads((run/'results.json').read_text())
    ax.plot([x['epoch'] for x in history],[100*x['val_dice'] for x in history],color='#222222',lw=1.2)
    ax.axvline(result['best_epoch'],color='#777777',ls='--',lw=.8,label=f"Selected epoch {result['best_epoch']}")
    ax.set(title=label,xlabel='Training epoch',ylabel='Validation Dice (%)')
    ax.legend(frameon=False,fontsize=8,loc='lower right')
    ax.grid(axis='y',color='#DDDDDD',lw=.5)
fig.savefig(work/'figures/learning_curves.png',dpi=200)
plt.close(fig)
