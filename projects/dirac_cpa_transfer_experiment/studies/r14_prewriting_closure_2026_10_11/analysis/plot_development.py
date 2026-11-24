"""All prespecified tuning candidates; descriptive development evidence only."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plan import BASE,sha,write

def main():
    selection=json.loads((BASE/'frozen/baseline_selection.json').read_text(encoding='utf-8'))
    data=pd.DataFrame(selection['development_rows']);cost=pd.read_csv(BASE/'analysis_outputs/development_costs_all_candidates.csv')
    data=data.merge(cost,on=['case','family','candidate','seed'],validate='one_to_one')
    assert len(data)==24 and set(data.seed)=={910,911}
    out=BASE/'figures';out.mkdir(exist_ok=True);source=out/'source_data';source.mkdir(exist_ok=True)
    data.to_csv(source/'development_tuning_all_candidates.csv',index=False)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    fig,axes=plt.subplots(2,2,figsize=(9.5,6.2),layout='constrained')
    labels=['SIREN\n30 / 1e-3','SIREN\n30 / 1e-4','SIREN\n60 / 1e-4','Fourier\n1 / 1e-3','Fourier\n4 / 1e-3','Fourier\n8 / 3e-4']
    colors={'siren':'#b07649','fourier':'#456d8d'}
    for col,case in enumerate(('T10','m10')):
        for row,(metric,ylabel) in enumerate((('score','Frozen development score'),('training_wall_seconds','Training wall time (s)'))):
            ax=axes[row,col];table=data[data.case==case]
            for family,offset in (('siren',0),('fourier',3)):
                for i in range(3):
                    t=table[(table.family==family)&(table.candidate==i)].sort_values('seed');x=i+offset
                    for seed,shift,marker in ((910,-.085,'o'),(911,.085,'^')):
                        v=t[t.seed==seed][metric].iloc[0]
                        ax.scatter(x+shift,v,s=30,color=colors[family],marker=marker,label=f'Seed {seed}' if family=='siren' and i==0 else None,zorder=3)
                    mean=float(t[metric].mean());ax.plot([x-.2,x+.2],[mean,mean],color='#262626',lw=1.4)
                    if i==selection['per_case_candidate'][case][family]:
                        ax.scatter(x,mean,s=140,facecolors='none',edgecolors='#262626',marker='s',linewidths=.8,zorder=2)
            ax.axvline(2.5,color='#dddddd',lw=.8);ax.set_xticks(range(6),labels,fontsize=7);ax.set_ylabel(ylabel)
            ax.set_title(('T = 10' if case=='T10' else 'm = 10')+(' — selection' if row==0 else ' — search cost'),loc='left',fontweight='bold')
            ax.text(-.11,1.04,chr(97+row*2+col),transform=ax.transAxes,fontweight='bold',fontsize=11)
            ax.set_xlim(-.5,5.5)
    axes[0,0].legend(loc='lower left',fontsize=7)
    for ext in ('png','svg','pdf'):fig.savefig(out/f'development_tuning.{ext}',dpi=220)
    plt.close(fig)
    caption=('All24 frozen development attempts: two families, three candidate settings, two cases, two independent development seeds910–911, 4000steps each. '
        'Dots show both seeds and horizontal bars their mean; open squares mark the automatically selected candidate in each case. '
        'Labels give SIREN omega/lr or Fourier feature scale/lr. Scores use the preregistered sum of normalized spinor, current, modal and fidelity errors. '
        'No uncertainty interval or formal-method conclusion is drawn from n=2. Search time includes every candidate, measured during concurrent CPU training; '
        'evaluation and setup costs remain in the accompanying ledger. The globally selected extension family is Fourier; formal810–814 results did not select it.')
    (out/'development_tuning_caption.txt').write_text(caption,encoding='utf-8')
    write(out/'development_tuning_metadata.json',dict(status='GENERATED_REQUIRES_VISUAL_CHECK',script_sha256=sha(__file__),
        source_sha256=sha(source/'development_tuning_all_candidates.csv'),selection_sha256=sha(BASE/'frozen/baseline_selection.json'),all24_attempts_shown=True))

if __name__=='__main__':main()
