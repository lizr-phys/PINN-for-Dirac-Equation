from __future__ import annotations
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plan import BASE,write,sha

def main():
    out=BASE/'figures';out.mkdir(exist_ok=True);sources=out/'source_data';sources.mkdir(exist_ok=True)
    calibration=json.loads((BASE/'references/calibration.json').read_text(encoding='utf-8'));rows=pd.DataFrame(calibration['comparisons'])
    rows.to_csv(sources/'reference_calibration.csv',index=False)
    order=['T10','m10','m5T5','V04','two_harmonics','chirped'];labels=['T = 10','m = 10','m5T5','V = 0.4','Two harmonics','Chirped IC']
    rows=rows.set_index('case').loc[order].reset_index()
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.8,'legend.frameon':False})
    fig,axes=plt.subplots(1,3,figsize=(10.3,3.5),layout='constrained')
    x=np.arange(6)
    for i,(col,limit,title) in enumerate((('trajectory_rel_difference',1e-6,'Full complex trajectory'),('lower_boundary_occupation',1e-8,'Outer 16-layer occupation'))):
        ax=axes[i];ax.scatter(x,rows[col],s=26,color='#456d8d',label='Lower boundary' if i==1 else '161 vs 193 modes')
        if i==1:ax.scatter(x,rows['upper_boundary_occupation'],s=25,color='#b07649',marker='s',label='Upper boundary')
        ax.axhline(limit,color='#8a3a3a',ls='--',lw=1,label='Frozen threshold');ax.set_yscale('log')
        ax.set_xticks(x,labels,rotation=35,ha='right');ax.set_ylabel('Relative difference' if i==0 else 'Total occupation');ax.set_title(title,loc='left',fontweight='bold')
        ax.legend(fontsize=7,loc='lower left');ax.text(-.15,1.04,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=12)
    test=json.loads((BASE/'engineering/TSFP_tests.json').read_text(encoding='utf-8'));dt=np.array([.025,.0125,.00625]);errors=np.array(test['Strang_dt_errors'])
    ax=axes[2];ax.loglog(dt,errors,'o-',color='#456d8d',lw=1,label='Independent test')
    ax.loglog(dt,errors[-1]*(dt/dt[-1])**2,'--',color='#9a9a9a',label=r'$O(\Delta t^2)$')
    ax.set_xlabel('Time step');ax.set_ylabel('Spinor relative error');ax.set_title('TSFP2 order check',loc='left',fontweight='bold');ax.legend(fontsize=7)
    ax.text(-.15,1.04,'c',transform=ax.transAxes,fontweight='bold',fontsize=12)
    for ext in ('png','svg','pdf'):fig.savefig(out/f'reference_calibration.{ext}',dpi=220)
    plt.close(fig)
    pd.DataFrame({'dt':dt,'spinor_error':errors}).to_csv(sources/'TSFP_order_test.csv',index=False)
    caption='Reference calibration before learning. (a) Zero-padded full complex trajectories on the frozen 161-mode grid vs193 modes, six independent cases. (b) Maximum total occupation in each outer16-layer boundary block. Dashed lines are the frozen1e-6 and1e-8 thresholds. (c) Independent TSFP2 dt-halving test over a short interval; all three preset steps are shown. These deterministic calibration results are not five-seed learning evidence. Source data and computation checks accompany the figure.'
    (out/'reference_calibration_caption.txt').write_text(caption,encoding='utf-8')
    write(out/'reference_figure_generation.json',dict(script_sha256=sha(__file__),calibration_sha256=sha(BASE/'references/calibration.json'),
        actual_visual_inspection_required=True,visual_QA_complete=False,files=[f'reference_calibration.{ext}' for ext in ('png','svg','pdf')]))
    print('REFERENCE_FIGURE_CREATED')

if __name__=='__main__':main()
