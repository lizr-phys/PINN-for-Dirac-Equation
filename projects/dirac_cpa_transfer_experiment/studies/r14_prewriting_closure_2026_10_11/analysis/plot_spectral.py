"""Complete TSFP2 error-cost curves, with primary and wide reference traceability."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from plan import BASE,sha,write

def main():
    data=pd.read_csv(BASE/'spectral_reference_check/spectral_error_cost_wide_reference.csv')
    assert len(data)==108 and all(len(t)==18 for _,t in data.groupby('case'))
    out=BASE/'figures';out.mkdir(exist_ok=True);source=out/'source_data';source.mkdir(exist_ok=True)
    data.to_csv(source/'TSFP_error_cost_all108.csv',index=False)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    fig,axes=plt.subplots(2,3,figsize=(10.5,6.0),layout='constrained')
    cases=('T10','m10','m5T5','V04','two_harmonics','chirped')
    labels=('T = 10','m = 10','m5T5','V = 0.4','Equal-RMS two harmonics','Chirped initial state')
    colors=('#456d8d','#b07649','#5d7764');markers=('o','s','^')
    for index,(ax,case,label) in enumerate(zip(axes.ravel(),cases,labels)):
        table=data[data.case==case]
        for grid,color,marker in zip((256,512,1024),colors,markers):
            t=table[table.grid==grid].sort_values('dt',ascending=False)
            ax.loglog(t.cost_median_seconds,t.spinor_rel_l2_gauge_aligned,marker=marker,color=color,lw=1,ms=4,label=f'Nx = {grid}')
        t=table[table.grid==1024].sort_values('dt',ascending=False)
        ax.loglog(t.cost_median_seconds,t.primary161_spinor_error,color='#979797',ls=':',lw=1.4,label='Primary161 ref (Nx1024)')
        ax.set_title(label,loc='left',fontweight='bold');ax.set_xlabel('Complete propagation time (s)');ax.set_ylabel('Spinor relative error')
        ax.text(-.13,1.04,chr(97+index),transform=ax.transAxes,fontweight='bold',fontsize=11)
        ax.set_ylim(5e-9,5e-5);ax.grid(axis='y',which='major',color='#ececec',lw=.5)
    axes[0,0].legend(fontsize=7,loc='upper right')
    for ext in ('png','svg','pdf'):fig.savefig(out/f'TSFP_error_cost.{ext}',dpi=220)
    plt.close(fig)
    caption=('All108 predefined TSFP2 configurations: six separate cases, Nx256/512/1024, six dt values from0.025 to0.00078125. '
        'Each cost is the median of five complete propagations including initial-state setup, free/interaction factors and FFT work; no neural training is used. '
        'The known free Dirac operator is an explicit prior. Errors include the full FFT spectrum and use a single spacetime constant phase, Nt201/Nx2048 quadrature. '
        'Solid curves use the independent wide reference [-128,160], validated against [-112,144]; dotted curves retain the original161-mode comparison. '
        'The wide check was added after observing the high-precision truncation floor of the mature integrator, using the previously fixed widest candidate pair. '
        'No learning criterion, reference training isolation or frozen PINN design was changed. The two-harmonic primary reference floor is visible. '
        'Costs are observations during concurrent work, not a hardware-isolated neural speedup claim.')
    (out/'TSFP_error_cost_caption.txt').write_text(caption,encoding='utf-8')
    write(out/'TSFP_error_cost_metadata.json',dict(status='GENERATED_REQUIRES_VISUAL_CHECK',script_sha256=sha(__file__),
        source_sha256=sha(source/'TSFP_error_cost_all108.csv'),all108_configurations_shown=True,
        primary_reference_preserved=True,wide_check_timing_explicit=True))

if __name__=='__main__':main()
