from __future__ import annotations
import csv,json,re
from plan import BASE,write,sha
L=BASE/'literature'
records=json.loads((L/'candidate_metadata.json').read_text(encoding='utf-8'))
original_candidate_count=len(records)
followup=L/'additional_candidate_metadata_20261011.json'
if followup.exists():records+=json.loads(followup.read_text(encoding='utf-8'))
rows=[
 ('Dirac PINN','Lorin and Yang (2022)','10.1016/j.cpc.2022.108474','https://arxiv.org/abs/2204.02959','Full paper, sections 1-2, pp1-6','时变Dirac旋量PINN、硬初值及L2范数约束已存在','本项目检验能支相位对齐、动态空通道切向及相干误差；不能称首次Dirac PINN'),
 ('Spectral PINN','Yu et al. (arXiv 2024; journal 2026)','10.1016/j.cam.2025.117178','https://arxiv.org/html/2408.16414v2','Full paper, Fig2 and section3, pp3-5','以(k,t)为输入、输出谱系数、谱乘法代替空间微分已有直接先例','本项目共享谱形式不是原创；贡献须落在旋量能支、耦合缺陷和验证'),
 ('Conservative phase-amplitude','Kumar, Dwivedi and Rajeev (2026)','10.1016/j.cam.2026.117607','https://www.sciencedirect.com/science/article/pii/S0377042726002505','Publisher indexed method/abstract and CrossRef; direct fulltext403','相位—幅度与守恒目标的CPA路线已有先例，也是原迁移来源','不能将相位分解、守恒约束单独列为新发明；未取得全文部分不排除细节重合'),
 ('WKB phase-envelope','Cen et al. (2026)','10.1016/j.physd.2026.135394','https://www.sciencedirect.com/science/article/pii/S0167278926002903','Publisher indexed abstract/method; direct fulltext403','高振荡Schrodinger型问题的相位表示及eikonal辅助已有邻近路线','与多能支Dirac及空通道缺陷对象不同；不能仅凭摘要排除完整方法重合'),
 ('WKB complex envelope','Kumar et al. (2026)','10.1016/j.cnsns.2026.110653','https://www.sciencedirect.com/science/article/abs/pii/S1007570426010075','Publisher abstract and CrossRef; fulltext unavailable','快速实相位与复包络、耦合代理和强形式残差已有直接相邻先例','本项目最终复包络不是原纯实幅度LTPB；部件创新应收窄'),
 ('Interaction picture','Hult (2007)','10.1109/jlt.2007.909373','https://opg.optica.org/jlt/abstract.cfm?uri=JLT-25-12-3770','Full author PDF, sectionII-D, p3','通过可解自由传播因子变换到相互作用绘景属于成熟计算路线','解析载波对照数学上与已知自由H0的相互作用绘景相关；学习近似载波需保留缺陷，不能主张首次'),
 ('Quantum normalization: CDF structure plus soft endpoints','Pu et al. (2023)','10.1088/1674-1137/acc518','https://arxiv.org/abs/2303.03934','Full arXiv and published PDF; sectionII, equations1-4; journal pp054104-2/3 visually checked','单调CDF结构产生非负实基态波函数；式4用CDF端点软损失控制归一','该文不是本项目全复谱硬球面投影，也不是时变复旋量；总积分等于CDF端点差，端点软项不能冒称任意权重下精确单位范数；一般结构化归一概念仍有先例'),
 ('Hard projection constraints','Chen, Constante Flores and Li (2024)','','https://arxiv.org/abs/2402.07251','Full paper, abstract and section3','KKT投影层严格实施线性等式约束','本项目是复谱二次球面范数；不可将此先例误称完全同一算法，也不宣称结构约束概念首创'),
 ('SIREN','Sitzmann et al. (2020)','','https://arxiv.org/abs/2006.09661','Full paper, sections2-3, initialization','正弦激活与专门初始化支持高频信号和导数表示','本轮谱PINN适配基线，不是逐字复现原论文的求解器'),
 ('Fourier features','Tancik et al. (2020)','','https://arxiv.org/abs/2006.10739','Full paper, section3','随机Fourier特征改变MLP的高频表示与核带宽','本轮固定小规模开发调参；不把tanh塌缩当最强神经能力'),
 ('Dirac TSFP','Bao, Feng and Yin (2021)','','https://arxiv.org/abs/2112.03616','Full paper, method and numerical analysis','Dirac二阶时间分裂与Fourier伪谱有成熟误差分析','本轮TSFP2是数值性能基准，包含已知自由算符先验，不算神经学习成果'),
 ('Complex NN neighbor','Barman et al. (2026)','','https://arxiv.org/abs/2607.15087','Original arXiv abstract; not reproduced','复权重及实虚分裂激活的PINN涵盖振荡PDE','与本项目实权重四输出复包络不同；此类未复现，强基线结论限于已测试两类'),
 ('Phase recovery by quadrature and wave ratio','Khairullin and Rykovanov (SSRN2025; journal2026)','10.1016/j.cnsns.2026.109926','https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5901547','Original author SSRN abstract and CrossRef; method fulltext not obtained; not reproduced','前/后向波幅比值、透明边界硬结构、相位积分恢复已有振荡PINN相邻路线','Helmholtz空间散射与Dirac时变能支对象不同；不能称首次以结构分离快相位；未作为冻结性能基准，不用报道的速度比替代本轮实测')]
with (L/'prior_art_map.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['topic','prior_work','DOI','original_source','reading_depth','overlap','distinction_and_claim_limit']);w.writerows(rows)
# Bibliography from retrieved originals, not invented author/venue metadata.
chosen=[]
for r in records:
    if r['source'].startswith('10.') or r['source']=='arxiv_known' or (r['source']=='hult_interaction' and r.get('DOI')=='10.1109/jlt.2007.909373'):
        if r.get('DOI') and any(x.get('DOI','').lower()==r['DOI'].lower() for x in chosen):continue
        if any(x['title'].lower()==r['title'].lower() for x in chosen):continue
        # Prefer publisher journal metadata to the preprint when titles differ but DOI matches.
        chosen.append(r)
def escape(s):return str(s).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('{','').replace('}','')
bib=[]
for i,r in enumerate(chosen):
    dates=r.get('dates') or {};parts=(dates.get('published') or {}).get('date-parts',[[None]])[0];year=parts[0] if parts and parts[0] else (r.get('published','')[:4] or '2026')
    fields={'title':r['title'],'author':' and '.join(r['authors']),'year':year,'url':r.get('url','')}
    if r.get('journal'):fields['journal']=r['journal']
    if r.get('DOI'):fields['doi']=r['DOI']
    if r.get('arxiv'):fields['eprint']=r['arxiv'].split('/')[-1].split('v')[0];fields['archivePrefix']='arXiv'
    if r.get('DOI')=='10.1088/1674-1137/acc518':fields.update(eprint='2303.03934',archivePrefix='arXiv')
    typ='article' if r.get('journal') else 'misc'
    bib.append('@'+typ+'{r14ref'+str(i+1)+',\n'+',\n'.join('  '+k+' = {'+escape(v)+'}' for k,v in fields.items())+'\n}')
(L/'references.bib').write_text('\n\n'.join(bib)+'\n',encoding='utf-8')
# Every retrieved candidate is screened, including irrelevant Dirac-structure hits.
screen=[]
for r in records:
    title=r['title'].lower();text=title+' '+r.get('abstract','').lower()
    if r['source'].startswith('10.') or r['source']=='arxiv_known' or (r['source']=='hult_interaction' and 'optical fibers' in title):label='KEY_PRIOR';reason='原始先例或预定方法来源'
    elif 'dirac-interconnected' in title or 'hamilton-dirac' in title:label='EXCLUDE';reason='Dirac几何/约束结构，非量子Dirac旋量演化'
    elif 'ntk positivity' in title:label='ADJACENT';reason='PINN线性约束理论；Dirac-data指delta数据，不是本方程方法'
    elif 'sinfonia' in title:label='ADJACENT';reason='引力轨道交互绘景神经流；对象与训练任务不同，仅摘要核对'
    elif any(k in text for k in ('schrödinger','schrodinger','spectral pinn','dirac equation','phase-amplitude')):label='CANDIDATE';reason='对象或方法相关，需原文逐项核查，不能据关键词认定重复'
    else:label='EXCLUDE';reason='宽查询发现项，未匹配本课题对象与方法'
    screen.append(dict(title=r['title'],DOI=r.get('DOI'),url=r.get('url'),source=r['source'],decision=label,reason=reason))
import pandas as pd
pd.DataFrame(screen).to_csv(L/'screening_all_candidates.csv',index=False,encoding='utf-8-sig')
write(L/'materials_status.json',{'status':'METHOD_MAP_PREPARED_FINAL_POSITIONING_PENDING_RESULTS','candidate_records':len(records),
    'original_candidate_records':original_candidate_count,'additional_targeted_records':len(records)-original_candidate_count,
    'prior_map_rows':len(rows),'bibliography_records':len(chosen),'full_papers_retrieved':8,'full_PDF_files':9,
    'priority_proof_claim':False,'all_candidates_screened':True,'publisher_fulltext_limitations_disclosed':True})
print('LITERATURE_MAP',len(rows),'BIB',len(chosen))
