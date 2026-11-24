"""Public original-source retrieval; all remote content is data, never code."""
from __future__ import annotations
import concurrent.futures,json,re,time,urllib.request,urllib.parse,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone
from plan import BASE,write,sha

OUT=BASE/'literature';OUT.mkdir(exist_ok=True)
UA='R14-Dirac-Prewriting-Evidence/1.0 (academic metadata audit)'
DOIS=['10.1016/j.cpc.2022.108474','10.1016/j.cam.2025.117178','10.1016/j.cam.2026.117607',
 '10.1016/j.physd.2026.135394','10.1016/j.cnsns.2026.110653','10.1088/1674-1137/acc518']
ARXIV=['2204.02959','2408.16414','2006.09661','2006.10739','2112.03616','1711.07193','2402.07251','2607.15087']
QUERIES=[
 ('dirac_all','all:Dirac AND (all:PINN OR all:"physics-informed")'),
 ('dirac_recent','(all:Dirac AND (all:PINN OR all:"physics-informed" OR all:neural)) AND submittedDate:[202609300000 TO 202610112359]'),
 ('phase_recent','((all:Dirac OR all:Schrodinger OR all:Schrödinger) AND (all:PINN OR all:"physics-informed") AND (all:phase OR all:amplitude OR all:spectral)) AND submittedDate:[202609300000 TO 202610112359]'),
 ('interaction','(all:"interaction picture" AND (all:neural OR all:PINN OR all:"physics-informed"))'),
 ('norm','((all:Schrodinger OR all:Dirac) AND (all:PINN OR all:"physics-informed") AND (all:norm OR all:normalization OR all:conservation))')]

def fetch(tag,url,ext='json',timeout=50):
    target=OUT/'sources'/f'{tag}.{ext}';target.parent.mkdir(exist_ok=True)
    meta=target.with_suffix(target.suffix+'.metadata.json')
    if meta.exists() and target.exists():return json.loads(meta.read_text(encoding='utf-8'))
    row=dict(tag=tag,url=url,retrieved_utc=datetime.now(timezone.utc).isoformat(),status='ERROR')
    for attempt in range(2):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':UA}),timeout=timeout) as r:
                data=r.read();row.update(status='OK',HTTP_status=r.status,content_type=r.headers.get('Content-Type'),bytes=len(data))
            target.write_bytes(data);row.update(path=str(target.relative_to(BASE)),sha256=sha(target));break
        except Exception as e:
            row.update(error=repr(e),attempts=attempt+1)
            if attempt==0:time.sleep(2)
    write(meta,row);print('SOURCE',tag,row['status'],flush=True);return row

def main():
    jobs=[]
    for doi in DOIS:jobs.append((doi.replace('/','_'),'https://api.crossref.org/works/'+urllib.parse.quote(doi,safe=''),'json'))
    for tag,q in QUERIES:
        url='https://export.arxiv.org/api/query?'+urllib.parse.urlencode(dict(search_query=q,start=0,max_results=100,sortBy='lastUpdatedDate',sortOrder='descending'))
        jobs.append((tag,url,'xml'))
    for topic,q in [('crossref_dirac','Dirac physics informed neural'),('crossref_spectral','spectral physics informed neural'),
        ('crossref_phase','WKB phase amplitude physics informed'),('crossref_norm','Schrodinger norm constraints neural'),('crossref_interaction','neural interaction picture')]:
        url='https://api.crossref.org/works?'+urllib.parse.urlencode({'query':q,'rows':20,'filter':'from-pub-date:2026-09-30,until-pub-date:2026-10-11'})
        jobs.append((topic,url,'json'))
    jobs += [('arxiv_known','https://export.arxiv.org/api/query?'+urllib.parse.urlencode({'id_list':','.join(ARXIV),'max_results':30}),'xml'),
        ('hult_interaction','https://api.crossref.org/works?'+urllib.parse.urlencode({'query.title':'A Fourth-Order Runge-Kutta in the Interaction Picture Method for Simulating Supercontinuum Generation in Optical Fibers','rows':3}),'json')]
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(fetch,*job) for job in jobs]
        for f in futures:results.append(f.result())
    write(OUT/'retrieval_log.json',dict(retrieved_utc=datetime.now(timezone.utc).isoformat(),queries=QUERIES,results=results,
        academic_MCP_unavailable=True,primary_APIs=['arXiv','CrossRef'],no_Scopus_WoS_access=True,
        bounded_queries_not_systematic_priority_proof=True))
    records=[]
    for row in results:
        if row['status']!='OK':continue
        path=BASE/row['path']
        if path.suffix=='.xml':
            root=ET.fromstring(path.read_bytes());ns={'a':'http://www.w3.org/2005/Atom','o':'http://a9.com/-/spec/opensearch/1.1/'}
            for ent in root.findall('a:entry',ns):
                records.append(dict(source=row['tag'],title=' '.join(ent.findtext('a:title','',ns).split()),
                    arxiv=ent.findtext('a:id','',ns),published=ent.findtext('a:published','',ns),updated=ent.findtext('a:updated','',ns),
                    authors=[a.findtext('a:name','',ns) for a in ent.findall('a:author',ns)],abstract=' '.join(ent.findtext('a:summary','',ns).split()),
                    DOI=ent.findtext('{http://arxiv.org/schemas/atom}doi',''),url=ent.findtext('a:id','',ns)))
        else:
            obj=json.loads(path.read_text(encoding='utf-8'))['message'];items=obj.get('items',[obj])
            for ent in items:
                records.append(dict(source=row['tag'],title=' '.join(ent.get('title',[])),DOI=ent.get('DOI'),
                    authors=[(a.get('given','')+' '+a.get('family','')).strip() for a in ent.get('author',[])],
                    journal=' '.join(ent.get('container-title',[])),dates={k:ent.get(k) for k in ('published-online','published-print','published','created','indexed')},
                    abstract=re.sub('<[^>]+>',' ',ent.get('abstract','')),url=ent.get('URL')))
    write(OUT/'candidate_metadata.json',records)
    # PDFs supplement full method checks; failures remain explicit.
    pdfs=[]
    for arxiv in ('2204.02959','2408.16414','2006.09661','2006.10739','2112.03616','2402.07251'):
        pdfs.append(fetch('paper_'+arxiv,'https://arxiv.org/pdf/'+arxiv,'pdf'))
    pdfs.append(fetch('hult_fulltext','https://laser.ceb.cam.ac.uk/system/files/documents/pdf-a-fourth-order-runge-kutta-in-the-interaction-picture-method-for-simulating-supercontinuum-generation-in-optical-fibres.pdf','pdf'))
    import fitz
    for row in pdfs:
        if row['status']=='OK':
            try:
                path=BASE/row['path'];doc=fitz.open(path)
                text='\n'.join(f'\n--- PAGE {i+1} ---\n'+page.get_text() for i,page in enumerate(doc))
                path.with_suffix('.txt').write_text(text,encoding='utf-8')
                row.update(pages=len(doc),text_path=str(path.with_suffix('.txt').relative_to(BASE)))
            except Exception as e:row['PDF_parse_error']=repr(e)
    write(OUT/'fulltext_retrieval_log.json',pdfs)

if __name__=='__main__':main()
