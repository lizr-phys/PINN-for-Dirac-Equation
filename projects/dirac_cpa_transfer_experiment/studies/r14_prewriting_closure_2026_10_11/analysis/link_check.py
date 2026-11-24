"""Verify existing evidence links; HTTP access limits remain distinct from broken links."""
from __future__ import annotations
import csv,json,re,time,hashlib
from pathlib import Path
from urllib.parse import urlparse,unquote,quote
from concurrent.futures import ThreadPoolExecutor
import requests
from plan import BASE,write,sha

HEADERS={'User-Agent':'R14-prewriting-evidence-link-verification/1.0'}
SCRIPT_SHA_AT_IMPORT=sha(__file__)

def external(url):
    record=dict(url=url,checked_utc=time.time())
    try:
        response=requests.get(url,headers=HEADERS,timeout=16,allow_redirects=True,stream=True)
        record.update(http_status=response.status_code,final_url=response.url,content_type=response.headers.get('Content-Type'))
        response.close()
        if 200<=record['http_status']<400:record['status']='RESOLVES'
        elif record['http_status'] in (401,403,429):record['status']='ACCESS_LIMITED'
        else:record['status']='HTTP_ERROR_REQUIRES_REVIEW'
    except Exception as error:record.update(status='NETWORK_ERROR_REQUIRES_REVIEW',error=repr(error))
    doi=None
    if 'doi.org/' in url:doi=unquote(url.split('doi.org/',1)[1])
    elif '/doi/' in url and re.search(r'/doi/(10\.[^?#]+)',url):doi=unquote(re.search(r'/doi/(10\.[^?#]+)',url).group(1))
    if doi:
        try:
            meta=requests.get('https://api.crossref.org/works/'+quote(doi,safe=''),headers=HEADERS,timeout=16)
            record['CrossRef_status']=meta.status_code
            if meta.ok:
                data=meta.json()['message'];record.update(DOI_metadata_verified=True,DOI_title=data.get('title'),DOI_metadata_url=data.get('URL'))
                if record['status']!='RESOLVES':record['status']='DOI_METADATA_VERIFIED_ACCESS_LIMITED'
            else:record['DOI_metadata_verified']=False
        except Exception as error:record['DOI_metadata_error']=repr(error)
    return record

def main():
    urls={};local=[];structured=[];structured_errors=[];input_sources=[]
    def content(path):
        data=path.read_bytes();input_sources.append(dict(path=path.relative_to(BASE).as_posix(),sha256=hashlib.sha256(data).hexdigest()))
        return data.decode('utf-8-sig')
    def artifact(document,label,target,expected_sha=None,expected_bytes=None):
        path=Path(unquote(target));path=path if path.is_absolute() else BASE/path
        record=dict(document=document,label=label,target=target,exists=path.is_file(),resolved=str(path.resolve()))
        if record['exists']:
            if expected_sha is not None:
                record.update(expected_sha256=expected_sha,current_sha256=sha(path))
                if record['current_sha256']!=expected_sha:structured_errors.append(dict(document=document,target=target,error='SHA256_MISMATCH'))
            if expected_bytes is not None:
                record.update(expected_bytes=expected_bytes,current_bytes=path.stat().st_size)
                if record['current_bytes']!=expected_bytes:structured_errors.append(dict(document=document,target=target,error='SIZE_MISMATCH'))
        local.append(record)
    documents=sorted(BASE.glob('*.md'))+[BASE/'literature/report.md']
    for document in documents:
        if not document.exists():continue
        text=content(document)
        for label,target in re.findall(r'\[([^\]]*)\]\(([^\n]+?)\)',text):
            target=target.strip().strip('<>')
            if target.startswith(('https://','http://')):urls.setdefault(target,[]).append(document.relative_to(BASE).as_posix())
            elif not target.startswith('#'):
                filepart=target.split('#',1)[0]
                # Markdown may include a :line suffix on a real absolute file link.
                filepart=re.sub(r':\d+$','',filepart)
                path=Path(unquote(filepart));path=path if path.is_absolute() else document.parent/path
                local.append(dict(document=document.relative_to(BASE).as_posix(),label=label,target=target,exists=path.exists(),resolved=str(path.resolve())))
    prior=BASE/'literature/prior_art_map.csv'
    for row in csv.DictReader(content(prior).splitlines()):
        for field in ('original_source','DOI'):
            value=row.get(field,'')
            if value.startswith('10.'):value='https://doi.org/'+value
            if value.startswith(('http://','https://')):urls.setdefault(value,[]).append(prior.relative_to(BASE).as_posix())
    bib=BASE/'literature/references.bib'
    for url in re.findall(r'url\s*=\s*\{([^}]+)\}',content(bib)):urls.setdefault(url,[]).append(bib.relative_to(BASE).as_posix())
    claims=BASE/'claims_evidence.csv'
    if claims.exists():
        try:
            import io
            reader=csv.DictReader(io.StringIO(content(claims)));assert {'claim_id','evidence_paths'}<=set(reader.fieldnames or []),'Claim identifiers and JSON evidence path arrays are required'
            rows=list(reader)
            assert rows,'Empty claim table';identifiers=set()
            for row in rows:
                identifier=row['claim_id'];assert identifier and identifier not in identifiers,'Empty or repeated claim identifier';identifiers.add(identifier)
                paths=json.loads(row['evidence_paths']);assert isinstance(paths,list) and paths and all(isinstance(p,str) and p for p in paths),'Nonempty JSON evidence path array required'
                for target in paths:
                    if target.startswith(('http://','https://')):urls.setdefault(target,[]).append('claims_evidence.csv')
                    else:artifact('claims_evidence.csv',identifier,target)
            structured.append(dict(path='claims_evidence.csv',sha256=sha(claims),rows=len(rows),status='PARSED_AND_REFERENCES_CHECKED'))
        except Exception as error:structured_errors.append(dict(document='claims_evidence.csv',error=repr(error)))
    manifest=BASE/'code_data_manifest.json'
    if manifest.exists():
        try:
            inventory=json.loads(content(manifest));files=inventory['files'];assert isinstance(files,list) and files,'Nonempty immutable file inventory required'
            paths=set()
            for row in files:
                assert row['path'] not in paths,'Duplicate manifest path';paths.add(row['path'])
                assert re.fullmatch('[0-9a-f]{64}',row['sha256']) and isinstance(row['bytes'],int),'SHA256 and byte count required'
                artifact('code_data_manifest.json',row.get('category','file'),row['path'],row['sha256'],row['bytes'])
            for row in inventory.get('linked_inventories',[]):
                assert re.fullmatch('[0-9a-f]{64}',row['sha256']),'Linked inventory SHA256 required'
                artifact('code_data_manifest.json','linked inventory',row['path'],row['sha256'])
            structured.append(dict(path='code_data_manifest.json',sha256=sha(manifest),files=len(files),status='PARSED_AND_FILES_HASHED'))
        except Exception as error:structured_errors.append(dict(document='code_data_manifest.json',error=repr(error)))
    with ThreadPoolExecutor(max_workers=3) as pool:external_rows=list(pool.map(external,sorted(urls)))
    for r in external_rows:r['citing_documents']=sorted(set(urls[r['url']]))
    changed_inputs=[r['path'] for r in input_sources if not (BASE/r['path']).exists() or sha(BASE/r['path'])!=r['sha256']]
    if SCRIPT_SHA_AT_IMPORT!=sha(__file__):changed_inputs.append('analysis/link_check.py')
    if changed_inputs:structured_errors.append(dict(error='INPUT_CHANGED_DURING_CHECK_REQUIRES_REPEAT',paths=changed_inputs))
    missing=[r for r in local if not r['exists']];unresolved=[r for r in external_rows if 'REQUIRES_REVIEW' in r['status']]
    pending=[name for name in ('total_report.md','claims_evidence.csv','code_data_manifest.json','submission_positioning.md') if not (BASE/name).exists()]
    ready=not pending and not missing and not unresolved and not structured_errors and len(structured)==2
    report=dict(updated_utc=time.time(),status='PASS_EXISTING_LINKS_WITH_ACCESS_LIMITS' if not missing and not unresolved and not structured_errors else 'REQUIRES_REVIEW',
        local_links=len(local),external_links=len(external_rows),missing_local_links=missing,unresolved_external_links=unresolved,
        access_limited_links=[r['url'] for r in external_rows if 'ACCESS_LIMITED' in r['status']],
        pending_final_documents=pending,whole_delivery_link_check_complete=ready,
        structured_documents=structured,structured_document_errors=structured_errors,
        link_completeness_is_not_whole_scientific_acceptance=True,
        access_limit_does_not_confirm_fulltext_reading=True,documents=input_sources,link_check_code_sha256=SCRIPT_SHA_AT_IMPORT,
        input_sources_unchanged_during_check=not changed_inputs,
        local=local,external=external_rows)
    write(BASE/'link_check.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('local','external','documents')},ensure_ascii=False))

if __name__=='__main__':main()
