from __future__ import annotations

import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
import fitz
from .r13_config import ROOT,FIGURES
from .freeze_r7_phase_release import write_json,sha256

def qa():
    registry=json.loads((FIGURES/'figure_registry.json').read_text(encoding='utf-8'))
    renders=FIGURES/'qa_renders';renders.mkdir(exist_ok=True);records=[];tiles=[]
    for record in registry['figures']:
        png=ROOT/record['exports']['png']['path'];pdf=ROOT/record['exports']['pdf']['path']
        image=Image.open(png).convert('RGB')
        with fitz.open(pdf) as document:
            page=document[0];target=renders/(record['name']+'_pdf.png')
            page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(target)
            pdf_image=Image.open(target).convert('RGB')
            blocks=page.get_text('dict')['blocks'];outside=[]
            for block in blocks:
                for line in block.get('lines',[]):
                    for span in line.get('spans',[]):
                        box=fitz.Rect(span['bbox'])
                        if not page.rect.contains(box):outside.append(span['text'])
            text=page.get_text();pages=len(document)
        source=ROOT/record['source_csv'];frame=pd.read_csv(source)
        numeric=frame.select_dtypes(include='number')
        finite=bool(np.isfinite(numeric.dropna(axis=1).to_numpy()).all())
        result=dict(name=record['name'],png_pixels=list(image.size),pdf_pixels=list(pdf_image.size),
            pdf_pages=pages,pdf_has_editable_text=len(text)>10,text_outside_pdf=outside,
            source_rows=len(frame),source_numeric_finite=finite,
            source_hash_matches=sha256(source)==record['source_sha256'],
            exported_hashes_match=all(sha256(ROOT/e['path'])==e['sha256'] for e in record['exports'].values()),
            pdf_render_path=target.relative_to(ROOT).as_posix(),pdf_render_sha256=sha256(target))
        result['automated_pass']=bool(result['source_hash_matches'] and result['exported_hashes_match'] and finite and
            result['pdf_has_editable_text'] and pages==1 and not outside and min(image.size)>=600)
        records.append(result)
        for source_image,label in ((image,record['name']+' PNG'),(pdf_image,record['name']+' PDF')):
            source_image.thumbnail((720,480));tile=Image.new('RGB',(760,520),'white')
            tile.paste(source_image,((760-source_image.width)//2,30));ImageDraw.Draw(tile).text((12,8),label,fill='black');tiles.append(tile)
    sheet=Image.new('RGB',(1520,520*int(np.ceil(len(tiles)/2))),'white')
    for index,tile in enumerate(tiles):sheet.paste(tile,((index%2)*760,(index//2)*520))
    sheet.save(renders/'contact_sheet.png')
    visual=FIGURES/'FIGURE_QA_VISUAL_REVIEW.json';review=json.loads(visual.read_text(encoding='utf-8')) if visual.exists() else None
    visual_pass=bool(review and review.get('status')=='PASS' and
        all(review.get('export_hashes',{}).get(e['path'])==e['sha256'] for r in registry['figures'] for e in r['exports'].values()))
    result=dict(status='PASS' if all(r['automated_pass'] for r in records) and visual_pass else
        ('AUTOMATED_PASS_VISUAL_REVIEW_PENDING' if all(r['automated_pass'] for r in records) else 'FAIL'),
        automated_checks=records,actual_pdf_rendered=True,visual_review_pass=visual_pass,
        no_visual_review_fabricated=True,checked_utc=datetime.now(timezone.utc).isoformat())
    write_json(FIGURES/'FIGURE_QA.json',result);return result

if __name__=='__main__':print(json.dumps(qa(),indent=2))
