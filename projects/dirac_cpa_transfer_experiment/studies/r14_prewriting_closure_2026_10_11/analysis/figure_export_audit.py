"""Render vector exports for additional visual inspection; no automatic visual PASS."""
import json
import xml.etree.ElementTree as ET
import fitz
from plan import BASE,write,sha

def main():
    names=tuple(p.stem for p in sorted((BASE/'figures').glob('*.pdf')));rows=[];output=BASE/'figures/pdf_render';output.mkdir(exist_ok=True)
    for name in names:
        pdf=BASE/'figures'/f'{name}.pdf';svg=BASE/'figures'/f'{name}.svg';png=output/f'{name}_PDF.png'
        with fitz.open(pdf) as document:
            assert len(document)==1
            page=document[0];page.get_pixmap(matrix=fitz.Matrix(1.6,1.6),alpha=False).save(png)
            boxes=[span['bbox'] for block in page.get_text('dict')['blocks'] if 'lines' in block for line in block['lines'] for span in line['spans']]
            inside=all(x0>=-.1 and y0>=-.1 and x1<=page.rect.width+.1 and y1<=page.rect.height+.1 for x0,y0,x1,y1 in boxes)
            fonts=[entry[3] for entry in page.get_fonts()]
        tree=ET.parse(svg);texts=len(list(tree.iter('{http://www.w3.org/2000/svg}text')))
        rows.append(dict(figure=name,pdf_sha256=sha(pdf),svg_sha256=sha(svg),rendered_PDF_png=png.relative_to(BASE).as_posix(),
            rendered_PDF_png_sha256=sha(png),pdf_fonts=fonts,pdf_text_boxes_inside_page=inside,svg_editable_text_nodes=texts,
            automatic_export_checks_pass=inside and texts>0,actual_visual_check_required=True))
    write(BASE/'figures/vector_export_checks.json',rows);print(json.dumps(rows,ensure_ascii=False))

if __name__=='__main__':main()
