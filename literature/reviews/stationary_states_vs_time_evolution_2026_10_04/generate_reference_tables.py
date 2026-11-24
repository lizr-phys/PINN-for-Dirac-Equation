"""Make reference-manager files from verified metadata; retain raw records."""
from __future__ import annotations
import csv
import hashlib
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUTHOR_OVERRIDES = {
    "B03": ["Du, Xiao-Kai", "Zhang, Shuang-Quan"],
    "R03": ["Gao, Wei", "Sun, Ting Ting", "Long, Wen Hui"],
    "S07": ["Awasthi, Ayushi", "Kant, Ishwar", "Sharma, Arushi",
            "Kumar, M. R. Ganesh", "Sastri, O. S. K. S."]
}
def clean(text):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", "", text))).strip().rstrip("*").strip()
def date(value):
    if not value or not value.get("date-parts"):
        return ""
    return "-".join(str(v).zfill(2) for v in value["date-parts"][0])
def bibsafe(text):
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("&", r"\&").replace("%", r"\%")

def main():
    records=json.loads((ROOT/"verified_metadata.json").read_text(encoding="utf-8"))["records"]
    annotations=json.loads((ROOT/"annotations.json").read_text(encoding="utf-8"))
    normalized=[]
    bib=[]
    ris=[]
    table=["# 核实的代表文献", "", "核查日：2026-10-04。34 条期刊/会议 DOI 由 Crossref 核实；另 6 条为已核实的 arXiv 版本，其中一条明确为硕士论文。arXiv 版本不等于已完成同行评议；没有核实期刊记录不等于确认没有发表。", "", "| ID | 年份/状态 | 原始论文与出处 | 相关性 |", "|---|---|---|---|"]
    for r in records:
        c=r.get("crossref",{})
        a=r.get("arxiv_metadata",{})
        title=clean((c.get("title") or a.get("citation_title") or ["UNKNOWN"])[0])
        if r["id"]=="A01":
            title="Über die Quantenmechanik der Elektronen in Kristallgittern"
        authors=AUTHOR_OVERRIDES.get(r["id"])
        if authors is None and c.get("author"):
            authors=[clean(x.get("family",""))+", "+clean(x.get("given","")) for x in c["author"]]
        if authors is None:
            authors=a.get("citation_author",[])
        printdate=date(c.get("published-print"))
        onlinedate=date(c.get("published-online"))
        pubdate=date(c.get("published"))
        year=(printdate or onlinedate or pubdate or (a.get("citation_date") or [""])[0])[:4]
        journal=clean((c.get("container-title") or ["arXiv"])[0])
        if r["id"]=="A02":
            journal="arXiv; Master's thesis"
        url="https://doi.org/"+r["doi"] if r["doi"] else "https://arxiv.org/abs/"+r["arxiv"]
        entry=dict(id=r["id"],topic=r["topic"],title=title,authors=authors,
                   year=year,journal=journal,doi=r["doi"],arxiv=r["arxiv"],url=url,
                   volume=c.get("volume",""),issue=c.get("issue",""),
                   pages=c.get("page") or c.get("article-number") or "",
                   online_date=onlinedate,print_date=printdate,registered_publication_date=pubdate,
                   arxiv_date=(a.get("citation_date") or [""])[0],
                   status="DOI metadata verified" if r["doi"] else "arXiv version verified; journal status not established",
                   note=annotations[r["id"]]["note"],
                   evidence_level=annotations[r["id"]]["evidence_level"])
        normalized.append(entry)
        kind="inproceedings" if c.get("type")=="proceedings-article" else "article" if r["doi"] else "misc"
        fields={"title":title,"author":" and ".join(authors),"year":year,"url":url}
        if r["doi"]:
            fields["doi"]=r["doi"]
            fields["booktitle" if kind=="inproceedings" else "journal"]=journal
        if c.get("volume"): fields["volume"]=c["volume"]
        if c.get("issue"): fields["number"]=c["issue"]
        if entry["pages"]: fields["pages"]=str(entry["pages"]).replace("-","--")
        if r["arxiv"]:
            fields["eprint"]=r["arxiv"]
            fields["archivePrefix"]="arXiv"
        if not r["doi"]: fields["note"]="Master's thesis" if r["id"]=="A02" else "Preprint version; journal status not established by this review"
        bib.append("@"+kind+"{"+r["id"]+",\n"+",\n".join("  "+k+" = {"+bibsafe(str(v))+"}" for k,v in fields.items())+"\n}\n")
        ris_entry=["TY  - "+("CONF" if kind=="inproceedings" else "JOUR" if kind=="article" else "UNPB"),
                   "ID  - "+r["id"],"TI  - "+title]
        ris_entry+=["AU  - "+x for x in authors]
        ris_entry+=["PY  - "+year,"T2  - "+journal,"UR  - "+url]
        if r["doi"]: ris_entry+=["DO  - "+r["doi"]]
        if r["arxiv"]: ris_entry+=["N1  - arXiv:"+r["arxiv"]]
        ris_entry+=["ER  - ",""]
        ris.append("\n".join(ris_entry))
        status=year+("；硕士论文" if r["id"]=="A02" else "；预印本版本" if not r["doi"] else "")
        if r["id"]=="B03": status="2026 卷期；2025-12 在线"
        if r["id"]=="S02": status="2026-09 期刊；2026-05 预印本"
        table.append("| "+r["id"]+" | "+status+" | ["+title.replace("|","/")+"]("+url+")；"+journal+" | "+entry["note"].replace("|","/")+" |")
    (ROOT/"references.json").write_text(json.dumps(normalized,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"references.bib").write_text("\n".join(bib),encoding="utf-8")
    (ROOT/"references.ris").write_text("\n".join(ris),encoding="utf-8")
    with (ROOT/"references.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=normalized[0].keys())
        writer.writeheader()
        writer.writerows({**x,"authors":"; ".join(x["authors"])} for x in normalized)
    table += ["", "元数据原响应保存在 raw/，读取层级见 annotations.json。条目标题去掉出版脚注星号、HTML 格式和多余空格；Bloch 原始标题包含编码替代符，展示为规范德文。Du/Gao 作者中英混排名字按原始出版页面规范化；S07 的五位作者按原 PDF 首页分开，原 arXiv 合并字段仍保留。S06 期刊标题改为 nuclear reactions，与预印本 scattering 归为一篇，不重复统计。", "", "年份优先采用期刊印刷年份，另保留线上、印刷、arXiv 日期。M03 当前 Crossref 卷期为 2026，不沿用旧综述的 2025 年简写；X02 的 arXiv 于 2015 年补存，原 Science 论文为 2013 年。P01 页面 DOI 标注注册待定，因此只提供已核实的 arXiv 链接，不把待注册字符串当成已生效 DOI。"]
    (ROOT/"references.md").write_text("\n".join(table)+"\n",encoding="utf-8")
    print(f"References: {len(records)}; DOI verified: {sum(bool(r['doi']) for r in records)}")
if __name__=="__main__":
    main()

