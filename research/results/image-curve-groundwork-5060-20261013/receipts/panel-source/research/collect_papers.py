"""Download primary papers and create searchable Markdown, retaining provenance.

Run with .venv/bin/python research/collect_papers.py [slug ...].
No LLM conversion: HTML is preferred for mathematics; PDF is the fallback.
"""

import argparse
import hashlib
import importlib.metadata
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify

ROOT = Path(__file__).resolve().parent
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "PufferLib-CNN-local-research/1.0 (paper archival)"


def download(url, path, pdf=False):
    if path.exists():
        data = path.read_bytes()
    else:
        for attempt in range(3):
            response = SESSION.get(url, timeout=(15, 90))
            if response.status_code not in (429, 500, 502, 503, 504):
                break
            time.sleep(4 * (attempt + 1))
        response.raise_for_status()
        data = response.content
        if pdf and not data.startswith(b"%PDF-"):
            raise ValueError(f"Not a PDF: {url}")
        path.write_bytes(data)
    if pdf and not data.startswith(b"%PDF-"):
        raise ValueError(f"Invalid cached PDF: {path}")
    return data


def html_markdown(data, url):
    soup = BeautifulSoup(data, "html.parser")
    article = soup.select_one("article.ltx_document")
    if article is None:
        raise ValueError("No full-text arXiv article in HTML")
    for item in article.select("script, style, nav, .ltx_page_logo"):
        item.decompose()
    equations = {}
    for n, math in enumerate(article.find_all("math")):
        tex = math.get("alttext")
        if not tex:
            annotation = math.find("annotation", encoding="application/x-tex")
            tex = annotation.get_text() if annotation else None
        if tex:
            token = f"MATHPLACEHOLDER{n}END"
            delim = "$$" if math.get("display") == "block" else "$"
            equations[token] = f"{delim}{tex}{delim}"
            math.replace_with(token)
    for item in article.find_all(["a", "img"]):
        attr = "href" if item.name == "a" else "src"
        if item.get(attr):
            item[attr] = urljoin(url, item[attr])
    result = markdownify(str(article), heading_style="ATX")
    for token, equation in equations.items():
        result = result.replace(token, equation)
    return result


def collect(paper):
    slug = paper["slug"]
    source = ROOT / "sources" / slug
    source.mkdir(parents=True, exist_ok=True)
    meta_path = source / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = dict(paper)
        meta["retrieved_utc"] = datetime.now(timezone.utc).isoformat()
        if "arxiv" in paper:
            aid = paper["arxiv"]
            raw = download(f"https://arxiv.org/abs/{aid}", source / "abstract.html")
            soup = BeautifulSoup(raw, "html.parser")
            title = soup.find("meta", attrs={"name": "citation_title"})
            if title is None:
                raise ValueError("Missing arXiv citation metadata")
            versions = re.findall(re.escape(aid) + r"v(\d+)", raw.decode())
            version = f"{aid}v{max(map(int, versions))}" if versions else aid
            meta.update(title=title["content"], version=version,
                        url=f"https://arxiv.org/abs/{version}",
                        pdf=f"https://arxiv.org/pdf/{version}",
                        html=f"https://arxiv.org/html/{version}",
                        authors=[x["content"] for x in soup.find_all("meta", attrs={"name": "citation_author"})])
            license_link = soup.find("a", string=re.compile("view license", re.I))
            if license_link:
                meta["license_url"] = urljoin(meta["url"], license_link["href"])
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    pdf_path = source / "paper.pdf"
    pdf = download(meta["pdf"], pdf_path, pdf=True)
    meta["pdf_sha256"] = hashlib.sha256(pdf).hexdigest()
    import pymupdf
    with pymupdf.open(pdf_path) as doc:
        meta["pdf_pages"] = len(doc)
        meta["pdf_text_words"] = sum(len(page.get_text().split()) for page in doc)

    body = None
    if meta.get("html"):
        try:
            raw = download(meta["html"], source / "paper.html")
            body = html_markdown(raw, meta["html"])
            meta["conversion"] = "arXiv HTML + markdownify; TeX math retained"
            if len(body.split()) < 0.75 * meta["pdf_text_words"]:
                meta["html_fallback_reason"] = "HTML text coverage below 75% of PDF text; possible missing appendices"
                body = None
        except (requests.RequestException, ValueError) as error:
            meta["html_fallback_reason"] = str(error)
    if body is None:
        import pymupdf4llm
        chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, use_ocr=False)
        body = "\n\n".join(f"<!-- PDF page {i + 1} -->\n\n{chunk['text']}" for i, chunk in enumerate(chunks))
        meta["conversion"] = "PyMuPDF4LLM PDF extraction; OCR disabled; page markers retained"
    if len(body.split()) < 600:
        raise ValueError(f"Suspiciously short conversion: {len(body.split())} words")
    meta["conversion_tools"] = {name: importlib.metadata.version(name) for name in ("pymupdf4llm", "markdownify", "beautifulsoup4")}
    meta["markdown_words"] = len(body.split())
    meta["markdown_sha256_body"] = hashlib.sha256(body.encode()).hexdigest()
    header = (f"# {meta['title']}\n\n"
              f"- Source: {meta['url']}\n- PDF: {meta['pdf']}\n"
              f"- Retrieved: {meta['retrieved_utc']}\n- Conversion: {meta['conversion']}\n"
              f"- Original PDF pages: {meta['pdf_pages']}\n- PDF SHA-256: `{meta['pdf_sha256']}`\n\n"
              "> Automatically converted source text, not an assistant summary. "
              "Check the original PDF for equations, tables, plots, and exact quotations. "
              "Figures in HTML remain linked to the source; PDF extraction does not reproduce all figures.\n\n---\n\n")
    (ROOT / "papers" / f"{slug}.md").write_text(header + body + "\n", encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slugs", nargs="*")
    args = parser.parse_args()
    papers = json.loads((ROOT / "papers.json").read_text())
    unknown = set(args.slugs) - {p["slug"] for p in papers}
    if unknown:
        parser.error(f"Unknown slugs: {sorted(unknown)}")
    (ROOT / "papers").mkdir(exist_ok=True)
    state_path = ROOT / "conversion_status.json"
    status = json.loads(state_path.read_text()) if state_path.exists() else {}
    failed = []
    for paper in papers:
        slug = paper["slug"]
        if args.slugs and slug not in args.slugs:
            continue
        try:
            meta = collect(paper)
            status[slug] = {"ok": True, **meta}
            print(f"OK {slug}: {meta['pdf_pages']} pages, {meta['markdown_words']} words ({meta['conversion']})", flush=True)
        except Exception as error:
            status[slug] = {"ok": False, "error": str(error)}
            failed.append(slug)
            print(f"FAILED {slug}: {error}", flush=True)
        state_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
        time.sleep(3)  # Gentle sequential requests to arXiv.
    if failed:
        raise SystemExit(f"Failed papers: {', '.join(failed)}")


if __name__ == "__main__":
    main()
