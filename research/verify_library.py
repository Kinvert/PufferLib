"""Verify downloaded paper provenance and actual retrieval, without training."""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "research"


def main():
    manifest = json.loads((RESEARCH / "papers.json").read_text())
    pages = words = 0
    for paper in manifest:
        slug = paper["slug"]
        print(f"Checking {slug}", flush=True)
        source = RESEARCH / "sources" / slug
        meta = json.loads((source / "metadata.json").read_text())
        if "arxiv" in paper:
            assert paper["arxiv"] == meta["version"], slug
        pdf = (source / "paper.pdf").read_bytes()
        assert pdf.startswith(b"%PDF-"), slug
        assert hashlib.sha256(pdf).hexdigest() == meta["pdf_sha256"], slug
        markdown = (RESEARCH / "papers" / f"{slug}.md").read_text()
        _, separator, body = markdown.partition("\n---\n\n")
        assert separator and body.endswith("\n"), slug
        assert hashlib.sha256(body[:-1].encode()).hexdigest() == meta["markdown_sha256_body"], slug
        assert meta["url"] in markdown[:1600] and meta["markdown_words"] > 600, slug
        pages += meta["pdf_pages"]
        words += meta["markdown_words"]
    print(f"Verified {len(manifest)} PDFs and Markdown bodies against receipts; {pages} pages, {words} converted words", flush=True)

    for document in RESEARCH.glob("*.md"):
        for target in re.findall(r"\]\(([^\s)]+)\)", document.read_text()):
            url = urlsplit(target)
            if url.scheme or not url.path:
                continue
            path = document.parent / unquote(url.path)
            assert path.exists(), (document.name, target)
    print("Verified local links in authored research documents", flush=True)

    cases = [
        ("global average pooling", "hybrid", "paper", "research/papers/impoola.md"),
        ("create_custom_encoder", "fts", "code", "src/algo.cu"),
        ("recurrent state reset on terminal", "vector", "code", "config/default.ini"),
    ]
    for text, mode, kind, expected in cases:
        result = subprocess.run(
            [sys.executable, str(RESEARCH / "search.py"), "query", text,
             "--mode", mode, "--kind", kind, "--limit", "5", "--json"],
            cwd=ROOT, text=True, capture_output=True, check=True, timeout=60,
        )
        rows = json.loads(result.stdout)
        assert any(row["path"] == expected for row in rows), (text, [r["path"] for r in rows])
        for row in rows:
            path = ROOT / row["path"]
            data = path.read_bytes()
            assert hashlib.sha256(data).hexdigest() == row["file_sha256"], row["path"]
            lines = data.decode().splitlines()
            assert 1 <= row["start_line"] <= row["end_line"] <= len(lines), row
            block = "\n".join(lines[row["start_line"] - 1:row["end_line"]])
            assert row["content"] in block, row["path"]
        print(f"Verified {mode} retrieval and source ranges: {text!r}")


if __name__ == "__main__":
    main()
