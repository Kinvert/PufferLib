"""Local paper/code retrieval. Build once; queries use cached CPU embeddings.

Commands: build, status, query TEXT [--kind paper|code|note] [--mode hybrid|fts|vector].
Returned line ranges refer to real source files. Always inspect those files.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Keep retrieval from consuming all cores or using training GPUs.
os.environ.setdefault("LANCE_CPU_THREADS", "2")
os.environ.setdefault("LANCE_IO_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "research"
STATE = RESEARCH / "index_state.json"
MODEL = "snowflake/snowflake-arctic-embed-xs"
EMBED_CONFIG = f"{MODEL}:max256:strip-links:v1"
SUFFIXES = {".c", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".py", ".sh", ".ini", ".md", ".toml"}
EXCLUDE = {"vendor", "resources", ".git", ".venv"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def corpus():
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).decode().split("\0")
    paths = {ROOT / name for name in names if name}
    paths.update((RESEARCH / "papers").glob("*.md"))
    files, skipped = {}, []
    for path in sorted(paths):
        rel = path.relative_to(ROOT)
        if rel.parts[0] in EXCLUDE or path.suffix not in SUFFIXES:
            continue
        if not path.is_file() or path.is_symlink():
            continue
        if path.stat().st_size > 2_000_000:
            skipped.append(str(rel))
            continue
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            skipped.append(str(rel))
            continue
        files[str(rel)] = (digest(data), text)
    return files, skipped


def chunks(path, content):
    lines = content.splitlines()
    kind = "paper" if path.startswith("research/papers/") else "note" if path.endswith(".md") else "code"
    source = ""
    if kind == "paper":
        source = next((line.removeprefix("- Source: ") for line in lines[:20] if line.startswith("- Source: ")), "")
    heading = ""
    i = 0
    while i < len(lines):
        start = i
        block = []
        size = 0
        while i < len(lines) and len(block) < 36:
            if block and size + len(lines[i]) > 1400:
                break
            if lines[i].startswith("#") and path.endswith(".md"):
                heading = lines[i].strip("# ")
            block.append(lines[i])
            size += len(lines[i]) + 1
            i += 1
        text = "\n".join(block).strip()
        # Very long paragraphs/table rows are split without losing their source line.
        for offset in range(0, len(text), 1400):
            part = text[offset:offset + 1400]
            if not part.strip():
                continue
            identity = f"{path}:{start}:{offset}:{part}"
            if heading:
                identity += f":heading={heading[:160]}"
            yield {"id": digest(identity.encode()),
                   "path": path, "start_line": start + 1, "end_line": i,
                   "kind": kind, "heading": heading[:160], "source_url": source,
                   "text": f"{path}\n{heading[:160]}\n{part}", "content": part}
        if i < len(lines) and i - start > 5:
            i -= 5


def embedder(download=False):
    import onnxruntime
    onnxruntime.disable_telemetry_events()
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name=MODEL, cache_dir=str(RESEARCH / ".models"),
                         threads=2, providers=["CPUExecutionProvider"],
                         local_files_only=not download)
    # Search previews retain the entire chunk. Only semantic encoding is capped.
    model.model.tokenizer.enable_truncation(max_length=256)
    return model


def semantic_text(text):
    return re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)


def check_state(state, files):
    current = {p: value[0] for p, value in files.items()}
    previous = state.get("files", {})
    changed = [p for p in current if previous.get(p) != current[p]]
    removed = sorted(set(previous) - set(current))
    return changed, removed


def build(args):
    import lancedb
    import pyarrow as pa
    files, skipped = corpus()
    old = json.loads(STATE.read_text()) if STATE.exists() else {}
    changed, removed = check_state(old, files)
    if not changed and not removed and old.get("embedding_config") == EMBED_CONFIG and not args.rebuild:
        print(f"Index current: {old['chunks']} chunks from {len(files)} files")
        return
    rows = [row for path, (_, content) in files.items() for row in chunks(path, content)]
    print(f"Indexing {len(rows)} chunks from {len(files)} files; {len(skipped)} oversized/non-UTF8 files skipped", flush=True)
    db = lancedb.connect(str(RESEARCH / ".lancedb"))
    # Reuse unchanged chunk vectors; content hashes make stale reuse impossible.
    vectors = {}
    cache_name = "vectors_" + digest(EMBED_CONFIG.encode())[:12]
    cache_schema = pa.schema([("id", pa.string()), ("vector", pa.list_(pa.float32(), 384))])
    cache = db.create_table(cache_name, schema=cache_schema, exist_ok=True)
    if not args.rebuild:
        for row in cache.to_arrow().to_pylist():
            vectors[row["id"]] = row["vector"]
    pending = [r for r in rows if r["id"] not in vectors]
    if pending:
        model = embedder(download=True)
        for start in range(0, len(pending), 128):
            batch = pending[start:start + 128]
            saved = []
            for row, vector in zip(batch, model.passage_embed([semantic_text(r["text"]) for r in batch], batch_size=16), strict=True):
                vectors[row["id"]] = vector.tolist()
                saved.append({"id": row["id"], "vector": vector.tolist()})
            cache.add(saved)  # Checkpoint each batch so interrupted work is reusable.
            print(f"Embedded {min(start + 128, len(pending))}/{len(pending)} new chunks", flush=True)
    for row in rows:
        row["vector"] = vectors[row["id"]]
        row["file_sha256"] = files[row["path"]][0]
    schema = pa.schema([
        ("id", pa.string()), ("path", pa.string()), ("start_line", pa.int32()),
        ("end_line", pa.int32()), ("kind", pa.string()), ("heading", pa.string()),
        ("source_url", pa.string()), ("text", pa.string()), ("content", pa.string()),
        ("vector", pa.list_(pa.float32(), 384)), ("file_sha256", pa.string()),
    ])
    # Alternate tables so an interrupted build leaves the last complete index usable.
    name = "chunks_b" if old.get("table") == "chunks_a" else "chunks_a"
    table = db.create_table(name, data=rows, schema=schema, mode="overwrite")
    from lancedb.index import FTS
    table.create_index("text", config=FTS())
    state = {"built_utc": datetime.now(timezone.utc).isoformat(), "model": MODEL,
             "embedding_config": EMBED_CONFIG,
             "table": name, "chunks": len(rows), "files": {p: x[0] for p, x in files.items()},
             "skipped": skipped,
             "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(STATE)
    print(f"Ready: {table.count_rows()} chunks; {len(rows) - len(pending)} embeddings reused")


def query(args, state, files):
    import lancedb
    changed, removed = check_state(state, files)
    if changed or removed:
        print(f"Index stale: {len(changed)} changed/new files, {len(removed)} removed. Run build. Stale hits are omitted.", file=sys.stderr)
    table = lancedb.connect(str(RESEARCH / ".lancedb")).open_table(state["table"])
    if args.mode != "fts" and state.get("embedding_config") != EMBED_CONFIG:
        raise SystemExit("Embedding configuration changed. Rebuild before semantic search.")
    if args.mode == "fts":
        search = table.search(args.text, query_type="fts", fts_columns="text")
    else:
        vector = next(embedder().query_embed(args.text)).tolist()
        if args.mode == "vector":
            search = table.search(vector, query_type="vector").distance_type("cosine")
        else:
            from lancedb.rerankers import RRFReranker
            search = table.search(query_type="hybrid", fts_columns="text").vector(vector).text(args.text).rerank(RRFReranker())
    if args.kind:
        search = search.where(f"kind = '{args.kind}'", prefilter=True)
    rows = search.limit(max(args.limit * 8, 40)).to_list()
    results = []
    for row in rows:
        if row["path"] not in files or row["file_sha256"] != files[row["path"]][0]:
            continue
        if any(r["path"] == row["path"] and r["start_line"] <= row["end_line"] and row["start_line"] <= r["end_line"] for r in results):
            continue
        results.append({k: v for k, v in row.items() if k not in {"vector", "text", "id"}})
        if len(results) >= args.limit:
            break
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for n, row in enumerate(results, 1):
            print(f"\n{n}. {row['path']}:{row['start_line']}-{row['end_line']} [{row['kind']}]")
            if row["source_url"]:
                print(row["source_url"])
            print(row["content"])
        if not results:
            print("No current matching chunks. Try --mode vector, a shorter query, or rebuild.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--rebuild", action="store_true", help="Re-embed everything")
    commands.add_parser("status")
    commands.add_parser("doctor", help="Check imports, CPU inference, and embedding shape")
    q = commands.add_parser("query")
    q.add_argument("text")
    q.add_argument("--kind", choices=["paper", "code", "note"])
    q.add_argument("--mode", choices=["hybrid", "fts", "vector"], default="hybrid")
    q.add_argument("--limit", type=int, default=5)
    q.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.command == "doctor":
        import importlib.metadata
        import numpy as np
        print(sys.version)
        for name in ("lancedb", "fastembed", "onnxruntime", "pyarrow", "pymupdf4llm"):
            print(name, importlib.metadata.version(name))
        model = embedder(download=True)
        start = time.perf_counter()
        sample = "Global average pooling compresses spatial feature maps before the recurrent encoder projection. " * 10
        vectors = np.asarray(list(model.passage_embed([sample] * 16, batch_size=16)))
        assert vectors.shape == (16, 384) and np.isfinite(vectors).all()
        print("Providers:", model.model.model.get_providers())
        print(f"16 passages in {time.perf_counter() - start:.2f}s; vectors finite, shape={vectors.shape}")
        return
    if args.command == "build":
        import fcntl
        with (RESEARCH / "index.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit("Another index build is running.")
            return build(args)
    if not STATE.exists():
        parser.error("No index. Run: .venv/bin/python research/search.py build")
    state = json.loads(STATE.read_text())
    files, _ = corpus()
    if args.command == "status":
        changed, removed = check_state(state, files)
        print(json.dumps({**{k: v for k, v in state.items() if k != "files"},
                          "file_count": len(state["files"]), "changed": changed, "removed": removed}, indent=2))
    else:
        if args.limit < 1:
            parser.error("--limit must be positive")
        query(args, state, files)


if __name__ == "__main__":
    main()
