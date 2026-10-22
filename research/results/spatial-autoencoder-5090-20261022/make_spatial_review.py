"""Arrange native reconstruction crops for inspection; no model execution."""
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "build/spatial-ae-5090-623b7cb9-01/spatial-run"
OUT = ROOT / "build/spatial-viewer-review-623b7cb9-01"
OUT.mkdir(exist_ok=False)
variants = ("s8-c8", "s16-c8", "s16-c16", "s32-c16")
classes = {1: "apple", 8: "spotted apple", 7: "striped mushroom", 10: "spotted toadstool"}

for skin in (0, 3, 4):
    rows = []
    for variant in variants:
        plain = RUN / variant / "plain/reconstruction"
        objects = RUN / variant / "objects/reconstruction"
        for kind, name in classes.items():
            candidates = sorted(plain.glob(f"split-2-skin-{skin}-kind-{kind}-cell-*-original.ppm"))
            if not candidates:
                continue
            original = candidates[0]
            assert hashlib.sha256(original.read_bytes()).digest() == hashlib.sha256((objects / original.name).read_bytes()).digest()
            plain_recon = original.with_name(original.name.replace("-original", "-reconstruction"))
            object_recon = objects / plain_recon.name
            row = OUT / f"skin-{skin}-{variant}-kind-{kind}.png"
            command = ["montage", "-font", "DejaVu-Sans", "-pointsize", "12", "-background", "white", "-fill", "black"]
            for title, path in (("original", original), ("plain", plain_recon), ("object loss", object_recon)):
                command += ["-label", f"{variant}: {name}\n{title}", str(path)]
            command += ["-filter", "point", "-resize", "128x128", "-tile", "3x1", "-geometry", "+4+4", str(row)]
            subprocess.run(command, check=True)
            rows.append(row)
    target = OUT / f"skin-{skin}-contact.png"
    subprocess.run(["montage", *map(str, rows), "-tile", "1x", "-geometry", "+0+0", "-background", "white", str(target)], check=True)
    print(target)

source = RUN / "s8-c8/objects/reconstruction"
subprocess.run(["montage", str(source / "split-2-skin-0-original.ppm"),
                str(source / "split-2-skin-0-reconstruction.ppm"), "-tile", "2x1",
                "-geometry", "+4+4", "-background", "white", str(OUT / "s8-c8-objects-full-frame.png")], check=True)
