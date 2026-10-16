"""Fetch the Question 3 datasets from the official UCI archives (only if they are missing).

Writes data/winequality-white.csv and data/har/{X,y}_{train,test}.txt + activity_labels.txt,
and records each archive's URL, size and SHA-256 hash in data/sources.json.
Adapted from the downloader in the OpenAI Codex review of this assignment.
"""
import hashlib
import io
import json
import os
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
HAR_FILES = ["X_train.txt", "X_test.txt", "y_train.txt", "y_test.txt", "activity_labels.txt"]
SOURCES = {
    "wine": "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
    "har": "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
}


def have(name):
    if name == "wine":
        return os.path.exists(os.path.join(DATA, "winequality-white.csv"))
    return all(os.path.exists(os.path.join(DATA, "har", f)) for f in HAR_FILES)


def main():
    os.makedirs(os.path.join(DATA, "har"), exist_ok=True)
    manifest = []
    for name, url in SOURCES.items():
        if have(name):
            print(f"{name}: already present, skipping download")
            continue
        print(f"{name}: downloading {url}", flush=True)
        with urllib.request.urlopen(url, timeout=300) as r:
            raw = r.read()
        z = zipfile.ZipFile(io.BytesIO(raw))
        if name == "wine":
            entry = next(n for n in z.namelist() if n.endswith("winequality-white.csv"))
            with open(os.path.join(DATA, "winequality-white.csv"), "wb") as f:
                f.write(z.read(entry))
        else:
            inner = [n for n in z.namelist() if n.endswith(".zip")]   # UCI nests the HAR zip
            if inner:
                z = zipfile.ZipFile(io.BytesIO(z.read(inner[0])))
            for base in HAR_FILES:
                entry = next(n for n in z.namelist() if n.endswith("/" + base) or n == base)
                with open(os.path.join(DATA, "har", base), "wb") as f:
                    f.write(z.read(entry))
        manifest.append(dict(dataset=name, url=url, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    if manifest:
        with open(os.path.join(DATA, "sources.json"), "w") as f:
            json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
