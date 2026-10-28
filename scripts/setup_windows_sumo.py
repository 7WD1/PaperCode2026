"""Install a verified standalone SUMO 1.22 Windows runtime, without pip metadata.

Official 1.22 Windows runtime wheel metadata advertises cp38 internally despite
its generic wheel filename. Extracting its unchanged application payload and the
matching native binding avoids registering that incompatible distribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import stat
import sys
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import uuid
import zipfile

VERSION = "1.22.0"


def select_wheel(metadata: dict, package: str, python_tag: str,
                  platform_tag: str = "win_amd64") -> dict:
    """Choose only the requested official Windows architecture and Python ABI."""
    if package not in ("eclipse-sumo", "libsumo"):
        raise ValueError("Only official SUMO runtime and native binding are selected")
    suffix = (f"-{python_tag}-{python_tag}-{platform_tag}.whl" if package == "libsumo"
              else f"-none-{platform_tag}.whl")
    expected_prefix = package.replace("-", "_")+"-"+VERSION+"-"
    candidates = [entry for entry in metadata["urls"]
                  if entry["filename"].startswith(expected_prefix) and entry["filename"].endswith(suffix)]
    if len(candidates) != 1:
        raise ValueError("No unique official wheel matches this Windows architecture/Python ABI")
    selected = dict(candidates[0])
    name = selected["filename"]
    if Path(name).name != name or any(char in name for char in ("/", "\\", ":")):
        raise ValueError("Unsafe wheel filename")
    url = urlparse(selected["url"])
    if url.scheme != "https" or url.hostname != "files.pythonhosted.org":
        raise ValueError("Wheel download must use official HTTPS PyPI hosting")
    if selected["size"] <= 0 or len(selected["digests"]["sha256"]) != 64:
        raise ValueError("Official length and SHA256 are required")
    return selected


def metadata_for(package: str) -> dict:
    request = Request(f"https://pypi.org/pypi/{package}/{VERSION}/json",
                      headers={"User-Agent": "rc-shapleynet-sumo-setup"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def verify(path: Path, record: dict) -> bool:
    if not path.is_file() or path.stat().st_size != record["size"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == record["digests"]["sha256"]


def verified_wheel(record: dict, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    target = cache/record["filename"]
    if verify(target, record):
        return target
    partial = target.with_suffix(".partial")
    last_error = None
    for _ in range(3):
        try:
            request = Request(record["url"]+"?fresh="+uuid.uuid4().hex,
                              headers={"Cache-Control": "no-cache", "User-Agent": "rc-shapleynet-sumo-setup"})
            with urlopen(request, timeout=45) as response, partial.open("wb") as stream:
                for chunk in iter(lambda: response.read(1024*1024), b""):
                    stream.write(chunk)
            if not verify(partial, record):
                raise ValueError("Downloaded wheel fails the official length/SHA256 check")
            partial.replace(target)
            return target
        except (OSError, ValueError) as exc:
            last_error = exc
    raise RuntimeError("No complete, verified official wheel was obtained") from last_error


def safe_target(root: Path, member: str) -> Path:
    """Reject traversal, alternate Windows separators/drives and metadata."""
    relative = PurePosixPath(member)
    if (relative.is_absolute() or ".." in relative.parts or "\\" in member or ":" in member
            or any(part.endswith(".dist-info") for part in relative.parts)):
        raise ValueError("Unsafe or distribution-metadata archive member")
    target = (root/Path(*relative.parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Archive member escapes the chosen runtime directory")
    return target


def extract_payload(wheel: Path, package: str, runtime: Path) -> dict:
    """Copy only unchanged application/binding bytes; never install dist-info."""
    prefix = "sumo/" if package == "eclipse-sumo" else "libsumo/"
    count, selected_hashes = 0, {}
    with zipfile.ZipFile(wheel) as archive:
        for item in archive.infolist():
            if not item.filename.startswith(prefix):
                continue
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Symlinks are not accepted in the runtime archive")
            member = item.filename if package == "eclipse-sumo" else "sumo/tools/"+item.filename
            target = safe_target(runtime, member)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = archive.read(item)
            target.write_bytes(payload)
            count += 1
            if package == "libsumo" or target.name in ("sumo.exe", "netconvert.exe"):
                selected_hashes[member] = hashlib.sha256(payload).hexdigest()
    if not count:
        raise ValueError("Official wheel contains no expected runtime payload")
    return {"extracted_files": count, "selected_payload_sha256": selected_hashes}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True,
                        help="External directory for an unchanged standalone SUMO runtime")
    parser.add_argument("--cache-dir", type=Path,
                        help="Optional cache of official wheels; every cached file is reverified")
    args = parser.parse_args(argv)
    if sys.platform != "win32" or platform.machine().lower() not in ("amd64", "x86_64"):
        parser.error("This helper supports 64-bit Windows; other platforms use the SUMO extra")
    runtime = args.runtime_dir.resolve()
    cache = args.cache_dir.resolve() if args.cache_dir else runtime/"wheel-cache"
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    report = {"version": VERSION, "python_abi": python_tag, "packages": {}}
    for package in ("eclipse-sumo", "libsumo"):
        record = select_wheel(metadata_for(package), package, python_tag)
        wheel = verified_wheel(record, cache)
        details = extract_payload(wheel, package, runtime)
        report["packages"][package] = {"filename": record["filename"], "url": record["url"],
                                       "size": record["size"], "sha256": record["digests"]["sha256"],
                                       **details}
    home = runtime/"sumo"
    (runtime/"runtime_provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Verified unchanged SUMO runtime and matching Python binding extracted.")
    print(f'$env:SUMO_HOME="{home.as_posix()}"')
    print('$env:PYTHONPATH="$env:SUMO_HOME/tools;$env:PYTHONPATH"')
    print('$env:Path="$env:SUMO_HOME/bin;$env:Path"')
    print('sumo --version; netconvert --version')
    print('python -m pytest tests/test_physics.py -q')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
