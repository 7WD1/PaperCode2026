"""Offline checks of official-artifact selection and safe runtime extraction."""
import hashlib
import importlib.util
from pathlib import Path
import zipfile

import pytest


spec = importlib.util.spec_from_file_location("windows_sumo_setup", Path(__file__).resolve().parents[1]/"scripts/setup_windows_sumo.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def wheel_record(name):
    return {"filename": name, "url": "https://files.pythonhosted.org/packages/example/"+name,
            "size": 10, "digests": {"sha256": "a"*64}}


def test_selector_requires_the_matching_windows_python_abi():
    cp310 = wheel_record("libsumo-1.22.0-cp310-cp310-win_amd64.whl")
    cp312 = wheel_record("libsumo-1.22.0-cp312-cp312-win_amd64.whl")
    unix = wheel_record("libsumo-1.22.0-cp312-cp312-manylinux_2_17_x86_64.whl")
    assert setup.select_wheel({"urls": [cp310, cp312, unix]}, "libsumo", "cp312") == cp312
    with pytest.raises(ValueError, match="matches"):
        setup.select_wheel({"urls": [cp310, unix]}, "libsumo", "cp312")


@pytest.mark.parametrize("member", ["sumo/../../escape", "/outside", "C:/outside", "sumo\\..\\escape", "sumo/x.dist-info/METADATA"])
def test_runtime_paths_cannot_escape_or_install_distribution_metadata(tmp_path, member):
    with pytest.raises(ValueError):
        setup.safe_target(tmp_path, member)


def test_verification_requires_both_exact_size_and_hash(tmp_path):
    source = tmp_path/"artifact.whl"
    source.write_bytes(b"complete payload")
    record = {"size": source.stat().st_size, "digests": {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}
    assert setup.verify(source, record)
    assert not setup.verify(source, {**record, "size": record["size"]+1})
    assert not setup.verify(source, {"size": record["size"], "digests": {"sha256": "0"*64}})


def test_payload_remains_unchanged_and_dist_info_is_not_extracted(tmp_path):
    source = tmp_path/"runtime.whl"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("sumo/bin/sumo.exe", b"binary fixture")
        archive.writestr("eclipse_sumo-1.22.0.dist-info/WHEEL", "Tag: cp38-cp38-win_amd64")
    runtime = tmp_path/"runtime"
    report = setup.extract_payload(source, "eclipse-sumo", runtime)
    assert (runtime/"sumo/bin/sumo.exe").read_bytes() == b"binary fixture"
    assert not list(runtime.rglob("*.dist-info"))
    assert report["extracted_files"] == 1


def test_payload_extraction_rejects_a_traversal_member(tmp_path):
    source = tmp_path/"bad.whl"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("sumo/../../escape", "bad")
    with pytest.raises(ValueError):
        setup.extract_payload(source, "eclipse-sumo", tmp_path/"runtime")
    assert not (tmp_path/"escape").exists()
