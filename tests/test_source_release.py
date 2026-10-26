import hashlib
import json
import zipfile

import pytest

from scripts import package_source as release


def source_fixture(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(release, "ROOT_FILES", ("README.md", ".env.example"))
    monkeypatch.setattr(release, "DOCS", ())
    monkeypatch.setattr(release, "PUBLICATION_DOCS", ())
    monkeypatch.setattr(release, "RESEARCH_MEDIA", ())
    monkeypatch.setattr(release, "TREES", {"endlexx": {".py"}})
    (root / "README.md").write_text("Research preview", encoding="utf-8")
    (root / ".env.example").write_text("API_KEY=", encoding="utf-8")
    (root / "endlexx").mkdir()
    (root / "endlexx/api.py").write_text("print('hello')", encoding="utf-8")
    return root


def test_release_is_reproducible_and_excludes_private_data(tmp_path, monkeypatch):
    root = source_fixture(tmp_path, monkeypatch)
    for name in (".env", "data/model-secrets.json", "PPO/outputs/models/private.pt", "endlexx/__pycache__/x.pyc"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private fixture", encoding="utf-8")
    first, second = tmp_path / "one.zip", tmp_path / "two.zip"
    result = release.build_archive(root, first)
    release.build_archive(root, second)
    assert first.read_bytes() == second.read_bytes()
    assert result["files"] == 3
    assert result["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    with zipfile.ZipFile(first) as archive:
        manifest = json.loads(archive.read(f"{release.PREFIX}/SOURCE_MANIFEST.json"))
    assert set(manifest["files"]) == {"README.md", ".env.example", "endlexx/api.py"}


def test_release_blocks_credentials_without_printing_them(tmp_path, monkeypatch):
    root = source_fixture(tmp_path, monkeypatch)
    token = "hf_" + "A" * 35
    (root / "endlexx/api.py").write_text(f"token = '{token}'", encoding="utf-8")
    with pytest.raises(ValueError, match="Potential credential") as error:
        release.build_archive(root, tmp_path / "unsafe.zip")
    assert token not in str(error.value)
    assert not (tmp_path / "unsafe.zip").exists()


def test_release_verification_rejects_tampered_source(tmp_path, monkeypatch):
    root = source_fixture(tmp_path, monkeypatch)
    original = tmp_path / "good.zip"
    release.build_archive(root, original)
    damaged = tmp_path / "damaged.zip"
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(damaged, "w") as target:
        for info in source.infolist():
            content = source.read(info)
            if info.filename.endswith("/endlexx/api.py"):
                content = b"print('tampered')"
            target.writestr(info, content)
    with pytest.raises(ValueError, match="Source hash mismatch"):
        release.verify_archive(damaged)


def test_release_rejects_vendor_code_that_disagrees_with_snapshot_inventory(tmp_path, monkeypatch):
    root = source_fixture(tmp_path, monkeypatch)
    vendor = root / "vendor/endless"
    vendor.mkdir(parents=True)
    (vendor / "source-manifest.json").write_text(json.dumps({"model.py": "0" * 64}), encoding="utf-8")
    (vendor / "model.py").write_text("# modified snapshot", encoding="utf-8")
    monkeypatch.setattr(release, "ROOT_FILES", release.ROOT_FILES + ("vendor/endless/source-manifest.json",))
    monkeypatch.setattr(release, "TREES", release.TREES | {"vendor/endless": {".py"}})
    with pytest.raises(ValueError, match="Vendor snapshot hash mismatch"):
        release.build_archive(root, tmp_path / "unsafe-vendor.zip")


@pytest.mark.parametrize("name", ["../escape.py", ".env.prod", "data/file.py", "x/secret.pem"])
def test_release_rejects_unsafe_member_paths(name):
    with pytest.raises(ValueError, match="Forbidden archive path"):
        release.validate_source(name, b"not a credential")


def test_publication_link_must_exist_inside_source_archive(tmp_path, monkeypatch):
    root = source_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(release, "PUBLICATION_DOCS", ("README.md",))
    (root / "README.md").write_text('<img src="data/private.png">', encoding="utf-8")
    with pytest.raises(ValueError, match="Missing publication target"):
        release.build_archive(root, tmp_path / "broken-docs.zip")


def test_release_carries_only_approved_media_and_rejects_edited_image(tmp_path, monkeypatch):
    root = source_fixture(tmp_path, monkeypatch)
    image = b"public historical image fixture"
    folder = root / "docs/media/research"
    folder.mkdir(parents=True)
    (folder / "example.png").write_bytes(image)
    (folder / "private.png").write_bytes(b"unregistered user asset")
    commit = "a" * 40
    manifest = {
        "source_repository": "https://github.com/ZerothXX/Endless-XXX", "source_commit": commit,
        "files": [{
            "file": "example.png", "source_path": "input/example.png", "bytes": len(image),
            "source_url": f"https://raw.githubusercontent.com/ZerothXX/Endless-XXX/{commit}/input/example.png",
            "sha256": hashlib.sha256(image).hexdigest(),
            "git_blob_sha": hashlib.sha1(b"blob " + str(len(image)).encode() + b"\0" + image).hexdigest(),
        }],
    }
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    media = tuple("docs/media/research/" + name for name in ("manifest.json", "example.png"))
    monkeypatch.setattr(release, "RESEARCH_MEDIA", media)
    monkeypatch.setattr(release, "PUBLICATION_DOCS", ("README.md",))
    (root / "README.md").write_text("![Example](docs/media/research/example.png)", encoding="utf-8")
    result = release.build_archive(root, tmp_path / "media.zip")
    assert result["files"] == 5
    with zipfile.ZipFile(tmp_path / "media.zip") as archive:
        assert not any(name.endswith("private.png") for name in archive.namelist())
    (folder / "example.png").write_bytes(b"silently changed output")
    with pytest.raises(ValueError, match="Research media provenance mismatch"):
        release.build_archive(root, tmp_path / "edited.zip")
