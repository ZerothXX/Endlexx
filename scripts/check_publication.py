"""Offline checks for publication links and explicitly approved research media."""

import hashlib
import json
import posixpath
import re
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if value and ((tag == "img" and key == "src") or (tag == "a" and key == "href")):
                self.targets.append(value)


def local_references(text):
    # Ignore code examples, which can contain placeholders instead of real links.
    text = re.sub(r"^```[^\n]*\n.*?^```\s*$", "", text, flags=re.M | re.S)
    text = re.sub(r"`[^`\n]+`", "", text)
    parser = References()
    parser.feed(text)
    markdown = re.findall(r"!?\[[^\]\n]*\]\((<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)", text)
    return parser.targets + [target.strip("<>") for target in markdown]


def check_links(sources, documents, available=None):
    """Check relative file targets against the archive (or a pinned upstream tree)."""
    available = set(sources) if available is None else set(available)
    count = 0
    for name in documents:
        text = sources[name].decode("utf-8")
        for target in local_references(text):
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path:
                continue
            path = unquote(url.path)
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), path))
            if path.startswith("/") or "\\" in path or resolved.startswith("../"):
                raise ValueError(f"Unsafe publication link in {name}: {target}")
            if resolved not in available and not any(p.startswith(resolved + "/") for p in available):
                raise ValueError(f"Missing publication target in {name}: {target}")
            count += 1
    return count


def check_publication(sources, documents, media):
    links = check_links(sources, documents)
    if not media:
        return {"documents": len(documents), "local_links": links, "research_images": 0}
    prefix = "docs/media/research/"
    manifest_name = prefix + "manifest.json"
    manifest = json.loads(sources[manifest_name])
    commit = manifest["source_commit"]
    if (
        manifest["source_repository"] != "https://github.com/ZerothXX/Endless-XXX"
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
    ):
        raise ValueError("Invalid research provenance repository/commit")
    listed = []
    for entry in manifest["files"]:
        name = prefix + entry["file"]
        if name not in media or entry["file"] != posixpath.basename(entry["file"]):
            raise ValueError("Unapproved research media entry")
        content = sources[name]
        sha256 = hashlib.sha256(content).hexdigest()
        blob_sha = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        expected_url = (
            f"https://raw.githubusercontent.com/ZerothXX/Endless-XXX/{commit}/{entry['source_path']}"
        )
        if (
            sha256 != entry["sha256"] or blob_sha != entry["git_blob_sha"]
            or len(content) != entry["bytes"] or entry["source_url"] != expected_url
        ):
            raise ValueError(f"Research media provenance mismatch: {name}")
        listed.append(name)
    if len(listed) != len(set(listed)) or set(listed) != set(media) - {manifest_name}:
        raise ValueError("Research media inventory does not match the publication allowlist")
    return {"documents": len(documents), "local_links": links, "research_images": len(listed)}


def main():
    try:
        from scripts import package_source as release
    except ImportError:
        import package_source as release
    sources = release.collect_sources(release.ROOT)
    print(json.dumps(check_publication(sources, release.PUBLICATION_DOCS, release.RESEARCH_MEDIA)))


if __name__ == "__main__":
    main()
