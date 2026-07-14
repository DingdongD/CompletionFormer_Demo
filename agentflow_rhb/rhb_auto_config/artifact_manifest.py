import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ArtifactEntry:
    path: str
    kind: str
    size_bytes: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class ArtifactManifest:
    name: str
    entries: List[ArtifactEntry]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry(path: Path, root: Path = None) -> ArtifactEntry:
    stat = path.stat()
    rel = str(path if root is None else path.relative_to(root))
    return ArtifactEntry(
        path=rel,
        kind="file",
        size_bytes=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
        sha256=_sha256_file(path),
    )


def build_artifact_manifest(name: str, paths: List[Path]) -> ArtifactManifest:
    entries: List[ArtifactEntry] = []
    for path in paths:
        if path.is_file():
            entries.append(_entry(path))
        elif path.is_dir():
            root = path
            for file_path in sorted(root.rglob("*")):
                if file_path.is_file():
                    entries.append(_entry(file_path, root=root))
    return ArtifactManifest(name=name, entries=entries)


def save_artifact_manifest(manifest: ArtifactManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")


def render_artifact_manifest(manifest: ArtifactManifest, limit: int = 120) -> str:
    total = sum(item.size_bytes for item in manifest.entries)
    lines = [
        "# Artifact Manifest",
        "",
        f"- name: `{manifest.name}`",
        f"- files: {len(manifest.entries)}",
        f"- total_size_bytes: {total}",
        "",
        "size_bytes\tsha256\tpath",
    ]
    for item in manifest.entries[:limit]:
        lines.append(f"{item.size_bytes}\t{item.sha256}\t{item.path}")
    if len(manifest.entries) > limit:
        lines.append(f"... truncated {len(manifest.entries) - limit} entries")
    return "\n".join(lines)
