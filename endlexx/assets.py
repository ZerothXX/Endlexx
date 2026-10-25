import hashlib
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import settings
from .db import Asset, Session, record, uid


def store_image(source, filename: str, parent_id: str | None = None, kind: str = "image") -> dict:
    try:
        image = Image.open(source) if not isinstance(source, Image.Image) else source
        if image.width * image.height > 24_000_000:
            raise ValueError("图片最大支持 2400 万像素")
        image = ImageOps.exif_transpose(image)
        image.load()
        image = image.convert("L" if kind == "mask" else "RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("无法读取图片，请使用 PNG、JPEG 或 WebP") from exc
    asset_id = uid()
    folder = settings().data_dir.resolve() / "assets"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{asset_id}.png"
    image.save(path)
    with Session() as db:
        asset = Asset(
            id=asset_id,
            filename=Path(filename).name[:240],
            path=str(path),
            kind=kind,
            parent_id=parent_id,
            width=image.width,
            height=image.height,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        db.add(asset)
        db.commit()
        return public_asset(asset)


def public_asset(asset: Asset) -> dict:
    value = record(asset)
    value.pop("path")
    return value | {"url": f"/api/assets/{asset.id}/file"}


def asset_path(asset_id: str) -> Path:
    with Session() as db:
        asset = db.get(Asset, asset_id)
        if not asset:
            raise ValueError("图片资产不存在")
        path = Path(asset.path).resolve()
        if not path.is_relative_to((settings().data_dir.resolve() / "assets")) or not path.is_file():
            raise ValueError("图片文件不可用")
        return path
