from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path

from .types import InputAsset


SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png"}
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".pdf"}


def load_inputs(paths: list[Path], scale: float = 2.0) -> list[InputAsset]:
    validated = _validate_input_paths(paths)
    assets: list[InputAsset] = []
    for source_index, path in enumerate(validated):
        if path.suffix.lower() == ".pdf":
            assets.extend(_load_pdf(path, source_index, scale=scale))
            continue
        assets.append(_load_image(path, source_index))
    return assets


def build_input_manifest(assets: list[InputAsset]) -> list[dict[str, object]]:
    return [
        {
            "source_path": asset.source_path,
            "source_index": asset.source_index,
            "page_index": asset.page_index,
            "media_type": asset.media_type,
            "filename": asset.filename,
        }
        for asset in assets
    ]


def _validate_input_paths(paths: list[Path]) -> list[Path]:
    if not 1 <= len(paths) <= 2:
        raise ValueError("Expected 1 to 2 input files")

    validated: list[Path] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Input path is not a file: {path}")
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        validated.append(path)
    return validated


def _load_image(path: Path, source_index: int) -> InputAsset:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if media_type not in SUPPORTED_IMAGE_TYPES:
        raise ValueError(f"Unsupported image media type: {media_type}")

    data = path.read_bytes()
    return InputAsset(
        source_path=str(path),
        source_index=source_index,
        page_index=0,
        media_type=media_type,
        filename=path.name,
        base64_data=base64.b64encode(data).decode("ascii"),
    )


def _load_pdf(path: Path, source_index: int, scale: float) -> list[InputAsset]:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError(
            "PDF support requires pypdfium2 and Pillow. Install dependencies from requirements.txt."
        ) from exc

    document = pdfium.PdfDocument(str(path))
    assets: list[InputAsset] = []

    for page_index in range(len(document)):
        page = document.get_page(page_index)
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image.close()
        assets.append(
            InputAsset(
                source_path=str(path),
                source_index=source_index,
                page_index=page_index,
                media_type="image/png",
                filename=f"{path.stem}-page-{page_index + 1}.png",
                base64_data=base64.b64encode(buffer.getvalue()).decode("ascii"),
            )
        )
        bitmap.close()
        page.close()

    return assets
