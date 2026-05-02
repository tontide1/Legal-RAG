"""
PaddleOCR-VL-1.5: Trich xuat van ban tieng Viet tu hinh anh
su dung model PaddlePaddle/PaddleOCR-VL-1.5 qua PaddleOCR Python API.

Doc tat ca anh trong data/images, gui den PaddleOCR-VL-1.5 de OCR,
luu ket qua duoi dang <ten_anh>.md trong data/paddle_ocr.

Cai dat:
    python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
    python -m pip install -U "paddleocr[doc-parser]"

    # Neu dung CPU:
    python -m pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    python -m pip install -U "paddleocr[doc-parser]"

Su dung:
    python paddle_ocr.py
"""

import sys
import time
from pathlib import Path

# Fix encoding cho Windows console
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# -------------------- Cau hinh --------------------
IMAGE_DIR  = Path("data/images")
OUTPUT_DIR = Path("data/paddle_ocr")
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

# -------------------- Khoi tao pipeline --------------------
print("[INFO ] Dang khoi tao PaddleOCR-VL-1.5 pipeline...")
from paddleocr import PaddleOCRVL  # noqa: E402 – import sau khi hieu thi thong bao
pipeline = PaddleOCRVL()
print("[INFO ] Pipeline san sang.")


def _extract_text_from_res(res) -> str:
    """Trich xuat van ban tu mot ket qua PaddleOCRVL theo nhieu cach."""

    # Cach 1: thuoc tinh .markdown (phien ban moi nhat)
    if hasattr(res, "markdown") and isinstance(res.markdown, str) and res.markdown.strip():
        return res.markdown

    # Cach 2: luu tam ra file markdown roi doc lai
    try:
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            res.save_to_markdown(save_path=tmp)
            md_files = list(Path(tmp).rglob("*.md"))
            if md_files:
                return md_files[0].read_text(encoding="utf-8")
    except Exception:
        pass

    # Cach 3: doc tu .json / dict ben trong result
    try:
        # res co the la dict hoac co thuoc tinh nhu .json / .res / .result
        data = None
        if isinstance(res, dict):
            data = res
        elif hasattr(res, "json"):
            data = res.json if isinstance(res.json, dict) else None
        elif hasattr(res, "res"):
            data = res.res

        if data:
            texts: list[str] = []
            # Duyet de quy lay tat ca gia tri kieu str co the la van ban
            def _collect(obj):
                if isinstance(obj, str):
                    texts.append(obj)
                elif isinstance(obj, dict):
                    for v in obj.values():
                        _collect(v)
                elif isinstance(obj, (list, tuple)):
                    for item in obj:
                        _collect(item)
            _collect(data)
            if texts:
                return "\n".join(texts)
    except Exception:
        pass

    # Cach 4: ep kieu str
    return str(res)


def ocr_image(image_path: Path) -> str:
    """
    Gui anh den PaddleOCR-VL-1.5 va tra ve noi dung Markdown.

    pipeline.predict() tra ve generator; moi phan tu la mot
    ket qua cho tung trang / khung.
    """
    results = list(pipeline.predict(str(image_path)))

    parts: list[str] = []
    for res in results:
        text = _extract_text_from_res(res)
        if text.strip():
            parts.append(text.strip())

    return "\n\n".join(parts)


def get_image_files(directory: Path) -> list[Path]:
    """Lay danh sach file anh duoc ho tro, sap xep theo ten."""
    return [
        f for f in sorted(directory.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    ]


def main():
    # Kiem tra thu muc anh
    if not IMAGE_DIR.exists():
        print(f"[ERROR] Khong tim thay thu muc '{IMAGE_DIR}'")
        sys.exit(1)

    # Tao thu muc output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Lay danh sach anh
    images = get_image_files(IMAGE_DIR)
    if not images:
        print(f"[ERROR] Khong tim thay anh nao trong '{IMAGE_DIR}'")
        sys.exit(1)

    total = len(images)
    print(f"[INFO ] Tim thay {total} anh trong '{IMAGE_DIR}'")
    print(f"[INFO ] Ket qua OCR se luu tai '{OUTPUT_DIR}'")
    print(f"[INFO ] Model: PaddlePaddle/PaddleOCR-VL-1.5")
    print("=" * 70)

    success    = 0
    failed     = 0
    skipped    = 0
    total_time = 0.0

    for idx, img_path in enumerate(images, 1):
        out_name = img_path.stem + ".md"
        out_path = OUTPUT_DIR / out_name

        # Bo qua neu da OCR roi
        if out_path.exists():
            print(f"[SKIP ] ({idx:>3}/{total}) {img_path.name} -> da ton tai")
            skipped += 1
            continue

        print(f"[OCR  ] ({idx:>3}/{total}) {img_path.name} ... ", end="", flush=True)
        start = time.time()

        try:
            text    = ocr_image(img_path)
            elapsed = time.time() - start
            total_time += elapsed

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)

            success += 1
            avg_time  = total_time / success
            remaining = (total - idx) * avg_time
            print(f"OK ({elapsed:.1f}s, {len(text)} chars) ~{remaining/60:.0f}m left")

        except KeyboardInterrupt:
            print("\n[WARN ] Dung boi nguoi dung (Ctrl+C)")
            break

        except Exception as e:
            elapsed = time.time() - start
            total_time += elapsed
            failed += 1
            print(f"FAIL ({elapsed:.1f}s): {type(e).__name__}: {e}")

    # Tong ket
    print("=" * 70)
    print(f"[DONE ] {success} thanh cong | {failed} that bai | {skipped} bo qua")
    print(f"[DONE ] Tong thoi gian xu ly: {total_time/60:.1f} phut")
    print(f"[DONE ] Cac file .md da luu tai: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()