"""
GLM-OCR: Trich xuat van ban tu hinh anh su dung model glm-ocr qua Ollama.

Doc tat ca anh trong data/images, gui den glm-ocr de OCR,
luu ket qua duoi dang <ten_anh>.md trong data/ocr.

Su dung:
    python glm_ocr.py
"""

import base64
import io
import sys
import time

import ollama
from pathlib import Path
from PIL import Image

# Fix encoding cho Windows console
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# -------------------- Cau hinh --------------------
MODEL_NAME = "glm-ocr"
NUM_CTX = 8192           # Context size cho vision model
MAX_IMAGE_DIM = 1024     # Resize anh ve max dimension nay
IMAGE_DIR = Path("data/images")
OUTPUT_DIR = Path("data/ocr")
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
PROMPT = "Text Recognition:"


def resize_and_encode(image_path: Path, max_dim: int = MAX_IMAGE_DIM) -> str:
    """Resize anh neu qua lon va tra ve base64 string."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ocr_image(image_path: Path) -> str:
    """Gui anh den glm-ocr qua Ollama va tra ve van ban trich xuat."""
    img_b64 = resize_and_encode(image_path)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": PROMPT,
            "images": [img_b64],
        }],
        options={"num_ctx": NUM_CTX},
    )

    return response["message"]["content"]


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
    print(f"[INFO ] Model: {MODEL_NAME} | num_ctx: {NUM_CTX} | max_dim: {MAX_IMAGE_DIM}px")
    print("=" * 70)

    success = 0
    failed = 0
    skipped = 0
    total_time = 0

    for idx, img_path in enumerate(images, 1):
        # Ten file output: <ten anh khong extension>.md
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
            text = ocr_image(img_path)
            elapsed = time.time() - start
            total_time += elapsed

            # Luu ket qua
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)

            success += 1
            remaining = (total - idx) * (total_time / max(success, 1))
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
