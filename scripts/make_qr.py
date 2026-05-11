"""Generate a QR code (PNG) and a printable A6 card (PDF) for the dashboard URL.

Usage:
    python scripts/make_qr.py "https://your-app.streamlit.app"
    python scripts/make_qr.py "https://..." --out presentation/qr --label "Live results"

Outputs (defaults under ``presentation/qr/``):
    dashboard_qr.png       — high-res QR for slide decks (white bg, dark cyan modules)
    dashboard_qr_card.pdf  — A6 (105 × 148 mm) printable card with QR + URL + caption

Dependencies: ``segno`` (pure-Python QR), ``reportlab`` (PDF). Both are listed
under [tool.uv]/dev or you can install them ad hoc:

    pip install --user segno reportlab
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Optional


REPO_ROOT     = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUT   = REPO_ROOT / "presentation" / "qr"
BRAND_CYAN    = "#22d3ee"
BRAND_BG      = "#171614"
BRAND_TEXT    = "#cdccca"
BRAND_MUTED   = "#7a7974"
THESIS_TITLE  = ("AI Trading Agent — A 2×2 factorial study of\n"
                 "LLM model class × multi-agent architecture")
AUTHOR_LINE   = ("Iker Sánchez Pereira  ·  "
                 "Universidad Pontificia Comillas — ICADE-ICAI")


def _ensure_deps() -> None:
    missing = []
    try:
        import segno  # noqa: F401
    except ImportError:
        missing.append("segno")
    try:
        import reportlab  # noqa: F401
    except ImportError:
        missing.append("reportlab")
    if missing:
        sys.stderr.write(
            f"Missing dependencies: {', '.join(missing)}.\n"
            f"Install with:  pip install --user {' '.join(missing)}\n"
        )
        sys.exit(1)


def make_png(url: str, out_path: pathlib.Path,
             *, scale: int = 16,
             dark: str = BRAND_CYAN, light: str = "white") -> None:
    """High-resolution QR PNG with brand colors. Scale 16 ≈ 600×600 px."""
    import segno
    qr = segno.make(url, error="h")  # high error correction (~30%) for print
    qr.save(str(out_path), scale=scale, dark=dark, light=light, border=2)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}  ({scale}× scale)")


def make_card_pdf(url: str, out_path: pathlib.Path, *,
                  label: str = "Live results dashboard",
                  thesis_title: str = THESIS_TITLE,
                  author: str = AUTHOR_LINE) -> None:
    """A6 portrait card (105 × 148 mm) with QR + URL + caption."""
    import io
    import segno
    from reportlab.lib.pagesizes import A6
    from reportlab.lib.units     import mm
    from reportlab.lib.utils     import ImageReader
    from reportlab.pdfgen        import canvas

    page_w, page_h = A6                   # 297.6, 419.5 pt = 105×148 mm

    # 1. Generate QR PNG in memory at high res
    qr = segno.make(url, error="h")
    qr_buf = io.BytesIO()
    qr.save(qr_buf, kind="png", scale=20, dark=BRAND_CYAN, light=BRAND_BG, border=1)
    qr_buf.seek(0)

    c = canvas.Canvas(str(out_path), pagesize=A6)

    # Background
    c.setFillColorRGB(*_hex_to_rgb(BRAND_BG))
    c.rect(0, 0, page_w, page_h, fill=1, stroke=0)

    # Top accent bar
    c.setFillColorRGB(*_hex_to_rgb(BRAND_CYAN))
    c.rect(0, page_h - 4 * mm, page_w, 4 * mm, fill=1, stroke=0)

    # Title strip
    c.setFillColorRGB(*_hex_to_rgb(BRAND_CYAN))
    c.setFont("Helvetica-Bold", 7)
    c.drawString(8 * mm, page_h - 12 * mm, "TFG · AI TRADING AGENT")
    c.setFillColorRGB(*_hex_to_rgb(BRAND_TEXT))
    c.setFont("Helvetica-Bold", 11)
    for i, line in enumerate(thesis_title.split("\n")):
        c.drawString(8 * mm, page_h - (18 + i * 5) * mm, line)

    # QR centered
    qr_size = 75 * mm
    qr_x = (page_w - qr_size) / 2
    qr_y = page_h / 2 - qr_size / 2 - 6 * mm
    c.drawImage(ImageReader(qr_buf), qr_x, qr_y, qr_size, qr_size,
                preserveAspectRatio=True, mask="auto")

    # Caption above QR
    c.setFillColorRGB(*_hex_to_rgb(BRAND_CYAN))
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(page_w / 2, qr_y + qr_size + 10 * mm, label)
    c.setFillColorRGB(*_hex_to_rgb(BRAND_MUTED))
    c.setFont("Helvetica", 8.5)
    c.drawCentredString(page_w / 2, qr_y + qr_size + 4 * mm,
                        "Scan to open the live results dashboard")

    # URL below QR
    c.setFillColorRGB(*_hex_to_rgb(BRAND_TEXT))
    c.setFont("Courier-Bold", 8)
    c.drawCentredString(page_w / 2, qr_y - 6 * mm, url)

    # Author line at bottom
    c.setFillColorRGB(*_hex_to_rgb(BRAND_MUTED))
    c.setFont("Helvetica", 7)
    c.drawCentredString(page_w / 2, 6 * mm, author)

    c.showPage()
    c.save()
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}  (A6, 105×148 mm)")


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255.0,
            int(h[2:4], 16) / 255.0,
            int(h[4:6], 16) / 255.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("url",
                    help='Deployed dashboard URL, e.g. "https://your-app.streamlit.app"')
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"Output directory. Default: {DEFAULT_OUT}.")
    ap.add_argument("--label", default="Live results dashboard",
                    help='Caption above the QR code on the printed card.')
    ap.add_argument("--scale", type=int, default=16,
                    help="QR PNG pixel scale (16 ≈ 600×600 px).")
    args = ap.parse_args()

    _ensure_deps()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating QR for: {args.url}")
    make_png(args.url, out / "dashboard_qr.png", scale=args.scale)
    make_card_pdf(args.url, out / "dashboard_qr_card.pdf", label=args.label)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
