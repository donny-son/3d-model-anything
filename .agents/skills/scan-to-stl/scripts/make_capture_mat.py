#!/usr/bin/env python3
"""Printable capture mat for object scans, and the matching ARKit reference image.

The app tracks the mat as an ARKit image of known width (180 mm). Under the object it:
  - gives ARKit texture to track on plain tables (the biggest real-world accuracy win)
  - defines a flat, level base plane and a repeatable object orientation
  - records ARKit's estimate of the mat's size, a check on ARKit's metric scale
    (object_to_stl.py --mat-scale applies it)

usage: make_capture_mat.py [-o capture-mat.pdf] [--paper a4|letter] [--app-image CaptureMat.png]

Print at 100 % ("Actual size"), then check the 100 mm ruler with a real ruler. If it is off,
pass the measured pattern width to object_to_stl.py as --mat-width-mm. Needs Pillow + numpy.
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PATTERN_MM = 180.0   # must match ObjectCaptureController.matPhysicalWidth
PATTERN_PX = 2048
CENTER = 0.42        # plain centre square (fraction of the width) where the object stands
SEED = 20261023      # fixed: the app's CaptureMat.png must be this exact pattern


def pattern():
    """High-contrast, non-repeating blocks around a plain centre with a faint 1 cm grid."""
    rng = np.random.default_rng(SEED)
    img = Image.new('L', (PATTERN_PX, PATTERN_PX), 255)
    d = ImageDraw.Draw(img)
    px_per_mm = PATTERN_PX / PATTERN_MM
    lo, hi = PATTERN_PX * (0.5 - CENTER / 2), PATTERN_PX * (0.5 + CENTER / 2)

    def in_centre(x0, y0, x1, y1):
        return x1 > lo and x0 < hi and y1 > lo and y0 < hi

    for size_mm, count in ((14, 60), (8, 220), (4.5, 700), (2.5, 1600)):
        s = size_mm * px_per_mm
        for _ in range(count):
            x, y = rng.uniform(0, PATTERN_PX - s, 2)
            w, h = s * rng.uniform(0.5, 1.6, 2)
            if in_centre(x, y, x + w, y + h):
                continue
            shade = int(rng.choice([0, 0, 0, 70, 255, 150]))
            if rng.random() < 0.3:
                d.ellipse([x, y, x + w, y + h], fill=shade)
            else:
                d.rectangle([x, y, x + w, y + h], fill=shade)
    # plain centre: light grey, faint centimetre grid, crosshair
    d.rectangle([lo, lo, hi, hi], fill=232)
    step = 10 * px_per_mm
    c = PATTERN_PX / 2
    k = np.arange(-20, 21)
    for v in c + k * step:
        if lo < v < hi:
            d.line([(v, lo), (v, hi)], fill=212, width=2)
            d.line([(lo, v), (hi, v)], fill=212, width=2)
    d.line([(c - 3 * step, c), (c + 3 * step, c)], fill=150, width=3)
    d.line([(c, c - 3 * step), (c, c + 3 * step)], fill=150, width=3)
    d.rectangle([lo, lo, hi, hi], outline=0, width=6)
    # an asymmetric corner mark so the mat's orientation is unambiguous
    m = 6 * px_per_mm
    d.polygon([(m, m), (m + 3 * m, m), (m, m + 3 * m)], fill=0)
    d.rectangle([0, 0, PATTERN_PX - 1, PATTERN_PX - 1], outline=0, width=8)
    return img


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def page(mat, paper):
    dpi = 300
    w_mm, h_mm = (210, 297) if paper == 'a4' else (215.9, 279.4)
    px = lambda mm: int(round(mm / 25.4 * dpi))  # noqa: E731
    sheet = Image.new('L', (px(w_mm), px(h_mm)), 255)
    side = px(PATTERN_MM)
    left = (sheet.width - side) // 2
    top = px(16)
    sheet.paste(mat.resize((side, side), Image.LANCZOS), (left, top))
    d = ImageDraw.Draw(sheet)
    # 100 mm ruler with centimetre ticks: measure it to check the print scale
    y = top + side + px(14)
    x0 = left
    d.line([(x0, y), (x0 + px(100), y)], fill=0, width=5)
    for i in range(11):
        x = x0 + px(10 * i)
        d.line([(x, y - px(4 if i % 5 == 0 else 2.5)), (x, y)], fill=0, width=4)
    big, small = font(52), font(36)
    d.text((x0 + px(104), y - px(5)), '100 mm', fill=0, font=big)
    lines = [
        '3D Model Anything object capture mat. Pattern: 180 mm wide.',
        'Print at 100% / Actual size. The bar above must measure exactly 100 mm;',
        'if it does not, pass the measured pattern width to object_to_stl.py (--mat-width-mm).',
        'Lay it flat on a table, put the object in the grey square, and keep the pattern visible',
        'while you scan. A matte print on thick paper (or glued to card) stays flattest.',
    ]
    for i, text in enumerate(lines):
        d.text((x0, y + px(10) + i * px(6), ), text, fill=0, font=small)
    return sheet, dpi


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('-o', default='capture-mat.pdf', help='printable page (.pdf or .png)')
    ap.add_argument('--paper', choices=['a4', 'letter'], default='a4')
    ap.add_argument('--app-image', help='also write the 1024 px reference image the app bundles (CaptureMat.png)')
    a = ap.parse_args()
    mat = pattern()
    sheet, dpi = page(mat, a.paper)
    sheet.save(a.o, resolution=dpi) if a.o.lower().endswith('.pdf') else sheet.save(a.o, dpi=(dpi, dpi))
    print(f'wrote {a.o} ({a.paper}, pattern {PATTERN_MM:.0f} mm)')
    if a.app_image:
        mat.resize((1024, 1024), Image.LANCZOS).save(a.app_image, optimize=True)
        print(f'wrote {a.app_image}')


if __name__ == '__main__':
    main()
