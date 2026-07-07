"""
overlay_grid.py

Overlay a horizontal pulse grid on a SAR scene image.
Gridlines are spaced by global pulse number, not pixels.

Global convention:
    pixel row 0  <->  global pulse PULSE_START
    pixel row H  <->  global pulse PULSE_STOP

Usage:
    python overlay_grid.py --img scene.png
    python overlay_grid.py --img scene.png --pulse-step 5000
    python overlay_grid.py --img scene.png --pulse-start 46528 --pulse-stop 124580
"""

import os
import argparse
from PIL import Image, ImageDraw, ImageFont

DEFAULT_IMG         = 'image.png'
DEFAULT_OUT         = 'scene_grid_overlay.png'
DEFAULT_PULSE_START = 46528
DEFAULT_PULSE_STOP  = 124580
DEFAULT_PULSE_STEP  = 5000


def overlay_grid(img_path, out_path, pulse_start, pulse_stop, pulse_step):
    img  = Image.open(img_path).convert('RGB')
    W, H = img.size
    draw = ImageDraw.Draw(img)

    font = None
    for fp in [
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf',
        '/usr/share/fonts/truetype/freefont/FreeMono.ttf',
    ]:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, size=36)
            break

    pixels_per_pulse = H / (pulse_stop - pulse_start)

    # Snap first gridline to the nearest pulse_step boundary above pulse_start
    first = (pulse_start // pulse_step + 1) * pulse_step

    pulse = first
    while pulse < pulse_stop:
        py = round((pulse - pulse_start) * pixels_per_pulse)
        py = max(0, min(py, H - 1))

        draw.line([(0, py), (W, py)], fill=(255, 255, 0), width=4)

        label = f'g={pulse}  px={py}'
        bbox  = draw.textbbox((0, 0), label, font=font)
        th    = bbox[3] - bbox[1]
        tw    = bbox[2] - bbox[0]
        ty    = py + 6 if py + th + 10 < H else py - th - 6
        draw.rectangle([6, ty - 2, tw + 14, ty + th + 2], fill=(0, 0, 0))
        draw.text((10, ty), label, fill=(255, 255, 0), font=font)

        pulse += pulse_step

    img.save(out_path)
    print(f'Saved -> {out_path}')
    print(f'Image : {W} x {H} px  |  {1/pixels_per_pulse:.2f} pulses/pixel')
    print(f'Pulse step : {pulse_step}  ({round(pulse_step * pixels_per_pulse):.0f} px between lines)')
    print()
    print(f'{"global pulse":>14}  {"px row":>8}')
    print(f'{"------------":>14}  {"------":>8}')
    pulse = first
    while pulse < pulse_stop:
        py = round((pulse - pulse_start) * pixels_per_pulse)
        print(f'{pulse:14d}  {py:8d}')
        pulse += pulse_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img',         default=DEFAULT_IMG)
    parser.add_argument('--out',         default=DEFAULT_OUT)
    parser.add_argument('--pulse-start', type=int, default=DEFAULT_PULSE_START)
    parser.add_argument('--pulse-stop',  type=int, default=DEFAULT_PULSE_STOP)
    parser.add_argument('--pulse-step',  type=int, default=DEFAULT_PULSE_STEP,
                        help='Global pulse interval between gridlines (default 5000).')
    args = parser.parse_args()

    if not os.path.exists(args.img):
        raise FileNotFoundError(f'Image not found: {args.img}')

    overlay_grid(args.img, args.out,
                 args.pulse_start, args.pulse_stop, args.pulse_step)


if __name__ == '__main__':
    main()