from pathlib import Path
import math
import random
from PIL import Image, ImageDraw, ImageFilter

dest_dir = Path(__file__).resolve().parent.parent / "assets" / "editions"
dest_dir.mkdir(parents=True, exist_ok=True)

# 1. Chapter 02 Centralized Sun Hub (1920x1080 WebP)
def generate_centralized_hub():
    w, h = 1920, 1080
    im = Image.new("RGB", (w, h), (7, 7, 9))
    draw = ImageDraw.Draw(im)

    cx, cy = w // 2, h // 2

    # Concentric astrolabe rings
    for r in [60, 120, 220, 360, 520, 700]:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(60, 50, 25), width=2)

    # Radial spokes (spoke network)
    for deg in range(0, 360, 15):
        rad = math.radians(deg)
        x2 = cx + int(math.cos(rad) * 850)
        y2 = cy + int(math.sin(rad) * 850)
        draw.line([(cx, cy), (x2, y2)], fill=(38, 32, 18), width=1)

    # Golden Sun-King Astrolabe Center
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    for r in range(400, 0, -4):
        alpha = int(120 * ((400 - r) / 400) ** 1.6)
        glow_draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(235, 87, 87, alpha))
    im.paste(Image.alpha_composite(im.convert("RGBA"), glow), (0, 0))

    draw = ImageDraw.Draw(im)
    draw.ellipse([cx - 80, cy - 80, cx + 80, cy + 80], fill=(255, 220, 90), outline=(235, 87, 87), width=6)

    # 42 Synchronized mechanical puppet hands / bots pulling threads toward the central seed hub
    random.seed(101)
    for b in range(42):
        angle = (b / 42) * 2 * math.pi
        dist = random.randint(320, 680)
        bx = cx + int(math.cos(angle) * dist)
        by = cy + int(math.sin(angle) * dist)
        # Pulling thread (red/gold line)
        draw.line([(cx, cy), (bx, by)], fill=(235, 87, 87, 180), width=2)
        # Node head
        draw.ellipse([bx - 12, by - 12, bx + 12, by + 12], fill=(212, 175, 55), outline=(255, 255, 255), width=2)

    out_path = dest_dir / "centralized-sun-hub.webp"
    im.save(out_path, "WEBP", quality=90)
    print(f"Generated {out_path} ({out_path.stat().st_size} bytes)")


# 2. Foreground Cutout: Federated Astrolabe (1200x1200 PNG with Transparent Alpha)
def generate_cutout_astrolabe():
    w, h = 1200, 1200
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)

    cx, cy = w // 2, h // 2

    # Renaissance Brass Armillary Sphere Rings
    rings = [
        (480, 14, (212, 175, 55, 255)),
        (420, 10, (180, 140, 40, 255)),
        (350, 8, (245, 215, 110, 255)),
        (260, 6, (212, 175, 55, 255)),
        (160, 6, (167, 139, 250, 255)),
    ]
    for r, width, col in rings:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=width)

    # Oblique Zodiac Band (Tilted Ellipse)
    for deg in range(0, 360, 3):
        rad = math.radians(deg)
        x = cx + int(440 * math.cos(rad) * math.cos(0.4) - 220 * math.sin(rad) * math.sin(0.4))
        y = cy + int(440 * math.cos(rad) * math.sin(0.4) + 220 * math.sin(rad) * math.cos(0.4))
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(212, 175, 55, 255))

    # Central Earth / Pod Core
    draw.ellipse([cx - 50, cy - 50, cx + 50, cy + 50], fill=(30, 25, 35, 255), outline=(212, 175, 55, 255), width=4)
    # Suspension Ring
    draw.ellipse([cx - 40, cy - 540, cx + 40, cy - 460], outline=(212, 175, 55, 255), width=10)

    out_path = dest_dir / "cutout-federated-astrolabe.png"
    im.save(out_path, "PNG")
    print(f"Generated {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    generate_centralized_hub()
    generate_cutout_astrolabe()
