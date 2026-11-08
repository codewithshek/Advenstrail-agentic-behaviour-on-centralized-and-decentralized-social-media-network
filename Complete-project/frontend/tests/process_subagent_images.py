from pathlib import Path
from PIL import Image

brain_dir = Path(r"C:\Users\dabhi\.gemini\antigravity\brain\3d474815-5a7e-4660-9dc8-768eeef4456f")
dest_dir = Path(__file__).resolve().parent.parent / "assets" / "editions"
dest_dir.mkdir(parents=True, exist_ok=True)

# 1. Hero Swarm -> WebP
hero_src = brain_dir / "hero_chiaroscuro_swarm_1794138599505.jpg"
if hero_src.exists():
    im = Image.open(hero_src).convert("RGB")
    hero_dst = dest_dir / "hero-chiaroscuro-swarm.webp"
    im.save(hero_dst, "WEBP", quality=92)
    print(f"Saved {hero_dst} ({hero_dst.stat().st_size} bytes)")

# 2. Federated Map -> WebP
map_src = brain_dir / "renaissance_celestial_map_1794138692681.jpg"
if map_src.exists():
    im = Image.open(map_src).convert("RGB")
    map_dst = dest_dir / "federated-armillary-map.webp"
    im.save(map_dst, "WEBP", quality=92)
    print(f"Saved {map_dst} ({map_dst.stat().st_size} bytes)")

# 3. Mercury Bust -> Alpha Cutout PNG
mercury_src = brain_dir / "mercury_automaton_bust_1794138672405.jpg"
if mercury_src.exists():
    im = Image.open(mercury_src).convert("RGBA")
    # Convert black studio background to transparent alpha
    datas = im.getdata()
    newData = []
    for item in datas:
        r, g, b, a = item
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        if lum < 12:
            newData.append((r, g, b, 0))
        elif lum < 32:
            alpha = int(255 * ((lum - 12) / 20))
            newData.append((r, g, b, alpha))
        else:
            newData.append((r, g, b, 255))
    im.putdata(newData)
    mercury_dst = dest_dir / "cutout-mercury-agent.png"
    im.save(mercury_dst, "PNG")
    print(f"Saved {mercury_dst} ({mercury_dst.stat().st_size} bytes)")
