from pathlib import Path
from PIL import Image

base_dir = Path("c:/Users/devin/MLUA")
test_dir = base_dir / "dataset" / "test"

imgs = list((test_dir / "images").glob("*.png"))
lbls = list((test_dir / "labels").glob("*.png"))
imgs_cut = list((test_dir / "images_cut").glob("*.png"))
lbls_cut = list((test_dir / "labels_cut").glob("*.png"))

print(f"images count: {len(imgs)}, labels count: {len(lbls)}")
print(f"images_cut count: {len(imgs_cut)}, labels_cut count: {len(lbls_cut)}")

if imgs:
    with Image.open(imgs[0]) as im:
        print(f"images[0] ({imgs[0].name}) size: {im.size}, mode: {im.mode}")

if imgs_cut:
    with Image.open(imgs_cut[0]) as im:
        print(f"images_cut[0] ({imgs_cut[0].name}) size: {im.size}, mode: {im.mode}")

if lbls:
    with Image.open(lbls[0]) as im:
        print(f"labels[0] ({lbls[0].name}) size: {im.size}, mode: {im.mode}")
