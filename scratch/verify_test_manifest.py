import os
import hashlib
import json
from pathlib import Path
from PIL import Image

base_dir = Path("c:/Users/devin/MLUA")
test_dir = base_dir / "dataset" / "test"
images_dir = test_dir / "images"
labels_dir = test_dir / "labels"
images_cut_dir = test_dir / "images_cut"
labels_cut_dir = test_dir / "labels_cut"

image_files = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))
label_files = sorted(list(labels_dir.glob("*.png")) + list(labels_dir.glob("*.jpg")))
images_cut_files = sorted(list(images_cut_dir.glob("*.png")) + list(images_cut_dir.glob("*.jpg")))
labels_cut_files = sorted(list(labels_cut_dir.glob("*.png")) + list(labels_cut_dir.glob("*.jpg")))

print(f"Full Panoramics -> Images: {len(image_files)}, Labels: {len(label_files)}")
print(f"Cut Patches (384x384) -> Images Cut: {len(images_cut_files)}, Labels Cut: {len(labels_cut_files)}")

assert len(image_files) == 100, f"Expected 100 test panoramics, found {len(image_files)}"
assert len(label_files) == 100, f"Expected 100 test labels, found {len(label_files)}"
assert len(images_cut_files) == 2100, f"Expected 2100 test cut images (100x21), found {len(images_cut_files)}"
assert len(labels_cut_files) == 2100, f"Expected 2100 test cut labels (100x21), found {len(labels_cut_files)}"

# Sample check dimensions
with Image.open(image_files[0]) as img:
    w, h = img.size
    print(f"Sample Panoramic Dimensions: {w}x{h} (Expected: 1536x768)")
    assert (w, h) == (1536, 768)

with Image.open(images_cut_files[0]) as patch:
    pw, ph = patch.size
    print(f"Sample Cut Patch Dimensions: {pw}x{ph} (Expected: 384x384)")
    assert (pw, ph) == (384, 384)

# Create deterministic checksum inventory
manifest = {
    "total_panoramic_cases": len(image_files),
    "total_cut_patches": len(images_cut_files),
    "patches_per_panoramic": 21,
    "panoramic_resolution": [768, 1536],
    "patch_resolution": [384, 384],
    "stride": 192,
    "grid_shape": [3, 7],
    "checksums": {
        "images_first_5_sha256": {},
        "labels_first_5_sha256": {},
    }
}

for img_p in image_files[:5]:
    with open(img_p, "rb") as f:
        manifest["checksums"]["images_first_5_sha256"][img_p.name] = hashlib.sha256(f.read()).hexdigest()

for lbl_p in label_files[:5]:
    with open(lbl_p, "rb") as f:
        manifest["checksums"]["labels_first_5_sha256"][lbl_p.name] = hashlib.sha256(f.read()).hexdigest()

# Save manifest to outputs/evaluation/final_100/TEST_SET_MANIFEST.json
out_eval_dir = base_dir / "outputs" / "evaluation" / "final_100"
out_eval_dir.mkdir(parents=True, exist_ok=True)
manifest_path = out_eval_dir / "TEST_SET_MANIFEST.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Successfully generated sealed benchmark manifest at {manifest_path}")
