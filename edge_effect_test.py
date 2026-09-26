import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import SegformerForSemanticSegmentation

MODEL_ID = "restor/tcd-segformer-mit-b0"
SCALE = 6
TILE_SIZE = 768
OVERLAP = 128
DEFAULT_PAD_SOURCE_PX = 128


def parse_args():
    p = argparse.ArgumentParser(description="Phase A edge-effect test at 6x scale")
    p.add_argument("input_image", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/edge_effect_test"))
    p.add_argument("--pad-source-px", type=int, default=DEFAULT_PAD_SOURCE_PX,
                   help="Padding in source-image pixels before 6x scaling")
    return p.parse_args()


def to_tensor(image):
    a = np.asarray(image, dtype=np.float32) / 255.0
    x = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def find_tree_id(model):
    for class_id, label in model.config.id2label.items():
        if str(label).lower() == "tree":
            return int(class_id)
    raise RuntimeError(f"tree class not found: {model.config.id2label}")


def starts(length):
    if length <= TILE_SIZE:
        return [0]
    step = TILE_SIZE - OVERLAP
    values = list(range(0, max(length - TILE_SIZE, 0) + 1, step))
    last = length - TILE_SIZE
    if values[-1] != last:
        values.append(last)
    return values


def infer_tiled(image, model, tree_id):
    w, h = image.size
    score_sum = torch.zeros((h, w), dtype=torch.float32)
    weight_sum = torch.zeros((h, w), dtype=torch.float32)
    xs, ys = starts(w), starts(h)
    total = len(xs) * len(ys)
    n = 0
    for y in ys:
        for x in xs:
            tile = image.crop((x, y, min(x + TILE_SIZE, w), min(y + TILE_SIZE, h)))
            tw, th = tile.size
            with torch.no_grad():
                logits = model(pixel_values=to_tensor(tile)).logits
            logits = F.interpolate(logits, size=(th, tw), mode="bilinear", align_corners=False)
            prob = torch.softmax(logits, dim=1)[0, tree_id].cpu()
            score_sum[y:y+th, x:x+tw] += prob
            weight_sum[y:y+th, x:x+tw] += 1
            n += 1
            print(f"  tile {n}/{total}", end="\r")
    print()
    return score_sum / weight_sum.clamp_min(1)


def mask_and_overlay(source, prob, out_prefix):
    mask = (prob >= 0.5).to(torch.uint8).numpy() * 255
    mask_img = Image.fromarray(mask).resize(source.size, Image.Resampling.NEAREST)
    mask_arr = np.asarray(mask_img) > 0
    mask_img.save(out_prefix.with_name(out_prefix.name + "_mask.png"))
    overlay = source.copy()
    alpha = Image.fromarray(mask_arr.astype(np.uint8) * 110)
    overlay.paste(Image.new("RGB", source.size, (0, 255, 0)), (0, 0), alpha)
    overlay.save(out_prefix.with_name(out_prefix.name + "_overlay.png"))
    return mask_arr


def main():
    args = parse_args()
    if not args.input_image.exists():
        raise FileNotFoundError(args.input_image)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = Image.open(args.input_image).convert("RGB")

    print("Loading model...")
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    model.eval()
    tree_id = find_tree_id(model)
    print(f"Model loaded. tree_id={tree_id}")

    # Baseline: identical concept to the prior 6x experiment.
    scaled = source.resize((source.width*SCALE, source.height*SCALE), Image.Resampling.BICUBIC)
    print("\nBaseline 6x inference...")
    base_prob = infer_tiled(scaled, model, tree_id)
    base_mask = mask_and_overlay(source, base_prob, args.output_dir / "baseline6x")

    # Edge test: reflect-pad at source resolution, scale, infer, then crop back.
    p = args.pad_source_px
    padded = ImageOps.expand(source, border=p, fill=0)
    # Replace constant padding with reflected image content using NumPy.
    arr = np.asarray(source)
    reflected = np.pad(arr, ((p, p), (p, p), (0, 0)), mode="reflect")
    padded = Image.fromarray(reflected.astype(np.uint8))
    padded_scaled = padded.resize((padded.width*SCALE, padded.height*SCALE), Image.Resampling.BICUBIC)
    print(f"\nPadded 6x inference (reflect padding: {p} source px)...")
    padded_prob = infer_tiled(padded_scaled, model, tree_id)
    ps = p * SCALE
    cropped_prob = padded_prob[ps:ps+source.height*SCALE, ps:ps+source.width*SCALE]
    padded_mask = mask_and_overlay(source, cropped_prob, args.output_dir / "reflectpad6x")

    changed = base_mask != padded_mask
    change_img = Image.fromarray(changed.astype(np.uint8) * 255)
    change_img.save(args.output_dir / "changed_pixels.png")

    # Quantify whether changes concentrate near the image boundary.
    border = min(64, source.width // 4, source.height // 4)
    border_zone = np.zeros_like(changed, dtype=bool)
    border_zone[:border, :] = True
    border_zone[-border:, :] = True
    border_zone[:, :border] = True
    border_zone[:, -border:] = True
    interior = ~border_zone

    summary = {
        "scale": SCALE,
        "tile_size": TILE_SIZE,
        "overlap": OVERLAP,
        "reflect_padding_source_px": p,
        "baseline_tree_fraction": float(base_mask.mean()),
        "reflectpad_tree_fraction": float(padded_mask.mean()),
        "overall_changed_fraction": float(changed.mean()),
        "border_width_source_px": border,
        "border_changed_fraction": float(changed[border_zone].mean()),
        "interior_changed_fraction": float(changed[interior].mean()),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Saved to: {args.output_dir}")
    print("DONE")


if __name__ == "__main__":
    main()
