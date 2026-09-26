import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation

MODEL_ID = "restor/tcd-segformer-mit-b0"
SCALES = (1, 2, 4, 6)
TILE_SIZE = 768
OVERLAP = 128


def parse_args():
    p = argparse.ArgumentParser(description="Phase A GSD/scale sensitivity test for TCD SegFormer B0")
    p.add_argument("input_image", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/gsd_test"))
    p.add_argument("--source-gsd", type=float, default=0.597164, help="Source GSD in m/pixel (metadata only)")
    return p.parse_args()


def to_tensor(image):
    w, h = image.size
    x = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(h, w, 3)
    x = x.permute(2, 0, 1).unsqueeze(0) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def find_tree_id(model):
    for class_id, label in model.config.id2label.items():
        if str(label).lower() == "tree":
            return int(class_id)
    raise RuntimeError(f"tree class not found: {model.config.id2label}")


def starts(length, tile, overlap):
    if length <= tile:
        return [0]
    step = tile - overlap
    out = list(range(0, max(length - tile, 0) + 1, step))
    last = length - tile
    if out[-1] != last:
        out.append(last)
    return out


def infer_tiled(image, model, tree_id):
    w, h = image.size
    score_sum = torch.zeros((h, w), dtype=torch.float32)
    weight_sum = torch.zeros((h, w), dtype=torch.float32)
    xs = starts(w, TILE_SIZE, OVERLAP)
    ys = starts(h, TILE_SIZE, OVERLAP)
    total = len(xs) * len(ys)
    done = 0

    for y in ys:
        for x in xs:
            tile = image.crop((x, y, min(x + TILE_SIZE, w), min(y + TILE_SIZE, h)))
            tw, th = tile.size
            pixel_values = to_tensor(tile)
            with torch.no_grad():
                logits = model(pixel_values=pixel_values).logits
            logits = F.interpolate(logits, size=(th, tw), mode="bilinear", align_corners=False)
            probs = torch.softmax(logits, dim=1)[0, tree_id].cpu()
            score_sum[y:y+th, x:x+tw] += probs
            weight_sum[y:y+th, x:x+tw] += 1.0
            done += 1
            print(f"  tile {done}/{total}", end="\r")
    print()
    return score_sum / weight_sum.clamp_min(1.0)


def save_outputs(source, scaled, tree_prob, scale, out_dir):
    sw, sh = scaled.size
    mask_scaled = (tree_prob >= 0.5).to(torch.uint8).numpy() * 255
    mask_scaled_img = Image.fromarray(mask_scaled)

    # Bring the result back to the original grid for apples-to-apples comparison.
    mask_original = mask_scaled_img.resize(source.size, resample=Image.Resampling.NEAREST)
    mask_arr = torch.from_numpy(__import__("numpy").array(mask_original)) > 0

    scale_dir = out_dir / f"scale_{scale}x"
    scale_dir.mkdir(parents=True, exist_ok=True)
    mask_original.save(scale_dir / "mask_original_grid.png")

    green = Image.new("RGB", source.size, (0, 255, 0))
    alpha = Image.fromarray((mask_arr.numpy().astype("uint8") * 110))
    overlay = source.copy()
    overlay.paste(green, (0, 0), alpha)
    overlay.save(scale_dir / "overlay_original_grid.png")

    tree_fraction = mask_arr.float().mean().item()
    return tree_fraction


def main():
    args = parse_args()
    if not args.input_image.exists():
        raise FileNotFoundError(args.input_image)

    source = Image.open(args.input_image).convert("RGB")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input: {args.input_image}")
    print(f"Source size: {source.width} x {source.height}")
    print(f"Source GSD (metadata): {args.source_gsd:.6f} m/px")
    print("Loading model...")
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    model.eval()
    tree_id = find_tree_id(model)
    print(f"Model loaded. tree_id={tree_id}")

    rows = []
    for scale in SCALES:
        target_size = (source.width * scale, source.height * scale)
        effective_scale_gsd = args.source_gsd / scale
        print(f"\n=== scale {scale}x: {target_size[0]} x {target_size[1]} (scale-equivalent {effective_scale_gsd:.4f} m/px) ===")
        scaled = source if scale == 1 else source.resize(target_size, resample=Image.Resampling.BICUBIC)
        tree_prob = infer_tiled(scaled, model, tree_id)
        fraction = save_outputs(source, scaled, tree_prob, scale, args.output_dir)
        rows.append({
            "scale": scale,
            "width_px": target_size[0],
            "height_px": target_size[1],
            "source_gsd_m_per_px": args.source_gsd,
            "scale_equivalent_gsd_m_per_px": effective_scale_gsd,
            "tree_fraction": fraction,
        })
        print(f"Tree fraction (original grid): {fraction:.4f}")

    comparison = args.output_dir / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    with (comparison / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (comparison / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"\nSaved comparison summary to: {comparison}")
    print("DONE")


if __name__ == "__main__":
    main()
