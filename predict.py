import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation


MODEL_ID = "restor/tcd-segformer-mit-b0"
DEFAULT_INPUT = Path(
    r"C:\tcd-upstream\docs\images\closed_canopy_example_rgb.jpg"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OAM-TCD SegFormer B0 semantic tree-cover inference."
    )
    parser.add_argument(
        "input_image",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input image path. Defaults to the upstream closed-canopy sample.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for mask and overlay outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_image = args.input_image
    output_dir = args.output_dir

    if not input_image.exists():
        raise FileNotFoundError(f"Input image not found: {input_image}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the input image without resizing it.
    image = Image.open(input_image).convert("RGB")
    width, height = image.size

    print(f"Input image: {input_image}")
    print(f"Image size: {width} x {height}")

    # Convert RGB values from 0-255 to a PyTorch tensor in 0-1 range.
    pixel_values = torch.tensor(
        list(image.getdata()),
        dtype=torch.float32,
    ).reshape(height, width, 3)
    pixel_values = pixel_values.permute(2, 0, 1).unsqueeze(0) / 255.0

    # Apply ImageNet normalization while preserving source image resolution.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    pixel_values = (pixel_values - mean) / std

    print("Loading model...")
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    model.eval()
    print("Model loaded.")

    print("Running inference...")
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    # Restore SegFormer logits to the original image dimensions.
    logits = F.interpolate(
        outputs.logits,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    prediction = logits.argmax(dim=1)[0].cpu()

    print("Classes:", model.config.id2label)
    print("Predicted class IDs:", torch.unique(prediction).tolist())

    # Resolve the tree class from model metadata rather than hard-coding its ID.
    tree_id = None
    for class_id, label in model.config.id2label.items():
        if str(label).lower() == "tree":
            tree_id = int(class_id)
            break

    if tree_id is None:
        raise RuntimeError(
            f"'tree' class was not found in id2label: {model.config.id2label}"
        )

    print("Tree class ID:", tree_id)

    stem = input_image.stem

    # Save a binary tree mask.
    mask = (prediction == tree_id).to(torch.uint8) * 255
    mask_array = mask.numpy()
    mask_image = Image.fromarray(mask_array)
    mask_path = output_dir / f"{stem}_mask.png"
    mask_image.save(mask_path)

    # Save a simple green overlay for visual inspection.
    overlay = image.copy()
    green = Image.new("RGB", image.size, (0, 255, 0))
    alpha = Image.fromarray(((mask_array > 0) * 110).astype("uint8"))
    overlay.paste(green, (0, 0), alpha)

    overlay_path = output_dir / f"{stem}_overlay.png"
    overlay.save(overlay_path)

    tree_fraction = (prediction == tree_id).float().mean().item()
    print(f"Tree fraction: {tree_fraction:.3f}")
    print(f"Saved: {mask_path}")
    print(f"Saved: {overlay_path}")
    print("DONE")


if __name__ == "__main__":
    main()
