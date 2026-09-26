from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation


MODEL_ID = "restor/tcd-segformer-mit-b0"
INPUT_IMAGE = Path(
    r"C:\tcd-upstream\docs\images\closed_canopy_example_rgb.jpg"
)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# Load the input image without resizing it.
image = Image.open(INPUT_IMAGE).convert("RGB")
width, height = image.size

print(f"Input image: {INPUT_IMAGE}")
print(f"Image size: {width} x {height}")


# Convert RGB values from 0-255 to a PyTorch tensor in 0-1 range.
pixel_values = torch.tensor(
    list(image.getdata()),
    dtype=torch.float32,
).reshape(height, width, 3)
pixel_values = pixel_values.permute(2, 0, 1).unsqueeze(0) / 255.0


# Apply ImageNet normalization while preserving the source image resolution.
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


# Save a binary tree mask.
mask = (prediction == tree_id).to(torch.uint8) * 255
mask_array = mask.numpy()
mask_image = Image.fromarray(mask_array)
mask_path = OUTPUT_DIR / "mask.png"
mask_image.save(mask_path)


# Save a simple green overlay for visual inspection.
overlay = image.copy()
green = Image.new("RGB", image.size, (0, 255, 0))
alpha = Image.fromarray(((mask_array > 0) * 110).astype("uint8"))
overlay.paste(green, (0, 0), alpha)

overlay_path = OUTPUT_DIR / "overlay.png"
overlay.save(overlay_path)


tree_fraction = (prediction == tree_id).float().mean().item()
print(f"Tree fraction: {tree_fraction:.3f}")
print(f"Saved: {mask_path}")
print(f"Saved: {overlay_path}")
print("DONE")
