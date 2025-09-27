Hopefully I will be able to create some useful nodes for ComfyUI. I just started making these, using LLMs for coding purposes, with no previous coding experience. However, I can see myself learning more and more.

### Path Tool
This node dynamically generates a filename prefix for saving files into an organized folder structure.

What it does:
* It constructs a path in the format: `[root_directory]/[current_date]/[filename]`.
* You can choose between a primary `base_path` and an `alt_path` using a boolean toggle.
* It automatically creates a subfolder named with the current date in YYYY_MM_DD format (e.g., 2023_10_27).
* It sanitizes the path and filename to remove any invalid characters, ensuring a safe output.
* The final string is intended to be connected to the `filename_prefix` input of a "Save Image" or similar node.

### Color Match Falloff
**Code is a modified version of Kijai's ComfyUI-KJNodes Color Match node**
https://github.com/kijai/ComfyUI-KJNodes

It keeps the original functionality with a small addition:

* Smoothly fades the color transfer effect over a sequence of frames (`falloff_duration`).
* Uses an ease-in-out curve (strength 1 → 0) to reduce color flicker.
* Handy for stitching video clips generated in separate batches (last→first) e.g. with WAN 2.1.

### Color Picker
Minimal color picker that outputs two convenient formats you can wire into other nodes.

* **Input**: `COLOR` widget (also accepts manual `"R,G,B"` or JSON `[R,G,B]`, 0–255).
* **Outputs**: `hex` (e.g., `#7F7F7F`) and `rgb_csv` (e.g., `127,127,127`).
* Great companion for `Sequence Blend` (`rgb_csv` → `blend_color_rgb`).

### Sequence Blend
Flexible sequence blending with either a solid color or an image, including common blend modes and HSL-based modes.

* **Modes**: `normal`, `screen`, `additive color`, `overlay`, `multiply`, `color burn`, `difference`, `saturation`, `hue`, `color`.
* **Source**: `color` (via `blend_color_rgb`) or `image` (`layer_sequence`).
* **Strength per-frame**: single float or list/tensor via `scales_any`; optional `override_last_strength`.
* **HSL modes**: `saturation` (H+L from base, S from source), `hue` (S+L from base, H from source), `color` (L from base, H+S from source).
* **Resize**: source auto-resized to base using selected interpolation (`lanczos`, `bilinear`, `bicubic`, `nearest`, `area` via OpenCV if available).
* **Alpha**: RGBA supported; source treated as premultiplied; base alpha preserved.

### Image Difference to Alpha
Builds a difference overlay between **A** and **B** (or a solid **HEX** color) and optionally composites it over a background.

* **Overlay RGB**: `diff` (colorized `abs(A−B)`), `image_b` (when `source=image`), or `black`.
* **Overlay Alpha**: strength of the difference by channel `max` (default) or `mean`; optional normalization.
* **Boost**: multiply difference to make subtle changes visible.
* **Special**: when `source=hex_color` **and** `image_b` is connected, the overlay (A vs HEX) is composited **over** `image_b` (`over`: `overlay*alpha + bg*(1−alpha)`) and returned. Otherwise the node outputs the standalone RGBA overlay.
* **Batch/size**: handles batches; auto-resizes secondary inputs to match A.

### Sequence Content Zoom
Zoom-in/out while keeping the frame size unchanged.

* **s ≥ 1.0 (zoom-in)**: center (or aligned) crop + resize back to full frame.
* **s < 1.0 (zoom-out)**: scale content down and pad with a background color (`pad_rgb`).
* **Alignment**: `crop_align` and `pad_align` (`center`, corners, edges).
* **Scales**: single or list/tensor via `scales_any`; optional `override_last_scale`.
* **Interpolation**: `lanczos`, `bilinear`, `bicubic`, `nearest`, `area` (OpenCV if available).
* **Alpha**: RGBA preserved; alpha resized bilinearly.

---

*More nodes and tweaks soon ✌️*

