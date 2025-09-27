import os
import datetime
import math
import torch
import json 
import numpy as np
import re

class ColorPicker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Ten typ "COLOR" dostanie customowy widget z pliku JS
                "color": ("COLOR", {"default": "#808080"}),
            },
            "optional": {
                # Jeśli chcesz: można tu dodać też "alpha" albo inne opcje
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("hex", "rgb_csv")
    FUNCTION = "emit"
    CATEGORY = "Utils/Color"
    DESCRIPTION = """
Prosty color picker. Zwraca:
- HEX: np. "#7F7F7F"
- RGB CSV: np. "127,127,127" (idealne do podpięcia pod `SequenceBlend.blend_color_rgb`)

Wejście `color` akceptuje też ręcznie wpisane "R,G,B" lub JSON "[R,G,B]".
"""

    def emit(self, color):
        hex_str, rgb_csv = self._normalize_color(color)
        return (hex_str, rgb_csv)

    # ---------- helpers ----------

    def _normalize_color(self, v):
        """
        Przyjmij: "#RRGGBB" lub "R,G,B" lub "[R,G,B]"
        Zwróć: ("#RRGGBB", "R,G,B")
        """
        if v is None:
            r, g, b = 127, 127, 127
            return ("#7F7F7F", f"{r},{g},{b}")

        s = str(v).strip()

        # JSON [r,g,b]
        try:
            arr = json.loads(s)
            if isinstance(arr, (list, tuple)) and len(arr) >= 3:
                r, g, b = [int(np.clip(float(arr[i]), 0, 255)) for i in range(3)]
                return (self._rgb_to_hex(r, g, b), f"{r},{g},{b}")
        except Exception:
            pass

        # CSV "r,g,b"
        parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
        if len(parts) >= 3:
            try:
                r = int(np.clip(float(parts[0]), 0, 255))
                g = int(np.clip(float(parts[1]), 0, 255))
                b = int(np.clip(float(parts[2]), 0, 255))
                return (self._rgb_to_hex(r, g, b), f"{r},{g},{b}")
            except Exception:
                pass

        # HEX "#RRGGBB" lub "RRGGBB"
        s = s.upper()
        if s.startswith("#"):
            s = s[1:]
        if len(s) == 6 and all(c in "0123456789ABCDEF" for c in s):
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
            return (f"#{s}", f"{r},{g},{b}")

        # fallback
        return ("#7F7F7F", "127,127,127")

    def _rgb_to_hex(self, r, g, b):
        return f"#{int(r):02X}{int(g):02X}{int(b):02X}"

import torch
import numpy as np
import json
import re

class SequenceBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_sequence": ("IMAGE",),
                "mode": (
                    [
                        "normal",
                        "screen",
                        "additive color",
                        "overlay",
                        "multiply",
                        "color burn",
                        "difference",
                        "saturation",
                        "hue",
                        "color",
                    ],
                    {"default": "normal"}
                ),
                "blend_color_rgb": ("STRING", {"default": "127,127,127", "multiline": False}),
                "source": (["color", "image"], {"default": "color"}),
                "interpolation": (
                    ["lanczos", "bilinear", "bicubic", "nearest", "area"],
                    {"default": "lanczos"}
                ),
                "override_last_strength": ("BOOLEAN", {"default": False}),
                "override_value": ("FLOAT", {"default": 1.0}),
            },
            "optional": {
                "layer_sequence": ("IMAGE",),              # tylko gdy source=image
                "scales_any": ("FLOAT", {"default": 1.0}), # siła: float lub lista
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blend_sequence"
    CATEGORY = "Image/Blend"
    DESCRIPTION = """
Blend z kolorem lub obrazem: normal, screen, additive color, overlay, multiply, color burn, difference, saturation, hue, color.
- `saturation`: H+L z bazy, S ze źródła.
- `hue`: S+L z bazy, H ze źródła.
- `color`: L z bazy, H+S ze źródła.
- `source=color` działa bez podłączonej warstwy.
- Siła z `scales_any` (float/lista); można nadpisać ostatnią wartość (override_*).
"""

    def blend_sequence(self, base_sequence, mode, blend_color_rgb, source, interpolation,
                       override_last_strength, override_value, layer_sequence=None, scales_any=1.0):
        try:
            from PIL import Image
        except ImportError:
            raise Exception("Brak biblioteki Pillow. Zainstaluj: pip install pillow")

        # ----- baza -----
        base = base_sequence
        if not isinstance(base, torch.Tensor):
            raise ValueError("base_sequence musi być tensorem torch (BxHxWxC)")
        device = base.device
        base_cpu = base.detach().clamp(0.0, 1.0).cpu().to(torch.float32)
        B, H, W, C = base_cpu.shape
        if C not in (3, 4):
            raise ValueError(f"Obsługiwane kanały: 3 (RGB) lub 4 (RGBA), otrzymano: {C}")

        # ----- siły -----
        strengths = self._coerce_any(scales_any)
        if not strengths:
            strengths = [1.0]
        if override_last_strength:
            try:
                ov = float(override_value)
            except Exception:
                ov = 1.0
            if len(strengths) == 0:
                strengths = [ov]
            else:
                strengths[-1] = ov

        # ----- źródło -----
        use_image_src = (str(source).lower() == "image")
        if use_image_src:
            if not isinstance(layer_sequence, torch.Tensor):
                raise ValueError("Ustawiono source=image, ale `layer_sequence` nie jest podłączone.")
            src_cpu = layer_sequence.detach().clamp(0.0, 1.0).cpu().to(torch.float32)
            SB, SH, SW, SC = src_cpu.shape
            if SC not in (3, 4):
                raise ValueError(f"layer_sequence musi mieć 3 lub 4 kanały, otrzymano: {SC}")
        else:
            cr, cg, cb = self._parse_rgb(blend_color_rgb)  # 0..1
            color_rgb = np.array([cr, cg, cb], dtype=np.float32)

        # ----- resize setup -----
        pil_methods = {
            "lanczos": Image.LANCZOS,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "nearest": Image.NEAREST,
        }
        use_cv2_area = False
        if interpolation == "area":
            try:
                import cv2
                use_cv2_area = True
            except Exception:
                use_cv2_area = False

        out_frames = []
        base_np = base_cpu.numpy()

        for i in range(B):
            t = strengths[min(i, len(strengths) - 1)]
            try:
                t = float(t)
            except Exception:
                t = 1.0
            t = float(np.clip(t, 0.0, 1.0))  # 0..1

            base_frame = base_np[i]  # H,W,C
            if C == 4:
                base_rgb = base_frame[:, :, :3]
                base_a = base_frame[:, :, 3:4]
            else:
                base_rgb = base_frame
                base_a = None

            # przygotuj src_rgb
            if use_image_src:
                j = min(i, src_cpu.shape[0] - 1)
                src_frame = src_cpu[j].numpy()
                if src_frame.shape[0] != H or src_frame.shape[1] != W:
                    if src_frame.shape[2] == 4:
                        srgb = src_frame[:, :, :3]
                        sa = src_frame[:, :, 3]
                        srgb = self._resize_rgb(srgb, W, H, interpolation, use_cv2_area, Image, pil_methods)
                        sa = self._resize_alpha(sa[:, :, None], W, H, use_cv2_area, Image).squeeze(-1)
                        src_rgb = srgb * sa[..., None]
                    else:
                        src_rgb = self._resize_rgb(src_frame, W, H, interpolation, use_cv2_area, Image, pil_methods)
                else:
                    if src_frame.shape[2] == 4:
                        src_rgb = src_frame[:, :, :3] * src_frame[:, :, 3:4]
                    else:
                        src_rgb = src_frame[:, :, :3]
            else:
                src_rgb = np.broadcast_to(color_rgb, (H, W, 3)).copy()

            # tryb (z aliasami) + HSL tryby
            mode_key = self._normalize_mode(mode)
            if mode_key == "saturation":
                blend_rgb = self._blend_saturation(base_rgb, src_rgb)
            elif mode_key == "hue":
                blend_rgb = self._blend_hue(base_rgb, src_rgb)
            elif mode_key == "color":
                blend_rgb = self._blend_color(base_rgb, src_rgb)
            else:
                blend_rgb = self._blend_mode(mode_key, base_rgb, src_rgb)

            # miks z siłą t
            out_rgb = (1.0 - t) * base_rgb + t * blend_rgb
            out_rgb = np.clip(out_rgb, 0.0, 1.0)

            if base_a is not None:
                out = np.concatenate([out_rgb, base_a], axis=-1)
            else:
                out = out_rgb

            out_frames.append(torch.from_numpy(out))

        out_tensor = torch.stack(out_frames, dim=0).to(torch.float32).clamp(0.0, 1.0)
        return (out_tensor.to(device),)

    # ---------------- helpers ----------------

    def _normalize_mode(self, mode):
        m = str(mode).strip().lower().replace(" ", "_")
        if m == "additive":
            m = "additive_color"
        if m in ("color_burn", "burn"):
            m = "color_burn"
        return m

    def _blend_mode(self, mode, a, b):
        if mode == "normal":
            m = b
        elif mode == "screen":
            m = 1.0 - (1.0 - a) * (1.0 - b)
        elif mode == "additive_color":
            m = np.clip(a + b, 0.0, 1.0)
        elif mode == "overlay":
            m = np.where(a <= 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))
        elif mode == "multiply":
            m = a * b
        elif mode == "color_burn":
            eps = 1e-6
            m = 1.0 - np.minimum(1.0, (1.0 - a) / np.clip(b, eps, 1.0))
            m = np.clip(m, 0.0, 1.0)
        elif mode == "difference":
            m = np.abs(a - b)
        else:
            m = b
        return m

    def _blend_saturation(self, base_rgb, src_rgb):
        # H+L z bazy, S ze źródła
        hb, sb, lb = self._rgb_to_hsl_np(base_rgb)
        hs, ss, ls = self._rgb_to_hsl_np(src_rgb)
        return self._hsl_to_rgb_np(hb, ss, lb)

    def _blend_hue(self, base_rgb, src_rgb):
        # S+L z bazy, H ze źródła
        hb, sb, lb = self._rgb_to_hsl_np(base_rgb)
        hs, ss, ls = self._rgb_to_hsl_np(src_rgb)
        return self._hsl_to_rgb_np(hs, sb, lb)

    def _blend_color(self, base_rgb, src_rgb):
        # L z bazy, H+S ze źródła
        hb, sb, lb = self._rgb_to_hsl_np(base_rgb)
        hs, ss, ls = self._rgb_to_hsl_np(src_rgb)
        return self._hsl_to_rgb_np(hs, ss, lb)

    # ---------- HSL <-> RGB (wektorowo, numpy) ----------

    def _rgb_to_hsl_np(self, rgb):
        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]

        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        l = (minc + maxc) / 2.0

        s = np.zeros_like(l)
        h = np.zeros_like(l)

        diff = maxc - minc
        mask = diff > 1e-12

        # Saturation
        denom1 = (maxc + minc) + 1e-12
        denom2 = (2.0 - maxc - minc) + 1e-12
        s_low = diff / denom1
        s_high = diff / denom2
        s = np.where(mask & (l <= 0.5), s_low, s)
        s = np.where(mask & (l > 0.5), s_high, s)

        # Hue
        rc = (maxc - r) / (diff + 1e-12)
        gc = (maxc - g) / (diff + 1e-12)
        bc = (maxc - b) / (diff + 1e-12)

        h_r = (bc - gc)
        h_g = (2.0 + rc - bc)
        h_b = (4.0 + gc - rc)

        h = np.where((mask) & (maxc == r), h_r, h)
        h = np.where((mask) & (maxc == g), h_g, h)
        h = np.where((mask) & (maxc == b), h_b, h)

        h = (h / 6.0) % 1.0
        return h, s, l

    def _hue2rgb(self, m1, m2, h):
        h = (h % 1.0)
        out = np.empty_like(h)

        cond1 = h < 1.0/6.0
        cond2 = (h >= 1.0/6.0) & (h < 1.0/2.0)
        cond3 = (h >= 1.0/2.0) & (h < 2.0/3.0)

        out = np.where(cond1, m1 + (m2 - m1) * 6.0 * h, 0.0)
        out = np.where(cond2, m2, out)
        out = np.where(cond3, m1 + (m2 - m1) * (2.0/3.0 - h) * 6.0, out)
        out = np.where(~(cond1 | cond2 | cond3), m1, out)
        return out

    def _hsl_to_rgb_np(self, h, s, l):
        m2 = np.where(l <= 0.5, l * (1.0 + s), l + s - l * s)
        m1 = 2.0 * l - m2

        achr = s <= 1e-12
        r = np.where(achr, l, self._hue2rgb(m1, m2, h + 1.0/3.0))
        g = np.where(achr, l, self._hue2rgb(m1, m2, h))
        b = np.where(achr, l, self._hue2rgb(m1, m2, h - 1.0/3.0))

        rgb = np.stack([r, g, b], axis=-1)
        return np.clip(rgb, 0.0, 1.0)

    # ---------- pozostałe utilsy ----------

    def _coerce_any(self, value):
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            try:
                return [float(x) for x in value.detach().cpu().flatten().tolist()]
            except Exception:
                return []
        if isinstance(value, np.ndarray):
            try:
                return [float(x) for x in value.flatten().tolist()]
            except Exception:
                return []
        if isinstance(value, (int, float)):
            return [float(value)]
        if isinstance(value, (list, tuple)):
            out = []
            for x in value:
                try:
                    out.append(float(x))
                except Exception:
                    continue
            return out
        return []

    def _parse_rgb(self, s):
        """
        Akceptuje:
          - HEX: "#RRGGBB" / "RRGGBB" / "#RGB" / "RGB"
          - CSS: "rgb(r,g,b)" lub "rgba(r,g,b,a)" (alfa ignorowana)
          - CSV: "r,g,b" lub "r;g;b"
          - JSON: "[r,g,b]"
        Wartości mogą być w 0..255 LUB 0..1. Zwraca (r,g,b) w 0..1.
        """
        # domyślny neutralny szary (pasuje do starego zachowania)
        default = (127/255.0, 127/255.0, 127/255.0)

        if s is None:
            return default

        txt = str(s).strip()
        if not txt:
            return default

        # --- HEX (#RRGGBB / #RGB) ---
        t = txt.lower().strip()
        if t.startswith("#"):
            t = t[1:]
        if re.fullmatch(r"[0-9a-f]{6}", t or ""):
            r = int(t[0:2], 16); g = int(t[2:4], 16); b = int(t[4:6], 16)
            return (r/255.0, g/255.0, b/255.0)
        if re.fullmatch(r"[0-9a-f]{3}", t or ""):
            r = int(t[0]*2, 16); g = int(t[1]*2, 16); b = int(t[2]*2, 16)
            return (r/255.0, g/255.0, b/255.0)

        # --- CSS rgb/rgba ---
        m = re.match(r"rgba?\\s*\\(\\s*([0-9.]+)\\s*,\\s*([0-9.]+)\\s*,\\s*([0-9.]+)(?:\\s*,\\s*[0-9.]+)?\\s*\\)", txt, flags=re.IGNORECASE)
        if m:
            try:
                rr = float(m.group(1)); gg = float(m.group(2)); bb = float(m.group(3))
                if rr <= 1.0 and gg <= 1.0 and bb <= 1.0:
                    return (np.clip(rr,0,1), np.clip(gg,0,1), np.clip(bb,0,1))
                else:
                    return (np.clip(rr,0,255)/255.0, np.clip(gg,0,255)/255.0, np.clip(bb,0,255)/255.0)
            except Exception:
                pass

        # --- JSON [r,g,b] ---
        if (txt.startswith("[") and txt.endswith("]")) or (txt.startswith("(") and txt.endswith(")")):
            try:
                val = json.loads(txt.replace("(", "[").replace(")", "]"))
                if isinstance(val, (list, tuple)) and len(val) >= 3:
                    rr, gg, bb = float(val[0]), float(val[1]), float(val[2])
                    if rr <= 1.0 and gg <= 1.0 and bb <= 1.0:
                        return (np.clip(rr,0,1), np.clip(gg,0,1), np.clip(bb,0,1))
                    else:
                        return (np.clip(rr,0,255)/255.0, np.clip(gg,0,255)/255.0, np.clip(bb,0,255)/255.0)
            except Exception:
                pass

        # --- CSV "r,g,b" (lub z ";") ---
        t2 = txt.replace(";", ",")
        parts = [p.strip() for p in t2.split(",") if p.strip()]
        if len(parts) >= 3:
            try:
                rr = float(parts[0]); gg = float(parts[1]); bb = float(parts[2])
                if rr <= 1.0 and gg <= 1.0 and bb <= 1.0:
                    return (np.clip(rr,0,1), np.clip(gg,0,1), np.clip(bb,0,1))
                else:
                    return (np.clip(rr,0,255)/255.0, np.clip(gg,0,255)/255.0, np.clip(bb,0,255)/255.0)
            except Exception:
                return default

        # fallback
        return default

    def _to_pil_rgb(self, arr, Image):
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def _to_pil_gray(self, arr, Image):
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")

    def _resize_rgb(self, src_np, out_w, out_h, interpolation, use_cv2_area, Image, pil_methods):
        if interpolation == "area" and use_cv2_area:
            import cv2
            return cv2.resize(
                src_np.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_AREA
            ).astype(np.float32)
        else:
            pil_img = self._to_pil_rgb(src_np, Image)
            resample = pil_methods.get(interpolation, Image.BILINEAR)
            pil_img = pil_img.resize((out_w, out_h), resample=resample)
            return np.asarray(pil_img).astype(np.float32) / 255.0

    def _resize_alpha(self, a_np, out_w, out_h, use_cv2_area, Image):
        if a_np.ndim == 3 and a_np.shape[-1] == 1:
            a2d = a_np.squeeze(-1)
        else:
            a2d = a_np
        try:
            if use_cv2_area:
                import cv2
                a_resized = cv2.resize(
                    a2d.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR
                ).astype(np.float32)
            else:
                pil_a = self._to_pil_gray(a2d, Image)
                pil_a = pil_a.resize((out_w, out_h), resample=Image.BILINEAR)
                a_resized = np.asarray(pil_a).astype(np.float32) / 255.0
        except Exception:
            pil_a = self._to_pil_gray(a2d, Image)
            pil_a = pil_a.resize((out_w, out_h), resample=Image.BILINEAR)
            a_resized = np.asarray(pil_a).astype(np.float32) / 255.0
        return a_resized[..., None]

class ImageDifferenceToAlpha:
    """
    Porównuje A z B lub A z kolorem HEX.
    • RGB: kolorowa różnica abs(A-B) (lub inne tryby)
    • ALPHA: siła różnicy (mean/max)
    DODATKOWO:
      Gdy source='hex_color' i podłączysz image_b, różnica (A vs HEX) jest
      kompozycjonowana NA image_b i to jest wynikiem (RGBA).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "source": (["image", "hex_color"], {"default": "image"}),

                # alfa = jak liczymy siłę różnicy
                "alpha_mode": (["mean", "max"], {"default": "max"}),
                "normalize_alpha": ("BOOLEAN", {"default": False}),

                # RGB wyjściowe nakładki (gdy nie kompozycjonujemy na tło)
                "rgb_mode": (["diff", "image_b", "black"], {"default": "diff"}),
                "boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
            },
            "optional": {
                # Użycie zależne od 'source':
                # - source='image'  -> image_b = obraz porównawczy
                # - source='hex_color' + image_b podpięte -> tło do kompozycji
                "image_b": ("IMAGE",),
                "color_hex": ("STRING", {"default": "#808080"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "diff_to_alpha"
    CATEGORY = "Image/Compare"
    DESCRIPTION = ("Kolorowa różnica + alfa; opcjonalny kompozyt na image_b, jeśli source=hex_color.")

    # ---------- utils ----------
    def _ensure_tensor(self, x, name):
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"{name} musi być tensorem torch (BxHxWxC).")
        if x.ndim != 4:
            raise ValueError(f"{name}: oczekiwany kształt BxHxWxC.")
        if x.shape[-1] not in (3, 4):
            raise ValueError(f"{name}: obsługiwane kanały: 3 (RGB) lub 4 (RGBA).")
        return x

    def _resize_like(self, src, target_h, target_w):
        # src: (H,W,C) -> (target_h,target_w,C) w [0,1]
        t = torch.from_numpy(src.transpose(2, 0, 1)).unsqueeze(0)  # 1xCxHxW
        t = torch.nn.functional.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return t.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()

    def _parse_hex(self, s):
        if s is None:
            return (0.5, 0.5, 0.5)
        txt = str(s).strip()
        if txt.startswith("#"):
            txt = txt[1:]
        txt = txt.lower()

        if re.fullmatch(r"[0-9a-f]{3}", txt):
            r = int(txt[0]*2, 16); g = int(txt[1]*2, 16); b = int(txt[2]*2, 16)
        elif re.fullmatch(r"[0-9a-f]{6}", txt):
            r = int(txt[0:2], 16); g = int(txt[2:4], 16); b = int(txt[4:6], 16)
        else:
            parts = [p.strip() for p in txt.replace(";", ",").split(",") if p.strip()]
            if len(parts) >= 3:
                try:
                    r = int(float(parts[0])); g = int(float(parts[1])); b = int(float(parts[2]))
                except Exception:
                    return (0.5, 0.5, 0.5)
            else:
                return (0.5, 0.5, 0.5)

        r = np.clip(r, 0, 255) / 255.0
        g = np.clip(g, 0, 255) / 255.0
        b = np.clip(b, 0, 255) / 255.0
        return (float(r), float(g), float(b))

    # --------- core ----------
    def diff_to_alpha(self, image_a, source, alpha_mode="max", normalize_alpha=False,
                      rgb_mode="diff", boost=1.0, image_b=None, color_hex="#808080"):

        a = self._ensure_tensor(image_a, "image_a").detach().clamp(0.0, 1.0)
        device = a.device
        A_B, A_H, A_W, _ = a.shape
        a_cpu = a.to(torch.float32).cpu().numpy()

        use_image = (str(source).lower() == "image")

        # Przygotowanie B (porównanie albo tło)
        b_cpu = None
        if use_image:
            if image_b is None:
                raise ValueError("Wybrano source=image, ale nie podłączono image_b.")
            b = self._ensure_tensor(image_b, "image_b").detach().clamp(0.0, 1.0)
            B_B, _, _, _ = b.shape
            b_cpu = b.to(torch.float32).cpu().numpy()
        else:
            base_rgb = np.array(self._parse_hex(color_hex), dtype=np.float32)
            # jeżeli image_b podpięte w tym trybie -> posłuży jako TŁO do kompozycji
            bg_cpu = None
            if image_b is not None:
                bg = self._ensure_tensor(image_b, "image_b (background)").detach().clamp(0.0, 1.0)
                BG_B, _, _, _ = bg.shape
                bg_cpu = bg.to(torch.float32).cpu().numpy()

        out_frames = []

        for i in range(A_B):
            ai = a_cpu[i]
            a_rgb = ai[..., :3]

            if use_image:
                # klasyczny tryb porównania A vs image_b
                j = min(i, B_B - 1)
                bi = b_cpu[j]
                b_rgb = bi[..., :3]
                if bi.shape[0] != A_H or bi.shape[1] != A_W:
                    b_rgb = self._resize_like(b_rgb, A_H, A_W)
            else:
                # porównujemy A do stałego koloru
                b_rgb = np.broadcast_to(base_rgb, (A_H, A_W, 3)).copy()

            # --- różnica kolorowa ---
            diff = np.abs(a_rgb - b_rgb)  # HxWx3, 0..1
            try:
                k = max(0.0, float(boost))
            except Exception:
                k = 1.0
            diff = np.clip(diff * k, 0.0, 1.0)

            # --- alfa ---
            if str(alpha_mode).lower() == "mean":
                alpha = diff.mean(axis=2)
            else:
                alpha = diff.max(axis=2)

            if normalize_alpha:
                m = float(alpha.max())
                if m > 1e-8:
                    alpha = alpha / m
            alpha = np.clip(alpha, 0.0, 1.0)

            # --- RGB nakładki (gdy nie kompozycjonujemy) ---
            mode = str(rgb_mode).lower()
            if mode == "image_b" and use_image:
                rgb_overlay = b_rgb
            elif mode == "black":
                rgb_overlay = np.zeros_like(diff)
            else:
                rgb_overlay = diff  # "diff"

            if not use_image and image_b is not None:
                # === SPECJALNY TRYB: source=hex_color + image_b podpięte → KOMPOZYT ===
                jbg = min(i, BG_B - 1)
                bg_i = bg_cpu[jbg]
                bg_rgb = bg_i[..., :3]
                if bg_i.shape[0] != A_H or bg_i.shape[1] != A_W:
                    bg_rgb = self._resize_like(bg_rgb, A_H, A_W)

                # standard 'over': out = overlay*alpha + bg*(1-alpha)
                out_rgb = rgb_overlay * alpha[..., None] + bg_rgb * (1.0 - alpha[..., None])

                # alfa wyjściowa: jeśli tło ma alfę — zostaw; w innym razie pełna 1
                if bg_i.shape[-1] == 4:
                    bg_a = bg_i[..., 3:4]
                    if bg_i.shape[0] != A_H or bg_i.shape[1] != A_W:
                        bg_a = self._resize_like(bg_a, A_H, A_W)
                    # złożenie alf (opcjonalnie można użyć bardziej złożonego wzoru)
                    out_a = np.clip(bg_a, 0.0, 1.0)
                else:
                    out_a = np.ones((A_H, A_W, 1), dtype=np.float32)

                out_rgba = np.concatenate([out_rgb, out_a], axis=-1).astype(np.float32)
            else:
                # klasyczny output: sama nakładka RGBA (bez kompozycji)
                out_rgba = np.concatenate([rgb_overlay, alpha[..., None]], axis=-1).astype(np.float32)

            out_frames.append(torch.from_numpy(out_rgba))

        out = torch.stack(out_frames, dim=0).to(torch.float32).to(device).clamp(0.0, 1.0)
        return (out,)


class SequenceContentZoom:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_sequence": ("IMAGE",),
                "interpolation": (
                    ["lanczos", "bilinear", "bicubic", "nearest", "area"],
                    {"default": "lanczos"}
                ),
                # kolor tła jako CSV/JSON: "127,127,127" lub "[127,127,127]"
                "pad_rgb": ("STRING", {"default": "127,127,127", "multiline": False}),
                # wyrównanie treści dla trybu CROP (zoom-in)
                "crop_align": (
                    ["center","top-left","top","top-right","left","right","bottom-left","bottom","bottom-right"],
                    {"default": "center"}
                ),
                # wyrównanie treści dla trybu PAD (zoom-out)
                "pad_align": (
                    ["center","top-left","top","top-right","left","right","bottom-left","bottom","bottom-right"],
                    {"default": "center"}
                ),
                # nadpisanie ostatniej wartości z listy skal
                "override_last_scale": ("BOOLEAN", {"default": False}),
                "override_value": ("FLOAT", {"default": 1.0}),
            },
            "optional": {
                # pojedynczy float LUB lista (torch/numpy/list/tuple) – kompatybilne z innymi nodami
                "scales_any": ("FLOAT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "zoom_sequence"
    CATEGORY = "Image/Transform"
    DESCRIPTION = """
Zoom do/z środka przy zachowaniu rozmiaru klatki.
- Skale: `scales_any` (float lub lista). Jeśli brak → 1.0.
- s >= 1.0 → CROP + resize (powiększenie treści), wyrównanie: `crop_align`.
- s < 1.0 → PAD (zmniejszenie treści na tle koloru `pad_rgb`), wyrównanie: `pad_align`.
- Dłuższa sekwencja niż lista → powtarza się ostatnia wartość.
- `override_last_scale` pozwala nadpisać ostatnią skalę na `override_value`.
- Interpolacje: lanczos, bilinear, bicubic, nearest; „area” używa OpenCV INTER_AREA (jeśli dostępne), inaczej bilinear.
"""

    def zoom_sequence(self, image_sequence, interpolation, pad_rgb, crop_align, pad_align,
                      override_last_scale, override_value, scales_any=None):
        try:
            from PIL import Image
        except ImportError:
            raise Exception("Brak biblioteki Pillow. Zainstaluj: pip install pillow")

        img = image_sequence
        if not isinstance(img, torch.Tensor):
            raise ValueError("image_sequence musi być tensorem torch (BxHxWxC)")

        device = img.device
        img_cpu = img.detach().clamp(0.0, 1.0).cpu()
        b, h, w, c = img_cpu.shape
        if c not in (3, 4):
            raise ValueError(f"Obsługiwane są obrazy z 3 (RGB) lub 4 (RGBA) kanałami, otrzymano: {c}")

        # skale z innego noda
        scale_list = self._coerce_any(scales_any)
        if not scale_list:
            scale_list = [1.0]

        # nadpisanie ostatniej skali (opcjonalnie)
        if override_last_scale:
            try:
                ov = float(override_value)
            except Exception:
                ov = 1.0
            if len(scale_list) == 0:
                scale_list = [ov]
            else:
                scale_list[-1] = ov

        # kolor tła
        pr, pg, pb = self._parse_rgb(pad_rgb)  # 0..1

        pil_methods = {
            "lanczos": Image.LANCZOS,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "nearest": Image.NEAREST,
        }

        use_cv2_area = False
        if interpolation == "area":
            try:
                import cv2  # opcjonalnie
                use_cv2_area = True
            except Exception:
                use_cv2_area = False  # fallback do bilinear

        out_frames = []
        np_frames = img_cpu.numpy()  # B,H,W,C w [0..1]

        for i in range(b):
            s = scale_list[min(i, len(scale_list) - 1)]
            try:
                s = float(s)
            except Exception:
                s = 1.0
            s = max(s, 1e-6)  # zabezpieczenie

            frame_np = np_frames[i]  # H,W,C

            if s >= 1.0:
                # ========= ZOOM-IN (CROP + RESIZE) =========
                crop_w = max(1, int(round(w / s)))
                crop_h = max(1, int(round(h / s)))
                left, top = self._aligned_offset(container_w=w, container_h=h,
                                                 content_w=crop_w, content_h=crop_h,
                                                 align=crop_align)
                left = max(0, min(left, w - crop_w))
                top = max(0, min(top, h - crop_h))
                right = left + crop_w
                bottom = top + crop_h

                if c == 3:
                    crop_np = frame_np[top:bottom, left:right, :]
                    resized = self._resize_rgb(crop_np, w, h, interpolation, use_cv2_area, Image, pil_methods)
                    out_frames.append(torch.from_numpy(resized))
                else:
                    rgb_np = frame_np[:, :, :3]
                    a_np = frame_np[:, :, 3:]
                    rgb_crop = rgb_np[top:bottom, left:right, :]
                    a_crop = a_np[top:bottom, left:right, :]

                    rgb_resized = self._resize_rgb(rgb_crop, w, h, interpolation, use_cv2_area, Image, pil_methods)
                    a_resized = self._resize_alpha(a_crop, w, h, use_cv2_area, Image)
                    merged = np.concatenate([rgb_resized, a_resized], axis=-1)
                    out_frames.append(torch.from_numpy(merged))

            else:
                # ========= ZOOM-OUT (PAD) =========
                content_w = max(1, int(round(w * s)))
                content_h = max(1, int(round(h * s)))
                left, top = self._aligned_offset(container_w=w, container_h=h,
                                                 content_w=content_w, content_h=content_h,
                                                 align=pad_align)
                left = max(0, min(left, w - content_w))
                top = max(0, min(top, h - content_h))
                right = left + content_w
                bottom = top + content_h

                if c == 3:
                    bg = np.empty((h, w, 3), dtype=np.float32)
                    bg[..., 0] = pr
                    bg[..., 1] = pg
                    bg[..., 2] = pb

                    resized = self._resize_rgb(frame_np, content_w, content_h, interpolation, use_cv2_area, Image, pil_methods)
                    bg[top:bottom, left:right, :] = resized
                    out_frames.append(torch.from_numpy(bg))
                else:
                    rgb_np = frame_np[:, :, :3]
                    a_np = frame_np[:, :, 3:]

                    bg_rgb = np.empty((h, w, 3), dtype=np.float32)
                    bg_rgb[..., 0] = pr
                    bg_rgb[..., 1] = pg
                    bg_rgb[..., 2] = pb
                    bg_a = np.ones((h, w, 1), dtype=np.float32)  # tło nieprzezroczyste

                    rgb_resized = self._resize_rgb(rgb_np, content_w, content_h, interpolation, use_cv2_area, Image, pil_methods)
                    a_resized = self._resize_alpha(a_np, content_w, content_h, use_cv2_area, Image)

                    bg_rgb[top:bottom, left:right, :] = rgb_resized
                    bg_a[top:bottom, left:right, :] = a_resized

                    merged = np.concatenate([bg_rgb, bg_a], axis=-1)
                    out_frames.append(torch.from_numpy(merged))

        out = torch.stack(out_frames, dim=0).to(torch.float32).clamp(0.0, 1.0)
        return (out.to(device),)

    # ===================== helpers =====================

    def _aligned_offset(self, container_w, container_h, content_w, content_h, align):
        """
        Zwraca (left, top) dla osadzenia contentu w kontenerze wg align.
        align ∈ {"center","top-left","top","top-right","left","right","bottom-left","bottom","bottom-right"}
        """
        if align == "top-left":
            left = 0
            top = 0
        elif align == "top":
            left = (container_w - content_w) // 2
            top = 0
        elif align == "top-right":
            left = container_w - content_w
            top = 0
        elif align == "left":
            left = 0
            top = (container_h - content_h) // 2
        elif align == "right":
            left = container_w - content_w
            top = (container_h - content_h) // 2
        elif align == "bottom-left":
            left = 0
            top = container_h - content_h
        elif align == "bottom":
            left = (container_w - content_w) // 2
            top = container_h - content_h
        elif align == "bottom-right":
            left = container_w - content_w
            top = container_h - content_h
        else:  # "center" i fallback
            left = (container_w - content_w) // 2
            top = (container_h - content_h) // 2
        return left, top

    def _coerce_any(self, value):
        """
        Przyjmij:
        - listę/krotkę,
        - pojedynczy float/int,
        - torch.Tensor (dowolny shape),
        - numpy.ndarray,
        - None → [].
        """
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            try:
                return [float(x) for x in value.detach().cpu().flatten().tolist()]
            except Exception:
                return []
        if isinstance(value, np.ndarray):
            try:
                return [float(x) for x in value.flatten().tolist()]
            except Exception:
                return []
        if isinstance(value, (int, float)):
            return [float(value)]
        if isinstance(value, (list, tuple)):
            out = []
            for x in value:
                try:
                    out.append(float(x))
                except Exception:
                    continue
            return out
        return []

    def _parse_rgb(self, s):
        """
        Parsuje '127,127,127' lub '[127,127,127]' → (r,g,b) w skali 0..1.
        Jeśli błąd, zwraca (0.498, 0.498, 0.498) ~ 127/255.
        """
        default = (127/255.0, 127/255.0, 127/255.0)
        if s is None:
            return default
        txt = str(s).strip()
        if not txt:
            return default
        # JSON?
        try:
            val = json.loads(txt)
            if isinstance(val, (list, tuple)) and len(val) >= 3:
                r, g, b = val[:3]
                return (np.clip(float(r), 0, 255)/255.0,
                        np.clip(float(g), 0, 255)/255.0,
                        np.clip(float(b), 0, 255)/255.0)
        except Exception:
            pass
        # CSV
        txt = txt.replace(";", ",")
        if txt.startswith("[") and txt.endswith("]"):
            txt = txt[1:-1]
        parts = [p.strip() for p in txt.split(",") if p.strip()]
        if len(parts) >= 3:
            try:
                r = np.clip(float(parts[0]), 0, 255)/255.0
                g = np.clip(float(parts[1]), 0, 255)/255.0
                b = np.clip(float(parts[2]), 0, 255)/255.0
                return (r, g, b)
            except Exception:
                return default
        return default

    def _to_pil_rgb(self, arr, Image):
        """arr: HxWx3 w [0..1] lub [0..255]"""
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def _to_pil_gray(self, arr, Image):
        """arr: HxW w [0..1] lub [0..255]"""
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")

    def _resize_rgb(self, src_np, out_w, out_h, interpolation, use_cv2_area, Image, pil_methods):
        if interpolation == "area" and use_cv2_area:
            import cv2
            return cv2.resize(
                src_np.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_AREA
            ).astype(np.float32)
        else:
            pil_img = self._to_pil_rgb(src_np, Image)
            resample = pil_methods.get(interpolation, Image.BILINEAR)
            pil_img = pil_img.resize((out_w, out_h), resample=resample)
            return np.asarray(pil_img).astype(np.float32) / 255.0

    def _resize_alpha(self, a_np, out_w, out_h, use_cv2_area, Image):
        # alfę wygładzamy bilinearnie (cv2.INTER_LINEAR / PIL.BILINEAR)
        if a_np.ndim == 3 and a_np.shape[-1] == 1:
            a2d = a_np.squeeze(-1)
        else:
            a2d = a_np
        try:
            if use_cv2_area:
                import cv2
                a_resized = cv2.resize(
                    a2d.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR
                ).astype(np.float32)
            else:
                pil_a = self._to_pil_gray(a2d, Image)
                pil_a = pil_a.resize((out_w, out_h), resample=Image.BILINEAR)
                a_resized = np.asarray(pil_a).astype(np.float32) / 255.0
        except Exception:
            pil_a = self._to_pil_gray(a2d, Image)
            pil_a = pil_a.resize((out_w, out_h), resample=Image.BILINEAR)
            a_resized = np.asarray(pil_a).astype(np.float32) / 255.0
        return a_resized[..., None]



class PathTool:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": ("STRING", {"default": "ComfyUI", "multiline": False}),
                "base_path": ("STRING", {"default": "Project_01/", "multiline": False}),
                "alt_path": ("STRING", {"default": "Project_01/Alternatives/", "multiline": False}),
                "use_alt_path": ("BOOLEAN", {"default": False}),
                "date_format": ("STRING", {"default": "%Y_%m_%d", "multiline": False}),
                "use_date_folder": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename_prefix",)
    FUNCTION = "build_path"
    CATEGORY = "Filename & Path Manager"
    DESCRIPTION = """
### Path Tool

This node dynamically generates a filename prefix for saving files into an organized folder structure.

What it does:
* It constructs a path in the format: `[root_directory]/[current_date]/[filename]`.
* You can choose between a primary `base_path` and an `alt_path` using a boolean toggle.
* It automatically creates a subfolder named with the current date in YYYY_MM_DD format (e.g., 2023_10_27).
* It sanitizes the path and filename to remove any invalid characters, ensuring a safe output.
* The final string is intended to be connected to the `filename_prefix input` of a "Save Image" or similar node. 
"""

    def build_path(self, filename, base_path, alt_path, use_alt_path, date_format, use_date_folder):
        root_dir = self._clean_path(alt_path if use_alt_path else base_path)
        path_components = [root_dir]
        
        if use_date_folder:
            current_date_str = datetime.datetime.now().strftime(date_format)
            path_components.append(current_date_str)
            
        sanitized_filename = self._sanitize(filename)
        path_components.append(sanitized_filename)
        
        full_path = os.path.join(*path_components)
        
        return (os.path.normpath(full_path),)

    def _clean_path(self, path):
        return os.path.normpath("".join(c for c in path.strip() if c not in '*?"<>|'))

    def _sanitize(self, name):
        return "".join(c for c in name.strip() if c not in '\\/:*?"<>|')


class ColorMatchFalloff:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
                "method": (
                    ['mkl', 'hm', 'reinhard', 'mvgd', 'hm-mvgd-hm', 'hm-mkl-hm'], 
                    {"default": 'mkl'}
                ),
                "falloff_duration": ("INT", {
                    "default": 50, 
                    "min": 2, 
                    "max": 1000, 
                    "display": "number"
                }),
            },
        }
    
    CATEGORY = "KJNodes/image"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "colormatch_dynamic"
    DESCRIPTION = """
Code is modified version of Kijai's ComfyUI-KJNodes (https://github.com/kijai/ComfyUI-KJNodes).
Color Match Falloff node smoothly fades the color correction effect over a sequence of frames (falloff_duration). It uses an Ease-In-Out curve (from strength 1 to 0) to mitigate color flickering, making it ideal for seamlessly stitching together video clips generated in separate batches (last frame to first frame approach) using WAN2.1 model
"""
    
    def colormatch_dynamic(self, image_ref, image_target, method, falloff_duration=50):
        try:
            from color_matcher import ColorMatcher
        except ImportError:
            raise Exception("Can't import color-matcher. Pip install color-matcher")
        
        cm = ColorMatcher()
        image_ref_tensor = image_ref.cpu()
        image_target_tensor = image_target.cpu()
        batch_size = image_target_tensor.size(0)
        
        out = []

        images_target_np = image_target_tensor.numpy()
        images_ref_np = image_ref_tensor.numpy()

        if image_ref_tensor.size(0) > 1 and image_ref_tensor.size(0) != batch_size:
            raise ValueError("ColorMatch: Use a single ref image or a matching batch of ref images.")

        for i in range(batch_size):
            frame_number = i + 1 
            strength = 0.0
            
            if frame_number <= 1:
                strength = 1.0
            elif frame_number <= falloff_duration:
                normalized_frame = (frame_number - 1) / (falloff_duration - 1)
                angle = normalized_frame * (math.pi / 2)
                strength = math.cos(angle)

            if strength < 0.0001:
                blended_image_np = images_target_np[i]
            else:
                image_target_np_single = images_target_np[i]
                image_ref_np_single = images_ref_np[0] if image_ref_tensor.size(0) == 1 else images_ref_np[i]
                
                try:
                    image_result_np = cm.transfer(src=image_target_np_single, ref=image_ref_np_single, method=method)
                except Exception as e:
                    print(f"Error during color transfer on frame {frame_number}: {e}")
                    image_result_np = image_target_np_single
                
                blended_image_np = image_target_np_single + strength * (image_result_np - image_target_np_single)
            
            out.append(torch.from_numpy(blended_image_np))
            
        out_tensor = torch.stack(out, dim=0).to(torch.float32)
        out_tensor.clamp_(0, 1)
     
        return (out_tensor,)
        
