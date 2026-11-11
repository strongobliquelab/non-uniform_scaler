# Non-Uniform Scalers

ComfyUI custom nodes for **non-uniform scaling** image and latent along a single axis (vertical or horizontal). Made to cope, revolt, and counter-stretch against SDXL/Illustrious's characteristic distortion with inter-pass processing.

![saya mau kamu susu](assets/ss1.png)
## Installation

Put dis repository into your ComfyUI `custom_nodes` folder and restart ComfyUI.
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/strongobliquelab/non-uniform_scaler

```
## Usage

### Non-Uniform Scaler (Image)

| Parameter            | Type     | Description |
|----------------------|----------|-------------|
| `image`              | IMAGE    | Input image batch |
| `scaling_mode`       | combo    | `factor` or `targeted_dimension` |
| `direction`          | combo    | `vertical` or `horizontal` |
| `crop_mode`          | combo    | (`targeted_dimension` mode only) How the node should trim some insignificant leftover pixels, just set this to `both` tbh |
| `scale_factor`       | FLOAT    | (`factor` mode only) Peak scale at the centre of the affected region |
| `targeted_dimension` | INT      | (`targeted_dimension` mode only) Desired output length of the whole image along the chosen axis |
| `affected_start`     | INT      | Inclusive start of the region to be affected |
| `affected_end`       | INT      | Inclusive end of the region to be affected |
| `mask_input_mode`    | BOOLEAN  | Mask input mode, to use mask's edges to input `affected_start` / `affected_end` instead |
| `mask`               | MASK     | (`mask_input_mode` only) Optional. Vertical uses the topmost and bottommost lit rows; horizontal uses the leftmost and rightmost lit columns. |

The node scales the image along the `direction` axis with a scale that's linear in the source-pixel space: it is one at the starting/ending edges and ramps to `scale_factor` at the span centre.

In `targeted_dimension` mode, the scale_factor would be computed from the `targeted_dimension` instead.

The `mask_input_mode` allows you to pick the affected region and should be useful for post-processing/inter-pass processing.

### Non-Uniform Scaler (Latent)
It has **almost** the same controls as the image node, plus:

| Parameter                  | Type | Description |
|----------------------------|------|-------------|
| `downsampling_factor_be_8` | INT  | It is 8, mostly, and is the downsampling factor |

The `downsampling_factor_be_8` parameter is 8 (eight), for most cases, and is the downsampling factor of your VAE model. It is, again, eight because it is eight for XL models, ignore this unless you're doing rocket science with a non-XL one. Consult your distilled Gemini-Instant-Flash-Lite for the correct downsampling factor if you use a non-XL model.

## Usage (cont.)
### Introduction

So... Stable Diffusion and its derivative models, including the good ol' Illustrious, are notorious for distorting details. It is painfully obvious even at 1 megapixel and would be amplified and worsened after hi-res passes. Here, I offer a straightforward and **barbaric** solution to cope against it, which is just counter-stretching.

Of course, as a barbaric solution, you have total control over where, when, and how to counter-stretch your works. Here, in this section, I offer how I personally use my nodes.

### Mask Input Inter-Pass Processing
Look at this.
![mah wife](assets/ss2.png)
You probably want to shorten her elongated torso and head top with a 0.6-0.8 `scale_factor`. Here, the mask input mode offers precise control over where to stretch. Note that the affected region is strictly defined by the mask's topmost and bottommost boundaries, so you would need to fix those two regions one by one.

That is pretty much post-processing. However, I recommend using it as an inter-pass processing tool instead, preferably right after t2i. Please respect your own artworks.

### Latent Scaler Usage
Anyway, in case you didn't notice hard enough, the SDXL/Illustrious's distortion is characteristically strongest at the centre of the image; you probably want to counter-stretch your work there with an affected area that fully covers your inputs. Here, you can put the latent scalers between your denoise nodes, with triple passes. It works by creating an extra space so that my non-uniform scaler can shrink the thing properly.
![vertical_denoise](assets/ss3.png)
In this example, it is a component of a t2i workflow. For i2i please use the **Non-Uniform Scaler (Image)** node before VAE encoding instead.

### Universal Deslop Stretcher
As a proof of concept, I also made this [universal deslop stretcher](workflows/deslop_stretching.json) that's based entirely on that said distortion pattern.
![64_nodes_from_hell](assets/ss4.png)

---

You may, and should use **[Kohya Deep Shrink](https://docs.comfy.org/built-in-nodes/PatchModelAddDownscale)** together with this pack as it helps solving the exact same problem as my pack does, in case you ain't aware of its existence. My nodes attack a different front than its approach. That's it, bye bye.
