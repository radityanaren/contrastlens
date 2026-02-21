# ContrastLens

> Stochastic luminance-based image contrast renderer with a Tkinter GUI.

ContrastLens is a fun Python projects that converts any photograph into a high contrast black and white rendering using a probabilistic sampling technique driven by pixel luminance. Basically it's very good for detecting contrast on an image.


| Original | Low Contrast | High Contrast |
|----------|-------------|---------------|
| ![Original](assets/input1.jpg) | ![Low Contrast](assets/output1.png) | ![High Contrast](assets/output2.png) |

> Hold **left mouse button** on the canvas to compare the processed result against the original image.

## Features

- Load any JPEG, PNG, or BMP image
- Real-time contrast slider (0.0 → 1.0)
- Two rendering modes: **Low Contrast** and **High Contrast**
- Hold-to-compare: press and hold the canvas to see the original
- Save the processed result as a PNG

---

## How It Works

1. **Luminance extraction** : The input image is converted to a single-channel float luminance map in `[0, 1]`.
2. **Adaptive contrast expansion** : Pixel values are stretched away from the midpoint (0.5) by a configurable gain factor.
3. **Probability mapping** : Each pixel's luminance is inverted to a sampling probability
4. **Stochastic sampling** : A random draw is made per pixel against its probability, producing a binary mask.
5. **Output** : The mask is mapped to a pure black-and-white image (0 or 255).

The two modes control the strength of the contrast gain:

| Mode | Contrast Gain Formula | Gamma |
|------|-----------------------|-------|
| Low | `1.0 + contrast × 1.5` | 1.0 |
| High | `1.0 + contrast × 4.0` | 0.6 |


## Installation

> [!WARNING]
> Make sure you have Git and Python 3.10 or better installed on your computer!

1. Clone the repo :
   ```bash
   git clone https://github.com/your-username/ContrastLens.git
   ```
2. Go to directory :
   ```bash
   cd ContrastLens
   ```
3. Install dependencies :
   ```bash
   pip install .
   ```
4. Run the app :
   ```
   python -m contrastlens
   ```

---

## How to use it

**Step by step guide**
1. Load an image with the **Load Image** button (supports JPEG, PNG, BMP)
2. Pick a rendering mode : **Low Contrast** or **High Contrast**
3. Adjust the **Contrast** slider to dial in the effect intensity
4. Hold the canvas with the **left mouse button** to compare against the original
5. Save the result with the **Save Image** button

> [!IMPORTANT]
> Each slider or mode change re-renders the image live.

---
