---
name: film-simulation
description: Apply classic film stock simulation effects (Fujifilm Classic Chrome, Velvia, Provia, Astia; Kodak Portra, Tri-X, Gold; Cinestill 800T; CCD/Leica camera sims; cartoon/line art; comic print) to user-provided images. Use this skill ANY TIME the user wants to add film looks, film simulations, retro photo effects, cinematic grades, analog-style color grading, cartoon/line art conversion, or comic print effects — even if they phrase it casually like "make this look like film", "give it a retro vibe", "add grain", "Fujifilm colors", "make it look cinematic", "vintage photo", "turn this into a cartoon/drawing/sketch/line art", or "comic book effect". Also trigger when discussing photo editing, color grading, film emulation, or image stylization.
---
# Film Simulation

Apply classic film stock looks to digital photos using Python (Pillow + NumPy).

## Capabilities

This skill can:
- Apply **15 built-in profiles**: film stocks, camera sensors, cartoon/line art, comic print
- Adjust effect **strength** (0.0–1.0) for subtle to full looks
- Add **film grain** with luminance-dependent noise
- Add **halation glow** (red-edge bloom around bright areas — Cinestill signature)
- Apply **tone curves** (characteristic S-curves per film stock)
- **3-way color grading** (shadows/midtones/highlights)
- **Per-hue selective** saturation & hue shift
- **Color matrix** transforms in linear space
- **XDoG/Sobel-based line art** with anti-aliased output
- **Cel-shading + edge inking** for cartoon effects
- **Ben-Day halftone dots** for comic/manga print
- **Lens vignetting** simulation
- Process any image format Pillow supports (JPEG, PNG, TIFF, WebP, etc.)
- **Retro game pixelation**: lower resolution + color quantization with adjustable block size
- **Retro console palettes**: NES (64-color), Game Boy (4-shade green), PICO-8 (16-color), CGA
- **Floyd-Steinberg dithering**: smooth color transitions in pixel art
- **Background removal**: remove white/solid backgrounds, output RGBA PNG with transparency
- **Edge trim/shrink**: shrink content inward to clean up white aliasing/halos around edges

## Available Profiles

| Profile | Film Stock | Character |
|---------|-----------|-----------|
| `classic-chrome` | Fujifilm Classic Chrome | Muted, warm shadows |
| `velvia` | Fujifilm Velvia | Vivid, high saturation |
| `provia` | Fujifilm Provia | Natural, balanced |
| `astia` | Fujifilm Astia | Soft, portrait-friendly |
| `portra` | Kodak Portra | Warm skin tones, soft |
| `tri-x` | Kodak Tri-X 400 | B&W, grainy, contrasty |
| `gold` | Kodak Gold 200 | Warm consumer film |
| `cinestill` | Cinestill 800T | Cinematic, halation glow |
| `ccd` | CCD Sensor | Cool tones, punchy blues, early digital |
| `leica` | Leica Look | Micro contrast, warm, smooth, 3D pop |
| `cartoon` | Cartoon | Cel-shaded with edge lines |
| `cartoon-inked` | Cartoon Inked | Bold ink outlines, flat colors |
| `lineart` | Line Art | Anti-aliased sketch, white bg, black strokes |
| `print` | Comic Print | Halftone dots, bold inks, manga style |
| `portra-bw` | Kodak Portra B&W | Smooth B&W conversion |
| `bw-high-contrast` | — | Dramatic B&W |
| `faded` | — | Faded vintage look |
| `pixel` | Retro Pixel Art | 8-bit/16-bit game style — chunky pixels, posterized colors |
| `pixel-dithered` | Pixel Art + Dither | Pixel art with Floyd-Steinberg dithering |
| `pixel-nes` | NES Palette | NES authentic 64-color palette + pixelation |
| `pixel-gameboy` | Game Boy | 4-shade green monochrome Game Boy look |
| `pixel-pico8` | PICO-8 | 16-color fantasy console palette |

## Usage Workflow

1. **Get the image path** from the user — accept drag/drop paths, URLs (download first), or clipboard references
2. **Ask which look** they want if not specified:
   - "Any particular film stock in mind? I have Fujifilm Classic Chrome / Velvia / Provia / Astia, Kodak Portra / Tri-X / Gold, Cinestill, or a faded vintage look."
   - If they're unsure, suggest based on the photo content (landscape → Velvia, portrait → Astia/Portra, street → Classic Chrome/Tri-X)
3. **Run the simulation**:
   ```bash
   python <skill_path>/scripts/film_sim.py <input_path> --profile <profile_name> [--strength 0.8]
   ```
4. **Present the result** — show before/after if possible, or open the output file for the user
5. **Iterate** — if the user wants adjustments, re-run with different profile/strength

## Example interactions

**User:** "Make this photo look like a 90s film camera"
**You:** "I'll apply a Kodak Gold look — warm tones and subtle grain."
→ `python film_sim.py photo.jpg --profile gold --strength 0.9`

**User:** "Give it that Wong Kar-wai vibe"
**You:** "Cinestill 800T with halation glow — the classic Hong Kong night look."
→ `python film_sim.py photo.jpg --profile cinestill --strength 0.9`

**User:** "Something moody and dramatic"
**You:** "Tri-X or B&W high contrast. Let me try the high-contrast B&W."
→ `python film_sim.py photo.jpg --profile bw-high-contrast`

**User:** "Turn this into a line drawing / sketch"
**You:** "I'll create a clean line art version."
→ `python film_sim.py photo.jpg --profile lineart`

**User:** "Make it look like a comic book / manga"
**You:** "I'll apply comic print with halftone dots."
→ `python film_sim.py photo.jpg --profile print --strength 0.6`

**User:** "Cartoonize this photo"
**You:** "I'll apply cel-shading with edge lines. Want it inked (bold outlines) or soft?"
→ `python film_sim.py photo.jpg --profile cartoon`

**User:** "Turn this into pixel art / 8-bit game style"
**You:** "I'll pixelate it with retro game style. Want a specific console palette (NES, Game Boy, PICO-8)?"
→ `python film_sim.py photo.jpg --profile pixel --block-size 8 --color-levels 4`
→ `python film_sim.py photo.jpg --profile pixel-nes --block-size 6`
→ `python film_sim.py photo.jpg --profile pixel-gameboy`

**User:** "Make this look like a Game Boy game"
**You:** "I'll apply the Game Boy pixel look — 4-shade green palette."
→ `python film_sim.py photo.jpg --profile pixel-gameboy --block-size 6`

**User:** "Remove the white background / make background transparent"
**You:** "I'll remove the white background with transparency."
→ `python film_sim.py photo.jpg --profile pixel --remove-bg --bg-tolerance 30`

**User:** "Clean up the white jagged edges around my pixel art"
**You:** "I'll shrink the content inward slightly and feather the edges."
→ `python film_sim.py photo.jpg --profile pixel --remove-bg --shrink 1`

**User:** "Pixel art with no white background and cleaner edges"
**You:** "I'll pixelate it, remove the white background, and trim edge artifacts."
→ `python film_sim.py photo.jpg --profile pixel --block-size 4 --remove-bg --shrink 1`

## Post-Processing Options

These flags work with **any** profile (film, cartoon, pixel, etc.):

| Flag | Description |
|------|-------------|
| `--remove-bg [R,G,B]` | Remove background color → RGBA PNG. Default removes white `255,255,255`. Specify custom color like `--remove-bg 0,255,0` |
| `--bg-tolerance N` | Tolerance for bg removal 0-255 (default: 30). Higher = more aggressive |
| `--shrink N` | Shrink content inward by N pixels, feather alpha edges to clean aliasing |

## Pixel Art Profiles — Extra CLI Flags

These flags tune the pixel/retro profiles:

| Flag | Description |
|------|-------------|
| `--block-size N` | Pixel block size (2-32, default per profile ~6-8). Lower = finer pixels |
| `--color-levels N` | Color quantization levels per channel (2-8). Lower = fewer colors |
| `--dither` / `--no-dither` | Enable/disable Floyd-Steinberg dithering |
| `--palette NAME` | Retro palette: nes, gameboy, cga, cga-high, zxs, pico8 |

## Script Path

The script lives at `scripts/film_sim.py` relative to this SKILL.md. Use `<skill_path>/scripts/film_sim.py` to reference it from bash.
