# Film Simulation

一套基于 Python (Pillow + NumPy) 的图像风格化工具，支持胶片模拟、卡通化、漫画印刷、像素游戏化等效果。作为 Claude Code 的 **film-simulation** skill 运行。

---

## 功能总览

### 🎞️ 胶片模拟（13种）
| 风格 | 效果 |
|------|------|
| Classic Chrome | 富士经典正片，暖调柔和 |
| Velvia | 富士 Velvia，鲜艳高饱和 |
| Provia | 富士 Provia，自然平衡 |
| Astia | 富士 Astia，人像柔和 |
| Portra | 柯达 Portra，暖调肤色 |
| Gold | 柯达 Gold 200，经典暖色 |
| Cinestill 800T | 电影感，青调 + 辉光 |
| Leica | 徕卡风格，微反差立体感 |
| CCD | 早期数码 CCD 冷调 |
| Tri-X 400 | 柯达黑白经典，颗粒感 |
| B&W High Contrast | 高对比黑白 |
| Portra B&W | Portra 黑白转换 |
| Faded | 褪色复古 |

### 🎨 特效（4种）
| 风格 | 效果 |
|------|------|
| Cartoon | 卡通化 — cel-shading + 边缘线 |
| Cartoon Inked | 卡通墨线 — 粗黑轮廓 + 平涂色 |
| Line Art | 线条画 — 白底黑线素描 |
| Comic Print | 漫画印刷 — 半调网点 + 墨线 |

### 🕹️ 像素游戏（5种）
| 风格 | 效果 |
|------|------|
| Pixel | 通用 8-bit 像素化 |
| Pixel Dithered | 像素化 + Floyd-Steinberg 抖动 |
| Pixel NES | 红白机 64 色调色板 |
| Pixel Game Boy | Game Boy 4 色绿屏 |
| Pixel PICO-8 | PICO-8 幻想主机 16 色 |

### 🛠️ 后处理
- **去背景** (`--remove-bg`) — 移除纯色背景，输出 RGBA 透明 PNG
- **缩边去锯齿** (`--shrink`) — 向内收缩 + 羽化，清理边缘白边锯齿

---

## 使用方法

```bash
# 基本用法
python scripts/film_sim.py <图片路径> --profile <风格名>

# 指定输出路径
python scripts/film_sim.py input.jpg output.png --profile portra

# 控制强度
python scripts/film_sim.py input.jpg --profile velvia --strength 0.7
```

### 像素类附加参数

```bash
# 调像素块大小（默认8，越小越精细）
python scripts/film_sim.py input.jpg --profile pixel --block-size 4

# 调颜色层级（2-8）
python scripts/film_sim.py input.jpg --profile pixel --color-levels 6

# 抖动
python scripts/film_sim.py input.jpg --profile pixel --dither
```

### 后处理参数（通用）

```bash
# 去白底
python scripts/film_sim.py input.jpg --profile cartoon --remove-bg

# 去指定颜色背景（如绿色）
python scripts/film_sim.py input.jpg --profile pixel --remove-bg 0,255,0

# 去底 + 缩边去锯齿
python scripts/film_sim.py input.jpg --profile pixel --remove-bg --shrink 2

# 调色容差（默认30）
python scripts/film_sim.py input.jpg --remove-bg --bg-tolerance 50
```

### 完整示例

```bash
# 像素化 + NES 调色板 + 去白底 + 缩边
python scripts/film_sim.py photo.png --profile pixel-nes --block-size 4 --remove-bg --shrink 1

# 卡通 + 去白底
python scripts/film_sim.py photo.png --profile cartoon --remove-bg

# 胶片 + 低强度
python scripts/film_sim.py photo.png --profile cinestill --strength 0.6
```

---

## 命令参考

```
usage: film_sim.py [-h] [--profile PROFILE] [--strength STRENGTH]
                   [--block-size BLOCK_SIZE] [--color-levels COLOR_LEVELS]
                   [--dither] [--no-dither] [--palette PALETTE]
                   [--remove-bg [REMOVE_BG]] [--bg-tolerance BG_TOLERANCE]
                   [--shrink SHRINK] [--list] [--quality QUALITY]
                   [input] [output]

Apply film simulation to an image

positional arguments:
  input                 Input image path
  output                Output image path

options:
  -h, --help            show this help message
  --profile, -p         Profile name (default: classic-chrome)
  --strength, -s        Effect strength 0-1 (default: 1.0)
  --block-size          Pixel block size for pixel profiles
  --color-levels        Color quantization levels for pixel profiles
  --dither / --no-dither Enable/disable Floyd-Steinberg dithering
  --palette             Retro palette: nes, gameboy, cga, pico8
  --remove-bg [R,G,B]   Remove background color, output RGBA PNG
  --bg-tolerance        Tolerance for background removal (default: 30)
  --shrink N            Shrink inward by N pixels to clean aliasing
  --list                List all available profiles
  --quality             JPEG save quality (default: 95)
```

---

## 文件结构

```
film-simulation/
├── README.md           ← 本文件
├── SKILL.md            ← Claude Code skill 定义
├── LICENSE
├── scripts/
│   └── film_sim.py     ← 核心处理脚本
└── references/         ← 参考素材
```

---

## 环境依赖

- Python 3.7+
- Pillow
- NumPy

```bash
pip install pillow numpy
```

---

## 许可证

MIT
