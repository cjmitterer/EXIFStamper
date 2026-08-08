# EXIFStamper

# DISCLAIMER:

This was created using extensive use of generative AI and honestly kind of sucks. But..... it does what I need to do with it. I may clean it up but likely will be abandonded. If I ever make this public and someone for somereason wants to use it hats off to you.

# Purpose

The purpose of this is to stamp missing timestamps into photo and video files so the original date survives even when copied between devices. I'm currently working on a personal project that requires me to add windows creation data to the meta data of images and videos. Some of those images/videos did not have such data so I am stamping it in pulling from the Windows information.

Windows CLI. Single script: `stamper.py`.

## What it does

1. Scans a directory for media files
2. Checks if each file already has a timestamp — if it does, leaves it **completely untouched**
3. For unprotected files, stamps them with `min(creation time, modified time)` 
4. Logs results to `stamper.log` (no console output)

Stamped metadata includes: date, time, timezone, device info (Make/Model), and software name.

## Supported formats

| Format | Support | Notes |
|---|---|---|
| JPEG, JPG, PNG, TIFF, WebP, HEIC | Full support | EXIF metadata |
| MP4, M4V | Full support | Container atoms (mutagen) |
| MOV | Full support | Container atoms (exiftool) |
| CR2, NEF, ARW, ORF, RW2, DNG | Detect-only | Inspected but never written |
| Other | Unsupported | Logged and skipped |

## Setup

**Requirements:**
- Python 3.12+
- `exiftool` executable (required for MOV files)

**Install Python packages:**
```bash
pip install --user piexif Pillow pillow-heif mutagen
```

**Get exiftool:**
- Download from https://exiftool.org/
- Or if you already have it, the stamper will auto-detect it

## Usage

```bash
python stamper.py <directory> [--recursive] [--dry-run] [--backup]
```

| Option | Effect |
|---|---|
| `--recursive` | Scan subdirectories (default: top-level only) |
| `--dry-run` | Log what would happen, don't write anything |
| `--backup` | Create `.original` backup before writing |

## Quick start

1. **Preview first** (no changes):
   ```bash
   python stamper.py "D:\My Photos" --recursive --dry-run
   ```
   Check `stamper.log` to see what will happen.

2. **Stamp with backup** (safe):
   ```bash
   python stamper.py "D:\My Photos" --recursive --backup
   ```

3. **Verify** (open a stamped file in Explorer or `exiftool` to confirm the date looks right)

## Key behaviors

- **Never overwrites existing dates** — only fills in missing timestamps
- **Never modifies camera hardware fields** — aperture, ISO, shutter speed, focal length, lens
- **RAW/DNG files are inspect-only** — never written to
- **Each run overwrites `stamper.log`** — keep a copy if you need a history
