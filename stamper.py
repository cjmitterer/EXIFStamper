"""EXIFStamper -- stamp durable, honest timestamps into unprotected media files.

Windows CLI. Pipeline: scan -> classify -> plan -> apply -> log
(see docs/architecture spine: _bmad-output/planning-artifacts/architecture/
architecture-EXIFStamper-2026-08-07/ARCHITECTURE-SPINE.md).

Usage:
    python stamper.py <directory> [--recursive] [--dry-run] [--backup]

This slice implements the full pipeline with real write support for
JPEG/TIFF/WebP/PNG/HEIC (via piexif+Pillow+pillow-heif), MP4/M4V (via mutagen),
and MOV (via exiftool). DNG and RAW formats (CR2/NEF/ARW/ORF/RW2) remain wired
into FORMAT_CAPABILITIES as detect-only (AD-8 default) pending their own promotion.
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import piexif
from PIL import Image
from PIL.ExifTags import IFD
from mutagen.mp4 import MP4, MP4StreamInfoError

try:
    import pillow_heif

    pillow_heif.register_heif_opener()  # lets Image.open() handle .heic transparently
except Exception:
    # Missing/broken HEIF codec must not take down the whole CLI -- HEIC
    # files simply fail per-file (same AD-4 channel) instead of crashing
    # every other format at startup.
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FILENAME = "stamper.log"

# ---------------------------------------------------------------------------
# Exiftool detection
# ---------------------------------------------------------------------------

def _find_exiftool_executable() -> str:
    """Find exiftool executable on the system. Raises RuntimeError if not found."""
    try:
        result = subprocess.run(
            ["exiftool", "-ver"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        if result.returncode == 0:
            return "exiftool"
    except Exception:
        pass
    
    # Windows common paths
    if sys.platform == "win32":
        username = os.environ.get("USERNAME", "")
        common_paths = [
            Path(r"C:\Program Files\exiftool\exiftool.exe"),
            Path(r"C:\Program Files (x86)\exiftool\exiftool.exe"),
            Path(r"C:\exiftool\exiftool.exe"),
        ]
        
        if username:
            # Check common user folders
            desktop_et = Path(f"C:\\Users\\{username}\\Desktop\\UsefulPrograms\\exiftool\\exiftool.exe")
            pictures_et = Path(f"C:\\Users\\{username}\\Pictures\\TestEXIF\\exiftool.exe")
            common_paths.extend([desktop_et, pictures_et])
        
        for path in common_paths:
            if path.exists():
                try:
                    subprocess.run(
                        [str(path), "-ver"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5,
                    )
                    return str(path)
                except Exception:
                    pass
    
    raise RuntimeError(
        "exiftool not found on system. Required for MOV/video file support. "
        "Download from https://exiftool.org/ and add to PATH, or install to a common location."
    )


# Initialize exiftool once at startup
EXIFTOOL_PATH = _find_exiftool_executable()

# ---------------------------------------------------------------------------
# RunOptions (AD-7 frozen schema)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunOptions:
    target_root: Path
    recursive: bool
    dry_run: bool
    backup: bool


# ---------------------------------------------------------------------------
# FormatClass / FORMAT_CAPABILITIES (AD-8 single source of truth)
# ---------------------------------------------------------------------------


class FormatClass(str, Enum):
    WRITE_SUPPORT = "write-support"
    DETECT_ONLY = "detect-only"
    BEST_EFFORT = "best-effort"
    UNSUPPORTED = "unsupported"


# extension -> (FormatClass, handler key). DNG remains DETECT_ONLY per AD-8
# (gated per PRD Pre-release Validation Gate) pending its own promotion.
# HEIC/MP4/M4V/MOV promoted per spec-exifstamper-video-heic-promotion.md.
FORMAT_CAPABILITIES: Dict[str, tuple] = {
    ".jpg": (FormatClass.WRITE_SUPPORT, "image"),
    ".jpeg": (FormatClass.WRITE_SUPPORT, "image"),
    ".tif": (FormatClass.WRITE_SUPPORT, "image"),
    ".tiff": (FormatClass.WRITE_SUPPORT, "image"),
    ".webp": (FormatClass.WRITE_SUPPORT, "image"),
    ".png": (FormatClass.WRITE_SUPPORT, "image"),
    ".heic": (FormatClass.WRITE_SUPPORT, "image"),  # promoted, AD-8
    ".dng": (FormatClass.DETECT_ONLY, "detect_only"),  # gated, AD-8
    ".mp4": (FormatClass.WRITE_SUPPORT, "video"),  # promoted, AD-8
    ".m4v": (FormatClass.WRITE_SUPPORT, "video"),  # promoted, AD-8
    ".mov": (FormatClass.BEST_EFFORT, "video"),  # best-effort, same handler/contract as MP4/M4V
    ".cr2": (FormatClass.DETECT_ONLY, "detect_only"),
    ".nef": (FormatClass.DETECT_ONLY, "detect_only"),
    ".arw": (FormatClass.DETECT_ONLY, "detect_only"),
    ".orf": (FormatClass.DETECT_ONLY, "detect_only"),
    ".rw2": (FormatClass.DETECT_ONLY, "detect_only"),
}

# Sidecar backup suffix; files carrying it are never re-scanned as content.
BACKUP_SUFFIX = ".original"


# ---------------------------------------------------------------------------
# Shared data shapes (AD-7 -- frozen, not just named)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProtectionFacts:
    datetime_original: Optional[str]
    create_date: Optional[str]
    container_creation_time: Optional[str] = None


@dataclass(frozen=True)
class FilePlan:
    path: Path
    format_class: FormatClass
    disposition: Literal["stamp", "skip-protected", "detect-only", "unsupported"]
    normalized_dt: Optional[datetime.datetime]
    plan_fields: Dict[str, Any] = field(default_factory=dict)
    log_detail: str = ""


@dataclass(frozen=True)
class ApplyResult:
    status: Literal["applied", "best_effort_failed"]
    written_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MachineIdentity:
    host_computer: str
    make: str
    model: str


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _decode_exif_str(value: Any) -> Optional[str]:
    """Decode a raw EXIF tag value into a clean string, or None if empty."""
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii", errors="ignore")
        except Exception:
            return None
    value = str(value).strip().strip("\x00").strip()
    return value or None


def _read_exif_facts(path: Path, raise_on_unreadable: bool = False) -> ProtectionFacts:
    """Format-agnostic read of DateTimeOriginal/CreateDate.

    Distinguishes "opened fine, no datetime tag" (a legitimate unprotected
    state) from "could not open/parse the file at all". The former always
    yields empty facts. The latter only raises when `raise_on_unreadable`
    is True (write-support formats) so it surfaces through plan() to the
    single AD-4 catch site instead of silently masquerading as
    "unprotected" and then reaching apply(). DetectOnlyHandler always
    passes False since RAW/gated formats must log detect-only regardless
    of readability (I/O matrix row) and apply() is never called for them.
    """
    opened = False
    try:
        with Image.open(path) as img:
            opened = True
            exif = img.getexif()
            exif_ifd = exif.get_ifd(IFD.Exif) if exif else {}
            dto = _decode_exif_str(exif_ifd.get(36867))
            cd = _decode_exif_str(exif_ifd.get(36868))
            if dto or cd:
                return ProtectionFacts(datetime_original=dto, create_date=cd)
    except Exception:
        pass
    try:
        exif_dict = piexif.load(str(path))
        opened = True
        dto = _decode_exif_str(exif_dict.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal))
        cd = _decode_exif_str(exif_dict.get("Exif", {}).get(piexif.ExifIFD.DateTimeDigitized))
        return ProtectionFacts(datetime_original=dto, create_date=cd)
    except Exception:
        pass
    if not opened and raise_on_unreadable:
        raise RuntimeError(f"unreadable/corrupt file, cannot verify protection state: {path}")
    return ProtectionFacts(datetime_original=None, create_date=None)


def _format_utc_offset(for_dt: datetime.datetime) -> str:
    """UTC offset for the *stamped* date, not 'now' -- required so the
    written OffsetTime is DST-correct for dates on the other side of a
    DST boundary from the run date."""
    offset = for_dt.astimezone().utcoffset()
    total_minutes = int((offset or datetime.timedelta()).total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    total_minutes = abs(total_minutes)
    return f"{sign}{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def resolve_machine_identity() -> MachineIdentity:
    """Resolve WMI PC identity (with motherboard fallback) and hostname
    exactly once per run (AD-6). Never raises -- falls back to 'Unknown'."""
    hostname = socket.gethostname()
    make: Optional[str] = None
    model: Optional[str] = None
    mb_make: Optional[str] = None
    mb_model: Optional[str] = None
    try:
        import wmi  # type: ignore

        conn = wmi.WMI()
        for sysinfo in conn.Win32_ComputerSystem():
            make = (sysinfo.Manufacturer or "").strip()
            model = (sysinfo.Model or "").strip()
        for board in conn.Win32_BaseBoard():
            mb_make = (board.Manufacturer or "").strip()
            mb_model = (board.Product or "").strip()
    except Exception:
        pass

    placeholder_markers = {"", "system manufacturer", "to be filled by o.e.m.", "default string"}
    if not make or make.lower() in placeholder_markers:
        make = mb_make or make or "Unknown"
    if not model or model.lower() in placeholder_markers:
        model = mb_model or model or "Unknown"
    return MachineIdentity(host_computer=hostname, make=make, model=model)


# ---------------------------------------------------------------------------
# Format handlers (AD-2: swappable strategy, pure data-in/data-out per AD-9)
# ---------------------------------------------------------------------------


PIEXIF_INSERT_EXTS = {".jpg", ".jpeg"}
PILLOW_SAVE_EXTS = {".png", ".webp", ".tif", ".tiff"}


def _build_exif_dict(f: Dict[str, Any]) -> dict:
    def enc(s: Any) -> bytes:
        return str(s).encode("ascii", errors="ignore")

    zeroth = {
        piexif.ImageIFD.Make: enc(f["Make"]),
        piexif.ImageIFD.Model: enc(f["Model"]),
        piexif.ImageIFD.Software: enc(f["Software"]),
        piexif.ImageIFD.HostComputer: enc(f["HostComputer"]),
        piexif.ImageIFD.Orientation: f["Orientation"],
        piexif.ImageIFD.ResolutionUnit: f["ResolutionUnit"],
        piexif.ImageIFD.XResolution: (int(f["XResolution"]), 1),
        piexif.ImageIFD.YResolution: (int(f["YResolution"]), 1),
        piexif.ImageIFD.DateTime: enc(f["ModifyDate"]),
    }
    exif_ifd: Dict[int, Any] = {
        piexif.ExifIFD.DateTimeOriginal: enc(f["DateTimeOriginal"]),
        piexif.ExifIFD.DateTimeDigitized: enc(f["CreateDate"]),
        piexif.ExifIFD.OffsetTime: enc(f["OffsetTime"]),
        piexif.ExifIFD.OffsetTimeOriginal: enc(f["OffsetTimeOriginal"]),
        piexif.ExifIFD.OffsetTimeDigitized: enc(f["OffsetTimeDigitized"]),
        piexif.ExifIFD.ExifVersion: enc(f["ExifVersion"]),
        piexif.ExifIFD.FlashpixVersion: enc(f["FlashpixVersion"]),
        piexif.ExifIFD.ColorSpace: f["ColorSpace"],
        piexif.ExifIFD.Flash: f["Flash"],
    }
    if f.get("ExifImageWidth") is not None:
        exif_ifd[piexif.ExifIFD.PixelXDimension] = int(f["ExifImageWidth"])
    if f.get("ExifImageHeight") is not None:
        exif_ifd[piexif.ExifIFD.PixelYDimension] = int(f["ExifImageHeight"])
    return {"0th": zeroth, "Exif": exif_ifd, "1st": {}, "GPS": {}, "Interop": {}, "thumbnail": None}


class ImageExifHandler:
    """JPEG/TIFF/WebP/PNG -- piexif in-place insert for JPEG (no
    recompression); Pillow save(exif=...) for TIFF/PNG/WebP (piexif's
    insert() only supports JPEG/WebP containers directly, and its WebP
    path re-encodes anyway, so Pillow save is used uniformly for the
    other three)."""

    format_class = FormatClass.WRITE_SUPPORT

    def read_protection_facts(self, path: Path) -> ProtectionFacts:
        return _read_exif_facts(path, raise_on_unreadable=True)

    def build_plan_fields(
        self,
        path: Path,
        normalized_dt: Optional[datetime.datetime],
        machine_identity: MachineIdentity,
    ) -> Dict[str, Any]:
        assert normalized_dt is not None
        dt_str = normalized_dt.strftime("%Y:%m:%d %H:%M:%S")
        offset = _format_utc_offset(normalized_dt)

        width = height = None
        xres = yres = 72
        try:
            with Image.open(path) as img:
                width, height = img.size
                dpi = img.info.get("dpi")
                if dpi:
                    xres, yres = dpi
        except Exception:
            pass

        return {
            "DateTimeOriginal": dt_str,
            "CreateDate": dt_str,
            "ModifyDate": dt_str,
            "OffsetTime": offset,
            "OffsetTimeDigitized": offset,
            "OffsetTimeOriginal": offset,
            "HostComputer": machine_identity.host_computer,
            "Make": machine_identity.make,
            "Model": machine_identity.model,
            "ExifImageWidth": width,
            "ExifImageHeight": height,
            "XResolution": xres,
            "YResolution": yres,
            "Software": "EXIFStamper",
            "Orientation": 1,
            "ResolutionUnit": 2,
            "ExifVersion": "0232",
            "FlashpixVersion": "0100",
            "ColorSpace": 1,
            "Flash": 0,
        }

    def apply(self, path: Path, plan_fields: Dict[str, Any]) -> ApplyResult:
        exif_dict = _build_exif_dict(plan_fields)
        exif_bytes = piexif.dump(exif_dict)
        ext = path.suffix.lower()
        if ext in PIEXIF_INSERT_EXTS:
            piexif.insert(exif_bytes, str(path))
        else:
            with Image.open(path) as img:
                if getattr(img, "n_frames", 1) > 1:
                    # Multi-frame/animated PNG/WebP: img.save() without
                    # save_all=True silently flattens to a single frame,
                    # dropping data. Refuse rather than corrupt -- surfaces
                    # as a per-file error (AD-4), run continues.
                    raise RuntimeError(f"animated/multi-frame image unsupported for write: {path}")
                img.save(path, exif=exif_bytes)
        return ApplyResult(status="applied", written_fields=dict(plan_fields))


class DetectOnlyHandler:
    """RAW (CR2/NEF/ARW/ORF/RW2) plus DNG pending promotion (AD-8).
    apply() must never be called (AD-3)."""

    format_class = FormatClass.DETECT_ONLY

    def read_protection_facts(self, path: Path) -> ProtectionFacts:
        return _read_exif_facts(path)

    def build_plan_fields(
        self,
        path: Path,
        normalized_dt: Optional[datetime.datetime],
        machine_identity: MachineIdentity,
    ) -> Dict[str, Any]:
        return {}

    def apply(self, path: Path, plan_fields: Dict[str, Any]) -> ApplyResult:
        raise NotImplementedError("DetectOnlyHandler.apply() must never be called (AD-3)")


class VideoAtomHandler:
    """MP4/M4V (mutagen) and MOV (exiftool) -- container atom writing for video metadata.
    MP4/M4V use mutagen to write the `©day` atom; MOV uses exiftool since mutagen's
    MOV support is unreliable. Requires exiftool for MOV support."""

    format_class = FormatClass.WRITE_SUPPORT

    DAY_ATOM = "\xa9day"

    def read_protection_facts(self, path: Path) -> ProtectionFacts:
        try:
            mp4 = MP4(str(path))
        except MP4StreamInfoError as exc:
            raise RuntimeError(f"not a valid MP4/MOV container: {path}") from exc
        day = None
        if mp4.tags:
            values = mp4.tags.get(self.DAY_ATOM)
            if values:
                day = _decode_exif_str(values[0])
        return ProtectionFacts(datetime_original=None, create_date=None, container_creation_time=day)

    def build_plan_fields(
        self,
        path: Path,
        normalized_dt: Optional[datetime.datetime],
        machine_identity: MachineIdentity,
    ) -> Dict[str, Any]:
        assert normalized_dt is not None
        return {"ContainerCreationTime": normalized_dt.strftime("%Y-%m-%d")}

    def apply(self, path: Path, plan_fields: Dict[str, Any]) -> ApplyResult:
        ext = path.suffix.lower()
        
        if ext == ".mov":
            # MOV files: use exiftool (required)
            creation_time = plan_fields["ContainerCreationTime"]
            
            try:
                # exiftool writes in-place and preserves original
                subprocess.run(
                    [
                        EXIFTOOL_PATH,
                        "-overwrite_original",
                        f"-CreationTime={creation_time}",
                        str(path),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=60,
                    check=True,
                )
                return ApplyResult(status="applied", written_fields=dict(plan_fields))
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"exiftool failed to write MOV metadata: {exc}") from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"exiftool timeout: {exc}") from exc
        else:
            # MP4/M4V: use mutagen
            mp4 = MP4(str(path))
            mp4.tags = mp4.tags or type(mp4).MP4Tags()
            mp4.tags[self.DAY_ATOM] = [plan_fields["ContainerCreationTime"]]
            mp4.save()
            return ApplyResult(status="applied", written_fields=dict(plan_fields))


HANDLERS = {
    "image": ImageExifHandler(),
    "detect_only": DetectOnlyHandler(),
    "video": VideoAtomHandler(),
}


# ---------------------------------------------------------------------------
# Pipeline: scan -> classify -> plan -> apply -> log
# ---------------------------------------------------------------------------


def scan(run_options: RunOptions):
    """Yield file paths under target_root. Skips our own backup sidecars
    and the run log so re-running never mistakes our own output for
    content."""
    root = run_options.target_root
    if run_options.recursive:
        iterator = root.rglob("*")
    else:
        iterator = root.glob("*")
    for candidate in iterator:
        if not candidate.is_file():
            continue
        if candidate.name.endswith(BACKUP_SUFFIX):
            continue
        if candidate.name == LOG_FILENAME:
            continue
        yield candidate


def classify(path: Path):
    """Return (FormatClass, handler) for path's extension, or None if
    unsupported (FR-1)."""
    ext = path.suffix.lower()
    cap = FORMAT_CAPABILITIES.get(ext)
    if cap is None:
        return None
    format_class, handler_key = cap
    return format_class, HANDLERS[handler_key]


def plan(path: Path, machine_identity: MachineIdentity) -> FilePlan:
    """Pure, read-only (AD-1): classification + protection decision + date
    normalization + full field computation, from a single stat() snapshot.
    Never opens the file in write mode."""
    classification = classify(path)
    if classification is None:
        return FilePlan(
            path=path,
            format_class=FormatClass.UNSUPPORTED,
            disposition="unsupported",
            normalized_dt=None,
            log_detail=f"unsupported extension '{path.suffix}'",
        )

    format_class, handler = classification
    facts = handler.read_protection_facts(path)
    protected = bool(facts.datetime_original) or bool(facts.create_date) or bool(
        facts.container_creation_time
    )

    if format_class == FormatClass.DETECT_ONLY:
        return FilePlan(
            path=path,
            format_class=format_class,
            disposition="detect-only",
            normalized_dt=None,
            log_detail=f"detect-only ({'protected' if protected else 'unprotected'})",
        )

    if protected:
        return FilePlan(
            path=path,
            format_class=format_class,
            disposition="skip-protected",
            normalized_dt=None,
            log_detail="skip-protected: existing datetime metadata present",
        )

    # Unprotected write-support file -- compute the full stamp plan (FR-3, FR-4).
    st = path.stat()
    ctime = datetime.datetime.fromtimestamp(st.st_ctime)
    mtime = datetime.datetime.fromtimestamp(st.st_mtime)
    normalized_dt = min(ctime, mtime)
    plan_fields = handler.build_plan_fields(path, normalized_dt, machine_identity)
    return FilePlan(
        path=path,
        format_class=format_class,
        disposition="stamp",
        normalized_dt=normalized_dt,
        plan_fields=plan_fields,
        log_detail=f"stamp with {normalized_dt.strftime('%Y:%m:%d %H:%M:%S')}",
    )


def apply(file_plan: FilePlan, run_options: RunOptions) -> ApplyResult:
    """Pipeline-owned: backup (AD-5) then dispatch write to the handler.
    Only ever called for disposition == 'stamp' (AD-3)."""
    if file_plan.disposition != "stamp":
        raise AssertionError(f"apply() called for non-stamp disposition: {file_plan.disposition}")

    format_class, handler = classify(file_plan.path)  # type: ignore[misc]

    if run_options.backup:
        backup_path = file_plan.path.with_name(file_plan.path.name + BACKUP_SUFFIX)
        if backup_path.exists():
            raise RuntimeError(f"backup sidecar already exists, refusing to overwrite: {backup_path}")
        shutil.copy2(file_plan.path, backup_path)
        if not backup_path.exists() or backup_path.stat().st_size != file_plan.path.stat().st_size:
            raise RuntimeError(f"backup verification failed for {file_plan.path}")

    return handler.apply(file_plan.path, file_plan.plan_fields)


def _log_line(run_options: RunOptions, path: Path, disposition: str, detail: str) -> str:
    prefix = "[DRY RUN] " if run_options.dry_run else ""
    return f"{prefix}{path} | {disposition} | {detail}"


def run(run_options: RunOptions) -> int:
    machine_identity = resolve_machine_identity()
    log_path = SCRIPT_DIR / LOG_FILENAME
    error_count = 0

    with open(log_path, "w", encoding="utf-8") as log_file:
        for path in scan(run_options):
            try:
                file_plan = plan(path, machine_identity)
                if file_plan.disposition == "stamp" and not run_options.dry_run:
                    apply(file_plan, run_options)
                log_file.write(_log_line(run_options, path, file_plan.disposition, file_plan.log_detail) + "\n")
            except Exception as exc:  # AD-4: the ONLY per-file catch site
                error_count += 1
                reason = f"{type(exc).__name__}: {exc}"
                log_file.write(_log_line(run_options, path, "error", reason) + "\n")

    return 1 if error_count else 0


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> RunOptions:
    parser = argparse.ArgumentParser(
        prog="stamper.py",
        description="Stamp durable timestamps into unprotected photo/video files.",
    )
    parser.add_argument("directory", type=str, help="Target directory to scan.")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories.")
    parser.add_argument("--dry-run", action="store_true", help="Compute and log without writing.")
    parser.add_argument("--backup", action="store_true", help="Create a .original sidecar before writing.")
    args = parser.parse_args(argv)
    return RunOptions(
        target_root=Path(args.directory).resolve(),
        recursive=args.recursive,
        dry_run=args.dry_run,
        backup=args.backup,
    )


def main(argv=None) -> int:
    run_options = parse_args(argv)
    if not run_options.target_root.is_dir():
        # Spec mandates zero console output; signal via exit code only.
        return 2
    try:
        return run(run_options)
    except Exception:
        # Top-level safety net for failures outside the per-file loop
        # (e.g. cannot open stamper.log itself). No console output per spec.
        return 2


if __name__ == "__main__":
    sys.exit(main())
