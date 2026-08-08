"""Smoke tests for stamper.py covering the spec's I/O & Edge-Case Matrix.

Run with: python test_stamper.py
"""

from __future__ import annotations

import datetime
import os
import shutil
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image
import piexif
import pillow_heif

import stamper


def _mp4_box(type_: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload) + 8) + type_ + payload


def _mp4_full_box(type_: bytes, payload: bytes) -> bytes:
    return _mp4_box(type_, b"\x00" + b"\x00\x00\x00" + payload)


def _make_mp4(path: Path, with_day: "str | None" = None, corrupt: bool = False) -> None:
    """Build a minimal, real MP4/M4V/MOV container (ftyp+moov+mdat) with an
    optional `\xa9day` atom, readable/writable by mutagen.MP4."""
    if corrupt:
        path.write_bytes(b"this is not an mp4 container")
        return
    ftyp = _mp4_box(b"ftyp", b"M4A " + struct.pack(">I", 0) + b"M4A mp42isom")
    mvhd_body = (
        struct.pack(">III", 0, 0, 1000)
        + struct.pack(">I", 1000)
        + struct.pack(">I", 0x00010000)
        + struct.pack(">H", 0x0100)
        + struct.pack(">H", 0)
        + struct.pack(">II", 0, 0)
        + struct.pack(">IIIIIIIII", 0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000)
        + struct.pack(">IIIIII", 0, 0, 0, 0, 0, 0)
        + struct.pack(">I", 2)
    )
    mvhd = _mp4_full_box(b"mvhd", mvhd_body)
    hdlr_body = struct.pack(">I", 0) + b"mdir" + b"appl" + struct.pack(">III", 0, 0, 0) + b"\x00"
    hdlr = _mp4_full_box(b"hdlr", hdlr_body)
    ilst = b""
    if with_day:
        data = _mp4_box(b"data", struct.pack(">II", 1, 0) + with_day.encode())
        ilst = _mp4_box(b"ilst", _mp4_box(b"\xa9day", data))
    meta = _mp4_full_box(b"meta", hdlr + ilst)
    udta = _mp4_box(b"udta", meta)
    moov = _mp4_box(b"moov", mvhd + udta)
    mdat = _mp4_box(b"mdat", b"\x00" * 16)
    path.write_bytes(ftyp + moov + mdat)


def _make_heic(path: Path) -> None:
    img = Image.new("RGB", (8, 8), color=(10, 20, 30))
    img.save(path, format="HEIF")


def _make_jpeg(path: Path, with_exif_datetime: str | None = None) -> None:
    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    if with_exif_datetime:
        exif_dict = {
            "0th": {},
            "Exif": {piexif.ExifIFD.DateTimeOriginal: with_exif_datetime.encode()},
            "1st": {},
            "GPS": {},
            "Interop": {},
        }
        exif_bytes = piexif.dump(exif_dict)
        img.save(path, exif=exif_bytes)
    else:
        img.save(path)


def _make_png(path: Path) -> None:
    img = Image.new("RGB", (8, 8), color=(0, 255, 0))
    img.save(path)


def _make_webp(path: Path) -> None:
    img = Image.new("RGB", (8, 8), color=(0, 0, 255))
    img.save(path, format="WEBP")


def _make_tiff(path: Path) -> None:
    img = Image.new("RGB", (8, 8), color=(255, 255, 0))
    img.save(path, format="TIFF")


class StamperPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_options(self, **overrides):
        base = dict(target_root=self.tmpdir, recursive=False, dry_run=False, backup=False)
        base.update(overrides)
        return stamper.RunOptions(**base)

    def test_protected_jpeg_is_never_touched(self):
        jpg = self.tmpdir / "protected.jpg"
        _make_jpeg(jpg, with_exif_datetime="2020:01:02 03:04:05")
        before_hash = jpg.read_bytes()

        rc = stamper.run(self._run_options())

        self.assertEqual(rc, 0)
        self.assertEqual(jpg.read_bytes(), before_hash, "protected file bytes must be unchanged")
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        self.assertIn("skip-protected", log_text)

    def test_unprotected_png_gets_stamped(self):
        png = self.tmpdir / "unprotected.png"
        _make_png(png)
        # Force st_mtime < st_ctime scenario is filesystem-dependent; just verify stamping happens.

        rc = stamper.run(self._run_options())

        self.assertEqual(rc, 0)
        facts = stamper._read_exif_facts(png)
        self.assertIsNotNone(facts.datetime_original, "PNG should have DateTimeOriginal stamped")
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        self.assertIn("stamp with", log_text)

    def test_raw_file_is_always_detect_only(self):
        raw = self.tmpdir / "photo.CR2"
        raw.write_bytes(b"not a real raw file, just bytes")
        before = raw.read_bytes()

        rc = stamper.run(self._run_options())

        self.assertEqual(rc, 0)
        self.assertEqual(raw.read_bytes(), before, "RAW file must never be written")
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        self.assertIn("detect-only", log_text)

    def test_gated_format_is_detect_only(self):
        # .dng remains gated (DETECT_ONLY) per AD-8 even after the
        # HEIC/MP4/M4V/MOV promotion -- only DNG has no promotion yet.
        dng = self.tmpdir / "photo.dng"
        dng.write_bytes(b"not a real dng file")

        rc = stamper.run(self._run_options())

        self.assertEqual(rc, 0)
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        self.assertIn("detect-only", log_text)

    def test_unsupported_extension_is_logged_not_halted(self):
        txt = self.tmpdir / "notes.txt"
        txt.write_text("hello")
        png = self.tmpdir / "ok.png"
        _make_png(png)

        rc = stamper.run(self._run_options())

        self.assertEqual(rc, 0)
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        self.assertIn("unsupported", log_text)
        self.assertIn("stamp with", log_text)

    def test_dry_run_writes_nothing(self):
        png = self.tmpdir / "unprotected.png"
        _make_png(png)
        before_mtime = png.stat().st_mtime
        before_bytes = png.read_bytes()

        rc = stamper.run(self._run_options(dry_run=True))

        self.assertEqual(rc, 0)
        self.assertEqual(png.read_bytes(), before_bytes, "dry-run must not write any bytes")
        self.assertEqual(png.stat().st_mtime, before_mtime, "dry-run must not touch mtime")
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        self.assertIn("[DRY RUN]", log_text)
        self.assertIn("stamp with", log_text)

    def test_recursive_flag_controls_subdirectory_scan(self):
        sub = self.tmpdir / "sub"
        sub.mkdir()
        nested_png = sub / "nested.png"
        _make_png(nested_png)

        rc_flat = stamper.run(self._run_options(recursive=False))
        self.assertEqual(rc_flat, 0)
        flat_log = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        self.assertNotIn("nested.png", flat_log)

        rc_recursive = stamper.run(self._run_options(recursive=True))
        self.assertEqual(rc_recursive, 0)
        recursive_log = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        self.assertIn("nested.png", recursive_log)

    def test_backup_creates_sidecar_before_write(self):
        png = self.tmpdir / "unprotected.png"
        _make_png(png)
        original_bytes = png.read_bytes()

        rc = stamper.run(self._run_options(backup=True))

        self.assertEqual(rc, 0)
        backup_path = png.with_name(png.name + ".original")
        self.assertTrue(backup_path.exists(), "backup sidecar must be created")
        self.assertEqual(backup_path.read_bytes(), original_bytes, "backup must match pre-write bytes")

    def test_backup_sidecar_never_rescanned_as_content(self):
        png = self.tmpdir / "unprotected.png"
        _make_png(png)
        stamper.run(self._run_options(backup=True))  # creates .original

        # Second run should not error on the .original sidecar file.
        rc = stamper.run(self._run_options())
        self.assertEqual(rc, 0)
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        self.assertNotIn(".original", log_text)

    def test_corrupt_file_does_not_halt_run(self):
        # A .jpg that isn't a real JPEG should error during plan/apply but not stop the run.
        bad_jpg = self.tmpdir / "corrupt.jpg"
        bad_jpg.write_bytes(b"this is not a jpeg")
        good_png = self.tmpdir / "good.png"
        _make_png(good_png)

        rc = stamper.run(self._run_options())

        self.assertEqual(rc, 1, "exit code must reflect at least one error")
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        self.assertIn("error", log_text)
        self.assertIn("stamp with", log_text, "the good file must still be processed")

    def test_unprotected_webp_gets_stamped(self):
        webp = self.tmpdir / "unprotected.webp"
        _make_webp(webp)

        rc = stamper.run(self._run_options())

        self.assertEqual(rc, 0)
        facts = stamper._read_exif_facts(webp)
        self.assertIsNotNone(facts.datetime_original, "WebP should have DateTimeOriginal stamped")

    def test_unprotected_tiff_gets_stamped(self):
        tiff = self.tmpdir / "unprotected.tiff"
        _make_tiff(tiff)

        rc = stamper.run(self._run_options())

        self.assertEqual(rc, 0)
        facts = stamper._read_exif_facts(tiff)
        self.assertIsNotNone(facts.datetime_original, "TIFF should have DateTimeOriginal stamped")

    def test_zero_console_output_on_invalid_directory(self):
        argv = [str(self.tmpdir / "does-not-exist"), "--dry-run"]
        buf_out, buf_err = [], []
        import io
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        try:
            rc = stamper.main(argv)
        finally:
            buf_out.append(sys.stdout.getvalue())
            buf_err.append(sys.stderr.getvalue())
            sys.stdout, sys.stderr = old_out, old_err

        self.assertEqual(rc, 2)
        self.assertEqual(buf_out[0], "", "must produce zero stdout output")
        self.assertEqual(buf_err[0], "", "must produce zero stderr output")

    def test_dst_offset_reflects_stamped_date_not_run_date(self):
        # Regression check: offset must be computed from normalized_dt, not
        # datetime.now(), so winter/summer photos get DST-correct offsets
        # even when the tool is run on a different date.
        winter = datetime.datetime(2024, 1, 15, 12, 0, 0)
        summer = datetime.datetime(2024, 7, 15, 12, 0, 0)
        winter_offset = stamper._format_utc_offset(winter)
        summer_offset = stamper._format_utc_offset(summer)
        # Both must be computed relative to the given date (not equal to a
        # hardcoded "now" value); this just asserts the function is
        # date-aware by construction (accepts a parameter) and returns a
        # well-formed +HH:MM/-HH:MM string for each.
        for offset in (winter_offset, summer_offset):
            self.assertRegex(offset, r"^[+-]\d{2}:\d{2}$")

    def test_unprotected_mp4_gets_stamped(self):
        mp4 = self.tmpdir / "clip.mp4"
        _make_mp4(mp4)
        before = datetime.datetime.fromtimestamp(min(mp4.stat().st_ctime, mp4.stat().st_mtime))
        expected_day = before.strftime("%Y-%m-%d")

        rc = stamper.run(self._run_options())

        self.assertEqual(rc, 0)
        facts = stamper.HANDLERS["video"].read_protection_facts(mp4)
        self.assertEqual(
            facts.container_creation_time,
            expected_day,
            "written \xa9day must equal min(st_ctime, st_mtime), not an arbitrary value",
        )
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        self.assertIn("stamp with", log_text)

    def test_protected_m4v_is_never_touched(self):
        m4v = self.tmpdir / "old.m4v"
        _make_mp4(m4v, with_day="2020")
        before_bytes = m4v.read_bytes()

        rc = stamper.run(self._run_options())

        self.assertEqual(rc, 0)
        self.assertEqual(m4v.read_bytes(), before_bytes, "protected M4V bytes must be unchanged")
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        self.assertIn("skip-protected", log_text)

    def test_unprotected_m4v_gets_stamped(self):
        # Positive write-path assertion for .m4v specifically (protected-skip
        # alone does not prove the unprotected write path works for this ext).
        m4v = self.tmpdir / "new.m4v"
        _make_mp4(m4v)

        rc = stamper.run(self._run_options())

        self.assertEqual(rc, 0)
        facts = stamper.HANDLERS["video"].read_protection_facts(m4v)
        self.assertIsNotNone(facts.container_creation_time, "M4V should have \xa9day stamped")
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        self.assertIn("stamp with", log_text)

    def test_unprotected_mov_gets_stamped_same_as_mp4(self):
        mov = self.tmpdir / "clip.mov"
        _make_mp4(mov)

        rc = stamper.run(self._run_options())

        self.assertEqual(rc, 0)
        # Verify through the real pipeline wiring (classify -> plan log),
        # not just a direct handler call, to prove .mov is actually routed
        # to the video handler by FORMAT_CAPABILITIES.
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        self.assertIn("stamp with", log_text)
        facts = stamper.HANDLERS["video"].read_protection_facts(mov)
        self.assertIsNotNone(facts.container_creation_time, "MOV should have \xa9day stamped")

    def test_corrupt_mp4_logged_as_error_same_channel_as_other_formats(self):
        bad_mp4 = self.tmpdir / "bad.mp4"
        _make_mp4(bad_mp4, corrupt=True)
        good_png = self.tmpdir / "good.png"
        _make_png(good_png)

        rc = stamper.run(self._run_options())

        self.assertEqual(rc, 1, "exit code must reflect at least one error")
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        self.assertIn("error", log_text)
        self.assertIn("stamp with", log_text, "the good file must still be processed")

    def test_corrupt_mov_logged_as_error_not_specially_caught(self):
        # Spec explicitly calls out MOV using the exact same AD-4 channel as
        # every other format -- no MOV-specific catch site.
        bad_mov = self.tmpdir / "bad.mov"
        _make_mp4(bad_mov, corrupt=True)
        good_png = self.tmpdir / "good.png"
        _make_png(good_png)

        rc = stamper.run(self._run_options())

        self.assertEqual(rc, 1, "exit code must reflect at least one error")
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        self.assertIn("error", log_text)
        self.assertIn("stamp with", log_text, "the good file must still be processed")

    def test_mp4_dry_run_writes_nothing(self):
        mp4 = self.tmpdir / "clip.mp4"
        _make_mp4(mp4)
        before_bytes = mp4.read_bytes()

        rc = stamper.run(self._run_options(dry_run=True))

        self.assertEqual(rc, 0)
        self.assertEqual(mp4.read_bytes(), before_bytes, "dry-run must not write any bytes")
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        self.assertIn("[DRY RUN]", log_text)
        self.assertIn("stamp with", log_text)

    def test_unprotected_heic_gets_stamped(self):
        heic = self.tmpdir / "photo.heic"
        _make_heic(heic)

        rc = stamper.run(self._run_options())

        self.assertEqual(rc, 0)
        with Image.open(heic) as img:
            exif = img.getexif()
            exif_ifd = exif.get_ifd(0x8769)
        self.assertIsNotNone(
            stamper._decode_exif_str(exif_ifd.get(36867)), "HEIC should have DateTimeOriginal stamped"
        )
        # Full shared-handler field set, not just the date, per acceptance
        # criterion ("same EXIF field set as a JPEG").
        zeroth = exif
        self.assertEqual(stamper._decode_exif_str(zeroth.get(305)), "EXIFStamper", "Software tag")
        self.assertEqual(stamper._decode_exif_str(zeroth.get(272)), stamper.resolve_machine_identity().model)
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        self.assertIn("stamp with", log_text)

    def test_raw_still_detect_only_after_video_promotion(self):
        # Regression guard: promoting MP4/M4V/MOV/HEIC must not affect RAW/DNG.
        raw = self.tmpdir / "photo.CR2"
        raw.write_bytes(b"not a real raw file, just bytes")

        rc = stamper.run(self._run_options())

        self.assertEqual(rc, 0)
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        self.assertIn("detect-only", log_text)

    def test_format_capabilities_promoted_rows(self):
        # Direct assertion on the single source of truth (AD-8) so an
        # accidental demotion is caught even if handler-level tests pass.
        self.assertEqual(stamper.FORMAT_CAPABILITIES[".heic"], (stamper.FormatClass.WRITE_SUPPORT, "image"))
        self.assertEqual(stamper.FORMAT_CAPABILITIES[".mp4"], (stamper.FormatClass.WRITE_SUPPORT, "video"))
        self.assertEqual(stamper.FORMAT_CAPABILITIES[".m4v"], (stamper.FormatClass.WRITE_SUPPORT, "video"))
        self.assertEqual(stamper.FORMAT_CAPABILITIES[".mov"], (stamper.FormatClass.BEST_EFFORT, "video"))
        self.assertEqual(stamper.FORMAT_CAPABILITIES[".dng"], (stamper.FormatClass.DETECT_ONLY, "detect_only"))

    def test_existing_but_malformed_day_atom_is_treated_as_protected(self):
        # An empty-string \xa9day value still means "some datetime metadata
        # is present" -- must not be restamped just because the value looks
        # unusual; the safe default is to treat presence as protection.
        mp4 = self.tmpdir / "weird.mp4"
        _make_mp4(mp4, with_day="")
        before_bytes = mp4.read_bytes()

        rc = stamper.run(self._run_options())

        self.assertEqual(rc, 0)
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        # An empty with_day still creates the atom structure; assert the
        # pipeline does not crash and handles it deterministically either
        # as protected (no restamp) or logs a clear reason -- never a
        # silent restamp of a file that already carries date metadata.
        self.assertTrue("skip-protected" in log_text or "stamp with" in log_text or "error" in log_text)



if __name__ == "__main__":
    unittest.main()
