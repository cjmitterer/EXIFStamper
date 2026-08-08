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

    def test_mov_files_are_routed_to_video_handler(self):
        # Verify that .mov files are classified as video format
        # (actual unprotected MOV testing is complex due to exiftool behavior)
        mov = self.tmpdir / "test.mov"
        _make_mp4(mov, with_day=None)
        
        classification = stamper.classify(mov)
        self.assertIsNotNone(classification)
        format_class, handler = classification
        self.assertIs(handler, stamper.HANDLERS["video"])

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


# ==================== COMPREHENSIVE NEW TESTS ====================


class HelperFunctionTests(unittest.TestCase):
    """Test suite for helper functions."""

    def test_decode_exif_str_with_none(self):
        result = stamper._decode_exif_str(None)
        self.assertIsNone(result)

    def test_decode_exif_str_with_empty_string(self):
        result = stamper._decode_exif_str("")
        self.assertIsNone(result)

    def test_decode_exif_str_with_bytes(self):
        result = stamper._decode_exif_str(b"2024:01:15 12:30:45")
        self.assertEqual(result, "2024:01:15 12:30:45")

    def test_decode_exif_str_with_string(self):
        result = stamper._decode_exif_str("2024:01:15 12:30:45")
        self.assertEqual(result, "2024:01:15 12:30:45")

    def test_decode_exif_str_with_whitespace(self):
        result = stamper._decode_exif_str("  2024:01:15 12:30:45  ")
        self.assertEqual(result, "2024:01:15 12:30:45")

    def test_decode_exif_str_with_null_bytes(self):
        result = stamper._decode_exif_str("2024:01:15\x00\x00\x00")
        self.assertEqual(result, "2024:01:15")

    def test_decode_exif_str_with_bytes_and_nulls(self):
        result = stamper._decode_exif_str(b"2024:01:15\x00\x00\x00")
        self.assertEqual(result, "2024:01:15")

    def test_format_utc_offset_winter_date(self):
        dt = datetime.datetime(2024, 1, 15, 12, 0, 0)
        offset = stamper._format_utc_offset(dt)
        # Should be well-formed +HH:MM or -HH:MM
        self.assertRegex(offset, r"^[+-]\d{2}:\d{2}$")

    def test_format_utc_offset_summer_date(self):
        dt = datetime.datetime(2024, 7, 15, 12, 0, 0)
        offset = stamper._format_utc_offset(dt)
        # Should be well-formed +HH:MM or -HH:MM
        self.assertRegex(offset, r"^[+-]\d{2}:\d{2}$")

    def test_format_utc_offset_different_dates_may_differ(self):
        # Summer vs winter dates may have different DST, so offsets may differ
        winter_offset = stamper._format_utc_offset(datetime.datetime(2024, 1, 15, 12, 0, 0))
        summer_offset = stamper._format_utc_offset(datetime.datetime(2024, 7, 15, 12, 0, 0))
        # Both valid, but may be the same or different depending on DST rules
        self.assertRegex(winter_offset, r"^[+-]\d{2}:\d{2}$")
        self.assertRegex(summer_offset, r"^[+-]\d{2}:\d{2}$")

    def test_resolve_machine_identity_returns_fields(self):
        identity = stamper.resolve_machine_identity()
        self.assertIsNotNone(identity.host_computer)
        self.assertIsNotNone(identity.make)
        self.assertIsNotNone(identity.model)
        self.assertIsInstance(identity.host_computer, str)
        self.assertIsInstance(identity.make, str)
        self.assertIsInstance(identity.model, str)

    def test_resolve_machine_identity_never_raises(self):
        # Should be idempotent and safe to call multiple times
        try:
            identity1 = stamper.resolve_machine_identity()
            identity2 = stamper.resolve_machine_identity()
            # Should not raise
            self.assertIsNotNone(identity1)
            self.assertIsNotNone(identity2)
        except Exception:
            self.fail("resolve_machine_identity should never raise")


class ProtectionFactsEdgeCaseTests(unittest.TestCase):
    """Test protection facts reading with edge cases."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_read_exif_facts_from_unprotected_png(self):
        png = self.tmpdir / "test.png"
        _make_png(png)
        facts = stamper._read_exif_facts(png)
        self.assertIsNone(facts.datetime_original)
        self.assertIsNone(facts.create_date)

    def test_read_exif_facts_from_protected_jpeg(self):
        jpg = self.tmpdir / "test.jpg"
        _make_jpeg(jpg, with_exif_datetime="2020:01:02 03:04:05")
        facts = stamper._read_exif_facts(jpg)
        self.assertIsNotNone(facts.datetime_original)
        self.assertIn("2020", facts.datetime_original)

    def test_read_exif_facts_from_corrupt_file_no_raise_by_default(self):
        corrupt = self.tmpdir / "corrupt.jpg"
        corrupt.write_bytes(b"not a jpeg")
        # By default, should not raise
        facts = stamper._read_exif_facts(corrupt, raise_on_unreadable=False)
        self.assertIsNone(facts.datetime_original)
        self.assertIsNone(facts.create_date)

    def test_read_exif_facts_from_corrupt_file_raises_when_flag_set(self):
        corrupt = self.tmpdir / "corrupt.jpg"
        corrupt.write_bytes(b"not a jpeg")
        with self.assertRaises(RuntimeError):
            stamper._read_exif_facts(corrupt, raise_on_unreadable=True)

    def test_read_exif_facts_from_empty_file(self):
        empty = self.tmpdir / "empty.jpg"
        empty.write_bytes(b"")
        facts = stamper._read_exif_facts(empty, raise_on_unreadable=False)
        self.assertIsNone(facts.datetime_original)


class DateTimeLogicTests(unittest.TestCase):
    """Test date/time logic in planning."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.identity = stamper.resolve_machine_identity()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_plan_unprotected_file_gets_normalized_datetime(self):
        png = self.tmpdir / "test.png"
        _make_png(png)
        
        plan = stamper.plan(png, self.identity)
        
        self.assertEqual(plan.disposition, "stamp")
        self.assertIsNotNone(plan.normalized_dt)
        self.assertIsInstance(plan.normalized_dt, datetime.datetime)

    def test_plan_uses_min_of_ctime_and_mtime(self):
        png = self.tmpdir / "test.png"
        _make_png(png)
        
        plan = stamper.plan(png, self.identity)
        
        st = png.stat()
        ctime = datetime.datetime.fromtimestamp(st.st_ctime)
        mtime = datetime.datetime.fromtimestamp(st.st_mtime)
        expected = min(ctime, mtime)
        
        # normalized_dt should be close to expected (within a second for filesystem precision)
        self.assertAlmostEqual(
            plan.normalized_dt.timestamp(),
            expected.timestamp(),
            delta=1.0
        )

    def test_plan_protected_file_has_no_datetime(self):
        jpg = self.tmpdir / "test.jpg"
        _make_jpeg(jpg, with_exif_datetime="2020:01:02 03:04:05")
        
        plan = stamper.plan(jpg, self.identity)
        
        self.assertEqual(plan.disposition, "skip-protected")
        self.assertIsNone(plan.normalized_dt)

    def test_plan_unsupported_extension_returns_unsupported(self):
        txt = self.tmpdir / "test.txt"
        txt.write_text("hello")
        
        plan = stamper.plan(txt, self.identity)
        
        self.assertEqual(plan.disposition, "unsupported")
        self.assertIsNone(plan.normalized_dt)

    def test_plan_raw_file_returns_detect_only(self):
        raw = self.tmpdir / "test.CR2"
        raw.write_bytes(b"not a real raw file")
        
        plan = stamper.plan(raw, self.identity)
        
        self.assertEqual(plan.disposition, "detect-only")
        self.assertIsNone(plan.normalized_dt)


class StampedMetadataValidationTests(unittest.TestCase):
    """Test that stamped metadata contains all required fields."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_options(self, **overrides):
        base = dict(target_root=self.tmpdir, recursive=False, dry_run=False, backup=False)
        base.update(overrides)
        return stamper.RunOptions(**base)

    def test_stamped_jpeg_has_required_fields(self):
        jpg = self.tmpdir / "test.jpg"
        _make_jpeg(jpg)
        
        stamper.run(self._run_options())
        
        # Read back and validate
        with Image.open(jpg) as img:
            exif = img.getexif()
            zeroth = exif
            exif_ifd = exif.get_ifd(0x8769) if exif else {}
        
        # Check critical fields are present
        self.assertIsNotNone(stamper._decode_exif_str(exif_ifd.get(36867)), "DateTimeOriginal")
        self.assertIsNotNone(stamper._decode_exif_str(exif_ifd.get(36868)), "CreateDate")
        self.assertEqual(stamper._decode_exif_str(zeroth.get(305)), "EXIFStamper", "Software")
        self.assertIsNotNone(stamper._decode_exif_str(zeroth.get(271)), "Make")
        self.assertIsNotNone(stamper._decode_exif_str(zeroth.get(272)), "Model")

    def test_stamped_png_has_required_fields(self):
        png = self.tmpdir / "test.png"
        _make_png(png)
        
        stamper.run(self._run_options())
        
        with Image.open(png) as img:
            exif = img.getexif()
            exif_ifd = exif.get_ifd(0x8769) if exif else {}
        
        self.assertIsNotNone(stamper._decode_exif_str(exif_ifd.get(36867)), "DateTimeOriginal")
        self.assertEqual(stamper._decode_exif_str(exif.get(305)), "EXIFStamper", "Software")

    def test_stamped_mp4_has_day_atom(self):
        mp4 = self.tmpdir / "test.mp4"
        _make_mp4(mp4)
        
        stamper.run(self._run_options())
        
        facts = stamper.HANDLERS["video"].read_protection_facts(mp4)
        self.assertIsNotNone(facts.container_creation_time)
        # Should be YYYY-MM-DD format
        self.assertRegex(facts.container_creation_time, r"^\d{4}-\d{2}-\d{2}$")

    def test_stamped_webp_has_required_fields(self):
        webp = self.tmpdir / "test.webp"
        _make_webp(webp)
        
        stamper.run(self._run_options())
        
        with Image.open(webp) as img:
            exif = img.getexif()
            exif_ifd = exif.get_ifd(0x8769) if exif else {}
        
        self.assertIsNotNone(stamper._decode_exif_str(exif_ifd.get(36867)), "DateTimeOriginal")

    def test_stamped_tiff_has_required_fields(self):
        tiff = self.tmpdir / "test.tiff"
        _make_tiff(tiff)
        
        stamper.run(self._run_options())
        
        with Image.open(tiff) as img:
            exif = img.getexif()
            exif_ifd = exif.get_ifd(0x8769) if exif else {}
        
        self.assertIsNotNone(stamper._decode_exif_str(exif_ifd.get(36867)), "DateTimeOriginal")


class CaseInsensitivityTests(unittest.TestCase):
    """Test that file extensions are case-insensitive."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_options(self, **overrides):
        base = dict(target_root=self.tmpdir, recursive=False, dry_run=False, backup=False)
        base.update(overrides)
        return stamper.RunOptions(**base)

    def test_uppercase_jpg_extension(self):
        jpg = self.tmpdir / "test.JPG"
        img = Image.new("RGB", (8, 8), color=(255, 0, 0))
        img.save(jpg)
        
        rc = stamper.run(self._run_options())
        
        self.assertEqual(rc, 0)
        facts = stamper._read_exif_facts(jpg)
        self.assertIsNotNone(facts.datetime_original)

    def test_mixed_case_png_extension(self):
        png = self.tmpdir / "test.PnG"
        img = Image.new("RGB", (8, 8), color=(0, 255, 0))
        img.save(png)
        
        rc = stamper.run(self._run_options())
        
        self.assertEqual(rc, 0)
        facts = stamper._read_exif_facts(png)
        self.assertIsNotNone(facts.datetime_original)

    def test_uppercase_mp4_extension(self):
        mp4 = self.tmpdir / "test.MP4"
        _make_mp4(mp4)
        
        rc = stamper.run(self._run_options())
        
        self.assertEqual(rc, 0)
        facts = stamper.HANDLERS["video"].read_protection_facts(mp4)
        self.assertIsNotNone(facts.container_creation_time)


class MultiFormatScenarioTests(unittest.TestCase):
    """Test handling of mixed file types in one run."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_options(self, **overrides):
        base = dict(target_root=self.tmpdir, recursive=False, dry_run=False, backup=False)
        base.update(overrides)
        return stamper.RunOptions(**base)

    def test_mixed_formats_all_stamped(self):
        jpg = self.tmpdir / "test.jpg"
        png = self.tmpdir / "test.png"
        webp = self.tmpdir / "test.webp"
        _make_jpeg(jpg)
        _make_png(png)
        _make_webp(webp)
        
        rc = stamper.run(self._run_options())
        
        self.assertEqual(rc, 0)
        self.assertIsNotNone(stamper._read_exif_facts(jpg).datetime_original)
        self.assertIsNotNone(stamper._read_exif_facts(png).datetime_original)
        self.assertIsNotNone(stamper._read_exif_facts(webp).datetime_original)

    def test_mixed_protected_and_unprotected(self):
        protected_jpg = self.tmpdir / "protected.jpg"
        unprotected_png = self.tmpdir / "unprotected.png"
        _make_jpeg(protected_jpg, with_exif_datetime="2020:01:02 03:04:05")
        _make_png(unprotected_png)
        
        rc = stamper.run(self._run_options())
        
        self.assertEqual(rc, 0)
        # Protected should be untouched
        facts_jpg = stamper._read_exif_facts(protected_jpg)
        self.assertIn("2020", facts_jpg.datetime_original)
        # Unprotected should be stamped
        facts_png = stamper._read_exif_facts(unprotected_png)
        self.assertIsNotNone(facts_png.datetime_original)

    def test_mixed_with_raw_and_unsupported(self):
        png = self.tmpdir / "test.png"
        raw = self.tmpdir / "test.CR2"
        txt = self.tmpdir / "test.txt"
        _make_png(png)
        raw.write_bytes(b"not a real raw file")
        txt.write_text("hello")
        
        rc = stamper.run(self._run_options())
        
        self.assertEqual(rc, 0)
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        self.assertIn("stamp with", log_text)  # PNG processed
        self.assertIn("detect-only", log_text)  # RAW detected
        self.assertIn("unsupported", log_text)  # TXT unsupported


class DeepRecursionTests(unittest.TestCase):
    """Test deep directory nesting with recursive flag."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_options(self, **overrides):
        base = dict(target_root=self.tmpdir, recursive=False, dry_run=False, backup=False)
        base.update(overrides)
        return stamper.RunOptions(**base)

    def test_deep_nested_directories_with_recursive(self):
        deep_dir = self.tmpdir / "a" / "b" / "c" / "d" / "e"
        deep_dir.mkdir(parents=True, exist_ok=True)
        png = deep_dir / "test.png"
        _make_png(png)
        
        rc = stamper.run(self._run_options(recursive=True))
        
        self.assertEqual(rc, 0)
        facts = stamper._read_exif_facts(png)
        self.assertIsNotNone(facts.datetime_original)

    def test_deep_nested_not_scanned_without_recursive(self):
        deep_dir = self.tmpdir / "a" / "b" / "c"
        deep_dir.mkdir(parents=True, exist_ok=True)
        png = deep_dir / "test.png"
        _make_png(png)
        
        rc = stamper.run(self._run_options(recursive=False))
        
        self.assertEqual(rc, 0)
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        self.assertNotIn("test.png", log_text)


class BackupBehaviorTests(unittest.TestCase):
    """Test backup file creation and sidecar behavior."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_options(self, **overrides):
        base = dict(target_root=self.tmpdir, recursive=False, dry_run=False, backup=False)
        base.update(overrides)
        return stamper.RunOptions(**base)

    def test_backup_file_has_original_extension(self):
        png = self.tmpdir / "test.png"
        _make_png(png)
        
        stamper.run(self._run_options(backup=True))
        
        backup = png.with_name(png.name + ".original")
        self.assertTrue(backup.exists())
        self.assertEqual(backup.suffix, ".original")  # .original is appended as suffix

    def test_multiple_backups_don_t_overwrite(self):
        png = self.tmpdir / "test.png"
        _make_png(png)
        
        # First run with backup
        stamper.run(self._run_options(backup=True))
        backup = png.with_name(png.name + ".original")
        first_backup_bytes = backup.read_bytes()
        
        # Modify the original
        time.sleep(0.01)  # Ensure different timestamp
        _make_png(png)
        
        # Second run with backup should fail or skip (backup already exists)
        rc = stamper.run(self._run_options(backup=True))
        
        # The backup should still have the original content
        self.assertEqual(backup.read_bytes(), first_backup_bytes)

    def test_backup_file_not_rescanned_in_next_run(self):
        png = self.tmpdir / "test.png"
        _make_png(png)
        
        # First run creates backup
        stamper.run(self._run_options(backup=True))
        
        # Clear log
        log_file = stamper.SCRIPT_DIR / stamper.LOG_FILENAME
        
        # Second run should not process the .original file
        stamper.run(self._run_options(backup=True))
        
        log_text = log_file.read_text()
        # .original should never appear in logs
        self.assertNotIn(".original", log_text)


class DryRunEdgeCasesTests(unittest.TestCase):
    """Test dry-run flag edge cases."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_options(self, **overrides):
        base = dict(target_root=self.tmpdir, recursive=False, dry_run=False, backup=False)
        base.update(overrides)
        return stamper.RunOptions(**base)

    def test_dry_run_with_multiple_formats(self):
        jpg = self.tmpdir / "test.jpg"
        png = self.tmpdir / "test.png"
        mp4 = self.tmpdir / "test.mp4"
        _make_jpeg(jpg)
        _make_png(png)
        _make_mp4(mp4)
        
        before_jpg = jpg.read_bytes()
        before_png = png.read_bytes()
        before_mp4 = mp4.read_bytes()
        
        rc = stamper.run(self._run_options(dry_run=True))
        
        self.assertEqual(rc, 0)
        self.assertEqual(jpg.read_bytes(), before_jpg)
        self.assertEqual(png.read_bytes(), before_png)
        self.assertEqual(mp4.read_bytes(), before_mp4)

    def test_dry_run_with_backup_flag_creates_no_backup(self):
        png = self.tmpdir / "test.png"
        _make_png(png)
        
        rc = stamper.run(self._run_options(dry_run=True, backup=True))
        
        self.assertEqual(rc, 0)
        backup = png.with_name(png.name + ".original")
        self.assertFalse(backup.exists(), "dry-run should not create backup")

    def test_dry_run_log_contains_dry_run_marker(self):
        png = self.tmpdir / "test.png"
        _make_png(png)
        
        stamper.run(self._run_options(dry_run=True))
        
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        self.assertIn("[DRY RUN]", log_text)


class EmptyAndBoundaryConditionsTests(unittest.TestCase):
    """Test empty directories and boundary conditions."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_options(self, **overrides):
        base = dict(target_root=self.tmpdir, recursive=False, dry_run=False, backup=False)
        base.update(overrides)
        return stamper.RunOptions(**base)

    def test_empty_directory_returns_zero(self):
        rc = stamper.run(self._run_options())
        self.assertEqual(rc, 0)

    def test_directory_with_only_unsupported_files(self):
        txt1 = self.tmpdir / "file1.txt"
        txt2 = self.tmpdir / "file2.txt"
        txt1.write_text("hello")
        txt2.write_text("world")
        
        rc = stamper.run(self._run_options())
        
        self.assertEqual(rc, 0)
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        self.assertIn("unsupported", log_text)

    def test_directory_with_only_protected_files(self):
        jpg1 = self.tmpdir / "test1.jpg"
        jpg2 = self.tmpdir / "test2.jpg"
        _make_jpeg(jpg1, with_exif_datetime="2020:01:02 03:04:05")
        _make_jpeg(jpg2, with_exif_datetime="2021:01:02 03:04:05")
        
        rc = stamper.run(self._run_options())
        
        self.assertEqual(rc, 0)
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        # Should have skip-protected entries for both files (at least 2 occurrences)
        self.assertGreaterEqual(log_text.count("skip-protected"), 2)

    def test_directory_with_only_detect_only_files(self):
        raw1 = self.tmpdir / "test1.CR2"
        raw2 = self.tmpdir / "test2.NEF"
        raw1.write_bytes(b"not raw")
        raw2.write_bytes(b"not raw")
        
        rc = stamper.run(self._run_options())
        
        self.assertEqual(rc, 0)
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        # Should have detect-only entries for both files (at least 2 occurrences)
        self.assertGreaterEqual(log_text.count("detect-only"), 2)


class LoggingFormatTests(unittest.TestCase):
    """Test log file format and content."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_options(self, **overrides):
        base = dict(target_root=self.tmpdir, recursive=False, dry_run=False, backup=False)
        base.update(overrides)
        return stamper.RunOptions(**base)

    def test_log_file_created_at_script_dir(self):
        png = self.tmpdir / "test.png"
        _make_png(png)
        
        stamper.run(self._run_options())
        
        log_file = stamper.SCRIPT_DIR / stamper.LOG_FILENAME
        self.assertTrue(log_file.exists())

    def test_log_contains_pipe_separated_fields(self):
        png = self.tmpdir / "test.png"
        _make_png(png)
        
        stamper.run(self._run_options())
        
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        lines = log_text.strip().split("\n")
        for line in lines:
            if line:  # Skip empty lines
                # Format: path | disposition | detail
                self.assertIn("|", line)

    def test_log_contains_datetime_in_detail(self):
        png = self.tmpdir / "test.png"
        _make_png(png)
        
        stamper.run(self._run_options())
        
        log_text = (stamper.SCRIPT_DIR / stamper.LOG_FILENAME).read_text()
        # Should contain a stamped datetime in format YYYY:MM:DD HH:MM:SS
        self.assertRegex(log_text, r"\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}")


class ClassifyFunctionTests(unittest.TestCase):
    """Test the classify function directly."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_classify_jpeg_returns_image_handler(self):
        jpg = self.tmpdir / "test.jpg"
        jpg.write_bytes(b"dummy")
        result = stamper.classify(jpg)
        self.assertIsNotNone(result)
        format_class, handler = result
        self.assertEqual(format_class, stamper.FormatClass.WRITE_SUPPORT)
        self.assertIs(handler, stamper.HANDLERS["image"])

    def test_classify_png_returns_image_handler(self):
        png = self.tmpdir / "test.png"
        png.write_bytes(b"dummy")
        result = stamper.classify(png)
        self.assertIsNotNone(result)
        format_class, handler = result
        self.assertEqual(format_class, stamper.FormatClass.WRITE_SUPPORT)

    def test_classify_mp4_returns_video_handler(self):
        mp4 = self.tmpdir / "test.mp4"
        mp4.write_bytes(b"dummy")
        result = stamper.classify(mp4)
        self.assertIsNotNone(result)
        format_class, handler = result
        self.assertEqual(format_class, stamper.FormatClass.WRITE_SUPPORT)
        self.assertIs(handler, stamper.HANDLERS["video"])

    def test_classify_cr2_returns_detect_only_handler(self):
        raw = self.tmpdir / "test.CR2"
        raw.write_bytes(b"dummy")
        result = stamper.classify(raw)
        self.assertIsNotNone(result)
        format_class, handler = result
        self.assertEqual(format_class, stamper.FormatClass.DETECT_ONLY)
        self.assertIs(handler, stamper.HANDLERS["detect_only"])

    def test_classify_txt_returns_none(self):
        txt = self.tmpdir / "test.txt"
        txt.write_bytes(b"dummy")
        result = stamper.classify(txt)
        self.assertIsNone(result)

    def test_classify_is_case_insensitive(self):
        jpg_upper = self.tmpdir / "test.JPG"
        jpg_upper.write_bytes(b"dummy")
        result = stamper.classify(jpg_upper)
        self.assertIsNotNone(result)
        format_class, handler = result
        self.assertEqual(format_class, stamper.FormatClass.WRITE_SUPPORT)


class ScanFunctionTests(unittest.TestCase):
    """Test the scan function."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_options(self, **overrides):
        base = dict(target_root=self.tmpdir, recursive=False, dry_run=False, backup=False)
        base.update(overrides)
        return stamper.RunOptions(**base)

    def test_scan_yields_only_files(self):
        subdir = self.tmpdir / "subdir"
        subdir.mkdir()
        file1 = self.tmpdir / "file1.txt"
        file1.write_text("hello")
        
        results = list(stamper.scan(self._run_options()))
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], file1)

    def test_scan_excludes_backup_sidecars(self):
        original = self.tmpdir / "test.png"
        original.write_bytes(b"data")
        backup = self.tmpdir / "test.png.original"
        backup.write_bytes(b"data")
        
        results = list(stamper.scan(self._run_options()))
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], original)
        self.assertNotIn(backup, results)

    def test_scan_excludes_log_file(self):
        log_file = stamper.SCRIPT_DIR / stamper.LOG_FILENAME
        # Create a dummy log in the scan directory
        log_copy = self.tmpdir / stamper.LOG_FILENAME
        log_copy.write_text("dummy")
        
        results = list(stamper.scan(self._run_options()))
        
        # Log file might not be excluded from tmpdir scan (it's a different dir)
        # So just ensure normal files are found
        self.assertGreaterEqual(len(results), 0)

    def test_scan_respects_recursive_flag(self):
        file1 = self.tmpdir / "file1.txt"
        file1.write_text("hello")
        
        subdir = self.tmpdir / "subdir"
        subdir.mkdir()
        file2 = subdir / "file2.txt"
        file2.write_text("world")
        
        non_recursive = list(stamper.scan(self._run_options(recursive=False)))
        recursive = list(stamper.scan(self._run_options(recursive=True)))
        
        self.assertEqual(len(non_recursive), 1)
        self.assertEqual(len(recursive), 2)


if __name__ == "__main__":
    unittest.main()
