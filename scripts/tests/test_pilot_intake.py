"""Data integrity and review-gate checks; no Slicer application required."""
import io
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

SYSTEM = Path(__file__).resolve().parent.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, SYSTEM / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


intake = load("NINS_Pilot_Intake")
review = load("NINS_Pilot_Review")
HEADER = (b"NRRD0005\ntype: short\ndimension: 3\nspace: left-posterior-superior\n"
          b"sizes: 2 2 2\nspace directions: (0.5,0,0) (0,0.5,0) (0,0,0.5)\n"
          b"space origin: (0,0,0)\nendian: little\nencoding: raw\n"
          b"PatientName:=DO_NOT_EXPORT\n# confidential comment\n\n")
PAYLOAD = bytes(range(16))


class IntakeTests(unittest.TestCase):
    def test_full_integrity_and_metadata_minimization(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "source.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("private-name.nrrd", HEADER + PAYLOAD)
            result = intake.inventory([archive], verify=True)
            row = result["volumes"][0]
            self.assertEqual(row["payload_status"], "verified")
            self.assertEqual(row["payload_sha256"], hashlib.sha256(PAYLOAD).hexdigest())
            self.assertNotIn("DO_NOT_EXPORT", json.dumps(result))
            self.assertNotIn("private-name", json.dumps(result))
            out = Path(temp) / "staged.nrrd"
            intake.stage(archive, 0, out, row["payload_sha256"])
            with out.open("rb") as f:
                intake.read_header(f)
                self.assertEqual(f.read(), PAYLOAD)
            self.assertNotIn(b"confidential", out.read_bytes())
            with self.assertRaises(FileExistsError):
                intake.stage(archive, 0, out, row["payload_sha256"])

    def test_short_payload_is_not_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "source.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("source.nrrd", HEADER + PAYLOAD[:-2])
            row = intake.inventory([archive], True)["volumes"][0]
            self.assertEqual(row["payload_status"], "size_mismatch")

    def test_nonfinite_geometry_rejected(self):
        fields = intake.read_header(io.BytesIO(HEADER))
        fields["space directions"] = "(nan,0,0) (0,0.5,0) (0,0,0.5)"
        with self.assertRaises(ValueError):
            intake.geometry(fields)

    def test_thick_and_single_slice_not_primary(self):
        fields = intake.read_header(io.BytesIO(HEADER))
        fields["space directions"] = "(0.5,0,0) (0,0.5,0) (0,0,5)"
        self.assertEqual(intake.geometry(fields)["sampling_category"], "thick_sampling_not_primary")
        fields["sizes"] = "512 512 1"
        self.assertEqual(intake.geometry(fields)["sampling_category"], "single_slice_not_eligible")

    def test_external_data_not_staged(self):
        fields = intake.read_header(io.BytesIO(HEADER))
        fields["data file"] = "external.raw"
        with self.assertRaises(ValueError):
            intake.validate_raw(fields)

    def test_unreadable_archive_is_a_recorded_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "broken.zip"
            archive.write_bytes(b"not a zip")
            result = intake.inventory([archive], True)
            self.assertEqual(result["archives"][0]["status"], "unreadable")
            self.assertEqual(result["patients_confirmed"], 0)

    def test_missing_review_never_passes(self):
        volume = dict(spacing_mm=[.5, .5, .5], dimensions_ijk=[20, 20, 20])
        points = dict(SPF_R=[[1, 2, 3]], SPF_L=[[4, 5, 6]])
        self.assertEqual(review.readiness(volume, {}, points)["status"], "review_incomplete")
        verified = dict.fromkeys(review.REVIEW_FIELDS, True)
        self.assertEqual(review.readiness(volume, verified, points)["status"], "ready_for_assisted_run")
        self.assertEqual(review.readiness(volume, verified, {})["status"], "review_incomplete")
        volume["spacing_mm"][-1] = 1.0
        self.assertEqual(review.readiness(volume, verified, points)["status"], "review_incomplete")


if __name__ == "__main__":
    unittest.main()
