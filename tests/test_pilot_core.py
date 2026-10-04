import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"NINSPilot"))
import pilot_core as core


class PilotCoreTests(unittest.TestCase):
    def test_rotated_rectangle_has_true_feret_and_area(self):
        points = np.array([[-2,-1,0], [2,-1,0], [2,1,0], [-2,1,0.]])
        rotation = np.array([[0,0,1], [1,0,0], [0,1,0.]])
        r = core.rim_geometry(points@rotation.T+[20,-10,8])
        self.assertAlmostEqual(r["minimum_feret_mm"], 2)
        self.assertAlmostEqual(r["maximum_feret_mm"], np.sqrt(20))
        self.assertAlmostEqual(r["projected_area_mm2"], 8)
        self.assertLess(r["plane_rms_mm"], 1e-10)

    def test_concave_area_is_not_hull_area(self):
        r = core.rim_geometry([[0,0,0], [4,0,0], [4,4,0], [2,2,0], [0,4,0]])
        self.assertAlmostEqual(r["projected_area_mm2"], 12)
        self.assertAlmostEqual(r["minimum_feret_mm"], 4)

    def test_nonplanarity_reported(self):
        r = core.rim_geometry([[-2,-2,.2], [2,-2,-.2], [2,2,.2], [-2,2,-.2]])
        self.assertAlmostEqual(r["plane_rms_mm"], .2)

    def test_invalid_rims_refused(self):
        for points in ([[0,0,0],[1,1,0],[2,2,0]], [[0,0,0],[2,2,0],[0,2,0],[2,0,0]], [[0,0,0],[1,0,0],[1,1,float("nan")]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                core.rim_geometry(points)

    def test_one_assessable_side_and_refusal_reason(self):
        sides = dict(R=dict(status="assessable"), L=dict(status="unassessable", reason="Rim obscured"))
        self.assertEqual(core.side_decisions(sides), ["R"])
        sides["L"]["reason"] = ""
        with self.assertRaises(ValueError):
            core.side_decisions(sides)

    def test_thresholds_respect_roi_and_exclude_padding(self):
        a = np.full((6,7,8), -1000.)
        a[2,3,4] = 900
        a[3,3,4] = -3024
        air, bone = core.proposal_masks(a, [[2,5],[2,4],[1,4]])
        self.assertEqual(air.sum(), 46)
        self.assertEqual(bone.sum(), 1)
        self.assertEqual(air[3,3,4], 0)
        self.assertFalse(air[0].any())

    def test_invalid_thresholds_and_roi_refused(self):
        for bounds, air, bone in [([[0,9],[0,2],[0,2]], -300,300), ([[0,2]]*3, 400,300), ([[0,2]]*3, -300,np.nan)]:
            with self.assertRaises(ValueError):
                core.proposal_masks(np.zeros((3,3,3)), bounds, air, bone)

    def test_fingerprint_detects_data_and_shape(self):
        a = np.zeros((2,3,4), np.int16)
        before = core.array_digest(a)
        self.assertNotEqual(before, core.array_digest(a.reshape(4,3,2)))
        a[0,0,0] = 1
        self.assertNotEqual(before, core.array_digest(a))

    def test_three_distinct_synthetic_fixtures(self):
        hashes = {core.array_digest(core.synthetic_case(k)) for k in ("straight", "narrow", "artifact")}
        self.assertEqual(len(hashes), 3)


if __name__ == "__main__":
    unittest.main()
