"""Analytical geometry tests; VTK adapter tests run when VTK is available."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

PATH = Path(__file__).resolve().parents[1] / "(C) NINS_Contact_Geometry_0912.py"
spec = importlib.util.spec_from_file_location("contact_geometry", PATH)
cg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cg)


def graph(fn):
    def sample(origins, n, half):
        assert np.allclose(n, [0, 0, 1])
        out = origins.copy()
        out[:, 2] = fn(origins[:, 0], origins[:, 1])
        return out
    return sample


def evaluate(surface, **kwargs):
    args = dict(center=[0, 0, 0], normal=[0, 0, 1], axis=[1, 0, 0],
                target=[0, 0, -4], width=4., length=6., angle_deg=0., tolerance_mm=.1)
    args.update(kwargs)
    return cg.evaluate(surface, **args)


class AnalyticalGeometry(unittest.TestCase):
    def test_plane_area_and_apposition(self):
        c = evaluate(graph(lambda x, y: 0*x))
        self.assertAlmostEqual(c['flat_gap_max_mm'], 0)
        self.assertAlmostEqual(c['flat_apposition_fraction'], 1)
        self.assertAlmostEqual(c['conforming_surface_area_mm2'], c['sampled_projected_area_mm2'])
        self.assertLess(abs(c['sampled_projected_area_mm2'] / (6*np.pi) - 1), .002)

    def test_cylinder_sagitta_and_orientation(self):
        radius = 10.
        cylinder = graph(lambda x, y: np.sqrt(radius**2 - x**2) - radius)
        a = evaluate(cylinder)
        b = evaluate(cylinder, angle_deg=90.)
        self.assertAlmostEqual(a['flat_gap_max_mm'], radius - np.sqrt(radius**2 - 2**2), places=9)
        self.assertAlmostEqual(b['flat_gap_max_mm'], radius - np.sqrt(radius**2 - 3**2), places=9)
        self.assertGreater(b['flat_gap_max_mm'], a['flat_gap_max_mm'])
        self.assertGreater(a['conforming_surface_area_mm2'], a['sampled_projected_area_mm2'])

    def test_quadratic_curvature_and_relief(self):
        c = evaluate(graph(lambda x, y: .05*x*x + .1*y*y))
        np.testing.assert_allclose(c['quadratic_principal_curvatures_per_mm'], [.1, .2], atol=1e-10)
        self.assertLess(c['quadratic_fit_rms_mm'], 1e-10)
        self.assertAlmostEqual(c['surface_relief_mm'], .9, places=9)

    def test_area_weights_not_vertex_count(self):
        c = evaluate(graph(lambda x, y: -.05*(x*x + y*y)), width=4, length=4,
                     tolerance_mm=.1, rings=40, sectors=128)
        # Contact disk radius sqrt(2), electrode radius 2 => area fraction 1/2.
        self.assertLess(abs(c['flat_apposition_fraction'] - .5), .03)

    def test_boundary_hole_refuses_entire_candidate(self):
        def missing(origins, n, half):
            out = origins.copy()
            out[origins[:, 0] > 1.5] = np.nan
            return out
        c = evaluate(missing)
        self.assertEqual(c['status'], 'rejected')
        self.assertNotIn('_flat_points', c)
        self.assertLess(c['sampled_coverage_fraction'], 1)

    def test_rigid_transform_invariance(self):
        base = graph(lambda x, y: .02*x*x)
        a = evaluate(base)
        rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0.]])
        translation = np.array([11, -12, 23])
        def transformed(origins, n, half):
            return base((origins-translation)@rotation, rotation.T@n, half)@rotation.T+translation
        b = evaluate(transformed, center=translation, normal=rotation@[0, 0, 1],
                     axis=rotation@[1, 0, 0], target=rotation@[0, 0, -4]+translation)
        for key in ['flat_gap_max_mm', 'conforming_surface_area_mm2', 'flat_target_distance_mean_mm']:
            self.assertAlmostEqual(a[key], b[key], places=10)

    def test_invalid_inputs(self):
        for kw in [dict(width=-1), dict(tolerance_mm=-1), dict(normal=[0, 0, 0]),
                   dict(axis=[0, 0, 1]), dict(target=[np.nan, 0, 0]), dict(angle_deg=np.nan)]:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                evaluate(graph(lambda x, y: 0*x), **kw)

    def test_pareto_removes_only_dominated(self):
        rows = []
        for name, area, gap, distance in [('a', 5, 1, 3), ('b', 4, 2, 4), ('c', 6, 2, 3)]:
            rows.append(dict(candidate_id=name, status='evaluated',
                             flat_apposition_projected_area_mm2=area,
                             flat_gap_max_mm=gap, flat_target_distance_mean_mm=distance))
        self.assertEqual(cg.pareto_ids(rows, 'flat'), ['a', 'c'])

    def test_search_and_export(self):
        report = cg.search(graph(lambda x, y: 0*x), [0, 0, 0], [0, 0, 1], [1, 0, 0],
                           [0, 0, -4], [(2, 4), (4, 6)], [0, 90], [(0, 0), (1, 0)], .1)
        self.assertEqual(len(report['candidates']), 8)
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(cg.export_report(report, tmp))
            second = Path(cg.export_report(report, tmp))
            self.assertNotEqual(first, second)
            clean = json.loads((first/'report.json').read_text())
            self.assertNotIn('_faces', clean['candidates'][0])
            self.assertEqual(len(list(first.glob('*.obj'))), 16)


try:
    import vtk
except ImportError:
    vtk = None


@unittest.skipIf(vtk is None, 'VTK not installed in this Python')
class VTKAdapter(unittest.TestCase):
    def patch(self, heights=(0.,)):
        append = vtk.vtkAppendPolyData()
        for z in heights:
            plane = vtk.vtkPlaneSource()
            plane.SetOrigin(-5, -5, z)
            plane.SetPoint1(5, -5, z)
            plane.SetPoint2(-5, 5, z)
            plane.SetXResolution(10)
            plane.SetYResolution(10)
            plane.Update()
            append.AddInputData(plane.GetOutput())
        append.Update()
        return cg.VTKSurface(append.GetOutput())

    def test_open_plane_shared_edges_and_outside(self):
        surface = self.patch()
        hits = surface(np.array([[0., 0., 0.], [1., 1., 0.], [6., 0., 0.]]), np.array([0., 0., 1.]), 3)
        np.testing.assert_allclose(hits[:2], [[0, 0, 0], [1, 1, 0]])
        self.assertTrue(np.isnan(hits[2]).all())
        self.assertEqual(evaluate(surface)['status'], 'evaluated')

    def test_multiple_layers_refused(self):
        c = evaluate(self.patch((0., 1.)))
        self.assertEqual(c['status'], 'rejected')
        self.assertEqual(c['sampled_coverage_fraction'], 0)

    def test_air_reference_controls_orientation(self):
        surface = self.patch()
        p, n, axis, distance = surface.closest_frame([0, 0, .5], [0, 0, 3])
        np.testing.assert_allclose(n, [0, 0, 1])
        self.assertAlmostEqual(distance, .5)
        _, reverse, _, _ = surface.closest_frame([0, 0, .5], [0, 0, -3])
        np.testing.assert_allclose(reverse, [0, 0, -1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
