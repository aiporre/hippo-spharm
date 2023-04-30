from unittest import TestCase
from hippospharm.surface import Ellipsoid

class TestSphereHarmonics(TestCase):
    def test_ellipsoid(self):
        # create an ellipsiod
        ellipsoid = Ellipsoid(a=1, b=1, c=3, N=20)
        # assert grid is 20x20
        self.assertEqual(ellipsoid.Phi.shape, (20, 20))
        self.assertEqual(ellipsoid.Theta.shape, (20, 20))
        self.assertEqual(ellipsoid.R.shape, (20, 20))
        # assert grid is 400x1
        self.assertEqual(ellipsoid.phi_vect.shape, (400,))
        self.assertEqual(ellipsoid.theta_vect.shape, (400,))
        self.assertEqual(ellipsoid.r_vect.shape, (400,))
        # assert compute harmonics


