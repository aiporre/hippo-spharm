from unittest import TestCase

import numpy as np

from hippospharm.spharm import SphereHarmonics
from hippospharm.surface import Ellipsoid, Surface
import os

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
    def test_compute_harmonis_ellipsoid(self):
        ellipsoid = Ellipsoid(a=1, b=1, c=3, N=20)
        # compute harmonics
        harmonics = ellipsoid.get_harmonics()
        # plot harmonics
        print(harmonics)
        harmonics.plot_spectrum()
    def test_harmonics_in_csv_exist(self):
        ellipsoid = Ellipsoid(a=1, b=1, c=3, N=20)
        # compute harmonics
        harmonics = ellipsoid.get_harmonics()
        # save harmonics to csv
        harmonics.save_to_csv('test.csv','./')
        # assert csv exists
        self.assertTrue(os.path.exists('test.csv'))
        # remove csv
        os.remove('test.csv')
    def test_constant_surface(self):
        N = 100
        data = np.ones([N,N])
        surface = Surface(grid=data)
        harmonics = surface.get_harmonics(normalization_method='zero')
        harmonics.plot_spectrum()
        harmonics.plot_spectrum2()
        self.assertEqual(tuple(harmonics.harmonics.shape), (2,N//2,N//2))
        # test max harmonic is one
        self.assertAlmostEqual(harmonics.harmonics.max(), 1, 90)


