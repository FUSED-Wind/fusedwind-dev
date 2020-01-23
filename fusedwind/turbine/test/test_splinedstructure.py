
import unittest
import numpy as np
import os
import pkg_resources

from openmdao.api import Group, Problem

from fusedwind.turbine.structure import read_bladestructure, \
                                        interpolate_bladestructure, \
                                        SplinedBladeStructure, \
                                        BladeStructureProperties
from fusedwind.turbine.geometry import read_blade_planform,\
                                       redistribute_planform,\
                                       PGLLoftedBladeSurface,\
                                       PGLRedistributedPlanform

PATH = pkg_resources.resource_filename('fusedwind', 'turbine/test')

def configure():

    st3d = read_bladestructure(os.path.join(PATH, 'data/DTU10MW'))
    st3dn = interpolate_bladestructure(st3d, np.linspace(0, 1, 8))

    p = Problem(root=Group())
    spl = p.root.add('st_splines', SplinedBladeStructure(st3dn), promotes=['*'])
    spl.add_spline('DP08', np.linspace(0, 1, 4), spline_type='bezier')
    spl.add_spline('DP09', np.linspace(0, 1, 4), spline_type='bezier')
    spl.add_spline(('DP04', 'DP05'), np.linspace(0, 1, 4), spline_type='bezier')
    spl.add_spline(('r04uniax00T', 'r04uniax01T'), np.linspace(0, 1, 4), spline_type='bezier')
    spl.add_spline('w02biax00T', np.linspace(0, 1, 4), spline_type='bezier')
    p.setup()
    return p

def configure_with_surface():

    nsec = 8
    st3d = read_bladestructure(os.path.join(PATH, 'data/DTU10MW'))
    st3dn = interpolate_bladestructure(st3d, np.linspace(0, 1, nsec))

    p = Problem(root=Group())

    pf = read_blade_planform(os.path.join(PATH, 'data/DTU_10MW_RWT_blade_axis_prebend.dat'))
    s_new = np.linspace(0, 1, nsec)
    pf = redistribute_planform(pf, s=s_new)

    cfg = {}
    cfg['redistribute_flag'] = False
    cfg['blend_var'] = np.array([0.241, 0.301, 0.36, 1.0])
    afs = []
    for f in [os.path.join(PATH, 'data/ffaw3241.dat'),
              os.path.join(PATH, 'data/ffaw3301.dat'),
              os.path.join(PATH, 'data/ffaw3360.dat'),
              os.path.join(PATH, 'data/cylinder.dat')]:

        afs.append(np.loadtxt(f))
    cfg['base_airfoils'] = afs
    d = PGLLoftedBladeSurface(cfg, size_in=nsec, size_out=(200, nsec, 3), suffix='_st')
    r = p.root.add('blade_surf', d, promotes=['*'])

    spl = p.root.add('st_splines', SplinedBladeStructure(st3dn), promotes=['*'])
    spl.add_spline('DP04', np.linspace(0, 1, 4), spline_type='bezier')
    spl.add_spline(('r04uniax00T', 'r04uniax01T'), np.linspace(0, 1, 4), spline_type='bezier')
    spl.add_spline('w02biax00T', np.linspace(0, 1, 4), spline_type='bezier')
    p.root.add('st_props', BladeStructureProperties((200, nsec, 3), st3dn, [4,5,8,9]), promotes=['*'])

    p.setup()
    for k, v in list(pf.items()):
        if k+'_st' in list(p.root.blade_surf.params.keys()):
            p.root.blade_surf.params[k+'_st'] = v

    return p


class TestSplinedBladeStructure(unittest.TestCase):

    def test_splines(self):

        r04uniax= np.array([ 0.008     ,  0.03097776,  0.04077973,  0.04464869,  0.04330389,
                             0.03712204,  0.02271313,  0.0015    ])
        w02biax = np.array([ 0.0026    ,  0.00334698,  0.00596805,  0.00784869,  0.00869437,
                             0.00836037,  0.00643492,  0.0013    ])
        DP04 = np.array([-0.49644126, -0.48696236, -0.34930237, -0.31670771, -0.33028392,
                         -0.35351936, -0.3801164 , -0.3808905 ])
        DP05 = np.array([-0.3638539 , -0.34034888, -0.21823594, -0.1873391 , -0.19636527,
                         -0.21358559, -0.23647097, -0.15612563])
        p = configure()
        p['r04uniax00T_C'][2] = 0.01
        p['w02biax00T_C'][2] = 0.01
        p['DP04_C'][1] = 0.1
        p.run()

        self.assertEqual(np.testing.assert_array_almost_equal(p['r04uniax00T'], r04uniax, decimal=4), None)
        self.assertEqual(np.testing.assert_array_almost_equal(p['r04uniax01T'], r04uniax, decimal=4), None)
        self.assertEqual(np.testing.assert_array_almost_equal(p['w02biax00T'], w02biax, decimal=4), None)
        self.assertEqual(np.testing.assert_array_almost_equal(p['DP04'], DP04, decimal=4), None)

    def test_props(self):

        r04_thickness = np.array([ 0.032     ,  0.06272709,  0.07806093,  0.083     ,  0.07821129,
                                   0.06549772,  0.03912888,  0.013     ])
        r04_width = np.array([ 0.01264879,  0.01255458,  0.01049935,  0.00894517,  0.00743887,
                               0.00596788,  0.00448234,  0.00164483])
        pacc_ss = np.array([[ -4.72540562e-06,   3.05060896e-02],
                           [ -5.76655195e-04,   2.24110499e-02],
                           [  4.98821153e-03,   1.01445966e-02],
                           [  4.91489605e-03,   3.40209671e-03],
                           [  3.85570137e-03,  -3.25207627e-03],
                           [  2.84071207e-03,  -1.14796226e-02],
                           [  1.80044657e-03,  -2.22351645e-02],
                           [  7.60925288e-04,  -3.77607698e-02]])
        web_angle02 = np.array([ 0.00848498, -0.15833175, -0.04299566, -0.05821014, -0.06484249,
        0.02652572,  0.01361219, -0.00427642])
        p = configure_with_surface()
        p.run()

        self.assertEqual(np.testing.assert_array_almost_equal(p['r04_width'], r04_width, decimal=1), None)
        self.assertEqual(np.testing.assert_array_almost_equal(p['r04_thickness'], r04_thickness, decimal=1), None)
        self.assertEqual(np.testing.assert_array_almost_equal(p['pacc_ss'], pacc_ss, decimal=1), None)
        self.assertEqual(np.testing.assert_array_almost_equal(p['web_angle02'], web_angle02, decimal=0), None)

if __name__ == '__main__':

    unittest.main()
    # p = configure()
    # p = configure_with_surface()
    # p.run()
    #p['r04uniaxT_C'][2] = 0.01
    #p['w02biaxT_C'][2] = 0.01
    #p['DP04_C'][1] = 0.1
    #st3d = read_bladestructure('data/DTU10MW')
    #st3dn = interpolate_bladestructure(st3d, np.linspace(0, 1, 8))
    # p.run()
