
import unittest
import numpy as np
import os
import pkg_resources

from openmdao.api import Group, Problem

from fusedwind.turbine.structure import read_bladestructure, \
                                        interpolate_bladestructure, \
                                        SplinedBladeStructure, \
                                        SplinedBladeStructureParam2, \
                                        BladeStructureProperties
from fusedwind.turbine.geometry import read_blade_planform,\
                                       redistribute_planform,\
                                       PGLLoftedBladeSurface,\
                                       PGLRedistributedPlanform

PATH = pkg_resources.resource_filename('fusedwind', 'turbine/test')


def configure_with_surface():

    nsec = 20
    st3dn = read_bladestructure(os.path.join(PATH, 'data_version_2/Param2_10MW'))
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

    spl = p.root.add('st_splines', SplinedBladeStructureParam2(st3dn, (200, nsec, 3)), promotes=['*'])

    p.root.add('st_props', BladeStructureProperties((200, nsec, 3), st3dn,
                                                    [4, 7, 10, 13]),
                                                    promotes=['*'])

    spl.add_spline(('cap_width_ss',
                    'cap_width_ps'),
                    np.array([0., 0.52631578947368418, 1.]), spline_type='linear')
    p.setup()
    for k, v in list(pf.items()):
        if k+'_st' in list(p.root.blade_surf.params.keys()):
            p[k+'_st'] = v
    return p


class TestSplinedBladeStructure(unittest.TestCase):

    def test_workflow(self):

        cap_width = np.linspace(1.1, 0.8, 20) / 86.366
        cap_width[-1] = 0.25 / 86.366
        le_width = np.linspace(1.6, 0.8, 20) / 86.366
        le_width[-2] = 0.6 / 86.366
        le_width[-1] = 0.1 / 86.366

        DPs_i4 = np.array([-1.        , -0.99      , -0.8930699 , -0.73888813, -0.4887113 ,
       -0.4714117 , -0.36119044, -0.34242498, -0.10098954,  0.10062569,
        0.4135107 ,  0.43302372,  0.53967981,  0.55926998,  0.77011888,
        0.89345515,  0.99      ,  1.        ])

        DPs_i17 = np.array([-1.        , -0.99      , -0.74739708, -0.74739708, -0.5414817 ,
       -0.49992362, -0.24561462, -0.20325066, -0.17981903,  0.17913871,
        0.20121045,  0.2436819 ,  0.49689473,  0.53816183,  0.74835276,
        0.74835276,  0.99      ,  1.        ])

        p = configure_with_surface()
        p.run()
        self.assertEqual(np.testing.assert_array_almost_equal(p['r04_width'] +
                                                              p['r05_width'] +
                                                              p['r06_width'],
                                                              cap_width,
                                                              decimal=6), None)
        self.assertEqual(np.testing.assert_array_almost_equal(p['r08_width'],
                                                              le_width,
                                                              decimal=6), None)


        self.assertEqual(np.testing.assert_array_almost_equal(
                                    p.root.st_splines.compute_dps._DPs.DPs[4,:],
                                                              DPs_i4,
                                                              decimal=1), None)
        self.assertEqual(np.testing.assert_array_almost_equal(
                                    p.root.st_splines.compute_dps._DPs.DPs[17,:],
                                                              DPs_i17,
                                                              decimal=1), None)

    def test_splines(self):

        # linear pertubation of cap width
        cap_width = np.linspace(1.1, 0.8, 20) / 86.366
        cap_width[-1] = 0.25 / 86.366
        dd = np.linspace(0, 0.005, 11)
        cap_width[:11] += dd
        dd = np.linspace(0.005, 0, 10)
        cap_width[11:] += dd[1:]

        p = configure_with_surface()

        p['cap_width_ss_C'][1] = 0.005

        p.run()

        self.assertEqual(np.testing.assert_array_almost_equal(p['r04_width'] +
                                                              p['r05_width'] +
                                                              p['r06_width'],
                                                              cap_width,
                                                              decimal=6), None)


if __name__ == '__main__':

    unittest.main()
    # p = configure_with_surface()
    # p.run()
