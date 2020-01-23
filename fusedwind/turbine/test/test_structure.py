
import numpy as np
import unittest

from fusedwind.turbine.structure import write_bladestructure,\
    read_bladestructure
import os
import shutil

st3d_desired = {}
st3d_desired['web_def'] = np.array([[-1, 0], [2, -3], [4, -5], [5, -6]])
st3d_desired['bond_def'] = np.array([[0, 1, -2, -1]])
st3d_desired['cap_width_ps'] = np.linspace(1.1, 0.8, 20) / 86.366
st3d_desired['cap_width_ps'][-1] = 0.25 / 86.366
st3d_desired['cap_DPs'] = [4, 7, 10, 13]
st3d_desired['le_DPs'] = [8, 9]
st3d_desired['te_DPs'] = [2, 15]


class StructureTests(unittest.TestCase):
    ''' This class contains the unit tests for
        :mod:`fusedwind.turbine.structure`.
    '''
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.data_version_0 = 'data'
        self.data_version_1 = 'data_version_1'
        self.data_version_2 = 'data_version_2'
        self.blade = 'DTU10MW'
        self.test_dir = 'test_dir'

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    def test_read_bladestructure_version_0(self):
        st3d_actual = read_bladestructure(os.path.join(self.data_version_0, self.blade))
        self.assertEqual(np.testing.assert_array_equal(
                         st3d_actual['web_def'],
                         st3d_desired['web_def']), None)

    def test_read_bladestructure_version_1(self):
        st3d_actual = read_bladestructure(os.path.join(self.data_version_1, self.blade))
        self.assertEqual(np.testing.assert_array_equal(
                         st3d_actual['web_def'],
                         st3d_desired['web_def']), None)

    def test_read_write_read_bladestructure_version_0(self):
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        st3d = read_bladestructure(os.path.join(self.data_version_0, self.blade))
        # convert to version 1
        st3d['version'] = 1
        write_bladestructure(st3d, os.path.join(self.test_dir, 'test'))
        st3dn = read_bladestructure(os.path.join(self.test_dir, 'test'))
        self.assertEqual(np.testing.assert_array_equal(
                         st3dn['web_def'],
                         st3d_desired['web_def']), None)
        shutil.rmtree(self.test_dir)

    def test_read_write_read_bladestructure_version_1(self):
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        st3d = read_bladestructure(os.path.join(self.data_version_1, self.blade))
        write_bladestructure(st3d, os.path.join(self.test_dir, 'test'))
        st3dn = read_bladestructure(os.path.join(self.test_dir, 'test'))
        self.assertEqual(np.testing.assert_array_equal(
                         st3dn['web_def'],
                         st3d_desired['web_def']), None)
        shutil.rmtree(self.test_dir)

    def test_read_write_read_bladestructure_version_2(self):
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        st3d = read_bladestructure(os.path.join(self.data_version_2, self.blade))
        write_bladestructure(st3d, os.path.join(self.test_dir, 'test'))
        st3dn = read_bladestructure(os.path.join(self.test_dir, 'test'))
        self.assertEqual(np.testing.assert_array_equal(
                         st3dn['web_def'],
                         st3d_desired['web_def']), None)
        self.assertEqual(np.testing.assert_array_equal(
                         st3dn['bond_def'],
                         st3d_desired['bond_def']), None)
        shutil.rmtree(self.test_dir)

    def test_version_2_geo3d(self):

        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        st3d = read_bladestructure(os.path.join(self.data_version_2, 'Param2_10MW'))
        write_bladestructure(st3d, os.path.join(self.test_dir, 'test'))
        st3dn = read_bladestructure(os.path.join(self.test_dir, 'test'))
        self.assertEqual(np.testing.assert_array_equal(
                         st3dn['cap_DPs'],
                         st3d_desired['cap_DPs']), None)
        self.assertEqual(np.testing.assert_array_equal(
                         st3dn['le_DPs'],
                         st3d_desired['le_DPs']), None)
        self.assertEqual(np.testing.assert_array_equal(
                         st3dn['te_DPs'],
                         st3d_desired['te_DPs']), None)
        self.assertEqual(np.testing.assert_array_equal(
                         st3dn['cap_width_ps'],
                         st3d_desired['cap_width_ps']), None)
        shutil.rmtree(self.test_dir)

if __name__ == '__main__':
    unittest.main()
    # data_version_0 = 'data'
    # data_version_1 = 'data_version_1'
    # data_version_2 = 'data_version_2'
    # blade = 'DTU10MW'
    # test_dir = 'test_dir'
    # st3d = read_bladestructure(os.path.join(data_version_2, blade))
    # write_bladestructure(st3d, os.path.join(test_dir, 'test'))
    # st3dn = read_bladestructure(os.path.join(test_dir, 'test'))
