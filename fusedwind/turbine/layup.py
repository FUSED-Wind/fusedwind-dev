import numpy as np
from collections import OrderedDict
import pickle


class Material(object):
    ''' Defines and handles material input data such as properties, resistances,
    and safety factors.

    Properties (Values should be set as mean values from measurements in SI units.)
    ----------
    :param E1: Young's modulus parallel (||) to fiber direction
    :param E2: Young's modulus perpendicular (_|_) to fiber direction (in lamina plane)
    :param E3: Young's modulus perpendicular (_|_) to fiber direction (out of lamina plane)
    :param nu12: major Poisson's ratio between fiber direction and perpendicular to fiber
        direction (in lamina plane)
    :param nu13: major Poisson's ratio between fiber direction and perpendicular to fiber
        direction (out of lamina plane)
    :param nu23: major Poisson's ratio between both in and out of lamina plane fiber
        directions
    :param nu21: minor Poisson's ratio between perpendicular direction to fiber and fiber
        direction (in lamina plane)
    :param nu31: minor Poisson's ratio between perpendicular direction to fiber and fiber
        direction (out of lamina plane)
    :param nu32: minor Poisson's ratio between both out of and in lamina plane fiber
        directions

    .. note:: 1st index = loading, 2nd index = contraction
              That is for a uniaxial layer nu12 is the larger (major) and nu21 is the
              smaller (minor) value.

    :param G12: Shear Modules (in lamina plane)
    :param G13: Shear Modules parallel to fiber direction and out of lamina plane
    :param G23: Shear Modules perpendicular to fiber direction and out of lamina plane
    :param rho: Density 
    :param failcrit: Failure criterion to be used for this material ('maximum_strain', 'maximum_stress', 'tsai_wu')
    :type failcrit: string 


    Resistances:
    -----------
    :param s11_t: allowable tensile stress parallel to fiber direction
    :param s22_t: allowable tensile stress perpendicular to fiber direction (in lamina plane)
    :param s33_t: allowable tensile stress perpendicular to fiber direction (out-of lamina plane)
    :param s11_c: allowable compressive stress parallel to fiber direction
    :param s22_c: allowable compressive stress perpendicular to fiber direction (in lamina plane)
    :param s33_c: allowable compressive stress perpendicular to fiber direction (out-of lamina plane)
    :param t12: allowable shear stress in lamina plane
    :param t13: allowable shear stress out of lamina plane, parallel to fiber direction
    :param t23: allowable shear stress out of lamina plane, perpendicular to fiber direction 
    :param e11_t: allowable tensile strain parallel to fiber direction
    :param e22_t: allowable tensile strain perpendicular to fiber direction (in lamina plane)
    :param e33_t: allowable tensile strain perpendicular to fiber direction (out-of lamina plane)
    :param e11_c: allowable compressive strain parallel to fiber direction
    :param e22_c: allowable compressive strain perpendicular to fiber direction (in lamina plane)
    :param e33_c: allowable compressive strain perpendicular to fiber direction (out-of lamina plane)
    :param g12: allowable shear strain in lamina plane
    :param g13: allowable shear strain out of lamina plane, parallel to fiber direction
    :param g23: allowable shear strain out of lamina plane, perpendicular to fiber direction 

    Safety Factors according to GL2010 scheme
    -----------------------------------------
    :param gM0: general material safety factor
    :param C1a: safety factor for influence of ageing
    :param C2a: safety factor for temperature effects
    :param C3a: safety factor for the manufacturing process
    :param C4a: safety factor for the effect of post-curing
    '''

    def __init__(self):
        self.E1 = None
        self.E2 = None
        self.E3 = None
        self.nu12 = None
        self.nu13 = None
        self.nu23 = None
        self.nu21 = None
        self.nu31 = None
        self.nu32 = None
        self.G12 = None
        self.G13 = None
        self.G23 = None
        self.rho = None
        self.s11_t = None
        self.s22_t = None
        self.s33_t = None
        self.s11_c = None
        self.s22_c = None
        self.s33_c = None
        self.t12 = None
        self.t13 = None
        self.t23 = None
        self.e11_c = None
        self.e22_c = None
        self.e33_c = None
        self.e11_t = None
        self.e22_t = None
        self.e33_t = None
        self.g12 = None
        self.g13 = None
        self.g23 = None
        self.gM0 = None
        self.C1a = None
        self.C2a = None
        self.C3a = None
        self.C4a = None

    def _minor_poissons_ratios(self):
        ''' Derives minor Poisson's ratios
        '''
        self.nu31 = self.nu13 * self.E3 / self.E1
        self.nu21 = self.nu12 * self.E2 / self.E1
        self.nu32 = self.nu23 * self.E3 / self.E2

    def set_props_iso(self, E1, nu12, rho):
        ''' Sets isotropic material properties.
        '''
        self.rho = rho
        self.E1 = E1
        self.nu12 = nu12

        # derived
        self.E2 = self.E1
        self.E3 = self.E1
        self.nu23 = self.nu12
        self.nu13 = self.nu12
        self.G12 = self.E1 / (2 * (1 + self.nu12))
        self.G23 = self.G12
        self.G13 = self.G12
        self._minor_poissons_ratios()

    def set_props_uniax(self, E1, E2, nu12, G12, nu23, rho):
        ''' Sets material properties for uniax.
        '''
        self.rho = rho
        self.E1 = E1
        self.E2 = E2
        self.nu12 = nu12
        self.G12 = G12
        self.nu23 = nu23

        # derived
        self.E3 = self.E2
        # Schuermann, p.202, eq. 8.35
        self.G23 = self.E2 / (2 * (1 + self.nu23))
        self.nu13 = self.nu12
        self.G13 = self.G12
        self._minor_poissons_ratios()

    def set_props(self, E1, E2, E3, nu12, nu13, nu23, G12, G13, G23, rho):
        ''' Set 3D material properties.
        '''
        self.rho = rho
        self.E1 = E1
        self.E2 = E2
        self.E3 = E3
        self.nu12 = nu12
        self.nu13 = nu13
        self.nu23 = nu23
        self.G12 = G12
        self.G13 = G13
        self.G23 = G23

        self._minor_poissons_ratios()

    def matprops(self):
        ''' Returns the list of material properties.

        :return matprops_list: list of material properties suitable for the st3d
            dict
        :return matprops_labels: list of labels used in matprops_list
        '''
        matprops_list = [self.E1,
                         self.E2,
                         self.E3,
                         self.nu12,
                         self.nu13,
                         self.nu23,
                         self.G12,
                         self.G13,
                         self.G23,
                         self.rho]

        matprops_labels = ['E1',
                           'E2',
                           'E3',
                           'nu12',
                           'nu13',
                           'nu23',
                           'G12',
                           'G13',
                           'G23',
                           'rho']

        return matprops_list, matprops_labels

    def set_resists_strains_iso(self, failcrit, e11_t, e11_c, g12):
        ''' Sets the characteristic allowable strains for an isotropic material.
        '''
        self.failcrit = failcrit
        self.e11_t = e11_t
        self.e11_c = e11_c
        self.g12 = g12

        # derived
        self.e22_t = self.e11_t
        self.e33_t = self.e11_t
        self.e22_c = self.e11_c
        self.e33_c = self.e11_c
        self.g13 = self.g12
        self.g23 = self.g12
        self._resists_stresses()

    def set_resists_strains_uniax(self, failcrit, e11_t, e22_t, e11_c, e22_c, g12):
        ''' Sets the characteristic allowable strains for a uniax material.
        '''
        self.failcrit = failcrit
        self.e11_t = e11_t
        self.e11_c = e11_c
        self.e22_t = e22_t
        self.e22_c = e22_c
        self.g12 = g12

        # derived
        self.e33_t = self.e22_t
        self.e33_c = self.e22_c
        self.g13 = self.g12
        self.g23 = self.g12
        self._resists_stresses()

    def set_resists_strains(self, failcrit, e11_t, e22_t, e33_t, e11_c, e22_c,
                            e33_c, g12, g13, g23):
        ''' Sets the characteristic allowable strains.
        '''
        self.failcrit = failcrit
        self.e11_t = e11_t
        self.e22_t = e22_t
        self.e33_t = e33_t
        self.e11_c = e11_c
        self.e22_c = e22_c
        self.e33_c = e33_c
        self.g12 = g12
        self.g13 = g13
        self.g23 = g23

        self._resists_stresses()

    def _resists_stresses(self):
        ''' Determines stress resistances from strain resistances and stiffnesses
        '''
        self.s11_t = self.e11_t * self.E1
        self.s22_t = self.e22_t * self.E2
        self.s33_t = self.e33_t * self.E3
        self.s11_c = self.e11_c * self.E1
        self.s22_c = self.e22_c * self.E2
        self.s33_c = self.e33_c * self.E3
        self.t12 = self.g12 * self.G12
        self.t13 = self.g13 * self.G13
        self.t23 = self.g23 * self.G23

    def set_safety_GL2010(self, gM0, C1a, C2a, C3a, C4a):
        ''' Sets the material safety factors.
        '''
        self.gM0 = gM0
        self.C1a = C1a
        self.C2a = C2a
        self.C3a = C3a
        self.C4a = C4a

    def failmat(self):
        ''' Returns the list of material resistances and safety factors.

        :return failmat_list: list of resistances and safety factors suitable
            for the st3d dict
        :return failmat_labels: list of labels used in failmat_list
        '''
        failmat_list = [self.s11_t,
                        self.s22_t,
                        self.s33_t,
                        self.s11_c,
                        self.s22_c,
                        self.s33_c,
                        self.t12,
                        self.t13,
                        self.t23,
                        self.e11_c,
                        self.e22_c,
                        self.e33_c,
                        self.e11_t,
                        self.e22_t,
                        self.e33_t,
                        self.g12,
                        self.g13,
                        self.g23,
                        self.gM0,
                        self.C1a,
                        self.C2a,
                        self.C3a,
                        self.C4a]

        failmat_labels = ['s11_t',
                          's22_t',
                          's33_t',
                          's11_c',
                          's22_c',
                          's33_c',
                          't12',
                          't13',
                          't23',
                          'e11_c',
                          'e22_c',
                          'e33_c',
                          'e11_t',
                          'e22_t',
                          'e33_t',
                          'g12',
                          'g13',
                          'g23',
                          'gM0',
                          'C1a',
                          'C2a',
                          'C3a',
                          'C4a']

        return failmat_list, failmat_labels


class DivisionPoint(object):
    '''Holds a division point's arc positions on the blade surface.

    :param arc: arc length positions on airfoil's surface
            -1.0 = trailing edge suction side
            1.0 = trailing edge pressure side
            0.0 = leading edge
    :type arc: array
    '''

    def __init__(self):
        self.arc = None


class Layer(object):
    """ Holds a layer's thickness and angle along the blade.

    :param thickness: layer thickness
    :type thickness: array
    :param angle: layup angle (deg)
    :type angle: array

    .. note:: A layer thickness can go to zero if material disappears at
              a certain spanwise position.
    """

    def __init__(self):
        self.thickness = None
        self.angle = None


class Region(object):
    """ Holds a region's layers along the blade.

    :param layers: Dictionary of Layer3D objects
    :type layers: dict
    """

    def __init__(self):
        self.layers = OrderedDict()

    def add_layer(self, name):
        ''' Inserts a layer into layers dict.

        :param name: Name of the material
        :return: The layer added to the region
        '''
        dubl = 0
        for k in self.layers.keys():
            if name == k[:-2]:
                dubl += 1

        lname = '%s%02d' % (name, dubl)

        layer = Layer()
        self.layers[lname] = layer
        return layer

    def init_stack(self):
        ''' Initiates thickness and angle matrices of the region plus the max
            thickness of each layer
        :return: thick_matrix: 2d np.array containing thicknesses of all layers (size: no_layers,s)
        :return: thick_max: 1d np.array max thickness per layer
        :return: angle_matrix: 2d np.array containing angles of all layers (size: no_layers,s)
        '''
        thickmatdata = []
        for v in self.layers.values():
            thickmatdata.append(v.thickness)
        self.thick_matrix = np.fliplr(np.rot90(np.array(thickmatdata), -1))

        thickmaxdata = []
        for l in range(self.thick_matrix.shape[1]):
            thickmaxdata.append(np.max(self.thick_matrix[:, l]))
        self.thick_max = np.array(thickmaxdata)

        anglematdata = []
        for v in self.layers.values():
            anglematdata.append(v.angle)
        self.angle_matrix = np.fliplr(np.rot90(np.array(anglematdata), -1))

        return self.thick_matrix, self.thick_max, self.angle_matrix


class BladeLayup(object):
    """ Span-wise layup definition of a blade.

    :param s: Spanwise discretization of the blade's layup
    :type s: array
    :param regions: Dictionary of Region3D objects representing regions on blade
                    surface
    :param webs: Dictionary of Region3D objects representing regions as webs
    :param iwebs: DP indices connecting webs to the surface
    :param DPs: Dictionary of DivisionPoint objects
    :param materials: Dictionary of Material objects

    """

    def __init__(self):
        self.s = None
        self.regions = OrderedDict()
        self.webs = OrderedDict()
        self.iwebs = None
        self.DPs = OrderedDict()
        self.materials = OrderedDict()

        self._warns = 0  # counter for inconsistencies

        self._version = 2  # file version

    def init_regions(self, nr, names=[]):
        ''' Initialize a number of nr regions.

        :param nr: Number of regions to be initialized
        :type nr: integer
        :param names: Names of regions (optional), must have the length of nr
        '''

        for i in range(nr + 1):
            self.DPs['DP%02d' % i] = DivisionPoint()

        for i in range(nr):
            try:
                name = names[i]
            except:
                name = 'region%02d' % i
            self._add_region(name)

    def init_webs(self, nw, iwebs, names=[]):
        ''' Initialize a number of nw webs.

        :param nw: Number of webs to be initialized
        :type nw: integer
        :param iwebs: List of DP index pairs connecting a web
            Example: [[-1, 0], [1, 4]] means 2 webs, web00 uses DP00-1 (clock-wise
            counting) and DP03 (its layup stacking direction is then inwards),
            web01 uses DP01 and DP04
        :param names: Names of webs (optional), must have the length of nw

        '''
        self.iwebs = iwebs
        for i in range(nw):
            try:
                name = names[i]
            except:
                name = 'web%02d' % i
            self._add_web(name)

    def init_bonds(self, nb, ibonds, names=[]):
        ''' Initialize a number of nb bondlines.

        :param nb: Number of webs to be initialized
        :type nb: integer
        :param ibonds: List of DP index quads connecting a web
            Example: [[0, 1, -2, -1]]
        :param names: Names of bonds (optional), must have the length of nb

        '''
        self.bonds = OrderedDict()
        # self.ibonds = None
        self.ibonds = ibonds
        for i in range(nb):
            try:
                name = names[i]
            except:
                name = 'bond%02d' % i
            self._add_bond(name)

    def _add_region(self, name):
        ''' Adds region to the blade
        '''
        region = Region()
        self.regions[name] = region
        return region

    def _add_web(self, name):
        ''' Adds web to the blade
        '''
        region = Region()
        self.webs[name] = region
        return region

    def _add_bond(self, name):
        ''' Adds bondline to the blade
        '''
        region = Region()
        self.bonds[name] = region
        return region

    def add_material(self, name):
        ''' Inserts material into materials dict.

        :param name: Name of the material.
        :return: The added material object.
        '''
        material = Material()
        self.materials[name] = material
        return material

    def check_consistency(self):
        ''' Checks the consistency of the BladeLayup.

        This method compares the length of any vectors in DPs, regions and webs
        with BladeLayup's s length. Further, materials set as layers are checked
        for their existence in the materials dict. Also initilized objects are
        checked if they have unset values.
        '''
        print('Starting consistency check of BladeLayup.')
        #  check BladeLayup attributes
        for attr, val in self.__dict__.items():
            if val is None:
                self._warns += 1
                print(('Attribute %s is not set.') % (attr))

        # check material attributes
        for km, vm in self.materials.items():
            for attr, val in vm.__dict__.items():
                if val is None:
                    self._warns += 1
                    print(('%s\'s attribute %s is not set.') % (km, attr))

        # calc BladeLayup's s length
        len_s = len(self.s)
        # check DPs
        for dpk, dpv in self.DPs.items():
            # check DP attributes
            for attr, val in dpv.__dict__.items():
                if val is None:
                    self._warns += 1
                    print(('%s\'s attribute %s is not set.') % (dpk, attr))
            # check DP lengths
            len_dp = len(dpv.arc)
            if len_dp != len_s:
                self._warns += 1
                print(('%s\'s size (%s) is unequal to size of s (%s).') % (
                    dpk, len_dp, len_s))

        def _check_regions(dictionary):
            ''' Check regions' consistency.

            :param dictionary: self.regions or self.webs or self.bonds
            '''
            for rk, rv in dictionary.items():
                # check dictionary attributes
                for attr, val in rv.__dict__.items():
                    if val is None:
                        self._warns += 1
                        print(('%s\'s attribute %s is not set.') % (rk, attr))
                for lk, lv in rv.layers.items():
                    # check layer attres
                    for attr, val in lv.__dict__.items():
                        if val is None:
                            self._warns += 1
                            print(('%s\'s %s attribute %s is not set.') % (
                                rk, lk, attr))
                    # check if layer's materials exist
                    # note: last two digits comply layer nr.
                    if lk[:-2] not in iter(self.materials.keys()):
                        # if lk not in self.materials.iterkeys():
                        self._warns += 1
                        print(('%s\'s %s does not exist in materials dict.') % (
                            rk, lk[:-2]))
                    # check vector lengths
                    len_thick = len(lv.thickness)
                    len_ang = len(lv.angle)
                    if len_thick != len_s:
                        self._warns += 1
                        print(('%s\'s %s thickness size (%s) is unequal to size of s (%s).') % (
                            rk, lk, len_thick, len_s))
                    if len_ang != len_s:
                        self._warns += 1
                        print(('%s\'s %s angle size (%s) is unequal to size of s (%s).') % (
                            rk, lk, len_ang, len_s))

        # check surface regions and webs
        _check_regions(self.regions)
        _check_regions(self.webs)
        if hasattr(self, 'bonds'):
            _check_regions(self.bonds)

        if self._warns:
            print(('%s inconsistencies detected!' % self._warns))
        else:
            print('OK.')

    def print_plybook(self, filename='plybook', vmode='stack',
                      include_materials=False, slim=[], add_filename=False,
                      add_page=False):
        ''' Prints a PDF file for layup visualization.

        :param filename: name of the PDF
        :param vmode: 'stack' or 'explode' visualization of layup
        :param include_materials: True if materials should be included (runs only
                when check_consistency() is OK)
        :param slim: [slim_lower, slim_upper] limits the x axis of the plots
        :param add_filename: if True filename is added to each plot 
        :param add_page: if True page number of plybook is added to each plot 

        '''

        import matplotlib.pylab as plt
        import matplotlib.patches as patches
        from matplotlib.backends.backend_pdf import PdfPages

        pb = PdfPages(filename + '.pdf')

        cm = plt.get_cmap('jet')

        # create material color dict (necessary for the case that materials list
        # is longer than 7)
        start = 0.2
        stop = 1.0
        number_of_lines = len(self.materials)
        mat_colors = [cm(x) for x in np.linspace(start, stop, number_of_lines)]
        cm_dict = {}
        for i, m in enumerate(self.materials.keys()):
            cm_dict[m] = mat_colors[i]

        page_pos_x = 0.98
        page_pos_y = 0.02
        filename_pos_x = 0.02
        filename_pos_y = 0.02
        if include_materials:
            # material properties
            fig, _ = plt.subplots()
            plt.title('MATPROPS')
            N = 10
            ind = np.arange(N)    # the x locations for the groups
            # the width of the bars: can also be len(x) sequence
            width = 1.0 / N
            for i, mat_name in enumerate(self.materials.keys()):
                plt.bar(
                    ind + i *
                    width, self.materials[mat_name].matprops()[0], width,
                    color=cm_dict[mat_name], label=mat_name, log=1)
                if i == 1:
                    matprops_labels = self.materials[mat_name].matprops()[1]

            plt.xticks(ind + .5, matprops_labels)
            plt.legend(loc='best', prop={'size': 10}, framealpha=0.5)
            # add page number and filename to figure
            if add_page:
                fig.text(
                    page_pos_x, page_pos_y, str(pb.get_pagecount() + 1), ha='left', fontsize=8)
            if add_filename:
                fig.text(filename_pos_x, filename_pos_y, str(
                    filename), ha='left', fontsize=8)
            pb.savefig()  # save fig to plybook

            # failmat_stress
            fig, _ = plt.subplots()
            plt.title('FAILMAT_STRESS')
            N = 9
            ind = np.arange(N)    # the x locations for the groups
            # the width of the bars: can also be len(x) sequence
            width = 1.0 / N
            for i, mat_name in enumerate(self.materials.keys()):
                plt.bar(
                    ind + i * width, self.materials[mat_name].failmat()[0][:N],
                    width,
                    color=cm_dict[mat_name], label=mat_name, log=1)
                if i == 1:
                    matprops_labels = self.materials[mat_name].failmat()[1]

            plt.xticks(ind + .5, matprops_labels[:N])
            plt.legend(loc='best', prop={'size': 10}, framealpha=0.5)
            # add page number and filename to figure
            if add_page:
                fig.text(
                    page_pos_x, page_pos_y, str(pb.get_pagecount() + 1), ha='left', fontsize=8)
            if add_filename:
                fig.text(filename_pos_x, filename_pos_y, str(
                    filename), ha='left', fontsize=8)
            pb.savefig()  # save fig to plybook

            # failmat_strain
            fig, _ = plt.subplots()
            plt.title('FAILMAT_STRAIN')
            N = 9
            ind = np.arange(N)    # the x locations for the groups
            # the width of the bars: can also be len(x) sequence
            width = 1.0 / N
            for i, mat_name in enumerate(self.materials.keys()):
                plt.bar(
                    ind + i *
                    width, self.materials[mat_name].failmat()[0][N:2 * N],
                    width,
                    color=cm_dict[mat_name], label=mat_name)
                if i == 1:
                    matprops_labels = self.materials[mat_name].failmat()[1]

            plt.xticks(ind + .5, matprops_labels[N:2 * N])
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.legend(loc='best', prop={'size': 10}, framealpha=0.5)
            # add page number and filename to figure
            if add_page:
                fig.text(
                    page_pos_x, page_pos_y, str(pb.get_pagecount() + 1), ha='left', fontsize=8)
            if add_filename:
                fig.text(filename_pos_x, filename_pos_y, str(
                    filename), ha='left', fontsize=8)
            pb.savefig()  # save fig to plybook

            # failmat_safety
            fig, _ = plt.subplots()
            plt.title('FAILMAT_SAFETY')
            N = 5
            ind = np.arange(N)    # the x locations for the groups
            # the width of the bars: can also be len(x) sequence
            width = 1.0 / 9
            for i, mat_name in enumerate(self.materials.keys()):
                plt.bar(
                    ind + i *
                    width, self.materials[
                        mat_name].failmat()[0][18:18 + N],
                    width,
                    color=cm_dict[mat_name], label=mat_name)
                if i == 1:
                    matprops_labels = self.materials[mat_name].failmat()[1]

            plt.xticks(ind + .5, matprops_labels[18:18 + N])
            plt.legend(loc='best', prop={'size': 10}, framealpha=0.5)
            # add page number and filename to figure
            if add_page:
                fig.text(
                    page_pos_x, page_pos_y, str(pb.get_pagecount() + 1), ha='left', fontsize=8)
            if add_filename:
                fig.text(filename_pos_x, filename_pos_y, str(
                    filename), ha='left', fontsize=8)
            pb.savefig()  # save fig to plybook

        fig, _ = plt.subplots()
        plt.title('DPs')
        for di, dk in enumerate(sorted(self.DPs, reverse=True)):
            plt.plot(self.s, self.DPs[dk].arc, label=dk[-2:])
        plt.legend(loc='best', prop={'size': 6}, bbox_to_anchor=(1, 1))
        # draw station lines
        for s in self.s:
            plt.plot([self.s, self.s], [-1, 1], 'k', linewidth=0.5)
        if slim:
            plt.xlim((slim[0], slim[1]))
        # add page number and filename to figure
        if add_page:
            fig.text(
                page_pos_x, page_pos_y, str(pb.get_pagecount() + 1), ha='left', fontsize=8)
        if add_filename:
            fig.text(filename_pos_x, filename_pos_y, str(
                filename), ha='left', fontsize=8)
        pb.savefig()  # save fig to plybook

        def _region_sets(reg_type):
            ''' Compares all regions of reg_type and creates a list of unique reg_types

            :param reg_type: self.regions or self.webs or self.bonds
            :return: list: unique region sets
            '''

            # list of rthicks and region cum thicknesses
            rthicks = []
            rmaxthicks = []
            for i, rv in enumerate(reg_type.values()):
                # init thicknesses
                rv.init_stack()
                rthicks.append(rv.thick_matrix)
                rmaxthicks.append(np.sum(rv.thick_max))
            # maximum thickness of sets
            rmaxthick = np.max(rmaxthicks)
            # check for identic regions
            rsets = []
            for rt0 in rthicks:
                i0_idents = []
                for i, rt in enumerate(rthicks):
                    if np.array_equal(rt0, rt):
                        i0_idents.append(i)
                rsets.append(i0_idents)
            # remove duplicate entries
            rsets = list(map(list, OrderedDict.fromkeys(list(map(tuple, rsets)))))
            return rsets, rmaxthick

        def _plot_region(rsets, reg_type):
            ''' Adds plots of region sets to plybook

            :param rsets: list of region sets
            :param reg_type: self.regions or self.webs or self.bonds
            '''
            if reg_type == self.regions:
                rtype = 'region'
            elif reg_type == self.webs:
                rtype = 'web'
            elif reg_type == self.bonds:
                rtype = 'bond'

            for rset in rsets:
                r = reg_type['%s%02d' % (rtype, rset[0])]
                fig1, ax1 = plt.subplots()
                #fig1 = plt.figure()
                #ax1 = fig1.add_subplot(111)
                plt.title(rtype.upper() + ' ' + str(rset))
                # draw station lines
                for s in self.s:
                    ax1.plot(
                        [self.s, self.s], [0, maxthick], 'k', linewidth=0.5)
                t = np.zeros_like(self.s)
                for k, l in r.layers.items():
                    mat_name = k[:-2]
                    mat_count = k[-2:]
                    if vmode == 'stack':
                        # draw layer box
                        plt.plot(self.s, t + l.thickness, 'k')
                        # draw layer thickness distro
                        plt.fill_between(self.s, t, t + l.thickness,
                                         color=cm_dict[mat_name],
                                         label=mat_name if int(mat_count) == 0 else "_nolegend_")
                        t = t + l.thickness
                    elif vmode == 'explode':
                        # check for layer drops
                        drops = []
                        drop_prev = 0
                        for s, thick in enumerate(l.thickness):
                            if t[s] + thick > t[s]:
                                drop = 1
                            else:
                                drop = 0
                            if drop != drop_prev and drop_prev == 0:
                                drops.append(s)
                            elif drop != drop_prev and drop_prev == 1:
                                drops.append(s - 1)
                            drop_prev = drop
                        if len(drops) == 1:
                            drops.append(len(l.thickness) - 1)
                        max_thick = np.max(l.thickness)
                        # draw layer box
                        for di in range(len(drops) - 1):
                            # draw box per layer
                            north_west = [
                                self.s[drops[di]], t[drops[di]] + max_thick]
                            south_west = [self.s[drops[di]], t[drops[di]]]
                            north_east = [
                                self.s[drops[di + 1]], t[drops[di + 1]] + max_thick]
                            south_east = [
                                self.s[drops[di + 1]], t[drops[di + 1]]]
                            if drops[di] > 0:  # we have a plydrop up
                                north_west = [
                                    self.s[drops[di] - 1], t[drops[di]] + max_thick]
                                south_west = [
                                    self.s[drops[di] - 1], t[drops[di]]]
                                south_south_west = [
                                    self.s[drops[di]], t[drops[di]]]
                                north_mid_west = [
                                    self.s[drops[di]], (t + l.thickness)[drops[di]]]
                                # draw layer thickness distro drop
                                ax1.add_patch(patches.Polygon(
                                    [south_west, north_mid_west,
                                        south_south_west],
                                    linewidth=0, facecolor=cm_dict[mat_name]))
                            # we have a plydrop down
                            if drops[di + 1] < len(self.s) - 1:
                                south_east = [
                                    self.s[drops[di + 1] + 1], t[drops[di + 1]]]
                                north_east = [
                                    self.s[drops[di + 1] + 1], t[drops[di + 1]] + max_thick]
                                south_south_east = [
                                    self.s[drops[di + 1]], t[drops[di + 1]]]
                                north_mid_east = [
                                    self.s[drops[di + 1]], (t + l.thickness)[drops[di + 1]]]
                                # draw layer thickness distro drop
                                ax1.add_patch(patches.Polygon(
                                    [north_mid_east, south_east,
                                        south_south_east],
                                    linewidth=0, facecolor=cm_dict[mat_name]))
                            ax1.add_patch(patches.Polygon(
                                [south_west, north_west,
                                    north_east, south_east],
                                fill=False))
                        # draw layer thickness distro
                        plt.fill_between(self.s, t, t + l.thickness,
                                         where=t < t + l.thickness,
                                         color=cm_dict[mat_name],
                                         label=mat_name if int(mat_count) == 0 else "_nolegend_")
                        t = t + max_thick
                plt.ylim([0, maxthick])  # set all plot limits to maxthickness
                plt.legend(loc='best', prop={'size': 10}, framealpha=0.5)
                if slim:
                    plt.xlim((slim[0], slim[1]))
                # add page number and filename to figure
                if add_page:
                    fig1.text(
                        page_pos_x, page_pos_y, str(pb.get_pagecount() + 1), ha='left', fontsize=8)
                if add_filename:
                    fig1.text(filename_pos_x, filename_pos_y, str(
                        filename), ha='left', fontsize=8)
                pb.savefig(fig1)  # save fig to plybook

        rsets, rmaxthick = _region_sets(self.regions)
        wsets, wmaxthick = _region_sets(self.webs)
        if hasattr(self, 'bonds'):
            bsets, bmaxthick = _region_sets(self.bonds)
            maxthick = np.max([rmaxthick, wmaxthick])  # , bmaxthick])
        else:
            maxthick = np.max([rmaxthick, wmaxthick])

        _plot_region(rsets, reg_type=self.regions)
        _plot_region(wsets, reg_type=self.webs)
        if hasattr(self, 'bonds'):
            _plot_region(bsets, reg_type=self.bonds)

        pb.close()  # close plybook


def _create_regions(dictionary):
    ''' create regions list

    :param dictionary: bl.regions or bl.webs or bl.bonds
    :return: List of regions
    '''
    regs = []
    for k, v in dictionary.items():
        r = {}
        r['layers'] = []
        andata = []
        thdata = []
        for k, v in v.layers.items():
            r['layers'].append(k)
            thdata.append(v.thickness)
            andata.append(v.angle)
        r['thicknesses'] = np.fliplr(np.rot90(np.r_[thdata], -1))
        r['angles'] = np.fliplr(np.rot90(np.r_[andata], 1))
        regs.append(r)
    return regs


def create_bladestructure(bl):
    """ Creator for BladeStructureVT3D data from a BladeLayup object

    :param bl: BladeLayupShell object
    :return: The st3d dictionary containing geometric and material properties
        definition of the blade structure
    """

    st3d = {}

    st3d['version'] = bl._version

    st3d['materials'] = OrderedDict()
    for i, name in enumerate(bl.materials.keys()):
        st3d['materials'][name] = i

    matprops = []
    failmat = []
    failcrit = []
    for v in bl.materials.values():
        matprops.append(v.matprops()[0])
        failmat.append(v.failmat()[0])
        failcrit.append(v.failcrit)

    st3d['matprops'] = np.r_[matprops]
    st3d['failmat'] = np.r_[failmat]
    st3d['failcrit'] = failcrit
    st3d['web_def'] = bl.iwebs
    if hasattr(bl, 'bonds'):
        st3d['bond_def'] = bl.ibonds
    st3d['s'] = bl.s

    dpdata = []
    for v in bl.DPs.values():
        dpdata.append(v.arc)
    st3d['DPs'] = np.fliplr(np.rot90(np.r_[dpdata], -1))

    st3d['regions'] = _create_regions(bl.regions)
    st3d['webs'] = _create_regions(bl.webs)
    if hasattr(bl, 'bonds'):
        st3d['bonds'] = _create_regions(bl.bonds)

    return st3d


def create_bladelayup(st3d):
    """ Creator for BladeLayup object from BladeStructureVT3D

    :param: st3d: The st3d dictionary containing geometric and material properties
        definition of the blade structure

    :return bl: BladeLayupShell object
    """

    bl = BladeLayup()

    bl._version = st3d['version']

    bl.s = st3d['s']

    for i, k in enumerate(st3d['materials'].keys()):
        mat = bl.add_material(k)
        mat.set_props(E1=st3d['matprops'][i][0],
                      E2=st3d['matprops'][i][1],
                      E3=st3d['matprops'][i][2],
                      nu12=st3d['matprops'][i][3],
                      nu13=st3d['matprops'][i][4],
                      nu23=st3d['matprops'][i][5],
                      G12=st3d['matprops'][i][6],
                      G13=st3d['matprops'][i][7],
                      G23=st3d['matprops'][i][8],
                      rho=st3d['matprops'][i][9])
        mat.set_resists_strains(failcrit=st3d['failcrit'][i],
                                e11_t=st3d['failmat'][i][9],
                                e22_t=st3d['failmat'][i][10],
                                e33_t=st3d['failmat'][i][11],
                                e11_c=st3d['failmat'][i][12],
                                e22_c=st3d['failmat'][i][13],
                                e33_c=st3d['failmat'][i][14],
                                g12=st3d['failmat'][i][15],
                                g13=st3d['failmat'][i][16],
                                g23=st3d['failmat'][i][17])
        mat.set_safety_GL2010(gM0=st3d['failmat'][i][18],
                              C1a=st3d['failmat'][i][19],
                              C2a=st3d['failmat'][i][20],
                              C3a=st3d['failmat'][i][21],
                              C4a=st3d['failmat'][i][22])

    bl.init_regions(len(st3d['regions']))

    for idp, dp in enumerate(bl.DPs.values()):
        dp.arc = st3d['DPs'][:, idp]

    for ir, reg in enumerate(st3d['regions']):
        for il, lay in enumerate(reg['layers']):
            layer = bl.regions['region%02d' % ir].add_layer(lay[:-2])
            layer.thickness = np.squeeze(reg['thicknesses'][:, il])
            layer.angle = np.squeeze(reg['angles'][:, il])

    bl.init_webs(nw=len(st3d['web_def']), iwebs=st3d['web_def'])

    for ir, reg in enumerate(st3d['webs']):
        for il, lay in enumerate(reg['layers']):
            layer = bl.webs['web%02d' % ir].add_layer(lay[:-2])
            layer.thickness = np.squeeze(reg['thicknesses'][:, il])
            layer.angle = np.squeeze(reg['angles'][:, il])

    if st3d['bond_def']:
        bl.init_bonds(nb=1, ibonds=st3d['bond_def'])

        for ir, reg in enumerate(st3d['bonds']):
            for il, lay in enumerate(reg['layers']):
                layer = bl.bonds['bond%02d' % ir].add_layer(lay[:-2])
                layer.thickness = np.squeeze(reg['thicknesses'][:, il])
                layer.angle = np.squeeze(reg['angles'][:, il])

    bl.check_consistency()

    return bl


def pickle_bladelayup(bl):
    ''' pickle a bl object into an ascii file

    :param bl
    :return: List of regions
    '''

    with open('bl.pkl', 'wb') as mysavedata:
        pickle.dump(bl, mysavedata)


def unpickle_bladelayup():
    ''' unpickle a bl object from an ascii file

    :return: bl
    '''
    with open('bl.pkl', 'rb') as myrestoredata:
        bl = pickle.load(myrestoredata)
    return bl
