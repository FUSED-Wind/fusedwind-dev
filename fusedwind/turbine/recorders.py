
import numpy as np

from openmdao.recorders.base_recorder import BaseRecorder
from fusedwind.turbine.structure import write_bladestructure
from fusedwind.turbine.geometry import write_blade_planform
from fusedwind.lib.geom_tools import calculate_length


def get_structure_recording_vars(st3d, with_props=False, with_CPs=False):
    """
    convenience method for generating list of variable names
    of the blade structure for adding to a recorder

    params
    ------
    st3d: dict
        dictionary of blade structure
    with_props: bool
        also add variable names with blade structure props
        computed by `BladeStructureProperties`

    returns
    -------
    recording_vars: list
        list of strings with names of all DPs, layer names,
        and material props.
    """
    recording_vars = []
    s = st3d['s']
    nsec = s.shape[0]
    nDP = st3d['DPs'].shape[1]

    DPs = ['DP%02d' % i for i in range(nDP)]
    DPs_C = ['DP%02d_C' % i for i in range(nDP)]
    regions = []
    webs = []
    for ireg, reg in enumerate(st3d['regions']):
        layers = []
        for i, lname in enumerate(reg['layers']):
            varname = 'r%02d%s' % (ireg, lname)
            layers.extend([varname + 'T', varname + 'A'])
            if with_CPs:
                layers.extend([varname + 'T_C', varname + 'A_C'])
        regions.extend(layers)
        if with_props:
            regions.append('r%02d_thickness' % ireg)
            regions.append('r%02d_width' % ireg)
    for ireg, reg in enumerate(st3d['webs']):
        layers = []
        for i, lname in enumerate(reg['layers']):
            varname = 'w%02d%s' % (ireg, lname)
            layers.extend([varname + 'T', varname + 'A'])
            if with_CPs:
                layers.extend([varname + 'T_C', varname + 'A_C'])
        if with_props:
            regions.append('r%02d_thickness' % ireg)
            regions.append('r%02d_width' % ireg)
        webs.extend(layers)

    recording_vars.extend(DPs)
    if with_CPs:
        recording_vars.extend(DPs_C)
    recording_vars.extend(regions)
    recording_vars.extend(webs)
    recording_vars.append('matprops')
    recording_vars.append('failmat')
    recording_vars.append('s_st')
    if 'cap_width_ps' in list(st3d.keys()):
        recording_vars.extend(['cap_center_ps',
                               'cap_center_ss',
                               'cap_width_ps',
                               'cap_width_ss',
                               'te_width',
                               'le_width', 'struct_angle'])
        recording_vars.extend(['w%02dpos' % i for i in range(1, len(st3d['web_def']))])
    if with_props:
        recording_vars.extend(['pacc_u',
                               'pacc_l',
                               'pacc_u_curv',
                               'pacc_l_curv'])
        for ireg, reg in enumerate(st3d['webs']):
            recording_vars.extend(['web_angle%02d' % ireg,
                                  'web_offset%02d' % ireg])

    return recording_vars


def get_recorded_bladestructure(st3d, db, coordinate):

    data = db[coordinate]['Unknowns']

    s = st3d['s']
    stnew = {}
    stnew['s'] = data['s_st']
    stnew['materials'] = st3d['materials']
    stnew['matprops'] = st3d['matprops']
    stnew['failcrit'] = st3d['failcrit']
    stnew['failmat'] = st3d['failmat']
    stnew['web_def'] = st3d['web_def']
    stnew['version'] = st3d['version']
    if 'cap_width_ps' in list(st3d.keys()):
        try:
            stnew['dominant_regions'] = st3d['dominant_regions']
            stnew['cap_DPs'] = st3d['cap_DPs']
            stnew['le_DPs'] = st3d['le_DPs']
            stnew['te_DPs'] = st3d['te_DPs']
            stnew['struct_angle'] = data['struct_angle']
            stnew['cap_center_ps'] = data['cap_center_ps']
            stnew['cap_center_ss'] = data['cap_center_ss']
            stnew['cap_width_ps'] = data['cap_width_ps']
            stnew['cap_width_ss'] = data['cap_width_ss']
            stnew['te_width'] = data['te_width']
            stnew['le_width'] = data['le_width']
            for i in range(1, len(st3d['web_def'])):
                stnew['w%02dpos' % i] = data['w%02dpos' % i]
        except:
            print('Failed retrieving Param2 structural data from db')
    nsec = s.shape[0]
    nDP = st3d['DPs'].shape[1]

    DPs = np.array([data['DP%02d' % i] for i in range(nDP)]).T
    stnew['DPs'] = DPs
    stnew['regions'] = []
    stnew['webs'] = []
    for ireg, reg in enumerate(st3d['regions']):
        rnew = {}
        rnew['layers'] = reg['layers']
        nl = len(reg['layers'])
        Ts = np.zeros((nsec, nl))
        As = np.zeros((nsec, nl))

        for i, lname in enumerate(reg['layers']):
            varname = 'r%02d%s' % (ireg, lname)
            Ts[:, i] = data[varname + 'T']
            As[:, i] = data[varname + 'A']
        rnew['thicknesses'] = Ts
        rnew['angles'] = As
        stnew['regions'].append(rnew)

    for ireg, reg in enumerate(st3d['webs']):
        rnew = {}
        rnew['layers'] = reg['layers']
        nl = len(reg['layers'])
        Ts = np.zeros((nsec, nl))
        As = np.zeros((nsec, nl))

        for i, lname in enumerate(reg['layers']):
            varname = 'w%02d%s' % (ireg, lname)
            Ts[:, i] = data[varname + 'T']
            As[:, i] = data[varname + 'A']
        rnew['thicknesses'] = Ts
        rnew['angles'] = As
        stnew['webs'].append(rnew)

    return stnew

def write_recorded_bladestructure(st3d, db, coordinate, filebase):

    stnew = get_recorded_bladestructure(st3d, db, coordinate)

    write_bladestructure(stnew, filebase)

def get_planform_recording_vars(suffix='', with_CPs=False):
    """
    convenience method for generating list of variable names
    of the blade planform for adding to a recorder

    params
    ------
    suffix: str
        to record pf vars with e.g. _st appended to the variable names
    with_CPs: bool
        flag for also adding spline CPs arrays to list

    returns
    recording_vars: list
        list of strings with names of all planform vars
    """

    recording_vars = []

    names = ['s', 'x', 'y', 'z', 'rot_x', 'rot_y', 'rot_z',
             'chord', 'rthick', 'p_le']

    if suffix != '':
        pf_vars = [name + suffix for name in names]
    else:
        pf_vars = names
    curv_vars = [name + '_curv' for name in pf_vars]

    cp_vars = []
    if with_CPs:
        cp_vars = [name + '_C' for name in names]

    recording_vars.extend(pf_vars)
    recording_vars.extend(curv_vars)
    recording_vars.extend(cp_vars)

    return recording_vars

def get_recorded_planform(db, coordinate):

    data = db[coordinate]['Unknowns']
    pf = {}
    names = ['x', 'y', 'z', 'rot_x', 'rot_y', 'rot_z',
             'chord', 'rthick', 'p_le']
    for name in names:
        try:
            pf[name] = data[name]
        except:
            print('%s not found in database' % name)
            pf[name] = np.zeros(data['x'].shape[0])

    s = calculate_length(np.array([pf['x'], pf['y'], pf['z']]).T)

    pf['blade_length'] = pf['z'][-1]
    pf['s'] = s / s[-1]
    pf['smax'] = s[-1]

    return pf

def write_recorded_planform(db, coordinate, filebase):

    pf = get_recorded_planform(db, coordinate)

    write_blade_planform(pf, filebase)
