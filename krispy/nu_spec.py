'''
The following code is used to read in NuSTAR spectral data and create spectral models.
'''

import sys
from os.path import *
import os
import numpy as np
from astropy.io import fits

def read_xspec_txt(f):
    ''' Takes a the output .txt file from XSPEC and extracts useful information from it.
    
    Parameters
    ----------
    f : Str
            String for the .txt file.
            
    Returns
    -------
    The counts, photons, and ratios from the XSPEC output. 
    '''

    f = f if f.endswith('.txt') else f+'.txt'
    
    asc = open(f, 'r')  # We need to re-open the file
    data = asc.read()
    asc.close()
    
    sep_lists = data.split('!')[1].split('NO NO NO NO NO') #seperate counts info from photon info from ratio info
    _file = {}
    for l in range(len(sep_lists)):
        tmp_list = sep_lists[l].split('\n')[1:-1] # seperate list by lines and remove the blank space from list ends
        num_list = []
        for s in tmp_list:
            new_line = s.split(' ') # numbers are seperated by spaces
            for c, el in enumerate(new_line):
                if el == 'NO':
                    # if a value is NO then make it a NaN
                    new_line[c] = np.nan
            num_line = list(map(float, new_line)) # have the values as strings, map them to floats
            num_list.append(num_line)
        num_list = np.array(num_list)
        ## first block of NO NO NO... should be counts, second photons, third ratio
        if l == 0:
            _file['counts'] = num_list
        if l == 1:
            _file['photons'] = num_list
        if l == 2:
            _file['ratio'] = num_list
    return _file


def seperate(read_xspec_data, fitting_mode='1apec1fpm'):
    ''' Takes a the output from read_xspec_txt() function and splits the output into data, model, erros, energies, etc.
    
    Parameters
    ----------
    read_xspec_data : Dict
            Dictionary output from read_xspec_txt().

    fitting_mode : Str
            Information about the fit in XSPEC, e.g. 1apec model fit with on focal plane modules data: '1apec1fpm'.
            Default: '1apec1fpm'
            
    Returns
    -------
    The counts, photons, and ratios from the XSPEC output. 
    '''
    if fitting_mode == '1apec1fpm' or fitting_mode == '1apec2fpm':
        seperated = {'energy':read_xspec_data['counts'][:,0], 
                     'e_energy':read_xspec_data['counts'][:,1], 
                     'data1':read_xspec_data['counts'][:,2], 
                     'edata1':read_xspec_data['counts'][:,3], 
                     'model':read_xspec_data['counts'][:,4]}
        
    return seperated


def read_pha(file):
    ''' Takes a .pha file and extracts useful information from it.
    
    Parameters
    ----------
    file : Str
            String for the .pha file of the spectrum under investigation.
            
    Returns
    -------
    The channel numbers, counts, and the livetime for the observation. 
    '''

    hdul = fits.open(file)
    data = hdul[1].data
    header_FOR_LIVETIME = hdul[0].header
    hdul.close()

    return data['channel'], data['counts'], header_FOR_LIVETIME['LIVETIME']


def nustar_FluxCtsSpec(file):
    ''' Takes a .pha file and returns plotting innformation.
    
    Parameters
    ----------
    file : Str
            String for the .pha file of the spectrum under investigation.
            
    Returns
    -------
    The energy is the middle of the energy bin for the counts (energy_binMid), the half-range that energy bin spans (energy_binMid_err), 
    count rate per keV (cts), and its error (cts_err). 
    '''

    channel, counts, livetime = read_pha(file)
    
    energy_binStart = channel*0.04+1.6
    energy_binMid = energy_binStart + 0.02 # add 0.02 to get value in the middle of the energy bin
    energy_binMid_err = 0.02
    
    bin_size_keV = 0.04
    cts = (counts / bin_size_keV) / livetime # now in cts keV^-1 s^-1
    
    cts_err = (np.sqrt(counts) / bin_size_keV) / livetime
    
    return energy_binMid, energy_binMid_err, cts, cts_err


def read_arf(file):
    ''' Takes a .arf file and extracts useful information from it.
    
    Parameters
    ----------
    file : Str
            String for the .arf file of the spectrum under investigation.
            
    Returns
    -------
    The low and high boundary of energy bins, and the ancillary response [cm^2] (data['specresp']).  
    '''

    hdul = fits.open(file)
    data = hdul[1].data
    hdul.close()
    
    return data['energ_lo'], data['energ_hi'], data['specresp']


def read_rmf(file):
    ''' Takes a .rmf file and extracts useful information from it.
    
    Parameters
    ----------
    file : Str
            String for the .rmf file of the spectrum under investigation.
            
    Returns
    -------
    The low and high boundary of energy bins (data['energ_lo'], data['energ_hi']), number of sub-set channels in the energy 
    bin (data['n_grp']), starting index of each sub-set of channels (data['f_chan']), 
    number of channels in each sub-set (data['n_chan']), redistribution matrix [counts per photon] (data['matrix']). 
    '''

    hdul = fits.open(file)
    data = hdul[2].data
    hdul.close()
    
    return data['energ_lo'], data['energ_hi'], data['n_grp'], data['f_chan'], data['n_chan'], data['matrix']


def col2arr_py(data, **kwargs):
    ''' Takes a list of parameters for each energy channel from a .rmf file and returns it in the correct format.

    From: https://lost-contact.mit.edu/afs/physics.wisc.edu/home/craigm/lib/idl/util/vcol2arr.pro
    
    Parameters
    ----------
    data : array/list-like object
            One parameter's array/list from the .rmf file.

    kwargs : idl_check=Bool or idl_way=Bool
            If idl_check=True the funciton will throw an error if the Python and IDL methods give different answers (they shouldn't).
            If idl_way=True the IDL method's result with be returned instead of the new Python method described.
            
    Returns
    -------
    A 2D numpy array of the correctly ordered input data.
    
    Example
    -------
    data = FITS_rec([(  1.6 ,   1.64,   1, [0]   , [18]  , [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), 
                     (  1.64,   1.68,   1, [0]   , [20]  , [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                     (  1.68,   1.72,   2, [0,22], [20,1], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), 
                     dtype=(numpy.record, [('ENERG_LO', '>f4'), ('ENERG_HI', '>f4'), ('N_GRP', '>i2'), 
                                           ('F_CHAN', '>i4', (2,)), ('N_CHAN', '>i4', (2,)), ('MATRIX', '>i4', (2,))]))
                          
    >>> col2arr_py(data['F_CHAN'])
    array([[  0.,   0.],
           [  0.,   0.],
           [  0.,  22.]])
    ## max row length of 2 so 2 columns, each row is an energy channel. 
    '''

    ## this is the quicker way I have chosen to do in Python (this may be revised later but is ~30x faster than way below in Python)
    max_len = np.max([len(r) for r in data]) # find max row length
    chan_array_py = np.array([[*r, *(max_len-len(r))*[0]] for r in data]) # make each row that length (padding with 0)

    #*************************************************************************************************************************************************
    # if you want to involve the IDL way
    # set defaults to help check how it is done in IDL (second dict rewrites the keys of the first)
    defaults = {**{"idl_check":False, "idl_way":False}, **kwargs}

    if defaults["idl_check"] or defaults["idl_way"]:
        ## this is the way IDL does col2arr.pro
        chan = np.array(data)

        nc = np.array([len(n) for n in data]) # number of entries in each row
        accum_nc_almost = [nc[i]+sum(nc[0:i]) for i in range(len(nc))] # running total in each row
    
        # need 0 as start with 0 arrays
        accum_nc = np.array([0] + accum_nc_almost) # this acts as the index as if the array has been unraveled

        ## number of columns is the length of the row with the max number of entries (nc)
        ncol = np.max(nc)
        ## number of rows is just the number of rows chan just has
        nrow = len(chan)

        chan_array = np.zeros(shape=(nrow, ncol))

        for c in range(ncol):
            # indices where the number of entries in the row are greater than the column
            where = (nc > c).nonzero()[0] 

            # cycle through the rows to be filled in:
            ## if this row is one that has more values in it than the current column number then use the appropriate chan 
            ## number else make it zero
            chan_array[:,c] = [chan[n][c] if (n in where) else 0 for n in range(nrow)] 

        if defaults["idl_check"]:
            assert np.array_equal(chan_array_py, chan_array), \
            "The IDL way and the Python way here do not produce the same result. \nPlease check this but trust the IDL way more (set idl_way=True)!"
        if defaults["idl_way"]:
            return chan_array
    #*************************************************************************************************************************************************

    return chan_array_py


def vrmf2arr_py(data=None, n_grp_list=None, f_chan_array=None, n_chan_array=None):
    ''' Takes redistribution parameters for each energy channel from a .rmf file and returns it in the correct format.

    From: https://lost-contact.mit.edu/afs/physics.wisc.edu/home/craigm/lib/idl/spectral/vrmf2arr.pro
    
    Parameters
    ----------
    data : array/list-like object
            Redistribution matrix parameter array/list from the .rmf file. Units are counts per photon.
            Default : None
            
    n_grp_list : list
            Number of groups (sub-channels) for each energy bin. Total number of channels is the length of this list.
            Default : None
            
    f_chan_array : numpy.array
            The index of each sub-set channel from each energy bin from the .rmf file run through col2arr_py().
            Default : None
            
    n_chan_array : numpy.array
            The number of sub-set channels in each index for each energy bin from the .rmf file run through col2arr_py().
            Default : None
            
    Returns
    -------
    A 2D numpy array of the correctly ordered input data with dimensions of energy in the rows and channels in 
    the columns.
    
    Example
    -------
    data = FITS_rec([(  1.6 ,   1.64,   1, [0]   , [18]  , [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), 
                     (  1.64,   1.68,   1, [0]   , [20]  , [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                     (  1.68,   1.72,   2, [0,22], [20,1], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), ...,
                     dtype=(numpy.record, [('ENERG_LO', '>f4'), ('ENERG_HI', '>f4'), ('N_GRP', '>i2'), 
                                           ('F_CHAN', '>i4', (2,)), ('N_CHAN', '>i4', (2,)), ('MATRIX', '>i4', (2,))]))
                          
    >>> vrmf2arr_py(data['MATRIX'])
    array([[0.00033627, 0.0007369 , 0.00113175, ..., 0.        , 0.        , 0.        ],
           [0.00039195, 0.00079259, 0.00138341, ..., 0.        , 0.        , 0.        ],
           [0.00042811, 0.00083381, 0.00157794, ..., 0.        , 0.        , 0.        ],
                                                ...,
           [0.        , 0.        , 0.        , ..., 0.00408081, 0.00409889, 0.00403308],
           [0.        , 0.        , 0.        , ..., 0.00405333, 0.00413722, 0.00413216],
           [0.        , 0.        , 0.        , ..., 0.        , 0.        , 0.        ]])
    ## rows = energy, columns = channels 
    '''
    
    # unravel matrix array, can't use numpy.ravel as this has variable length rows
    ## now can index the start of each row with the running total
    unravel_dmat = []
    for n in data:
        for nn in n:
            unravel_dmat.append(nn)

    no_of_channels = len(n_grp_list)

    nrows = len(data)
    ncols = no_of_channels
    nc = np.array([len(n) for n in data])
    accum_nc_almost = [nc[i]+sum(nc[0:i]) for i in range(len(nc))]
    accum_nc = np.array([0] + accum_nc_almost) 
    # sorted wobble of diagonal lines, the indices were off by one left and right
    ## i.e. this is the running index so should start at zero

    mat_array = np.zeros(shape=(nrows, ncols))

    for r in range(nrows):
        if nc[r] > 0:
            # in IDL code the second index is -1 but that's because IDL's index boundaries 
            ## are both inclusive so drop the -1, i.e. was accum_nc[r+1]-1
            row = unravel_dmat[accum_nc[r]:accum_nc[r+1]] 

            c=0

            # for number of sub-set channels in each energy channel groups
            for ng in range(n_grp_list[r]):
                # want redist. prob. for number of sub-set channels 
                ## if c+m is larger than len(row)-1 then only want what we can get
                wanted_r = [row[int(c+m)] for m in np.arange(n_chan_array[r,ng]) if c+m <= len(row)-1 ]

                # now fill in the entries in mat_array from the starting number of the sub-set channel, 
                ## the fchan_array[r, ng]
                for z,wr in enumerate(wanted_r):
                    mat_array[r, int(f_chan_array[r, ng])+z] = wr

                # move the place that the that the index for row starts from along 
                c = c + n_chan_array[r,ng]

            # if dgrp[r] == 0 then above won't do anything, need this as not to miss out the 0th energy channel
            if n_grp_list[r] == 0:
                wanted_r = [row[int(c+m)] for m in np.arange(n_chan_array[r,0]) if c+m <= len(row)-1 ]
                for z,wr in enumerate(wanted_r):
                    mat_array[r, int(f_chan_array[r, 0])+z] = wr
                    
    return mat_array


def make_nusrm(rmf_matrix=(), arf_array=()):
    ''' Takes rmf and arf and produces the spectral response matrix fro NuSTAR.

    From: https://github.com/ianan/nsigh_nov14/blob/master/make_ns_srm.pro
    
    Parameters
    ----------
    rmf_matrix : numpy 2D array
            Array representing the redistribution matrix.
            Default : None
            
    arf_array : numpy 1D array/list
            List representing the ancillary response.
            Default : None
            
    Returns
    -------
    An array that is the spectral response (srm).
    '''
    
    if len(rmf_matrix) == 0 or len(arf_array) == 0:
        print('Need both RMF and ARF information to proceed.')
        return
    
    srm = np.array([rmf_matrix[r, :] * arf_array[r] for r in range(len(arf_array))]) # each energy bin row in the rmf is multiplied the arf value for the same energy bin
    return srm


def make_model(energies=None, photon_model=None, parameters=None, srm=None):
    ''' Takes a photon model array ( or function if you provide the pinputs with parameters), the spectral response matrix and returns a model count spectrum.
    
    Parameters
    ----------
    energies : array/list
            List of energies.
            Default : None

    photon_model : function/array/list
            Array -OR- function representing the photon model (if it's a function, provide the parameters of the function as a list, e.g. paramters = [energies, const, power]).
            Default : None
            
    parameters : list
            List representing the inputs a photon model function, if a function is provided, excludeing the energies the spectrum is over.
            Default : None

    srm : matrix/array
            Spectral response matrix.
            Default : None
            
    Returns
    -------
    A model count spectrum.
    '''

    ## if parameters is None then assume the photon_model input is already a spectrum to test, else make the model spectrum from the funciton and parameters
    if type(parameters) == type(None):
        photon_spec = photon_model
    else:
        photon_spec = photon_model(energies, *parameters)
    
    model_cts_matrix = np.array([srm[r, :] * photon_spec[r] for r in range(len(photon_spec))])
    model_cts_spectrum = model_cts_matrix.sum(axis=0) # sum the rows together
    
    return model_cts_spectrum
