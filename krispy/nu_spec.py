'''
Functions to go in here (I think!?):
    KC: 06/08/2019, added-
    ~grppha/minimum count grouping check
    ~NuSTAR spectral stuff
'''

import sys
from os.path import *
import os
import numpy as np
from astropy.io import fits


def grppha_min_check(pha_file, group_min=None, print_tries=False):
    ''' Takes a .pha file, loads in the counts, and checks the bins left over from grouping the bins with a minimum 
    value.
    
    Parameters
    ----------
    pha_file : Str
            String for the .pha file of the spectrum under investigation.
            
    group_min : Int
            The minimum number of counts allowed in a bin. This input is a starting number and the is checked 
            incrementally.
            Default: None
            
    print_tries : Bool
            States whether the result of every try of 'group_min' should be displayed (True) or only the final 
            result (False, default).
            Default: False
            
    Returns
    -------
    The minimum bin number that gives zero counts left over at the end, if it exists, else None. 
    '''
    
    if type(group_min)!=int or group_min<=0: 
        print('The \'group_min\' parameter must be an integer and > 0.')
        return
    
    # grppha groups in counts, not counts s^-1 or anything
    n = 1
    hdul = fits.open(file)
    data = hdul[n].data
    hdul.close()
    
    orig_counts = data['counts']
    orig_channel = data['channel']
    total_counts = np.sum(orig_counts)
    
    if type(group_min) == int:
        combin = [1] # just to establish the variable
        while len(combin) != 0:
            binned_counts = []
            binned_channel = []
            combin = []
            for c in range(len(orig_counts)):
                if orig_counts[c] >= group_min and len(combin) == 0:
                    binned_counts.append(orig_counts[c])
                    binned_channel.append(orig_channel[c])
                elif orig_counts[c] > 0:
                    combin.append(orig_counts[c])
                    if len(combin) == 1:
                        binned_channel.append(orig_channel[c])
                    if np.sum(combin) >= group_min:
                        binned_counts.append(np.sum(combin)) 
                        combin = []

            if print_tries == True:
                print('Group min: ', group_min, ' has counts left over: ', len(combin), ' of bins ', combin)
                
            if len(combin) != 0:
                group_min += 1
            elif group_min >= total_counts:
                print('The minimum group number being tried is the same as the total number of counts.')
                return
            else:
                print('Group minimum that works is: ', group_min)
                return group_min


def read_pha(file):
    ''' Takes a .pha file and extracts useful information from it.
    
    Parameters
    ----------
    file : Str
            String for the .pha file of the spectrum under investigation.
            
    Returns
    -------
    The counts, channel numbers, and the livetime for the observation. 
    '''

    hdul = fits.open(file)
    data = hdul[1].data
    header_FOR_LIVETIME = hdul[0].header
    hdul.close()

    return data['counts'], data['channel'], header_FOR_LIVETIME['LIVETIME']


def read_arf(file):
    ''' Takes a .arf file and extracts useful information from it.
    
    Parameters
    ----------
    file : Str
            String for the .arf file of the spectrum under investigation.
            
    Returns
    -------
    The low and high boundary of energy bins, and the ancillary response [cm^2].  
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
    The low and high boundary of energy bins, number of sub-set channels in the energy bin, starting index of each sub-set of channels, 
    number of channels in each sub-set, redistribution matrix [counts per photon]. 
    '''

    hdul = fits.open(file)
    data = hdul[2].data
    hdul.close()
    
    return data['energ_lo'], data['energ_hi'], data['n_grp'], data['f_chan'], data['n_chan'], data['matrix']


def col2arr_py(data):
    ''' Takes a list of parameters for each energy channel from a .rmf file and returns it in the correct format.
    
    Parameters
    ----------
    data : array/list-like object
            One parameter's array/list from the .rmf file.
            
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
        
    return chan_array


def vrmf2arr_py(data=None, no_of_channels=None, n_grp_list=None, f_chan_array=None, n_chan_array=None):
    ''' Takes redistribution parameters for each energy channel from a .rmf file and returns it in the correct format.
    
    Parameters
    ----------
    data : array/list-like object
            Redistribution matrix parameter array/list from the .rmf file. Units are counts per photon.
            Default : None
            
    no_of_channels : int
            Number of channels/ energy bins.
            Default : None

    n_grp_list : list
            Number of sub-set channels in that energy bin.
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
            ## are both inclusive sod rop the -1, i.e. was accum_nc[r+1]-1
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


def make_srm(rmf_array=(), arf_array=()):
    ''' Takes rmf and arf and produces the spectral response matrix fro NuSTAR.
    
    Parameters
    ----------
    rmf_array : numpy 2D array
            Array representing the redistribution matrix.
            Default : None
            
    no_of_channels : numpy 1D array/list
            List representing the ancillary response.
            Default : None
            
    Returns
    -------
    An array that is the spectral response (srm).
    '''
    
    if len(rmf_array) == 0 or len(arf_array) == 0:
        print('Need both RMF and ARF information to proceed.')
        return
    
    srm = np.array([rmf_array[r, :] * arf_array[r] for r in range(len(arf_array))]) # each energy bin row in the rmf is multiplied the arf value for the same energy bin
    return srm
