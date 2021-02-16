'''
Functions to go in here (I think!?):
	KC: 19/12/2018, ideas-
	~getTimeAverage()
        ~ anything that helps with data handling or extraction, not really plotting, file working, or image making.
'''
from . import file_working
from . import contour

import numpy as np
import sunpy
import sunpy.map
import datetime
import os
import matplotlib
#import sunpy.cm # replaced by line below for sunpy >v1.0
import sunpy.visualization.colormaps
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from skimage import future
from skimage.transform import resize
from scipy.ndimage import convolve
import csv
from os import path

#make a light curve
def make_lightcurve(directory, bottom_left, top_right, time_filter=None, mask=None, isHMI=None):
    """Takes a directory and coordinates for a region in arcseconds returns start time of the light curve, times and 
    DN per second per pixel.
    
    Parameters
    ----------
    directory : str
            The directory that holds the .fits files of the observation, e.g. /home/.../. Can now take multiple directories, 
            just make sure they are in order, e.g. [dir1,dir2,...]. The data must be prepped.
    
    bottom_left : 1-d array
            The bottom left coordinates, [x,y] (as floats), for the light curve region in arcseconds. 
            **If a mask is given then this is the bottom left [x,y] of the field-of-view for the mask.
            
    top_right : 1-d array
            The top right coordinates, [x,y] (as floats), for the light curve region in arcseconds.
            **If a mask is given then this is the bottom left [x,y] of the field-of-view for the mask.

    time_filter : list, 2 strings
            If you provide a directory but only want a lightcurve made from a certain time range, e.g. 
            ["%Y/%m/%d, %H:%M:%S", "%Y/%m/%d, %H:%M:%S"].
            Defualt: None

    mask : array
            An array the same size as the region between bottom_left and top_right that has 1s where the feature you want a 
            lightcurve of is and 0s everywhere else (can use the draw_mask() funciton to create this mask over the same region, 
            where the 'bottom_left' and 'top_right' arguments become the region of the mask).
            Defualt: None

    isHMI : int, float, or None
            This number identifies what values should be summed for an HMI lightcurve. If >=0 then only fluxes greater than this value 
            are summed, if <0 then only fluxes less than this value are summed. Each entry is return with the number of pixels that 
            contributed to its sum. 
            Defualt: None
            
    Returns
    -------
    Array of times (as datetime objects) and light curve values in input_unit per pixel (AIA).
    -AND/OR-
    Dictionary of times (as datetime objects) and input_unit per pixel seperated by filter combo and exposure time (XRT).
    -AND/OR-
    Array of times (as datetime objects) and total + or - Gauss in the region with number of contributing pixels (HMI).
    """

    if type(directory) == str:
        directory = [directory] # no point in writing out the loop below twice
        
    if type(directory) == list: # directory lists must be in time order for now, hopefully this will change (and look better with *args?)
        directory_with_files = []
        for _d in directory:
            _aia_files = os.listdir(_d)
            _aia_files.sort()
            _aia_files = file_working.only_fits(_aia_files)
            _directory_with_files = [_d+f for f in _aia_files]
            directory_with_files += _directory_with_files

    if time_filter is not None:
        _q = contour.Contours()
        _file_times = _q.aia_file_times(aia_file_dir="", aia_file_list=directory_with_files)
        _in_time = _q.useful_time_inds(times_list=_file_times, time_interval=time_filter)
        directory_with_files = np.array(directory_with_files)[_in_time]

    no_of_files = len(directory_with_files)

    lc_values = [] #average value for selected area: DN/s/pix
    lc_times = []
    lc_values_xrt = {} #average value for selected area: DN/s/pix
    lc_times_xrt = {}
    for d,f in enumerate(directory_with_files):
        aia_map = sunpy.map.Map(f)
        times = aia_map.meta['date-obs']

        if aia_map.meta['instrume'][:3] == 'AIA': #if the files are aia then define the instrument and obs time
            map_type = 'AIA'
            obs_time = getTimeFromFormat(times)
        elif aia_map.meta['instrume'] == 'XRT': #if the files are xrt then define the instrument, filter combo and obs time
            map_type = 'XRT'     
            xrt_filter = aia_map.meta['ec_fw1_'] + '_' + aia_map.meta['ec_fw2_']  
            key = 'filter' + xrt_filter + '_exptime' + str(aia_map.meta['exptime']).replace('.','-')
            obs_time = getTimeFromFormat(times)
        elif 'HMI' in aia_map.meta['instrume']:
            map_type = 'HMI' 
            obs_time = getTimeFromFormat(times)
        
        bl = SkyCoord(bottom_left[0]*u.arcsec, bottom_left[1]*u.arcsec, frame=aia_map.coordinate_frame)
        tr = SkyCoord(top_right[0]*u.arcsec, top_right[1]*u.arcsec, frame=aia_map.coordinate_frame)
        
        aia_submap_lc = aia_map.submap(bl,top_right=tr)

        del aia_map

        #check that the data is normalised, aia and xrt tell you in different ways
        if aia_submap_lc.meta['instrume'][:3] == 'AIA' and aia_submap_lc.meta['lvl_num'] == 1.5:
            if aia_submap_lc.meta['exptime'] == 1.0:
                t_norm_data = aia_submap_lc.data
            else:
                t_norm_data = aia_submap_lc.data / aia_submap_lc.meta['exptime']
                #aia_submap_lc = sunpy.map.Map(t_norm_data, aia_submap_lc.meta)
        elif aia_submap_lc.meta['instrume'] == 'XRT' and 'XRT_PREP' in aia_submap_lc.meta['history']:
            if 'Normalized from' in aia_submap_lc.meta['history'] and 'sec --> 1.00 sec.' in aia_submap_lc.meta['history']:
                t_norm_data = aia_submap_lc.data
            else:
                t_norm_data = aia_submap_lc.data / aia_submap_lc.meta['exptime']
                #aia_submap_lc = sunpy.map.Map(t_norm_data, aia_submap_lc.meta)
        elif 'HMI' in aia_submap_lc.meta['instrume']:
            if 'history' in aia_submap_lc.meta:
                t_norm_data = aia_submap_lc.data
        else:
            print('The data either: isn\'t from the AIA or XRT or HMI, or it has not been prepped.')
            return

        del aia_submap_lc

        # check mask size. This shouldn't be any more out than 1 pixel in x and/or y from the mask being from different Sunpy v or something
        # or the mask could be from one instrument and you want the mask applied to another instrument with a different res.
        if type(mask) != type(None):
            if not np.array_equal(np.shape(t_norm_data), np.shape(mask)) and d==0:
                print('Needing to re-shape the mask to the map size. Re-shaping ', np.shape(mask), ' to ', np.shape(t_norm_data), '.')
                mask = maskFill(mask, np.shape(t_norm_data))

        # if a mask is provided of the region with 1 in the desired pixels and 0s everywhere else
        if (type(isHMI) == type(None)) and (map_type != 'HMI'):
            if type(mask) != type(None):
                emiss = np.array(t_norm_data) * np.array(mask)
                # np.sum( np.array(t_norm_data) * np.array(mask) ) / np.sum(np.array(mask))
            else:
                emiss = np.array(t_norm_data)
                # np.sum(np.array(t_norm_data)) / ((np.shape(np.array(t_norm_data))[0] * np.shape(np.array(t_norm_data))[1]))
            ave_value = np.mean(emiss[emiss>0])
        elif (map_type == 'HMI') and (type(isHMI) != type(None)) and (type(isHMI)==type(1) or type(isHMI)==type(1.0)):
            if type(mask) != type(None):
                # only want values in the mask, everything else is 0
                t_norm_data = np.array(t_norm_data) * np.array(mask)

            if isHMI >= 0:
                t_norm_data[t_norm_data < isHMI] = 0 # set everything less than this value to zero
            elif isHMI < 0:
                t_norm_data[t_norm_data > isHMI] = 0 # set everything greater than this value to zero
            ave_value = [np.sum(t_norm_data), np.count_nonzero(t_norm_data)] # return total Gauss (- or +) with the number of pixels that produced that total
        else:
            print('isHMI has to be set to a number if HMI data is provided (flux below negative inputs will be summed and flux about positive inputs will be summed). Otherwise it should be set to None')


        if map_type == 'XRT': #xrt need keys to seperate different filter combos and exptimes
            if key in lc_values_xrt:
                lc_values_xrt[key].append(ave_value)
                lc_times_xrt[key].append(obs_time)
            elif key not in lc_values_xrt:
                lc_values_xrt[key] = [ave_value]
                lc_times_xrt[key] = [obs_time]
        elif (map_type == 'AIA') or (map_type == 'HMI'):
            lc_times.append(obs_time)
            lc_values.append(ave_value)

        print(f'\r[function: make_lightcurve()] Looked at {d+1} files of {no_of_files}.        ', end='')

    if (map_type == 'AIA') or (map_type == 'HMI'):
        return lc_times, lc_values
    elif map_type == 'XRT': 
        return lc_times_xrt, lc_values_xrt


def getBetweenTime(name=None, time_ranges=None, lightcurve_data=None, get='average'):
    """Given a list of datetime times and values (i.e. {'times': [...] ,'data': [...]}) then the average, sum, or 
    values of lightcurve_data will be found between those times.
    
    Parameters
    ----------
    name : Str
        The name given to the data set.
        Defualt: None

    time_ranges : list of datetimes
        The times that you want the average value in between, e.g. [dt1, dt2, dt3, ...].
        Defualt: None

    lightcurve_data : dict
        A dictionary in the form {'times': [...] ,'data': [...]}.
        Defualt: None

    get : Str
        Decide whether you want the 'average', 'sum', 'values' back in between the times given.
        Defualt: 'average'
            
    Returns
    -------
    A list of average, sum, or values of lightcurve_data between the times specified.
    """
    
    # to keep track of what wavelength I'm on
    if type(name) != type(None):
        print('For '+name+': ')
        
    if type(time_ranges) == type(None):
        print('Need a list of time ranges in datetime objects.')
        return
    
    if type(time_ranges) == type(None):
        print('Need the lightcurve data as a dictionary, e.g. {\'times\':[dt_obj], \'data\':[...]}.')
        return
    
    # find average values, sum values, or values in the time periods indicated
    ave = []
    for tr in range(len(time_ranges)-1): 
        vals = []
        for c,t in enumerate(lightcurve_data['times']):
            
            # the times within the time range, add corresponding values into the vals list to be averaged, summed, or just returned
            if time_ranges[tr] < t <= time_ranges[tr+1]:
                vals.append(lightcurve_data['data'][c])
        if get == 'average':
            ave.append(np.average(vals))
        elif get == 'sum':
            ave.append(np.sum(vals))
        elif get == 'values':
            ave.append(vals)
    
    # e.g. return the average lightcurve value between the time range values
    # so len(averages) = len(time_ranges)-1
    return ave


def draw_mask(array, save_mask=None):
    """Given an array to draw a mask(s) area on it.
    
    Parameters
    ----------
    array : np.array
        The array with a feature on it you want to isolate with a mask.

    save_mask : Str
        String with the directory and filename of the mask you have created to be saved.
        Defualt: None
            
    Returns
    -------
    An array with 1s enclosed within the region you specified and 0s everywhere else.
    """

    # avoid maps being plotted upside down
    matplotlib.pyplot.rcParams["image.origin"] = "lower"

    mask = future.manual_lasso_segmentation(array)

    if type(save_mask) != type(None):
        np.save(save_mask, mask)

    return mask


## get boundaries for lasso region
def get_region_boundaries(region):
    """Given an array/mask, this will produce the x,y lists to plot the boundary of the regions given in the mask.
    
    Parameters
    ----------
    region : np.array
        The array/mask where the selected region is all 1s and 0s everywhere else.
            
    Returns
    -------
    The x-position (columns) and y-position (rows) lists for the boundary for the given region. 
    This output will not be ordered to plot as a line/loop
    """

    ## find the sum of the 8 neighbouring pixels 
    ## middle is 10 instead of 0 as I want the middle value to be >0 to start with 
    kernel = np.array([[1, 1,  1],
                       [1, 10, 1],
                       [1, 1,  1]])
    
    ## find sum of neightbour pixels plus 10x of pixel's value
    c = convolve(region, kernel, mode='constant')
    
    ## c[i]>10 means that pixel has a 1 (not a 0), c[i]<18 means not all of its neighbours have a 1 in them
    boundary = np.array([list(i) for i,n in np.ndenumerate(c) if c[i]>10 and c[i]<18])
    
    return boundary[:,1], boundary[:,0]


## if the mask is at a different resolution than the map. Is this because:
# 1. The map and mask have given slightly different shapes of matrices? Different versions of Sunpy for example?
def maskFill(mask, shape):
    """Given an array/mask, this will try to pad it with zeros sensibly (or remove rows/columns) to reach the given shape.
    
    Parameters
    ----------
    mask : np.array or list
        The array/mask where the selected region is all 1s and 0s everywhere else but is a different shape to the map to which it is being applied.

    shape: tuple, len=2
        The shape of the map to which the mask is being applied, e.g. (rows, columns).
            
    Returns
    -------
    The new, padded (or reduced) mask as a numpy array (right and top of the image is padded first then alternate, 
    important for odd number of pixel difference).
    """

    # find difference in mask and shape
    mask_size, mask = np.shape(mask), np.array(mask)
    delta_row, delta_col = shape[0]-mask_size[0], shape[1]-mask_size[1]

    # pad top row and right column first
    # fix rows
    if delta_row>0:
        # if more rows are needed
        for dr in range(delta_row):
            if dr%2==0:
                mask = np.concatenate((mask, np.array([0]*mask_size[1]).reshape(1,mask_size[1])), axis=0)
            else:
                mask = np.concatenate((np.array([0]*mask_size[1]).reshape(1,mask_size[1]), mask), axis=0)
    elif delta_row<0:
        for dr in range(abs(delta_row)):
            if dr%2==0:
                mask = mask[:-1, :]
            else:
                mask = mask[1:, :]
            
    # fix columns
    shifted_rowcol = np.shape(mask) # new rows has been added, this changed the padding for the columns/rows
    if delta_col>0:
        # if more columns are needed
        for dc in range(delta_col):
            if dc%2==0:
                mask = np.concatenate((mask, np.array([0]*shifted_rowcol[0]).reshape(shifted_rowcol[0], 1)), axis=1)
            else:
                mask = np.concatenate((np.array([0]*shifted_rowcol[0]).reshape(shifted_rowcol[0], 1), mask), axis=1)
    elif delta_col<0:
        for dc in range(abs(delta_col)):
            if dc%2==0:
                mask = mask[:, :-1]
            else:
                mask = mask[:, 1:]

    return mask

# 2. different instruments and so different resolutions
def maskResize(mask, shape):
    """Given an array/mask, this will try to resize it sensibly to the given shape.

    ******** This is not getting used at the moment but a function *like* this may be needed soon. ********
    
    Parameters
    ----------
    mask : np.array
        The array/mask where the selected region is all 1s and 0s everywhere else but is a different shape to the map to which it is being applied.

    shape: tuple, len=2
        The shape of the map to which the mask is being applied, e.g. (rows, columns).
            
    Returns
    -------
    The new, resized mask as a numpy array.
    """

    # make it very obvious where the mask is mathematically
    mask[mask==0] = -1

    # now resize the mask to the correct shape and convert back to 1s and 0s
    mask_resized = resize(mask, (shape[0], shape[1]))
    mask_resized[mask_resized > 0.5*mask_resized.max()] = 1
    mask_resized[mask_resized <= 0.5*mask_resized.max()] = 0
    return mask_resized


# function to increase the area slection from the draw_mask function
def maskPadOnes(maskToIncrease, pixelPad=1):
    """Increases the size of the region of ones in the mask by padding around the already existing ones by the pixelPad amount.
    
    Parameters
    ----------
    mask : np.array
        The array/mask where the selected region is all 1s and 0s everywhere else.

    pixelPad: int
        The number of pixels to pad the already existing ones with.
            
    Returns
    -------
    Array with the regions of ones increased.

    Example
    -------
    mask = [[0,0,0,0], 
            [0,1,0,0], 
            [0,0,0,0]]
    maskPadOnes(mask, pixelPad=1)

    >>> [[1,1,1,0], 
         [1,1,1,0], 
         [1,1,1,0]]
    """

    mask = np.copy(maskToIncrease)

    kernel = np.array([[1, 1,  1],
                       [1, 10, 1],
                       [1, 1,  1]])
    
    for r in range(pixelPad):
        ## find sum of neightbour pixels plus 10x of pixel's value
        c = convolve(mask, kernel, mode='constant')
    
        ## 0<c[i]<10 means that pixel has a 0 but has a 1 in at least one contigous pixel
        boundary = np.array([list(i) for i,n in np.ndenumerate(c) if c[i]<10 and c[i]>0])

        for br, bc in zip(boundary[:,0], boundary[:,1]):
            mask[br, bc] = 1

    return mask


# function to increase the area slection from the draw_mask function
def maskUnPadOnes(maskToIncrease, pixelUnPad=1):
    """Decreases the size of the region of ones in the mask by setting the boundary pixels to 0 .
    
    Parameters
    ----------
    mask : np.array
        The array/mask where the selected region is all 1s and 0s everywhere else.

    pixelPad: int
        The number of times to remove the boundary pixels.
            
    Returns
    -------
    Array with the regions of ones decreases.

    Example
    -------
    mask = [[1,1,1,0], 
            [1,1,1,0], 
            [1,1,1,0]]
    maskUnPadOnes(mask, pixelUnPad=1)

    >>> [[0,0,0,0], 
         [0,1,0,0], 
         [0,0,0,0]]
    """

    mask = np.copy(maskToIncrease)

    kernel = np.array([[1, 1,  1],
                       [1, 10, 1],
                       [1, 1,  1]])
    
    for r in range(pixelUnPad):
        ## find sum of neightbour pixels plus 10x of pixel's value
        c = convolve(mask, kernel, mode='constant')
    
        ## 10<c[i]<18 means that pixel has a 1 but has a 0 in at least one contigous pixel, i.e. it is a boundary pixel
        boundary = np.array([list(i) for i,n in np.ndenumerate(c) if c[i]<18 and c[i]>10])

        for br, bc in zip(boundary[:,0], boundary[:,1]):
            mask[br, bc] = 0

    return mask



# if you have multiple masks and want to combine them
def combineMasks(*args):
    """Combines the multiple masks you give it and gives back an array with them all combined.
    
    Parameters
    ----------
    *args : np.arrays
        The masks, all need to be the same shape as least.
            
    Returns
    -------
    Array with the regions of ones.

    Example
    -------
    mask1 = [[0,0,0],    mask2 = [[1,0,0] 
             [0,1,1],             [0,1,0]
             [0,1,0]]             [1,1,0]]
    combineMasks(mask1, mask2)
    >>> [[1,0,0],
         [0,1,1],
         [1,1,0]] 
    """
    added = sum(list(args))
    added[added>0] = 1
    return added


# a function to try and guess what time format you give it
def getTimeFromFormat(timeString, **kwargs):
    """Give it a string of a date, it will return the UTC datetime object for it.
    
    Parameters
    ----------
    timeString : str
        The date as a string. Format must be in the date_format_defualts.

    **kwargs : name=datetime format
        Your own datetime formats.
            
    Returns
    -------
    A datetime object in UTC (hopefully) of the string you gave it.
    """
    date_format_defualts = {"fmt0":'%Y-%m-%dT%H:%M:%S.%fZ', "fmt1":'%Y-%m-%dT%H:%M:%S.%f', 
                            "fmt2":'%Y/%m/%dT%H:%M:%S.%f', "fmt3":'%Y/%m/%dT%H:%M:%S.%fZ', 
                            "fmt4":'%Y/%m/%d %H:%M:%S.%f', "fmt5":'%Y/%m/%d, %H:%M:%S.%f', 
                            "fmt6":'%Y/%m/%d %H:%M:%S', "fmt7":'%Y/%m/%d, %H:%M:%S', 
                            "fmt8":'%Y-%m-%d, %H:%M:%S', "fmt9":'%Y-%m-%d %H:%M:%S', 
                            "fmt10":'%Y%m%d_%H%M%S'}
    date_formats = {**date_format_defualts, **kwargs}

    for f in date_formats.keys():
        # run through formats
        try:
            time = datetime.datetime.strptime(timeString, date_formats[f]) 
            # turn time into an "aware" object, don't keep as a "naive" one
            time = time.replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue

        return time


def readAIAresponse(csvFile):
    """Obtains the SDO/AIA responses from a .csv file.
    
    Parameters
    ----------
    csvFile : str
        The .csv file with the SDO/AIA responses in it. First line should be labels for each 
        column as a comma seperated string, e.g. row1=['logt, A94, A171'].
            
    Returns
    -------
    A dictionary with the responses where the keys are determined by the labels for each column.
    """
    
    with open(csvFile) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        resp = {}
        for c,row in enumerate(readCSV):
            if c == 0:
                for t in row[0].split(sep=", "):
                    resp[t]=[] # find and make the keys from the first row of the .csv file
            else:
                for a, key in enumerate(resp.keys()):
                    resp[key].append(float(row[a])) # match each key to its index in the row
        
    return resp


# a function to extract keywords from the output XSPEC files
def xspecParams(xspec_fits, *args):
    """Takes an XSPEC fits file and your guess at some keywords and returns those parameters 
    with some other (hopefully) useful information.
    
    Parameters
    ----------
    xspec_fits : str
        The fits file in question.
        
    *args : str
        The keywords (or general guess) at the parameters you want from the fits file. 
        E.g. "temp" will give temperature ("kt#" in the file) but if you want a specific 
        parameter/temperature then can have "kt1".
            
    Returns
    -------
    A dictionary with the keyword from the fits file as the keys, each with a list of 
    the values obtained from that keyword and the arg you provided which got that keyword.
    """
    
    # map easy to search for terms to the terms used in the xspec files
    xspec_words = []
    for a in args:
        if a.lower() in ["t", "temp", "temperature"] and "kt" not in xspec_words:
            xspec_words.append("kt")
        elif a.lower() in ["norm", "normalisation"] and "norm" not in xspec_words:
            xspec_words.append("norm")
        elif a.lower() in ["break", "ebreak", "e_break"] and "break" not in xspec_words:
            xspec_words.append("break")
        elif a.lower() in ["photonindex", "phoindx", "index"] and "phoindx" not in xspec_words:
            xspec_words.append("phoindx")
        else:
            xspec_words.append(a.lower())
    
    # what keywords are in the fits file
    hdul = fits.open(xspec_fits)
    keys = list(dict(hdul[1].header).values())
    values = hdul[1].data
    hdul.close()
    
    # find the keywords your words refer to
    findings = []
    for w in xspec_words:
        for x in keys:
            if w in str(x).lower():
                findings.append([x, w])
    
    # now find the values of the keywords found, include the conversion factors for the EM and T
    output = {"emfact":3.5557e-42, "kev2mk":0.0861733}
    for f in findings:
        output[f[0]] = [values[f[0]][0], f[1]]
    
    return output



# a function to read in frames already generated from a folder or create the frames
def frames(fitsDirectory=None, time_range=None, where=None, submap=None, framesFolder=None, framesFile=None, overwrite=False, needs_prepping=False):
    """Reads or creates the frames you want from a directory. Reading takes priority over creating the files again unless 
    overwrite=True.
    
    Parameters
    ----------
    fitsDirectory : list of str
        Directory for the fits files from where the frames are to come from. E.g. fitsDirectory=["dir1", "dir2"]
        
    time_range : list of lists
        For time filtering for the files in the fitsDirectory, e.g. time_range=[["yyyy/mm/dd, HH:MM:SS", "yyyy/mm/dd, HH:MM:SS"]].

    where : str
        Should the frame be the middle fram of the time range or the average etc. Options are "start", "middle", "end", "average", 
        or a time string "yyyy/mm/dd, HH:MM:SS". Alternatively it could be a list of these for each resulting frame.

    submap : list of lists
        Take a submap of the files in the fitsDirectory, e.g. submap=[[bottomLeftX, bottomLeftY, topRightX, topRightY]].

    framesFolder : str
        If this has already been run then just want the directory they are in.

    framesFile : 
        If this has already been run then what is the name of the file.

    overwrite : bool
        Set to true if you want the frames to be calculated again regardless if they are there or not.

    needs_prepping : bool or list of bool
        Set to True if the fits files need prepping (uses aiapy).
        Default: False
            
    Returns
    -------
    The frames, the frame's maxima and minima.

    Example
    -------
    # read, if not there then create
    frames(fitsDirectory=["dir1"], time_range=[["yyyy/mm/dd, HH:MM:SS", "yyyy/mm/dd, HH:MM:SS"]], where="average", submap=[[blx,bly,trx,try]], framesFolder="saveDir", framesFile="saveFile")

    # read
    frames(framesFolder="saveDir", framesFile="saveFile")

    # create even if the file already exists
    frames(fitsDirectory=["dir1"], time_range=[["yyyy/mm/dd, HH:MM:SS", "yyyy/mm/dd, HH:MM:SS"]], where="average", submap=[[blx,bly,trx,try]], framesFolder="saveDir", framesFile="saveFile", overwrite=True)
    """
    
    # what do we need
    create = ((type(fitsDirectory)!=type(None)) and (type(time_range)!=type(None)) and (type(where)!=type(None)) and (type(submap)!=type(None)) and (type(framesFolder)!=type(None)) and (type(framesFile)!=type(None)))
    read = ((type(framesFolder)!=type(None)) and (type(framesFile)!=type(None)))
    if create or read:
        pass
    else:
        print("Need either fitsDirectory, time_range, and where OR framesFolder and framesFile.")
        return

    # map easy to search for terms to the terms used in the xspec files
    if path.exists(framesFolder+"/"+framesFile) and overwrite==False:
        frames_and_max = np.load(framesFolder+"/"+framesFile, allow_pickle=True)
        # need to index [inds] here when the whole list of possible events is being used
        aia_frames, aia_maxima, aia_minima = frames_and_max[0], frames_and_max[1], frames_and_max[2]
    else:
        aia_frames =[]
        aia_maxima = []
        aia_minima = []
        for d in range(len(fitsDirectory)):
            aiamap = contour.Contours()

            aia_file_list = np.array(os.listdir(fitsDirectory[d]))
                      
            times_list = aiamap.aia_file_times(fitsDirectory[d], aia_file_list=aia_file_list)

            good_indices = aiamap.useful_time_inds(times_list, time_interval=time_range[d])

            files_in_trange = aia_file_list[good_indices]


            if type(where)==list:
                needs_prepping = [needs_prepping]*len(where) if type(needs_prepping)!=list else needs_prepping
                background_map = aiamap.which_background(fitsDirectory[d], files_in_trange, where=where[d], needs_prepping=needs_prepping[d])
            else:
                background_map = aiamap.which_background(fitsDirectory[d], files_in_trange, where=where, needs_prepping=needs_prepping)

            aia_frame, _ = aiamap.aia_frame(background_map, submap=submap[d])
            aia_frames.append(aia_frame)
            aia_maxima.append(np.max(aia_frame.data))
            aia_minima.append(np.min(aia_frame.data))

            print(f'\rDone {d+1} maps of {len(fitsDirectory)}.        ', end='')

        np.save(framesFolder+"/"+framesFile, [aia_frames, aia_maxima, aia_minima])

    return aia_frames, aia_maxima, aia_minima