'''
Functions to go in here (I think!?):
	KC: 19/12/2018, ideas-
	~getTimeAverage()
        ~ anything that helps with data handling or extraction, not really plotting, file working, or image making.
'''
from . import file_working

import numpy as np
import sunpy
import sunpy.map
import datetime
import os
import matplotlib
import sunpy.cm
from astropy.coordinates import SkyCoord
import astropy.units as u
from skimage import future

#make a light curve
def make_lightcurve(directory, bottom_left, top_right, mask=None):
    """Takes a directory and coordinates for a region in arcseconds returns start time of the light curve, times and 
    DN per second per pixel.
    
    Parameters
    ----------
    directory : str
            The directory that holds the .fits files of the observation, e.g. /home/.../. Can now take multiple directories, 
            just make sure they are in order, e.g. [dir1,dir2,...]. The data must be prepped.
    
    bottom_left : 1-d array
            The bottom left coordinates, [x,y] (as floats), for the light curve region in arcseconds.
            
    top_right : 1-d array
            The top right coordinates, [x,y] (as floats), for the light curve region in arcseconds.

    mask : array
            An array the same size as the region between bottom_left and top_right that has 1s where the feature you want a 
            lightcurve of is and 0s everywhere else (can use the draw_mask() funciton to create this mask over the same region).
            
    Returns
    -------
    Array of times (as datetime objects) and light curve values in input_unit per pixel (AIA).
    -AND/OR-
    Dictionary of times and input_unit per pixel seperated by filter combo and exposure time (XRT).
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
    no_of_files = len(directory_with_files)

    lc_values = [] #average value for selected area: DN/s/pix
    lc_times = []
    lc_values_xrt = {} #average value for selected area: DN/s/pix
    lc_times_xrt = {}
    for d,f in enumerate(directory_with_files):
        aia_map = sunpy.map.Map(f)
        times = aia_map.meta['date_obs']

        if aia_map.meta['instrume'][:3] == 'AIA': #if the files are aia then define the instrument and obs time
            map_type = 'AIA'
            obs_time = datetime.datetime.strptime(times, '%Y-%m-%dT%H:%M:%S.%fZ') 
        elif aia_map.meta['instrume'] == 'XRT': #if the files are xrt then define the instrument, filter combo and obs time
            map_type = 'XRT'     
            xrt_filter = aia_map.meta['ec_fw1_'] + '_' + aia_map.meta['ec_fw2_']  
            key = 'filter' + xrt_filter + '_exptime' + str(aia_map.meta['exptime']).replace('.','-')
            obs_time = datetime.datetime.strptime(times, '%Y-%m-%dT%H:%M:%S.%f')
        
        bl = SkyCoord(bottom_left[0]*u.arcsec, bottom_left[1]*u.arcsec, frame=aia_map.coordinate_frame)
        tr = SkyCoord(top_right[0]*u.arcsec, top_right[1]*u.arcsec, frame=aia_map.coordinate_frame)
        
        aia_submap_lc = aia_map.submap(bl,tr)

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
        else:
            print('The data either: isn\'t from the AIA or XRT, or it has not been prepped.')
            return

        del aia_submap_lc

        # if a mask is provided of teh region with 1 in the desired pixels and 0s everywhere else
        if type(mask) == type(None):
            ave_value = np.sum( np.array(t_norm_data) * np.array(mask) ) / np.sum(np.array(mask))
        else:
            ave_value = np.sum(np.array(t_norm_data)) / ((np.shape(np.array(t_norm_data))[0] * \
                                              np.shape(np.array(t_norm_data))[1]))

        if map_type == 'XRT': #xrt need keys to seperate different filter combos and exptimes
            if key in lc_values_xrt:
                lc_values_xrt[key].append(ave_value)
                lc_times_xrt[key].append(obs_time)
            elif key not in lc_values_xrt:
                lc_values_xrt[key] = [ave_value]
                lc_times_xrt[key] = [obs_time]
        elif map_type == 'AIA':
            lc_times.append(obs_time)
            lc_values.append(ave_value)

        print(f'\r[function: make_lightcurve()] Looked at {d+1} files of {no_of_files}.', end='')

    if map_type == 'AIA':
        return lc_times, lc_values
    elif map_type == 'XRT': 
        return lc_times_xrt, lc_values_xrt


def getTimeAverage(name=None, time_ranges=None, lightcurve_data=None):
    """Given a list of datetime times then the average value of lightcurve_data will be found between those times.
    
    Parameters
    ----------
    name : Str
        The name given to the data set.
        Defualt: None

    time_ranges : list of datetimes
        The times that you want the average value in between.
        Defualt: None

    lightcurve_data : dict
        A dictionary in the form {'times': [...] ,'data': [...]}.
        Defualt: None
            
    Returns
    -------
    A list of average values of lightcurve_data between the times specified.
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
    
    # find average values in the time periods indicated
    ave = []
    for tr in range(len(time_ranges)-1): 
        vals = []
        for c,t in enumerate(lightcurve_data['times']):
            
            # the times within the time range, add corresponding values into the vals list to be averaged
            if time_ranges[tr] < t <= time_ranges[tr+1]:
                vals.append(lightcurve_data['data'][c])
        ave.append(np.average(vals))
    
    # return the average lightcurve value between the time range values
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
    mask = future.manual_lasso_segmentation(array)

    if type(save_mask) != type(None):
        np.save(save_mask)

    return mask
