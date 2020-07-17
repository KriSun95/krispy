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
from skimage import future

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
            
    top_right : 1-d array
            The top right coordinates, [x,y] (as floats), for the light curve region in arcseconds.

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
        elif 'HMI' in aia_submap_lc.meta['instrume']:
            if 'history' in aia_submap_lc.meta:
                t_norm_data = aia_submap_lc.data
        else:
            print('The data either: isn\'t from the AIA or XRT or HMI, or it has not been prepped.')
            return

        del aia_submap_lc

        # if a mask is provided of the region with 1 in the desired pixels and 0s everywhere else
        if (type(isHMI) == type(None)) and (map_type != 'HMI'):
            if type(mask) != type(None):
                ave_value = np.sum( np.array(t_norm_data) * np.array(mask) ) / np.sum(np.array(mask))
            else:
                ave_value = np.sum(np.array(t_norm_data)) / ((np.shape(np.array(t_norm_data))[0] * \
                                              np.shape(np.array(t_norm_data))[1]))
        elif (map_type == 'HMI') and (type(isHMI) != type(None)) and (type(isHMI)==type(1) or type(isHMI)==type(1.0)):
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


def getTime(name=None, time_ranges=None, lightcurve_data=None, get='average'):
    """Given a list of datetime times then the average, sum, or values value of lightcurve_data will be found between those times.
    
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

    get : Str
        Decide whether you want the 'average', 'sum', 'values' back over the time range.
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
    
    # find average values in the time periods indicated
    ave = []
    for tr in range(len(time_ranges)-1): 
        vals = []
        for c,t in enumerate(lightcurve_data['times']):
            
            # the times within the time range, add corresponding values into the vals list to be averaged
            if time_ranges[tr] < t <= time_ranges[tr+1]:
                vals.append(lightcurve_data['data'][c])
        if get == 'average':
            ave.append(np.average(vals))
        elif get == 'sum':
            ave.append(np.sum(vals))
        elif get == 'values':
            ave.append(vals)
    
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

    # avoid maps being plotted upside down
    matplotlib.pyplot.rcParams["image.origin"] = "lower"

    mask = future.manual_lasso_segmentation(array)

    if type(save_mask) != type(None):
        np.save(save_mask, mask)

    return mask


## get boundaries for lasso region
def draw_region_boundaries(region):
    """Given an array/mask, this will produce the x,y lists to plot the boundary of the one region given in the mask.
    
    Parameters
    ----------
    region : np.array
        The array/mask where the selected region is all 1s and 0s everywhere else (can only manage one region at the moment).
            
    Returns
    -------
    The x-position (columns) and y-position (rows) lists for the boundary for the given region.
    """

    # rows (y) and columns (x) for all mask entries that are 1
    allOnes = np.where(region==1)

    ## inColumns = [ [column, [rows in this column]], ... ] (this will have duplicate entries though)
    inColumns = [[col , allOnes[0][np.where(allOnes[1]==col)]] for col in allOnes[1]] # rows in column

    # remove duplicates, and find min/max row for the edge
    removedCols=[]
    bot_cont = []
    top_cont = []
    for c in inColumns:
        ## if we haven't come across the column before, register it and continue
        if c[0] not in removedCols:
            removedCols.append(c[0])
            if np.min(c[1]) == np.max(c[1]):
                ## if the column only has one row then just add it to the bottom 
                bot_cont.append([c[0], np.max(c[1])])
            else:
                ## else add the bottom and top boundary entries into the correct list
                bot_cont.append([c[0], np.min(c[1])])
                top_cont.append([c[0], np.max(c[1])])

    colsb, rowsb = np.array(bot_cont)[:, 0], np.array(bot_cont)[:, 1] # rows and columns for bottom boundary line
    ## order columns and rows in the order of increasing column
    bot_ordered = [[c,r] for c,r in sorted(zip(colsb,rowsb))] 

    colst, rowst = np.array(top_cont)[:, 0], np.array(top_cont)[:, 1]
    ## bottom goes left-to-right so top needs to be ordered from right-to-left
    top_ordered = [[c,r] for c,r in sorted(zip(colst,rowst),  reverse = True)] 

    allIn = np.array([*bot_ordered, *top_ordered]) # combine bottom and top boundary lines
    ## seperate columns and rows, and add first entry to the end (so it loops back to the start)
    return [*allIn[:, 0], allIn[:, 0][0]], [*allIn[:, 1], allIn[:, 1][0]] 


# a function to try and guess what time format you give it
def getTimeFromFormat(timeString, **kwargs):
    """Give it a string of a date, it will return the UTC datetime object for it.
    
    Parameters
    ----------
    timeString : str
        The date as a string. Format must be in the date_format_defualts

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
                            "fmt8":'%Y-%m-%d, %H:%M:%S', "fmt9":'%Y-%m-%d %H:%M:%S'}
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