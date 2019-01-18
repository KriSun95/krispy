'''
Functions to go in here (I think!?):
	KC: 01/12/2018, ideas-
	~make_lightcurve()		<
	~dt_to_md()				<
	~stepped_plot()			<
	~cmap_midcolours()      <   
	~iron_18_title()
	~plot_lightcurve()

	KC: 18/12/2018, added-
	~make_lightcurve()
	~dt_to_md()
	~stepped_lightcurve()
    KC: 19/12/2018, added-	
    ~cmap_midcolours()
'''
from . import file_working

import numpy as np
import sunpy
import sunpy.map
import datetime
import os
import matplotlib.dates as mdates
import matplotlib
import sunpy.cm

#make a light curve
def make_lightcurve(directory, bottom_left, top_right):
    """Takes a directory and coordinates for a ragion in arcseconds returns start time of the light curve, times and 
    DN per second per pixel.
    
    Parameters
    ----------
    directory : str
            The directory that holds the .fits files of the observation, e.g. /home/.../.
    
    bottom_left : 2-d array
            The bottom left coordinates, [x,y] (as floats), for the light curve region in arcseconds.
            
    top_right : 2-d array
            The top right coordinates, [x,y] (as floats), for the light curve region in arcseconds.
            
    Returns
    -------
    Start time of the observation as well as an array of times (as datetime objects) and light curve values in
    input_unit per pixel.
    """
    files = os.listdir(directory)
    files = file_working.only_fits(files)
    files.sort()
    
    
    lc_values = [] #average value for selected area: DN/s/pix
    lc_times = [] #times for each average value
    tstart = []
    for f in files:
        aia_map = sunpy.map.Map(directory + f)
        
        times = aia_map.meta['date_obs']
        obs_time = datetime.datetime.strptime(times, '%Y-%m-%dT%H:%M:%S.%fZ') 
        if lc_times == []:
            tstart.append(obs_time.strftime('%Y/%m/%d %H:%M:%S')) #1st time converted back to a string for easier use
        lc_times.append(obs_time)
        
        bl = SkyCoord(bottom_left[0]*u.arcsec, bottom_left[1]*u.arcsec, frame=aia_map.coordinate_frame)
        tr = SkyCoord(top_right[0]*u.arcsec, top_right[1]*u.arcsec, frame=aia_map.coordinate_frame)
        
        aia_submap_lc = aia_map.submap(bl,tr)
        
        ave_value = np.sum(np.array(aia_submap_lc.data)) / ((np.shape(np.array(aia_submap_lc.data))[0] * \
                                              np.shape(np.array(aia_submap_lc.data))[1])) 
        #averages all pixel values within the boxed region
        
        lc_values.append(ave_value)
    
    return tstart, lc_times, lc_values


#change a list of datetime objects to matplotlib dates
def dt_to_md(dt_array):
    """Takes a datetime list and returns the times in a matplotlib date format.
    
    Parameters
    ----------
    dt_array : list/array
            A list/array of datetime.datetime() objects.
            
    Returns
    -------
    Matplotlib date list.
    """
    new_array = np.zeros(len(dt_array))
    for c, d in enumerate(dt_array): #allows the index of each entry to be used as well as the entry itself
        plt_date = mdates.date2num(d)
        new_array[c] = plt_date
    return new_array



# make a stepped lightcurve
def stepped_lightcurve(x, y, inc_edges=True):
    """Takes an x and y input, duplicates the x values and y values with the offset as to produce a new x and y which 
    will produce a stepped graph once all the scatter points are plotted.
    
    Parameters
    ----------
    x : 1-d list/array
            This is the original set of x values.
    
    y : 1-d list/array
            This is the original set of y values.
            
    inc_edges : bool
            This determines whether the ends should go from their value to zero (True) or stop where they are (False).
            Default: True
            
    Returns
    -------
    New x and y values that, when plotted, will produce a stepped graph. Can be used to represent binning along the x
    axis.
    """
    new_x = np.array(np.zeros(2*len(x)))
    new_y = np.array(np.zeros(2*len(y)))
    for i in range(len(x)): #x and y should be the same length to plot anyway
        if i == 0: #start with the 1st and 2nd x value having the same y.
            new_x[i] = x[i]
            new_y[2*i], new_y[2*i+1] = y[i], y[i]
        elif i == len(x)-1: #the last new_x should be one beyond the last x as this value for the start of its bin
            new_x[2*i-1], new_x[2*i], new_x[2*i+1] = x[i], x[i], x[i]+(x[i]-x[i-1])
            new_y[2*i], new_y[2*i+1] = y[i], y[i]
            break
        else: #else keep the pattern going that two adjacent x's should share a y
            new_x[2*i-1], new_x[2*i] = x[i], x[i]
            new_y[2*i], new_y[2*i+1] = y[i], y[i]
    if inc_edges == True: #create first and last coordinates to have a new_y of zero
        new_x = np.insert(new_x, 0, [new_x[0]])
        new_x = np.append(new_x,[new_x[-1]])
        new_y = np.insert(new_y, 0, [0])
        new_y = np.append(new_y,[0])
    return new_x, new_y



#central colours of colourmaps
def cmap_midcolours(**kwargs):
    """Can take multiple name = 'colourmap' and find its centre colour to be used in line colours for plotting 
    graphs.
    
    Parameters
    ----------
    **kwargs : name equals an assigned colourmap
            If a new colourmap mid-colour is needed quickly then a name can be given to it and then referenced to 
            as such later.
            
    Returns
    -------
    A dictionary of rgba values (between 0 and 1) for the middle of given colourmaps.
    """
    #default/standard colourmaps with appropriate names for their use 
    cmap_names = ['sdoaia94', 'sdoaia171', 'sdoaia211'] #default colourmaps

    cmap_dict ={}

    for n in cmap_names:
        cmap = matplotlib.cm.get_cmap(n)
        cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=10) #index the colours from 0 to 10 throughout the colourmap
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)

        colorval = scalarMap.to_rgba(5) #pick the colour in the middle

        cmap_dict.update({n:colorval}) 
        
    #standard colourmaps given custom names to be used for plotting etc.
    custom_cmaps = {'sdoaiaFeXVIII':'Blues'} #custom names for standard colourmaps
    
    for c_cmap_name, std_cmap in custom_cmaps.items():
        cmap = matplotlib.cm.get_cmap(std_cmap)
        cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=10) #index the colours from 0 to 10 throughout the colourmap
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)

        colorval = scalarMap.to_rgba(5) #pick the colour in the middle

        cmap_dict.update({c_cmap_name:colorval})
    
    #colourmaps to be given with a name, e.g. nustar_2to4='red',and used as such
    for key, value in kwargs.items():
        cmap = matplotlib.cm.get_cmap(value)
        cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=10) #index the colours from 0 to 10 throughout the colourmap
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)

        colorval = scalarMap.to_rgba(5) #pick the colour in the middle

        cmap_dict.update({key:colorval}) 
            
            
    return cmap_dict #a dictionary with key names and the corresponding rgba values    