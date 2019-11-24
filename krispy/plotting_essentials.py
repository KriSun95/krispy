'''
Functions to go in here (I think!?):
    KC: 01/12/2018, ideas-
    ~make_lightcurve()      <
    ~dt_to_md()             <
    ~stepped_plot()         <
    ~cmap_midcolours()      <   
    ~iron_18_title()
    ~plot_lightcurve()

    KC: 18/12/2018, added-
    ~make_lightcurve() NOW REMOVED AND IN DATA_HANDLING.PY
    ~dt_to_md()
    ~stepped_lightcurve()

    KC: 19/12/2018, added-    
    ~cmap_midcolours()
'''

import numpy as np
import sunpy
import sunpy.map
import datetime
import os
import matplotlib.dates as mdates
import matplotlib
import sunpy.cm
from astropy.coordinates import SkyCoord
import astropy.units as u

'''
Alterations:
    KC: 05/02/2019 - cmap_midcolours() now has the 'Purples' colour pallet assigned to iron 16 (FeXVI).
    KC: 21/03/2019 - make_lightcurve() now can take multiple directories, and deal with both AIA and XRT prepped data.
                   - cmap_midcolours() now has sdoaia335 and hinodexrt added. 
'''

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

    #05/02/2019: ~colourbar used for iron 16 is added as 'Purples'

    #default/standard colourmaps with appropriate names for their use 
    cmap_names = ['sdoaia94', 'sdoaia131', 'sdoaia171', 'sdoaia193', 'sdoaia211', 'sdoaia335', 'hinodexrt'] #default colourmaps

    cmap_dict ={}

    for n in cmap_names:
        cmap = matplotlib.cm.get_cmap(n)
        cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=10) #index the colours from 0 to 10 throughout the colourmap
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)

        colorval = scalarMap.to_rgba(5) #pick the colour in the middle

        cmap_dict.update({n:colorval}) 
        
    #standard colourmaps given custom names to be used for plotting etc.
    custom_cmaps = {'sdoaiaFeXVIII':'Blues', 'sdoaiaFeXVI':'Purples'} #custom names for standard colourmaps
    
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
