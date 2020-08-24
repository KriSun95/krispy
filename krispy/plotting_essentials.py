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
#import sunpy.cm # replaced by line below for sunpy >v1.0
import sunpy.visualization.colormaps
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import pickle

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
    custom_cmaps = {'sdoaiaFeXVIII':'GnBu', 'sdoaiaFeXVI':'Purples'} #custom names for standard colourmaps
    
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


def plotSDOlightcurves(instrument, directory="./", files=None, data_list=None, title="Lightcurves", nustardo_obj=None, samePlot=False, other_data=None):
    """Takes a directory and a list of (pickle) files of lightcurves and produces a plot with all the lightcurves plotted.
    
    Parameters
    ----------
    instrument : Str
            Either from 'AIA' or 'HMI' so far (for average magnetic field that contributed to the total Gauss then 'HMIPIXAVERAGE').

    directory : Str
            String for the directory where the pickle files reside.
            Default: "./"
    
    files : list of Strings
            Files of the pickled lightcurve data. This input takes priority over the data input.
            Default: None

    data_list : list of dictionaries
            List of dictionaries of the pickled lightcurve data.
            Default: None
            
    title : Str
            Title for the plot.
            Default: "Lightcurves"

    nustardo_obj : NustarDo Object
            If you want to plot colours for where there is a constant CHU combination and allow the x limits to be determined by the object.
            Default: None

    samePlot : Bool
            Set to True if you want all the curves to be plotted on the same axis.
            Default: False

    other_data : dictionary of length 2 dictionaries
            If you have your own lightcurves that you want tagged on at the end of the plot/included, 
            e.g. {"name":{"times":[dt_times], "units":[data]}} where "name" will label the axis and "units" provides y-axis label.
            Default: None
            
    Returns
    -------
    Displays a figure comprised of lightcurve subplots.
    """

    # use the function above to use the colours for AIA
    cmap_dict = cmap_midcolours()

    if other_data is not None:
        extra_plots = len(other_data)
    else:
        extra_plots = 0

    # manually set the number of plots to 1 if they are to all be plotted on the same axis
    if files is not None:
        n = len(files) + extra_plots if samePlot is False else 1
        ps = range(len(files))
    else:
        n = len(data_list) + extra_plots if samePlot is False else 1
        ps = range(len(data_list))
    
    fig, axs = plt.subplots(n,1,figsize=(16, 1.5*n+4), sharex=True)
    # make sure axs can still be indexed
    axs = axs if n>1 else [axs] #"samePlot is False" was the old statement
    fig.subplots_adjust(hspace=0.)

    for plot in ps:

        # load in each lightcurve and plot
        if files is not None:
            with open(directory+files[plot], "rb") as input_file:
                data = pickle.load(input_file)
        else:
        	data = data_list[plot] 

        name = list(data.keys())[0]

        plot = plot if samePlot is False else 0

        # set time labels for x-axis
        fmt = mdates.DateFormatter('%H:%M')
        tickTime = mdates.MinuteLocator(byminute=[0, 10, 20, 30, 40, 50], interval = 1)
        axs[plot].xaxis.set_major_formatter(fmt)
        axs[plot].xaxis.set_major_locator(tickTime)
        
        # if it's AIA lightcurves
        if instrument.upper() == 'AIA':
            if samePlot is False:
                axs[plot].plot(dt_to_md(data[name]['times']), data[name]['DN_per_sec_per_pixel'], color=cmap_dict[name])
                axs[plot].set_ylabel('DN s$^{-1}$ pix$^{-1}$', color=cmap_dict[name])
                axs[plot].tick_params(axis='y', labelcolor=cmap_dict[name])

                # set up twin axis to label each subplot
                twinx_ax = axs[plot].twinx()
                twinx_ax.set_ylabel(name)
                twinx_ax.yaxis.label.set_color(cmap_dict[name])
                twinx_ax.set_yticks([])
                twinx_ax.xaxis.set_major_formatter(fmt)
                twinx_ax.xaxis.set_major_locator(tickTime)
            elif samePlot is True:
                axs[plot].plot(dt_to_md(data[name]['times']), data[name]['DN_per_sec_per_pixel']/np.max(data[name]['DN_per_sec_per_pixel']), color=cmap_dict[name], label=name)
                axs[plot].set_ylabel('Normalized DN s$^{-1}$ pix$^{-1}$')
                axs[plot].legend(loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

            ylims = [0.99*min(data[name]['DN_per_sec_per_pixel']), 1.01*max(data[name]['DN_per_sec_per_pixel'])] if samePlot is False else [0, 1.05]
            axs[plot].set_ylim(ylims)

        # if it's HMI 'lightcurves'
        elif instrument[:3].upper() == 'HMI':
            gauss = np.array(data[name]['Gauss_and_numOfPix'])[:,0]
            # if it's negative Gauss then make it positive to plot for now and label it so
            g = -gauss if gauss[0] < 0 else gauss
            l = 'Neg. Mag. Flux' if gauss[0] < 0 else 'Pos. Mag. Flux'
            c = 'green' if gauss[0] < 0 else 'fuchsia'

            if instrument[3:].upper() == 'PIXAVERAGE':
                g = g / np.array(data[name]['Gauss_and_numOfPix'])[:,1]
                axs[plot].plot(data[name]['times'], g, color=c, label=l+'/contributing pix')
                axs[plot].set_ylabel('Gauss per contributing pixel')
            else:
                axs[plot].plot(data[name]['times'], g, color=c, label=l)
                axs[plot].set_ylabel('Total Gauss')
            axs[plot].set_ylim([0.99*min(g), 1.01*max(g)])
            axs[plot].legend()

        # plot CHU changes to match the NuSTAR plots?
        if nustardo_obj is not None and n>0:
            nustardo_obj.plotChuTimes(axis=axs[plot])
            # avoid plotting the chu changes over and over all on the same plot
            n = n if samePlot is False else -1

    if other_data is not None:
        for c, custom_name in enumerate(other_data.keys()):
            if samePlot is True:
                pass
                #axs[0].plot(dt_to_md(other_data[name]["times"]), data[name]['DN_per_sec_per_pixel']/np.max(data[name]['DN_per_sec_per_pixel']), label=name)
            else:
                plot = plot+c+1
                units = [unts for unts in other_data[custom_name].keys() if unts is not "times"]
                axs[plot].plot(other_data[custom_name]["times"], other_data[custom_name][units[0]], color="red", label=custom_name)
                axs[plot].set_ylabel(units[0])
                axs[plot].tick_params(axis='y')

                # set up twin axis to label each subplot
                twinx_ax = axs[plot].twinx()
                twinx_ax.set_ylabel(custom_name)
                twinx_ax.yaxis.label.set_color("red")
                twinx_ax.set_yticks([])
                twinx_ax.xaxis.set_major_formatter(fmt)
                twinx_ax.xaxis.set_major_locator(tickTime)


    axs[0].set_title(title)
    # set x limits
    if nustardo_obj is None:
        axs[0].set_xlim([np.min(data[name]['times']), np.max(data[name]['times'])])
    else:
        axs[0].set_xlim([np.min(nustardo_obj.lc_times), np.max(nustardo_obj.lc_times)])
    axs[-1].set_xlabel('Time (UTC)') 

    return fig, axs


def plotMarkers(markers, span=True, axis=None):
    """Takes markers to be plotted on an axis as vertical lines or a spanned shaded region.
    
    Parameters
    ----------
    markers : list of a list of ranges
            Points on the x-axis that want to be marked. If span=True then len(markers)>=2.

    span : Bool
            Set true for the regions to be shaded, false for just vertical lines.
            Default: True
    
    axis : Axis Object
            Axis to be plotted on. If None then "plt" is used.
            Default: None
            
    Returns
    -------
    The colour and marker range.
    """
    colours = ['k', 'r', 'g', 'c', 'm', 'b', 'y']
    markers_out = {}
    axis = {'ax':plt} if axis is None else {'ax':axis}

    for m in range(len(markers)):
        # make sure c cycles through the colours
        c = int( m - (m//len(colours))*len(colours) )

        # plot a shaded region or just the time boundaries for the chu changes
        if span:
            axis['ax'].axvspan(*markers[m], alpha=0.1, color=colours[c])
        else:
            axis['ax'].axvline(markers[m][0], color=colours[c])
            axis['ax'].axvline(markers[m][1], color=colours[c])
        markers_out[colours[c]] = markers[m]

    return markers_out