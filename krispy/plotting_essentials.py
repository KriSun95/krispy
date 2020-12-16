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

    KC: 16/12/2020, added-    
    ~textBox()
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


def plotSDOlightcurves(instrument, directory="./", files=None, data_list=None, title="Lightcurves", nustardo_obj=None, samePlot=False, 
                       other_data=None, **kwargs):
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

    kwargs : e.g. nustar_is_other_data = True -> to step the lightcurve is the other_data is NuSTAR data.
            
    Returns
    -------
    Displays a figure comprised of lightcurve subplots.
    """

    defaults = {"nustar_is_other_data":False, **kwargs}

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

    tickTime = None

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
        if type(tickTime) == type(None):
            tmin_length  = (data[name]['times'][-1] - data[name]['times'][0]).total_seconds()/60
            if 20 < tmin_length:
                tickTime = mdates.MinuteLocator(byminute=[0, 10, 20, 30, 40, 50], interval = 1)
            elif 10 < tmin_length <= 20:
                tickTime = mdates.MinuteLocator(byminute=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55], interval = 1)
            elif 0 < tmin_length <= 10:
                tickTime = mdates.MinuteLocator(interval = 1)
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
                if defaults["nustar_is_other_data"]:
                    axs[plot].plot(other_data[custom_name]["times"], other_data[custom_name][units[0]], color="red", label=custom_name, drawstyle='steps-post')
                else:
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


def plotMarkers(markers, span=True, axis=None, customColours=None):
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

    customColours : list 
            If you want ot provide your own colours for the markers. This list replaces the 
            default colours used.
            Default: None
            
    Returns
    -------
    The colour and marker range.
    """

    colours = ['k', 'r', 'g', 'c', 'm', 'b', 'y'] if type(customColours)==type(None) else customColours
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


def textBox(text, position, colour=None, **kwargs):
    """Draws text on a plot for you like plt.annotate or plt.text; however, this allows multicoloured text easily.
    
    Parameters
    ----------
    text : str
            The string you want printed.
    
    position : tuple, length 2
            The left x and lower y value of the axis the text is to be printed into.
            
    colour : str -or- list
            This is either one colour as a string or a list of colours with the same 
            length as the space seperated text. 
            Default: 'k'
            
    kwargs
    ----------
    axes : matplolitb axes object
            The axes of the plot you want the text on.
            Default: plt.gca()
    
    facecolour : str
            The colour of the background for the text.
            Default: 'white'
            
    axes_alpha : 0.0<=float<=1
            The opacity of the background for the text. Set to 0 for 100% transparent 
            and 1 for 100% opaque. 
            Default: 0.5
            
    fontsize : float
            Fontsize of the text. 
            Default: 18
            
    fontweight : float -or- str
            Fontweight of the text. 
            Default: 'normal'
            
    text_ax_width_scale : float
            Scale the background width for the text.
            Default: 1.05
    
    text_ax_height_scale : float
            Scale the background height for the text.
            Default: 1.25
            
    leftx_text_point_scale : float
            Alter the position of where the left of the text gets aligned to within its axes. 
            Default: 1.00
            
    middley_text_point_scale : float
            Alter the position of where the middle height of the text gets aligned to within its axes. 
            Default: 0.45
            
    letter_width_scale : float
            Scale the width of the text characters incase they bunch up or spread out. 
            Default: 1.00
            
    Any other kwargs just get passed to matplotlib.pyplot.annotate()
            
    Returns
    -------
    Nothing. This just plots the text.
    
    Examples
    --------
    To draw the word 'Text.' in red with a green background
    -> textBox('Text.', (0.5,0.5), colour='r', facecolour='g')
    
    To draw the words 'What about more text.' in red with a green background
    -> textBox('What about more text.', (0.5,0.5), colour='r', facecolour='g')
    
    To draw the words 'What about more text.' in red, blue, and cyan with a green background
    -> textBox('What about more text.', (0.5,0.5), colour=['r','r','b','c'], facecolour='g')
    (Note: text will get broken up into ['What','about','more','text.'] and so each need a colour.)
    """
    
    # sort out parameters I use here and the ones to pass onto annotate
    axes = plt.gca() if "axes" not in kwargs else kwargs["axes"]
    facecolour = 'white' if "facecolour" not in kwargs else kwargs["facecolour"]
    axes_alpha = 0.5 if "axes_alpha" not in kwargs else kwargs["axes_alpha"]
    fontsize = 18 if "fontsize" not in kwargs else kwargs["fontsize"]
    fontweight = "normal" if "fontweight" not in kwargs else kwargs["fontweight"]
    # now remove the ones I use directly so that kwargs can get passed to annotate
    kwargs.pop("axes", None)
    kwargs.pop("facecolour", None)
    kwargs.pop("axes_alpha", None)
    kwargs.pop('fontsize', None)
    kwargs.pop("fontweight", None)
    
    ## ***** coded numbers to adjust text/text axes the way you want *****
    # scaling factors to make the axes created for the text slightly larger than the text dimensions
    text_ax_width_scale = 1.05 if "text_ax_width_scale" not in kwargs else kwargs["text_ax_width_scale"]
    text_ax_height_scale = 1.25 if "text_ax_height_scale" not in kwargs else kwargs["text_ax_height_scale"]
    kwargs.pop("text_ax_width_scale", None)
    kwargs.pop("text_ax_height_scale", None)
    # text is aligned vertically (VA) center and horizontally (HA) left 
    # but 0.45 works better than 0.5 here for VA and HA is caluclated in a weird way, 
    # change if you want/need to move the text about in the axes created for it
    leftx_text_point_scale = 1.00 if "leftx_text_point_scale" not in kwargs else kwargs["leftx_text_point_scale"]
    middley_text_point_scale = 0.45 if "middley_text_point_scale" not in kwargs else kwargs["middley_text_point_scale"]
    kwargs.pop("leftx_text_point_scale", None)
    kwargs.pop("middley_text_point_scale", None)
    # incase the width of the letters need scaled
    letter_width_scale = 1.00 if "letter_width_scale" not in kwargs else kwargs["letter_width_scale"]
    kwargs.pop("letter_width_scale", None)
    
    # need to know text box dimensions
    text_extent = axes.annotate(text, 
                                (0, 0), 
                                fontsize=fontsize, fontweight=fontweight,
                                bbox=dict(boxstyle="square, pad=0"))
    canvas = axes.figure.canvas
    text_extent.draw(canvas.get_renderer())
    # get the text box size
    s = text_extent.get_window_extent()
    # get the axes size (window) to be able to calculate the axis fraction for teh text box
    window = axes.get_window_extent()
    # done here so remove this text box
    text_extent.remove()

    # now make axes the correct size for the text with a little border (hence the 1.05 and 1.05)
    tb_ax = axes.inset_axes([*position, 
                             text_ax_width_scale*s.width/window.width, 
                             text_ax_height_scale*s.height/window.height])# make axes for the text
    
    # colour and transparency of axis
    tb_ax.patch.set_facecolor(facecolour)
    tb_ax.patch.set_alpha(axes_alpha)
    
    # remove ticks
    tb_ax.xaxis.set_visible(False)
    tb_ax.yaxis.set_visible(False)
    
    # Hide the right and top spines
    for w in ['right', 'left', 'top', 'bottom']:
        tb_ax.spines[w].set_visible(False)
        
    # now to deal with the text
    words = text.split() # split at spaces
    # if there are no colour(s) given, make it all black
    colours = ['black']*len(words) if type(colour)==type(None) else colour
    # if there are colour(s) given, make it a list if it is a single colour or just keep as is
    colours = [colours]*len(words) if type(colours)!=list else colours
    
    ## now let's draw the text
    # work in axes pixels units
    coords = 'axes pixels'
    # define the x and y positions for text alignment in it axes
    # can scale with 'leftx_text_point_scale' and 'middley_text_point_scale' if needed
    x = leftx_text_point_scale * (tb_ax.get_window_extent().width-s.width)/(2)
    y = middley_text_point_scale * tb_ax.get_window_extent().height
    # length of total text already down
    length=0
    for c, w in zip(colours, words):
        text_draw = tb_ax.annotate(w, (x+length, y), 
                                   fontsize=fontsize, fontweight=fontweight, 
                                   va='center', ha='left', 
                                   color=c, 
                                   xycoords=coords, 
                                   **kwargs)

        canvas = tb_ax.figure.canvas
        text_draw.draw(canvas.get_renderer())
        length += letter_width_scale * text_draw.get_window_extent().width
        
        # add a space if it isn't the last word
        if w!=words[-1]:
            text_draw = tb_ax.annotate(" ", (x+length, y), 
                                       fontsize=fontsize, fontweight=fontweight,
                                       va='center', ha='left', 
                                       xycoords=coords, 
                                       **kwargs)
            canvas = tb_ax.figure.canvas
            text_draw.draw(canvas.get_renderer())
            length += letter_width_scale * text_draw.get_window_extent().width
            
    # nothing to return
    return 