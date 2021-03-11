'''
Functions to go in here (I think!?):
    KC: 06/08/2019, added-
    ~plot XSPEC output
'''

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from copy import copy

from .nu_spec import read_xspec_txt, seperate
from .data_handling import xspecParams
from .plotting_essentials import plotMarkers


def fpmFromFilename(filename):
    ''' What FPM is the file from?
    
    Parameters
    ----------
    filename : str
            The name of the XSPEC output .txt and .fits file.
            
    Returns
    -------
    As tring of A, B, or A&B depending on what was in filename. 
    '''
    if "fpmab" in filename.lower():
        return "A&B"
    elif "fpma" in filename.lower():
        return "A"
    elif "fpmb" in filename.lower():
        return "B"


def searchAndLoad(xspec_output, fitting_mode):
    ''' Get the XSPEC output parameters, the count rate data, and the output file names without the extension.
    
    Parameters
    ----------
    xspec_output : str
            The name of the XSPEC output .txt and .fits file.

    fitting_mode : str
            The fitting mode used when fitting the spectrum. This gets passed to seperate() in nu_spec.
            
    Returns
    -------
    The fitted parameters, the count rates for the data, and the XSPEC filname without the extension. 
    '''
    xspec_output = xspec_output if not xspec_output.endswith(".fits") else xspec_output[:-5]
    xspec_output = xspec_output if not xspec_output.endswith(".txt") else xspec_output[:-4]

    keys_to_check = ["normalisation"]
    if "apec" in fitting_mode:
        keys_to_check.append("temperature")
    if "bknpower" in fitting_mode:
        keys_to_check.append("break")
        keys_to_check.append("photonindex")

    fitting_values = xspecParams(xspec_output+".fits", *keys_to_check)

    xspec_data = read_xspec_txt(xspec_output+".txt")
    counts_data = seperate(xspec_data, fitting_mode=fitting_mode)
    return fitting_values, counts_data, xspec_output


def meta_info(xspec_output):
    ''' Get meta data for the spectral fitting like the final statistic, livetime, FPMA/B scaling factor.
    
    Parameters
    ----------
    xspec_output : str
            The name of the XSPEC output .txt and .fits file.
            
    Returns
    -------
    A dictionary of the meta data for the spectral fit. 
    '''
    # other plotting info
    livetime = 'EXPOSURE'
    c_stat = 'STATISTIC'
    factor = 'factor'  # if A and B have been fit together this is the relative scaling
    return xspecParams(xspec_output+".fits", livetime, c_stat, factor)


def gain_info(xspec_output):
    ''' Get final gain used in the fitting.
    
    Parameters
    ----------
    xspec_output : str
            The name of the XSPEC output .txt and .fits file.
            
    Returns
    -------
    A dictionary of the meta data for the gain fit. 
    '''
    # other plotting info
    gain_params = ['gainSlope', 'gainSlopeElow', 'gainSlopeEhi']
    gain_values = xspecParams(xspec_output+"_gain.fits", *gain_params)
    
    gain = gain_values['gainSlope'][0]
    ext1, ext2 = gain-gain_values['gainSlopeElow'][0], gain-gain_values['gainSlopeEhi'][0]
    plusG = ext1 if ext1>0 else ext2
    minusG = ext1 if ext1<0 else ext2
    
    return " ($Gain: {{{0:.2f}}}^{{+{1:.2f}}}_{{-{2:.2f}}}$)".format(gain, plusG, abs(minusG))


def getSubModels(data):
    ''' Get mcount rates from the spectral fit models.
    
    Parameters
    ----------
    data : str
            The counts data from the seperate() method (a dictionary).
            
    Returns
    -------
    A dictionary of the count rates. 
    '''
    subModels = {}
    has_total_model_been = False
    for key in data.keys():
        if has_total_model_been:
            subModels[key] = data[key]
        if key == "model_total":
            has_total_model_been = True
    return subModels


def matchSubModels2Meta(subModels, fit_values, fpms):
    ''' Match teh count rates from the .txt file to the fitted parameters from the .fits file.
    
    Parameters
    ----------
    subModels : dict
            Output from getSubModels(). Just a dictionary of the count rates with appropriate keys.

    fit_values : dict
            Output from searchAndLoad(). Just a dictionary of the fitted parameters from the .fits XS{EC output file.

    fpms : str
            Output from fpmFromFilename(). "A", "B", or "A&B"
            
    Returns
    -------
    A dictionary where each entry is a list of the count rates, fitted parameters and errors. 
    '''
    sm = copy(subModels)
    
    counter = 1 if fpms!="A&B" else 2 # if "A&B" fit together then a constant is the first value
    for key in sm.keys():
        # if this is a thermal model
        if "apec" in key.lower():
            temp  = fit_values["kT"+str(counter)][0]/fit_values["kev2mk"]
            # calc. +/- errors, if zero then model was fixed so just keep it as 0
            etemp = [v/fit_values["kev2mk"] - temp if v!=0 else 0 for v in fit_values["EkT"+str(counter)][0]]
            counter += 3 # skip over the redshift and abundacnces
            EM  = fit_values["norm"+str(counter)][0]/fit_values["emfact"]
            eEM = [v/fit_values["emfact"] - EM if v!=0 else 0 for v in fit_values["Enorm"+str(counter)][0]]
            
            sm[key] = [subModels[key], temp, etemp, EM, eEM] #the data and the T&EM
        # if this is a non-thermal model
        elif "bknpower" in key.lower():
            counter += 1 # skip over the lower, fixed photon index
            breakE = fit_values["BreakE"+str(counter)][0]
            ebreakE = [v - breakE if v!=0 else 0 for v in fit_values["EBreakE"+str(counter)][0]]
            counter += 1
            gamma = fit_values["PhoIndx2"+str(counter)][0]
            egamma = [v - gamma if v!=0 else 0 for v in fit_values["EPhoIndx2"+str(counter)][0]]
            counter += 1
            norm_at_1kev = fit_values["norm"+str(counter)][0]
            enorm_at_1kev = [v - norm_at_1kev if v!=0 else 0 for v in fit_values["Enorm"+str(counter)][0]]
            sm[key] = [subModels[key], breakE, ebreakE, gamma, egamma, norm_at_1kev, enorm_at_1kev] #the data and the break,Gamma,Norm
        counter += 1
    return sm



def plotXspec(xspec_output, counts_data, matched_submodels, fitting_ranges=None, title=None, x_lim=None, y_lim=None, **kwargs):
    ''' Plots the output from an XSPEC fit.
    
    Parameters
    ----------
    xspec_output : str
            The name of the XSPEC output .txt and .fits file.

    counts_data: dict
            The observed data and total model data, "counts_data" from searchAndLoad().

    matched_submodels: dict
            The model count rates matched to the fitted parameters, "counts_data" from matchSubModels2Meta().
            
    fitting_ranges : list of length==2 lists
            The fitting range(s) the fit(s) took ploace over.
    
    title : str
            Title of the plot.
            
    x_lim : list, length==2
            The x-limits for plotting.
            
    y_lim : list, length==2
            The y-limits for plotting.
            
    **kwargs -- axes : matplotlib axes object
                        The axes for the data to be plotted on.
                        Default: plt
                        
                meta_data : dict
                        A dictionary for the meta data, i.e. livetime via "EXPOSURE" key.
                        
                total_model_colour : str
                        The colour for the total model.
                        Default: purple
                
                model_colours : list
                        Colours for the models that make up the total model.
                        Default: matplotlib default colours
                        
                fitting_range_colours : list
                        Colours of the shaded regions that indicate the fitting range.

                fitting_range_display : bool
                        Set to False if you don't want the fitting range to be plotted for some reason,
                        e.g. you want to plot it yourself in some other way. Fitting ranges should 
                        always be plotted one way or another though.
                        Default: True
                        
                EM_orders : list
                        The orders of magnitude you want the emission measures displayed as.
                        
                plot_parameters : bool or list
                        True to plot the fitted parameters, False to not. If you want some, and 
                        not others, plotted then provide list of bool.
                        Default: True
                        
                x_param : float
                        Starting x value (axes fraction) of where the parameters are plotted.
                        Default: 0.25
                        
                y_param : float
                        Starting y value (axes fraction) of where the parameters are plotted.
                        Default: 0.95
                        
                y_param_inc : float
                        Increment to decrease y_param with after plotting each parameter value.
                        Default: 0.07
                        
                param_size : float
                        Text size for the plotted parameters.
                        Default: 11
                              
                plot_res : bool
                        True to plot an appended residuals plot. The appended axes object will 
                        also be returned.
                        Default: True

                res_axes : matplotlib axes object
                        The axes for the residuals to be plotted on.
                        Default: An appended axes below the spectral plot

                add_to_title : string
                        If you want to tag on extra information in the default title.
                        Default: empty

    Returns
    -------
    The axes object of the plotted XSPEC fit (if plot_res=True then the axes for the data and 
    residuals plot is returned). 
    '''
    defaults = {"axes":plt, 
                "meta_data":None,
                "total_model_colour":"purple", "model_colours":plt.rcParams['axes.prop_cycle'].by_key()['color'], "fitting_range_colours":None, "fitting_range_display":True, 
                "EM_orders":None, "plot_parameters":True,
                "x_param":0.25, "y_param":0.95, "y_param_inc":0.07, "param_size":11, 
                "plot_res":True, "res_axes":None, 
                "add_to_title":""}
    defaults.update(kwargs)
    
    # be lazy, just unpack the dict instead of changing the variable names later
    axs = defaults["axes"]
    meta_data = defaults["meta_data"]
    total_model_colour, model_colours, fitting_range_colours = defaults["total_model_colour"], defaults["model_colours"], defaults["fitting_range_colours"]
    EM_orders, plot_parameters = defaults["EM_orders"], defaults["plot_parameters"]
    x_param, y_param, y_param_inc, param_size = defaults["x_param"], defaults["y_param"], defaults["y_param_inc"], defaults["param_size"]
    plot_res = defaults["plot_res"]
    add_to_title = defaults["add_to_title"]
    
    fpm_used = fpmFromFilename(xspec_output)
    statistic_used = " C-stat"
    livetime_str = " ({0:.2f} s)".format(meta_data['EXPOSURE'][0]) if type(meta_data)!=type(None) else ""
    
    # include FPMB to A scaling if it's there
    scaler = ""
    if type(meta_data)!=type(None):
        factor_there = [f for f in meta_data.keys() if "factor" in f]
        if len(factor_there)>0:
            factor_key = factor_there[0]
            factor, efactor = meta_data[factor_key], meta_data["E"+factor_key]
            ext1, ext2 = factor[0]-efactor[0][0], factor[0]-efactor[0][1]
            plusF = ext1 if ext1>0 else ext2
            minusF = ext1 if ext1<0 else ext2
            scaler = " ($C_{{B}}: {{{0:.2f}}}^{{+{1:.2f}}}_{{-{2:.2f}}}$)".format(factor[0], plusF, abs(minusF))
        
    title_str = "FPM" + fpm_used + statistic_used + livetime_str + scaler + add_to_title if type(title)==type(None) else title
    
    # values for plotting the params
    y_lim = [1e-1,3.5e3] if type(y_lim)==type(None) else y_lim
    x_lim = [2, 10] if type(x_lim)==type(None) else x_lim
    # because plt can't use set_
    if defaults["axes"]==plt:
        axs = plt.gca()
    axs.set_ylabel('Count Spectrum [cts s$^{-1}$ keV$^{-1}$]')
    axs.set_ylim(y_lim)
    axs.set_yscale('log')
    axs.set_title(title_str)
    axs.set_xlim(x_lim)
    axs.set_xlabel('Energy [keV]')
        
    ## plot data
    axs.errorbar(counts_data["energy"], counts_data["data"],xerr=counts_data["e_energy"], yerr=counts_data["e_data"], color='k', fmt='.',markersize=0.01, label='Data')

    ## plot total model
    tmc = "purple" if type(total_model_colour)==type(None) else total_model_colour
    axs.plot(counts_data["energy"], counts_data["model_total"], label='Total Model', color=total_model_colour)

    # it just became easier to have this seperate
    times_sign = str(u'\u00d7'.encode('utf-8').decode('utf-8')) 

    # what values do I want to plot? If True: plot all, False: plot none, if list then select the ones to plot
    plot_params = [plot_parameters]*len(matched_submodels.keys()) if type(plot_parameters)==bool else plot_parameters

    # plot sub-models
    cumulative_y_param_inc = -y_param_inc
    for n,sm in enumerate(matched_submodels.keys()):
        if "apec" in sm.lower():
            minusT = matched_submodels[sm][2][0] if matched_submodels[sm][2][0]<0 else matched_submodels[sm][2][1]
            plusT = matched_submodels[sm][2][0] if matched_submodels[sm][2][0]>0 else matched_submodels[sm][2][1]
            minusEM = matched_submodels[sm][4][0] if matched_submodels[sm][4][0]<0 else matched_submodels[sm][4][1]
            plusEM = matched_submodels[sm][4][0] if matched_submodels[sm][4][0]>0 else matched_submodels[sm][4][1]
            em_order = int(np.floor(np.log10(matched_submodels[sm][3]))) if type(EM_orders)==type(None) else EM_orders[n] # how big is the EM? 
            em_norm = 10**em_order
            if plusT==0 or minusT==0:
                param_str = "{0:.2f} MK".format(matched_submodels[sm][1]) +\
                            ", ${{{0:.2f}}}$".format(matched_submodels[sm][3]/em_norm) +\
                            times_sign+"$10^{{{0:.0f}}}$".format(em_order) + " cm$^{-3}$"
            else:

                param_str = "${{{0:.2f}}}^{{+{1:.2f}}}_{{-{2:.2f}}}$ MK".format(matched_submodels[sm][1], plusT, abs(minusT)) +\
                            ", ${{{0:.2f}}}^{{+{1:.2f}}}_{{-{2:.2f}}}$".format(matched_submodels[sm][3]/em_norm, plusEM/em_norm, abs(minusEM)/em_norm) +\
                            times_sign+"$10^{{{0:.0f}}}$".format(em_order) + " cm$^{-3}$"
            no_of_lines = 1

        elif "bknpower" in sm.lower():
            # break
            minusB = matched_submodels[sm][2][0] if matched_submodels[sm][2][0]<0 else matched_submodels[sm][2][1]
            plusB = matched_submodels[sm][2][0] if matched_submodels[sm][2][0]>0 else matched_submodels[sm][2][1]
            # gamma
            minusG = matched_submodels[sm][4][0] if matched_submodels[sm][4][0]<0 else matched_submodels[sm][4][1]
            plusG = matched_submodels[sm][4][0] if matched_submodels[sm][4][0]>0 else matched_submodels[sm][4][1]
            # normalisation at 1 keV
            minusN = matched_submodels[sm][6][0] if matched_submodels[sm][6][0]<0 else matched_submodels[sm][6][1]
            plusN = matched_submodels[sm][6][0] if matched_submodels[sm][6][0]>0 else matched_submodels[sm][6][1]
            if plusB==0 or minusB==0:
                param_str = "E$_{b}$: "+"{0:.2f} keV".format(matched_submodels[sm][1]) +\
                            ", $\gamma$: "+"${{{0:.2f}}}$".format(matched_submodels[sm][3]) +\
                            ", \n${{{0:.2f}}}$".format(matched_submodels[sm][5]) +\
                            " ph keV$^{-1}$ cm$^{-2}$ s$^{-1}$ @ 1 keV"
            else:

                param_str = "E$_{b}$: "+"${{{0:.2f}}}^{{+{1:.2f}}}_{{-{2:.2f}}}$ keV".format(matched_submodels[sm][1], plusB, abs(minusB)) +\
                            ", $\gamma$: "+"${{{0:.2f}}}^{{+{1:.2f}}}_{{-{2:.2f}}}$".format(matched_submodels[sm][3], plusG, abs(minusG)) +\
                            ", \n${{{0:.2f}}}^{{+{1:.2f}}}_{{-{2:.2f}}}$".format(matched_submodels[sm][5], plusN, abs(minusN)) +\
                            " ph keV$^{-1}$ cm$^{-2}$ s$^{-1}$ @ 1 keV"
            no_of_lines = 2
        c = int( n - (n//len(model_colours))*len(model_colours) ) # cycle through line_colours
        axs.plot(counts_data["energy"], matched_submodels[sm][0], label=param_str, color=model_colours[c])
        if plot_params[n]:
            cumulative_y_param_inc += y_param_inc*no_of_lines
            axs.annotate(param_str, (x_param,y_param - cumulative_y_param_inc), color=model_colours[c], xycoords='axes fraction', size=param_size)
            
    if defaults["fitting_range_display"]:
        markers = plotMarkers(fitting_ranges, span=True, axis=axs, customColours=fitting_range_colours)
    
    #residuals plotting
    if plot_res:
        residuals = (counts_data["data"] - counts_data["model_total"]) / counts_data["e_data"]
        
        if defaults["axes"]==plt:
            axs = plt.gca()
            
        if type(defaults["res_axes"])==type(None):
            divider = make_axes_locatable(axs)
            res = divider.append_axes('bottom', 1.2, pad=0.2, sharex=axs)
        else:
            res = defaults["res_axes"]
        # replace nans and infs with zeros (i.e., where there arent data so there is a divide by 0)
        residuals[~np.isfinite(residuals)] = 0
        residuals[np.isnan(residuals)] = 0
        res.plot(counts_data["energy"], residuals, drawstyle='steps-mid', color=total_model_colour)
        res.axhline(0, linestyle=':', color='k')
        res.set_xlim(x_lim)
        res.set_ylim([-7,7])
        res.set_ylabel('(y$_{Data}$ - y$_{Model}$)/$\sigma_{Data}$')
        res.set_xlabel('Energy [keV]')

        statistic = "C-stat: {0:.2f}".format(meta_data['STATISTIC'][0]) if type(meta_data)!=type(None) else ""
        res.annotate(statistic, (0.98,0.95), color="k", xycoords='axes fraction', size=param_size, ha="right", va="top")
        
        axs.xaxis.set_tick_params(labelbottom=False)
        axs.get_xaxis().set_visible(False)
        
        return axs, res
    return axs



def plotXspec_allTogether(xspec_output, fitting_mode, **kwargs):
    ''' Plots the output from an XSPEC fit while doing all the data seperation too.
    So sets everything up and runs plotXspec().
    
    Parameters
    ----------
    xspec_output : str
            The name of the XSPEC output .txt and .fits file.

    fitting_mode : str
            The fitting mode used when fitting the spectrum. This gets passed to seperate() in nu_spec.

    **kwargs -- all passed to plotXspec()
            
    Returns
    -------
    The axes object of the plotted XSPEC fit (if plot_res=True then the axes for the data and 
    residuals plot is returned). 
    '''

    # get the fitted parameters, the count rate data and the file name without an extension
    fitting_values, counts_data, xspec_output = searchAndLoad(xspec_output, fitting_mode)

    # get the livetime, c-stat, and FPMA&B scaling factor if you can
    plotting_values = meta_info(xspec_output)

    # get all the models that go into making the total model
    subMods = getSubModels(counts_data)

    # create a dict the has a key indicating a thermal or non-thermal model that is assigned to a list of the 
    #  count rates and fitted parameters for the model
    matched_submodels = matchSubModels2Meta(subMods, fitting_values, fpmFromFilename(xspec_output))

    # what if the gain was varied? I save this out in a xspec_output+_gain.fits file where xspec_output endswith gainVary
    if xspec_output.endswith("gainVary"):
        gain_string = gain_info(xspec_output)
        kwargs.update(add_to_title=gain_string)
    
    # now pass everything on to be plotted and return the axes object(s) created
    return plotXspec(xspec_output, counts_data, matched_submodels, meta_data=plotting_values, **kwargs)

