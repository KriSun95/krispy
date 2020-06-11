'''
Functions to go in here (I think!?):
    KC: 01/12/2018, ideas-
    ~make_maps_from_dir()       <
    ~contour_maps_from_dir()    <
    ~iron_18_cmap()             <

    KC: 19/12/2018, added-
    ~aiamaps()
    ~aiamaps_from_dir() ****Use aiamaps instead right now, big memory leak****
    ~contourmaps_from_dir()
    ~iron18 cmap is now 'Blues' and the mid-colour can be found in plotting_essentials
'''
from . import file_working
from . import contour
from . import data_handling

import matplotlib.pyplot as plt
import matplotlib
import os
from astropy.io import fits
import astropy.units as u
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from astropy.coordinates import SkyCoord
import numpy as np
import nustar_pysolar as nustar
import sunpy.map
from scipy import ndimage
import scipy.io
import sunpy
import datetime
import gc
import re
import warnings
import astropy.units as u
from astropy.time import Time
import matplotlib.patches as patches

'''
Alterations:
    KC: 20/12/2018 - made it possible to add more than one rectangle in 'aiamaps()'.
    KC: 19/01/2019 - the sunpy.map.Map() now takes in a list instead of map objects being created
                        for every file.
    KC: 19/01/2019 - aiamaps() can now take a wavelength for a title, look at two iron channels,
                        change the scale of the colour bar, and can now subtract average of 
                        observation.                        
    KC: 19/01/2019 - contourmaps_from_dir() now takes an AIA file from within time range instead
                        having time bins all having to match bfore input.
    KC: 22/01/2019 - cm_scale is changed to 'Normalize' now when a difference image is requested.
    KC: 05/02/2019 - aiamaps() memory leak now solved!! By calling the 'del' operator on the variable 'aia_map'
                     before too much gets loaded into memory releases the data efficiently.
                   - aiamaps() now makes use of the method 'collect()' from garbage collection (gc) to keep on top of memory.
    KC: 21/03/2019 - aiamaps() now can take multiple directories, save directories, difference types, change the resolution, 
                     and save the submaps being worked with.
                   - contourmaps_from_dir() can now specify when the aia map comes from in the time range that is being looked at. 
    KC: 03/04/2020 - multiple rectangles plotted should now be different colours.
'''

#make images from the aia fits files
def aiamaps(directory, save_directory, submap=None, cmlims=None, rectangle=None, rectangle_colour=None, save_inc=True, iron='',
           cm_scale='Normalize', diff_image=None, res=None, save_smap=None, colourbar=True, time_filter=None):      
    """Takes a directory with fits files, constructs a map or submap of the full observation with/without a rectangle and
    saves the image in the requested directory.
    
    Parameters
    ----------
    directory : Data directory
            The directory which contains the list of fits files from the AIA. Must end with a '/'. Can now accept a list of directories
            of multiple data directories, e.g. [dir1,dir2,...]. Make sure they're in order.
    
    save_directory : Save directory
            The directory in which the new fits files are saved. Must end with a '/'. Can now accept a list of directories
            of where to save all of the images, e.g. [sav1,sav2,...].
    
    save_directory : Save directory
            The directory in which the new fits files are saved. Must end with a '/'.
        
    savefile_fmt : Str
            File extension for the saved file's format, e.g. '.png', '.jpg', '.pdf', etc.
            Default: '.png'
            
    submap : One-dimensional list/array, length 4
            Contains the bottom left (bl) and top right (tr) coordinates for a submap, e.g. [blx,bly,trx,try]. Must be 
            in arcseconds, of type float or integer and NOT an arcsec object.

    cmlims : One-dimensional list/array of type float or int, length 2
            Limits of the colourmap, e.g. [vmin, vmax]. 
            Default: Empty list, []
            
    rectangle : two-dimensional list/array, shape=n,4
            Contains lists of the bottom left (bl) and top right (tr) coordinates to draw a rectangle on the constructed 
            map, e.g. [[blx1,bly1,trx1,try1], [blx2,bly2,trx2,try2], ...]. Must be in arcseconds, of type float or 
            integer and NOT an arcsec object.
            Default: Empty list, []

    rectangle_colour : one-dimensional list of strings
            Contains the colour or colours you want associated with the rectangle you want to plot. Must contain one colour or
            the same number of colours as there are rectangles.
            Default: ["black"] for iron, else ["white"] 
            
    save_inc : Bool
            Indicates whether or not the save file should be named with respect to the file it was produced from or be
            named incrementally, e.g. AIA123456_123456_1234.png (False) or map_0000 (True) respectively.
            Default: True
    
    iron : Str
            Indicates whether or not the save file is the iron 16 or 18 channel. Set to '16' or '18'.
            Default: ''

    cm_scale : Str
            Scale for the colour bar for the plot. Set to 'Normalize' or 'LogNorm'.
            Default: 'Normalize'

    diff_image : Str
            Can set the output images to have it's previous image subtracted (running difference) or have the first image of the 
            observation subtracted, e.g. 'running' or 'subtratct_first'.
            Default: None

    res : float
            A float <1 which has the resolution of the image reduced, e.g. 0.5 produces an image at 50% resolution to the original.
            Default: None
            
    save_smap : Str
            Can be set to a string for the directory if you want to save the subamps data being worked with to that directory. These
            submap files are not overwritten so delete them manually if you run it again with a different sub-region.
            Default: None

    colourbar : Bool
            Indicates whether or not to draw the colour bar for the map.
            Default: True

    time_filter : list, 2 strings
            If you provide a directory but only want maps made from a certain time range, e.g. 
            ["%Y/%m/%d, %H:%M:%S", "%Y/%m/%d, %H:%M:%S"].
            Defualt: None

    Returns
    -------
    AIA maps saved to the requested directory (so doesn't really return anythin).
    """
    
    rectangle_colour = ["black"] if rectangle_colour is None else rectangle_colour
    if type(rectangle_colour) is not list:
        rectangle_colour = [rectangle_colour]

    rectangle = [] if rectangle is None else rectangle
    cmlims = [] if cmlims is None else cmlims

    np.seterr(divide='ignore', invalid='ignore') #ignore warnings resulting from missing header info
    warnings.simplefilter('ignore', Warning)

    matplotlib.rcParams['font.sans-serif'] = "Arial" #sets up plots
    matplotlib.rcParams['font.family'] = "sans-serif"
    matplotlib.rcParams['font.size'] = 12

    title_addition = ''
    rescale_cml = False
    first_time_through = True

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

    d = 0

    for f in range(no_of_files):

        aia_map = sunpy.map.Map(directory_with_files[f])

        
        if diff_image == None:
            pass
        elif diff_image == 'subtract_first':
            first_map = sunpy.map.Map(directory_with_files[0]) #first map and keep it the now
            diff_data = aia_map.data - first_map.data
            del first_map
            aia_map = sunpy.map.Map(diff_data, aia_map.meta)
            del diff_data
            if f == 0: #define limits for the full observation from the difference between the first and last observation
                first_map = sunpy.map.Map(directory_with_files[0])
                last_map = sunpy.map.Map(directory_with_files[-1])
                if cmlims != []: #offer a way for the limits to be scaled
                    #scale factor of 0.3 just to make things look nice
                    cmlims = [cmlims[0]*0.3*np.min(last_map.data - first_map.data), cmlims[1]*0.3*abs(np.min(last_map.data - first_map.data))]
                else:
                    cmlims = [0.3*np.min(last_map.data - first_map.data), 0.3*abs(np.min(last_map.data - first_map.data))]
                del first_map
                del last_map
                rescale_cml = True #this is only used if 'res' is set since if the matrix is rescaled then the limits need to be adjusted too
            cm_scale='Normalize' #Normalized colour bar for difference images
            title_addition = ' (with First Obs. Subtracted)' # with space before first bracket, extra info for plot title
        elif diff_image == 'running':
            if f == 0: #first map doesn't have one prior to subtract
                continue
            previous_map = sunpy.map.Map(directory_with_files[f-1])
            diff_data = aia_map.data - previous_map.data
            del previous_map
            aia_map = sunpy.map.Map(diff_data, aia_map.meta)
            del diff_data
            if f == 1:
                first_map = sunpy.map.Map(directory_with_files[0])
                second_map = sunpy.map.Map(directory_with_files[1])
                if cmlims != []:
                    cmlims = [cmlims[0]*0.3*np.min(second_map.data - first_map.data), cmlims[1]*0.3*abs(np.min(second_map.data - first_map.data))]
                else:
                    cmlims = [0.3*np.min(second_map.data - first_map.data), 0.3*abs(np.min(second_map.data - first_map.data))]
                del first_map
                del second_map
                rescale_cml = True
            cm_scale='Normalize'
            title_addition = ' (Running Difference)' #with space before first bracket
        else:
            assert 1==0, 'Ahhhhhhhhhhhhhhhh!'

        if (_aia_files[0][0:3] == 'HMI') or (_aia_files[0][0:3] == 'hmi'):
            # HMI is mount "upside down"
            aia_map = aia_map.rotate()
        
        if submap != None:
            bl_fi = SkyCoord(submap[0]*u.arcsec, submap[1]*u.arcsec, frame=aia_map.coordinate_frame)
            tr_fi = SkyCoord(submap[2]*u.arcsec, submap[3]*u.arcsec, frame=aia_map.coordinate_frame)
        
            smap = aia_map.submap(bl_fi,tr_fi) 
            del aia_map
        else:
            smap = aia_map

        if res is not None:
            res = res if res<1 else 1
            orig_size = np.shape(smap.data)
            smap = smap.resample(u.Quantity([orig_size[0]*res,orig_size[1]*res], u.pixel)) #new dimensions are the fraction you want of the original
            if first_time_through == True: #only add the new title info on the first go through
                title_res = ' ({:.0f}'.format(res*100)+'% res)'
                title_addition = title_addition + title_res
                first_time_through = False
            if rescale_cml == True: #rescale limits if the resolution is changed
                cmlims = [cmlims[0]*(0.5*(1+res)), cmlims[1]*(0.5*(1+res))]
                rescale_cml = False     

        if cm_scale == 'LogNorm':
            #set to min positive value to avoid nans in the log plot for values <=0
            m_data = smap.data
            m_data[m_data<=0] = np.min(np.min(m_data[m_data>0])) 
            smap = sunpy.map.Map(m_data,smap.meta)
            del m_data
            
        if iron == '18':
            smap.plot_settings['cmap'] = plt.cm.GnBu
        if iron == '16':
            smap.plot_settings['cmap'] = plt.cm.Purples
        if diff_image is not None:
            smap.plot_settings['cmap'] = plt.cm.coolwarm
        if (_aia_files[0][0:3] == 'HMI') or (_aia_files[0][0:3] == 'hmi'):
            smap.plot_settings['cmap'] = matplotlib.cm.get_cmap('PiYG_r') # use a simpler cmap than 'hmimag', pink=positive, green=negative

        #save submap that you're working with to save time later
        if save_smap is not None:
            filename_regex = re.compile(r'\w+\.\w+') #find the filename from the full path
            filename = filename_regex.findall(directory_with_files[f])[0]
            if 'submapped_'+filename in os.listdir(save_smap):
                # the file has already been saved and this is a duplicate so ignore
                pass
            else:
                smap.save(save_smap+'submapped_'+filename) #save the submap with the prefix 'submapped_' to the original filename

        fig, ax = plt.subplots(figsize=(9,8)) 
        
        compmap = sunpy.map.Map(smap, composite=True) #comp image as to keep formatting the same as NuSTAR
        
        if cm_scale == 'Normalize': #tell the plot the choices for the limits and colour map
            if cmlims != []:
                if cmlims[0] <= 0 and diff_image == False: #vmin > 0 or error
                    cmlims[0] = 0.1
                    compmap.plot(vmin=cmlims[0], vmax=cmlims[1], norm=colors.Normalize())
                else:
                    compmap.plot(vmin=cmlims[0], vmax=cmlims[1], norm=colors.Normalize())
            elif cmlims == []:
                compmap.plot(norm=colors.Normalize())
            if colourbar == True:
                if res is not None: #res makes the units per pixel
                    plt.colorbar(label='DN pix$^{-1}$ s$^{-1}$')
                elif res is None:
                    plt.colorbar(label='DN s$^{-1}$')
            
        elif cm_scale == 'LogNorm':
            if cmlims != []:
                if cmlims[0] <= 0: #vmin > 0 or error
                    cmlims[0] = 0.1
                    compmap.plot(vmin=cmlims[0], vmax=cmlims[1], norm=colors.LogNorm()) 
                else:
                    compmap.plot(vmin=cmlims[0], vmax=cmlims[1], norm=colors.LogNorm())
            elif cmlims == []:
                compmap.plot(norm=colors.LogNorm())
            if colourbar == True:
                if res is not None:
                    plt.colorbar(label='DN pix$^{-1}$ s$^{-1}$')
                elif res is None:
                    plt.colorbar(label='DN s$^{-1}$')

        
        if rectangle != []: #if a rectangle(s) is specified, make it
            assert len(rectangle_colour)==len(rectangle) or len(rectangle_colour)==1, "Check you have either given 1 colour in the \'rectangle_colour\' list or the same number of colours as rectangles!"
            rectangle_colour = rectangle_colour if len(rectangle_colour)==len(rectangle) else rectangle_colour*len(rectangle)
            x, y, counter = submap[2], submap[3], 0 # x and y for box titles if needed, plus a counter for the "for" loop

            for rect, rcol in zip(rectangle, rectangle_colour):
                
                bl_rect = (rect[0], rect[1]) #SkyCoord(rect[0]*u.arcsec, rect[1]*u.arcsec, frame=smap.coordinate_frame)
                length = rect[2] - rect[0]
                height = rect[3] - rect[1]
                if (iron != '') or (diff_image != None): #if iron or a diff map is needed then make the rectangles black
                    rcol = "black" if len(rectangle_colour)==1 else rcol
                    plt.gca().add_patch(patches.Rectangle(bl_rect, length, height, facecolor="none", linewidth=2, edgecolor=rcol))
                    #smap.draw_rectangle(bl_rect, length*u.arcsec, height*u.arcsec, color = rcol, axes=plt.gca())
                else:
                    rcol = "white" if len(rectangle_colour)==1 else rcol
                    plt.gca().add_patch(patches.Rectangle(bl_rect, length, height, facecolor="none", linewidth=2, edgecolor=rcol))
                    #smap.draw_rectangle(bl_rect, length*u.arcsec, height*u.arcsec, color = rcol, axes=plt.gca())

                # if there are multiple boxes then label them with the colour, tough if you're using the same colour the now
                if len(rectangle_colour) > 1:
                    # lazy check for no repeats
                    if rectangle_colour[0] not in rectangle_colour[1:]:
                        plt.text(x, y-counter*0.06*(submap[3]-submap[1]), "Box "+str(counter+1), 
                            verticalalignment="top", horizontalalignment="right",
                            color=rcol)
                        counter += 1
        
        #make titles
        time = smap.meta['t_obs'] 
        wavelength = str(smap.meta['wavelnth'])
        if iron == '18': #sets title for Iron 18
            plt.title(f'AIA FeXVIII {time[:10]} {time[11:19]}'+title_addition)  
        elif iron == '16': #sets title for Iron 16
            plt.title(f'AIA FeXVI {time[:10]} {time[11:19]}'+title_addition)
        else:
            plt.title('AIA '+wavelength + r'$\AA$ ' + f'{time[:10]} {time[11:19]}'+title_addition)

        if type(save_directory) == str: #save writing what is below twice
            save_directory = [save_directory]
        if type(save_directory) == list:
            for _save_d in save_directory:
                if save_inc == False:
                    plt.savefig(_save_d + f'aia_image{wavelength}.png', dpi=300, bbox_inches='tight')
                elif save_inc == True:
                    plt.savefig(_save_d + 'maps{:04d}.png'.format(d), dpi=300, bbox_inches='tight')
        d+=1
                
        plt.clf() 
        plt.cla()
        plt.close('all')

        
        bl_fi = 0 #reassign variables that take up a lot of space the delete them just to be sure
        tr_fi = 0 
        aia_map = 0
        smap = 0
        compmap = 0
        del bl_fi
        del tr_fi
        del fig
        del ax
        del aia_map
        del smap
        del compmap

        gc.collect()
        print(f'\r[function: aiamaps()] Saved {d} submap(s) of {no_of_files}.        ', end='')

    aia_map = 0
    smap = 0
    compmap = 0
    del aia_map
    del smap
    del compmap
    del directory_with_files
    gc.collect()
    print('\nLook everyone, it\'s finished!')


#make contour maps
def contourmaps_from_dir(aia_dir, nustar_dir, nustar_file, save_dir, hk_file=None, chu='', fpm='', energy_rng=[], submap=[], 
                         cmlims = [], nustar_shift=None, time_bins=[], resample_aia=[], counter=0, contour_lvls=[],
                         contour_fmt='percent', contour_colour='black', aia='ns_overlap_only', iron18=True, 
                         save_inc=False, gauss_sigma=4, background_time='begin', save_bg=False, A_and_B=False, frame_to_correlate=0, 
                         AB_pixshift=None, deconvolve=False, it=20):
    """Takes a list of times, a nustar fits file and and list of iron 18 AIA fits files and produces an AIA map with 
    contours from the nustar observation between adjacent time in the time list.
    
    Parameters
    ----------
    aia_dir : Data directory, str
            The directory which contains the list of fits files from the AIA. Must end with a '/'.
    
    nustar_dir : Data directory, str
            The directory in which the nustar fits file is saved. Must end with a '/'.
        
    nustar_file : Filename, str
            NuSTAR file's name from nustar_dir which should be split into a chu state and be mapped in helioprojection 
            units, e.g. 'nu80415202001A06_chu13_S_cl_sunpos.evt'.
            
    chu : Camera head unit, str
            The camera head unit configuration of the NuSTAR file's observation, e.g. 'chu1', 'chu2', 'chu13', etc.
            Default: ''

    fpm : Focal plane module, str
            The focal plane module of the NuSTAR file's observation, e.g. 'A', 'B'.
            Default: ''
            
    energy_rng : One-dimensional list/array, length 2
            The energy range being inspected from the NuSTAR observation, e.g. [2,5], [6,8].
            Default: []
            
    submap : One-dimensional list/array, length 4
            Contains the bottom left (bl) and top right (tr) coordinates for a submap, e.g. [blx,bly,trx,try]. Must be 
            in arcseconds, of type float or integer and NOT an arcsec object. 
            Default: []

    cmlims : One-dimensional list/array of type float or int, length 2
            Limits of the colourmap, e.g. [vmin, vmax]. To avoid error, if vmin <= 0 then vmin is set to 0.1.
            Default: []
            
    nustar_shift : One-dimensional list/array of type int, length 2, or str
            Gives an [x,y] shift to move the NuSTAR contours to align with the AIA image as the coordinates of the NuSTAR
            map may not be spot on. This can also accept 'cc' fort eh shift to be done by cross-correlation.
            Default: []
    
    time_bins : One-dimensional list/array of type Str, length N
            This list/array provides the time boundariesfort he NuSTAR time bins in the form '%Y/%m/%d, %H:%M:%S'.
            Default: []
            
    resample_aia : One-dimensional list/array of type int, length 2
            The number of pixels the AIA map should get rebinned to, e.g. [1000,1000].
            Default: []
            
    counter : Number, int
            The number that is padded for the incremental save of the maps. This makes it possible to run the function 
            seperately for different time intervals with different variables but still keep them in sequence. 
            Default: 0
            
    contour_lvls : One-dimensional list/array of type int, length N
            This list dictates what values the contour levels represent.
            Default: []
            
    contour_fmt : str
            A string indicating whether to accept the values of the contour levels as percentages or their values, 
            e.g. 'percent' or 'actual_values' respectively.
            Default: 'percent'
            
    contour_colour : str 
            The colour of the contour lines, e.g. 'red', 'white', etc.
            Default: 'black'
            
    aia : str
            This string describes whether the maps created come only from times with NuSTAR AND AIA data, NuSTAR OR AIA 
            data, or just AIA data e.g. 'ns_overlap_only', 'all', or 'solo' respectively. All these maps are plotted the
            same way as to easily allow them to be compared directly or to be stitched together in a movie.
            Default: 'ns_overlap_only'
        
    iron18 : Bool
            This indicates that the aia fits files represent the iron 18 channel, any other channel already has the 
            colour map in its header information.
            Default: True
            
    save_inc : Bool
            Indicates whether or not the save file should be named with respect to the file it was produced from or be
            named incrementally, e.g. nustar contours1_on_iron18_chu1_fpmA.png or contours000 respectively.
            Default: False

    gauss_sigma : float
            The value of the gaussian kernal for NuSTAR data to be smoothed over.
            Default: 4

    background_time : Str
            Specify if you want the background map to be taken from the start, middle, end, average of the time range, e.g.
            'begin', 'middle', 'end', 'average'.
            Default: 'begin'

    save_bg : Bool
            Do you want to save the background array?
            Default: False

    A_and_B : Bool
            Specify whether, if given FPMA at first, if you want to combine it with FPMB.
            Default: False 

    frame_to_corr : int
            Index of the start time for the frame to be cross-correlated with, default is the first frame. For every individual 
            frame to be cross-correlated then frame_to_corr='individual'.
            Default: 0

    AB_pixshift : 2d-array
            A list of the A pixel shift and the B pixel shift, e.g. [[Ax, Ay], [Bx, By]]. Any row can be None. If you only are 
            working with, and only want to shift one, make it the first row, i.e. working with FPMB only then [[Bx, By], None].
            Default: None
            
    Returns
    -------
    A dictionary with the values of the largest values of the NuSTAR map to help with contour value setting 
    (labelled as 'max_contour_levels'), the final value for the incremental counter (labelled as 'last_incremental_value'), 
    'nustar_shift' gives the manual shift in arcseconds the user gives, 'cc_A_and_B_pixel_shifts' provides the 
    cross-correlation pixel shift values. AIA maps, with NuSTAR contours, are also saved to the requested directory.
    """
    #20/11/2018: ~if statement for the definition of cleanevt.
    #            ~two iron 18 if statements for the colour map.
    #            ~two iron 18 if statements for the colour map.
    #26/11/2018: ~added try and except to the cleanevt bit.
    #08/04/2019: ~can now combine A and B.
    #23/05/2019: ~fixed some issues from when I enabled combining two FPMs

    warnings.warn('This function, contourmaps_from_dir(), is now deprecated and should not be used. Use the Contour class instead.', DeprecationWarning)
    
    import filter_with_tmrng # this file has to be in the directory
    
    matplotlib.rcParams['font.sans-serif'] = "Arial" #sets up plots
    matplotlib.rcParams['font.family'] = "sans-serif"
    matplotlib.rcParams['font.size'] = 14
    
    hdulist = fits.open(nustar_dir + nustar_file) #chu, sunpos file
    evtdata=hdulist[1].data
    hdr = hdulist[1].header
    hdulist.close()

    #hk_file
    nu1 = krispy.nustardo.NustarDo(nustar_dir + nustar_file)
    nu1.livetime(hk_filename=hk_file, show_fig=False)
    lvt_t1 = [(datetime.datetime(2010,1 ,1 ,0 ,0 ,0) + timedelta(seconds=t)) for t in nu1.hk_times]
    lvt_lvt1 = nu1.hk_livetimes
    
    if A_and_B == True:
        hdulist = fits.open(nustar_dir + nustar_file.replace('A', 'B')) #chu, sunpos file
        evtdata_other = hdulist[1].data
        hdr_other = hdulist[1].header
        hdulist.close()
        nu2 = krispy.nustardo.NustarDo(nustar_dir + nustar_file.replace('A', 'B'))
        nu2.livetime(hk_filename=hk_file.replace('A', 'B'), show_fig=False)
        lvt_t2 = [(datetime.datetime(2010,1 ,1 ,0 ,0 ,0) + timedelta(seconds=t)) for t in nu2.hk_times]
        lvt_lvt2 = nu2.hk_livetimes
    
    aia_files = os.listdir(aia_dir)
    aia_files.sort()
    aia_files = only_fits(aia_files)

    manual_pixel_shift = False
    if AB_pixshift != None:
        manual_pixel_shift = True
        if len(np.shape(AB_pixshift[0])) == 1:
            AB_pixshift = [[AB_pixshift[0]], AB_pixshift[1]]
        if len(np.shape(AB_pixshift[1])) == 1:
            AB_pixshift = [AB_pixshift[0], [AB_pixshift[1]]]
    
    d = counter
    max_contours = []

    all_A_shift = []
    all_B_shift = []

    saved_backgrounds = []
    
    for t in range(len(time_bins)-1):
        if frame_to_correlate == 'individual':
            frame_to_corr = t
        else:
            frame_to_corr = frame_to_correlate
        try:
            cleanevt = filter_with_tmrng.event_filter(evtdata,fpm=fpm,energy_low=energy_rng[0], energy_high=energy_rng[1], 
                                                      tmrng=[time_bins[t], time_bins[t+1]])
            if A_and_B == True:
                            cleanevt_other = filter_with_tmrng.event_filter(evtdata_other,fpm=fpm,energy_low=energy_rng[0], energy_high=energy_rng[1], 
                                                      tmrng=[time_bins[t], time_bins[t+1]])
            else:
                cleanevt_other = []
        except IndexError:
            cleanevt = [] #if time range is outwith nustar obs (i.e. IndexError) then this still lets aia to be looked at
            if A_and_B == True:
                            cleanevt_other = []
            
        if len(cleanevt) != 0 and aia == ('ns_overlap_only' or 'all'): #AIA data and NuSTAR data
            
            t_1 = datetime.datetime.strptime(time_bins[t], '%Y/%m/%d, %H:%M:%S')
            t_2 = datetime.datetime.strptime(time_bins[t+1], '%Y/%m/%d, %H:%M:%S')
            #livetime correction A
            ltimes_in_rangeA = []
            for lv in range(len(lvt_t1)):
                if ((lvt_t1[lv]>=t_1) & (lvt_t1[lv]<t_2)):
                    ltimes_in_rangeA.append(lvt_lvt1[lv])
            ave_livetime_A = np.average(ltimes_in_rangeA)

            if A_and_B == True:
                #livetime correction B
                ltimes_in_rangeB = []
                for lv in range(len(lvt_t2)):
                    if ((lvt_t2[lv]>=t_1) & (lvt_t2[lv]<t_2)):
                        ltimes_in_rangeB.append(lvt_lvt2[lv])
                ave_livetime_B = np.average(ltimes_in_rangeB)

            nustar_map = nustar.map.make_sunpy(cleanevt, hdr, norm_map=False)
            final_map_meta = nustar_map.meta
            nustar_map_normdata = nustar_map.data / ((t_2 - t_1).total_seconds() * ave_livetime_A)

            '''
            Need to find shift before all of this, CANNOT do it each time
            '''
            
            if (manual_pixel_shift == False): #(len(cleanevt_other) != 0) and 
                if (A_and_B == True) or (nustar_shift == 'cc'):
                        if (t == 0) or (frame_to_correlate == 'individual'): #only want the shift at the start
                                cleanevt_c = filter_with_tmrng.event_filter(evtdata,fpm=fpm,energy_low=energy_rng[0], energy_high=energy_rng[1], 
                                                      tmrng=[time_bins[frame_to_corr], time_bins[frame_to_corr+1]])
                                if (A_and_B == True):
                                    cleanevt_other_c = filter_with_tmrng.event_filter(evtdata_other,fpm=fpm,energy_low=energy_rng[0], energy_high=energy_rng[1], 
                                                      tmrng=[time_bins[frame_to_corr], time_bins[frame_to_corr+1]])
                                t_1_c = datetime.datetime.strptime(time_bins[frame_to_corr], '%Y/%m/%d, %H:%M:%S')
                                t_2_c = datetime.datetime.strptime(time_bins[frame_to_corr+1], '%Y/%m/%d, %H:%M:%S')
                                #livetime correction
                                ltimes_in_rangeA_c = []
                                for lv in range(len(lvt_t1)):
                                    if ((lvt_t1[lv]>=t_1_c) & (lvt_t1[lv]<t_2_c)):
                                        ltimes_in_rangeA_c.append(lvt_lvt1[lv])
                                ave_livetime_A_c = np.average(ltimes_in_rangeA_c)

                                nustar_map_c = nustar.map.make_sunpy(cleanevt_c, hdr, norm_map=False)
                                nustar_map_normdata_c = nustar_map_c.data / ((t_2_c - t_1_c).total_seconds() * ave_livetime_A_c)
                       
                                nustar_map_first_c = sunpy.map.Map(nustar_map_normdata_c, nustar_map_c.meta)
                                if (A_and_B == True):
                                    #livetime correction B
                                    ltimes_in_rangeB_c = []
                                    for lv in range(len(lvt_t2)):
                                        if ((lvt_t2[lv]>=t_1_c) & (lvt_t2[lv]<t_2_c)):
                                            ltimes_in_rangeB_c.append(lvt_lvt2[lv])
                                    ave_livetime_B_c = np.average(ltimes_in_rangeB_c)

                                    nustar_map_other_c = nustar.map.make_sunpy(cleanevt_other_c, hdr_other, norm_map=False)
                                    nustar_map_normdata_other_c = nustar_map_other_c.data / ((t_2_c - t_1_c).total_seconds() * ave_livetime_B_c)
                                    nustar_map_other_c = sunpy.map.Map(nustar_map_normdata_other_c, nustar_map_other_c.meta)

                                #make submap for quickness to get the shift
                                bl = SkyCoord((submap[0]-100)*u.arcsec, (submap[1]-100)*u.arcsec, frame=nustar_map_c.coordinate_frame)
                                tr = SkyCoord((submap[2]+100)*u.arcsec, (submap[3]+100)*u.arcsec, frame=nustar_map_c.coordinate_frame)
                                submap_first = nustar_map_first_c.submap(bl,tr)
                                dataA = submap_first.data
                                if (A_and_B == True):
                                    submap_other = nustar_map_other_c.submap(bl,tr)
                                    dataB = submap_other.data
                                    del submap_other
                                    del nustar_map_normdata_other_c
                                    del nustar_map_other_c

                                del nustar_map_first_c
                                del nustar_map_normdata_c
                                del submap_first
            if (A_and_B == True):
                fpm = 'A&B'
            #del nustar_map_normdata
            
            background_in_trng_data = []
            background_in_trng_header = []
            aia_map = 0
            for f in aia_files: #make a list of the aia files in the range
                aia_time_string = f[3:7]+'/'+f[7:9]+'/'+f[9:11]+', '+f[12:14]+':'+ f[14:16]+':'+ f[16:18]
                aia_time = datetime.datetime.strptime(aia_time_string, '%Y/%m/%d, %H:%M:%S')
                t_bin_edge1 = datetime.datetime.strptime(time_bins[t], '%Y/%m/%d, %H:%M:%S')
                t_bin_edge2 = datetime.datetime.strptime(time_bins[t+1], '%Y/%m/%d, %H:%M:%S')
                if t_bin_edge1 <= aia_time < t_bin_edge2:
                    aia_map = sunpy.map.Map(aia_dir + f);
                    if resample_aia != []: #resample the data here to the new dimensions given now
                        dimensions = u.Quantity([resample_aia[0], resample_aia[1]], u.pixel)
                        aia_map = aia_map.resample(dimensions, method='linear');
                    background_in_trng_data.append(aia_map.data)
                    background_in_trng_header.append(aia_map.meta)
                    del aia_map
                    if background_time == 'begin': #only need the first one if 'begin' is set so save time and stop here
                        break

            if len(background_in_trng_data) == 0:
                print(f'\rNo AIA data in this time range: {time_bins[t]}, {time_bins[t+1]}.', end='')
                continue

            background_in_trng_data = np.array(background_in_trng_data)
            num_of_maps = len(background_in_trng_data)
            if background_time == 'begin':
                bg = background_in_trng_data[0]
                aia_map = sunpy.map.Map(bg, background_in_trng_header[0])
            elif background_time == 'middle':
                mid_of_maps = num_of_maps // 2
                bg = background_in_trng_data[mid_of_maps]
                aia_map = sunpy.map.Map(bg, background_in_trng_header[mid_of_maps])
            elif background_time == 'end':
                bg = background_in_trng_data[-1]
                aia_map = sunpy.map.Map(bg, background_in_trng_header[-1])
            elif background_time == 'average':
                bg = background_in_trng_data.sum(axis=0) / num_of_maps
                aia_map = sunpy.map.Map(bg, background_in_trng_header[0])
            else:
                print('Choose where the background map comes from: begin, middle, end, or average of time range')

            del bg
            del background_in_trng_data
            del background_in_trng_header

            ## can now cross correlate all the maps
            if ('A_shift' not in locals()) and ('B_shift' not in locals()):
                A_shift = None
                B_shift = None

            if deconvolve == True:
                bl_c = SkyCoord((submap[0]+0.01)*u.arcsec, (submap[1]+0.01)*u.arcsec, frame=nustar_map.coordinate_frame)
                tr_c = SkyCoord((submap[2]-0.01)*u.arcsec, (submap[3]-0.01)*u.arcsec, frame=nustar_map.coordinate_frame)
                #0.01 arcsec padding to make sure the contours aren't cut off and so that the final plot won't have weird 
                #blank space bits
                dconvfA = '/home/kris/Desktop/link_to_kris_ganymede/old_scratch_kris/data_and_coding_folder/nustar_psfs/nuA2dpsfen1_20100101v001.fits'
                dconvfB = '/home/kris/Desktop/link_to_kris_ganymede/old_scratch_kris/data_and_coding_folder/nustar_psfs/nuB2dpsfen1_20100101v001.fits'
                psfhdu = fits.open(dconvfA)
                psfA = psfhdu[1].data
                psfhdu.close()
                if A_and_B == True:
                    psfhdu = fits.open(dconvfB)
                    psfB = psfhdu[1].data
                    psfhdu.close()
                elif fpm == 'B':
                    psfhdu = fits.open(dconvfB)
                    psfA = psfhdu[1].data
                    psfhdu.close()

            #print(manual_pixel_shift, nustar_shift, t, frame_to_correlate)
            if (manual_pixel_shift == False) and ((nustar_shift == 'cc') or (A_and_B == True)):
                if (t == 0) or (frame_to_correlate == 'individual'):

                    background_in_trng_data_c = []
                    background_in_trng_header_c = []
                    aia_map_c = 0
                    for f in aia_files: #make a list of the aia files in the range
                        aia_time_string = f[3:7]+'/'+f[7:9]+'/'+f[9:11]+', '+f[12:14]+':'+ f[14:16]+':'+ f[16:18]
                        aia_time = datetime.datetime.strptime(aia_time_string, '%Y/%m/%d, %H:%M:%S')
                        t_bin_edge1 = datetime.datetime.strptime(time_bins[frame_to_corr], '%Y/%m/%d, %H:%M:%S')
                        t_bin_edge2 = datetime.datetime.strptime(time_bins[frame_to_corr+1], '%Y/%m/%d, %H:%M:%S')
                        if t_bin_edge1 <= aia_time < t_bin_edge2:
                            aia_map_c = sunpy.map.Map(aia_dir + f);
                            background_in_trng_data_c.append(aia_map_c.data)
                            background_in_trng_header_c.append(aia_map_c.meta)
                            del aia_map_c
                            if background_time == 'begin': #only need the first one if 'begin' is set so save time and stop here
                                break

                    if len(background_in_trng_data_c) == 0:
                        print(f'\rNo AIA data in time range to cross-corrrelate: {time_bins[frame_to_corr]}, {time_bins[frame_to_corr+1]}.', end='')
                        break

                    background_in_trng_data_c = np.array(background_in_trng_data_c)
                    num_of_maps_c = len(background_in_trng_data_c)
                    if background_time == 'begin':
                        aia_map_c = sunpy.map.Map(background_in_trng_data_c[0], background_in_trng_header_c[0])
                    elif background_time == 'middle':
                        mid_of_maps_c = num_of_maps_c // 2
                        aia_map_c = sunpy.map.Map(background_in_trng_data_c[mid_of_maps], background_in_trng_header_c[mid_of_maps])
                    elif background_time == 'end':
                        aia_map_c = sunpy.map.Map(background_in_trng_data_c[-1], background_in_trng_header_c[-1])
                    elif background_time == 'average':
                        ave_c = background_in_trng_data_c.sum(axis=0) / num_of_maps_c
                        aia_map_c = sunpy.map.Map(ave_c, background_in_trng_header_c[0])
                    else:
                        print('Choose where the background map comes from: begin, middle, end, or average of time range')

                    del background_in_trng_data_c
                    del background_in_trng_header_c

                    submap_aia_c = aia_map_c.submap(bl,tr)
                    data_for_corr = submap_aia_c.data
                    del submap_aia_c

                    #must rescale aia to suit the nustar images
                    data_for_corr_A = resize(data_for_corr, np.shape(dataA))
                    if deconvolve == True:
                        dataA = restoration.richardson_lucy(dataA, psfA, iterations=it, clip=False)
                    else:
                        dataA = ndimage.gaussian_filter(dataA, gauss_sigma, mode='nearest')

                    corr_with_A = signal.correlate2d(data_for_corr_A, dataA, boundary='symm', mode='same')
                    y_A, x_A = np.unravel_index(np.argmax(corr_with_A), corr_with_A.shape)  # find the match
                    x_pix_shift_A = -(np.shape(data_for_corr_A)[1]/2 - x_A) #negative because the positive number means shift to the left/down
                    y_pix_shift_A = -(np.shape(data_for_corr_A)[0]/2 - y_A)
                    A_shift = np.array([x_pix_shift_A, y_pix_shift_A]) 
                    all_A_shift.append(A_shift)
 
                    if A_and_B == True:
                        data_for_corr_B = resize(data_for_corr, np.shape(dataB))
                        if deconvolve == True:
                            dataB = restoration.richardson_lucy(dataB, psfB, iterations=it, clip=False)
                        else:
                            dataB = ndimage.gaussian_filter(dataB, gauss_sigma, mode='nearest')

                        corr_with_B = signal.correlate2d(data_for_corr_B, dataB, boundary='symm', mode='same')
                        y_B, x_B = np.unravel_index(np.argmax(corr_with_B), corr_with_B.shape)  # find the match
                        x_pix_shift_B = -(np.shape(data_for_corr_B)[1]/2 - x_B) #negative because the positive number means shift to the left/down
                        y_pix_shift_B = -(np.shape(data_for_corr_B)[0]/2 - y_B)
                        B_shift = np.array([x_pix_shift_B, y_pix_shift_B])
                        all_B_shift.append(B_shift)
                        del corr_with_B
                        del dataB

                    del data_for_corr
                    del dataA
                    del corr_with_A
                #print('A shift is ', A_shift)
                #print('shape A is ', [np.shape(dataA)[0], np.shape(dataA)[1]])
                #print('aia shape is ', [np.shape(data_for_corr_A)[0], np.shape(data_for_corr_A)[1]])
                #print('corr shape is ', [np.shape(corr_with_A)[0], np.shape(corr_with_A)[1]])
                #print('Corr max is ', [y_A, x_A])
                #fig, (ax1, ax2, ax3) = plt.subplots(1,3)
                #ax1.imshow(data_for_corr_A)
                #ax2.imshow(dataA)
                #ax3.imshow(corr_with_A)
                #plt.show()

                shift_evt_A = krispy.nustardo.shift(cleanevt, pix_xshift=A_shift[0], pix_yshift=A_shift[1])
                nustar_map_A = nustar.map.make_sunpy(shift_evt_A, hdr, norm_map=False)
                if deconvolve == True:
                    nustar_map_A = nustar_map_A.submap(bl_c,tr_c)
                    deconv_A = restoration.richardson_lucy(nustar_map_A.data, psfA, iterations=it, clip=False)
                    nustar_map_A = sunpy.map.Map(deconv_A, nustar_map_A.meta)
                final_map_meta = nustar_map_A.meta
                nustar_map_normdata_A = nustar_map_A.data / ((t_2 - t_1).total_seconds() * ave_livetime_A)

                if A_and_B == True:
                    shift_evt_B = krispy.nustardo.shift(cleanevt_other, pix_xshift=B_shift[0], pix_yshift=B_shift[1])
                    nustar_map_B = nustar.map.make_sunpy(shift_evt_B, hdr_other, norm_map=False)
                    if deconvolve == True:
                        nustar_map_B = nustar_map_B.submap(bl_c,tr_c)
                        deconv_B = restoration.richardson_lucy(nustar_map_B.data, psfB, iterations=it, clip=False)
                        nustar_map_B = sunpy.map.Map(deconv_B, nustar_map_B.meta)
                    nustar_map_normdata_B = nustar_map_B.data / ((t_2 - t_1).total_seconds() * ave_livetime_B)
                    
                    nustar_map_normdata = nustar_map_normdata_A + nustar_map_normdata_B
                else:
                    nustar_map_normdata = nustar_map_normdata_A
                if nustar_shift != 'cc':
                    nustar_shift = None

            if AB_pixshift != None:
                if len(AB_pixshift[0]) >= 1:
                    if frame_to_correlate == 'individual':
                        A_shift = AB_pixshift[0][t]
                    else:
                        A_shift = AB_pixshift[0][0]

                    shift_evt_A = krispy.nustardo.shift(cleanevt, pix_xshift=A_shift[0], pix_yshift=A_shift[1])
                    
                    nustar_map_A = nustar.map.make_sunpy(shift_evt_A, hdr, norm_map=False)
                    if deconvolve == True:
                        nustar_map_A = nustar_map_A.submap(bl_c,tr_c)
                        array = nustar_map_A.data
                        deconv_A = restoration.richardson_lucy(array, psfA, iterations=it, clip=False)
                        nustar_map_A = sunpy.map.Map(deconv_A, nustar_map_A.meta)
                    final_map_meta = nustar_map_A.meta
                    nustar_map_normdata_A = nustar_map_A.data / ((t_2 - t_1).total_seconds() * ave_livetime_A)
                if len(AB_pixshift[1]) >= 1:
                    if frame_to_correlate == 'individual':
                        B_shift = AB_pixshift[1][t]
                    else:
                        B_shift = AB_pixshift[1][0]
                    
                    shift_evt_B = krispy.nustardo.shift(cleanevt_other, pix_xshift=B_shift[0], pix_yshift=B_shift[1])
                    nustar_map_B = nustar.map.make_sunpy(shift_evt_B, hdr_other, norm_map=False)
                    if deconvolve == True:
                        nustar_map_B = nustar_map_B.submap(bl_c,tr_c)
                        deconv_B = restoration.richardson_lucy(nustar_map_B.data, psfB, iterations=it, clip=False)
                        nustar_map_B = sunpy.map.Map(deconv_B, nustar_map_B.meta)
                    nustar_map_normdata_B = nustar_map_B.data / ((t_2 - t_1).total_seconds() * ave_livetime_B)
                if A_and_B == True:
                    nustar_map_normdata = nustar_map_normdata_A + nustar_map_normdata_B
                else:
                    nustar_map_normdata = nustar_map_normdata_A
                if nustar_shift != 'cc':
                    nustar_shift = None 

            if deconvolve == True:
                dd = nustar_map_normdata
            else:
                dd=ndimage.gaussian_filter(nustar_map_normdata, gauss_sigma, mode='nearest');
            
            # Tidy things up before plotting
            dmin=1e-3
            dmax=1e1
            dd[dd < dmin]=0
            nm=sunpy.map.Map(dd, final_map_meta);

            del nustar_map_normdata
                
            # Let's shift it ############################################################################################
            if (nustar_shift != None) and (nustar_shift != 'cc'):
                shifted_nustar_map = nm.shift(nustar_shift[0]*u.arcsec, nustar_shift[1]*u.arcsec)
            else:
                shifted_nustar_map = nm
            #print(shifted_nustar_map.data)
            #print(shifted_nustar_map.meta)

            # Submap to plot ############################################################################################
            if deconvolve == False:
                bl_s = SkyCoord((submap[0]+0.01)*u.arcsec, (submap[1]+0.01)*u.arcsec, frame=shifted_nustar_map.coordinate_frame)
                tr_s = SkyCoord((submap[2]-0.01)*u.arcsec, (submap[3]-0.01)*u.arcsec, frame=shifted_nustar_map.coordinate_frame)
                #0.01 arcsec padding to make sure the contours aren't cut off and so that the final plot won't have weird 
                #blank space bits
                shifted_nustar_map = shifted_nustar_map.submap(bl_s,tr_s)
            
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [contour_colour,contour_colour])
            shifted_nustar_map.plot_settings['norm'] = colors.LogNorm(vmin=dmin,vmax=dmax)
            shifted_nustar_map.plot_settings['cmap'] = cmap

            #############################################################################################################
            bl_fi = SkyCoord(submap[0]*u.arcsec, submap[1]*u.arcsec, frame=aia_map.coordinate_frame)
            tr_fi = SkyCoord(submap[2]*u.arcsec, submap[3]*u.arcsec, frame=aia_map.coordinate_frame)
            #0.01 arcsec padding
            smap = aia_map.submap(bl_fi,tr_fi)    

            if save_bg == True: # do you want to keep the background used to be returned
                saved_backgrounds.append(smap.data)
            
            del aia_map

            if iron18 == True:
                smap.plot_settings['cmap'] = plt.cm.GnBu

            fig = plt.figure(figsize=(9,8));
            compmap = sunpy.map.Map(smap, composite=True);
            del smap
            compmap.add_map(shifted_nustar_map);
            
            if contour_fmt == 'percent':
                compmap.set_levels(1,contour_lvls,percent = True);
            elif contour_fmt == 'actual_values':
                compmap.set_levels(1,contour_lvls,percent = False);

            if cmlims != []:
                if cmlims[0] <= 0: #vmin > 0 or error
                    cmlims[0] = 0.1
                    compmap.plot(vmin=cmlims[0], vmax=cmlims[1]);  
                else:
                    compmap.plot(vmin=cmlims[0], vmax=cmlims[1]); 
            elif cmlims == []:
                compmap.plot();
            
            #plt.colorbar(label='DN s$^{-1}$');
            
            char_to_arcsec = 3.0 #-ish
            border = 2
            if energy_rng[1] >= 79:
                clabel_title = f'NuSTAR >{energy_rng[0]} keV FPM'+fpm+' '+chu
            else:
                clabel_title = f'NuSTAR {energy_rng[0]}-{energy_rng[1]} keV FPM'+fpm+' '+chu
            #for sig figures
            max_length = np.max([len(str(c))-len(str(float(int(c))))+1 for c in contour_lvls])
            if max_length < 0:
                max_length = 0
            if contour_fmt == 'percent':
                x = submap[2] - border
                y = submap[1] + border        
                for lvl in contour_lvls[::-1]:
                    clabel = str(format(lvl, '.'+str(max_length)+'f')) + '%'
                    plt.annotate(clabel ,(x,y), ha='right')
                    y += 6
                plt.annotate(clabel_title ,(x,y), ha='right')
            elif contour_fmt == 'actual_values':
                x = submap[2] - border
                y = submap[1] + 2 
                for lvl in contour_lvls[::-1]:
                    clabel = str(format(lvl, '.'+str(max_length)+'f')) + ' Counts s$^{-1}$'
                    plt.annotate(clabel ,(x,y), ha='right') #, color='r')
                    y += 6 
                plt.annotate(clabel_title ,(x,y), ha='right') #, color='r')

            plt.title(f'FeXVIII at {time_bins[t]} to {time_bins[t+1][12:]}');

            if save_inc == False:
                plt.savefig(save_dir + f'nustar_contours{d}_on_iron18_chu{chu}_fpm{fpm}.png', dpi=300, 
                            bbox_inches='tight')
            elif save_inc == True:
                plt.savefig(save_dir + 'contours{:04d}.png'.format(d), dpi=300, bbox_inches='tight')
            d+=1
                
            plt.close(fig)
            print(f'\rSaved {d} submap(s).', end='')
            
            max_contours.append(shifted_nustar_map.data.max())
            del shifted_nustar_map

        elif (len(cleanevt) == 0 and aia == 'all') or aia == 'solo': #just AIA maps wih the same settings
            background_in_trng_data = []
            background_in_trng_header = []
            aia_map = 0
            for f in aia_files:
                aia_time_string = f[3:7]+'/'+f[7:9]+'/'+f[9:11]+', '+f[12:14]+':'+ f[14:16]+':'+ f[16:18]
                aia_time = datetime.datetime.strptime(aia_time_string, '%Y/%m/%d, %H:%M:%S')
                t_bin_edge1 = datetime.datetime.strptime(time_bins[t], '%Y/%m/%d, %H:%M:%S')
                t_bin_edge2 = datetime.datetime.strptime(time_bins[t+1], '%Y/%m/%d, %H:%M:%S')
                
                if t_bin_edge1 <= aia_time < t_bin_edge2:
                    aia_map = sunpy.map.Map(aia_dir + f);
                    if resample_aia != []:
                        dimensions = u.Quantity([resample_aia[0], resample_aia[1]], u.pixel)
                        aia_map = aia_map.resample(dimensions, method='linear');
                    background_in_trng_data.append(aia_map.data)
                    background_in_trng_header.append(aia_map.meta)
                    del aia_map
                    if background_time == 'begin':
                        break
                
            background_in_trng_data = np.array(background_in_trng_data)
            num_of_maps = len(background_in_trng_data)
            if background_time == 'begin':
                aia_map = sunpy.map.Map(background_in_trng_data[0], background_in_trng_header[0])
            elif background_time == 'middle':
                mid_of_maps = num_of_maps // 2
                aia_map = sunpy.map.Map(background_in_trng_data[mid_of_maps], background_in_trng_header[mid_of_maps])
            elif background_time == 'end':
                aia_map = sunpy.map.Map(background_in_trng_data[-1], background_in_trng_header[-1])
            elif background_time == 'average':
                aia_map = sunpy.map.Map(background_in_trng_data.sum()/num_of_maps, background_in_trng_header[0])
            else:
                print('Choose where the background map comes from: begin, middle, end, or average of time range')

            del background_in_trng_data
            del background_in_trng_header

            if aia_map == 0:
                print(f'\rNo AIA data in this time range: {time_bins[t]}, {time_bins[t+1]}.', end='')
                continue
                
            bl_fi = SkyCoord(submap[0]*u.arcsec, submap[1]*u.arcsec, frame=aia_map.coordinate_frame)
            tr_fi = SkyCoord(submap[2]*u.arcsec, submap[3]*u.arcsec, frame=aia_map.coordinate_frame)
            #0.1 arcsec padding
            smap = aia_map.submap(bl_fi,tr_fi)    

            if save_bg == True: # do you want to keep the background used to be returned
                saved_backgrounds.append(smap.data)  
            
            if iron18 == True:
                smap.plot_settings['cmap'] = plt.cm.GnBu

            fig = plt.figure(figsize=(9,8));
            compmap = sunpy.map.Map(smap, composite=True)
            
            if cmlims != []:
                if cmlims[0] <= 0: #vmin > 0 or error
                    cmlims[0] = 0.1
                    compmap.plot(vmin=cmlims[0], vmax=cmlims[1]);  
                else:
                    compmap.plot(vmin=cmlims[0], vmax=cmlims[1]);
            elif cmlims == []:
                compmap.plot();
                
            plt.colorbar(label='DN s$^{-1}$');
            
            plt.title(f'FeXVIII at {time_bins[t][:-7]} to {time_bins[t+1][10:18]}')
            print(f'{time_bins[t][:-7]} to {time_bins[t+1][10:18]}')         

            if save_inc == False:
                plt.savefig(save_dir + f'nustar_contours{d}_on_iron18_chu{chu}_fpm{fpm}.png', dpi=300, 
                            bbox_inches='tight')
            elif save_inc == True:
                plt.savefig(save_dir + 'maps{:04d}.png'.format(d), dpi=300, bbox_inches='tight')
            d+=1
                
            plt.close(fig)
            del smap
            del aia_map
            print(f'\rSaved {d} submap(s).', end='')
        
        else:
            print(f'\rNo NuSTAR data in this time range: {time_bins[t]} to {time_bins[t+1]}.', end='')

    if fpm == 'B' and type(B_shift) == type(None) and nustar_shift == 'cc' and type(A_shift) != type(None):
        B_shift = A_shift
        A_shift = None
    
    print('\nLook everyone, it\'s finished!')
    return {'max_contour_levels': max_contours, 'last_incremental_value': d-1, 'nustar_shift': nustar_shift, 'cc_A_and_B_pixel_shifts': [all_A_shift, all_B_shift], 'bgs_saved':saved_backgrounds} 
    #helps find the values for the contour lines and the last number padded for the incremental saves




#make maps from the fits file
############################################### **Warning** ###############################################
###########################################################################################################
###### This function has a memory leak and I don't know where yet so don't run a huge list of files #######
###### through it at once! ################################################################################
###########################################################################################################
###### This may be fixed by running it through the command line. IPython environments keep ################
###### references, especially to plots, even after they're not used meaning garbage collection ############
###### wouldn't get rid of them. ##########################################################################
###########################################################################################################
###### Delete the aia_map as soon as you can, this solves the problem. ####################################
###########################################################################################################
def aiamaps_from_dir(fits_dir, out_dir, savefile_fmt='.png', dpi=300, cmlims = [], submap=[], rectangle=[], 
                     resample=[], save_inc=False, fexviii=False):
    """Takes a directory with fits files, constructs a map or submap of the full observation with/without a rectangle and
    saves the image in the requested directory.
    
    Parameters
    ----------
    data_dir : Data directory
            The directory which contains the list of fits files from the AIA. Must end with a '/'.
    
    out_dir : Save directory
            The directory in which the new fits files are saved. Must end with a '/'.
        
    savefile_fmt : Str
            File extension for the saved file's format, e.g. '.png', '.jpg', '.pdf', etc.
            Default: '.png'
            
    dpi : Int
            Dots per inch for the save file resolution, e.g. 100, 150, 300, etc.
            Default: 300

        cmlims : One-dimensional list/array of type float or int, length 2
            Limits of the colourmap, e.g. [vmin, vmax].
            
    submap : One-dimensional list/array, length 4
            Contains the bottom left (bl) and top right (tr) coordinates for a submap, e.g. [blx,bly,trx,try]. Must be 
            in arcseconds, of type float or integer and NOT an arcsec object. 
            
    rectangle : One-dimensional list/array, length 4
            Contains the bottom left (bl) and top right (tr) coordinates to draw a rectangle on the constructed map,
            e.g. [blx,bly,trx,try]. Must be in arcseconds, of type float or integer and NOT an arcsec object.
            
    resample : One-dimensional list/array of type int, length 2
            The number of pixels the map should get rebinned to, e.g. [1000,1000].
            
    save_inc : Bool
            Indicates whether or not the save file should be named with respect to the file it was produced from or be
            named incrementally, e.g. AIA123456_123456_1234.png or map_000 respectively.
            Default: False
    
    fexviii : Bool
            Indicates whether or not the save file is the iron 18 channel.
            Default: False
            
    Returns
    -------
    AIA maps saved to the requested directory.
    """
    warn = input('This function has a memory leak and shouldn\'t be used for loads of files.\
        Do you still want to use this function (yes or no)? ')
    if warn == 'yes':
        print('OK, but I warned you!')
    elif warn != 'yes':
        assert 0 == 1, 'Deciding not to use this function, try and use the function aiamaps() instead.'

    files = list(os.listdir(fits_dir))
    files = file_working.only_fits(files)
    files.sort() #orders the files
    
    num_f = len(files) #total number of images
    d = 0 #save image with an incremental counter
    done = 0 #counter for the number of images created
    
    for f in files:
        aia_map = sunpy.map.Map(fits_dir + f)
        
        if fexviii == True: #set colourmap and title for Iron 18
            aia_map.plot_settings['cmap'] = plt.cm.GnBu
            
            time_dt = datetime.datetime.strptime(aia_map.meta['date_obs'], '%Y-%m-%dT%H:%M:%S.%f')
            YMDhms = []
            for num in [time_dt.year, time_dt.month, time_dt.day, time_dt.hour, time_dt.minute, time_dt.second]:
                if num < 10:
                    YMDhms.append(f'0{num}') #pads a number if it doesn't have two digits (except for the year)
                else:
                    YMDhms.append(f'{num}')
            time = f'{YMDhms[0]}-{YMDhms[1]}-{YMDhms[2]} {YMDhms[3]}:{YMDhms[4]}:{YMDhms[5]}'
        
        if resample != []: #rebins the map
            dimensions = u.Quantity(resample, u.pixel)
            aia_map = aia_map.resample(dimensions, method='linear')
        
        if submap != []: #creates a submap
            bottom_left = SkyCoord(submap[0]*u.arcsec, submap[1]*u.arcsec, frame=aia_map.coordinate_frame)
            top_right = SkyCoord(submap[2]*u.arcsec, submap[3]*u.arcsec, frame=aia_map.coordinate_frame)
            aia_submap = aia_map.submap(bottom_left, top_right)
            
            if fexviii == True: #set colourmap and title for Iron 18
                aia_submap.plot_settings['cmap'] = plt.cm.GnBu
            
            fig = plt.figure()
            if cmlims != []: #if there are colourmap limits, use them
                aia_submap.plot(vmin=cmlims[0], vmax=cmlims[1]);
            else:
                aia_submap.plot();
            plt.colorbar()
            
            if fexviii == True: #sets title for Iron 18
                plt.title('AIA FeXVIII ' + time)
                 
            if rectangle != []: #if a rectangle is specified, make it
                bl_rect = SkyCoord(rectangle[0]*u.arcsec, rectangle[1]*u.arcsec, frame=aia_map.coordinate_frame)
                length = rectangle[2] - rectangle[0]
                height = rectangle[3] - rectangle[1]
                if fexviii == True:
                    aia_submap.draw_rectangle(bl_rect, length*u.arcsec, height*u.arcsec, color = 'black');
                else:
                    aia_submap.draw_rectangle(bl_rect, length*u.arcsec, height*u.arcsec);
                
            #plt.show(fig);
            
            if save_inc == True: #saves the images with an incremental naming system (3 digit padding)
                fig.savefig(out_dir + 'map_{:03d}'.format(d) + savefile_fmt, dpi = dpi, bbox_inches='tight')
                #bbox_inches=removes unnecessary white space
                d+=1
            elif save_inc == False:
                fig.savefig(out_dir + f[:-5] + savefile_fmt, dpi = dpi, bbox_inches='tight')
                #save image w/ original file with diff. extension, bbox_inches=removes unnecessary white space
                
            plt.close(fig)

            del aia_submap

            done += 1
            print('\rSaved {} submap(s) of {}'.format(done, num_f), end='')
        else: #just makes the full map
            fig = plt.figure();
            if cmlims != []: #if there are colourmap limits, use them
                aia_map.plot(vmin=cmlims[0], vmax=cmlims[1]);
            else:
                aia_map.plot();
            plt.colorbar();
            
            if fexviii == True: #sets title for Iron 18
                plt.title('AIA FeXVIII ' + time)
            
            if rectangle != []: #if a rectangle is specified, make it
                bl_rect = SkyCoord(rectangle[0]*u.arcsec, rectangle[1]*u.arcsec, frame=aia_map.coordinate_frame)
                length = rectangle[2] - rectangle[0]
                height = rectangle[3] - rectangle[1]
                if fexviii == True:
                    aia_map.draw_rectangle(bl_rect, length*u.arcsec, height*u.arcsec, color = 'black');
                else:
                    aia_map.draw_rectangle(bl_rect, length*u.arcsec, height*u.arcsec);
            
            #plt.show(fig);
            
            if save_inc == True: #saves the images with an incremental naming system (3 digit padding)
                fig.savefig(out_dir + 'map_{:03d}'.format(d) + savefile_fmt, dpi = dpi, bbox_inches='tight')
                #bbox_inches=removes unnecessary white space
                d+=1
            elif save_inc == False:
                fig.savefig(out_dir + f[:-5] + savefile_fmt, dpi = dpi, bbox_inches='tight') 
                #save image w/ original file with diff. extension, bbox_inches=removes unnecessary white space
                
            plt.close(fig);

            del aia_map

            done += 1
            print('\rSaved {} map(s) of {}'.format(done, num_f), end='')
    print('\nAll files saved!')


#make composite images from the aia fits files
def overlay_aiamaps(directory, second_directory, save_directory, submap=None, cmlims = None,cmlims2=[], rectangle=None, rectangle_colour=None, save_inc=True, iron='',
                    cm_scale='Normalize', res=None, in_order=True, alphas=[0.5,0.5], lvls=None, dpi=300, time_filter=None):      
    """Takes a directory with fits files, constructs a map or submap of the full observation with/without a rectangle and
    saves the image in the requested directory.
    
    Parameters
    ----------
    directory : Data directory
            The directory which contains the list of fits files from the AIA. Must end with a '/'. Can now accept a list of directories
            of multiple data directories, e.g. [dir1,dir2,...]. Make sure they're in order.

    second_directory : Data directory
            The directory which contains the list of fits files for the second maps from the AIA. Must end with a '/'. Can now accept a list of directories
            of multiple data directories, e.g. [dir1,dir2,...]. Make sure they're in order.
    
    save_directory : Save directory
            The directory in which the new fits files are saved. Must end with a '/'. Can now accept a list of directories
            of where to save all of the images, e.g. [sav1,sav2,...].
            
    submap : One-dimensional list/array, length 4
            Contains the bottom left (bl) and top right (tr) coordinates for a submap, e.g. [blx,bly,trx,try]. Must be 
            in arcseconds, of type float or integer and NOT an arcsec object.

    cmlims : One-dimensional list/array of type float or int, length 2
            Limits of the colourmap, e.g. [vmin, vmax]. 

    cmlims2 : One-dimensional list/array of type float or int, length 2
            Limits of the colourmap for the second map, e.g. [vmin, vmax]. 
            
    rectangle : two-dimensional list/array, shape=n,4
            Contains lists of the bottom left (bl) and top right (tr) coordinates to draw a rectangle on the constructed 
            map, e.g. [[blx1,bly1,trx1,try1], [blx2,bly2,trx2,try2], ...]. Must be in arcseconds, of type float or 
            integer and NOT an arcsec object.

    rectangle_colour : one-dimensional list of strings
            Contains the colour or colours you want associated with the rectangle you want to plot. Must contain one colour or
            the same number of colours as there are rectangles.
            Default: ["black"] for iron, else ["white"] 
            
    save_inc : Bool
            Indicates whether or not the save file should be named with respect to the file it was produced from or be
            named incrementally, e.g. AIA123456_123456_1234.png or map_000 respectively.
            Default: True
    
    iron : Str
            Indicates whether or not the save file is the iron 16 or 18 channel. Set to '16' or '18'.
            Default: ''

    cm_scale : Str
            Scale for the colour bar for the first maps given. Set to 'Normalize' or 'LogNorm'.
            Default: 'Normalize'

    res : float
            A float <1 which has the resolution of the image reduced, e.g. 0.5 produces an image at 50% resolution to the original.
            Default: None

    in_order : Bool
            If True then the second map will be overlain on the first, if False the this order switches. This is important as the first maps 
            dictate the frames created, i.e. the first map is paired with its closest (in time) second map, so you might want all the frames of
            the first lot of maps but have that go on top of the second maps when plotted.
            Default: True 

    alphas : One-dimensional list/array of type float or int, length 2
            Alpha channel for map1 and map2, in order of plot order, respectively, e.g. [a1, a2]. 
    
    lvls : 1D array
            Contour levels as true values (not percentages) fort the second overlying map. If lvls is set to None then the two maps are
            just combined.
            Default: None

    dpi : Int
            Express the dots per inch that the images produced should have.
            Default: 300

    time_filter : list, 2 strings
            If you provide a directory but only want maps made from a certain time range, e.g. 
            ["%Y/%m/%d, %H:%M:%S", "%Y/%m/%d, %H:%M:%S"].
            Defualt: None

    Returns
    -------
    AIA maps saved to the requested directory (so doesn't really return anythin).
    """

    rectangle_colour = ["black"] if rectangle_colour is None else rectangle_colour
    if type(rectangle_colour) is not list:
        rectangle_colour = [rectangle_colour]

    rectangle = [] if rectangle is None else rectangle
    cmlims = [] if cmlims is None else cmlims

    np.seterr(divide='ignore', invalid='ignore') #ignore warnings resulting from missing header info
    warnings.simplefilter('ignore', Warning)

    matplotlib.rcParams['font.sans-serif'] = "Arial" #sets up plots
    matplotlib.rcParams['font.family'] = "sans-serif"
    matplotlib.rcParams['font.size'] = 12

    title_addition = ''
    rescale_cml = False
    first_time_through = True

    if cmlims2 == []:
        cmlims2 = cmlims

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

    if type(second_directory) == str:
        second_directory = [second_directory] # no point in writing out the loop below twice
        
    if type(second_directory) == list: # directory lists must be in time order for now, hopefully this will change (and look better with *args?)
        second_directory_with_files = []
        for _d in second_directory:
            _aia_files = os.listdir(_d)
            _aia_files.sort()
            _aia_files = file_working.only_fits(_aia_files)
            _directory_with_files = [_d+f for f in _aia_files]
            second_files = _aia_files
            second_directory_with_files += _directory_with_files

    if time_filter is not None:
        _q = contour.Contours()

        _file_times = _q.aia_file_times(aia_file_dir="", aia_file_list=directory_with_files)
        _in_time = _q.useful_time_inds(times_list=_file_times, time_interval=time_filter)
        directory_with_files = np.array(directory_with_files)[_in_time]

        _file_times = _q.aia_file_times(aia_file_dir="", aia_file_list=second_directory_with_files)
        _in_time = _q.useful_time_inds(times_list=_file_times, time_interval=time_filter)
        second_directory_with_files = np.array(second_directory_with_files)[_in_time]

    no_of_files = len(directory_with_files)

    time_of_second_lot = []
    for td in second_directory_with_files:
        z = sunpy.map.Map(td)
        # but HMI does time in TAI (international atomic time) I hear you cry? Well 'date-obs' is in utc for both
        time_of_second_lot.append(z.meta['date-obs'])
        del z

    d = 0

    for f in range(no_of_files):

        aia_map = sunpy.map.Map(directory_with_files[f])

        #find closest overlay file
        first_image_time = data_handling.getTimeFromFormat(aia_map.meta['date-obs'])
        
        distance = [abs(first_image_time - data_handling.getTimeFromFormat(over)) for over in time_of_second_lot]
        closest = np.argmin(distance)
        second_aia_map = sunpy.map.Map(second_directory_with_files[closest])
        
        if submap != None:
            bl_fi = SkyCoord(submap[0]*u.arcsec, submap[1]*u.arcsec, frame=aia_map.coordinate_frame)
            tr_fi = SkyCoord(submap[2]*u.arcsec, submap[3]*u.arcsec, frame=aia_map.coordinate_frame)
        
            smap = aia_map.submap(bl_fi,tr_fi) 
            second_smap = second_aia_map.submap(bl_fi,tr_fi)
            del aia_map
            del second_aia_map
        else:
            smap = aia_map
            second_smap = second_aia_map

        if res is not None:
            orig_size = np.shape(smap.data)
            smap = smap.resample(u.Quantity([orig_size[0]*res,orig_size[1]*res], u.pixel)) #new dimensions are the fraction you want of the original
            if first_time_through == True: #only add the new title info on the first gop through
                title_res = ' ({:.0f}'.format(res*100)+'% res)'
                title_addition = title_addition + title_res
                first_time_through = False
            if rescale_cml == True: #rescale limits is the resolution is changed
                cmlims = [cmlims[0]*(0.5*(1+res)), cmlims[1]*(0.5*(1+res))]
                rescale_cml = False     

        if cm_scale == 'LogNorm':
            #set to min positive value to avoid nans in the log plot for values <=0
            m_data = smap.data
            m_data[m_data<=0] = np.min(np.min(m_data[m_data>0])) 
            smap = sunpy.map.Map(m_data,smap.meta)
            del m_data

        #fig = plt.figure()
        #plt.imshow(second_smap.data, vmin=-50,vmax=50)
        #plt.show()
            
        if iron == '18':
            smap.plot_settings['cmap'] = plt.cm.GnBu
        if iron == '16':
            smap.plot_settings['cmap'] = plt.cm.Purples
        if (_aia_files[0][0:3] == 'HMI') or (_aia_files[0][0:3] == 'hmi'):
            #second_smap = second_smap.rotate(angle = 180 * u.deg)
            second_smap.plot_settings['cmap'] = matplotlib.cm.get_cmap('PiYG_r') # use a simpler cmap than 'hmimag', pink=positive, green=negative

        #fig = plt.figure()
        #plt.imshow(second_smap.data, vmin=-50,vmax=50)
        #plt.show()

        if lvls is None:
            if cm_scale == 'LogNorm':
                if cmlims[0] <= 0:
                    cmlims[0] = 0.1
                smap.plot_settings['norm'] = colors.LogNorm(vmin=cmlims[0],vmax=cmlims[1])
            else:
                smap.plot_settings['norm'] = colors.Normalize(vmin=cmlims[0],vmax=cmlims[1])
            second_smap.plot_settings['norm'] = colors.Normalize(vmin=cmlims2[0],vmax=cmlims2[1])

        fig, ax = plt.subplots(figsize=(9,8)) 
        
        if in_order == True:
            compmap = sunpy.map.Map(smap, second_smap, composite=True)
        elif in_order == False:
            compmap = sunpy.map.Map(second_smap, smap, composite=True)
        
        if lvls is not None:
            compmap.set_alpha(0, alphas[0])
            compmap.set_alpha(1, alphas[1])
            compmap.set_levels(index=1, levels=lvls)
        else:
            compmap.set_alpha(0, alphas[0])
            compmap.set_alpha(1, alphas[1])
        compmap.plot()

        if rectangle != []: #if a rectangle(s) is specified, make it
            assert len(rectangle_colour)==len(rectangle) or len(rectangle_colour)==1, "Check you have either given 1 colour in the \'rectangle_colour\' list or the same number of colours as rectangles!"
            rectangle_colour = rectangle_colour if len(rectangle_colour)==len(rectangle) else rectangle_colour*len(rectangle)
            x, y, counter = submap[2], submap[3], 0 # x and y for box titles if needed, plus a counter for the "for" loop
            for rect, rcol in zip(rectangle, rectangle_colour):
                
                bl_rect = (rect[0], rect[1]) #SkyCoord(rect[0]*u.arcsec, rect[1]*u.arcsec, frame=smap.coordinate_frame)
                length = rect[2] - rect[0]
                height = rect[3] - rect[1]
                if (iron != '') or (diff_image != None): #if iron or a diff map is needed then make the rectangles black
                    rcol = "black" if len(rectangle_colour)==1 else rcol
                    plt.gca().add_patch(patches.Rectangle(bl_rect, length, height, facecolor="none", linewidth=2, edgecolor=rcol))
                    #smap.draw_rectangle(bl_rect, length*u.arcsec, height*u.arcsec, color = rcol)
                else:
                    rcol = "white" if len(rectangle_colour)==1 else rcol
                    plt.gca().add_patch(patches.Rectangle(bl_rect, length, height, facecolor="none", linewidth=2, edgecolor=rcol))
                    #smap.draw_rectangle(bl_rect, length*u.arcsec, height*u.arcsec, color = rcol)

                # if there are multiple boxes then label them with the colour, tough if you're using the same colour the now
                if len(rectangle_colour) > 1:
                    # lazy check for no repeats
                    if rectangle_colour[0] not in rectangle_colour[1:]:
                        plt.text(x, y-counter*0.06*(submap[3]-submap[1]), "Box "+str(counter+1), 
                            verticalalignment="top", horizontalalignment="right",
                            color=rcol)
                        counter += 1
        
        #make titles
        time = smap.meta['t_obs'] 
        wavelength = str(smap.meta['wavelnth'])
        second_w = str(second_smap.meta['wavelnth'])
        if iron == '18': #sets title for Iron 18
            plt.title(f'AIA FeXVIII and {second_w} $\AA$ {time[:10]} {time[11:19]}'+title_addition)  
        elif iron == '16': #sets title for Iron 16
            plt.title(f'AIA FeXVI and {second_w} $\AA$ {time[:10]} {time[11:19]}'+title_addition)
        else:
            plt.title('AIA '+wavelength + r'$\AA$ and {second_w} $\AA$ ' + f'{time[:10]} {time[11:19]}'+title_addition)

        if type(save_directory) == str: #save writing what is below twice
            save_directory = [save_directory]
        if type(save_directory) == list:
            for _save_d in save_directory:
                if save_inc == False:
                    plt.savefig(_save_d + f'compaia_image{wavelength}.png', dpi=dpi, bbox_inches='tight')
                elif save_inc == True:
                    plt.savefig(_save_d + 'maps{:04d}.png'.format(d), dpi=dpi, bbox_inches='tight')
        d+=1
                
        plt.clf() 
        plt.cla()
        plt.close('all')

        
        bl_fi = 0 #reassign variables that take up a lot of space the delete them just to be sure
        tr_fi = 0 
        aia_map = 0
        smap = 0
        second_aia_map = 0
        second_smap = 0
        compmap = 0
        del bl_fi
        del tr_fi
        del fig
        del ax
        del aia_map
        del smap
        del second_aia_map
        del second_smap
        del compmap

        gc.collect()
        print(f'\r[function: overlay_aiamaps()] Saved {d} submap(s) of {no_of_files}.        ', end='')

    aia_map = 0
    smap = 0
    compmap = 0
    del aia_map
    del smap
    del compmap
    del directory_with_files
    gc.collect()

    print('\nLook everyone, it\'s finished!')

