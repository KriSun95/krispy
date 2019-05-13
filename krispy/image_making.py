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
'''

#make images from the aia fits files
def aiamaps(directory, save_directory, submap=None, cmlims = [], rectangle=[], save_inc=True, iron='',
           cm_scale='Normalize', diff_image=None, res=None, save_smap=None, colourbar=True):      
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
            
    rectangle : two-dimensional list/array, shape=n,4
            Contains lists of the bottom left (bl) and top right (tr) coordinates to draw a rectangle on the constructed 
            map, e.g. [[blx1,bly1,trx1,try1], [blx2,bly2,trx2,try2], ...]. Must be in arcseconds, of type float or 
            integer and NOT an arcsec object.
            
    save_inc : Bool
            Indicates whether or not the save file should be named with respect to the file it was produced from or be
            named incrementally, e.g. AIA123456_123456_1234.png or map_000 respectively.
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

    Returns
    -------
    AIA maps saved to the requested directory (so doesn't really return anythin).
    """

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
        
        if submap != None:
            bl_fi = SkyCoord(submap[0]*u.arcsec, submap[1]*u.arcsec, frame=aia_map.coordinate_frame)
            tr_fi = SkyCoord(submap[2]*u.arcsec, submap[3]*u.arcsec, frame=aia_map.coordinate_frame)
        
            smap = aia_map.submap(bl_fi,tr_fi) 
            del aia_map
        else:
            smap = aia_map

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
            
        if iron == '18':
            smap.plot_settings['cmap'] = plt.cm.Blues
        if iron == '16':
            smap.plot_settings['cmap'] = plt.cm.Purples
        if diff_image is not None:
            smap.plot_settings['cmap'] = plt.cm.coolwarm
        if (_aia_files[0][0:3] == 'HMI') or (_aia_files[0][0:3] == 'hmi'):
            smap.plot_settings['cmap'] = matplotlib.cm.get_cmap('hmimag')

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
            for rect in rectangle:
                
                bl_rect = SkyCoord(rect[0]*u.arcsec, rect[1]*u.arcsec, frame=smap.coordinate_frame)
                length = rect[2] - rect[0]
                height = rect[3] - rect[1]
                if (iron != '') or (diff_image != None): #if iron or a diff map is needed then make the rectangles black
                    smap.draw_rectangle(bl_rect, length*u.arcsec, height*u.arcsec, color = 'black')
                else:
                    smap.draw_rectangle(bl_rect, length*u.arcsec, height*u.arcsec)
        
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
                    plt.savefig(_save_d + f'aia_image{wavelength}.png', dpi=600, bbox_inches='tight')
                elif save_inc == True:
                    plt.savefig(_save_d + 'maps{:04d}.png'.format(d), dpi=600, bbox_inches='tight')
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
        print(f'\r[function: aiamaps()] Saved {d} submap(s) of {no_of_files}.', end='')

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
def contourmaps_from_dir(aia_dir, nustar_dir, nustar_file, save_dir, chu='', fpm='', energy_rng=[], submap=[], 
                             cmlims = [], nustar_shift=[], time_bins=[], resample_aia=[], counter=0, contour_lvls=[],
                         contour_fmt='percent', contour_colour='black', aia='ns_overlap_only', iron18=True, 
                         save_inc=False, gauss_sigma=4, background_time='begin', A_and_B=False, B_shift=None):
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
            
    nustar_shift : One-dimensional list/array of type int, length 2
            Gives an [x,y] shift to move the NuSTAR contours to align with the AIA image as the coordinates of the NuSTAR
            map may not be spot on.
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

    A_and_B : Bool
            Specify whether, if given FPMA at first, if you want to combine it with FPMB.
            Default: False 

    B_shift : list/tuple, length 2
            If 'A_and_B=True' the you can apply a shift to B manually rather than doing the 2D correlation.
            Default: None
            
    Returns
    -------
    A dictionary with the values of the largest values of the NuSTAR map to help with contour value setting 
    (labelled as 'max_contour_levels'), the final value for the incremental counter (labelled as 
    'last_incremental_value'), and the B shift applied (manual or otherwise) labelled as 'B_shift'. AIA maps, 
    with NuSTAR contours, are also saved to the requested directory.
    """
    #20/11/2018: ~if statement for the definition of cleanevt.
    #            ~two iron 18 if statements for the colour map.
    #            ~two iron 18 if statements for the colour map.
    #26/11/2018: ~added try and except to the cleanevt bit.
    #08/04/2019: ~can now combine A and B.

    import filter_with_tmrng # this file has to be in the directory

    matplotlib.rcParams['font.sans-serif'] = "Arial" #sets up plots
    matplotlib.rcParams['font.family'] = "sans-serif"
    matplotlib.rcParams['font.size'] = 14
    
    hdulist = fits.open(nustar_dir + nustar_file) #chu, sunpos file
    evtdata=hdulist[1].data
    hdr = hdulist[1].header
    hdulist.close()

    if A_and_B == True:
        hdulist = fits.open(nustar_dir + nustar_file.replace('A', 'B')) #chu, sunpos file
        evtdata_other = hdulist[1].data
        hdr_other = hdulist[1].header
        hdulist.close()
    
    aia_files =[]
    for f in os.listdir(aia_dir):
        aia_files.append(f)
    aia_files.sort()
    aia_files = only_fits(aia_files)
    
    d = counter
    max_contours = []

    for t in range(len(time_bins)-1):
        try:
            cleanevt = filter_with_tmrng.event_filter(evtdata,fpm=fpm,energy_low=energy_rng[0], energy_high=energy_rng[1], 
                                                      tmrng=[time_bins[t], time_bins[t+1]])
            if A_and_B == True:
                            cleanevt_other = filter_with_tmrng.event_filter(evtdata_other,fpm=fpm,energy_low=energy_rng[0], energy_high=energy_rng[1], 
                                                      tmrng=[time_bins[t], time_bins[t+1]])
        except IndexError:
            cleanevt = [] #if time range is outwith nustar obs (i.e. IndexError) then this still lets aia to be looked at
            if A_and_B == True:
                            cleanevt_other = []
            
        if len(cleanevt) != 0 and aia == ('ns_overlap_only' or 'all'): #AIA data and NuSTAR data
            t_1 = datetime.datetime.strptime(time_bins[t], '%Y/%m/%d, %H:%M:%S')
            t_2 = datetime.datetime.strptime(time_bins[t+1], '%Y/%m/%d, %H:%M:%S')
            t_start = datetime.datetime.strptime(time_bins[0], '%Y/%m/%d, %H:%M:%S')
            t_end = datetime.datetime.strptime(time_bins[-1], '%Y/%m/%d, %H:%M:%S')
            nustar_map = nustar.map.make_sunpy(cleanevt, hdr, norm_map=True)
            nustar_map_normdata = nustar_map.data / ((t_2 - t_1) / (t_end - t_start))

            '''
            Need to find shift before all of this, CANNOT do it each time
            '''
            
            if (A_and_B == True) and (len(cleanevt_other) != 0):
                        if t == 0: #only want the shift at the start
                            if  B_shift == None:
                                nustar_map_first = sunpy.map.Map(nustar_map_normdata, nustar_map.meta)

                                nustar_map_other = nustar.map.make_sunpy(cleanevt_other, hdr_other, norm_map=True)
                                nustar_map_normdata_other = nustar_map_other.data / ((t_2 - t_1) / (t_end - t_start))
                                nustar_map_other = sunpy.map.Map(nustar_map_normdata_other, nustar_map_other.meta)

                                #make submap for quickness to get the shift
                                bl = SkyCoord((submap[0]-200)*u.arcsec, (submap[1]-200)*u.arcsec, frame=nustar_map_first.coordinate_frame)
                                tr = SkyCoord((submap[2]+200)*u.arcsec, (submap[3]+200)*u.arcsec, frame=nustar_map_first.coordinate_frame)
                                submap_first = nustar_map_first.submap(bl,tr)
                                dataA = submap_first.data
                                submap_other = nustar_map_other.submap(bl,tr)
                                dataB = submap_other.data

                                corr = signal.correlate2d(dataA, dataB, boundary='symm', mode='same')

                                y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match
                                x_pix_shift = -(np.shape(dataB)[1]/2 - x) #negative because the positive number means shift to the left/down
                                y_pix_shift = -(np.shape(dataB)[0]/2 - y)
                                B_shift = [x_pix_shift, y_pix_shift]
                                
                                del corr
                                del nustar_map_first
                                del submap_first
                                del submap_other
                                del nustar_map_normdata_other
                                del nustar_map_other
                            elif B_shift != None:
                                x_pix_shift = B_shift[0]
                                y_pix_shift = B_shift[1]

                        shift_evt_other = krispy.nustardo.shift(cleanevt_other, pix_xshift=x_pix_shift, pix_yshift=x_pix_shift)
                        nustar_map_other = nustar.map.make_sunpy(shift_evt_other, hdr_other, norm_map=True)
                        nustar_map_normdata_other = nustar_map_other.data / ((t_2 - t_1) / (t_end - t_start))
                       
                        nustar_map_normdata = nustar_map_normdata + nustar_map_normdata_other
                        del nustar_map_other
                        fpm = 'A&B'
            

            dd=ndimage.gaussian_filter(nustar_map_normdata, gauss_sigma, mode='nearest');
            
            # Tidy things up before plotting
            dmin=1e-3
            dmax=1e1
            dd[dd < dmin]=0
            nm=sunpy.map.Map(dd, nustar_map.meta);

            del nustar_map_normdata
            
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
                ave = background_in_trng_data.sum(axis=0) / num_of_maps
                aia_map = sunpy.map.Map(ave, background_in_trng_header[0])
            else:
                print('Choose where the background map comes from: begin, middle, end, or average of time range')

            del background_in_trng_data
            del background_in_trng_header
   
            if aia_map == 0:
                print(f'\rNo AIA data in this time range: {time_bins[t]}, {time_bins[t+1]}.', end='')
                continue
                
            # Let's shift it ############################################################################################
            if nustar_shift != []:
                shifted_nustar_map = nm.shift(nustar_shift[0]*u.arcsec, nustar_shift[1]*u.arcsec)
            else:
                shifted_nustar_map = nm
            #print(shifted_nustar_map.data)
            #print(shifted_nustar_map.meta)

            # Submap to plot ############################################################################################
            bl_s = SkyCoord((submap[0]+0.01)*u.arcsec, (submap[1]+0.01)*u.arcsec, frame=shifted_nustar_map.coordinate_frame)
            tr_s = SkyCoord((submap[2]-0.01)*u.arcsec, (submap[3]-0.01)*u.arcsec, frame=shifted_nustar_map.coordinate_frame)
            #0.01 arcsec padding to make sure the contours aren't cut off and so that the final plot won't have weird 
            #blank space bits
            shifted_nustar_submap = shifted_nustar_map.submap(bl_s,tr_s)
            
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [contour_colour,contour_colour])
            shifted_nustar_submap.plot_settings['norm'] = colors.LogNorm(vmin=dmin,vmax=dmax)
            shifted_nustar_submap.plot_settings['cmap'] = cmap

            #############################################################################################################
            bl_fi = SkyCoord(submap[0]*u.arcsec, submap[1]*u.arcsec, frame=aia_map.coordinate_frame)
            tr_fi = SkyCoord(submap[2]*u.arcsec, submap[3]*u.arcsec, frame=aia_map.coordinate_frame)
            #0.01 arcsec padding
            smap = aia_map.submap(bl_fi,tr_fi)    
            
            del aia_map

            if iron18 == True:
                smap.plot_settings['cmap'] = plt.cm.Blues

            fig = plt.figure(figsize=(9,8));
            compmap = sunpy.map.Map(smap, composite=True);
            del smap
            compmap.add_map(shifted_nustar_submap);
            
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
            
            plt.colorbar(label='DN s$^{-1}$');
            
            char_to_arcsec = 3.0 #-ish
            border = 2
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
                plt.savefig(save_dir + f'nustar_contours{d}_on_iron18_chu{chu}_fpm{fpm}.png', dpi=600, 
                            bbox_inches='tight')
            elif save_inc == True:
                plt.savefig(save_dir + 'contours{:04d}.png'.format(d), dpi=600, bbox_inches='tight')
            d+=1
                
            plt.close(fig)
            print(f'\rSaved {d} submap(s).', end='')
            
            max_contours.append(shifted_nustar_submap.data.max())
            del shifted_nustar_submap

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
            
            if iron18 == True:
                smap.plot_settings['cmap'] = plt.cm.Blues

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
                plt.savefig(save_dir + f'nustar_contours{d}_on_iron18_chu{chu}_fpm{fpm}.png', dpi=600, 
                            bbox_inches='tight')
            elif save_inc == True:
                plt.savefig(save_dir + 'maps{:04d}.png'.format(d), dpi=600, bbox_inches='tight')
            d+=1
                
            plt.close(fig)
            del smap
            del aia_map
            print(f'\rSaved {d} submap(s).', end='')
        
        else:
            print(f'\rNo NuSTAR data in this time range: {time_bins[t]} to {time_bins[t+1]}.', end='')
    
    print('\nLook everyone, it\'s finished!')
    return {'max_contour_levels': max_contours, 'last_incremental_value': d-1, 'B_shift': B_shift} 
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
def aiamaps_from_dir(fits_dir, out_dir, savefile_fmt='.png', dpi=600, cmlims = [], submap=[], rectangle=[], 
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
            Dots per inch for the save file resolution, e.g. 100, 150, 600, etc.
            Default: 600

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
            aia_map.plot_settings['cmap'] = plt.cm.Blues
            
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
                aia_submap.plot_settings['cmap'] = plt.cm.Blues
            
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
def overlay_aiamaps(directory, second_directory, save_directory, submap=None, cmlims = [], lvls=[-50, 50], rectangle=[], save_inc=True, iron='',
                    cm_scale='Normalize', res=None, colourbar=True):      
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
    
    lvls : 1D array
            Contour levels as true values (not percentages) fort the second overlying map. If lvls is set to None then the two maps are
            just combined.
            Default: [-50, 50]
            
    rectangle : two-dimensional list/array, shape=n,4
            Contains lists of the bottom left (bl) and top right (tr) coordinates to draw a rectangle on the constructed 
            map, e.g. [[blx1,bly1,trx1,try1], [blx2,bly2,trx2,try2], ...]. Must be in arcseconds, of type float or 
            integer and NOT an arcsec object.
            
    save_inc : Bool
            Indicates whether or not the save file should be named with respect to the file it was produced from or be
            named incrementally, e.g. AIA123456_123456_1234.png or map_000 respectively.
            Default: True
    
    iron : Str
            Indicates whether or not the save file is the iron 16 or 18 channel. Set to '16' or '18'.
            Default: ''

    cm_scale : Str
            Scale for the colour bar for the plot. Set to 'Normalize' or 'LogNorm'.
            Default: 'Normalize'

    res : float
            A float <1 which has the resolution of the image reduced, e.g. 0.5 produces an image at 50% resolution to the original.
            Default: None

    colourbar : Bool
            Indicates whether or not to draw the colour bar for the map.
            Default: True

    Returns
    -------
    AIA maps saved to the requested directory (so doesn't really return anythin).
    """

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

    no_of_files = len(directory_with_files)

    d = 0

    for f in range(no_of_files):

        aia_map = sunpy.map.Map(directory_with_files[f])

        #find closest overlay file
        first_image_time = datetime.datetime.strptime(aia_map.meta['t_obs'], '%Y-%m-%dT%H:%M:%S.%fZ')
        distance = [abs(first_image_time - datetime.datetime.strptime(over[3:18], '%Y%m%d_%H%M%S')) for over in second_files]
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
            
        if iron == '18':
            smap.plot_settings['cmap'] = plt.cm.Blues
        if iron == '16':
            smap.plot_settings['cmap'] = plt.cm.Purples
        if (_aia_files[0][0:3] == 'HMI') or (_aia_files[0][0:3] == 'hmi'):
            second_smap.plot_settings['cmap'] = matplotlib.cm.get_cmap('hmimag')

        fig, ax = plt.subplots(figsize=(9,8)) 
        
        compmap = sunpy.map.Map(smap, second_smap, composite=True) #comp image as to keep formatting the same as NuSTAR
        
        if lvls is not None:
            compmap.set_alpha(0, 1)
            compmap.set_alpha(1, 0.4)
            compmap.set_levels(index=1, levels=lvls)
        else:
            compmap.set_alpha(0, 0.5)
            compmap.set_alpha(1, 0.5)
        
        if cm_scale == 'Normalize': #tell the plot the choices for the limits and colour map
            if cmlims != []:
                if False:#cmlims[0] <= 0 and diff_image == False: #vmin > 0 or error
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
            for rect in rectangle:
                
                bl_rect = SkyCoord(rect[0]*u.arcsec, rect[1]*u.arcsec, frame=smap.coordinate_frame)
                length = rect[2] - rect[0]
                height = rect[3] - rect[1]
                if (iron != ''): #if iron or a diff map is needed then make the rectangles black
                    smap.draw_rectangle(bl_rect, length*u.arcsec, height*u.arcsec, color = 'black')
                else:
                    smap.draw_rectangle(bl_rect, length*u.arcsec, height*u.arcsec)
        
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
                    plt.savefig(_save_d + f'aia_image{wavelength}.png', dpi=600, bbox_inches='tight')
                elif save_inc == True:
                    plt.savefig(_save_d + 'maps{:04d}.png'.format(d), dpi=600, bbox_inches='tight')
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
        print(f'\r[function: aiamaps()] Saved {d} submap(s) of {no_of_files}.', end='')

    aia_map = 0
    smap = 0
    compmap = 0
    del aia_map
    del smap
    del compmap
    del directory_with_files
    gc.collect()
    print('\nLook everyone, it\'s finished!')

