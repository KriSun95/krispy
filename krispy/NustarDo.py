'''
Functions to go in here (I think!?):
    KC: 01/12/2018, ideas-

    KC: 19/12/2018, added-
    ~NuSTAR class
'''

import sys
from os.path import *
import os
import astropy
from astropy.io import fits
import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from pylab import figure, cm
from astropy.coordinates import SkyCoord
import numpy as np
import nustar_pysolar as nustar
import filter_with_tmrng ######Kris
import custom_map ######Kris
import sunpy.map
from scipy import ndimage
import re #for regular expressions
import warnings #suppress astropy warnings
import datetime
from datetime import timedelta
from astropy.io.fits.verify import VerifyWarning
import matplotlib.dates as mdates
import pickle

'''
Alterations:
    KC: 22/01/2019 - . 
'''

#NuSTAR class for Python
class NustarDo:
    
    np.seterr(divide='ignore', invalid='ignore') #ignore warnings resulting from missing header info
    warnings.simplefilter('ignore', VerifyWarning)
    warnings.simplefilter('ignore', RuntimeWarning) 
    warnings.simplefilter('ignore', UserWarning)
    
    def __init__(self, evt_filename, energy_range=[2.5,79], time_range = None): #set-up parameters
        #directory of the file
        directory_regex = re.compile(r'\w+/')
        directory = directory_regex.findall(evt_filename)
        self.evt_directory = ''.join(directory)

        #search of the form of stuff (no slashes included), dot, then more stuff
        evt_filename_regex = re.compile(r'\w+\.\w+') 
        name_of_file = evt_filename_regex.findall(evt_filename)[0]
        
        #for a sunpy map object to be made then the file has to be positioned on the Sun
        sunpos_regex = re.compile(r'sunpos')
        sunpos = sunpos_regex.findall(name_of_file)
        if sunpos == []:
            raise ValueError('\nThe file must be a \'sunpos\' file, i.e. the observation is converted to appropriate solar coordinates.')
        
        #search for 2 digits, a non-digit, then 2 digits again
        fpm_regex = re.compile(r'\d{2}\D\d{2}')
        focal_plane_module = fpm_regex.findall(name_of_file)[0][2]
        
        #search for chu followed by however many consecutive digits
        chu_regex = re.compile(r'chu\d+')
        chu = chu_regex.findall(name_of_file)
        if chu != []:
            chu_state = chu[0]
        else:
            chu_state = 'not_split'
        
        #search for all seperate sub-strings composed of digits, first one in evt_filename is observation id
        obs_id_regex = re.compile(r'\d+')
        obs_id = obs_id_regex.findall(evt_filename)[0]
        self.obs_id = obs_id
        
        #set attributes of the file and parameters used in other functions on the class
        self.evt_filename = name_of_file 
        self.fpm = focal_plane_module
        self.time_range = time_range 
        self.energy_range = energy_range
        self.chu_state = chu_state
        self.rectangles = None #set so that you don't have to plot a map to get a light curve
        
        #extract the data within the provided parameters
        hdulist = fits.open(evt_filename) #not self.evt_filename as fits.open needs to know the full path to the file
        self.evt_data = hdulist[1].data
        self.evt_header = hdulist[1].header
        hdulist.close()
        
        #check evt_filename matches evt_header info
        assert obs_id == self.evt_header['OBS_ID'], 'Observation ID in the .evt filename does not match ID in the .evt header info. {} =/= {}'.format(obs_id, self.evt_header['OBS_ID'])
        assert focal_plane_module == self.evt_header['INSTRUME'][-1], 'Focal Plane Module (FPM) in the .evt filename does not match FPM in the .evt header info. {} =/= {}'.format(focal_plane_module, self.evt_header['INSTRUME'][-1])
        
        if self.time_range == None:
            #filter away the non grade zero counts and bad pixels
            self.cleanevt = filter_with_tmrng.event_filter(self.evt_data, fpm=focal_plane_module, 
                                                           energy_low=self.energy_range[0], 
                                                           energy_high=self.energy_range[1])
            #NuSTAR time is the number of seconds from 1/1/2010
            nustar_measured_from = datetime.datetime(2010,1 ,1 ,0 ,0 ,0)
            
            #start and end time of the NuSTAR observation as datetime objects
            self.time_range = [(nustar_measured_from + timedelta(seconds=np.min(self.cleanevt['TIME']))).strftime('%Y/%m/%d, %H:%M:%S'),
                              (nustar_measured_from + timedelta(seconds=np.max(self.cleanevt['TIME']))).strftime('%Y/%m/%d, %H:%M:%S')]
        elif len(self.time_range) == 2:
            try:
                self.cleanevt = filter_with_tmrng.event_filter(self.evt_data, fpm=focal_plane_module, 
                                                               energy_low=self.energy_range[0], 
                                                               energy_high=self.energy_range[1], 
                                                               tmrng=self.time_range) ######Kris
            except TypeError as error:
                raise TypeError('\nTimes need to be a string in the form \'%y/%m/%d, %H:%M:%S\', '
                                'e.g.\'2018/12/25, 12:30:52\'')
        else:
            raise TypeError('\nCheck that it is only a start time and end time you are giving.')

        #if there are no counts in cleanevt
        if len(self.cleanevt) == 0: 
            raise ValueError('\nThere there are no counts within these paramenters. '
                             '\nThis may be because no counts were recorded or that the paramenters are outwith the '
                             'scope of NuSTAR and/or the observation.')
    
    
    gaussian_filter = {'apply':True, 'sigma':4, 'mode':'nearest'}
    
    def nustar_map(self, time_norm=True, lose_off_limb=True, limits=[], 
                   submap=[], rebin_factor=1, norm='linear'):
        
        # Map the filtered evt, into one corrected for livetime (so units count/s) 
        self.nustar_map = custom_map.make_sunpy(self.cleanevt, self.evt_header, norm_map=time_norm)
        
        self.time_norm = time_norm
        
        if lose_off_limb == True:
            #fix really large plot, instead of going from -3600 to 3600 in x and y
            bl = SkyCoord(-1200*u.arcsec, -1200*u.arcsec, frame=self.nustar_map.coordinate_frame)
            tr = SkyCoord(1200*u.arcsec, 1200*u.arcsec, frame=self.nustar_map.coordinate_frame)
            self.nustar_map = self.nustar_map.submap(bl,tr)

        if self.gaussian_filter['apply'] == True:
            gaussian_width = self.gaussian_filter['sigma']
            m = self.gaussian_filter['mode']
            #Apply a guassian blur to the data to bring out the faint feature
            dd = ndimage.gaussian_filter(self.nustar_map.data, gaussian_width, mode=m)
            if limits == []:
                dmin = np.min(dd[np.nonzero(dd)])*1e6 #factor here as the lowest value will come from the gaussian 
                #filter and not the actual lowest count rate hence the factor 
                dmax = np.max(dd[np.isfinite(dd)])
            elif len(limits) == 2:
                if norm == 'lognorm':
                    if limits[0] <= 0:
                        dmin = 0.1
                        dmax=limits[1]
                    else:
                        dmin=limits[0]
                        dmax=limits[1]
                elif norm == 'linear':
                    dmin=limits[0]
                    dmax=limits[1]
                #dmin, dmax = *limits #i.e. dmin=limits[0], dmax=limits[1]
            else:
                raise TypeError('\nCheck the limits that were given please.')
        else:
            dd = self.nustar_map.data
            if limits == []:
                dmin = np.min(dd[np.nonzero(dd)])
                dmax = np.max(dd[np.isfinite(dd)])
            elif len(limits) == 2:
                if norm == 'lognorm':
                    if limits[0] <= 0:
                        dmin = 0.1
                        dmax=limits[1]
                    else:
                        dmin=limits[0]
                        dmax=limits[1]
                elif norm == 'linear':
                    dmin=limits[0]
                    dmax=limits[1]
                #dmin, dmax = *limits #i.e. dmin=limits[0], dmax=limits[1]
            else:
                raise TypeError('\nCheck the limits that were given please. It should be a list with two float/int '
                                'entries')

        # Tidy up before plotting
        dd[dd < dmin]=0
        nm = sunpy.map.Map(dd, self.nustar_map.meta)

        #Submap to plot?
        if len(submap) == 4:
            bottom_left = {'x':submap[0], 'y':submap[1]} 
            top_right = {'x':submap[2], 'y':submap[3]} 

            bl = SkyCoord(bottom_left['x']*u.arcsec, bottom_left['y']*u.arcsec, frame=nm.coordinate_frame)
            tr = SkyCoord(top_right['x']*u.arcsec, top_right['y']*u.arcsec, frame=nm.coordinate_frame)
            nustar_submap = nm.submap(bl,tr)
        elif len(submap) == 0:
            pass
        else:
            raise TypeError('\nCheck the submap coordinates that were given please. It should be a list with four '
                            'float/int entries in arcseconds in the form [bottom left x, bottom left y, top right x, '
                            'top right y].')

        if rebin_factor != 1:
            if len(submap) == 4:
                #can rebin the pixels if we want to further bring out faint features
                #set to 1 means no actual rebinning
                nx,ny = np.shape(self.nustar_submap.data)
                if rebin_factor > 1/nx and rebin_factor > 1/ny:
                    dimensions = u.Quantity([nx*rebin_factor, ny*rebin_factor], u.pixel)
                    rsn_map = nustar_submap.resample(dimensions)
                else:
                    raise TypeError(f'\nRebin factor must be greater than one over the x,y dimensions (1/{nx} and '
                                    f'1/{ny}) as to rebin to more than one pixel.')

            elif len(submap) == 0:
                #can rebin the pixels if we want to further bring out faint features
                #set to 1 means no actual rebinning
                nx,ny = np.shape(nm.data)
                if rebin_factor > 1/nx and rebin_factor > 1/ny:
                    dimensions = u.Quantity([nx*rebin_factor, ny*rebin_factor], u.pixel)
                    rsn_map = nm.resample(dimensions)
                else:
                    raise TypeError(f'\nRebin factor must be greater than one over the x,y dimensions (1/{nx} and '
                                    f'1/{ny}) as to rebin to more than one pixel.')
        elif rebin_factor == 1:
            if len(submap) == 4:
                rsn_map = nustar_submap
                del nustar_submap

            elif len(submap) == 0:
                rsn_map = nm
                del nm
        
        if norm == 'linear':
            #change all zeros to NaNs so they appear white in the plot otherwise zeros appear as the lowest colour 
            #on the colourbar
            rsn_map_data = rsn_map.data
            rsn_map_data[rsn_map_data <= 0] = np.nan
            rsn_map = sunpy.map.Map(rsn_map_data, rsn_map.meta)

            # Setup the scaling of the map and colour table
            rsn_map.plot_settings['norm'] = colors.Normalize(vmin=dmin,vmax=dmax)
            rsn_map.plot_settings['cmap'] = cm.get_cmap('Spectral_r')

        elif norm == 'lognorm':
            #log(0) produces a NaN (-inf) here anyway so appears white
            # Setup the scaling of the map and colour table
            rsn_map.plot_settings['norm'] = colors.LogNorm(vmin=dmin,vmax=dmax) 
            rsn_map.plot_settings['cmap'] = cm.get_cmap('Spectral_r')
        
        self.rsn_map = rsn_map
        return rsn_map
        
        
    def nustar_plot(self, boxes=None, show_fig=True, save_fig=None):
        matplotlib.rcParams['font.sans-serif'] = "Arial"
        #matplotlib.rcParams['font.family'] = "sans-serif"
        #matplotlib.rcParams['font.size'] = 12
        plt.rcParams["figure.figsize"] = (6,6)
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.size'] = 18
        plt.rcParams['font.family'] = 'STIXGeneral'
        # Start the plot - many things here just to make matplotlib look decent
        
        self.rectangles = boxes

        fig = plt.figure(figsize=(9, 8), frameon=False)
        ax = plt.subplot(projection=self.rsn_map, frame_on=False) #rsn_map nustar_submap

        self.rsn_map.plot()
        self.rsn_map.draw_limb(color='black',linewidth=1,linestyle='dashed',zorder=0)   

        # Manually plot a heliographic overlay - hopefully future no_ticks option in draw_grid
        overlay = ax.get_coords_overlay('heliographic_stonyhurst')
        lon = overlay[0]
        lat = overlay[1]
        lon.set_ticks_visible(False)
        lat.set_ticks_visible(False)
        lat.set_ticklabel_visible(False)
        lon.set_ticklabel_visible(False)
        lon.coord_wrap = 180
        lon.set_major_formatter('dd')
        overlay.grid(color='grey', linewidth=0.5, linestyle='dashed')

        # Tweak the titles and labels
        title_obsdate = '{:.20}'.format('{:%Y-%b-%d}'.format(self.rsn_map.date))
        e_range_str = str(self.energy_range[0])+'-'+str(self.energy_range[1])
        fpm = 'FPM'+self.fpm
        title_obstime_start = self.time_range[0][-8:]
        title_obstime_end = self.time_range[1][-8:]

        if self.chu_state == 'not_split':
            ax.set_title('NuSTAR '+e_range_str+' keV '+fpm+' '+ title_obsdate+' '+title_obstime_start+' to '+title_obstime_end)
        else:
            ax.set_title('NuSTAR '+e_range_str+' keV '+fpm+' '+self.chu_state+' '+ title_obsdate+' '+title_obstime_start+' to '+title_obstime_end)
        
        ax.set_ylabel('y [arcsec]')
        ax.set_xlabel('x [arcsec]')
        tx, ty = ax.coords
        tx.set_major_formatter('s')
        ty.set_major_formatter('s')
        ax.grid(False)
        # Add a colour bar
        if self.time_norm == True:
            plt.colorbar(fraction=0.035, pad=0.03,label='Counts $s^{-1}$')
        else:
            plt.colorbar(fraction=0.035, pad=0.03,label='Counts')
            
        if boxes is not None:
            if np.shape(boxes)==(4,):
                rect = boxes
                bottom_left_rectangle = SkyCoord(rect[0]*u.arcsec, rect[1]*u.arcsec, frame=self.rsn_map.coordinate_frame)
                length = rect[2] - rect[0]
                height = rect[3] - rect[1]
                self.rsn_map.draw_rectangle(bottom_left_rectangle, length*u.arcsec, height*u.arcsec, color='black')
            else:
                b = 1
                for rect in boxes:
                    bottom_left_rectangle = SkyCoord(rect[0]*u.arcsec, rect[1]*u.arcsec, frame=self.rsn_map.coordinate_frame)
                    length = rect[2] - rect[0]
                    height = rect[3] - rect[1]
                    self.rsn_map.draw_rectangle(bottom_left_rectangle, length*u.arcsec, height*u.arcsec, color='black')
                    for_text = self.arcsec_to_pixel([rect[0]-10,rect[3]+20], centre_pix_val= [self.rsn_map.meta['crpix1']+0.5, self.rsn_map.meta['crpix2']])
                    plt.text(for_text[0][0], for_text[0][1], 'Box '+str(b), fontsize=10)
                    b += 1
                    
        if save_fig != None:
            plt.savefig(save_fig)
        if show_fig == True:
            plt.show('all')
            
    
    def nustar_peek(self):
        #just to view the map with all default settings
        self.nustar_map() 
        self.nustar_plot()
            
    
    @staticmethod
    def stepped_lc_from_hist(x, y, inc_edges=True):
        """Takes an x and y input, duplicates the x values and y values with the offset as to produce a new x and y which 
        will produce a stepped graph once all the scatter points are plotted.

        Parameters
        ----------
        x : 1-d list/array
                This is the original set of x values or, in the case for a histogram, the bin edges.

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
        if len(x) == len(y)+1: #since histogram gives one more as they are the boundaries of the bins
            old_x = x
            x = x[:-1]
        elif len(x) == len(y):
            x = x #not necessary, but more readable just now
        else:
            raise ValueError('Either the x-axis array is the edge of the bins (len(x) == len(y)+1) or the x-axis is the '
                             'value for the beginning of each bin (len(x) == len(y)), you haven\'t satisfied either of '
                             'these.')

        new_x = np.array(np.zeros(2*len(x))) 
        new_y = np.array(np.zeros(2*len(y)))
        for i in range(len(x)): #x and y should be the same length to plot anyway
            if i == 0: #start with the 1st and 2nd x value having the same y.
                new_x[i] = x[i]
                new_y[2*i], new_y[2*i+1] = y[i], y[i]
            elif i == len(x)-1: #the last new_x should be one beyond the last x as this value for the start of its bin
                if len(x) == len(y)+1:
                    new_x[2*i-1], new_x[2*i], new_x[2*i+1] = x[i], x[i], old_x[-1]
                elif len(x) == len(y):
                    new_x[2*i-1], new_x[2*i], new_x[2*i+1] = x[i], x[i], x[i]+(x[i]-x[i-1])
                new_y[2*i] , new_y[2*i+1] = y[i], y[i]
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
    
        
    @staticmethod
    def dt_to_md(dt_array):
        if type(dt_array) != list:
            dt_array = [dt_array]
            
        new_array = np.zeros(len(dt_array))
        for c, d in enumerate(dt_array):
            plt_date = mdates.date2num(d)
            new_array[c] = plt_date
        return new_array
    
    
    @staticmethod
    def arcsec_to_pixel(*args, **kwargs):
        #NuSTAR values: ['crpix1'+0.5,'crpix2','cdelt1']
        meta = {'centre_pix_val': [1499.5+0.5, 1500], 'arc_per_pix':[2.45810736]} 
        #change list with kwargs
        for key, kwarg in kwargs.items():   
            meta[key] = kwarg

        #convert numbers so that they are easier to work with
        indices_for_centre = {'x':meta['centre_pix_val'][0]-1, 'y':meta['centre_pix_val'][1]-1}
        assert 1 <= len(meta['arc_per_pix']) <= 2, '\'arc_per_pix\' needs to have one or two arguments only.'
        if len(meta['arc_per_pix']) == 2:
            delta_x = meta['arc_per_pix'][0]
            delta_y = meta['arc_per_pix'][1]
        elif len(meta['arc_per_pix']) == 1:
            delta_x = meta['arc_per_pix'][0]
            delta_y = meta['arc_per_pix'][0]

        #input coordinates as [x,y] in arcseconds
        pixel_coords = []
        for arg in args:
            x_index = indices_for_centre['x'] + (arg[0] / delta_x)
            y_index = indices_for_centre['y'] + (arg[1] / delta_y)
            pixel_coords.append([int(round(x_index,0)), int(round(y_index,0))])
        return pixel_coords


    @staticmethod
    def spatial_filter(evt_data, sub_region_in_pixels):
        x = evt_data['X']
        y = evt_data['Y']
        #find indices within the x and y pixel range
        indices = (sub_region_in_pixels[0][0] < x)&(x<= sub_region_in_pixels[1][0]) & \
        (sub_region_in_pixels[0][1] < y)&(y <= sub_region_in_pixels[1][1])
        evt_data = evt_data[indices]
        return evt_data
    
    
    @staticmethod
    def time_filter(evtdata, tmrng=None):
        ''' ***** From filter function ***** >4x quicker to just filter with time than with full filter ***** '''
        if tmrng is None:
            tmrng = [evtdata['TIME'][0], evtdata['TIME'][-1]]
        elif tmrng is not None:
            tstart = datetime.datetime.strptime(tmrng[0], '%Y/%m/%d, %H:%M:%S') #date must be in this format 'yyyy/mm/dd, HH:MM:SS'
            tend = datetime.datetime.strptime(tmrng[1], '%Y/%m/%d, %H:%M:%S')
            rel_t = datetime.datetime(2010,1 ,1 ,0 ,0 ,0) #the date NuSTAR times are defined from
            tstart_s = (tstart - rel_t).total_seconds() #both dates are converted to number of seconds from 2010-Jan-1  
            tend_s = (tend - rel_t).total_seconds()
            tmrng = [tstart_s, tend_s] 

        time_filter = ( (evtdata['TIME']>tmrng[0]) & (evtdata['TIME']<tmrng[1]) )
        inds = (time_filter).nonzero()  
        goodinds=inds[0]       

        return evtdata[goodinds]
        
        
    def livetime(self, hk_filename=None, show_fig=True):
        #file = '/Users/kris/Documents/PhD/data/nustar/nu80414201001A_fpm.hk'
        '''
        This has to be moved above the time profile function so it is defined to be called
        '''
        plt.rcParams["figure.figsize"] = (10,6)
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.size'] = 18
        plt.rcParams['font.family'] = 'STIXGeneral'
        
        if hk_filename == None:
            #expression for everything that ends in a slash
            search_directory_regex = re.compile(r'\w+/')
            #find all the folders in the evt directory (they end with a slash)
            search_directory = search_directory_regex.findall(self.evt_directory)
            #don't includce the last folder to go back a directory
            search_directory = '/'+''.join(search_directory[:-1]) #go back a directory to search for the house keeping file
            for _dirpath, _dirnames, _filenames in os.walk(search_directory):
                for _file in _filenames:
                    if _file == 'nu'+self.obs_id+self.fpm+'_fpm.hk':
                        hk_filename = os.path.join(_dirpath, _file)
                        self.hk_directory = _dirpath
                        self.hk_filename = _file
            if hk_filename == None: #if there is still no hk_filename then there won't be one used
                print('Unable to find appropriate .hk file.')
                self.hk_times = 0
                self.hk_livetimes = 0
                return #stops the function here but doesn't stop the code, this is the same as 'return None'
        
        name_of_hk_file_regex = re.compile(r'\w+\.\w+')
        name_of_hk_file = name_of_hk_file_regex.findall(hk_filename)[0]
        
        hk_obs_id_regex = re.compile(r'\d+')
        hk_obs_id = hk_obs_id_regex.findall(name_of_hk_file)[0]
        
        hk_fpm_regex = re.compile(r'[A-Z]')
        hk_fpm = hk_fpm_regex.findall(name_of_hk_file)[0]
        
        #check .evt file and .hk file match
        assert self.obs_id == hk_obs_id, 'The observation id from the .evt file and the .hk are different, i.e. {} =/= {}'.format(self.obs_id, hk_obs_id)
        assert self.fpm == hk_fpm, 'The FPM from the .evt file and the .hk are different, i.e. {} =/= {}'.format(self.fpm, hk_fpm)
        
        hdulist = fits.open(hk_filename)
        self.hk_header = hdulist[1].header
        self.hk_data = hdulist[1].data
        hdulist.close()
        
        #check .hk filename matches its header info
        assert self.hk_header['OBS_ID'] == hk_obs_id, 'The observation id from the .hk file header and the .hk filename are different, i.e. {} =/= {}'.format(self.hk_header['OBS_ID'], hk_obs_id)
        assert self.hk_header['INSTRUME'][-1] == hk_fpm, 'The FPM from the .hk header and the .hk filename are different, i.e. {} =/= {}'.format(self.hk_header['INSTRUME'][-1], hk_fpm)

        self.hk_times = self.hk_data['time']
        self.hk_livetimes = self.hk_data['livetime']
        
        if show_fig == True:
            hktime = self.hk_times - self.hk_times[0]
            dt_times = [(datetime.datetime(2010,1 ,1 ,0 ,0 ,0) + timedelta(seconds=t)) for t in self.hk_times]
            lt_start_hhmmss = str((datetime.datetime(2010,1 ,1 ,0 ,0 ,0) + timedelta(seconds=np.min(self.hk_times))).strftime('%Y/%m/%d, %H:%M:%S'))
            # plt.semilogy(hktime[0:20],lvt[0:20],drawstyle='steps-mid')
            # plt.semilogy(hktime[0:20],lvt[0:20],'r')
            #plt.semilogy(hktime, self.hk_livetimes, drawstyle='steps-mid')
            fig = plt.figure()
            ax = plt.axes()
            plt.semilogy(self.dt_to_md(dt_times), self.hk_livetimes, drawstyle='steps-mid')
            plt.title('Livetime - '+lt_start_hhmmss[:10]) #get the date in the title
            plt.xlabel('Start Time - '+lt_start_hhmmss[12:])
            plt.ylabel('Livetime Fraction')
            plt.xlim([dt_times[0], dt_times[-1]])
            plt.ylim([0,1])
            fmt = mdates.DateFormatter('%H:%M')
            ax.xaxis.set_major_formatter(fmt)
            ax.xaxis.set_major_locator(plt.LinearLocator(9))
            plt.xticks(rotation=30)
            plt.show()        
        
        
    def light_curve(self, cleanevt=None, hdr=None, t_bin=[10,'approx'], sub_reg=None, tstart=None, tend=None, 
                    count_rate=True, house_keeping_file=None, make_final_graph=True):     
        plt.rcParams["figure.figsize"] = (10,6)
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.size'] = 18
        plt.rcParams['font.family'] = 'STIXGeneral'
        
        if cleanevt == None:
            cleanevt = self.cleanevt
        if hdr == None:
            hdr = self.evt_header
        if sub_reg == 'boxes':
            sub_reg = self.rectangles
            self.sub_reg_lc = sub_reg
            
        rel_t = datetime.datetime(2010,1 ,1 ,0 ,0 ,0)
        if tstart == None:
            tstart = np.min(cleanevt['TIME'])
            rel_tstart = tstart #already relative to 1/1/2010 and in seconds
        else:
            tstart = datetime.datetime.strptime(tstart, '%Y/%m/%d, %H:%M:%S') #date must be in this format 
            rel_tstart = (tstart - rel_t).total_seconds()

        if tend == None:
            tend = np.max(cleanevt['TIME'])
            rel_tend = tend #already relative to 1/1/2010 and in seconds
        else:
            tend = datetime.datetime.strptime(tend, '%Y/%m/%d, %H:%M:%S')
            rel_tend = (tend - rel_t).total_seconds() 
            
        if count_rate == True:
            self.livetime(hk_filename=house_keeping_file, show_fig=False) #run to get times and livetimes
            if len(self.hk_times) == 0:
                decision = input('No livetimes present. Do you just want to see the counts vs. time instead: ')
                if decision in ['Yes', 'yes', 'Y', 'y']:
                    count_rate = False
                else:
                    print('Will not show plot.')
                    return

        if t_bin[1] == 'approx':
            if (type(cleanevt) == astropy.io.fits.fitsrec.FITS_rec) and (sub_reg == None): #data form of NuSTAR
                t_bin_conversion = int((rel_tend - rel_tstart) // t_bin[0]) #get approximately t_bin seconds per bin as start and end of 
                #data are fixed when the histogram is created
                assert t_bin_conversion >= 1, 'Number of bins cannot be <1. Decrease \'t_bin\' value to get more bins.'

                counts = np.histogram(cleanevt['TIME'], t_bin_conversion) #gives out bin values and bin edges
                self.counts = counts[0]
                times = counts[1][:-1]
                t_bin_edges = counts[1]
                
                start_hhmmss = str((datetime.datetime(2010,1 ,1 ,0 ,0 ,0) + timedelta(seconds=np.min(times))).strftime('%H:%M:%S'))
                start_yyyymmdd = str((datetime.datetime(2010,1 ,1 ,0 ,0 ,0) + timedelta(seconds=np.min(times))).strftime('%Y/%m/%d'))
                
            elif (type(cleanevt) == astropy.io.fits.fitsrec.FITS_rec) and (sub_reg != None):
                #this is to plot the light curve of a sub-region.
                print('Inconvenient to approximate the time bins for the light curve of a sub_region.'
                      '\nChanging to \'exact\'.')
                t_bin[1] = 'exact'
            else:
                raise TypeError('\'astropy.io.fits.fitsrec.FITS_rec\' is the only supported data type at the moment.')

        if t_bin[1] == 'exact': #if since if the 'approx' flag is up and also submap!=None then time profile should be made here
            t_bin_number = int((rel_tend - rel_tstart) // t_bin[0]) #get whole number of bins that are t_bin seconds long and
            #doesn't include any time at the end that only has data for some of the last range

            assert t_bin_number >= 1, 'Number of bins cannot be <1. Decrease \'t_bin\' value to get more bins.'

            edge = rel_tstart
            t_bin_edges = np.zeros(t_bin_number+1) #+1 for the last edge
            for t in range(len(t_bin_edges)):
                t_bin_edges[t] = edge
                edge += t_bin[0]
            times = t_bin_edges[:-1]
            
            start_hhmmss = str((datetime.datetime(2010,1 ,1 ,0 ,0 ,0) + timedelta(seconds=np.min(times))).strftime('%H:%M:%S'))
            start_yyyymmdd = str((datetime.datetime(2010,1 ,1 ,0 ,0 ,0) + timedelta(seconds=np.min(times))).strftime('%Y/%m/%d'))
            
            if (type(cleanevt) == astropy.io.fits.fitsrec.FITS_rec) and (sub_reg == None): #data form of NuSTAR

                counts = np.histogram(cleanevt['TIME'], t_bin_edges) #gives out bin values and bin edges
                self.lc_counts = counts[0]
                    
            elif (type(cleanevt) == astropy.io.fits.fitsrec.FITS_rec) and (sub_reg != None):
                if np.shape(sub_reg) == (4,):
                    counts = []
                    pixels = self.arcsec_to_pixel([sub_reg[0],sub_reg[1]], [sub_reg[2],sub_reg[3]])
                    spacial_evtdata = self.spatial_filter(self.cleanevt, pixels)
                    for t in range(len(t_bin_edges)-1):
                        ts = (datetime.datetime(1970, 1, 1) + timedelta(seconds=(float(rel_t.strftime("%s"))+t_bin_edges[t]))).strftime('%Y/%m/%d, %H:%M:%S')
                        te = (datetime.datetime(1970, 1, 1) + timedelta(seconds=(float(rel_t.strftime("%s"))+t_bin_edges[t+1]))).strftime('%Y/%m/%d, %H:%M:%S')
      
                        sub_cleanevt = self.time_filter(spacial_evtdata, tmrng=[ts, te])
                        counts.append(len(sub_cleanevt['TIME']))
           
                    self.counts = np.array(counts)

                elif np.shape(sub_reg)[1] == 4:  
                    all_counts = {}
                    all_count_rates = {}
                    for b, sub_r in enumerate(sub_reg, start=1):
                        counts = []
                        
                        pixels = self.arcsec_to_pixel([sub_r[0],sub_r[1]], [sub_r[2],sub_r[3]])
                        spacial_evtdata = self.spatial_filter(self.cleanevt, pixels)
                        
                        for t in range(len(t_bin_edges)-1):
                            ts = (datetime.datetime(1970, 1, 1) + timedelta(seconds=(float(rel_t.strftime("%s"))+t_bin_edges[t]))).strftime('%Y/%m/%d, %H:%M:%S')
                            te = (datetime.datetime(1970, 1, 1) + timedelta(seconds=(float(rel_t.strftime("%s"))+t_bin_edges[t+1]))).strftime('%Y/%m/%d, %H:%M:%S')
                  
                            sub_cleanevt = self.time_filter(spacial_evtdata, tmrng=[ts, te])
                            counts.append(len(sub_cleanevt['TIME']))
          
                        box = ' (Box '+str(b)+')'
                        all_counts[box] = np.array(counts)
                        if make_final_graph == True:
                            
                            if count_rate == True:
                                
                                #livetime correction
                                livetimes = np.zeros(len(t_bin_edges)-1)
                                for t in range(len(t_bin_edges)-1):
                                    indices = ((self.hk_times>=t_bin_edges[t]) & (self.hk_times<t_bin_edges[t+1]))
                                    ltimes_in_range = self.hk_livetimes[indices]
                                    livetimes[t] = np.average(ltimes_in_range)
                                
                                counts_per_second = np.array(counts) / (livetimes * (times[1]-times[0]))
                                
                                fig = plt.figure()
                                ax = plt.axes()
                                dt_times = [(datetime.datetime(2010,1 ,1 ,0 ,0 ,0) + timedelta(seconds=t)) for t in times]

                                plt.plot(*self.stepped_lc_from_hist(self.dt_to_md(dt_times), counts_per_second))
                                plt.title('NuSTAR FPM'+self.fpm+' '+str(self.energy_range[0])+'-'+str(self.energy_range[1])+' keV Light Curve - '+start_yyyymmdd + box)

                                plt.xlim([dt_times[0] - timedelta(seconds=60), dt_times[-1] + timedelta(seconds=60)])
                                plt.xlabel('Start Time - '+start_hhmmss)
                                
                                plt.ylim([0, np.max(counts_per_second[np.isfinite(counts_per_second)])*1.05])
                                plt.ylabel('Counts $s^{-1}$')

                                fmt = mdates.DateFormatter('%H:%M:%S')
                                ax.xaxis.set_major_formatter(fmt)
                                ax.xaxis.set_major_locator(plt.LinearLocator(9))
                                plt.xticks(rotation=30)
                                plt.show()
                                all_count_rates[box] = counts_per_second
                            else:
                                fig = plt.figure()
                                ax = plt.axes()
                                dt_times = [(datetime.datetime(2010,1 ,1 ,0 ,0 ,0) + timedelta(seconds=t)) for t in times]

                                plt.plot(*self.stepped_lc_from_hist(self.dt_to_md(dt_times), counts))
                                plt.title('NuSTAR FPM'+self.fpm+' '+str(self.energy_range[0])+'-'+str(self.energy_range[1])+' keV Light Curve - '+start_yyyymmdd + box)

                                plt.xlim([dt_times[0] - timedelta(seconds=60), dt_times[-1] + timedelta(seconds=60)])
                                plt.xlabel('Start Time - '+start_hhmmss)
                                
                                plt.ylim([0, np.max(counts[np.isfinite(counts)])*1.05])
                                plt.ylabel('Counts')

                                fmt = mdates.DateFormatter('%H:%M:%S')
                                ax.xaxis.set_major_formatter(fmt)
                                ax.xaxis.set_major_locator(plt.LinearLocator(9))
                                plt.xticks(rotation=30)
                                plt.show()
                    self.lc_counts = all_counts
                    if all_count_rates == []:
                        self.lc_count_rates = None
                    else:
                        self.lc_count_rates = all_count_rates
                        
                    make_final_graph = False
                        
                else:
                    raise TypeError('Check the form of the sub-region was given in, e.g. need [bx,by,tx,ty] or [[bx,by,tx,ty], ...].')
            else:
                raise TypeError('\'astropy.io.fits.fitsrec.FITS_rec\' is the only supported data type at the moment.')
            
        else:
            if (t_bin[1] != 'exact') and (t_bin[1] != 'approx'):
                raise ValueError('Only options for the time bins is \'approx\' or \'exact\'.')
                
        if make_final_graph == True: #only in case multiple regions are plotted then they are handled in its own 'for' loop
            if count_rate == True:
                
                #livetime correction
                livetimes = np.zeros(len(t_bin_edges)-1)
                for t in range(len(t_bin_edges)-1):
                    indices = ((self.hk_times>=t_bin_edges[t]) & (self.hk_times<t_bin_edges[t+1]))
                    ltimes_in_range = self.hk_livetimes[indices]
                    livetimes[t] = np.average(ltimes_in_range)
                
                counts_per_second = self.lc_counts / (livetimes * (times[1]-times[0])) 
                fig = plt.figure()
                ax = plt.axes()
                
                dt_times = [(datetime.datetime(2010,1 ,1 ,0 ,0 ,0) + timedelta(seconds=t)) for t in times]

                plt.plot(*self.stepped_lc_from_hist(self.dt_to_md(dt_times), counts_per_second))
                plt.title('NuSTAR FPM'+self.fpm+' '+str(self.energy_range[0])+'-'+str(self.energy_range[1])+' keV Light Curve - '+start_yyyymmdd)

                plt.xlim([dt_times[0] - timedelta(seconds=60), dt_times[-1] + timedelta(seconds=60)])
                plt.xlabel('Start Time - '+start_hhmmss)
                
                plt.ylim([0, np.max(counts_per_second[np.isfinite(counts_per_second)])*1.05])
                plt.ylabel('Counts $s^{-1}$')
                
                fmt = mdates.DateFormatter('%H:%M:%S')
                ax.xaxis.set_major_formatter(fmt)
                ax.xaxis.set_major_locator(plt.LinearLocator(9))
                plt.xticks(rotation=30)
                
                plt.show()
                self.lc_times = dt_times
                self.lc_count_rates = counts_per_second
            else:
                fig = plt.figure()
                ax = plt.axes()
                dt_times = [(datetime.datetime(2010,1 ,1 ,0 ,0 ,0) + timedelta(seconds=t)) for t in times]

                plt.plot(*self.stepped_lc_from_hist(self.dt_to_md(dt_times), self.lc_counts))
                plt.title('NuSTAR FPM'+self.fpm+' '+str(self.energy_range[0])+'-'+str(self.energy_range[1])+' keV Light Curve - '+start_yyyymmdd)
                
                plt.xlim([dt_times[0] - timedelta(seconds=60), dt_times[-1] + timedelta(seconds=60)])
                plt.xlabel('Start Time - '+start_hhmmss)
                
                plt.ylim([0, np.max(self.lc_counts[np.isfinite(lc_counts)])*1.05])
                plt.ylabel('Counts')
                
                fmt = mdates.DateFormatter('%H:%M:%S')
                ax.xaxis.set_major_formatter(fmt)
                ax.xaxis.set_major_locator(plt.LinearLocator(9))
                plt.xticks(rotation=30)
                plt.show()
                self.lc_count_rates = None

     
    def save(self, save_dir='./', overwrite=False, **kwargs):
        #replace folder of saved data if run twice or just make a new one?
        """
        Can I automate the process using dir(nu) since this has every variable created?
        Or at least add to a list of attributes to be saved.
        Use os module to create appropriate directory structure for saved attributes.
        """
        #print(dir(nuA))
        '''
        Variables/info to save:
            ***** evt_file_used *****
            ~evt_directory, evt_filename, evt_data, evt_header #where did the data come from?
                ~meta data, chu_state, energy range, fpm, obs id

            ***** house_keeping_file_used *****
            ~self.hk_directory, self.hk_filename, hk_data, hk_header #what hk file was used?

            ***** nustar_livetime_data *****
            ~hk_livetimes, hk_times, livetimes plot

            ***** nustar_map_data *****
            ~rsn_map and plot (for plot need to run the nustar_plot() with save enabled) #what does it look like?
                ~gaussian filter applied, rectangle coordinates

            ***** nustar_light_curve_data *****
            ~lc_counts/lc_count_rates, lc_times, lightcurve plot(s)
                ~rectangle coordinates
        '''
        nustar_folder = save_dir + self.obs_id + '_nustar_folder'
        
        # Create target Directory if don't exist
        if not os.path.exists(nustar_folder + '/'):
            nustar_folder = nustar_folder + '/'
            os.mkdir(nustar_folder) #make empty folder
            print("Directory " , nustar_folder , " Created.", end='')
            
        # If the folder exists and overwrite is True then replace the first one
        elif os.path.exists(nustar_folder + '/') and (overwrite == True):
            nustar_folder = nustar_folder + '/'
            subprocess.check_output(['rm', '-r', nustar_folder]) #remove evrything in it too
            os.mkdir(nustar_folder) #make empty folder
            print("Replacing directory " , nustar_folder, end='')
            
        # If the folder exists and overwrite is False then just make another file with an index
        elif os.path.exists(nustar_folder + '/') and (overwrite == False): 
            number_exist = len(np.nonzero(['nustar_folder' in f for f in os.listdir(save_dir)])[0])
            nustar_folder = nustar_folder + '(' + str(number_exist) + ')/'
            os.mkdir(nustar_folder)
            print("Directory " , nustar_folder + '/' , " already exists. Creating another.", end='')
            
        # Now 'nustar_folder' is the folder things will be save into
        # Start with evt file information
        evt_folder = nustar_folder + 'evt_file_used/'
        os.mkdir(evt_folder)
        evt_list_to_save = ['evt_directory', 'evt_filename', 'obs_id', 'fpm', 'chu_state', 'energy_range',
                            'time_range', 'evt_data', 'evt_header', 'cleanevt']
        evt_info = list(set(dir(self)) & set(evt_list_to_save))
        evt_to_store = {}
        for name in evt_info:
            evt_to_store[name] = self.__dict__[name]
        with open(evt_folder + 'evt_file_info.pickle', 'wb') as evt_save_file:
            pickle.dump(evt_to_store, evt_save_file, protocol=pickle.HIGHEST_PROTOCOL)
            
        # hk file information
        hk_folder = nustar_folder + 'hk_file_used/'
        os.mkdir(hk_folder)
        hk_list_to_save = ['hk_directory', 'hk_filename', 'hk_data', 'hk_header']
        hk_info = list(set(dir(self)) & set(hk_list_to_save))
        hk_to_store = {}
        for name in hk_info:
            hk_to_store[name] = self.__dict__[name]
        with open(hk_folder + 'hk_file_info.pickle', 'wb') as hk_save_file:
            pickle.dump(hk_to_store, hk_save_file, protocol=pickle.HIGHEST_PROTOCOL)
            
        # Livetime info
        lvt_folder = nustar_folder + 'livetime_data/'
        os.mkdir(lvt_folder)
        lvt_list_to_save = ['hk_times', 'hk_livetimes']
        lvt_info = list(set(dir(self)) & set(lvt_list_to_save))
        lvt_to_store = {}
        for name in lvt_info:
            lvt_to_store[name] = self.__dict__[name]
        with open(lvt_folder + 'livetime_data.pickle', 'wb') as lvt_save_file:
            pickle.dump(lvt_to_store, lvt_save_file, protocol=pickle.HIGHEST_PROTOCOL)
            
        # Map info
        map_folder = nustar_folder + 'map_data/'
        os.mkdir(map_folder)
        map_list_to_save = ['rsn_map', 'gaussian_filter', 'time_norm', 'rectangles']
        map_info = list(set(dir(self)) & set(map_list_to_save))
        map_to_store = {}
        for name in map_info:
            try:
                map_to_store[name] = self.__dict__[name]
            except KeyError:
                map_to_store[name] = NustarDo.__dict__[name]     
        with open(map_folder + 'map_data.pickle', 'wb') as map_save_file:
            pickle.dump(map_to_store, map_save_file, protocol=pickle.HIGHEST_PROTOCOL)
            
        # Light curve info
        lc_folder = nustar_folder + 'light_curve_data/'
        os.mkdir(lc_folder)
        lc_list_to_save = ['lc_times', 'lc_counts', 'lc_count_rates', 'sub_reg_lc']
        lc_info = list(set(dir(self)) & set(lc_list_to_save))
        lc_to_store = {}
        for name in lc_info:
            lc_to_store[name] = self.__dict__[name]
        with open(lc_folder + 'light_curve_data.pickle', 'wb') as lc_save_file:
            pickle.dump(lc_to_store, lc_save_file, protocol=pickle.HIGHEST_PROTOCOL)
            
        # Can save your own stuff
        if len(kwargs) > 0:
            own_folder = nustar_folder
            with open(own_folder + 'own_data.pickle', 'wb') as own_save_file:
                pickle.dump(kwargs, own_save_file, protocol=pickle.HIGHEST_PROTOCOL)
                
        print(' Now Populated.')