'''
Functions to go in here (I think!?):
    KC: 01/12/2018, ideas-

    KC: 19/12/2018, added-
    ~NuSTAR class
'''

from . import data_handling

import sys
#from os.path import *
import os
from os.path import isfile
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
from . import filter_with_tmrng ######Kris
from . import custom_map ######Kris
import sunpy.map
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.ndimage import rotate
import re #for regular expressions
import warnings #suppress astropy warnings
import datetime
from datetime import timedelta
from astropy.io.fits.verify import VerifyWarning
import matplotlib.dates as mdates
import pickle
import subprocess
import pytz
from skimage import restoration
# from . import interp
from scipy import interpolate

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters() # was told to do this by the machine

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
    
    def __init__(self, evt_filename='', energy_range=[2.5,79], time_range = None): #set-up parameters
        #if a filename is not given then the static functions can still be used
        if evt_filename == '':
            return 

        #directory of the file
        directory_regex = re.compile(r'\w+/')
        directory = directory_regex.findall(evt_filename)
        self.evt_directory = '/'+''.join(directory)

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

        # search for a underscore, a non-digit, and an underscore (for the mode the pipeline was run if a chu file is given)
        mode_regex = re.compile(r"_\D_")
        mode = mode_regex.findall(name_of_file)
        self.pipeline_mode = mode[0] if len(mode)>0 else ""
        
        #search for all seperate sub-strings composed of digits, first one in evt_filename is observation id
        obs_id_regex = re.compile(r'\d+')
        obs_id = obs_id_regex.findall(name_of_file)[0]
        self.obs_id = obs_id
        
        #set attributes of the file and parameters used in other functions on the class
        self.evt_filename = name_of_file 
        self.fpm = focal_plane_module
        self.time_range = time_range 
        self.energy_range = energy_range
        self.chu_state = chu_state
        self.rectangles = None #set so that you don't have to plot a map to get a light curve

        # for plot title
        self.e_range_str = str(self.energy_range[0])+'-'+str(self.energy_range[1]) if self.energy_range[1]<79 else ">"+str(self.energy_range[0])

        self.rel_t = data_handling.getTimeFromFormat("2010/01/01, 00:00:00") # nustar times are measured in seconds from this date
        
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
            
            #start and end time of the NuSTAR observation as datetime objects
            self.time_range = [(self.rel_t+ timedelta(seconds=np.min(self.cleanevt['TIME']))).strftime('%Y/%m/%d, %H:%M:%S'),
                              (self.rel_t + timedelta(seconds=np.max(self.cleanevt['TIME']))).strftime('%Y/%m/%d, %H:%M:%S')]
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

        # now for the time tick marks...
        clevt_duration = np.max(self.cleanevt['TIME'])-np.min(self.cleanevt['TIME'])
        if clevt_duration > 3600*0.5:
            self.xlocator = mdates.MinuteLocator(byminute=[0, 10, 20, 30, 40, 50], interval = 1)
        elif 600 < clevt_duration <= 3600*0.5:
            self.xlocator = mdates.MinuteLocator(byminute=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55], interval = 1)
        elif 240 < clevt_duration <= 600:
            self.xlocator = mdates.MinuteLocator(interval = 2)
        else: 
            self.xlocator = mdates.MinuteLocator(interval = 1)
        


    @staticmethod
    def shift(evt_data, pix_xshift=None, pix_yshift=None):
        if pix_xshift != None:
            for X in evt_data:
                X['X'] = X['X'] + pix_xshift 
        if pix_yshift != None:
            for Y in evt_data:
                Y['Y'] = Y['Y'] + pix_yshift 
        return evt_data


    @staticmethod
    def arcsec_to_pixel(*args, **kwargs):
        #NuSTAR values: ['crpix1'+0.5,'crpix2','cdelt1']
        meta = {'centre_pix_val': [1499.5+0.5, 1500], 'arc_per_pix':[2.45810736], 'length':False} 
        #change list with kwargs
        for key, kwarg in kwargs.items():   
            meta[key] = kwarg

        #convert numbers so that they are easier to work with
        indices_for_centre = {'x':meta['centre_pix_val'][0], 'y':meta['centre_pix_val'][1]}
        assert 1 <= len(meta['arc_per_pix']) <= 2, '\'arc_per_pix\' needs to have one or two arguments only.'
        if len(meta['arc_per_pix']) == 2:
            delta_x = meta['arc_per_pix'][0]
            delta_y = meta['arc_per_pix'][1]
        elif len(meta['arc_per_pix']) == 1:
            delta_x = meta['arc_per_pix'][0]
            delta_y = meta['arc_per_pix'][0]

        # if have an arcsec length and want the length in pixels
        pixel_lengths = []
        if meta['length'] == True:
            for arg in args:
                x_length = (arg[0] / delta_x)
                y_length = (arg[1] / delta_y)
                pixel_lengths.append([int(round(x_length,0)), int(round(y_length,0))])
            return pixel_lengths

        #input coordinates as [x,y] in arcseconds
        pixel_coords = []
        for arg in args:
            x_index = indices_for_centre['x'] + (arg[0] / delta_x)
            y_index = indices_for_centre['y'] + (arg[1] / delta_y)
            pixel_coords.append([int(round(x_index,0)), int(round(y_index,0))])
        return pixel_coords


    @staticmethod
    def pixel_to_arcsec(*args, **kwargs):
        #NuSTAR values: ['crpix1'+0.5,'crpix2','cdelt1']
        meta = {'centre_pix_val': [1499.5+0.5, 1500], 'arc_per_pix':[2.45810736], 'length':False} 
        #change list with kwargs
        for key, kwarg in kwargs.items():   
            meta[key] = kwarg

        #convert numbers so that they are easier to work with
        indices_for_centre = {'x':meta['centre_pix_val'][0], 'y':meta['centre_pix_val'][1]}
        assert 1 <= len(meta['arc_per_pix']) <= 2, '\'arc_per_pix\' needs to have one or two arguments only.'
        if len(meta['arc_per_pix']) == 2:
            delta_x = meta['arc_per_pix'][0]
            delta_y = meta['arc_per_pix'][1]
        elif len(meta['arc_per_pix']) == 1:
            delta_x = meta['arc_per_pix'][0]
            delta_y = meta['arc_per_pix'][0]

        # if have a pixel length and want the length in arcsec
        arcsec_lengths = []
        if meta['length'] == True:
            for arg in args:
                x_length = arg[0] * delta_x
                y_length = arg[1] * delta_y
                arcsec_lengths.append([x_length, y_length])
            return arcsec_lengths

        #input coordinates as [col,row] in pixels
        arcsec_coords = []
        for arg in args:
            # arg[0] is x pixel position, so column
            x_arcsec = (arg[0] - indices_for_centre['x']) * delta_x
            # arg[1] is y pixel position, so row
            y_arcsec = (arg[1] - indices_for_centre['y']) * delta_y
            arcsec_coords.append([x_arcsec, y_arcsec])
        return arcsec_coords


    def nustar_shift_map(self, x_shift_arc, y_shift_arc):   
        #find shift in pix
        shift_pix = self.arcsec_to_pixel([x_shift_arc, y_shift_arc], length=True)
        #shift data now
        shift_cleanevt = self.shift(self.cleanevt, pix_xshift=shift_pix[0][0], pix_yshift=shift_pix[0][1]) 

        self.cleanevt = shift_cleanevt

    @staticmethod
    def fov_rotation(evt_data):
        """ Returns the average rotation of the NuSTAR FoV from the gradient of the edges between 
        det0&3 and 1&2.
        
        Parameters
        ----------
        *args : list [rawx0, rawy0, solx0, soly0, int]
            Each input should contain the raw X and Y coordinates from the (sunpos) evt file and the 
            solar X and Y coordinates from the sunpos evt file as well as the detector these 
            coordinates come from as an integer from 0 to 3.
            
        Returns
        -------
        A float of the average rotation from "North" in degrees where anticlockwise is positive.
        This assumes the rotation is between 90 and -90 degrees.
        
        Examples
        --------
        getMeanAngle([rawx0, rawy0, solx0, soly0, 0], 
                     [rawx1, rawy1, solx1, soly1, 1],
                     [rawx2, rawy2, solx2, soly2, 2], 
                     [rawx3, rawy3, solx3, soly3, 3])
        >>> a number
        """

        ## split the detectors
        d0_counts = evt_data[evt_data["det_id"]==0]
        d1_counts = evt_data[evt_data["det_id"]==1]
        d2_counts = evt_data[evt_data["det_id"]==2]
        d3_counts = evt_data[evt_data["det_id"]==3]

        ## now split up for the coordinates
        rawx0, rawy0, solx0, soly0 = d0_counts["RAWX"], d0_counts["RAWY"], d0_counts["X"], d0_counts["Y"]
        rawx1, rawy1, solx1, soly1 = d1_counts["RAWX"], d1_counts["RAWY"], d1_counts["X"], d1_counts["Y"]
        rawx2, rawy2, solx2, soly2 = d2_counts["RAWX"], d2_counts["RAWY"], d2_counts["X"], d2_counts["Y"]
        rawx3, rawy3, solx3, soly3 = d3_counts["RAWX"], d3_counts["RAWY"], d3_counts["X"], d3_counts["Y"]
        args = [[rawx0, rawy0, solx0, soly0, 0], 
                [rawx1, rawy1, solx1, soly1, 1],
                [rawx2, rawy2, solx2, soly2, 2], 
                [rawx3, rawy3, solx3, soly3, 3]]
        
        gradients = 0
        for a in args:
            rawx, rawy, solx, soly, det = a
            
            # use the pixel edges between det 0&3 and 1&2, use the raw pixel coordinates for this
            # orientation from the nustar_swguide.pdf, Figure 3
            if det==0:
                cols = collectSameXs(rawy, rawx, solx, soly)
                m_row_per_col = maxRowInCol(cols)
            elif det==1:
                cols = collectSameXs(rawx, rawy, solx, soly)
                m_row_per_col = maxRowInCol(cols)
            elif det==2:
                cols = collectSameXs(rawy, rawx, solx, soly)
                m_row_per_col = maxRowInCol(cols)
            elif det==3:
                cols = collectSameXs(rawx, rawy, solx, soly)
                m_row_per_col = maxRowInCol(cols)

            # working with rawx and y to make sure using correct edge then find the 
            # corresponding entries in solar coords
            aAndY = getXandY(m_row_per_col)
            x, y = aAndY[0], aAndY[1]
            
            ## do I want to filter some out?
            ## leave for now
            #if det in [0, 1]:
            #    x = x[y>np.median(y)]
            #    y = y[y>np.median(y)]
            #elif det in [2, 3]:
            #    x = x[y<np.median(y)]
            #    y = y[y<np.median(y)]
            
            # fit a straight line to the edge
            popt, pcov = curve_fit(straightLine, x, y, p0=[0, np.mean(y)])
            
            gradients += getDegrees(popt[0])

        return gradients/len(args)


    def nustar_deconv(self, map_array=None, psf_array=None, it=10, OA2source_offset=None, hor2SourceAngle=None, clip=False):
        """Class mathod to take a map (map_array) and a point spread function (psf_array) and deconvolve using 
        the Richardson-Lucy method with a number of iterations (it). 
    
        Parameters
        ----------
        map_array : 2d array
                The map of the data. Should be over the field of view. If "None" then the self.nustar_map class 
                attribute is used.
                Default: None

        psf_array : file string or 2d array
                The PSF you want to use. This can be a string of the fits file for the PSF or a 2d numpy array.
                If "None" then several common paths for nu'+self.fpm+'2dpsfen1_20100101v001.fits' are check and 
                if the file cannot be found the original map is returned. Currently this won't be rescaled if 
                it is a different resolution to the map data, it will just crash instead.
                Default: None

        it : int
                Number of iterations for the deconvolution.
                Default: 10

        OA2source_offset : float
                Angle subtended between the optical axis (OA), observer, and the X-ray source in arcminutes 
                (0<=OA2source_angle<8.5 arcminutes), i.e. radial distance to the source from the OA. Chooses 
                the correct PSF data to use.
                Default: None

        hor2SourceAngle : float
                Angle subtended between horizontal through the optical axis (OA), and the line through the X-ray source and OA in degrees.
                Clockwise is positive and anticlockwise is negative. Symmetric reflected in the origin so -90<=hor2SourceAngle<=90.
                Default: None

        clip : bool
                Set values >1 and <-1 to 1 and -1 respectively after each iteration. Unless working with a 
                normalised image this should be "False" otherwise it's a mess.
                Default: False
            
        Returns
        -------
        A 2d numpy array of the deconvolved map.

        Examples
        --------
        *Use within the class:
            NU_SUNPOS_FILE, ITERATIONS = "nustar_filename", 10
            nu = NustarDo(NU_SUNPOS_FILE)
            nu.deconvolve['apply'] = True
            nu.deconvolve['iterations'] = ITERATIONS
            nu.nustar_setmap(submap='FoV')
            deconv_map = nu.nustar_map.data

        *Use without class:
            STRING, FPM = "psf_filename", "A" or "B"
            nu = NustarDo()
            nu.fpm = FPM
            nu.nustar_map = Sunpy NuSTAR map
            deconv_map = nu.nustar_deconv(psf_array=STRING)

            -or-

            MAP, ARRAY, FPM = nustar data 2d numpy array, psf 2d numpy array, "A" or "B"
            nu = NustarDo()
            nu.fpm = FPM
            deconv_map = nu.nustar_deconv(map_array=MAP, psf_array=ARRAY)
        """

        ## for defaults
        if type(map_array) == type(None):
            map_array = self.nustar_map.data

        if type(psf_array) == type(None):
            # defualt is to check for the nu'+self.fpm+'2dpsfen1_20100101v001.fits' PSF file (the one used in Glesener code)
            trials = ['/opt/caldb/data/nustar/fpm/bcf/psf/nu'+self.fpm+'2dpsfen1_20100101v001.fits', 
                      '/usr/local/caldb/data/nustar/fpm/bcf/psf/nu'+self.fpm+'2dpsfen1_20100101v001.fits', 
                      '/home/kris/Desktop/link_to_kris_ganymede/old_scratch_kris/data_and_coding_folder/nustar_psfs/nu'+self.fpm+'2dpsfen1_20100101v001.fits',
                      '/home/kris/Desktop/nustar_psfs/nu'+self.fpm+'2dpsfen1_20100101v001.fits']

            if type(OA2source_offset) != type(None):
                psf_OA_angles = np.arange(0,9,0.5) # angles of 0 to 8.5 arcmin in 0.5 arcmin increments
                index = np.argmin([abs(psfoaangles - OA2source_offset) for psfoaangles in psf_OA_angles]) # find the closest arcmin array
                hdr_unit = index+1 # header units 1 to 18 (one for each of the arcmin entries) and 0 arcmin would be hdr_unit=1, hence the +1
                # print("using angle: ", hdr_unit)
            else:
                hdr_unit = 1

            #assume we can't find the file
            found_psf = False
            for t in trials:
                # try the files, if one exists use it
                if os.path.exists(t):
                    psfhdu = fits.open(t)
                    psf_h = psfhdu[hdr_unit].header['CDELT1'] # increment in degrees/pix
                    psf_array = psfhdu[hdr_unit].data
                    psfhdu.close()
                    psf_used = t
                    found_psf = True

            # if we still couldn't find a defualt PSF then print this, set self.deconvole to False, and just return the original map
            if found_psf == False:
                print('Could not find PSF file. Please provide the PSF filename or array.') 
                print('Returning original map.')
                self.deconvolve['apply'] = False
                self.deconv_settings_info = {'map':None, 'psf_file':None, 'psf_array':None, 'iterations':None}
                return map_array
                
            # check same res, at least in 1-D
            assert psf_h*3600 == self.nustar_map.meta['CDELT1'], "The resolution in the PSF and the current map are different."

        # if you have provided your own psf file use that instead
        elif type(psf_array) == str:
            psf_used = psf_array
            psfhdu = fits.open(psf_array)
            psf_h = psfhdu[1].header['CDELT1'] # increment in degrees/pix
            psf_array = psfhdu[1].data
            psfhdu.close()

            # check same res, at least in 1-D
            assert psf_h*3600 == self.nustar_map.meta['CDELT1'], "The resolution in the PSF and the current map are different."
            
        else:
            psf_used = 'Custom Array. Hopefully some numbers though.'

        if type(hor2SourceAngle)!=type(None):
            assert -90<=hor2SourceAngle<=90, "Please give \"hor2SourceAngle\" as an angle from horzontal to the source -90<=hor2SourceAngle<=90 where clockwise is positive and anticlockwise is negative"
            psf_array = rotate(psf_array, hor2SourceAngle, reshape=True)

        # deconvolve
        deconvolved_RL = restoration.richardson_lucy(map_array, psf_array, iterations=it, clip=False)
        
        # deconvolution info for later use
        self.deconv_settings_info = {'map':map_array, 'psf_file':psf_used, 'psf_array':psf_array, 'iterations':it}
        return deconvolved_RL


    @staticmethod
    def find_boxOfData(array):
        '''If there is an array with loads of 0s or nans and a region of numbers then this returns the rows 
           and columns the block of numbers is encased between'''
        array = np.array(array)
        array[np.isnan(array)] = 0

        # first and last row
        dataRows = []
        for i,row in enumerate(array):
            rSum = np.sum(row)
            if rSum > 0:
                dataRows.append(i)
        between_rows = [dataRows[0], dataRows[-1]]
       
        # first and last column
        dataCols = []
        for j,col in enumerate(array.T):
            cSum = np.sum(col)
            if cSum > 0:
                dataCols.append(j)
        between_cols = [dataCols[0], dataCols[-1]]

        return {'rowIndices':between_rows, 'columnIndices':between_cols}


    @staticmethod    
    def create_submap(sunpy_map_obj, lose_off_limb, submap):
        if (lose_off_limb == True) and (len(submap) == 0):
            #fix really large plot, instead of going from -3600 to 3600 in x and y
            bl = SkyCoord(-1200*u.arcsec, -1200*u.arcsec, frame=sunpy_map_obj.coordinate_frame)
            tr = SkyCoord(1200*u.arcsec, 1200*u.arcsec, frame=sunpy_map_obj.coordinate_frame)
            return sunpy_map_obj.submap(bl,top_right=tr)

        elif len(submap) == 4: #Submap to plot?
            bottom_left = {'x':submap[0], 'y':submap[1]}
            top_right = {'x':submap[2], 'y':submap[3]}
            
            bl = SkyCoord(bottom_left['x']*u.arcsec, bottom_left['y']*u.arcsec, frame=sunpy_map_obj.coordinate_frame)
            tr = SkyCoord(top_right['x']*u.arcsec, top_right['y']*u.arcsec, frame=sunpy_map_obj.coordinate_frame)
            return sunpy_map_obj.submap(bl,top_right=tr)

        else:
            raise TypeError('\nCheck the submap coordinates that were given please. It should be a list with four '
                            'float/int entries in arcseconds in the form [bottom left x, bottom left y, top right x, '
                            'top right y].')

        if (self.deconvolve['apply'] == True) and (self.gaussian_filter['apply'] == True):
            print('Caution! Did you mean to set deconvolve AND gaussian blurr to True? If so, then the'
                  'deconvolution will happen first then the Gaussian filter is applied.')



    # might be best to only allow one of these at a time, either deconvolve OR gaussian filter
    deconvolve = {'apply':False, 'iterations':10, 'OA2source_offset':None, 'hor2SourceAngle':None, 'clip':False} # set before nustar_setmap to run deconvolution on map
    gaussian_filter = {'apply':False, 'sigma':2, 'mode':'nearest'}
    sub_lt_zero = np.nan # replace less than zeroes with this value for plotting in a linear scale
    own_map = None # if you already have a map that you want a submap of then set this, be careful not to time normalize again though
    
    def nustar_setmap(self, time_norm=True, lose_off_limb=True, limits=None,
                   submap=None, rebin_factor=1, norm='linear', house_keeping_file=None):
        # adapted from Iain's python code

        # Map the filtered evt, into one corrected for livetime (so units count/s) 
        if type(self.own_map) == type(None):
            self.nustar_map = custom_map.make_sunpy(self.cleanevt, self.evt_header, norm_map=False)
        else:
            self.nustar_map = self.own_map
            if time_norm == True:
                time_norm = input('Caution! Do you mean to time normalize your \'own_map\'? True or False: ')

        # field of view in arcseconds
        FoVlimits = self.find_boxOfData(self.nustar_map.data)
        bottom_left = self.pixel_to_arcsec([FoVlimits['columnIndices'][0], FoVlimits['rowIndices'][0]])[0]
        top_right = self.pixel_to_arcsec([FoVlimits['columnIndices'][1]+1, FoVlimits['rowIndices'][1]+1])[0] # plus one as index stops one short
        self.FoV = [*bottom_left, *top_right]

        if limits == None:
            limits = []
        if submap == None:
            submap = []
        elif type(submap) == str:
            if submap.upper() == 'FOV':
                submap = self.FoV
            else:
                print('The only string input to submap that is supported at the moment is FOV, fov, FoV, etc.')
        self.submap = submap
        
        self.time_norm = time_norm
        if self.time_norm == True:
            self.livetime(hk_filename=house_keeping_file, set_up_plot=False, show_fig=False)
            #livetime correction
            time_range = [(data_handling.getTimeFromFormat(tm) - self.rel_t).total_seconds() for tm in self.time_range]
            indices = ((self.hk_times>=time_range[0]) & (self.hk_times<time_range[1]))
            ltimes_in_range = self.hk_livetimes[indices]
            livetime = np.average(ltimes_in_range)
            lc_cor_nustar_map = self.nustar_map.data / (livetime * (time_range[1] - time_range[0]))
            self.nustar_map = sunpy.map.Map(lc_cor_nustar_map, self.nustar_map.meta)
            
        if (self.deconvolve['apply'] == False):
            self.nustar_map = self.create_submap(self.nustar_map, lose_off_limb, self.submap)
        
        elif (self.deconvolve['apply'] == True):
            # make sure it's over the FoV
            self.nustar_map = self.create_submap(self.nustar_map, lose_off_limb, self.FoV)
            dconv = self.nustar_deconv(it=self.deconvolve['iterations'], OA2source_offset=self.deconvolve['OA2source_offset'], 
            	                       hor2SourceAngle=self.deconvolve['hor2SourceAngle'], clip=self.deconvolve['clip'])
            # make new map
            self.nustar_map = sunpy.map.Map(dconv, self.nustar_map.meta)
            # now cut to the shape you want
            self.nustar_map = self.create_submap(self.nustar_map, lose_off_limb, self.submap)

            
        if self.gaussian_filter['apply'] == True:
            gaussian_width = self.gaussian_filter['sigma']
            m = self.gaussian_filter['mode']
            #Apply a guassian blur to the data to bring out the faint feature
            dd = ndimage.gaussian_filter(self.nustar_map.data, gaussian_width, mode=m)
            if limits == []:
                dmin = np.min(dd[np.nonzero(self.nustar_map.data)])#*1e6 factor was here as the lowest value will come (came from dd) from the gaussian
                #filter and not the actual lowest count rate hence the factor 
                dmax = np.max(dd[np.isfinite(self.nustar_map.data)])
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
            else:
                raise TypeError('\nCheck the limits that were given please.')
        else:
            dd = self.nustar_map.data
            if limits == []:
                finite_vals = dd[np.isfinite(dd)]
                dmin = np.min(finite_vals[np.nonzero(finite_vals)])
                dmax = np.max(finite_vals)
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
            else:
                raise TypeError('\nCheck the limits that were given please. It should be a list with two float/int '
                                'entries')

        self.dmin = dmin # make it possible to get min and max normalisation values of the NuSTAR map
        self.dmax = dmax

        # Tidy up before plotting
        dd[dd < dmin]=0
        nm = sunpy.map.Map(dd, self.nustar_map.meta)
        
        if rebin_factor != 1:
            #can rebin the pixels if we want to further bring out faint features
            #set to 1 means no actual rebinning
            nx,ny = np.shape(nm.data)
            if rebin_factor >= 1/nx and rebin_factor >= 1/ny:
                dimensions = u.Quantity([nx*rebin_factor, ny*rebin_factor], u.pixel)
                rsn_map = nm.resample(dimensions)
            else:
                raise TypeError(f'\nRebin factor must be greater than one over the x,y dimensions (1/{nx} and '
                                f'1/{ny}) as to rebin to get one, or more, pixel(s) fro the entire image, i.e. can\'t rebin to half a pixel.')
        elif rebin_factor == 1:
            rsn_map = nm
            del nm
        
        if norm == 'linear':
            #change all zeros to NaNs so they appear white in the plot otherwise zeros appear as the lowest colour 
            #on the colourbar
            rsn_map_data = rsn_map.data
            rsn_map_data[rsn_map_data <= 0] = self.sub_lt_zero
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
        

    annotations = {'apply':False, 'text':'Some text', 'position':(0,0), 'color':'black', 'fontsize':12, 'weight':'normal'}
    rcParams_default_setup = True
    cbar_title = 'Counts'
    ax_label_size = 18
        
    def nustar_plot(self, boxes=None, show_fig=True, save_fig=None, usr_title=None):
        # adapted from Iain's python code

        if self.rcParams_default_setup:
            matplotlib.rcParams['font.sans-serif'] = "Arial"
            matplotlib.rcParams['font.family'] = "sans-serif"
            plt.rcParams["figure.figsize"] = (10,8)
            plt.rcParams['font.size'] = 18
            plt.rcParams['axes.facecolor']='white'
            plt.rcParams['savefig.facecolor']='white'
            # Start the plot - many things here just to make matplotlib look decent
        
        self.rectangles = boxes

        #fig = plt.figure(figsize=(9, 8), frameon=False)
        ax = plt.subplot(projection=self.rsn_map, frame_on=False) #rsn_map nustar_submap
        ax.set_facecolor((1.0, 1.0, 1.0))

        self.rsn_map.plot()
        self.rsn_map.draw_limb(color='black',linewidth=1,linestyle='dashed',zorder=0)

        if self.annotations['apply'] == True:
            plt.annotate(self.annotations['text'], self.annotations['position'], color=self.annotations['color'], fontsize=self.annotations['fontsize'], weight=self.annotations['weight'])

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
        title_obsdate = self.rsn_map.date.strftime('%Y-%b-%dT%H:%M:%S.%f')[:-13] #'{:.20}'.format('{:%Y-%b-%d}'.format(self.rsn_map.date))
        fpm = 'FPM'+self.fpm
        title_obstime_start = self.time_range[0][-8:]
        title_obstime_end = self.time_range[1][-8:]
        
        if type(usr_title) == type(None):
            if self.chu_state == 'not_split':
                ax.set_title('NuSTAR '+self.e_range_str+' keV '+fpm+' '+ title_obsdate+' '+title_obstime_start+' to '+title_obstime_end)
            else:
                ax.set_title('NuSTAR '+self.e_range_str+' keV '+fpm+' '+self.chu_state+' '+ title_obsdate+' '+title_obstime_start+' to '+title_obstime_end)
        else:
            ax.set_title(usr_title)
        
        ax.set_ylabel('y [arcsec]', fontsize=self.ax_label_size)
        ax.set_xlabel('x [arcsec]', fontsize=self.ax_label_size)
        tx, ty = ax.coords
        tx.set_major_formatter('s')
        ty.set_major_formatter('s')
        ax.grid(False)

        # Add a colour bar
        if self.time_norm == True:
            plt.colorbar(fraction=0.035, pad=0.03,label=self.cbar_title+' $s^{-1}$')
        else:
            plt.colorbar(fraction=0.035, pad=0.03,label=self.cbar_title)
            
        if boxes is not None:
            if np.shape(boxes)==(4,):
                rect = boxes
                bottom_left_rectangle = SkyCoord(rect[0]*u.arcsec, rect[1]*u.arcsec, frame=self.rsn_map.coordinate_frame)
                length = rect[2] - rect[0]
                height = rect[3] - rect[1]
                self.rsn_map.draw_rectangle(bottom_left_rectangle, width=length*u.arcsec, height=height*u.arcsec, color='black')
            else:
                b = 1
                for rect in boxes:
                    bottom_left_rectangle = SkyCoord(rect[0]*u.arcsec, rect[1]*u.arcsec, frame=self.rsn_map.coordinate_frame)
                    length = rect[2] - rect[0]
                    height = rect[3] - rect[1]
                    self.rsn_map.draw_rectangle(bottom_left_rectangle, width=length*u.arcsec, height=height*u.arcsec, color='black')
                    for_text = self.arcsec_to_pixel([rect[0]-10,rect[3]+20], centre_pix_val= [self.rsn_map.meta['crpix1']+0.5, self.rsn_map.meta['crpix2']])
                    plt.text(for_text[0][0], for_text[0][1], 'Box '+str(b), fontsize=10)
                    b += 1
                    
        if save_fig != None:
            plt.savefig(save_fig, dpi=300, bbox_inches='tight')
        if show_fig == True:
            plt.show('all')
            
    
    def nustar_peek(self):
        #just to view the map with all default settings
        self.nustar_setmap()
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
    def spatial_filter(evt_data, sub_region_in_pixels):
        x = evt_data['X']
        y = evt_data['Y']
        #find indices within the x and y pixel range
        indices = (sub_region_in_pixels[0][0] < x)&(x<= sub_region_in_pixels[1][0]) & \
        (sub_region_in_pixels[0][1] < y)&(y <= sub_region_in_pixels[1][1])
        evt_data = evt_data[:len(indices)][indices] # [:len(indices)] is a quick fix, doesn't work otherwise if cleanevt is loaded from pickle
        return evt_data
    
    
    @staticmethod
    def time_filter(evtdata, tmrng=None):
        ''' ***** From filter function ***** >4x quicker to just filter with time than with full filter ***** '''
        if tmrng is None:
            tmrng = [evtdata['TIME'][0], evtdata['TIME'][-1]]
        elif tmrng is not None:
            tstart = data_handling.getTimeFromFormat(tmrng[0]) #date must be in this format 'yyyy/mm/dd, HH:MM:SS'
            tend = data_handling.getTimeFromFormat(tmrng[1])
            rel_t = data_handling.getTimeFromFormat("2010/01/01, 00:00:00") #the date NuSTAR times are defined from
            tstart_s = (tstart - rel_t).total_seconds() #both dates are converted to number of seconds from 2010-Jan-1  
            tend_s = (tend - rel_t).total_seconds()
            tmrng = [tstart_s, tend_s] 

        time_filter = ( (evtdata['TIME']>tmrng[0]) & (evtdata['TIME']<tmrng[1]) )
        inds = (time_filter).nonzero()  
        goodinds=inds[0]       

        return evtdata[goodinds]


    @staticmethod
    def nustar_file_finder(start_directory='', obs_id='', descriptor='', fpm='', ext=''):

        full_filename = None
        file_directory = None
        file_name = None
        #expression for everything that ends in a slash
        search_directory_regex = re.compile(r'\w+/')

        #find all the folders in the evt directory (they end with a slash)
        search_directory = search_directory_regex.findall(start_directory)

        #don't includce the last folder to go back a directory
        search_directory = '/'+''.join(search_directory[:-1]) #go back a directory to search for the house keeping file
        for _dirpath, _dirnames, _filenames in os.walk(search_directory):
            for _file in _filenames:
                if _file == 'nu' + obs_id + fpm + descriptor + ext:
                    full_filename = os.path.join(_dirpath, _file)
                    file_directory = _dirpath
                    file_name = _file

        return full_filename, file_directory, file_name
        
        
    def livetime(self, hk_filename=None, set_up_plot=True, show_fig=True):
        #file = '/Users/kris/Documents/PhD/data/nustar/nu80414201001A_fpm.hk'
        '''
        This has to be moved above the time profile function so it is defined to be called
        '''
        if self.rcParams_default_setup:
            matplotlib.rcParams['font.sans-serif'] = "Arial"
            matplotlib.rcParams['font.family'] = "sans-serif"
            plt.rcParams["figure.figsize"] = (10,6)
            plt.rcParams['font.size'] = 18
        
        if hk_filename == None:
            hk_filename, self.hk_directory, self.hk_filename = self.nustar_file_finder(start_directory=self.evt_directory, obs_id=self.obs_id, descriptor='_fpm', fpm=self.fpm, ext='.hk')

            if hk_filename == None: #if there is still no hk_filename then there won't be one used
                print('Unable to find appropriate .hk file.')
                self.hk_times = 0
                self.hk_livetimes = [] # so the this length is 0
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
        self.lvt_times = [(self.rel_t + timedelta(seconds=t)) for t in self.hk_times]
        self.hk_livetimes = self.hk_data['livetime']
        
        if set_up_plot:
            hktime = self.hk_times - self.hk_times[0]
            dt_times = self.lvt_times
            lt_start_hhmmss = str((self.rel_t + timedelta(seconds=np.min(self.hk_times))).strftime('%Y/%m/%d, %H:%M:%S'))
            fig = plt.figure()
            ax = plt.axes()
            plt.semilogy(self.dt_to_md(dt_times), self.hk_livetimes, drawstyle='steps-mid')
            plt.title('Livetime - '+lt_start_hhmmss[:10]) #get the date in the title
            plt.xlabel('Start Time - '+lt_start_hhmmss[12:])
            plt.ylabel('Livetime Fraction')
            plt.xlim([data_handling.getTimeFromFormat(t) for t in self.time_range])#[dt_times[0], dt_times[-1]])
            plt.ylim([0,1])
            fmt = mdates.DateFormatter('%H:%M')
            ax.xaxis.set_major_formatter(fmt)
            ax.xaxis.set_major_locator(self.xlocator) # xlocator was plt.LinearLocator(9)
            plt.xticks(rotation=30)

            if show_fig == True:
                plt.show()        

        
    t_bin = {'seconds_per_bin':10, 'method':'approx'}

    def light_curve(self, cleanevt=None, hdr=None, sub_reg=None, tstart=None, tend=None, 
                    count_rate=True, house_keeping_file=None, show_fig=True):     

        if self.rcParams_default_setup:
            matplotlib.rcParams['font.sans-serif'] = "Arial"
            matplotlib.rcParams['font.family'] = "sans-serif"
            plt.rcParams["figure.figsize"] = (10,6)
            plt.rcParams['font.size'] = 18
        
        if cleanevt == None:
            cleanevt = self.cleanevt
        if hdr == None:
            hdr = self.evt_header
        if sub_reg == 'boxes':
            sub_reg = self.rectangles
            self.sub_reg_lc = sub_reg
        single_lc = True # just start by assuming one light curve, don't worry, this only gets set to False if not

        if tstart == None:
            tstart = np.min(cleanevt['TIME'])
            self.rel_tstart = tstart #already relative to 1/1/2010 and in seconds
        else:
            tstart = data_handling.getTimeFromFormat(tstart) 
            self.rel_tstart = (tstart - self.rel_t).total_seconds()

        if tend == None:
            tend = np.max(cleanevt['TIME'])
            self.rel_tend = tend #already relative to 1/1/2010 and in seconds
        else:
            tend = data_handling.getTimeFromFormat(tend)
            self.rel_tend = (tend - self.rel_t).total_seconds() 
            
        if count_rate == True:
            self.livetime(hk_filename=house_keeping_file, set_up_plot=False, show_fig=False) #run to get times and livetimes
            if len(self.hk_times) == 0:
                decision = input('No livetimes present. Do you just want to see the counts vs. time instead: ')
                if decision in ['Yes', 'yes', 'Y', 'y']:
                    count_rate = False
                else:
                    print('Will not show plot.')
                    return

        self.lc_livetimes = 0 # just to have it defined

        if self.t_bin['method'] == 'approx':
            if (type(cleanevt) == astropy.io.fits.fitsrec.FITS_rec) and (sub_reg == None): #data form of NuSTAR
                t_bin_conversion = int((self.rel_tend - self.rel_tstart) // self.t_bin['seconds_per_bin']) #get approximately t_bin seconds per bin as start and end of 
                #data are fixed when the histogram is created
                assert t_bin_conversion >= 1, 'Number of bins cannot be <1. Decrease \'t_bin\' value to get more bins.'

                counts = np.histogram(cleanevt['TIME'], t_bin_conversion) #gives out bin values and bin edges
                self.lc_counts = counts[0]
                times = counts[1][:-1]
                self.t_bin_edges = counts[1]
                
                start_hhmmss = str((self.rel_t + timedelta(seconds=np.min(times))).strftime('%H:%M:%S'))
                start_yyyymmdd = str((self.rel_t + timedelta(seconds=np.min(times))).strftime('%Y/%m/%d'))
                
            elif (type(cleanevt) == astropy.io.fits.fitsrec.FITS_rec) and (sub_reg != None):
                #this is to plot the light curve of a sub-region.
                print('Inconvenient to approximate the time bins for the light curve of a sub_region.'
                      '\nChanging to \'exact\'.')
                self.t_bin['method'] = 'exact'
            else:
                raise TypeError('\'astropy.io.fits.fitsrec.FITS_rec\' is the only supported data type at the moment.')

        if self.t_bin['method'] == 'exact': #if since if the 'approx' flag is up and also submap!=None then time profile should be made here
            t_bin_number = int((self.rel_tend - self.rel_tstart) // self.t_bin['seconds_per_bin']) #get whole number of bins that are t_bin seconds long and
            #doesn't include any time at the end that only has data for some of the last range

            assert t_bin_number >= 1, 'Number of bins cannot be <1. Decrease \'t_bin\' value to get more bins.'

            edge = self.rel_tstart
            self.t_bin_edges = np.zeros(t_bin_number+1) #+1 for the last edge
            for t in range(len(self.t_bin_edges)):
                self.t_bin_edges[t] = edge
                edge += self.t_bin['seconds_per_bin']
            times = self.t_bin_edges[:-1]
            
            start_hhmmss = str((self.rel_t + timedelta(seconds=np.min(times))).strftime('%H:%M:%S'))
            start_yyyymmdd = str((self.rel_t + timedelta(seconds=np.min(times))).strftime('%Y/%m/%d'))
            
            if (type(cleanevt) == astropy.io.fits.fitsrec.FITS_rec) and (sub_reg == None): #data form of NuSTAR

                counts = np.histogram(cleanevt['TIME'], self.t_bin_edges) #gives out bin values and bin edges
                self.lc_counts = counts[0]
                    
            elif (type(cleanevt) == astropy.io.fits.fitsrec.FITS_rec) and (sub_reg != None):
                if np.shape(sub_reg) == (4,):
                    counts = []
                    pixels = self.arcsec_to_pixel([sub_reg[0],sub_reg[1]], [sub_reg[2],sub_reg[3]])
                    spatial_evtdata = self.spatial_filter(self.cleanevt, pixels)
                    for t in range(len(self.t_bin_edges)-1):
                        # ts = (datetime.datetime(1970, 1, 1) + timedelta(seconds=(float(self.rel_t.strftime("%s"))+self.t_bin_edges[t]))).strftime('%Y/%m/%d, %H:%M:%S')
                        # te = (datetime.datetime(1970, 1, 1) + timedelta(seconds=(float(self.rel_t.strftime("%s"))+self.t_bin_edges[t+1]))).strftime('%Y/%m/%d, %H:%M:%S')
                        ts = (self.rel_t + timedelta(seconds=self.t_bin_edges[t])).strftime('%Y/%m/%d, %H:%M:%S')
                        te = (self.rel_t + timedelta(seconds=self.t_bin_edges[t+1])).strftime('%Y/%m/%d, %H:%M:%S')
      
                        sub_cleanevt = self.time_filter(spatial_evtdata, tmrng=[ts, te])
                        counts.append(len(sub_cleanevt['TIME']))
           
                    self.lc_counts = np.array(counts)

                elif np.shape(sub_reg)[1] == 4:  
                    all_counts = {}
                    all_count_rates = {}
                    for b, sub_r in enumerate(sub_reg, start=1):
                        counts = []
                        
                        pixels = self.arcsec_to_pixel([sub_r[0],sub_r[1]], [sub_r[2],sub_r[3]])
                        spatial_evtdata = self.spatial_filter(self.cleanevt, pixels)
                        
                        for t in range(len(self.t_bin_edges)-1):
                            ts = (self.rel_t + timedelta(seconds=self.t_bin_edges[t])).strftime('%Y/%m/%d, %H:%M:%S')
                            te = (self.rel_t + timedelta(seconds=self.t_bin_edges[t+1])).strftime('%Y/%m/%d, %H:%M:%S')
                  
                            sub_cleanevt = self.time_filter(spatial_evtdata, tmrng=[ts, te])
                            counts.append(len(sub_cleanevt['TIME']))
          
                        box = ' (Box '+str(b)+')'
                        all_counts[box] = np.array(counts)

                        #if make_final_graph == True:
                            
                        if count_rate == True:
                                
                                #livetime correction
                                livetimes = np.zeros(len(self.t_bin_edges)-1)
                                for t in range(len(self.t_bin_edges)-1):
                                    indices = ((self.hk_times>=self.t_bin_edges[t]) & (self.hk_times<self.t_bin_edges[t+1]))
                                    ltimes_in_range = self.hk_livetimes[indices]
                                    livetimes[t] = np.average(ltimes_in_range)
                                self.lc_livetimes = livetimes
                                
                                counts_per_second = np.array(counts) / (livetimes * (times[1]-times[0]))
                                
                                fig = plt.figure()
                                ax = plt.axes()
                                dt_times = [(self.rel_t + timedelta(seconds=t)) for t in times]

                                plt.plot(*self.stepped_lc_from_hist(self.dt_to_md(dt_times), counts_per_second))
                                plt.title('NuSTAR FPM'+self.fpm+' '+self.e_range_str+' keV Light Curve - '+start_yyyymmdd + box)

                                plt.xlim([data_handling.getTimeFromFormat(t) for t in self.time_range])
                                plt.xlabel('Start Time - '+start_hhmmss)
                                
                                plt.ylim([0, np.max(counts_per_second[np.isfinite(counts_per_second)])*1.05])
                                plt.ylabel('Counts $s^{-1}$')

                                fmt = mdates.DateFormatter('%H:%M')
                                ax.xaxis.set_major_formatter(fmt)
                                ax.xaxis.set_major_locator(self.xlocator)
                                plt.xticks(rotation=30)
                                #plt.show()
                                all_count_rates[box] = counts_per_second
                        else:
                                fig = plt.figure()
                                ax = plt.axes()
                                dt_times = [(self.rel_t + timedelta(seconds=t)) for t in times]

                                plt.plot(*self.stepped_lc_from_hist(self.dt_to_md(dt_times), counts))
                                plt.title('NuSTAR FPM'+self.fpm+' '+self.e_range_str+' keV Light Curve - '+start_yyyymmdd + box)

                                plt.xlim([data_handling.getTimeFromFormat(t) for t in self.time_range])
                                plt.xlabel('Start Time - '+start_hhmmss)
                                
                                plt.ylim([0, np.max(counts[np.isfinite(counts)])*1.05])
                                plt.ylabel('Counts')

                                fmt = mdates.DateFormatter('%H:%M')
                                ax.xaxis.set_major_formatter(fmt)
                                ax.xaxis.set_major_locator(self.xlocator)
                                plt.xticks(rotation=30)
                                #plt.show()
                    self.lc_counts = all_counts
                    if all_count_rates == []:
                        self.lc_count_rates = None
                    else:
                        self.lc_count_rates = all_count_rates
                     
                    self.lc_times = dt_times
                    if show_fig:
                        plt.show()
                        
                    single_lc = False
                        
                else:
                    raise TypeError('Check the form of the sub-region was given in, e.g. need [bx,by,tx,ty] or [[bx,by,tx,ty], ...].')
            else:
                raise TypeError('\'astropy.io.fits.fitsrec.FITS_rec\' is the only supported data type at the moment.')
            
        else:
            if (self.t_bin['method'] != 'exact') and (self.t_bin['method'] != 'approx'):
                raise ValueError('Only options for the time bins is \'approx\' or \'exact\'.')
                
        if single_lc == True: #only in case multiple regions are plotted then they are handled in its own 'for' loop
            if count_rate == True:
                
                #livetime correction
                livetimes = np.zeros(len(self.t_bin_edges)-1)
                for t in range(len(self.t_bin_edges)-1):
                    indices = ((self.hk_times>=self.t_bin_edges[t]) & (self.hk_times<self.t_bin_edges[t+1]))
                    ltimes_in_range = self.hk_livetimes[indices]
                    livetimes[t] = np.average(ltimes_in_range)
                self.lc_livetimes = livetimes
                
                counts_per_second = self.lc_counts / (livetimes * (times[1]-times[0])) 
                fig = plt.figure()
                ax = plt.axes()
                
                dt_times = [(self.rel_t + timedelta(seconds=t)) for t in times]

                plt.plot(*self.stepped_lc_from_hist(self.dt_to_md(dt_times), counts_per_second))
                plt.title('NuSTAR FPM'+self.fpm+' '+self.e_range_str+' keV Light Curve - '+start_yyyymmdd)

                plt.xlim([data_handling.getTimeFromFormat(t) for t in self.time_range])
                plt.xlabel('Start Time - '+start_hhmmss)
                
                plt.ylim([0, np.max(counts_per_second[np.isfinite(counts_per_second)])*1.05])
                plt.ylabel('Counts $s^{-1}$')
                
                fmt = mdates.DateFormatter('%H:%M')
                ax.xaxis.set_major_formatter(fmt)
                ax.xaxis.set_major_locator(self.xlocator)
                plt.xticks(rotation=30)
                
                #plt.show()
                self.lc_times = dt_times
                self.lc_count_rates = counts_per_second
            else:
                fig = plt.figure()
                ax = plt.axes()
                dt_times = [(self.rel_t + timedelta(seconds=t)) for t in times]

                plt.plot(*self.stepped_lc_from_hist(self.dt_to_md(dt_times), self.lc_counts))
                plt.title('NuSTAR FPM'+self.fpm+' '+self.e_range_str+' keV Light Curve - '+start_yyyymmdd)
                
                plt.xlim([data_handling.getTimeFromFormat(t) for t in self.time_range])
                plt.xlabel('Start Time - '+start_hhmmss)
                
                plt.ylim([0, np.max(self.lc_counts[np.isfinite(self.lc_counts)])*1.05])
                plt.ylabel('Counts')
                
                fmt = mdates.DateFormatter('%H:%M')
                ax.xaxis.set_major_formatter(fmt)
                ax.xaxis.set_major_locator(self.xlocator)
                plt.xticks(rotation=30)
                #plt.show()
                self.lc_times = dt_times
                self.lc_count_rates = None
        if show_fig:
            plt.show()


    def full_obs_chus(self, start_directory=None, obs_id=None, descriptor='_chu123', ext='.fits' ,show_fig=True):
        '''
        Apapted from: 
        https://github.com/ianan/nustar_sac/blob/master/idl/load_nschu.pro
        and
        https://github.com/NuSTAR/nustar_solar/blob/master/depricated/solar_mosaic_20150429/read_chus.pro
        '''
        if start_directory == None:
            start_directory=self.evt_directory
        if obs_id == None:
            obs_id=self.obs_id

        chu_filename, self.chu_directory, self.chu_filename = self.nustar_file_finder(start_directory=start_directory, obs_id=obs_id, descriptor=descriptor, ext=ext)

        #not self.chu_filename as fits.open needs to know the full path to the file
        hdulist = fits.open(chu_filename) 
        data1 = hdulist[1].data
        data2 = hdulist[2].data
        data3 = hdulist[3].data
        hdulist.close()

        # easier to work with numpy arrays later
        data_c1 = np.array(data1)
        data_c2 = np.array(data2)
        data_c3 = np.array(data3)

        maxres = 20
        
        for chu_num, dat in enumerate([data_c1, data_c2, data_c3]):
            chu_bool = ((dat['VALID']==1) & 
                        (dat['RESIDUAL']<maxres) &
                        (dat['STARSFAIL']<dat['OBJECTS']) &
                        (dat['CHUQ'][:,3]!=1))
            chu_01 = chu_bool*1 # change true/false into 1/0
    
            chu_mask = chu_01* (chu_num+1)**2 # give each chu a unique number that when it is added to another it gives a unique chu combo, like file permissions
    
            if chu_num == 0:
                chu_all = chu_mask # after chu 1 file have an array with 1s and 0s
            else:
                chu_all += chu_mask # after the others (chu2 and chu3) have an array with 1,4,9,5,10,13,14
        
        # last data array in the for loop can give the time, no. of seconds from 1-Jan-2010
        chu_time = dat['TIME']

        # reassigned values are at 100, etc. as to not accidently double sort the values again
        # e.g. if mask value was changed to 10, then if it was accidently run again it would get sorted into chu state 13 etc.
        chu_all[chu_all == 1] = 100 #chu1 # mask value in array is changed to chu state, e.g. mask value=5, chu state is 12, and value 102
        chu_all[chu_all == 4] = 101 #chu2 
        chu_all[chu_all == 5] = 102 #chu12
        chu_all[chu_all == 9] = 103 #chu3
        chu_all[chu_all == 10] = 104 #chu13
        chu_all[chu_all == 13] = 105 #chu23
        chu_all[chu_all == 14] = 106 #chu123

        chu_time = chu_time[chu_all > 0] # if there is still no chu assignment for that time then remove
        chu_all = chu_all[chu_all > 0]

        self.chu_all = chu_all

        self.chu_reference = {'chu1':100, 'chu2':101, 'chu12':102, 'chu3':103, 'chu13':104, 'chu23':105, 'chu123':106}

        tick_labels = ['','1', '2', '12', '3', '13', '23', '123'] 

        self.chu_times = [(self.rel_t + datetime.timedelta(seconds=t)) for t in chu_time]

        
        dt_times = self.chu_times
        fig = plt.figure(figsize=(10,5))
        ax = plt.axes()
        plt.plot(dt_times, chu_all,'x')
        plt.title('CHU States of NuSTAR on ' + dt_times[0].strftime('%Y/%m/%d')) #get the date in the title
        plt.xlabel('Start Time - ' + dt_times[0].strftime('%H:%M:%S'))
        plt.ylabel('NuSTAR CHUs')
        plt.xlim([data_handling.getTimeFromFormat(t) for t in self.time_range])
        fmt = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(self.xlocator)
        ax.axes.set_yticklabels(tick_labels)
        plt.xticks(rotation=30)
        if show_fig == True:
            plt.show()


    lc_3D_params = {'energy_low':1.6, 'energy_high':80, 'time_range':None} # start at 1.6 keV as this is the lowest (yet not trusted) bin for NuSTAR for binning in 0.04 keV steps

    def lightcurves_3D(self, all_evt_data=None, energy_increment=0.04, aspect=6):
        '''***Under Construction***'''

        if all_evt_data == None:
            all_evt_data = self.evt_data
        if self.lc_3D_params['time_range'] == None:
            self.lc_3D_params['time_range'] = self.time_range

        cleaned_all_evt = filter_with_tmrng.event_filter(all_evt_data, fpm = self.fpm, 
                                                     energy_low = self.lc_3D_params['energy_low'], 
                                                     energy_high = self.lc_3D_params['energy_high'], 
                                                     tmrng=self.lc_3D_params['time_range'])

        energies = np.arange(1.6 , self.lc_3D_params['energy_high'], energy_increment)
        no_of_time = 200
        times = np.arange(no_of_time, 1)

        er_and_tc = []
        for e in range(len(energies)-1):
            specific_lc_inds = filter_with_tmrng.by_energy(cleaned_all_evt, energies[e], energies[e+1])
            specific_lc_data = cleaned_all_evt[specific_lc_inds]
            counts = np.histogram(specific_lc_data['TIME'], no_of_time)[0]
            er_and_tc.append(counts)
        er_and_tc = np.array(er_and_tc)
        print(np.max(er_and_tc))
        fig = plt.figure(figsize=(6,8))
        plt.imshow(er_and_tc, origin='lower', aspect=aspect, vmax=1)
        plt.ylim([self.lc_3D_params['energy_low'], self.lc_3D_params['energy_high']])
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.show()

        ## event list for each energy bin (get energy filter function)
        ## get lightcurve for each energy bin
        ## Get 2D array for counts for each energy along rows, and time steps along the columns
        ## 1D array for the energies, 1D array for time steps
        ## get seperate, static method for 3D plot creation, return axis object
        ## axis limits to 2.5--80 keV (range of NuSTAR that's well calibrated)


    def detectors(self, show_fig=True):
        self.all_detectors = {}
        plt.figure()
        ax = plt.axes()
        for d in range(4):
            # if the detector is the one I want then I want the time of it, else leave it alone
            self.all_detectors['det'+str(d)] = [self.cleanevt['TIME'][c] for c,i in enumerate(self.cleanevt['DET_ID']) if i==d]
            # get percentage of counts each detector contributed to the full time
            self.all_detectors['det'+str(d)+'%'] = len(self.all_detectors['det'+str(d)]) / len(self.cleanevt['TIME']) * 100

            dets = np.histogram(self.all_detectors['det'+str(d)], self.t_bin_edges) #gives out bin values and bin edges
            dt_times = [(self.rel_t + timedelta(seconds=t)) for t in dets[1]]
            
            plt.plot(*self.stepped_lc_from_hist(self.dt_to_md(dt_times), dets[0]), label='det'+str(d)+': '+'{:.1f}'.format(self.all_detectors['det'+str(d)+'%'])+'%')
        plt.legend()

        fmt = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(self.xlocator)
        plt.xticks(rotation=30)

        plt.title('Detector Contribution '+self.e_range_str+" keV")
        plt.ylabel('Counts from detector')
        plt.xlabel('Time')

        if show_fig:
            plt.show()

        #return plt.g


    def plotChuTimes(self, span=True, axis=None):
        # remember to show_fig=False for the plotting methods as to allow alterations of the figures once run
        # look for and get the start and end times for each CHU file
        chus = ['chu1', 'chu2', 'chu12', 'chu3', 'chu13', 'chu23', 'chu123']
        colours = ['k', 'r', 'g', 'c', 'm', 'b', 'y']
        chuChanges = {}
        axis = {'ax':plt} if axis is None else {'ax':axis}
        pipeline_modes = ["_S_", "_N_"]

        for c, chu in enumerate(chus):
            for pm in pipeline_modes:
                chuFile = self.evt_directory+'nu' + self.obs_id + self.fpm + '06_' + chu + pm + 'cl_sunpos.evt'
                if isfile(chuFile):
                    break
            if not isfile(chuFile):
                continue
            hdulist = fits.open(chuFile) 
            evt_data = hdulist[1].data
            hdulist.close()

            chuChanges[chu] = [self.rel_t + timedelta(seconds=min(evt_data['time'])), 
                               self.rel_t + timedelta(seconds=max(evt_data['time']))]

            # plot a shaded region or just the time boundaries for the chu changes
            if span:
                axis['ax'].axvspan(*chuChanges[chu], alpha=0.1, color=colours[c])
            else:
                axis['ax'].axvline(chuChanges[chu][0], color=colours[c])
                axis['ax'].axvline(chuChanges[chu][1], color=colours[c])

        self.chuChanges = chuChanges

     
    def save(self, save_dir='./', folder_name=None, overwrite=False, **kwargs):
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

        New stuff to save:
            ***** chu function ***** deconvolve settings *****
        '''
        
        if self.chu_state != 'not_split' and folder_name is None:
            nustar_folder = save_dir + self.obs_id + self.fpm + '_' + self.chu_state + '_nustar_folder'
        elif folder_name is not None:
            nustar_folder = folder_name
        else:
            nustar_folder = save_dir + self.obs_id + self.fpm + '_nustar_folder'

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
            print("Directory " , nustar_folder , " already exists. Creating another.", end='')

        self.nustar_folder = nustar_folder
            
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
        lc_list_to_save = ['lc_times', 'lc_counts', 'lc_count_rates', 'sub_reg_lc', 'lc_livetimes']
        lc_info = list(set(dir(self)) & set(lc_list_to_save))
        lc_to_store = {}
        for name in lc_info:
            lc_to_store[name] = self.__dict__[name]
        with open(lc_folder + 'light_curve_data.pickle', 'wb') as lc_save_file:
            pickle.dump(lc_to_store, lc_save_file, protocol=pickle.HIGHEST_PROTOCOL)
            
        # Can save your own stuff
        if len(kwargs) > 0:
            own_folder = nustar_folder
            with open(own_folder + 'kwargs_data.pickle', 'wb') as own_save_file:
                pickle.dump(kwargs, own_save_file, protocol=pickle.HIGHEST_PROTOCOL)

        # save the object that can be loaded back in
        with open(nustar_folder + nustar_folder[:-1].split('/')[-1] + '.pickle', 'wb') as object_file:
            pickle.dump(self.__dict__, object_file, protocol=pickle.HIGHEST_PROTOCOL)
        self.object_file = nustar_folder + nustar_folder[:-1].split('/')[-1] + '.pickle'
                
        print(' Now Populated.')


    def load(self, object_file=None):
        '''Takes the object's namespace from the save() method and loads it back in to all it's attributes.'''
        if not hasattr(self, 'object_file') and object_file is None:
            print('\'object_file\' attribute and input to this function are both \'None\', please provide one. \n Note: the input for this method takes priority.')
            return
        object_file = object_file if (object_file is not None) else self.object_file
        
        with open(object_file, "rb") as input_file:
            self.__dict__ = pickle.load(input_file)


def shift(evt_data, pix_xshift=None, pix_yshift=None):
    if pix_xshift != None:
        for X in evt_data:
            X['X'] = X['X'] + pix_xshift 
    if pix_yshift != None:
        for Y in evt_data:
            Y['Y'] = Y['Y'] + pix_yshift 
    return evt_data


def nustars_synth_count(temp_response_dataxy, plasma_temp, plasma_em, source_area, errors=None, Tresp_syserror=0, log_data=False):
    """Takes data for a channel's temperature response, plasma temperature and emission measure and area of source and 
    returns the expected DN/s per pixel.
    *** Check output and make sure your units  work ***
    
    Parameters
    ----------
    temp_response_dataxy : dict
            The x and y data for the temperature response of the channel of interest, e.g. {'x':[...], 'y':[...]}.
    
    plasma_temp : float
            Temperature of the response you want in MK.
            
    plasma_em : float
            Volumetric emission measure of the plasma in cm^-3. 
            (If you have column emission measure, i.e. cm^-5, then set source_area=1.)
    
    source_area : float
            Area of the source in cm^2.
            
    errors : dict
            A dictionary of dictionaries containing the errors on T and EM, e.g. {'T':{'+':a, '-':b}, 
            'EM':{'+':c, '-':d}}.
            Defualt: None

    Tresp_syserror : float
            Fractional systematic error on the temperature response, e.g. 20% error on temp_response_dataxy['y'] means Tresp_error=0.2
            Default: 0
            
    log_data : bool
            Do you want the data (x and y) logged (base 10) for the interpolation?
            Default: False
            
    Returns
    -------
    A dictionary of floats that is the synthetic DN/s per pixel for the data given, temperature response, 
    temperature, and emission measure with units and errors.
    """
    # find temperature response at the given plasma temperature in DN cm^5 pix^-1 s^-1
    if log_data:
        f = interpolate.interp1d(np.log10(temp_response_dataxy['x']), np.log10(temp_response_dataxy['y']))
        temp_response = [10**f(np.log10(plasma_temp))]
    else:
        f = interpolate.interp1d(temp_response_dataxy['x'], temp_response_dataxy['y'])
        temp_response = [f(plasma_temp)]
        
    syn_flux = [tr * plasma_em * (1 / source_area) for tr in temp_response]
    
    # For errors
    if errors is not None:
        min_T, max_T = plasma_temp - errors['T']['-'], plasma_temp + errors['T']['+']
        min_EM, max_EM = plasma_em - errors['EM']['-'], plasma_em + errors['EM']['+']
        
        e_response = []
        for Ts in [min_T, max_T]:
            # find temperature response at the given plasma temperature in DN cm^5 pix^-1 s^-1
            r = [f(Ts)]
            e_response.append(r[0])
        
        temp_max_response = temp_response_dataxy['x'][np.argmax(temp_response_dataxy['y'])]
        
        # what if there is a bump between central value and error range
        if (e_response[0] < temp_response[0]) and (e_response[1] < temp_response[0]):
            if min_T < temp_max_response < plasma_temp:
                e_response[0] = np.max(temp_response_dataxy['y'])
            elif plasma_temp < temp_max_response < max_T:
                e_response[1] = np.max(temp_response_dataxy['y'])
        
        min_R, max_R = e_response[0], e_response[1] #R from min_T and R from max_T

        # include temperature response error
        up_resp = 1 + Tresp_syserror
        down_resp = 1 - Tresp_syserror
        
        #flux from min_T(max_EM) and flux from max_T(min_EM)
        min_flux, max_flux = min_R * max_EM * (1 / source_area), max_R * min_EM * (1 / source_area)
        flux_range = [min_flux, max_flux]
        
        e_response = np.array(e_response)[np.isfinite(e_response)]
        flux_range = np.array(flux_range)[np.isfinite(flux_range)]
        # max flux could be up_resp more, and min flux could be be down_resp more
        f_err = [up_resp*np.max(flux_range) - syn_flux[0], syn_flux[0] - down_resp*np.min(flux_range)]
        for n,f in enumerate(f_err):
            if f < 0:
                f_err[n] = np.max(f_err)
        
        errors = {'syn_flux_err':{'+': f_err[0], '-':f_err[1]}, 
                  't_res_err':{'+': abs(up_resp*np.max(e_response) - temp_response[0]), '-':abs(temp_response[0] - down_resp*np.min(e_response))}, 
                  't_res_syserr':[Tresp_syserror*100, '%'], 
                  'T_err':{'+': errors['T']['+'], '-':errors['T']['-']}, 
                  'EM_err':{'+': errors['EM']['+'],' -':errors['EM']['-']}}

    return {'syn_flux':[syn_flux[0],'DN pix^-1 s^-1'], 't_res':[temp_response, 'DN cm^5 pix^-1 s^-1'], 'T':[plasma_temp, 'MK'], 'EM':[plasma_em, 'cm^-3'], 'errors':errors}


def timefilter_evt(file, time_range=None, save_dir=None):
    """Takes a .evt file and filters the events list to a given time range. Only for region selection, do not use directly with spectral fitting software.
    
    Parameters
    ----------
    file : Str
            File (or directory/file) of the .evt file to be filtered by time.
    
    time_range : list
            A list of length 2 with the start and end date and time. Must be given in a specific format, e.g. time_range=['2018/09/10, 16:22:30', '2018/09/10, 16:24:30'].
            Default: None
            
    save_dir : Str
            String of the directory for the filtered file to be saved.
            Default: None
            
    Returns
    -------
    Creates a new file file with '_tf' before the file extension (meaning time filtered) and returns the name of the new file.
    """
    
    if time_range == None:
        print('No time_range given. Nothing will be done.')
        return
    
    file_regex = re.compile(r'.\w+') # form to split up filename string
    ext = file_regex.findall(file) # splits up file into all components, directories, filename, extension
    if save_dir == None:
        new_file_name = ''.join(ext[:-1]) + '_tf' + ext[-1] # '_tf' for time filtered
    else:
        new_file_name = save_dir + ext[-2] + '_tf' + ext[-1]
    
    hdulist = fits.open(file)
    evtdata=hdulist[1].data # data to be filtered

    evt_in_time = NustarDo().time_filter(evtdata, tmrng=time_range) # picks events inside time range

    hdulist[1].data = evt_in_time # replaces this hdu with the filtered events list
    hdulist.writeto(new_file_name, overwrite=True) # saves the edited file, original stays as is
    hdulist.close()
    
    return new_file_name



def CheckGrade0ToAllGrades(evtFile, wholeRangeToo=False, saveFig=None, timeRange=None, printOut=False, shortTitle=""):
    """Takes a NuSTAR evt file and compares the grade 0 events to the events of all grades.
       Adapted from: https://github.com/ianan/ns_proc_test/blob/main/test_proc_jun20_002.ipynb
    
    Parameters
    ----------
    evtFile : str
            The .evt file.
    
    wholeRangeToo : Bool
            If you want to plot the whole energy range in a second plot, next to the one ranging from 
            1.6--10 keV, set thi to True.
            Default: False
            
    saveFig : str
            If you want to save the figure made as a PDF then set this to a string of the save name.
            Defualt: None

    timeRange : list, 2 strings
            If you only want a certain time range of the total file's spectrum to be plotted, e.g. 
            ["%Y/%m/%d, %H:%M:%S", "%Y/%m/%d, %H:%M:%S"].
            Defualt: None

    printOut : Bool
            If you want to print out the output nicely(-ish) set this to True.
            Default: False

    shortTitle : Str
            Add a quick title to help keep track of the plots
            Default: ""
            
    Returns
    -------
    Dictionary containing the file name used ["file"], the time range of the file ["fileTimeRange"], 
    time range you asked it to plot ["timeRangeGivenToPlot"], effective exposure of full file ["eff_exp"], 
    ontime of full file ["ontime"], and percentage livetime ["lvtime_percent"] of full file given.
    """
    
    # read in .pha files for grade 0 and all grades
    hdulist = fits.open(evtFile)
    evt_data = hdulist[1].data
    evt_header = hdulist[1].header
    hdulist.close()
    
    # what is the time range of the file before filtering with time if you want
    ## nustar times are measured in seconds from this date
    rel_t = data_handling.getTimeFromFormat("2010/01/01, 00:00:00") 
    file_start = str((rel_t + timedelta(seconds=np.min(evt_data["time"]))).strftime('%Y/%m/%d, %H:%M:%S'))
    file_end   = str((rel_t + timedelta(seconds=np.max(evt_data["time"]))).strftime('%Y/%m/%d, %H:%M:%S'))
    
    # filter evt file by time?
    if type(timeRange) == list:
        if len(timeRange) == 2:
            evt_data = NustarDo().time_filter(evt_data, tmrng=timeRange)
    
    # get the data
    hist_gradeAll, be_gradeAll = np.histogram(evt_data['pi']*0.04+1.6,bins=np.arange(1.6,79,0.04))
    # work out the grade 0 spectra as well
    data_grade0 = evt_data['pi'][evt_data['grade']==0]
    hist_grade0, be_grade0 = np.histogram(data_grade0*0.04+1.6,bins=np.arange(1.6,79,0.04))
    
    # plotting info
    width   = 11 if wholeRangeToo else 5
    columns = 2  if wholeRangeToo else 1
    y_lims_spec  = [1e-1, 1.1*np.max(hist_gradeAll)]
    
    ratio = hist_gradeAll/hist_grade0
    fintie_vals = np.isfinite(ratio)
    y_lims_ratio = [0.95, 1.05*np.max(ratio[fintie_vals])] if wholeRangeToo else [0.95, 1.05*np.max(ratio[fintie_vals][:int((10-1.6)/0.04)])]

    axes_made = []

    plt.figure(figsize=(width,7))
    
    # define subplots for close look
    ax1 = plt.subplot2grid((4, columns), (0, 0), colspan=1, rowspan=3)
    axes_made.append(ax1)
    ax2 = plt.subplot2grid((4, columns), (3, 0), colspan=1, rowspan=1)
    axes_made.append(ax2)
    plt.tight_layout()

    # axis 1: the plots for all grades and grade 0
    ax1.plot(be_gradeAll[:-1], hist_gradeAll, drawstyle="steps-pre", label="Grade All")
    ax1.plot(be_grade0[:-1],   hist_grade0,   drawstyle="steps-pre", label="Grade 0")

    ax1.set_yscale("log")
    ax1.set_ylim(y_lims_spec)
    ax1.set_ylabel("Counts")# s$^{-1}$ keV$^{-1}$")
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_xlim([1.6,10])

    ax1.set_title("Grade 0 vs All Grades - "+shortTitle)
    ax1.legend()

    # axis 2: the difference between all grades and grade 0
    ax2.plot(be_grade0[:-1],   ratio,   drawstyle="steps-pre", color='k')

    ax2.set_ylabel("All Grades / Grade0")
    ax2.set_ylim(y_lims_ratio)
    ax2.set_xlim([1.6,10])
    ax2.set_xlabel("Energy [keV]")
    ax2.grid(axis='y')


    # define subplots for whole energy range
    if wholeRangeToo:
        # define subplots for close look
        ax3 = plt.subplot2grid((4, 2), (0, 1), colspan=1, rowspan=3)
        axes_made.append(ax3)
        ax4 = plt.subplot2grid((4, 2), (3, 1), colspan=1, rowspan=1)
        axes_made.append(ax4)
        plt.tight_layout()

        # axis 1: the plots for all grades and grade 0
        ax3.plot(be_gradeAll[:-1], hist_gradeAll, drawstyle="steps-pre", label="Grade All")
        ax3.plot(be_grade0[:-1],   hist_grade0,   drawstyle="steps-pre", label="Grade 0")

        ax3.set_yscale("log")
        ax3.set_ylim(y_lims_spec)
        ax3.set_xscale("log")
        plt.setp(ax3.get_xticklabels(), visible=False)
        ax3.set_xlim([1.6,79])

        ax3.set_title("Same But Whole E-range")
        ax3.legend()

        # axis 2: the difference between all grades and grade 0
        ax4.plot(be_grade0[:-1],   ratio,   drawstyle="steps-pre", color='k')
        
        ax4.set_ylim(y_lims_ratio)
        ax4.set_xscale("log")
        ax4.set_xlim([1.6,79])
        ax4.set_xlabel("Energy [keV]")
        ax4.grid(axis='y')
    
    if type(saveFig) == str:
        plt.savefig(saveFig, bbox_inches="tight")

    # plt.show()
    
    inform = {"file":evtFile,
              "fileTimeRange":[file_start, file_end], 
              "timeRangeGivenToPlot":timeRange,
              "eff_exp":evt_header['livetime'], 
              "ontime":evt_header['ontime'], 
              "lvtime_percent":100*evt_header['livetime']/evt_header['ontime']}
    if printOut:
        for key in inform.keys():
            print(key, " : ", inform[key])
            
    return inform, axes_made



def CheckGrade0ToAnyGrades(evtFile, grades, wholeRangeToo=False, saveFig=None, timeRange=None, printOut=False, shortTitle="", xlims=None):
    """Takes a NuSTAR evt file and compares the grade 0 events to the events of all grades.
       Adapted from: https://github.com/ianan/ns_proc_test/blob/main/test_proc_jun20_002.ipynb
    
    Parameters
    ----------
    evtFile : str
            The .evt file.

    grades : list of length 1 or 2 list
            A list of the lists of grades you want the grade 0 counts to be compared against. E.g. grades=[[1], [0,4]]
            means that grade zero will be checked against grade 1 counts and grade 0-4 counts inclusive.
    
    wholeRangeToo : Bool
            If you want to plot the whole energy range in a second plot, next to the one ranging from 
            1.6--10 keV, set thi to True.
            Default: False
            
    saveFig : str
            If you want to save the figure made as a PDF then set this to a string of the save name.
            Defualt: None

    timeRange : list, 2 strings
            If you only want a certain time range of the total file's spectrum to be plotted, e.g. 
            ["%Y/%m/%d, %H:%M:%S", "%Y/%m/%d, %H:%M:%S"].
            Defualt: None

    printOut : Bool
            If you want to print out the output nicely(-ish) set this to True.
            Default: False

    shortTitle : Str
            Add a quick title to help keep track of the plots
            Default: ""
            
    Returns
    -------
    Dictionary containing the file name used ["file"], the time range of the file ["fileTimeRange"], 
    time range you asked it to plot ["timeRangeGivenToPlot"], effective exposure of full file ["eff_exp"], 
    ontime of full file ["ontime"], percentage livetime ["lvtime_percent"] of full file given, Grade 0 
    plotting info, and you custom grade info too.
    """
    
    # read in .pha files for grade 0 and all grades
    hdulist = fits.open(evtFile)
    evt_data = hdulist[1].data
    evt_header = hdulist[1].header
    hdulist.close()
    
    # what is the time range of the file before filtering with time if you want
    ## nustar times are measured in seconds from this date
    rel_t = data_handling.getTimeFromFormat("2010/01/01, 00:00:00") 
    file_start = str((rel_t + timedelta(seconds=np.min(evt_data["time"]))).strftime('%Y/%m/%d, %H:%M:%S'))
    file_end   = str((rel_t + timedelta(seconds=np.max(evt_data["time"]))).strftime('%Y/%m/%d, %H:%M:%S'))
    
    # filter evt file by time?
    if type(timeRange) == list:
        if len(timeRange) == 2:
            evt_data = NustarDo().time_filter(evt_data, tmrng=timeRange)

    # work out the grade 0 spectra as well
    data_grade0 = evt_data['pi'][evt_data['grade']==0]
    hist_grade0, be_grade0 = np.histogram(data_grade0*0.04+1.6,bins=np.arange(1.6,79,0.04))
    
    other_grades = {}
    ratios = []
    max_ratios, min_ratios = [], []
    # get the data
    for g in grades:
        if len(g)==1:
            data_grade = evt_data['pi'][evt_data['grade']==g[0]]
            g_str = "Grade "+str(g[0])
            other_grades[g_str] = np.histogram(data_grade*0.04+1.6,bins=np.arange(1.6,79,0.04))
        else:
            data_grade = evt_data['pi'][(evt_data['grade']>=g[0]) & (evt_data['grade']<=g[1])]
            g_str = "Grade "+str(g[0])+"-"+str(g[1])
            other_grades[g_str] = np.histogram(data_grade*0.04+1.6,bins=np.arange(1.6,79,0.04))
        ratio = other_grades[g_str][0]/hist_grade0
        ratios.append(ratio)
        maximum = np.max(ratio[np.isfinite(ratio)]) if wholeRangeToo else np.max(ratio[np.isfinite(ratio)][:int((10-1.6)/0.04)])
        minimum = np.min(ratio[np.isfinite(ratio)]) if wholeRangeToo else np.min(ratio[np.isfinite(ratio)][:int((10-1.6)/0.04)])
        max_ratios.append(maximum)
        min_ratios.append(minimum)
    
    # plotting info
    width   = 11 if wholeRangeToo else 5
    columns = 2  if wholeRangeToo else 1
    y_lims_spec  = [1e-1, 1.1*np.max(hist_grade0)]
    
    y_lims_ratio = [0.95*np.min(min_ratios), 1.05*np.max(max_ratios)]

    axes_made = []

    plt.figure(figsize=(width,7))
    
    # define subplots for close look
    ax1 = plt.subplot2grid((4, columns), (0, 0), colspan=1, rowspan=3)
    axes_made.append(ax1)
    ax2 = plt.subplot2grid((4, columns), (3, 0), colspan=1, rowspan=1)
    axes_made.append(ax2)
    plt.tight_layout()

    # axis 1: the plots for all grades and grade 0
    for key, r in zip(other_grades.keys(), ratios):
        ax1.plot(other_grades[key][1][:-1], other_grades[key][0], drawstyle="steps-pre", label=key)
        ax2.plot(other_grades[key][1][:-1],   r,   drawstyle="steps-pre")
    ax1.plot(be_grade0[:-1],   hist_grade0,   drawstyle="steps-pre", label="Grade 0")

    ax1.set_yscale("log")
    ax1.set_ylim(y_lims_spec)
    ax1.set_ylabel("Counts")# s$^{-1}$ keV$^{-1}$")
    plt.setp(ax1.get_xticklabels(), visible=False)

    xlims = xlims if type(xlims)!=type(None) else [1.6,10]
    ax1.set_xlim(xlims)

    ax1.set_title("Grade 0 vs Chosen Grades - "+shortTitle)
    ax1.legend()

    # axis 2: the difference between all grades and grade 0
    # ax2.plot(be_grade0[:-1],   ratio,   drawstyle="steps-pre", color='k')

    ax2.set_ylabel("Chosen Grades / Grade0")
    ax2.set_ylim(y_lims_ratio)
    ax2.set_xlim(xlims)
    ax2.set_xlabel("Energy [keV]")
    ax2.grid(axis='y')


    # define subplots for whole energy range
    if wholeRangeToo:
        # define subplots for close look
        ax3 = plt.subplot2grid((4, 2), (0, 1), colspan=1, rowspan=3)
        axes_made.append(ax3)
        ax4 = plt.subplot2grid((4, 2), (3, 1), colspan=1, rowspan=1)
        axes_made.append(ax4)
        plt.tight_layout()

        # axis 1: the plots for all grades and grade 0
        for key, r in zip(other_grades.keys(), ratios):
            ax3.plot(other_grades[key][1][:-1], other_grades[key][0], drawstyle="steps-pre", label=key)
            ax4.plot(other_grades[key][1][:-1],   r,   drawstyle="steps-pre")
        ax3.plot(be_grade0[:-1],   hist_grade0,   drawstyle="steps-pre", label="Grade 0")

        ax3.set_yscale("log")
        ax3.set_ylim(y_lims_spec)
        ax3.set_xscale("log")
        plt.setp(ax3.get_xticklabels(), visible=False)
        ax3.set_xlim([1.6,79])

        ax3.set_title("Same But Whole E-range")
        ax3.legend()

        # axis 2: the difference between all grades and grade 0
        # ax4.plot(be_grade0[:-1],   ratio,   drawstyle="steps-pre", color='k')
        
        ax4.set_ylim(y_lims_ratio)
        ax4.set_xscale("log")
        ax4.set_xlim([1.6,79])
        ax4.set_xlabel("Energy [keV]")
        ax4.grid(axis='y')
    
    if type(saveFig) == str:
        plt.savefig(saveFig, bbox_inches="tight")

    # plt.show()
    
    inform = {"file":evtFile,
              "fileTimeRange":[file_start, file_end], 
              "timeRangeGivenToPlot":timeRange,
              "eff_exp":evt_header['livetime'], 
              "ontime":evt_header['ontime'], 
              "lvtime_percent":100*evt_header['livetime']/evt_header['ontime'],
              "Grade 0":[hist_grade0, be_grade0],
              **other_grades}
    if printOut:
        for key in inform.keys():
            print(key, " : ", inform[key])
            
    return inform, axes_made



## functions to help find the FoV rotation
def collectSameXs(rawx, rawy, solx, soly):
    """ Returns a dictionary where each column is given a unique entry with a list 
    of the rows that correspond to that one column from the evt file. Also saves the 
    solar coordinates for that raw coordinate column with the rawx column key+"map2sol".
    
    Parameters
    ----------
    rawx, rawy : lists
        Raw coordinates of the evt counts.
        
    solx, soly : lists
        Solar coordinates of the sunpos evt counts.
        
    Returns
    -------
    A dictionary.
    
    Examples
    --------
    rawx, rawy = [1,2,3,3], [7,8,4,9]
    solx, soly = [101, 102, 103, 104], [250, 252, 254, 256]

    collectSameXs(rawx, rawy, solx, soly)
    >>> {"1":[7], "1map2sol":[101, 250], 
         "2":[8], "2map2sol":[102, 252], 
         "3":[4, 9], "3map2sol":[[103, 254], [104, 256]]}
    """
    output = {}
    for c,xs in enumerate(rawx):
        if str(xs) not in output:
            output[str(xs)] = [rawy[c]]
            output[str(xs)+"map2sol"] = [[solx[c], soly[c]]]
        else:
            output[str(xs)].append(rawy[c])
            output[str(xs)+"map2sol"].append([solx[c], soly[c]])
        assert len([solx[c], soly[c]])==2
    return output

def minRowInCol(columns):
    """ Returns a dictionary where each key is the solar X position of each raw 
    coordinate chosen (edges between det0&3 and 1&2) with its value being the 
    solar Y coordinate.
    
    Parameters
    ----------
    columns : dictionary
        Information of the raw and solar coordinates of the counts in order to 
        each other.
        
        
    Returns
    -------
    A dictionary.
    
    Examples
    --------
    cols = {"1":[7], "1map2sol":[101, 250], 
            "2":[8], "2map2sol":[102, 252], 
            "3":[4, 9], "3map2sol":[[103, 254], [104, 256]]}

    minRowInCol(cols)
    >>> {"101":250, "102":252, "103":254}
    """
    output_sol = {}
    for key in columns.keys():
        if "map2sol" not in key:
            # find the corresponding solar coords to the minimum rawy
            sol_coords = columns[key+"map2sol"][np.argmin(columns[key])]
            # now have the solarX key with the solarY as its value
            assert len(sol_coords)==2
            output_sol[str(sol_coords[0])] = sol_coords[1]
    return output_sol

def maxRowInCol(columns):
    """ Returns a dictionary where each key is the solar X position of each raw 
    coordinate chosen (edges between det0&3 and 1&2) with its value being the 
    solar Y coordinate.
    
    Parameters
    ----------
    columns : dictionary
        Information of the raw and solar coordinates of the counts in order to 
        each other.
        
        
    Returns
    -------
    A dictionary.
    
    Examples
    --------
    cols = {"1":[7], "1map2sol":[101, 250], 
            "2":[8], "2map2sol":[102, 252], 
            "3":[4, 9], "3map2sol":[[103, 254], [104, 256]]}

    minRowInCol(cols)
    >>> {"101":250, "102":252, "104":256}
    """
    output_sol = {}
    for key in columns.keys():
        if "map2sol" not in key:
            # find the corresponding solar coords to the maximum rawy
            sol_coords = columns[key+"map2sol"][np.argmax(columns[key])]
            # now have the solarX key with the solarY as its value
            output_sol[str(sol_coords[0])] = sol_coords[1]
    return output_sol

def getXandY(colsAndRows):
    """ Returns solar X and Y coordinates.
    
    Parameters
    ----------
    colsAndRows : dictionary
        Keys as the solar X and values of solar Y coordinates.
        
    Returns
    -------
    Two numpy arrays.
    
    Examples
    --------
    colsAndRows = {"101":250, "102":252, "104":256}

    getXandY(colsAndRows)
    >>> [101, 102, 104], [250, 252, 256]
    """
    return np.array([int(c) for c in list(colsAndRows.keys())]), np.array(list(colsAndRows.values()))

def getDegrees(grad):
    """ Returns angle of rotation in degrees.
    
    Parameters
    ----------
    grad : float
        Gradient.
        
    Returns
    -------
    Angle in degrees.
    
    Examples
    --------
    grad = 1

    getDegrees(grad)
    >>> 45
    """
    return np.arctan(grad)*(180/np.pi)

def straightLine(x, m, c):
    """ A straight line model.
    
    Parameters
    ----------
    x : numpy list
        X positions.
    
    m : float
        Gradient.
    
    c : float
        Y-intercept.
        
    Returns
    -------
    Ys for a straight line.
    
    Examples
    --------
    x, m, c = [1, 2], 0.25, 1

    straightLine(x, m, c)
    >>> [1.25, 1.5]
    """
    return m*x + c

def getAngle_plot(rawx, rawy, solx, soly, det, **kwargs):
    """ Returns the rotation of the NuSTAR FoV from the gradient of the edges between 
    det0&3 and 1&2 for whatever detector(s) you give it.
    
    Parameters
    ----------
    rawx, rawy : lists
        Raw coordinates of the evt counts.
        
    solx, soly : lists
        Solar coordinates of the sunpos evt counts.
    
    det : int
        The detector for the counts (0--3).
        
    **kwargs : Can pass an axis to it.
        
    Returns
    -------
    A float of the rotation from "North" in degrees where anticlockwise is positive.
    This assumes the rotation is between 90 and -90 degrees.
    
    Examples
    --------
    fig, axs = plt.subplots(2,2, figsize=(14,10))

    # get orientation from the nustar_swguide.pdf, Figure 3

    gradient0 = getAngle_plot(rawx0, rawy0, solx0, soly0, 0, axes=axs[0][0])
    gradient1 = getAngle_plot(rawx1, rawy1, solx1, soly1, 1, axes=axs[0][1])
    gradient2 = getAngle_plot(rawx2, rawy2, solx2, soly2, 2, axes=axs[1][1])
    gradient3 = getAngle_plot(rawx3, rawy3, solx3, soly3, 3, axes=axs[1][0])

    plt.show()
    """
    
    k = {"axes":plt}
    for kw in kwargs:
        k[kw] = kwargs[kw]
        
    if det==0:
        cols = collectSameXs(rawy, rawx, solx, soly)
        m_row_per_col = maxRowInCol(cols)
    elif det==1:
        cols = collectSameXs(rawx, rawy, solx, soly)
        m_row_per_col = maxRowInCol(cols)
    elif det==2:
        cols = collectSameXs(rawy, rawx, solx, soly)
        m_row_per_col = maxRowInCol(cols)
    elif det==3:
        cols = collectSameXs(rawx, rawy, solx, soly)
        m_row_per_col = maxRowInCol(cols)

    # working with rawx and y to make sure using correct edge then find the 
    # corresponding entries in solar coords
    aAndY = getXandY(m_row_per_col)
    x, y = aAndY[0], aAndY[1]
    
    xlim, ylim = [np.min(x)-5, np.max(x)+5], [np.min(y)-5, np.max(y)+5]
    #if det in [0, 1]:
    #    x = x[y>np.median(y)]
    #    y = y[y>np.median(y)]
    #elif det in [2, 3]:
    #    x = x[y<np.median(y)]
    #    y = y[y<np.median(y)]
    
    popt, pcov = curve_fit(straightLine, x, y, p0=[0, np.mean(y)])
    
    k["axes"].plot(x, y, '.')
    k["axes"].plot(x, straightLine(x, *popt))
    
    if k["axes"] != plt:
        k["axes"].set_ylim(ylim)
        k["axes"].set_xlim(xlim)
        k["axes"].set_ylabel("Solar-Y")
        k["axes"].set_xlabel("Solar-X")
    else:
        k["axes"].ylim(ylim)
        k["axes"].xlim(xlim)
        k["axes"].ylabel("Solar-Y")
        k["axes"].xlabel("Solar-X")
    k["axes"].text(np.min(x), (ylim[0]+ylim[1])/2+5, "Grad: "+str(popt[0]))
    k["axes"].text(np.min(x), (ylim[0]+ylim[1])/2, "Angle: "+str(np.arctan(popt[0]))+" rad")
    k["axes"].text(np.min(x), (ylim[0]+ylim[1])/2-5, "Angle: "+str(np.arctan(popt[0])*(180/np.pi))+" deg")
    k["axes"].text(np.max(x)*0.99, ylim[0]*1.001, "DET: "+str(det), fontweight="bold")
    
    return np.arctan(popt[0])*(180/np.pi)