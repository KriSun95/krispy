'''
Functions to do with helping to creat contour plots:
* Can now set up multiple contour plots seperately then combine them. 
'''

import sys
sys.path.insert(0, '../')
import krispy
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import sunpy
from astropy.coordinates import SkyCoord
import astropy.units as u
from skimage.transform import resize
from scipy import signal
from astropy.io import fits
import matplotlib.colors as mc
from copy import deepcopy
import re #for regular expressions
import time


class Contours:
    '''Returns a figure with a background and NuSTAR contours
    
    Attributes
    ----------
    nu_file_directory : str
        A string of the path of the file given from the nu_file input.
        
    nu_filename : str
        A string of the filename given from the nu_file input.
    
    aia_directory :str
        A string of the AIA directory used to find the background.
        
    colour_dict : dict
        Dictionary that holds the information on the colour, energies, contour type, and levels of the contours 
        wanted, e.g. colour_dict={'green':{'energy':[2,4], 'format':'percent', 'lvls':[20,60,90]}, 
                                  'red':{'energy':[4,6], 'format':'real_values', 'lvls':[1,10,150]}}.
                                  
    time_range : indexable list, length 2
        A list of two times, start and end time, for the image produced at the end, e.g. 
        times = ['2018/09/10, 16:18:30', '2018/09/10, 16:20:30'].
        
    submap : indexable list, length 4
        List wchich has the boundaries of a submap in arcseconds as [leftX, bottomY, rightX, topY].
        
    Notes
    -----
    self.nu_final_maps =  nu_final
        self.nu_final_objects = nu_objs
        self.background_frame = background_frame
        iterations
        self.nu_shift = nu_shift
    '''
    
    def __init__(self, nu_file=None, aia_dir=None, colour_dict=None, time_range=None, submap=None):
        ''' Take in the inputs to the class and set them to class attributes. Seperate them for use in 
        later functions'''
        
        # so the class functions can potentially be used without specifying a function 
        if type(nu_file) == type(None):
            return
        
        self.file_given = nu_file
        #directory of the file
        directory_regex = re.compile(r'\w+/')
        directory = directory_regex.findall(nu_file)
        self.nu_file_directory = ''.join(directory)

        #search of the form of stuff (no slashes included), dot, then more stuff
        nu_filename_regex = re.compile(r'\w+\.\w+') 
        name_of_file = nu_filename_regex.findall(nu_file)[0]
        self.nu_filename = name_of_file
        
        self.aia_directory = aia_dir
        self.colour_dict = colour_dict
        self.time_range = time_range
        self.submap = submap
        self.nuFoV = None # this will get replaced with the NuSTAR FoV if we need it, otherwise an extended submap
        
        # now seperate colour_dict up for later function use into colour and energy, and colour and contour info
        self.colour_and_energy = {}
        self.colour_and_contours = {}
        for key in colour_dict:
            self.colour_and_energy[key] = colour_dict[key]['energy']
            self.colour_and_contours[key] = {'format':colour_dict[key]['format'], 'lvls':colour_dict[key]['lvls']}
        
    
    def multi_energy_nustar_frame(self, nu_file=None, colour_and_energy=None, time_interval=None, submap=None):
        
        if nu_file == None:
            nu_file=self.file_given
        if colour_and_energy == None:
            colour_and_energy=self.colour_and_energy
        if time_interval == None:
            time_interval=self.time_range
        if submap == None:
            submap=self.submap

        nustar_obj = {}
        nustar_maps_corr = {}

        for key in colour_and_energy:
            # get array for plotting
            nu1 = krispy.nustardo.NustarDo(nu_file, energy_range=[colour_and_energy[key][0], colour_and_energy[key][1]], time_range=time_interval)
            nu1.deconvolve['apply'] = False # just to be sure
            nu1.sub_lt_zero = 0
            nustar_obj[key] = nu1.nustar_setmap(submap = [submap[0]-0.01, submap[1]-0.01, submap[2]+0.01, submap[3]+0.01]) # to avoid white border
            del nu1

            # get array for cross-correlation
            nu2 = krispy.nustardo.NustarDo(nu_file, energy_range=[colour_and_energy[key][0], colour_and_energy[key][1]], time_range=time_interval)
            nu2.deconvolve['apply'] = False
            nu2.sub_lt_zero = 0
            nustar_maps_corr[key] = nu2.nustar_setmap(submap = [submap[0]-100, submap[1]-100, submap[2]+100, submap[3]+100])
            del nu2

        return nustar_obj, nustar_maps_corr

    
    iterations = 10 # set to change the number of iterations
    
    def nu_deconv(self, nu_file=None, colour_and_energy=None, 
                       time_interval=None, submap=None, iterations=None):
        
        if nu_file == None:
            nu_file=self.file_given
        if colour_and_energy == None:
            colour_and_energy=self.colour_and_energy
        if time_interval == None:
            time_interval=self.time_range
        if submap == None:
            submap=self.submap
        if iterations == None:
            iterations=self.iterations

        nustar_obj = {}
        nustar_maps_corr = {}

        for key in colour_and_energy:
            # get array for plotting 
            nu1 = krispy.nustardo.NustarDo(nu_file, energy_range=[colour_and_energy[key][0], colour_and_energy[key][1]], time_range=time_interval)
            nu1.deconvolve['apply'] = True
            nu1.deconvolve['iterations'] = iterations
            # make sure to deconvolve over FoV
            nu1.nustar_setmap(submap='FoV')
            self.nuFoV = nu1.FoV
            nu1.deconvolve['apply'] = False
            # use deconvolved map as own map, avoid time norm a second time, and only submap the region
            # if own_map isn't set then the map is created from scratch again and so isn't deconvolved
            nu1.own_map = nu1.rsn_map
            nustar_obj[key] = nu1.nustar_setmap(submap = [submap[0]-0.01, submap[1]-0.01, submap[2]+0.01, submap[3]+0.01], time_norm=False) # to avoid white border
            nu1.deconvolve['apply'] = False # to be extra safe
            del nu1

            # get array for cross-correlation
            nu2 = krispy.nustardo.NustarDo(nu_file, energy_range=[colour_and_energy[key][0], colour_and_energy[key][1]], time_range=time_interval)
            nu2.deconvolve['apply'] = True
            nu2.deconvolve['iterations'] = iterations
            # make sure to deconvolve over FoV
            nu2.nustar_setmap(submap='FoV')
            nu2.deconvolve['apply'] = False
            # use deconvolved map as own map, avoid time norm a second time, and only submap the region
            # if own_map isn't set then the map is created from scratch again and so isn't deconvolved
            nu2.own_map = nu2.rsn_map
            nustar_maps_corr[key] = nu2.nustar_setmap(submap = self.nuFoV, time_norm=False)
            # [submap[0]-100, submap[1]-100, submap[2]+100, submap[3]+100]
            nu2.deconvolve['apply'] = False
            del nu2

        return nustar_obj, nustar_maps_corr
    
    
    def aia_file_times(self, aia_file_dir=None):
        
        if aia_file_dir == None:
            aia_file_dir=self.aia_directory
        
        aia_file_list = np.array(os.listdir(aia_file_dir))
        times = []

        for file in aia_file_list:
            if file.endswith('.fits'):
                hdulist = fits.open(aia_file_dir + file)
                header = hdulist[0].header
                hdulist.close()
                time = datetime.datetime.strptime(header['date-obs'], '%Y-%m-%dT%H:%M:%S.%fZ')
                times.append(time)
            else:
                # just to provide an entry so indices can match up
                time = datetime.datetime.strptime('1979-01-01T00:00:00.000000Z', '%Y-%m-%dT%H:%M:%S.%fZ')
                times.append(time)

        return times
    
    
    def useful_time_inds(self, times_list=None, time_interval=None):
        
        if time_interval == None:
            time_interval=self.time_range
        
        if type(times_list) == type(None):
            # sys._getframe().f_code.co_name give the funciton name
            print('No times list has been given to: ', sys._getframe().f_code.co_name)
            return []
        
        ti_1 = datetime.datetime.strptime(time_interval[0], '%Y/%m/%d, %H:%M:%S')
        ti_2 = datetime.datetime.strptime(time_interval[1], '%Y/%m/%d, %H:%M:%S')

        dt = np.array(times_list)

        time_fil = ( (dt > ti_1) & (dt <= ti_2) )
        gd_inds = (time_fil).nonzero()[0]

        return gd_inds
    

    @staticmethod
    def which_background(aia_dir=None, aia_files=None, where='average'):
        
        if type(aia_dir) == type(None) or type(aia_files) == type(None):
            # sys._getframe().f_code.co_name give the funciton name
            print('Parameter missing (either directory or filenames) from: ', sys._getframe().f_code.co_name)
            return []

        bgs = []
        bgs_header = []
        for file in aia_files:
            hdulist = fits.open(aia_dir + file)
            bgs_header.append(hdulist[0].header)
            bgs.append(hdulist[0].data)
            hdulist.close()
     
        if where == 'start':
            bg = bgs[0]
            hdr = bgs_header[0]
        elif where == 'middle':
            i = (len(bgs)-1)//2 # middle index
            bg = bgs[i]
            hdr = bgs_header[i]
        elif where == 'end':
            bg = bgs[-1]
            hdr = bgs_header[-1]
        elif where == 'average':
            bg = np.sum(bgs, axis=0) / len(bgs)
            hdr = bgs_header[0]
        elif where == 'sum':
            bg = np.sum(bgs, axis=0)
            hdr = bgs_header[0]
        else:
            print('Need a valid \'where\' input (start, middle, end, average, sum): ', sys._getframe().f_code.co_name)
            return None
        return sunpy.map.Map(bg, hdr)
        
    
    def aia_frame(self, aia_map, submap=None, print_max=False):
        
        if submap == None:
            submap=self.submap
        
        aia_map_for_corr = deepcopy(aia_map)
    
        bl = SkyCoord((submap[0])*u.arcsec, (submap[1])*u.arcsec, frame=aia_map.coordinate_frame)
        tr = SkyCoord((submap[2])*u.arcsec, (submap[3])*u.arcsec, frame=aia_map.coordinate_frame)
        aia_data = aia_map.submap(bl,tr)


        if print_max == True:
            print('Max AIA map value is ', np.max(aia_data.data))

        if type(self.nuFoV) == type(None):
            self.nuFoV = [submap[0]-100, submap[1]-100, submap[2]+100, submap[3]+100]

        bl_corr = SkyCoord((self.nuFoV[0])*u.arcsec, (self.nuFoV[1])*u.arcsec, frame=aia_map.coordinate_frame)
        tr_corr = SkyCoord((self.nuFoV[2])*u.arcsec, (self.nuFoV[3])*u.arcsec, frame=aia_map.coordinate_frame)
        aia_corr_data = aia_map_for_corr.submap(bl_corr, tr_corr).data

        return aia_data, aia_corr_data
    

    def corr_fpm(self, nu_arrays, aia_array):

        # take first map the now
        for key in nu_arrays:
            nu_arr = nu_arrays[key].data
            nu_arr[np.isnan(nu_arr)] = 0
            break
        

        data_for_corr = resize(aia_array, np.shape(nu_arr))
        self.corr_data = signal.correlate2d(data_for_corr, nu_arr, boundary='symm', mode='same')
        y, x = np.unravel_index(np.argmax(self.corr_data), self.corr_data.shape)  # find the match

        self.corr_ar = data_for_corr, nu_arr

        # need the number of NuSTAR pixels is needed for the shift
        x_pix_shift = -(np.shape(data_for_corr)[1]/2 - x) #negative because the positive number means shift to the left/down
        y_pix_shift = -(np.shape(data_for_corr)[0]/2 - y)
        shift = np.array([x_pix_shift, y_pix_shift])

        return shift
    
    
    print_max_nu = False

    def apply_nu_shift_fpm(self, nu_file=None, nu_shift=None, colour_and_energy=None, 
                           time_interval=None, submap=None, iterations=None):
        
        if type(nu_shift) == type(None):
            # sys._getframe().f_code.co_name give the funciton name
            print('Need an nu_shift in pixels, [x,y], for this function: ', sys._getframe().f_code.co_name)
            return []
        
        if nu_file == None:
            nu_file=self.file_given
        if colour_and_energy == None:
            colour_and_energy=self.colour_and_energy
        if time_interval == None:
            time_interval=self.time_range
        if submap == None:
            submap=self.submap
        if iterations == None:
            iterations=self.iterations

        nustar_final = {}
        nu_objs = {}

        for key in colour_and_energy:
            # get array for plotting 
            nu1 = krispy.nustardo.NustarDo(nu_file, energy_range=[colour_and_energy[key][0], 
                                                                  colour_and_energy[key][1]], 
                                           time_range=time_interval)

            nu1.cleanevt = krispy.nustardo.shift(nu1.cleanevt, pix_xshift=nu_shift[0], pix_yshift=nu_shift[1])
            
            nu1.deconvolve['apply'] = False # to be extra safe
            tn = True
            if type(iterations) == int and iterations >= 1:
                nu1.deconvolve['apply'] = True
                nu1.deconvolve['iterations'] = iterations

                # make sure to deconvolve over FoV
                nu1.nustar_setmap(submap='FoV')
                nu1.deconvolve['apply'] = False
                # use deconvolved map as own map, avoid time norm a second time, and only submap the region
                # if own_map isn't set then the map is created from scratch again and so isn't deconvolved
                nu1.own_map = nu1.rsn_map
                tn = False

            nu1.sub_lt_zero = 0
            nustar_final[key] = nu1.nustar_setmap(submap = [submap[0]-0.01, submap[1]-0.01, submap[2]+0.01, submap[3]+0.01], time_norm=tn) # to avoid white border
            if self.print_max_nu == True:
                print('Max NuSTAR map value is ', np.max(nustar_final[key].data))
            nu1.deconvolve['apply'] = False # to be extra safe
            nu_objs[key] = nu1

        return nustar_final, nu_objs
    
    
    @staticmethod
    def unique_entries(meta_info):
        e_all = []
        fpm_all = []
        chu_all = []
        
        # get all factors seperated
        for key in meta_info:
            e_all.append(meta_info[key][0])
            fpm_all.append(meta_info[key][1])
            chu_all.append(meta_info[key][2])
        
        # remove duplicate entries in each list
        ## extra step for e as a list of lists is no hashable but a list of tuples is
        e_all = [tuple(row) for row in e_all]
        e_unique = list(set(e_all))
        fpm_unique = list(set(fpm_all))
        chu_unique = list(set(chu_all))
        
        # if, by removing duplicates, there is one entry then that is the constant for the figure, else return all
        e_return = e_unique if len(e_unique) == 1 else e_all
        fpm_return = fpm_unique if len(fpm_unique) == 1 else fpm_all
        chu_return = chu_unique if len(chu_unique) == 1 else chu_all

        return e_return, fpm_return, chu_return
    
       
    @staticmethod
    def complete_legend_title(legend_title, e_unique, fpm_unique, chu_unique):
    
        # structure of FPM?
        fpm_s = [' FPM'+l[0] for l in [e_unique, fpm_unique, chu_unique] if len(l)==1 and type(l[0]) == str and len(l[0]) == 1]
        # structure of energies?
        e_s = [' '+str(l[0][0])+'-'+str(l[0][1])+' keV' for l in [e_unique, fpm_unique, chu_unique] if len(l)==1 and type(l[0]) == tuple]
        # structure of chu state?
        chu_s = [' '+l[0] for l in [e_unique, fpm_unique, chu_unique] if len(l)==1 and type(l[0]) == str and len(l[0]) > 1] 

        # if there is only one element in the array, want it in the overall legend title
        for l in [fpm_s, e_s, chu_s]:
            if len(l) == 1:
                legend_title += l[0]

        return legend_title


    def create_contours(self, nusun_objects=None, nu_objects=None, aia_object=None, 
                        iron='', contours=None, submap=None, annotate=True, 
                        background_contours=False, bg_limits=None, plot=True, 
                        background_cmap=None, usr_title=None):

        if plot == False:
            plt.rcParams['figure.frameon'] = False
        
        if contours == None:
            contours=self.colour_and_contours
        if submap == None:
            submap=self.submap

        plt.rcParams['font.size'] = 14

        if iron == '18':
            aia_object.plot_settings['cmap'] = plt.cm.Blues
            map_title = 'SDO/AIA FeXVIII'
        else:
            map_title = 'SDO/AIA ' + str(aia_object.meta['WAVELNTH']) + ' \AA'

        if type(background_cmap) != type(None):
            aia_object.plot_settings['cmap'] = background_cmap

        compmap = sunpy.map.Map(aia_object, composite=True)
        tot_num_of_levels = 0 # to avoid non-assignment error later on
        
        # scale annotations to fit all contour info if > 4
        preconts = 1 if type(background_contours) == dict else 0
        if ( preconts + len(nusun_objects) ) > 4:
            scale = 4 / ( preconts + len(nusun_objects) )
        else:
            scale = 1
        
        # all annotation spacing
        size = np.shape(aia_object.data) #should the font, etc be dependent on the array size?
        # spacing probably should but charactersize should be constant...probably
        char_to_arcsec = 0.045*350 * scale #(350, 359) #
        border = 0.01*size[0]
        xspacing = 0.15*size[1] * scale - 2 # old spacing = 0.15*size[1] * scale
        x = submap[0] + border
        y = submap[1] + border 
        y_reset = deepcopy(y)
        yspacing = 0.025*size[0] * scale
        yspacing_reset = deepcopy(yspacing)

        maps_already_here = 1
        if type(background_contours) == dict:
            bgc = deepcopy(aia_object)
            key = list(background_contours.keys())[0]
            cmap = mc.LinearSegmentedColormap.from_list("", [key,key])
            bgc.plot_settings['cmap'] = cmap
            compmap.add_map(bgc)
            contour_lvls = background_contours[key]
            if contour_lvls['format'] == 'values':
                compmap.set_levels(maps_already_here,contour_lvls['lvls'],percent = False)
                string = ' DN s$^{-1}$'
            elif contour_lvls['format'] == 'percent':
                compmap.set_levels(maps_already_here,contour_lvls['lvls'],percent = True)
                string = ' %'
            else:
                print('Please choose percent or values for the way to display contour numbers.')
                return
            maps_already_here = 2
            if annotate == True:
                max_length = np.max([len(str(c))-len(str(float(int(c))))+1 for c in contour_lvls['lvls']])
                if max_length < 0:
                    max_length = 0
                tot_num_of_levels = len(contours[list(contours.keys())[0]]['lvls'])
                num_of_levels_of_bg = len(contour_lvls['lvls'])
                y_correction = 0.5*( (tot_num_of_levels - num_of_levels_of_bg) / (tot_num_of_levels))
                y = y + tot_num_of_levels*yspacing*y_correction
                for lvl in contour_lvls['lvls'][::-1]:
                    clabel = str(format(lvl, '.'+str(max_length)+'f')) + string
                    plt.annotate(clabel ,(x,y), ha='left', size=char_to_arcsec, color=key) #+0.6*xspacing
                    y += yspacing
                x += xspacing
        
        del aia_object
        meta_info = {}
        len_contour_lists = []
        for c, key in enumerate(nusun_objects):
            contour_lvls = contours[key]
            len_contour_lists.append(len(contour_lvls['lvls']))

        for c, key in enumerate(nusun_objects):

            cmap = mc.LinearSegmentedColormap.from_list("", [key,key])
            nusun_objects[key].plot_settings['cmap'] = cmap

            # get info for plotting
            e_range =  nu_objects[key].energy_range
            fpm = nu_objects[key].fpm
            time_range = nu_objects[key].time_range
            if nu_objects[key].chu_state == 'not_split':
                chu = ''
            else:
                chu = nu_objects[key].chu_state

            meta_info[key] = [e_range, fpm, chu]

            compmap.add_map(nusun_objects[key])

            contour_lvls = contours[key]
            if contour_lvls['format'] == 'values':
                compmap.set_levels(c+maps_already_here,contour_lvls['lvls'],percent = False)
                string = ' Cts s$^{-1}$'
            elif contour_lvls['format'] == 'percent':
                compmap.set_levels(c+maps_already_here,contour_lvls['lvls'],percent = True)
                string = ' %'
            else:
                print('Please choose percent or values for the way to display contour numbers.')
                return

            # from here on it is mainly annotation stuff with colourmap limits at the end
            if annotate == True:
                #for string precision
                max_length = np.max([len(str(c))-len(str(float(int(c))))+1 for c in contour_lvls['lvls']])
                if max_length < 0:
                    max_length = 0
                y = y_reset
                if len_contour_lists[c] < np.max(len_contour_lists):
                    y_correction = 0.5*( (tot_num_of_levels - len_contour_lists[c]) / (tot_num_of_levels))
                    y += tot_num_of_levels*yspacing*y_correction
                yspacing = yspacing_reset
                for lvl in contour_lvls['lvls'][::-1]:
                    clabel = str(format(lvl, '.'+str(max_length)+'f')) + string
                    plt.annotate(clabel ,(x,y), ha='left', size=char_to_arcsec, color=key) #+0.6*xspacing
                    y += yspacing
                x += xspacing

        y = y_reset+tot_num_of_levels*(yspacing)

        e_unique, fpm_unique, chu_unique = self.unique_entries(meta_info)
        
        # varying factors being investigated
        if annotate == True:
            varying = [l for l in [e_unique, chu_unique, fpm_unique] if len(l)==len(nusun_objects) and len(nusun_objects)!=1] 
            if type(background_contours) == dict:
                orig_x = x-xspacing*(len(nusun_objects)+maps_already_here-1)
                # make sure the title for BG is at the top column titles
                y = y+(len(varying)-1)*yspacing if len(varying) > 0 else y
                plt.annotate('BG Image' ,(orig_x,y), ha='left', size=char_to_arcsec, color=list(background_contours.keys())[0])
                y = y-(len(varying)-1)*yspacing # reset for titles of columns
            for v in varying:
                orig_x = x-xspacing*len(nusun_objects)
                for c, key in enumerate(nusun_objects):
                    e_string = ''
                    fpm_string = ''
                    chu_string = ''
                    if type(v[c]) == tuple: #energy
                        e_string = str(v[c][0])+'-'+str(v[c][1])+' keV'
                    elif type(v[c]) == str and len(v[c]) == 1: # FPM
                        fpm_string = 'FPM'+v[c]
                    elif type(v[c]) == str and len(v[c]) > 1: # chu
                        chu_string = v[c]
                    plt.annotate(fpm_string+e_string+chu_string ,(orig_x,y), ha='left', size=char_to_arcsec, color=key)
                    orig_x += xspacing
                y += yspacing

        legend_title = 'NuSTAR' 
        complete_legend = self.complete_legend_title(legend_title, e_unique, fpm_unique, chu_unique)

        if annotate == True:
            # if there are no sub-heading for the values then I don't want a gap where they would be
            x_subtitle = x-xspacing*len(nusun_objects)
            if len(nusun_objects) == 1:
                yspacing = -yspacing
            if type(background_contours) == dict:
                x_subtitle -= xspacing
            plt.annotate(complete_legend ,(x_subtitle,y), ha='left', size=char_to_arcsec)
        
        del nusun_objects
        
        if type(bg_limits) != type(None):
            if bg_limits[0] <= 0: #vmin > 0 or error
                bg_limits[0] = 0.1
                compmap.plot(vmin=bg_limits[0], vmax=bg_limits[1]) 
            else:
                compmap.plot(vmin=bg_limits[0], vmax=bg_limits[1])
        elif bg_limits == None:
            compmap.plot()

        if type(usr_title) == type(None):
            plt.title(f'{map_title} at {time_range[0]} to {time_range[1][-8:]}')
        else:
            plt.title(usr_title)

        plt.rcParams['figure.frameon']=True
        return compmap
    
    
    nu_shift = None
    
    def setup_deconvolved_contours(self, nu_file=None, colour_and_energy=None, 
                                   time_range=None, submap=None, iterations=None, 
                                   aia_dir=None, deconvolve=True):
        
        # set defualt values is inputs are None
        nu_file = self.file_given if nu_file == None else nu_file
        colour_and_energy = self.colour_and_energy if colour_and_energy == None else colour_and_energy       
        time_range = self.time_range if time_range == None else time_range 
        submap = self.submap if submap == None else submap
        iterations = self.iterations if iterations == None else iterations
        aia_dir = self.aia_directory if aia_dir == None else aia_dir
        
        if deconvolve == True:
            nustar_obj, nustar_maps_corr = self.nu_deconv(nu_file=nu_file, colour_and_energy=colour_and_energy, 
                                                     time_interval=time_range, submap=submap, iterations=iterations)
        else:
            self.iterations = 0
            iterations = 0
            nustar_obj, nustar_maps_corr = self.multi_energy_nustar_frame(nu_file=nu_file, 
                                                                      colour_and_energy=colour_and_energy, 
                                                                      time_interval=time_range, submap=submap)

        aia_file_list = np.array(os.listdir(aia_dir))
        
        times_list = self.aia_file_times(aia_dir)

        good_indices = self.useful_time_inds(times_list, time_interval=time_range)

        files_in_trange = aia_file_list[good_indices]

        background_map = self.which_background(aia_dir, files_in_trange)

        background_frame, background_corr = self.aia_frame(background_map, submap=submap)
        
        if type(self.nu_shift) == type(None):
            self.nu_shift = self.corr_fpm(nustar_maps_corr, background_corr)
        else:
            self.nu_shift = self.nu_shift
        
        nu_final, nu_objs = self.apply_nu_shift_fpm(nu_file=nu_file, nu_shift=self.nu_shift, 
                                                    colour_and_energy=colour_and_energy, 
                                                    time_interval=time_range, submap=submap, 
                                                    iterations=iterations)
        # what do I need for plotting that isn't an attribute yet?
        self.nu_final_maps =  nu_final
        self.nu_final_objects = nu_objs
        self.background_frame = background_frame
    
    
    def plot_contours(self, iron='', background_limits=None, background_contours=False, background_cmap=None, save_name='', annotate=True, plot=True, usr_title=None):
        
        ax = self.create_contours(nusun_objects=self.nu_final_maps, nu_objects=self.nu_final_objects, 
                                  aia_object=self.background_frame, iron=iron, contours=self.colour_and_contours, 
                                  submap=self.submap, annotate=annotate, background_contours=background_contours, 
                                  bg_limits=background_limits, plot=plot, background_cmap=background_cmap, usr_title=usr_title)
        
        if save_name != '':
            plt.savefig(save_name, dpi=300, bbox_inches='tight')
        
        return ax
    
    
    def __add__(self, other):
        # magic method:
        ## what do I do if I sort out the NuSTAR stuff first then want to combine them? Do this!
        
        new_object = deepcopy(self)
        
        att_self = getattr(self, 'nu_final_maps', 'DoesNotExist') # if attribute is there then att get that value, 
        ## if not then it gets assigned 'DoesNotExist'
        att_other = getattr(other, 'nu_final_maps', 'DoesNotExist')
        if att_self == 'DoesNotExist':
            print('Running setup for first object.')
            self.setup_deconvolved_contours()
        if att_other == 'DoesNotExist':
            print('Running setup for second object.')
            other.setup_deconvolved_contours()
        
        # setup new attributes
        new_object.files_given = [self.file_given, other.file_given]
        #directory of the files
        new_object.nu_file_directorys = [self.nu_file_directory, other.nu_file_directory]
        new_object.nu_filenames = [self.nu_filename, other.nu_filename]
        
        new_object.aia_directorys = [self.aia_directory, other.aia_directory]
        new_object.colour_dict = {**self.colour_dict, **other.colour_dict}
        
        # the submap/background/time range of the first object is used, self.submap remains unchanged
        ## this should mean it matters on the order of adding
        new_object.time_ranges = [self.time_range, other.time_range]
        new_object.submaps = [self.submap, other.submap]
        new_object.background_frames = [self.background_frame, other.background_frame]
        
        # now seperate colour_dict up for later function use into colour and energy, and colour and contour info
        new_object.colour_and_energy = {**self.colour_and_energy, **other.colour_and_energy}
        new_object.colour_and_contours = {**self.colour_and_contours, **other.colour_and_contours}
        
        
        new_object.nu_final_maps =  {**self.nu_final_maps, **other.nu_final_maps}
        new_object.nu_final_objects = {**self.nu_final_objects, **other.nu_final_objects}
        
        return new_object
