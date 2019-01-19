'''
Functions to go in here (I think!?):
	KC: 01/12/2018, ideas-
	~make_maps_from_dir()		<
	~contour_maps_from_dir()	<
	~iron_18_cmap()				<

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

'''
Alterations:
	KC: 20/12/2018 - made it possible to add more than one rectangle in 'aiamaps()'.
	KC: 19/01/2019 - the sunpy.map.Map() now takes in a list instead of map objects being created
						for every file.
	KC: 19/01/2019 - aiamaps() can nowtake a wavelength for a title, look at two iron channels,
						change the scale of the colour bar, and can now subtract average of 
						observation.						
	KC: 19/01/2019 - contourmaps_from_dir() now takes an AIA file from within time range instead
						having time bins all having to match bfore input.
'''

#make images from the aia fits files
def aiamaps(directory, save_directory, submap, wavelength='', cmlims = [], rectangle=[], save_inc=True, 
	iron='', cm_scale='Normalize', ave_diff_image=False):      
	"""Takes a directory with fits files, constructs a map or submap of the full observation with/without a rectangle and
	saves the image in the requested directory.
	
	Parameters
	----------
	directory : Data directory
			The directory which contains the list of fits files from the AIA. Must end with a '/'.
	
	save_directory : Save directory
			The directory in which the new fits files are saved. Must end with a '/'.
		
	savefile_fmt : Str
			File extension for the saved file's format, e.g. '.png', '.jpg', '.pdf', etc.
			Default: '.png'
			
	submap : One-dimensional list/array, length 4
			Contains the bottom left (bl) and top right (tr) coordinates for a submap, e.g. [blx,bly,trx,try]. Must be 
			in arcseconds, of type float or integer and NOT an arcsec object.

	wavelength : Str
			Indicates what wavelength is being passed into the function.
			Default: '' 

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
	
	iron : Bool
			Indicates whether or not the save file is the iron 16 or 18 channel. Set to '16' or '18'.
			Default: ''

	cm_scale : Str
			Scale for the colour bar for the plot. Set to 'Normalize' or 'LogNorm'.
			Default: 'Normalize'

	ave_diff_image : Bool
			For the images to be plotted to have the average value for that observation subtracted.
			Default: False
			
	Returns
	-------
	AIA maps saved to the requested directory (so doesn't really return anythin).
	"""
	#sets up plots
	matplotlib.rcParams['font.sans-serif'] = "Arial" 
	matplotlib.rcParams['font.family'] = "sans-serif"
	matplotlib.rcParams['font.size'] = 12
	
	aia_files = os.listdir(directory)
	aia_files = only_fits(aia_files)
	aia_files.sort()
	
	d = 0
	no_of_files = len(aia_files)

	directory_with_files = [directory+f for f in aia_files]
	aia_maps = sunpy.map.Map(directory_with_files, sequence=True)
	if ave_diff_image == True:
		mean = aia_maps.as_array().mean()
		min_for_diff = np.min(aia_maps.as_array() - aia_maps.as_array().mean())
		max_for_diff = np.max(aia_maps.as_array())
		cmlims = [min_for_diff, max_for_diff]
	
	for f in range(no_of_files):
		if no_of_files == 1:
			aia_map = aia_maps
		elif no_of_files > 1:
			aia_map = aia_maps[f]

		if ave_diff_image == True:
			diff_data = aia_map.data - mean
			aia_map = sunpy.map.Map(diff_data, aia_map.meta)
		
		bl_fi = SkyCoord(submap[0]*u.arcsec, submap[1]*u.arcsec, frame=aia_map.coordinate_frame)
		tr_fi = SkyCoord(submap[2]*u.arcsec, submap[3]*u.arcsec, frame=aia_map.coordinate_frame)
		
		smap = aia_map.submap(bl_fi,tr_fi)    
			
		if iron == '18':
			smap.plot_settings['cmap'] = plt.cm.Blues
		if iron == '16':
			smap.plot_settings['cmap'] = plt.cm.Purples

		fig = plt.figure(figsize=(9,8));
		compmap = sunpy.map.Map(smap, composite=True)
			
		if cm_scale == 'Normalize':
			if cmlims != []:
				if cmlims[0] <= 0: #vmin > 0 or error
					cmlims[0] = 0.1
					compmap.plot(vmin=cmlims[0], vmax=cmlims[1], norm=colors.Normalize()); #LogNorm  Normalize
				else:
					compmap.plot(vmin=cmlims[0], vmax=cmlims[1], norm=colors.Normalize());
			elif cmlims == []:
				compmap.plot(norm=colors.Normalize());
			plt.colorbar(label='Linear Scale');
			
		elif cm_scale == 'LogNorm':
			if cmlims != []:
				if cmlims[0] <= 0: #vmin > 0 or error
					cmlims[0] = 0.1
					compmap.plot(vmin=cmlims[0], vmax=cmlims[1], norm=colors.LogNorm()); #LogNorm  Normalize
				else:
					compmap.plot(vmin=cmlims[0], vmax=cmlims[1], norm=colors.LogNorm());
			elif cmlims == []:
				compmap.plot(norm=colors.LogNorm());
			plt.colorbar(label='Log Scale');
		
		if rectangle != []: #if a rectangle is specified, make it
			for rect in rectangle:
				bl_rect = SkyCoord(rect[0]*u.arcsec, rect[1]*u.arcsec, frame=aia_map.coordinate_frame)
				length = rect[2] - rect[0]
				height = rect[3] - rect[1]
				if iron18 == True:
					smap.draw_rectangle(bl_rect, length*u.arcsec, height*u.arcsec, color = 'black')
				else:
					smap.draw_rectangle(bl_rect, length*u.arcsec, height*u.arcsec)
		
		if ave_diff_image == False:
			if iron == '18': #sets title for Iron 18
				time = aia_map.meta['t_obs']
				plt.title(f'AIA FeXVIII {time[:10]} {time[11:19]}')  
			elif iron == '16': #sets title for Iron 16
				time = aia_map.meta['t_obs']
				plt.title(f'AIA FeXVI {time[:10]} {time[11:19]}')
			else:
				time = aia_map.meta['t_obs']
				plt.title('AIA '+ wavelength + r'$\AA$ ' + f'{time[:10]} {time[11:19]}')
		elif ave_diff_image == True:
			if iron == '18': #sets title for Iron 18
				time = aia_map.meta['t_obs']
				plt.title(f'AIA FeXVIII {time[:10]} {time[11:19]} (with Mean of Obs. Subtracted)')  
			elif iron == '16': #sets title for Iron 16
				time = aia_map.meta['t_obs']
				plt.title(f'AIA FeXVI {time[:10]} {time[11:19]} (with Mean of Obs. Subtracted)')
			else:
				#print(aia_map.meta)
				time = aia_map.meta['t_obs']
				plt.title('AIA '+ wavelength + r'$\AA$ ' + f'{time[:10]} {time[11:19]} (Mean of Obs. Subtracted)')       

		if save_inc == False:
			plt.savefig(save_dir + f'AIA_image_{time[:10]}_{time[11:19]}.png', dpi=600, bbox_inches='tight')
		elif save_inc == True:
			plt.savefig(save_dir + 'maps{:03d}.png'.format(d), dpi=600, bbox_inches='tight')
		d+=1
				
		plt.close(fig)
		bl_fi = 0
		tr_fi = 0 
		del bl_fi
		del tr_fi
		print(f'\rSaved {d} submap(s) of {no_of_files}.', end='')
	
	aia_map = 0
	aia_maps = 0
	aia_files = 0
	smap = 0
	compmap = 0
	del aia_map
	del aia_maps
	del aia_files
	del smap
	del compmap
	print('\nLook everyone, it\'s finished!')



#make contour maps
def contourmaps_from_dir(aia_dir, nustar_dir, nustar_file, save_dir, chu='', fpm='', energy_rng=[], submap=[], 
							 cmlims = [], nustar_shift=[], time_bins=[], resample_aia=[], counter=0, contour_lvls=[],
						 contour_fmt='percent', contour_colour='black', aia='ns_overlap_only', iron18=True, 
						 save_inc=False):
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
	
	time_bins : One-dimensional list/array of type int, length N
			This list/array provides the time boundariesfort he NuSTAR time bins.
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
			
	Returns
	-------
	A dictionary with the values of the largest values of the NuSTAR map to help with contour value setting 
	(labelled as 'max_contour_levels') and the final value for the incremental counter (labelled as 
	'last_incremental_value'). AIA maps, with NuSTAR contours, are also saved to the requested directory.
	"""
	#20/11/2018: ~if statement for the definition of cleanevt.
	#            ~two iron 18 if statements for the colour map.
	#            ~two iron 18 if statements for the colour map.
	#26/11/2018: ~added try and except to the cleanevt bit.

	import filter_with_tmrng # this file has to be in the directory

	matplotlib.rcParams['font.sans-serif'] = "Arial" #sets up plots
	matplotlib.rcParams['font.family'] = "sans-serif"
	matplotlib.rcParams['font.size'] = 12
	
	hdulist = fits.open(nustar_dir + nustar_file) #chu, sunpos file
	evtdata=hdulist[1].data
	hdr = hdulist[1].header
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
		except IndexError:
			cleanevt = [] #if time range is outwith nustar obs (i.e. IndexError) then this still lets aia to be looked at
			
		if len(cleanevt) != 0 and aia == ('ns_overlap_only' or 'all'): #AIA data and NuSTAR data
			nustar_map = nustar.map.make_sunpy(cleanevt, hdr, norm_map=False)
			nustar_map_normdata = nustar_map.data / (24) #24 second time bins
			dd=ndimage.gaussian_filter(nustar_map_normdata, 4, mode='nearest');
			
			# Tidy things up before plotting
			dmin=1e-3
			dmax=1e1
			dd[dd < dmin]=0
			nm=sunpy.map.Map(dd, nustar_map.meta);
			
			aia_map = 0
			for f in aia_files:
				if time_bins[t][10:18] <= f[12:14]+':'+ f[14:16]+':'+ f[16:18] < time_bins[t+1][10:18]:
					aia_map = sunpy.map.Map(aia_dir + f);
					if resample_aia != []:
						dimensions = u.Quantity([resample_aia[0], resample_aia[1]], u.pixel)
						aia_map = aia_map.resample(dimensions, method='linear');
					break
   
			if aia_map == 0:
				print(f'\rNo AIA data in this time range: {time_bins[t]}, {time_bins[t+1]}.', end='')
				continue
				
			# Let's shift it ############################################################################################
			if nustar_shift != []:
				shifted_nustar_map = nm.shift(nustar_shift[0]*u.arcsec, nustar_shift[1]*u.arcsec)
			else:
				shifted_nustar_map = nm

			# Submap to plot ############################################################################################
			bl = SkyCoord((submap[0]+0.01)*u.arcsec, (submap[1]+0.01)*u.arcsec, frame=shifted_nustar_map.coordinate_frame)
			tr = SkyCoord((submap[2]-0.01)*u.arcsec, (submap[3]-0.01)*u.arcsec, frame=shifted_nustar_map.coordinate_frame)
			#0.01 arcsec padding to make sure the contours aren't cut off and so that the final plot won't have weird 
			#blank space bits
			shifted_nustar_submap = shifted_nustar_map.submap(bl,tr)

			cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [contour_colour,contour_colour])
			shifted_nustar_submap.plot_settings['norm'] = colors.LogNorm(vmin=dmin,vmax=dmax)
			shifted_nustar_submap.plot_settings['cmap'] = cmap

			#############################################################################################################
			bl_fi = SkyCoord(submap[0]*u.arcsec, submap[1]*u.arcsec, frame=aia_map.coordinate_frame)
			tr_fi = SkyCoord(submap[2]*u.arcsec, submap[3]*u.arcsec, frame=aia_map.coordinate_frame)
			#0.01 arcsec padding
			smap = aia_map.submap(bl_fi,tr_fi)    
			
			if iron18 == True:
				smap.plot_settings['cmap'] = plt.cm.Blues

			fig = plt.figure(figsize=(9,8));
			compmap = sunpy.map.Map(smap, composite=True);
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
				
			plt.colorbar();
			
			plt.title(f'FeXVIII at {time_bins[t][:-7]} to {time_bins[t+1][10:18]} fpm ' + fpm + ' ' + chu +
					  f' {energy_rng[0]}-{energy_rng[1]} keV');

			if save_inc == False:
				plt.savefig(save_dir + f'nustar_contours{d}_on_iron18_chu{chu}_fpm{fpm}.png', dpi=600, 
							bbox_inches='tight')
			elif save_inc == True:
				plt.savefig(save_dir + 'contours{:03d}.png'.format(d), dpi=600, bbox_inches='tight')
			d+=1
				
			plt.close(fig)
			print(f'\rSaved {d} submap(s).', end='')
			
			max_contours.append(shifted_nustar_submap.data.max())
			
		elif (len(cleanevt) == 0 and aia == 'all') or aia == 'solo': #just AIA data
			
			aia_map = 0
			for f in aia_files:
				if f[12:14]+':'+ f[14:16]+':'+ f[16:18] == str(time_bins[t][10:18]):
					aia_map = sunpy.map.Map(aia_dir + f);
					if resample_aia != []:
						dimensions = u.Quantity([resample_aia[0], resample_aia[1]], u.pixel)
						aia_map = aia_map.resample(dimensions, method='linear');
					break
					
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
				
			plt.colorbar();
			
			plt.title(f'FeXVIII at {time_bins[t][:-7]} to {time_bins[t+1][10:18]}')
			print(f'{time_bins[t][:-7]} to {time_bins[t+1][10:18]}')         

			if save_inc == False:
				plt.savefig(save_dir + f'nustar_contours{d}_on_iron18_chu{chu}_fpm{fpm}.png', dpi=600, 
							bbox_inches='tight')
			elif save_inc == True:
				plt.savefig(save_dir + 'contours{:03d}.png'.format(d), dpi=600, bbox_inches='tight')
			d+=1
				
			plt.close(fig)
			print(f'\rSaved {d} submap(s).', end='')
		
		else:
			print(f'\rNo NuSTAR data in this time range: {time_bins[t]} to {time_bins[t+1]}.', end='')
	
	print('\nLook everyone, it\'s finished!')
	return {'max_contour_levels': max_contours, 'last_incremental_value': d-1} 
	#helps find the values for the contour lines and the last number padded for the incremental saves



#make maps from the fits file
############################################### **Warning** ###############################################
###########################################################################################################
###### This function has a memory leak and I don't know where yet so don't run a huge list of files #######
###### through it at once! ################################################################################
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
	files = only_fits(files)
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
			
			done += 1
			print('\rSaved {} map(s) of {}'.format(done, num_f), end='')
	print('\nAll files saved!')