'''
Functions to go in here (I think!?):
	KC: 01/12/2018, ideas-
	~time_binning()			<
	~only_fits()			<
	~missing_aia_files()	<

	KC: 19/12/2018, added-
	~time_bin_fits()
	~only_fits()
	~find_aia_files()
'''

import sunpy
import sunpy.map
import numpy as np
from astropy.io import fits
import datetime
from datetime import timedelta
import os

#group fits files in time bins
def time_bin_fits(data_dir, tstart, tend, time_bin, out_dir):
    """Takes a directory with fits files, bins the data to the specified time bin value, then saves each bin as a new
    fits file and returns the list of newly saved files in the out_dir with new observation start time [obs_time] and 
    pixel unit [DN to DN/s].
    
    Parameters
    ----------
    data_dir : Data directory
            The directory which contains the list of fits files from the AIA. Must end with a '/'.
    
    tstart : Start time for binning
            Must be of a form that can be converted to a datetime.datetime() object, e.g. '2018/06/28, 16:53:29.002',
            in order to match the format in line 36 (datetime.datetime.strptime(tstart, '%y/%m/%d, %H:%M:%S.%f')).
            
    tend : End time for binning
            Must be in the same form as the start time. The time bins are made by adding on the number of seconds for
            the time bin increments on to the start time until that value becomes larger than the end time. So the 
            true end time for the binning should end up seconds larger (< time bin increment) than tend.
            
    time_bin : Time bin increment in seconds, float
            This is the width of the time bins and must be in seconds.
    
    out_dir : Save directory
            The directory in which the new fits files are saved. Must end with a '/'.
            
    Returns
    -------
    A list of the time bin array for plotting.
    """
    counter = 1
    
    tstart = datetime.datetime.strptime(tstart, '%y/%m/%d, %H:%M:%S.%f')
    tend = datetime.datetime.strptime(tend, '%y/%m/%d, %H:%M:%S.%f')
    
    New_timebins = []
    inc = tstart
    
    while inc < tend: #creates the boundaries for the time bins
        New_timebins.append(inc)
        inc = inc + timedelta(seconds = time_bin)   
    
    files_in_bins = []
    for bins in range(len(New_timebins)-1): #populates the empty list so that each bin can be indexed next
        files_in_bins.append([])
    
    for file in os.listdir(data_dir): #puts each file into the correct index of the time bin array: files_in_bins
        if file.endswith('.fits'):
            aia_map = sunpy.map.Map(data_dir+file)
            obs_time = aia_map.meta['t_obs']
            obs_time_object = datetime.datetime.strptime(obs_time, '%Y-%m-%dT%H:%M:%S.%fZ')
            for r in range(len(New_timebins)-1):
                if New_timebins[r] <= obs_time_object < New_timebins[r+1]:      
                    files_in_bins[r].append(file)
                    files_in_bins[r].sort()
                    break

    for row, time in zip(files_in_bins, New_timebins): #cycles through each file for each time bin group
        if row != []:
            temp_list = []
            for file in row: #takes the files for a time bin, changes units from DN to DN per sec, appends this to a
                #temporary list which holds this new DN per sec data for a given time bin. Also pulls out header info 
                #to create new fits
                aia_map = sunpy.map.Map(data_dir+file)
                data_DN = aia_map.data
                data_DN_s = aia_map.data / aia_map.meta['exptime']
                data_DN_s[data_DN_s < 0] = 0
                temp_list.append(data_DN_s)

                file = fits.open(data_dir+file)
                hdr = file[0].header
                file.close()

            ave_data_for_bin = np.sum(temp_list, axis=0) / np.shape(temp_list)[0]

            primary_hdu = fits.PrimaryHDU(data = ave_data_for_bin ,header=hdr) #header from last file from loop above

            hdul = fits.HDUList([primary_hdu])

            YMDhms = []
            for num in [time.year, time.month, time.day, time.hour, time.minute, time.second]:
                if num < 10:
                    YMDhms.append(f'0{num}') #pads a number if it doesn't have two digits (except for the year)
                else:
                    YMDhms.append(f'{num}')

            wavelength = str(aia_map.meta['wavelnth']) #pads the wavelength with zeros for the naming convention
            while len(wavelength) < 4:
                wavelength = '0' + wavelength

            hdul.writeto(out_dir + 'AIA'+f'{YMDhms[0]}{YMDhms[1]}{YMDhms[2]}'+'_'+f'{YMDhms[3]}{YMDhms[4]}{YMDhms[5]}'+\
                        '_'+f'{wavelength}'+ f'_tbinned{int(time_bin)}.fits', overwrite=True) 
            #creates and names the fits file as 'AIAYYYYMMDD_HHmmss_1234_tbinned.fits'

            eff_time = time.isoformat(timespec='microseconds') #give the time out as YYYY-MM-DDTHH:mm:ss.ms

            fits.setval(out_dir + 'AIA'+f'{YMDhms[0]}{YMDhms[1]}{YMDhms[2]}'+'_'+f'{YMDhms[3]}{YMDhms[4]}{YMDhms[5]}'+\
                        '_'+f'{wavelength}'+f'_tbinned{int(time_bin)}.fits', 'date_obs', value=f'{eff_time}') 
            #changes the time in the header to the start of the appropriate time bin
            fits.setval(out_dir + 'AIA'+f'{YMDhms[0]}{YMDhms[1]}{YMDhms[2]}'+'_'+f'{YMDhms[3]}{YMDhms[4]}{YMDhms[5]}'+\
                        '_'+f'{wavelength}'+f'_tbinned{int(time_bin)}.fits', 'pixlunit', value='DN/s') 
            #changes the pixel unit to DN to DN/s

            print(f'\rFile(s) saved: {counter}', end='')
            counter += 1
        
    print('\nI\'m Finished!')
    return New_timebins #returns time bins.



#remove any non-fits file
def only_fits(fits_list, ext = '.fits'):
    """Takes a list of files and makes a new list only containing the wanted file type from the original list.
    
    Parameters
    ----------
    fits_list : List
            A list of the files which contain the wanted file type as well as other, unwanted files/directories.
    
    ext : Str
            The wanted file extension, e.g. '.fits', '.fts', '.pdf', etc.
            Default: '.fits'
            
    Returns
    -------
    A list which only contains the wanted file type from the original list.
    """
    
    fits_files = [ f for f in fits_list if f.endswith(ext)]

    return fits_files



#The missing file finder function
def find_aia_files(directory, cadence=12):
    files_list = list(os.listdir(directory))
    files = [ f for f in files_list if f.endswith('.fits')]
    files.sort()
    
    assert files != [], f'No .fits files in {directory}.'

    no_friends = [] #files that do not have a friend within the cadence time after it

    for f in range(len(files)-2): #don't want to look at the last one as it will never have anything after it anyway
        time_0 = datetime.datetime.strptime(files[f][3:18], '%Y%m%d_%H%M%S')
        time_1 = datetime.datetime.strptime(files[f+1][3:18], '%Y%m%d_%H%M%S')
        if time_0 <= time_1 <= time_0 + timedelta(seconds=cadence): #if there is a file <=12s ahead move on
            continue
        else: #if there is not a file <=12s ahead add it to the no friends list
            no_friends.append(files[f])

    if no_friends == []:
        print('All files here!')
    else:
        print(f'Here are files without friends {cadence} seconds ahead of them from directory \n{directory}:')
        print(no_friends)
        
    print('Please check there are no files missing after the last file.')
        
    return no_friends