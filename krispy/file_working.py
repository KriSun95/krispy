'''
Functions to go in here (I think!?):
    KC: 01/12/2018, ideas-
    ~time_binning()         <
    ~only_fits()            <
    ~missing_aia_files()    <

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
from sunpy.net.vso import VSOClient
import glob

'''
Alterations:
    KC: 22/01/2019 - documentation for the find_aia_files() function is now added.
    KC: 22/01/2019 - the find_aia_files() function now has the option to automatically search for 
                        the missing files and also deletes any double files that are downloaded by 
                        accident. 
'''

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
def find_aia_files(directory, wavelength='', time_limits=None, cadence=12, download=None, double_check='Yes'):
    """Checks a directory for missing files from downloading AIA images and can check/download the 
    files from the missing time.

    ***This function may need to run several times. It depends on how well the files are downloaded***
    
    Parameters
    ----------
    directory : Str
            A string of the path to the files to be checked.
    
    wavelength : Str
            The wavelength of the files to check. Only important if download = 'yes' or 'auto'.
            Default: ''

    time_limits : list
            A list of two entries for the start and end time of the observation. To check if any files
            were missed before the first and after the last file you have.
            Default: None

    cadence : Int
            An integer number of seconds that should be the temporal seperation of the files.
            Default: 12

    download : Str
            Indicates whether missing files should be searched for/downloaded. If set to None then there
            will be a prompt to ask, enter 'Yes' or 'No'. Setting to 'auto' will search for the data 
            automatically without user input.
            Default: None

    double_check : Str
            After checking for more files to download check again - without downloading - to see if 
            there are any still missing, e.g. 'Yes' or 'No'.
            Default: 'Yes'
            
    Returns
    -------
    A list of the files with no friends cadence seconds after them.
    """
    files_list = list(os.listdir(directory))
    files = [ f for f in files_list if f.endswith('.fits')]
    files.sort()
    
    assert files != [], f'No .fits files in {directory}.' #make sure there are files in the first place

    no_friends = [] #files that do not have a friend within the cadence time after it
    t_of_no_friends = [] #time of files that have no friends

    for f in range(len(files)-2): #don't want to look at the last one as it will never have anything after it anyway
        time_0 = datetime.datetime.strptime(files[f][4:19], '%Y%m%d_%H%M%S')
        time_1 = datetime.datetime.strptime(files[f+1][4:19], '%Y%m%d_%H%M%S')
        if time_0 <= time_1 <= time_0 + timedelta(seconds=cadence): #if there is a file <=12s ahead move on
            continue
        else: #if there is not a file <=12s ahead add it to the no friends list
            no_friends.append(files[f])
            t_of_no_friends.append(time_0)
            
    if time_limits != None:
        time_first = datetime.datetime.strptime(files[0][4:19], '%Y%m%d_%H%M%S')
        time_last = datetime.datetime.strptime(files[-1][4:19], '%Y%m%d_%H%M%S')
        
        time_limits_first = datetime.datetime.strptime(time_limits[0], "%Y-%m-%d %H:%M:%S")
        time_limits_last = datetime.datetime.strptime(time_limits[1], "%Y-%m-%d %H:%M:%S")

        if time_first - time_limits_first >= timedelta(seconds=cadence): #if the diff between the start time and the 
            #time of the first file then there should be one there
            t_of_no_friends.append(time_limits_first)
        if time_limits_last - time_last  >= timedelta(seconds=cadence): #if the diff between the final time and the 
            #time of the last file then there should be one there
            t_of_no_friends.append(time_last)
            
    if (download == None) and (len(t_of_no_friends) > 0 ): #asks the user if the files should be downloaded 
        download = input('Would you like the times of the missing files checked? ('Yes' or 'No') ')
    elif download == 'No':
        pass
    
    if (download == 'auto' or download == 'Yes') and (len(t_of_no_friends) > 0 ):
        client = VSOClient()
        start_times = [t.strftime("%Y-%m-%d %H:%M:%S") for t in t_of_no_friends]
        #search a minute ahead
        end_times = [(t + timedelta(seconds=5*cadence)).strftime("%Y-%m-%d %H:%M:%S") for t in t_of_no_friends]
        for ts, te in zip(start_times, end_times):
            query_response = client.query_legacy(tstart=ts, tend=te, instrument='AIA', wave=wavelength)

            n = len(query_response) - 1 #will be used to index the list 'query_response' when downloading

            #Download the first two from ROB to /tmp folder and wait for download to complete
            results = client.get(query_response[0:n], path=directory, site='rob')
            files = results.wait()
            
    filelist=glob.glob(directory + '*.*.fits') #removes files that downloaded twice
    for each_file_path in filelist:
        os.remove(each_file_path)
    
    if double_check == 'No':
        if no_friends == []:
            print('All files here!')
        else:
            print(f'Here are files without friends {cadence} seconds ahead of them from directory \n{directory}:')
            print(no_friends)
        return no_friends
    if double_check == 'Yes': #double check to see if we have all the files
        still_no_friends = find_aia_files(directory, wavelength, time_limits=time_limits, download='No', double_check='No')
    
    return still_no_friends