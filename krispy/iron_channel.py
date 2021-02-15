'''
Functions to go in here (I think!?):
	KC: 29/04/2020, ideas-
	~create_iron16()			

    KC: 29/04/2020, added-	
    ~create_iron18()
'''
from . import data_handling

import os
import sys
import sunpy
import sunpy.map
import datetime
from datetime import timedelta
from astropy.io import fits

def prep(aia_sunpy_map):
    from aiapy.calibrate import register, update_pointing, normalize_exposure
    m_updated_pointing = update_pointing(aia_sunpy_map)
    del aia_sunpy_map
    m_registered = register(m_updated_pointing)
    del m_updated_pointing
    return normalize_exposure(m_registered)

def create_iron18(dir_094=None, dir_171=None, dir_211=None, outdir=None, tr_degradation_corr=[True, '2018-09-09T12:00:00'], needing_prepped=False):
    """Takes the 94, 171, 211 channels from SDO/AIA to create an iron18 emission proxy (Del Zanna 2013).
    
    Parameters
    ----------
    dir_*** : str
            The string of the directory with the 94A (dir_094), 171A (dir_171), and 211A (dir_211) files.
            Default: None
            
    outdir : str
            Directory for the output iron18 files.
            Default: None

    tr_degradation_corr : list [bool, str]
            Do you want the iron18 to be created where 94, 171, 211  have been corrected for the instrument degredation?
            Also provide a time for the data-set of the form "YYYY-MM-DDTHH:mm:ss".
            (This should always really be True with the right time.)
            Default: [True, '2018-09-09T12:00:00']

    needing_prepped : bool
            Is the AIA data in dir_*** needing prepped?
            Default: False
            
    Returns
    -------
    Filenames of the iron 18 files.
    """

    files_094_list = list(os.listdir(dir_094))
    files_094 = []
    files_094 = [ f for f in files_094_list if f.endswith('.fits')]
    files_094.sort()

    files_171_list = list(os.listdir(dir_171))
    files_171 = []
    files_171 = [ f for f in files_171_list if f.endswith('.fits')]
    files_171.sort()

    files_211_list = list(os.listdir(dir_211))
    files_211 = []
    files_211 = [ f for f in files_211_list if f.endswith('.fits')]
    files_211.sort()

    degs = [1, 1, 1]
    if tr_degradation_corr[0] is True:
        import warnings
        warnings.simplefilter('ignore')
        from aiapy.calibrate import degradation
        from aiapy.calibrate.util import get_correction_table, CALIBRATION_VERSION
        import astropy.units as u
        from astropy import time
        
        correction_table = get_correction_table()
        time_obs = time.Time(tr_degradation_corr[1], scale='utc')
        channels=[94,171,211]*u.angstrom
        for i in range(3):
            deg = degradation(channels[i], time_obs, correction_table=correction_table)
            degs[i] = deg.value

    co_094 = []
    co_171 = []
    co_211 = []

    output = []

    for fn094 in files_094:
        print(fn094[39:-20])
        if needing_prepped:
            print(fn094[39:-20])
            time_094 = data_handling.getTimeFromFormat(fn094[39:-20], custom_fmt='%Y_%m_%dt%H_%M_%S')
        else:
            time_094 = data_handling.getTimeFromFormat(fn094[3:18]) #datetime.datetime.strptime(fn094[3:18], '%Y%m%d_%H%M%S')

        for fn171 in files_171:
            if needing_prepped:
                print(fn171[39:-20])
                time_171 = data_handling.getTimeFromFormat(fn171[39:-20], custom_fmt='%Y_%m_%dt%H_%M_%S')
            else:
                time_171 = data_handling.getTimeFromFormat(fn171[3:18]) #datetime.datetime.strptime(fn171[3:18], '%Y%m%d_%H%M%S')
            
            if time_094 <= time_171 < time_094 + timedelta(seconds=12):

                for fn211 in files_211:     
                    if needing_prepped:
                        time_211 = data_handling.getTimeFromFormat(fn211[39:-20], custom_fmt='%Y_%m_%dt%H_%M_%S')
                    else:
                        time_211 = data_handling.getTimeFromFormat(fn211[3:18]) #datetime.datetime.strptime(fn211[3:18], '%Y%m%d_%H%M%S')
                    
                    if time_094 <= time_211 < time_094 + timedelta(seconds=12):
                        co_094.append(fn094)
                        co_171.append(fn171)
                        co_211.append(fn211)
                        break
                break
    files_094 = co_094
    files_171 = co_171
    files_211 = co_211
    
    d = 1
    d_total = len(files_094)
    for f094, f171, f211 in zip(files_094, files_171, files_211):
        aia_map_094 = sunpy.map.Map(dir_094+f094)

        if needing_prepped:
            aia_map_094 = prep(aia_map_094)

        data_094 = aia_map_094.data / aia_map_094.meta['exptime']
        data_094[data_094 < 0] = 0
        
        aia_map_171 = sunpy.map.Map(dir_171+f171)

        if needing_prepped:
            aia_map_171 = prep(aia_map_171)

        data_171 = aia_map_171.data / aia_map_171.meta['exptime']
        data_171[data_171 < 0] = 0
        
        aia_map_211 = sunpy.map.Map(dir_211+f211)

        if needing_prepped:
            aia_map_211 = prep(aia_map_211)
        
        data_211 = aia_map_211.data / aia_map_211.meta['exptime']
        data_211[data_211 < 0] = 0

        Iron_18 = data_094/degs[0] - data_211/(120*degs[2]) - data_171/(450*degs[1])
        Iron_18[Iron_18 < 0] = 0
        aia_map_Fe18 = sunpy.map.Map(Iron_18, aia_map_094.meta)
        
        del aia_map_094
        del aia_map_171
        del aia_map_211
        
        file = fits.open(dir_094+f094)
        hdr = file[0].header
        file.close()

        primary_hdu = fits.PrimaryHDU(data = Iron_18 ,header=hdr)

        hdul = fits.HDUList([primary_hdu])

        hdul.writeto(outdir + f094[:18] + '_FeXVIII.fits', overwrite=True) 
        #names the file as 'AIAYYYYMMDD_HHmmss_FeXVIII.fits'
        
        #change header info
        fits.setval(outdir + f094[:18] + '_FeXVIII.fits', 'pixlunit', value='DN/s')
        fits.setval(outdir + f094[:18] + '_FeXVIII.fits', 'exptime', value=1)
        #add in info that this is iron 18 apart from just the filename
        fits.setval(outdir + f094[:18] + '_FeXVIII.fits', 'iron_channel', value='iron18')
        fits.setval(outdir + f094[:18] + '_FeXVIII.fits', 'file094', value=f094)
        fits.setval(outdir + f094[:18] + '_FeXVIII.fits', 'file171', value=f171)
        fits.setval(outdir + f094[:18] + '_FeXVIII.fits', 'file211', value=f211)
        
        if tr_degradation_corr[0] is True:
            fits.setval(outdir + f094[:18] + '_FeXVIII.fits', 'temp_resp_info', value='atLaunch')
            deg_94_str, deg_171_str, deg_211_str, cal_ver_str = str(degs[0]), str(degs[1]), str(degs[2]), str(CALIBRATION_VERSION)
            fits.setval(outdir + f094[:18] + '_FeXVIII.fits', 'cal_ver', value=cal_ver_str)
            fits.setval(outdir + f094[:18] + '_FeXVIII.fits', 'deg_94', value=deg_94_str)
            fits.setval(outdir + f094[:18] + '_FeXVIII.fits', 'deg_171', value=deg_171_str)
            fits.setval(outdir + f094[:18] + '_FeXVIII.fits', 'deg_211', value=deg_211_str)

        output.append(f094[:18] + '_FeXVIII.fits')

        print(f'\r[function: {sys._getframe().f_code.co_name}] Saved {d} submap(s) of {d_total}.        ', end='') 
        d += 1
    return output