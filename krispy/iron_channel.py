'''
Functions to go in here (I think!?):
	KC: 29/04/2020, ideas-
	~create_iron16()			

    KC: 29/04/2020, added-	
    ~create_iron18()
'''
import os
import sunpy
import sunpy.map
import datetime
from datetime import timedelta
from astropy.io import fits

def create_iron18(dir_094=None, dir_171=None, dir_211=None, outdir=None):
    """Takes the 94, 171, 211 channels from SDO/AIA to create an iron18 emission proxy.
    
    Parameters
    ----------
    dir_*** : str
            The string of the directory with the 94A (dir_094), 171A (dir_171), and 211A (dir_211) files.
            Default: None
            
    outdir : str
            Directory for the output iron18 files.
            Default: None
            
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

    co_094 = []
    co_171 = []
    co_211 = []

    output = []

    for fn094 in files_094:
        time_094 = datetime.datetime.strptime(fn094[3:18], '%Y%m%d_%H%M%S')

        for fn171 in files_171:
            time_171 = datetime.datetime.strptime(fn171[3:18], '%Y%m%d_%H%M%S')
            if time_094 <= time_171 < time_094 + timedelta(seconds=12):

                for fn211 in files_211:     
                    time_211 = datetime.datetime.strptime(fn211[3:18], '%Y%m%d_%H%M%S')
                    if time_094 <= time_211 < time_094 + timedelta(seconds=12):
                        co_094.append(fn094)
                        co_171.append(fn171)
                        co_211.append(fn211)
                        break
                break
    files_094 = co_094
    files_171 = co_171
    files_211 = co_211
    
    for f094, f171, f211 in zip(files_094, files_171, files_211):
        aia_map_094 = sunpy.map.Map(dir_094+f094)
        data_094 = aia_map_094.data / aia_map_094.meta['exptime']
        data_094[data_094 < 0] = 0
        
        aia_map_171 = sunpy.map.Map(dir_171+f171)
        data_171 = aia_map_171.data / aia_map_171.meta['exptime']
        data_171[data_171 < 0] = 0
        
        aia_map_211 = sunpy.map.Map(dir_211+f211)
        data_211 = aia_map_211.data / aia_map_211.meta['exptime']
        data_211[data_211 < 0] = 0

        Iron_18 = data_094 - data_211/120 - data_171/450
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
        #names the file as 'AIAYYYYMMDD_HHmmss_FeXVIII_tbinned.fits'
        
        #change header info
        fits.setval(outdir + f094[:18] + '_FeXVIII.fits', 'pixlunit', value='DN/s')
        fits.setval(outdir + f094[:18] + '_FeXVIII.fits', 'exptime', value=1)
        #add in info that this is iron 18 apart from just the filename
        fits.setval(outdir + f094[:18] + '_FeXVIII.fits', 'iron_channel', value='iron18')

        output.append(f094[:18] + '_FeXVIII.fits')
    return output