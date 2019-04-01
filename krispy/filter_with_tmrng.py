import numpy as np
import datetime # edited by Kris ~16/10/2018


def bad_pix(evtdata, fpm='A'):
	"""Do some basic filtering on known bad pixels.
	
	Parameters
	----------
    evtdata: FITS data class
		This should be an hdu.data structure from a NuSTAR FITS file.

    fpm: {"FPMA" | "FPMB"}
		Which FPM you're filtering on. Assumes A if not set.


	Returns
    -------

	goodinds: iterable
		Index of evtdata that passes the filtering.
	"""
    

	
	# Hot pixel filters
	
	# FPMA or FPMB
	
	if fpm.find('B') == -1 :
		pix_filter = np.invert( ( (evtdata['DET_ID'] == 2) & (evtdata['RAWX'] == 16) & (evtdata['RAWY'] == 5) |
								(evtdata['DET_ID'] == 2) & (evtdata['RAWX'] == 24) & (evtdata['RAWY'] == 22) |
								(evtdata['DET_ID'] == 2) & (evtdata['RAWX'] == 27) & (evtdata['RAWY'] == 6) |
								(evtdata['DET_ID'] == 2) & (evtdata['RAWX'] == 27) & (evtdata['RAWY'] == 21) |
								(evtdata['DET_ID'] == 3) & (evtdata['RAWX'] == 22) & (evtdata['RAWY'] == 1) |
								(evtdata['DET_ID'] == 3) & (evtdata['RAWX'] == 15) & (evtdata['RAWY'] == 3) |
								(evtdata['DET_ID'] == 3) & (evtdata['RAWX'] == 5) & (evtdata['RAWY'] == 5) | 
								(evtdata['DET_ID'] == 3) & (evtdata['RAWX'] == 22) & (evtdata['RAWY'] == 7) | 
								(evtdata['DET_ID'] == 3) & (evtdata['RAWX'] == 16) & (evtdata['RAWY'] == 11) | 
								(evtdata['DET_ID'] == 3) & (evtdata['RAWX'] == 18) & (evtdata['RAWY'] == 3) | 
								(evtdata['DET_ID'] == 3) & (evtdata['RAWX'] == 24) & (evtdata['RAWY'] == 4) | 
								(evtdata['DET_ID'] == 3) & (evtdata['RAWX'] == 25) & (evtdata['RAWY'] == 5) ) )
	else:
		pix_filter = np.invert( ( (evtdata['DET_ID'] == 0) & (evtdata['RAWX'] == 24) & (evtdata['RAWY'] == 24)) )


	inds = (pix_filter).nonzero()
	goodinds=inds[0]
	
	return goodinds
	
def by_energy(evtdata, energy_low=2.5, energy_high=10.):
	""" Apply energy filtering to the data.
	
	Parameters
	----------
	evtdata: FITS data class
		This should be an hdu.data structure from a NuSTAR FITS file.
		
	energy_low: float
		Low-side energy bound for the map you want to produce (in keV).
		Defaults to 2.5 keV.

    energy_high: float
		High-side energy bound for the map you want to produce (in keV).
		Defaults to 10 keV.
	"""		
	pilow = (energy_low - 1.6) / 0.04
	pihigh = (energy_high - 1.6) / 0.04
	pi_filter = ( ( evtdata['PI']>pilow ) &  ( evtdata['PI']<pihigh))
	inds = (pi_filter).nonzero()
	goodinds=inds[0]
	
	return goodinds
	
def gradezero(evtdata):
	""" Only accept counts with GRADE==0.
		
	Parameters
	----------
	evtdata: FITS data class
		This should be an hdu.data structure from a NuSTAR FITS file.
		
	Returns
    -------

	goodinds: iterable
		Index of evtdata that passes the filtering.
	"""

	# Grade filter
	
	grade_filter = ( evtdata['GRADE'] == 0)
	inds = (grade_filter).nonzero()
	goodinds = inds[0]
	
	return goodinds

# this function, time_range, was not here # created by Kris ~16/10/2018
def time_range(evtdata, tmrng):    
	""" Only include counts within a given time range.
	(********************************This doc string was edited from one above ********************************)
	Parameters
	----------
	evtdata: FITS data class
		This should be an hdu.data structure from a NuSTAR FITS file.
    
	tmrng : list of length 2   
		Input two times in the form 'yyyy/mm/dd, HH:MM:SS'
		(e.g. '2019/03/06, 16:45:30') for the time range,
		default is the whole observation for the file. 
	Returns
	-------

	goodinds: iterable
		Index of evtdata that lies within the time range.
	(********************************This doc string was edited from one above ********************************)
	"""
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
 
	return goodinds 

def event_filter(evtdata, fpm='FPMA',
	energy_low=2.5, energy_high=10, tmrng = None):
	# was event_filter(evtdata, fpm='FPMA', energy_low=2.5, energy_high=10) # Kris #
	""" All in one filter module. By default applies an energy cut, 
		selects only events with grade == 0, and removes known hot pixel.
		
		Note that this module returns a cleaned eventlist rather than
		the indices to the cleaned events.

	Parameters
	----------
	evtdata: FITS data structure
		This should be an hdu.data structure from a NuSTAR FITS file.
	
    fpm: {"FPMA" | "FPMB"}
		Which FPM you're filtering on. Defaults to FPMA.
		
	energy_low: float
		Low-side energy bound for the map you want to produce (in keV).
		Defaults to 2.5 keV.

    energy_high: float
		High-side energy bound for the map you want to produce (in keV).
		Defaults to 10 keV.

	#tmrng : list of strings, length 2                           Kris
	#	Input two times in the form 'yyyy/mm/dd, HH:MM:SS'       Kris
	#	(e.g. '2019/03/06, 16:45:30') for the time range,        Kris
	#	default is the whole observation for the file.           Kris
		
	Returns
    -------

	cleanevt: FITS data class.
		This is the subset of evtdata that pass the data selection cuts.
	"""
    
	goodinds = time_range(evtdata, tmrng) #######                                             Kris
	evt_timefilter = evtdata[goodinds] ##########                                             Kris
	goodinds = bad_pix(evt_timefilter, fpm=fpm) # was  goodinds = bad_pix(evtdata, fpm=fpm)   Kris
	evt_badfilter = evt_timefilter[goodinds] #### was  evt_badfilter = evtdata[goodinds]      Kris
	goodinds = by_energy(evt_badfilter,
						energy_low=energy_low, energy_high = energy_high)
	evt_energy = evt_badfilter[goodinds]
	goodinds = gradezero(evt_energy)
	cleanevt = evt_energy[goodinds]
	return cleanevt


