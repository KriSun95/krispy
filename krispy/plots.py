'''
Functions to go in here (I think!?):
    KC: 06/08/2019, added-
    ~plot XSPEC output
'''

from .nu_spec import read_xspec_txt, seperate
from .data_handling import xspecParams


def xspecPlot(xspec_output, fitting_mode=None, **kwargs):
    """Takes XSPEC output files from a fitting and plots them.
    
    Parameters
    ----------
    xspec_output : str
        The fits and text file name from teh XSPEC output, e.g Xoutput.fits and Xoutput.txt means 
        that xspec_output="Xoutput".
        
    fitting_mode : str
        Information about the fit in XSPEC, e.g. 1apec model fit with on focal plane modules data: '1apec'. 
        Can set: '1apec', '1apec1bknpower', '3apec1bknpower', '4apec'
        Any fitting mode defined in nu_spec's seperate().

    kwargs : various
        Plotting variables
            
    Returns
    -------
    Axes object of plot?
    """
    # even though I say it in the doc string to remove .txt and .fits just check it anyway, I don't trust myself
    xspec_output = xspec_output if not xspec_output.endswith(".fits") else xspec_output[:-5]
    xspec_output = xspec_output if not xspec_output.endswith(".txt") else xspec_output[:-4]

    keys_to_check = []

    fitting_values = xspecParams(xspec_output+".fits", *keys_to_check)

    xspec_data = read_xspec_txt(xspec_output+".txt")
    counts_data = seperate(xspec_data, fitting_mode=fitting_mode)