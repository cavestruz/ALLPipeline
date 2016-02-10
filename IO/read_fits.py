import astropy.io.fits as pyfits
import pylab as pl

def get_fits_obj(fitsfile) :
    '''Returns contents of fitsfile in numpy array format'''
    return pyfits.getdata(fitsfile)

def visualize_fits(fits_io,figname) :
    '''Saves fitsfile contents as an image, <figname>.png'''
    pl.figure()
    pl.contour(fits_io)
    pl.colorbar()
    pl.savefig(figname+'.png')
