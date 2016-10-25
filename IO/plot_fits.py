import sys, os
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.colors import LogNorm
from _tools import read_failed_ids

datadir = sys.argv[1]+'/images/'
os.mkdir(datadir)

# Should return a list of all failed ids
failed_ids = read_failed_ids(datadir)

lensed_failed_ids = [ fid for fid in failed_ids if 'unlensed' not in fid ]
unlensed_failed_ids = [ fid for fid in failed_ids if 'unlensed' in fid ]

def multiplot( fitsfiles, name ) :
    ncols = len(fitsfiles)/2
    fig, axes = plt.subplots(nrows=2, ncols=ncols)
    for ax,d in zip(axes.flat,fitsfiles) :
        im = ax.imshow(d,cmap='gray',norm=LogNorm())
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(name+'.pdf')
