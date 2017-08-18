''' 
Read in a list of fits files, and tile them
'''
from StrongCNN.IO.load_images import load_images, normalized_positive_image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fitsfiles', nargs='+', help='glob for fitsfiles to tile')
parser.add_argument('-d', '--directory', help='directory to save location', required=False)
parser.add_argument('-n', '--name', help='savename', required=False)
parser.add_argument('-c', '--cmap', help='colormap name', required=False)
parser.add_argument('-l', '--labels', nargs='+', help='labels for each fits file', required=False)
parser.add_argument('-N', '--ncols', help='Number of columns in tile', required=False)
parser.add_argument('-H', '--hog', help='hog transform option', required=False)

args = vars(parser.parse_args())

def plot_set( fits_list, name, labels=None,cmap_name='viridis',ncols=2 ) :
    images = load_images( fits_list ) 
    print [image.shape for image in images]
    images = [normalized_positive_image(image) for image in images]
    fig, axes = plt.subplots(nrows=len(images)/ncols, ncols=ncols)

    if labels != None : 
        for ax, data, label in zip(axes.flat,images, labels) :
            if args['hog'] != None : data = hog_image(data,label)
            im = ax.imshow(data, cmap=plt.get_cmap(cmap_name))
            if len(label) > 6 :
                plt.text(0.1,0.9,'%.2f'%float(label), ha='left', va='center', transform=ax.transAxes,fontsize='large',color='white')
            else :
                plt.text(0.1,0.9,label, ha='left', va='center', transform=ax.transAxes,fontsize='large',color='white')

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    else :
        for ax, data in zip(axes.flat,images) :
            im = ax.imshow( data, cmap=plt.get_cmap(cmap_name) )
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    #plt.subplots_adjust(wspace=-0.8, hspace=-0.4)
    plt.subplots_adjust(wspace=-0.4, hspace=-0.4)
    print "saving "+name
    plt.tight_layout()
    plt.savefig(name,bbox_inches='tight',pad_inches=0)

def contrast_image( image ) :
    return np.log(np.clip(image,1e-6,1e100)+1.0)/(image+1.0).max()

def hog_image( image, hog_kw_set=None ) :
    from skimage.feature import hog
    print hog_kw_set
    hog_kw_sets = {'HST':{'orientations' : 6,
                          'pixels_per_cell' : (16,16),
                          'cells_per_block' : (4, 4),
                          },
                   'nHST':{'orientations' : 5,
                           'pixels_per_cell' : (32,32),
                           'cells_per_block' : (4, 4),
                           },
                   'LSST1':{'orientations' : 5,
                            'pixels_per_cell' : (5, 5),
                            'cells_per_block' : (2, 2),
                            },
                   'LSST10':{'orientations' : 3,
                             'pixels_per_cell' : (4, 4),
                             'cells_per_block' : (3, 3),
                             },
                   'nLSST1':{'orientations' : 5,
                             'pixels_per_cell' : (4, 4),
                             'cells_per_block' : (5, 5),
                             },
                   'nLSST10':{'orientations' : 5,
                              'pixels_per_cell' : (6, 6),
                              'cells_per_block' : (5, 5),
                              },
        }
    hog_kw = hog_kw_sets[hog_kw_set]
    print hog_kw
    return hog(image,
               orientations = hog_kw['orientations'],
               pixels_per_cell = hog_kw['pixels_per_cell'],
               cells_per_block = hog_kw['cells_per_block'],
               visualise = True )[1]


    
if __name__ == '__main__' :
    plot_set( args['fitsfiles'], args['directory']+'/'+args['name'], labels=args['labels'], ncols=int(args['ncols']))

