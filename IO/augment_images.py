'''Collection of methods to augment data.'''
from scipy.ndimage import rotate

def _rotate_images(images_X, images_y, degrees=[0]) :

    rotated_images_X = []
    rotated_y = []
    for d in degrees :
        rotated_images_X += [ rotate(image, int(d)) for image in images_X ]
        rotated_y += images_y
    return rotated_images_X, rotated_y

augment_methods = { 'rotate_images': _rotate_images,
                    }

def augment_data( X, y, augment_data_method_label, *args, **kwargs ) :
    return augment_methods[augment_data_method_label](X, y, *args, **kwargs)

