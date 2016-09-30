from scipy.ndimage import rotate

def rotate_images(degrees, images_X, images_y) :

    rotated_images_X = []
    rotated_y = []
    for d in degrees :
        rotated_images_X += [ rotate(image, int(d)) for image in images_X ]
        rotated_y += images_y
    return rotated_images_X, rotated_y

