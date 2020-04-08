import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image


def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    ##### your code here #####
    ##########################

    image = skimage.filters.gaussian(image, sigma=2, multichannel=True)
    image = skimage.color.rgb2gray(image)
    threshold = skimage.filters.threshold_otsu(image)

    bw = skimage.morphology.closing(
        image < threshold, skimage.morphology.square(10))
    clear = skimage.segmentation.clear_border(bw)

    label, num = skimage.measure.label(
        clear, background=0, return_num=True, connectivity=2)
    for i in skimage.measure.regionprops(label):
        if i.area >= 200:
            bboxes.append(i.bbox)

    return bboxes, bw
