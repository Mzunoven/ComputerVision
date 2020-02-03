from alignChannels import alignChannels
import numpy as np
import cv2
import imageio
# Problem 1: Image Alignment

# 1. Load images (all 3 channels)
red = np.load('Computer Vision/Assignments/hw0/data/red.npy')
green = np.load('Computer Vision/Assignments/hw0/data/green.npy')
blue = np.load('Computer Vision/Assignments/hw0/data/blue.npy')

# 2. Find best alignment
rgbResult = alignChannels(red, green, blue)

# 3. save result to rgb_output.jpg (IN THE "results" FOLDER)
imageio.imwrite(
    'Computer Vision/Assignments/hw0/results/rgb_output.jpg', rgbResult)
