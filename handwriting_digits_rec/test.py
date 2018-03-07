import cv2
import numpy as np

sample = cv2.imread('data/sample.jpg', cv2.IMREAD_GRAYSCALE)
cols, rows = sample.shape[:2]
cells = [np.hsplit(row, 9) for row in np.vsplit(sample, 4)]


