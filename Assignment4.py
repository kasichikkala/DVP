import cv2
import numpy as np

# load image
input_image = cv2.imread('/Users/kasi/Downloads/nadia2.jpg', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('/Users/kasi/Downloads/nadiatemplate.jpg', cv2.IMREAD_GRAYSCALE)

# search matches
result = cv2.matchTemplate(input_image, template, cv2.TM_CCOEFF_NORMED)

# normalize result
normalized_result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# find min max
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# white retangle over the image
top_left = max_loc
bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
cv2.rectangle(input_image, top_left, bottom_right, 255, 2)

cv2.imshow('Detected', input_image)
cv2.imshow('Error Map', normalized_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
