import cv2
import numpy as np

def ebma(target, anchor, block_size, search_region):
    motion_vectors = []

    
    h, w = target.shape

    # loop over the image
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            
            best_match = (0, 0)
            min_error = float('inf')
            actual_block_size_x = min(block_size, w - x)
            actual_block_size_y = min(block_size, h - y)

            # define search
            x_start = max(x - search_region, 0)
            x_end = min(x + actual_block_size_x + search_region, w)
            y_start = max(y - search_region, 0)
            y_end = min(y + actual_block_size_y + search_region, h)

            # Loop inside the search region
            for yy in range(y_start, y_end - actual_block_size_y + 1):
                for xx in range(x_start, x_end - actual_block_size_x + 1):

                    block_target = target[y:y+actual_block_size_y, x:x+actual_block_size_x]
                    block_anchor = anchor[yy:yy+actual_block_size_y, xx:xx+actual_block_size_x]
                    
                    # calculate mean speed
                    mse = np.mean((block_target - block_anchor) ** 2)

                    if mse < min_error:
                        min_error = mse
                        best_match = (xx, yy)
            
            #save the motion vector
            motion_vectors.append(((x, y), (best_match[0] - x, best_match[1] - y)))

    return motion_vectors

target = cv2.imread('/Users/kasi/Downloads/nadia2.jpg', cv2.IMREAD_GRAYSCALE)
anchor = cv2.imread('/Users/kasi/Downloads/nadia3.jpg', cv2.IMREAD_GRAYSCALE)

block_size = 8
search_region = 7

#use ebma
motion_vectors = ebma(target, anchor, block_size, search_region)
rows, cols = target.shape
quiver_img = np.zeros((rows, cols, 3), dtype=np.uint8)

for (x, y), (dx, dy) in motion_vectors:
    cv2.arrowedLine(quiver_img, (x, y), (x + dx, y + dy), (0, 255, 0), 1, tipLength=0.3)

cv2.imshow('Motion Vector', quiver_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
