import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

cap = cv2.VideoCapture("/Users/kasi/Downloads/Foreman360p.mp4")

if (cap.isOpened()== False):
    print("Error opening file")

video = []

# Read 10 frames
for x in range(10):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret == True:
        video.append(frame)
        print(frame.shape, ' ', frame[x][x].shape)
cap.release()


# adding gausian noise
noisy_video = []
for frame in video:
    standard_dev = 25 
    mean = 0
    
    gaussian_noise = np.random.normal(mean, standard_dev, frame.shape).astype(np.uint8)
    noisy_frame = cv2.add(frame, gaussian_noise)
    noisy_video.append(noisy_frame)

noisy_video = np.array(noisy_video)


print("Noisy Video:", noisy_video.shape)


distorted_video = []
for frame in video:

    distorted_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    distorted_video.append(distorted_frame)

distorted_video = np.array(distorted_video)
print("Distorted Video:", distorted_video.shape)


original_mssim_distorted = compare_ssim(video[0], cv2.resize(distorted_video[0], (video[0].shape[1], video[0].shape[0]))) #compare ssim
original_mssim = compare_ssim(video[0], noisy_video[0]) #comapre mssim

print("MSSIM for Original vs. Distorted Video:", original_mssim_distorted)
print("MSSIM for Original vs. Noisy Video:", original_mssim)
