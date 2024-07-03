import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
plt.rcParams['animation.ffmpeg_path'] = '/Users/kasi/Downloads/ffmpeg'

def make_zone_plate(rows, cols, f):
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)
    radius_squared = X**2 + Y**2
    zone_plate = 0.5 + 0.5 * np.cos(np.pi * f * radius_squared)
    return zone_plate

def show_zone_plate(zone_plate):
    plt.imshow(zone_plate, cmap='gray')
    plt.title(f"Zone Plate (f = {f})")
    plt.axis('off')
    plt.show()


f_values = [0.5, 1.0, 1.5, 2.0, 2.5, 20, 50, 75, 100]

for f in f_values:
    rows, cols = 512, 512  
    zone_plate = make_zone_plate(rows, cols, f)
    show_zone_plate(zone_plate)

NROWS, NCOLS = 240, 320

def zone_plate(f, Nx, Ny):
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    X, Y = np.meshgrid(x, y)
    radius_squared = X**2 + Y**2
    zone_plate = 0.5 + 0.5 * np.cos(np.pi * f * radius_squared)
    return zone_plate

fig = plt.figure()
f = 1
def animate(i):
    image = zone_plate(f+1.0*i, NCOLS, NROWS)
    plt.imshow(image, cmap='gray')


ani = animation.FuncAnimation(fig, animate, frames=50)
FFwriter = animation.FFMpegWriter(codec='rawvideo')
ani.save('zoneplate.mov', writer=FFwriter)
