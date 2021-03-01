import numpy as np
import os
from opts_get_mask_coords import *


os.chdir(dirpath)

############################################
# Classes and Functions


class InputSMF:
    """
    A simple class to hold data from .SMF files.

    According to http://users.obs.carnegiescience.edu/clardy/imacs/smdfile.txt:
        x, y, width, xminus, and xplus are given in mm
        tilt is the degrees of counterclockwise rotation (on the sky)
        codes are 0=circle, 1=square, 2=rectangle, 3=special
    """

    def __init__(self, filename, skiplines=18):
        self.mask = filename
        self.type = np.array([])
        self.name = np.array([])
        self.ra = np.array([])
        self.dec = np.array([])
        self.x = np.array([])
        self.y = np.array([])
        self.width = np.array([])
        self.code = np.array([])
        self.xminus = np.array([])
        self.xplus = np.array([])
        self.tilt = np.array([])

        with open(filename) as f:
            for _ in range(skiplines):
                next(f)
            for line in f:
                splitted = line.split()
                if len(splitted) == 0:
                    break
                elif splitted[0] == "\x00!":
                    break
                self.type = np.append(self.type, splitted[0])
                self.name = np.append(self.name, splitted[1])
                self.ra = np.append(self.ra, splitted[2])
                self.dec = np.append(self.dec, splitted[3])
                self.x = np.append(self.x, splitted[4])
                self.y = np.append(self.y, splitted[5])
                self.width = np.append(self.width, splitted[6])
                if splitted[0] == "SLIT":
                    self.code = np.append(self.code, 2)
                    self.xminus = np.append(self.xminus, splitted[7])
                    self.xplus = np.append(self.xplus, splitted[8])
                    self.tilt = np.append(self.tilt, splitted[9])
                else:
                    self.code = np.append(self.code, float(splitted[7]))
                    self.xminus = np.append(self.xminus, splitted[8])
                    self.xplus = np.append(self.xplus, splitted[9])
                    self.tilt = np.append(self.tilt, splitted[10])
        self.x = self.x.astype(np.float)
        self.y = self.y.astype(np.float)
        self.width = self.width.astype(np.float)
        self.code = self.code.astype(np.int)
        self.xminus = self.xminus.astype(np.float)
        self.xplus = self.xplus.astype(np.float)
        self.tilt = self.tilt.astype(np.float)


'''
Old linear fits.  Better ones are in the following function.
def get_pixel(x_mm,y_mm):
    """
    Takes an x and y mm position from the .SMF file and returns the chip and pixel positions.
    """
    toprow = [1, 2, 3, 4]
    botrow = [6, 5, 8, 7]
    m = 14.5

    if y_mm >= 0:
        chips = toprow
        bs = [4137.5, 2043.8, -102.65, -2222.0]
        by = 4064.3
    elif y_mm < 0:
        chips = botrow
        bs = [4167., 2041., -85., -2211.]
        by = -53.3
    if x_mm <= -141:
        y_idx = 0
    elif -141 < x_mm <= 7:
        y_idx = 1
    elif 7 < x_mm <= 153:
        y_idx = 2
    elif 153 < x_mm:
        y_idx = 3
    chip = chips[y_idx]
    b = bs[y_idx]
        
    x_pix = m*x_mm + b
    y_pix = -m*y_mm + by
    
    return chip, x_pix, y_pix
    #return chip, x_pix, 2048
'''


def get_pixel(x_mm, y_mm):
    """
    Takes an x and y mm position from the .SMF file and returns the chip and pixel positions.
    """
    toprow = [1, 2, 3, 4]
    botrow = [6, 5, 8, 7]

    if y_mm >= 0:
        chips = toprow
        mxs = [
            14.390,
            14.553,
            14.538,
            14.407,
        ]  # m and b values found with linear fits to x positions
        bxs = [
            4150.8,
            2039.6,
            -99.267,
            -2216.9,
        ]  # m and b values for y position linear fit are consistent across chips
        my = -14.525
        by = 4113.3
    elif y_mm < 0:
        chips = botrow
        mxs = [14.552, 14.545, 14.558, 14.552]
        bxs = [4177.9, 2037.8, -102.30, -2242.4]
        my = -14.547
        by = -50.0
    if x_mm <= -141:
        y_idx = 0
    elif -141 < x_mm <= 7:
        y_idx = 1
    elif 7 < x_mm <= 153:
        y_idx = 2
    elif 153 < x_mm:
        y_idx = 3
    chip = chips[y_idx]
    mx = mxs[y_idx]
    bx = bxs[y_idx]

    x_pix = mx * x_mm + bx
    y_pix = my * y_mm + by

    return chip, x_pix, y_pix


vget_pixel = np.vectorize(get_pixel)

############################################
# Input Parameters

"""
binning = 2
mask_folder = '../'
SMFfile = 'hp26s.SMF'
output_file = 'hp26s_coords'
"""
############################################
# Main Code

SMF = InputSMF(f"{SMFfile}")
idx = np.where(SMF.type == "SLIT")[0]
chips, xs, ys = vget_pixel(SMF.x[idx], SMF.y[idx])

toprow = [1, 2, 3, 4]
botrow = [6, 5, 8, 7]

f = open(output_file, "w")
for i in range(len(SMF.name[idx])):
    obj = SMF.name[idx][i]
    chip = chips[i]
    x = xs[i] / binning
    y = ys[i] / binning
    if y > 4096 / binning - 10:  # Moving y positions away from chip gap slightly
        y = y - 20
    elif y < 10:
        y = y + 20
    line = (
        str(obj)
        + "_"
        + str(chip)
        + "\t"
        + "c"
        + str(chip)
        + "\t"
        + str(x)
        + "\t"
        + str(y)
        + "\n"
    )
    f.write(line)
    y2good = False  # Assume the spectrum doesn't stretch to the other chip
    if chip in toprow:
        chip2 = botrow[np.where(toprow == chip)[0][0]]
        if y >= 2048 / binning:
            y2good = True  # The slit is close enough to the center to spread the spectrum to the other chip.
            y2 = 20
    elif chip in botrow:
        chip2 = toprow[np.where(botrow == chip)[0][0]]
        if y <= 2048 / binning:
            y2good = True  # The slit is close enough to the center to spread the spectrum to the other chip.
            y2 = 4096 / binning - 20
    if y2good == True:
        line = (
            str(obj)
            + "_"
            + str(chip2)
            + "\t"
            + "c"
            + str(chip2)
            + "\t"
            + str(x)
            + "\t"
            + str(y2)
            + "\n"
        )
        f.write(line)
f.close()
