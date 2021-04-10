import dateutil
import numpy as np
import os, sys
from astropy.io import fits as pyfits
import pickle
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.signal import medfilt
from opts_check_transit_interactive_WASP107_ut210326 import *
import glob
from tqdm import tqdm

# from astropy.io import ascii

##DERIVED VALUES#############################

ing_time.append(dateutil.parser.parse(ing_time[0]))
ing_time.append(mdates.date2num(ing_time[1]))
mid_time.append(dateutil.parser.parse(mid_time[0]))
mid_time.append(mdates.date2num(mid_time[1]))
egr_time.append(dateutil.parser.parse(egr_time[0]))
egr_time.append(mdates.date2num(egr_time[1]))

ing_hr = ing_time[1].hour
ing_mn = ing_time[1].minute
mid_hr = mid_time[1].hour
mid_mn = mid_time[1].minute
egr_hr = egr_time[1].hour
egr_mn = egr_time[1].minute

date = mid_time[0][0:10]

delta = p ** 2
delta_min = (p - p_err) ** 2
delta_max = (p + p_err) ** 2

##CLASSES & FUNCTIONS##########################

# GrabNewdata = True
# if GrabNewdata:
#    os.system('rsync -avr obs2@llama.lco.cl:/Volumes/data_Llama/IMACS/ut140429/ .')


class InputCoords:
    """
    A simple class to hold coordinate data.
    """

    def __init__(self, filename, skiplines=0):
        self.fname = filename
        self.obj = np.array([])
        self.chip = np.array([])
        self.x = np.array([])
        self.y = np.array([])

        print(filename)
        with open(filename) as f:
            for _ in range(skiplines):
                next(f)
            for line in f:
                splitted = line.split()
                if len(splitted) == 0:
                    break
                self.obj = np.append(self.obj, splitted[0])
                self.chip = np.append(self.chip, splitted[1][-1])
                self.x = np.append(self.x, splitted[2])
                self.y = np.append(self.y, splitted[3])
        self.chip = self.chip.astype(np.int)
        self.x = self.x.astype(np.float)
        self.y = self.y.astype(np.float)


def biassec(x):
    x = x.split(",")
    fnum = ((x[0])[1:]).split(":")
    snum = ((x[1])[: len(x[1]) - 1]).split(":")
    fnum[0] = int(fnum[0])
    fnum[1] = int(fnum[1])
    snum[0] = int(snum[0])
    snum[1] = int(snum[1])
    return fnum, snum


def zero_oscan(d):
    zrows = len(np.where(d[:, 0] == 0)[0])
    zcols = len(np.where(d[0, :] == 0)[0])
    if zrows > zcols:
        mode = "r"
        length = d.shape[1]
    else:
        mode = "c"
        length = d.shape[0]
    cmed = []
    for i in range(length):
        if mode == "r":
            data = d[:, i]
        else:
            data = d[i, :]
        I = np.where(data != 0)[0]
        cmed.append(np.median(data[I]))
    return np.median(np.array(cmed))


def BiasTrim(d, c, h, otype, datasec=None):
    """
    Overscan/Bias correct and Trim an IMACS chip
    """
    oxsec, oysec = biassec(h["biassec"])
    if datasec == None:
        dxsec, dysec = biassec(h["datasec"])
    else:
        dxsec, dysec = biassec(datasec)
    if otype == "ift":
        oscan_data = d[(oysec[0] - 1) : oysec[1], (oxsec[0] - 1) : oxsec[1]]
        overscan = np.median(oscan_data)
        if overscan == 0:
            overscan = zero_oscan(oscan_data)
        newdata = d[(dysec[0] - 1) : dysec[1], (dxsec[0] - 1) : dxsec[1]] - overscan
    else:
        d = d.transpose()
        oscan_data = d[oxsec[0] - 1 : oxsec[1], oysec[0] - 1 : oysec[1]]
        overscan = np.median(oscan_data)
        if overscan == 0:
            overscan = zero_oscan(oscan_data)
        newdata = d[dxsec[0] - 1 : dxsec[1], dysec[0] - 1 : dysec[1]] - overscan
    if (c == "c5") or (c == "c6") or (c == "c7") or (c == "c8"):
        if otype == "iff":
            newdata = newdata[::-1, :]
        else:
            newdata = newdata[::-1, ::-1]
    return newdata


class baseObject:
    """
    Empty object container.
    """

    # 2010-01-24 15:13 IJC: Added to spitzer.py (from my ir.py)
    def __init__(self):
        return

    def __class__(self):
        return "baseObject"


def dict2obj(dic):
    """
    Take an input Dict, and turn it into an object with fields
    corresponding to the dict's keys.
    """
    # 2011-02-17 09:41 IJC: Created
    ret = baseObject()
    names = []
    if not isinstance(dic, dict):
        raise Exception("Input was not a Python dict!  Exiting.")
    else:
        for key in dic.keys():
            exec('ret.%s = dic["%s"]' % (key, key))
            names.append(key)
        ret.names = names
    return ret


def loadpickle(filename):
    fin = open(filename, "r")
    pkl = pickle.load(fin)
    fin.close()
    return pkl


def loadpickle_as_class(filename):
    fin = open(filename, "rb")
    pkl = pickle.load(fin)
    fin.close()
    return dict2obj(pkl)


def savepickle(obj, filename):
    """
    Save a pickle to a given filename.  If it can't be saved by
    pickle, return -1 -- otherwise return the file object.

    To save multiple objects in one file, use (e.g.) a dict:

       tools.savepickle(dict(a=[1,2], b='eggs'), filename)
    """
    # 2011-05-21 11:22 IJMC: Created from loadpickle.
    # 2011-05-28 09:36 IJMC: Added dict example
    try:
        f = open(filename, "wb")
    except:
        raise Exception("Could not open file: {filename} : for writing.")
    try:
        ret = pickle.dump(obj, f)
    except:
        raise Exception("Could not write object to pickle file: {filename}")
    try:
        f.close()
    except:
        raise Exception("Could not close pickle file {filename}")
    return f


def get_totflux(image, basewindow, otype):
    flux = 0.0
    if otype == "iff":
        for i in range(image.shape[1]):
            base = medfilt(image[:, i], basewindow)
            flux = flux + np.sum(image[:, i] - base)
    else:
        for i in range(image.shape[0]):
            base = medfilt(image[i, :], basewindow)
            flux = flux + np.sum(image[i, :] - base)
    return flux


##INITIALIZING########################################

print("\n###########################################################")
print("\n              IMACS Interactive Transit Plotter\n")
print("  authors:  Nestor Espinoza (nespino@astro.puc.cl)")
print("            Jonathan Fraine (jfraine@astro.umd.edu)")
print("            Benjamin Rackham (brackham@as.arizona.edu)")
print("\n###########################################################")

try:
    coords = InputCoords(f"{filedir}/{coords_file}")
except:
    sys.exit("\nCouldn't find coordinates file. Run get_mask_coords.py to make one.")

if Interactive:
    print("\nInteractive mode enabled.")

PlotExists = False
Updating = True
while Updating:
    # Listing all files in directory
    # dirlist = sorted(glob.glob(f"{filedir}/*")) #os.listdir(filedir)
    dirlist = os.listdir(filedir)

    # Finding all frames
    all_frames = []
    for ifile in sorted(dirlist):
        if "c1.fits" in ifile and len(ifile) == 14:  # len('ift0001c1.fits') = 14
            all_frames.append(ifile)

    # Finding all science frames
    science_frames = []
    for f in all_frames:
        object = pyfits.getval(f"{filedir}/{f}", "OBJECT")
        if object == science_object:
            science_frames.append(f[0:7])

    # Truncating the list if first_frame, last_frame given
    try:
        s1 = (
            i for i, f in enumerate(science_frames) if str(first_frame).zfill(4) in f
        ).next()
    except:
        s1 = 0
    try:
        s2 = (
            len(science_frames)
            - (
                i
                for i, f in enumerate(science_frames[::-1])
                if str(last_frame).zfill(4) in f
            ).next()
        )
    except:
        s2 = len(science_frames)

    science_frames = np.array(science_frames[s1:s2])
    nFrames = len(science_frames)
    otype = science_frames[0][0:3]
    if otype == "ift":
        mode = "f/2"
    elif otype == "iff":
        mode = "f/4"

    print(
        "\nFound science frames "
        + science_frames[0]
        + " to "
        + science_frames[-1]
        + " : "
        + str(nFrames)
        + " frames total."
    )

    pickle_dict = None
    for ifile in dirlist:
        if ifile == pklname:
            pickle_dict = loadpickle_as_class(f"{filedir}/{pklname}")

    if pickle_dict != None:
        print("\nStarting from save file:", pklname, "\n")
        time = pickle_dict.time
        fluxes = pickle_dict.fluxes
        flux_ratio = pickle_dict.flux_ratio

    else:
        print("\nStarting from nothing.\n")
        time = np.array([])
        fluxes = {obj: np.array([]) for obj in coords.obj}
        flux_ratio = np.array([])

    ##CALCULATING FLUX#################################

    idx_start = time.size

    if idx_start == nFrames:
        print("No more data to add!")
        data_added = False
    else:
        data_added = True
        n_added = 0
        chips = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
        for frame in tqdm(science_frames[idx_start:]):
        #for frame in tqdm(science_frames[idx_start:][np.array([1, 150, 300, 363])]):
            if Interactive:
                if n_added >= n_to_add:
                    break
            n_added += 1
            #print("Working on frame", frame)
            for chip in chips:
                try:
                    d, h = pyfits.getdata(f"{filedir}/{frame}{chip}.fits", header=True)
                except:
                    continue
                d = BiasTrim(d, chip, h, otype)
                if chip == chips[0]:
                    etime = h["EXPTIME"]
                    ut_time_start = dateutil.parser.parse(
                        h["UT-DATE"] + "T" + h["UT-TIME"]
                    )
                    ut_time_end = dateutil.parser.parse(h["UT-DATE"] + "T" + h["UT-END"])
                    if ut_time_end < ut_time_start:
                        ut_time_end = ut_time_end + timedelta(days=1)
                    time = np.append(
                        time, ut_time_start + (ut_time_end - ut_time_start) / 2
                    )
                for c in range(len(coords.chip)):
                    if str(coords.chip[c]) in chip:
                        obj = coords.obj[c]
                        x = int(coords.x[c])
                        slice = d[:, int(x - slitwidth / 2) : int(x + slitwidth / 2)]
                        flux = get_totflux(slice / etime, basewindow, otype)
                        fluxes[obj] = np.append(fluxes[obj], flux)
    # Summing all reference stars
    tflux, cflux = np.zeros(len(time)), np.zeros(len(time))
    for key in fluxes.keys():
        if target in key:
            tflux += fluxes[key]
        else:
            if (key not in bad_objs) and (key.split("_")[0] not in bad_comps):
                cflux += fluxes[key]
    if np.max(tflux) == 0:
        print("Can't identify target.  Check target name vs. name in coordinates file.")
    flux_ratio = tflux / cflux

    numtime = mdates.date2num(time)
    idx_oot = np.append(
        np.where((time < ing_time[1]))[0], np.where((time > egr_time[1]))[0]
    )

    if len(idx_oot) > 0:
        oot_flux = flux_ratio[idx_oot]
    else:
        oot_flux = flux_ratio[0]

    if data_added:
        print("\nSaving pickle with new data added.")
        pickle_out = {"time": time, "fluxes": fluxes, "flux_ratio": flux_ratio}
        print(f"{filedir}/{pklname}")
        savepickle(pickle_out, f"{filedir}/{pklname}")
    # print flux_ratio,np.median(oot_flux)
    print(flux_ratio, oot_flux)
    flux_ratio = flux_ratio / np.median(oot_flux)

    print("\n\tTotal number of frames:", len(time))
    print(
        "\tTotal time coverage:   ", round(24 * (max(numtime) - min(numtime)), 2), "hours"
    )

    ##PLOT######################################

    # plot window
    xmin = min(min(numtime) - 0.2 / 24, ing_time[2] - 1.5 / 24)
    xmax = max(max(numtime) + 0.2 / 24, egr_time[2] + 1.5 / 24)
    ymin = 0.971
    ymax = 1.02  # max ( 1.01, max(flux_ratio)+0.001)

    if Interactive:
        plt.ion()

    if not PlotExists:
        # Make the plot
        fig = plt.figure(1, figsize=(16, 8), facecolor="w", edgecolor="k")
        ax = fig.add_subplot(111)
        plt.rc("figure.subplot", top=0.85)
        ax.set_title(
            "Target: "
            + str(target)
            + " | UT Date: "
            + date
            + " | Mode: "
            + mode
            + " | Slit width: "
            + str(slitwidth)
            + " pixels | Base window: "
            + str(basewindow)
            + " pixels | No wav. cal"
        )
        ax.set_xlabel("UT Time")
        ax.set_ylabel("Normalized total flux ratio")
        ax.grid()

        # expected depth
        ax.plot_date(
            [min(numtime) - 0.5, max(numtime) + 0.5],
            [1.0 - delta, 1.0 - delta],
            "g-",
            label="Expected transit depth " + source,
            linewidth=5,
        )
        ax.plot_date(
            [min(numtime) - 0.5, max(numtime) + 0.5],
            [1.0 - delta_min, 1.0 - delta_min],
            "g--",
            label="68% credibility interval",
        )
        ax.plot_date(
            [min(numtime) - 0.5, max(numtime) + 0.5],
            [1.0 - delta_max, 1.0 - delta_max],
            "g--",
        )
        ax.fill_between(
            [min(numtime) - 0.5, max(numtime) + 0.5],
            1.0 - delta_min,
            1.0 - delta_max,
            facecolor="green",
            interpolate=True,
            alpha=0.1,
        )

        # expected times
        ax.plot_date(
            [ing_time[2], ing_time[2]],
            [min(flux_ratio) - 0.1, max(flux_ratio) + 0.1],
            "r--",
            label="Expected ingress ("
            + str(int(ing_hr)).zfill(2)
            + ":"
            + str(int(ing_mn)).zfill(2)
            + " UT) / egress ("
            + str(int(egr_hr)).zfill(2)
            + ":"
            + str(int(egr_mn)).zfill(2)
            + " UT) times",
        )
        ax.plot_date(
            [egr_time[2], egr_time[2]],
            [min(flux_ratio) - 0.1, max(flux_ratio) + 0.1],
            "r--",
        )
        ax.plot_date(
            [mid_time[2], mid_time[2]],
            [min(flux_ratio) - 0.1, max(flux_ratio) + 0.1],
            "r-",
            label="Expected mid-transit time ("
            + str(int(mid_hr)).zfill(2)
            + ":"
            + str(int(mid_mn)).zfill(2)
            + " UT)",
        )

        # data
        (line,) = ax.plot_date(numtime, flux_ratio, "b-", alpha=0.4)

        # write to dat file
        # ascii.write([numtime, flux_ratio], '/Users/mango/Desktop/LC.dat', names=['time', 'flux'])

        (medline,) = ax.plot_date(
            numtime[int(medfilter / 4) : int(-medfilter / 4)],
            medfilt(flux_ratio, medfilter)[int(medfilter / 4) : int(-medfilter / 4)],
            "b-",
            linewidth=2,
            label="Median-filtered data (" + str(medfilter) + " time-sample window)",
        )
        (points,) = ax.plot_date(
            numtime,
            flux_ratio,
            "wo-",
            markeredgecolor="b",
            label="Flux ratio between " + str(target) + " and Comparison Stars",
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        # ax.legend(bbox_to_anchor=(0.01, 0.02, 0.98, .102), loc=3,
        #                  ncol=2, mode="expand", borderaxespad=0.)
        ax.legend(loc="best", ncol=2, mode="expand", borderaxespad=0.0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        if not Interactive:
            print(filedir, target)
            plt.savefig(f"{filedir}/{target}.png", dpi=250)
        plt.show(block=not Interactive)
        PlotExists = True

    elif PlotExists:
        # Updating existing plot
        line.set_xdata(numtime)
        line.set_ydata(flux_ratio)
        medline.set_xdata(numtime[medfilter / 4 : -medfilter / 4])
        medline.set_ydata(medfilt(flux_ratio, medfilter)[medfilter / 4 : -medfilter / 4])
        points.set_xdata(numtime)
        points.set_ydata(flux_ratio)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.draw()

    if Interactive:
        user_response = input("\nEnter 'y' to update, or any other key to exit:\n\n>  ")
        if "y" not in user_response.lower():
            Updating = False
            print("\nExiting.\n")
            print("###########################################################\n")
        else:
            print("\n#####  Updating plot. #####")
    else:
        Updating = False
