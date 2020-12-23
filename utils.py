import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import juliet
import seaborn as sns
import mplcyberpunk
import numpy as np
import scipy as sp
import pandas as pd
import glob
import json
import os
import pickle
import corner
import pathlib
import itertools
import batman
import matplotlib.patheffects as PathEffects
import sys
import re
import george

from astropy.io import fits, ascii
from astropy.time import Time
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import AxesGrid
from datetime import datetime, timedelta
from dateutil import parser

def weighted_err(errs):
    weights = 1/errs**2
    partials = weights / np.sum(weights, axis=0)
    deltas = np.sqrt( np.sum( (partials*errs)**2, axis=0 ) )
    return deltas

def savefig(fpath):
    pathlib.Path(fpath).parents[0].mkdir(parents=True, exist_ok=True)
    plt.savefig(fpath, bbox_inches="tight")

def savepng(fpath):
    plt.savefig(fpath, dpi=250, bbox_inches="tight")

# pickle loader convenience function
def pkl_load(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f, encoding='latin') # Python 2 -> 3
    return data

# fits loader convenience function
def fits_open(fpath):
    with open(fpath, "rb") as f:
        hdu = fits.open(f)
        return hdu

def fits_data(fpath):
    with open(fpath, 'rb') as f:
        data = fits.open(f)[0].data
        return data

def fits_header(fpath):
    with open(fpath, 'rb') as f:
        header = fits.open(f)[0].header
        return header
def write_latex_row(row):
    v, vu, vd = row
    return f'{v:.3f}^{{+{vu:.3f}}}_{{-{vd:.3f}}}'
def write_latex_wav(row):
    wav_d, wav_u = row
    return f'{wav_d:.1f} - {wav_u:.1f}'

def write_latex_single(row):
    v, v_unc = row
    return f'{v:.3f} \pm {v_unc:.3f}'

def write_latex(p=None, df=None, v=None, vu=None, vd=None):
    if v is not None:
        pass
    else:
        v, vu, vd = df.loc[p]
    return f'{v:.5f}^{{+{vu:.5f}}}_{{-{vd:.5f}}}'

def write_latex2(val, unc):
    return f'{val:.2f} \pm {unc:.2f}'

def read_table(fpath, usecols=None, sep=None):
    engine = 'python' if sep is None else None
    df = pd.read_table(
        fpath,
        usecols=usecols,
        escapechar='#',
        sep=sep,
        engine=engine,
    )
    return df

def get_phases(t,P,t0):
    """
    Given input times, a period (or posterior dist of periods)
    and time of transit center (or posterior), returns the
    phase at each time t. From juliet =]
    """
    if type(t) is not float:
        phase = ((t - np.median(t0))/np.median(P)) % 1
        ii = np.where(phase>=0.5)[0]
        phase[ii] = phase[ii]-1.0
    else:
        phase = ((t - np.median(t0))/np.median(P)) % 1
        if phase>=0.5:
            phase = phase - 1.0
    return phase

def plot_subaperture(fpath):
    data = fits_data(fpath)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)

    im = axes[0].imshow(data, cmap='CMRmap')
    plt.colorbar(im, ax=axes[0])
    axes[0].grid(False) # get rid of gridlines on imshow
    #axes.set_title(name + ' -- sci frame 25')
    #axes[0].set_xlabel('%s - %s' % (ut_start, ut_end))
    #axes[0].set_xlim(600, 80)
    #axes[1].set_ylim(0, 25000)

    def drag(event): # mouse drag
        x = event.xdata
        y = event.ydata

        if (event.inaxes == axes[0]): # only detect mouse drags in imshow plot
            axes[1].cla()
            y = int(y)
            y_upper = y - 1
            y_lower = y + 1
            axes[1].plot(data[y_upper, :], label=f"x = {y_upper}")
            axes[1].plot(data[y, :], label=f"x = {y}")
            axes[1].plot(data[y_lower, :], label=f"x = {y_lower}")
            #axes[1].set_xlim(x - 50, x + 50)
            #axes[1].set_ylim(0, 30000)
            axes[1].legend(loc=1)

        #fig.tight_layout()
        fig.canvas.draw()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(fpath)
    cid = fig.canvas.mpl_connect('motion_notify_event', drag)

    return fig, axes, data

def plot_traces(dirpath, start=0, end=-2):
    XX = pkl_load(f'{dirpath}/XX.pkl')
    YY = pkl_load(f'{dirpath}/YY.pkl')

    # plot
    ncols = len(XX.keys())
    fig, axes = plt.subplots(1, ncols, figsize=(16, 6))

    # sort keys
    objs = sorted(XX.keys())

    for ax, obj in zip(axes.flatten(), objs):
        XX_obj, YY_obj = XX[obj][start:end+1], YY[obj][start:end+1]
        traces =  zip(XX_obj, YY_obj)
        segs = (list(zip(*trace)) for trace in traces)

        lines = LineCollection(segs, cmap='plasma')
        lines.set_array(np.arange(len(XX_obj))) # Color by time index

        ax.add_collection(lines)

        xmin = min([min(xs) for xs in XX_obj])
        xmax = max([max(xs) for xs in XX_obj])
        ymin = min([min(ys) for ys in YY_obj])
        ymax = max([max(ys) for ys in YY_obj])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(obj)

    axcb = fig.colorbar(lines, cmap='Blues', label='index')
    fig.suptitle(dirpath, y=1.01)
    fig.tight_layout()

    return fig, axes, XX, YY

def get_1D_spec(data, x):
    aperture = slice(x-25, x+25)
    peak_fluxes = []
    for row in data[None:None:-1, aperture]:
        peak_fluxes.append(np.max(row))

    return peak_fluxes

def plot_aperture(fpath):
    header = fits_header(fpath)
    data = fits_data(fpath)
    name = header['filename']
    ut_start = header['UT-TIME']
    ut_end = header['UT-END']

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    im = axes[0].imshow(data, cmap='CMRmap', vmin=0, vmax=2000)
    plt.colorbar(im, ax=axes[0])
    axes[0].grid(False) # get rid of gridlines on imshow
    #axes.set_title(name + ' -- sci frame 25')
    axes[0].set_xlabel('%s - %s' % (ut_start, ut_end))
    #axes[0].set_xlim(600, 80)
    #axes[1].set_ylim(0, 25000)

    def drag(event): # mouse drag
        x = event.xdata
        y = event.ydata

        if (event.inaxes == axes[0]): # only detect mouse drags in imshow plot
            axes[1].cla()
            axes[1].plot(data[int(y), :], lw=2)
            axes[1].set_xlim(x - 50, x + 50)
            axes[1].set_ylim(0, 30000)
            axes[1].set_ylabel('counts')

            axes[2].cla()
            axes[2].plot(get_1D_spec(data, int(x)), lw=2)

        #fig.tight_layout()
        fig.canvas.draw()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(fpath)
    cid = fig.canvas.mpl_connect('motion_notify_event', drag)

    return fig, axes, data

def plot_chips(dirpath, fpathame, vmin=0, vmax=2_000, spec_ap=0, sky_ap=0):
    # This plots the chips by numbers:
    #
    # 1 2 3 4
    # 6 5 8 7
    #
    class CoordinateData:
        """
        A simple class to hold coordinate data.
        """
        def __init__(self, filename, skiplines=0):
            self.fpathame = filename
            self.obj = np.array([])
            self.chip = np.array([])
            self.x = np.array([])
            self.y = np.array([])

            with open(filename) as f:
                for _ in range(skiplines):
                    next(f)
                for line in f:
                    splitted = line.split()
                    if(len(splitted)==0):
                        break
                    self.obj = np.append(self.obj, splitted[0])
                    self.chip = np.append(self.chip, splitted[1][-1])
                    self.x = np.append(self.x, splitted[2])
                    self.y = np.append(self.y, splitted[3])
            self.chip = self.chip.astype(np.int)
            self.x = self.x.astype(np.float)
            self.y = self.y.astype(np.float)

    target = dirpath.split('/')[7]
    coords_file = f"{dirpath}/{target}.coords"
    coords = CoordinateData(coords_file)

    def biassec(x):
        x = x.split(',')
        fpathum = ((x[0])[1:]).split(':')
        snum = ((x[1])[:len(x[1])-1]).split(':')
        fpathum[0] = int(fpathum[0])
        fpathum[1] = int(fpathum[1])
        snum[0] = int(snum[0])
        snum[1] = int(snum[1])
        return fpathum,snum

    def zero_oscan(d):
        zrows = len(np.where(d[:,0]==0)[0])
        zcols = len(np.where(d[0,:]==0)[0])
        if(zrows>zcols):
           mode = 'r'
           length = d.shape[1]
        else:
           mode = 'c'
           length = d.shape[0]
        cmed = []
        for i in range(length):
            if(mode == 'r'):
               data = d[:,i]
            else:
               data = d[i,:]
            I = np.where(data!=0)[0]
            cmed.append(np.median(data[I]))
        return np.median(np.array(cmed))

    def BiasTrim(d,c,h,otype,datasec=None):
        """
        Overscan/Bias correct and Trim an IMACS chip
        """
        # bias has no significant structure, so a single median suffices, I think
        # overscan = [0:49] [2097:2145]
        oxsec,oysec = biassec(h['biassec'])
        if(datasec == None):
           dxsec,dysec = biassec(h['datasec'])
        else:
           dxsec,dysec = biassec(datasec)
        if(otype=='ift'):
           oscan_data = d[(oysec[0]-1):oysec[1],(oxsec[0]-1):oxsec[1]]
           overscan = np.median(oscan_data)
           if(overscan == 0):
              overscan = zero_oscan(oscan_data)
           newdata = d[(dysec[0]-1):dysec[1],(dxsec[0]-1):dxsec[1]] - overscan
        else:
           d = d.transpose()
           oscan_data = d[oxsec[0]-1:oxsec[1],oysec[0]-1:oysec[1]]
           overscan = np.median(oscan_data)
           if(overscan == 0):
              overscan = zero_oscan(oscan_data)
           newdata = d[dxsec[0]-1:dxsec[1],dysec[0]-1:dysec[1]] - overscan
        #overscan = np.median(d[:,2048:2112])
        #newdata = d[:4095,0:2048] - overscan
        if ((c == 'c5') or (c == 'c6') or (c == 'c7') or (c == 'c8')):
           if(otype == 'iff'):
              newdata = newdata[::-1,:]
           else:
              newdata = newdata[::-1,::-1]

        return newdata

    ############
    # Plot chips
    ############
    image = fpathame
    otype = image[0:3]
    chips = ['c1','c2','c3','c4','c6','c5','c8','c7']
    order = [1, 2, 3, 4, 6, 5, 8, 7]
    ranges = [vmin, vmax]

    fig = plt.figure(figsize=(8, 11))
    grid = AxesGrid(
        fig,
        111,
        nrows_ncols=(2, 4),
        axes_pad=0.05,
        cbar_mode="single",
        cbar_location="right",
        cbar_pad=0.1,
    )

    #fig, axes = plt.subplots(
    #        2, 4,
    #        sharex=True, sharey=True,
    #        figsize=(8, 11),
    #        )
    #images = []
    for i, ax in enumerate(grid):
        #print('Chip number '+(chips[i])[1:]+' overscan:')
        #ax.plot(int('24'+str(i+1)))
        #print(image, chips)
        d,h = fits.getdata(f"{dirpath}/{image}{chips[i]}.fits", header=True)
        d = BiasTrim(d,chips[i],h,'ift')
        im = ax.imshow(d, vmin=ranges[0], vmax=ranges[1], cmap="magma")
        #images.append(im)
        #ax.plot([1,2,3])
        ax.xaxis.set_ticks(np.arange(250, 1_000, 250))
        ax.yaxis.set_ticks(np.arange(250, 2_000, 250))
        ax.tick_params(axis='both', which='major', labelsize=10)
        va = 'bottom'
        if chips[i] in ['c6', 'c5', 'c8', 'c7']:
            va = 'top'
        for c in range(len(coords.chip)):
            if order[i]==coords.chip[c]:
                col = 'w'
                if target in coords.obj[c]: col='y'
                #ax.scatter(coords.x[c], coords.y[c], s=100, marker='+', color='r')
                #ax.axvline(coords.x[c], color='C2', lw=0.3)
                ax.axvline(coords.x[c] - sky_ap, color='g', lw=1)
                #if 'comp7' not in coords.obj[c]:
                ax.axvline(coords.x[c] + sky_ap, color='g', lw=1)
                txt = ax.text(coords.x[c], coords.y[c], coords.obj[c], color=col,
                        ha="right", va=va, fontsize=14, rotation=90)
                txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='k')])
        #print('Shape:',d.shape)
        y = 1.05
        if i > 3: y=-0.2
        ax.set_title('Chip '+(chips[i])[1:], fontsize=12, y=y)
        #im.set_clim(ranges[0],ranges[1])
        #im.set_clim(0, 40_000)
        ax.grid(False)
        date_time = f"{h['DATE-OBS']} -- {h['TIME-OBS']}"
        #fig.text(0.1, 0.5, date_time, ha='center', va='center',
        #         rotation='vertical', fontsize=12)
        #fig.suptitle(date_time, y=1.01, fontsize=16)

    # Find the min and max of all colors for use in setting the color scale.
    #vmin = min(image.get_array().min() for image in images)
    #vmax = max(image.get_array().max() for image in images)
    #norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    #for im in images:
    #    im.set_norm(norm)
    #fig.colorbar(images[0], ax=axes, aspect=40)
    #fig.text(0.9, 0.5, date_time, ha='center', va='center',
    #         rotation='vertical', fontsize=12)
    #cbar.set_title("hey")
    #plt.subplots_adjust(hspace=0.03, wspace=0.02)
    # Must be in this order!!!
    plt.axis('off')
    cbar = ax.cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)
    #plt.subplots_adjust(wspace=0, hspace=0)
    #fig.tight_layout()
    #fig.colorbar(im, ax=axes.ravel().tolist(), aspect=30, label='counts')
    return fig, im

def _to_arr(idx_or_slc):
    # Converts str to 1d numpy array
    # or slice to numpy array of ints.
    # This format makes it easier for flattening multiple arrays in `_bad_idxs`
    if ':' in idx_or_slc:
        lower, upper = map(int, idx_or_slc.split(':'))
        return np.arange(lower, upper+1)
    else:
        return np.array([int(idx_or_slc)])

def _bad_idxs(s):
    if s == '[]':
        return []
    else:
        # Merges indices/idxs specified in `s` into a single numpy array of
        # indices to omit
        s = s.strip("[]").split(',')
        bad_idxs = list(map(_to_arr, s))
        bad_idxs = np.concatenate(bad_idxs, axis=0)
        return bad_idxs

def _plot_arc(ax, fluxes, wavs=None, pixels=None, c="C0", ann_c="lightgrey"):
    # Plot arc flux
    #fluxes = fluxes/np.max(fluxes)
    idxs = np.arange(0, len(fluxes))
    ax.plot(idxs, fluxes, c=c)

    # Annotate lines if provided
    if (pixels is not None) and (wavs is not None):
        ys = []
        for pixel in pixels:
            ys.append(fluxes[int(pixel)])
        for wav, pixel, y in zip(wavs, pixels, ys):
            ax.annotate(
                f"{wav:.4f}", (pixel, y),
                fontsize=10, alpha=0.5, picker=1, color=ann_c,
            )
            ax.axvline(pixel, lw=0.5, ls = '--', c='r', alpha=0.5)

    ax.set_xlabel("Pixel")
    ax.set_ylabel("Normalized flux")
    #ax.set_ylim(-0.01, 1.1)

    return ax

def compare_arc_lines(
        fpath_arc_ref=None, fpath_lines_ref=None,
        fpath_arc=None, fpath_lines=None,
        sharex=False, sharey=False
    ):
    ################################
    # Plot reference and target arcs
    ################################
    fig, axes = plt.subplots(
                    2, 1, figsize=(10, 10),
                    sharex=sharex, sharey=sharey,
                )
    ax_ref, ax_lco = axes

    # Check if reference lines specified
    if fpath_lines_ref is not None:
        if f"noao" in fpath_lines_ref.lower():
            col_name_pixel = "Pixel"
            col_name_wav = "User"
        else:
            col_name_pixel = "Pix"
            col_name_wav = "Wav"

        df_ref = pd.read_csv(
            fpath_lines_ref,
            #usecols=["Wav", "Pix", "Chip"],
            escapechar='#',
        )
        pixels = df_ref[col_name_pixel].to_numpy().flatten()
        wavs = df_ref[col_name_wav].to_numpy().flatten()
    else:
        pixels, wavs = None, None

    # Plot reference flux in top panel
    arc_flux_ref = fits_data(fpath_arc_ref)
    _plot_arc(
        ax_ref,
        arc_flux_ref,
        pixels=pixels,
        wavs=wavs,
        c='y',
    )
    path = pathlib.PurePath(fpath_arc_ref)
    parents = path.parents
    fname = path.name
    title = f"{parents[1].name}/{parents[0].name}/{fname}"
    ax_ref.set_title(title)

    # Plot target lines in bottom panel
    arc_flux_lco = fits_data(fpath_arc)

    # Create guesses file if not done already
    if not os.path.exists(fpath_lines):
        data = {'#Wav':[], "Pix":[], "Chip":[]}
        df_guesses = pd.DataFrame(data)
        df_guesses.to_csv(fpath_lines, index=False)

    print(fpath_lines)
    df_lco = pd.read_csv(
        fpath_lines,
        usecols=["Wav", "Pix", "Chip"],
        escapechar='#',
    )
    wavs = df_lco["Wav"].to_numpy().flatten()
    pixels = df_lco["Pix"].to_numpy().flatten()
    _plot_arc(
        ax_lco,
        arc_flux_lco,
        wavs=wavs,
        pixels=pixels,
        c='b',
        ann_c='g',
    )
    # Highlight wavelengths in top panel if already selected in bottom
    annotations = [
        child for child in ax_ref.get_children()
        if isinstance(child, plt.matplotlib.text.Annotation)
    ]
    for ann in annotations:
        if float(ann.get_text()) in wavs:
            ann.set_color('g')

    path = pathlib.PurePath(fpath_arc)
    parents = path.parents
    fname = path.name
    title = f"{parents[1].name}/{parents[0].name}/{fname}"
    ax_lco.set_title(title)

    ########################
    # Record wav/pixel pairs
    ########################
    wavs, pixels = [], []
    def on_click(event, cid, x, y):
        # Highlights selected annotation and stores the associated wavelength value
        ann = event.artist
        ann.set_color("g")
        wav = ann.get_text()
        wavs.append(float(wav))
        # Annotes the wavelength value on bottom panel for ease of reference
        ax_lco.annotate(wav, xy=(x+1, y+0.02), fontsize=10, alpha=0.5, c='g')
        fig.canvas.draw()
        fig.canvas.mpl_disconnect(cid) # Only run once per `X` press on keyboard

    def on_key(event):
        # Display and record pixel location on 'X' key press on bottom panel
        if (event.key == 'x' or event.key == 'X') and (event.inaxes == ax_lco):
            pixel = event.xdata
            pixels.append(pixel)
            y = event.ydata
            ax_lco.axvline(pixel, c='r', ls='--', lw=0.5)
            ax_lco.plot(pixel, y, 'rX')

            # Collect wavelength info after mouse press
            cid0 = fig.canvas.mpl_connect(
                       'pick_event',
                        lambda event: on_click(event, cid0, pixel, y),
                   )
            fig.canvas.draw()

    cid1 = fig.canvas.mpl_connect('key_press_event', on_key)

    fig.tight_layout()

    return wavs, pixels

def plot_pix_wav(ax, df, x_name, y_name, comp_name):
    groups = df.groupby("Chip")

    x_interp = np.linspace(0, 2048, 1000)
    for chip, group in groups:
        label = f"{comp_name}_{chip}"
        x, y = group[x_name], group[y_name]
        m, b = np.polyfit(x, y, 1)
        p = ax.plot(x, y, '.', lw=0.5, label=f"{label}, m={m:.3f}")
        c = p[0].get_color()
        ax.plot(x_interp, m*x_interp + b, c=c, lw=0.5)

def _plot_spec_file(ax, fpath=None, data=None, wavs=None, i=1, label=None,
        median_kwargs=None):
    """
    plots items in <object>_spec.fits files.
    has shape time x spec_item x ypix [it's ~ 2048] (|| to wav direction)
    `i`:
    0: Wavelength
    1: Simple extracted object spectrum
    2: Simple extracted flat spectrum
    3: Pixel sensitivity (obtained by the flat)
    4: Simple extracted object spectrum/pixel sensitivity
    5: Sky flag (0 = note uneven sky, 1 = probably uneven profile,
       2 = Sky_Base failed)
    6: Optimally extracted object spectrum
    7: Optimally extracted object spectrum/pixel sensitivity
    """
    if median_kwargs is None:
        median_kwargs = {}

    if fpath is not None:
        data = fits_data(fpath)
        specs = data[:, i, :]
        wavs = range(specs.shape[1]) # TODO, switch to wavlength if desired
    else:
        specs = data
        wavs = wavs

    specs[specs <= 0] = np.nan # sanitize data
    specs_med = np.nanmedian(specs, axis=0)
    specs_std = np.nanstd(specs, axis=0)
    factor = np.nanmedian(specs_med)
    specs_med #/= factor
    specs_std #/= factor


    # empty plot for custom legend
    p = ax.plot([], '-o', label=label, **median_kwargs)

    # plot normalized median and 1-sigma region
    specs_med #/= np.nanmax(specs_med)
    specs_std #/= np.nanmax(specs_med)
    c = p[0].get_color()
    ax.fill_between(wavs, specs_med - specs_std, specs_med + specs_std, color=c, alpha=0.25, lw=0)
    ax.plot(wavs, specs_med, lw=2, color=c)
    return ax, wavs, data

def plot_spec_file_objects(ax, fpath, i=1, c=None, label=None):
    obj = fpath.split('/')[-1].split("_spec")[0]
    ax, wavs, data =  _plot_spec_file(ax, fpath, i=i, label=obj)
    return ax, obj, wavs, data

def plot_spec_file_dates(ax, fpath, i=1, c=None, label=None):
    date = fpath.split('/')[4].split('_')[0]
    ax, data =  _plot_spec_file(ax, fpath, i=i, label=date)
    return ax, date, data

def plot_divided_wlcs(
        ax,
        data,
        t0 = 0,
        ferr = 0,
        c = 'b',
        comps_to_use=None,
        div_kwargs=None,
        bad_div_kwargs=None,
        bad_idxs_user=None,
    ):
    # Create plotting config dictionaries if needed
    if div_kwargs is None: div_kwargs = {}

    # Unpack data
    flux_target = np.c_[data["oLC"]]
    flux_comps = data["cLC"]
    cNames = data["cNames"] # Original unsorted order from tepspec
    time = data['t']
    airmass = data['Z']
    flux_comps = flux_comps[:, np.argsort(cNames)] # Sort columns by cName
    cNames = sorted(cNames)
    comps_to_use = sorted(comps_to_use)
    comps_to_use_idxs = [cNames.index(cName) for cName in comps_to_use]
    flux_comps_used = flux_comps[:, comps_to_use_idxs]

    ##############################################
    # Plot division by individual comparison stars
    ##############################################
    flux_divs = flux_target / flux_comps_used
    flux_divs /= np.median(flux_divs, axis=0)

    for flux_div, cName in zip(flux_divs.T, comps_to_use):
        #ax.plot(flux_div, label=cName, **div_kwargs)

        bad_idxs = []
        for i in range(1, len(flux_div) - 1):
            diff_left = np.abs(flux_div[i] - flux_div[i-1])
            diff_right = np.abs(flux_div[i] - flux_div[i+1])
            if (diff_left > 2*ferr) and (diff_right > 2*ferr):
                bad_idxs.append(i)
        #ax.plot(bad_idxs, flux_div[bad_idxs], "o", color="r", alpha=0.5, ms=3, mew=0)
        print(f"{cName} 1-sigma bad_idxs: {bad_idxs}")
        if bad_idxs_user is not None:
            bad_idxs_user = _bad_idxs(bad_idxs_user)
            ax.errorbar(
                    (time[bad_idxs_user] - t0)*24.0,
                    flux_div[bad_idxs_user],
                    yerr=ferr, zorder=10,
                    **bad_div_kwargs,
            )
            bad_idxs = set(bad_idxs_user).union(set(bad_idxs))
            bad_idxs = list(bad_idxs)

        idxs = np.arange(len(flux_div))
        flux_div_used = flux_div[idxs != bad_idxs_user]
        idxs_used = idxs[idxs != bad_idxs_user]
        ax.errorbar((time[idxs_used] - t0)*24.0, flux_div_used, yerr=ferr, label=cName, **div_kwargs)
        high_am = airmass > 2
        ax.plot((time[high_am] - t0)*24.0, flux_div[high_am], 'ro', alpha=0.5)

    # Label axes[]
    #ax.set_xlabel("index")
    #ax.set_ylabel("normalized flux")

    return ax

def plot_sum_divided_wlcs(
        ax,
        data,
        ferr=0.001,
        bad_idxs_user=None,
        comps_to_use=None,
        div_sum_kwargs=None,
    ):
    # Create plotting config dictionaries if needed
    if div_sum_kwargs is None: div_sum_kwargs = {}

    # Unpack data
    flux_target = np.c_[data["oLC"]]
    flux_comps = data["cLC"]
    cNames = data["cNames"] # Original unsorted order from tepspec
    flux_comps = flux_comps[:, np.argsort(cNames)] # Sort columns by cName
    cNames = sorted(cNames)
    comps_to_use = sorted(comps_to_use)
    comps_to_use_idxs = [cNames.index(cName) for cName in comps_to_use]
    flux_comps_used = flux_comps[:, comps_to_use_idxs]

    # Select comp star columns from already sorted data["cLC"]
    if comps_to_use is not None:
        comps_to_use_idxs = [cNames.index(cName) for cName in comps_to_use]
        flux_comps_used = flux_comps[:, comps_to_use_idxs]
    else:
        flux_comps_used = flux_comps

    # Plot
    flux_comps_used_sum = np.c_[np.sum(flux_comps_used, axis=1)]
    flux_div_sum = flux_target / flux_comps_used_sum
    flux_div_sum /= np.median(flux_div_sum, axis=0)
    label = rf"$\frac{{target}}{{\sum{comps_to_use}}}$"
    ax.errorbar(range(len(flux_div_sum)), flux_div_sum, yerr=ferr, label=label, **div_sum_kwargs)

    bad_idxs = []
    for i in range(1, len(flux_div_sum) - 1):
        diff_left = np.abs(flux_div_sum[i] - flux_div_sum[i-1])
        diff_right = np.abs(flux_div_sum[i] - flux_div_sum[i+1])
        if (diff_left > 2*ferr) and (diff_right > 2*ferr):
            bad_idxs.append(i)
    ax.plot(bad_idxs, flux_div_sum[bad_idxs], "ro")
    print(bad_idxs)

    if bad_idxs_user is not None:
        bad_idxs_user = _bad_idxs(bad_idxs_user)
        ax.plot(bad_idxs_user, flux_div_sum[bad_idxs_user], "wX", ms=5)
        bad_idxs = set(bad_idxs_user).union(set(bad_idxs))
        bad_idxs = list(bad_idxs)

    # Annotate Rp/Rs approximation
    delta = 1.0 - np.min(flux_div_sum)
    RpRs = np.sqrt(delta)
    label = rf"$R_\mathrm{{p}}/R_\mathrm{{s}} \approx {RpRs:.3f}$"
    ax.annotate(label, xy=(0.9, 0.1), xycoords="axes fraction", ha="center")

    # Label axes
    ax.set_xlabel("index")
    ax.set_ylabel("normalized flux")

    # Save indices, times, and flux for target / sum comp flux used
    time = data['t']

    idxs = range(len(flux_div_sum))
    idxs_used = np.delete(idxs, bad_idxs)
    return ax, idxs_used, time, flux_div_sum, bad_idxs, np.array(idxs)

def plot_binned(
    ax,
    idxs_used,
    fluxes,
    bins,
    offset,
    colors,
    annotate=False,
    utc=False,
    species=None,
    bold_species=True,
    plot_kwargs=None,
    annotate_kwargs=None,
    annotate_rms_kwargs=None,
    models=None,
    ):
    """
    Plots binned light curves.

    Parameters
    ----------
    ax : matplotib.axes object
        Current axis to plot on
    idxs_used: index, time, phase, etc.
    fluxes : ndarray
        `time[idxs_used]` x `wbin` array of fluxes. Each column corresponds to a wavelength
        binned LC, where `wbin` is the number of wavelength bins
    bins : ndarray
        `wbin` x 2 array of wavelength bins. The first column holds the lower
        bound of each bin, and the second column holds the upper bound for each.
    offset : int, float
        How much space to put between each binned LC on `ax`
    colors : ndarray
        `wbin` x 3 array of RGB values to set color palette
    annotate : bool, optional
        Whether to annotate wavelength bins on plot. Default is True.
    utc : bool, optional
        Whether to convert `time` to UTC or not. Default is False.
    bold_species : bool, optional
        Whether to make annotated bins bold if they are in
    plot_kwargs : dict, optional
        Optional keyword arguments to pass to plot function
    annotate_kwargs : dict, optional
        Optional keyword arguments to pass to annotate function

    Returns
    -------
    ax : matplotib.axes object
        Current axis that was plotted on.
    """
    if plot_kwargs is None: plot_kwargs = {}
    if annotate_kwargs is None: annotate_kwargs = {}
    if annotate_rms_kwargs is None: annotate_rms_kwargs = {}

    offs = 0
    if idxs_used is None:
        idx_used = range
        slc = slice(0, len(fluxes.shape[0]) + 1)
    else:
        slc = idxs_used

    #fluxes = fluxes[slc, :]
    N = bins.shape[0] # number of wavelength bins

    for i in range(N):
        wav_bin = [round(bins[i][j], 3) for j in range(2)]

        if (utc):
            t_date = Time(time, format='jd')
            ax.plot_date(t_date.plot_date, fluxes[:, i] + offs, c=colors[i],
                    label=wav_bin, **plot_kwargs)
        else:
            ax.plot(
                idxs_used,
                fluxes[:, i] + offs,
                c=0.9*colors[i],
                label=wav_bin,
                #mec=0.9*colors[i],
                **plot_kwargs,
            )
            if models is not None:
                ax.plot(idxs_used, models[:, i] + offs, c=0.6*colors[i], lw=2)

        if annotate:
            #trans = transforms.blended_transform_factory(
            #            ax.transAxes, ax.transData
            #        )
            trans = transforms.blended_transform_factory(
                        ax.transData, ax.transData
                    )
            # Annotate wavelength bins
            ann = ax.annotate(
                    wav_bin,
                    #xy=(0, 1.004*(1 + offs)),
                    xy=(idxs_used[-1], 1.002*(1 + offs)),
                    xycoords=trans,
                    **annotate_kwargs,
                  )
            rms = np.std(fluxes[:, i]) * 1e6
            ann_rms = ax.annotate(
                    f"{int(rms)}",
                    #xy=(0, 1.004*(1 + offs)),
                    xy=(idxs_used[0], 1.002*(1 + offs)),
                    xycoords=trans,
                    **annotate_rms_kwargs,
                  )

            # Make annotations bold if bin is a species bin
            if bold_species:
                if species is None: species = dict()
                for spec, spec_wav in species.items():
                    if wav_bin[0] <= spec_wav <= wav_bin[1]:
                        ann.set_text(f'{spec}: {ann.get_text()}')
                        ann.set_weight('bold')

        offs += offset

    return ax

def plot_corner(samples, fpath_truths=None, title=None, fig=None, c='C0', plot_truth=True,
        weights=None, ranges=None, params=None, corner_kwargs=None,
        hist_kwargs=None):
    if corner_kwargs is None: corner_kwargs = {}
    if hist_kwargs is None: hist_kwargs = {}
    # Load data
    if fpath_truths is not None:
        with open(fpath_truths) as f:
            params_dict = json.load(f)
        labels = [p['symbol'] for p in params_dict.values()]
    else:
        labels = list(params.values())
    #truths = [v['truth'][0] for v in params_dict.values()]

    """
    samples = pd.read_csv(
        f"{dirpath}/samples.dat",
        sep=' ',
    )[params]
    if "incl" in samples.columns: samples["incl"] = samples["incl"]*(180.0/np.pi)
    """

    #if (t0_offs is not None) and ("t0" in params_dict.keys()): # adjust t0
        #samples["t0"] -= 2450000
        #params_dict["t0"]["truth"][0] -= 2450000

    #############
    # Plot corner
    #############
    fig = corner.corner(
        samples,
        plot_datapoints=False,
        color=c,
        labels=labels,
        #show_titles=False,
        label_kwargs = {'rotation':45},
        fill_contours=True,
        #hist_kwargs={'histtype':'step', 'lw':2, 'density':True},
        hist_kwargs=hist_kwargs,
        #title_fmt='.4e',
        #title_fmt='.5f',
        title_kwargs={'ha':'left', 'x':-0.03},
        #quantiles=[0.16, 0.5, 0.84],
        use_math_text=True,
        labelpad=0.4,
        fig=fig,
        weights=weights,
        range=ranges,
        #truths=truths,
        #truth_color="grey",
        **corner_kwargs,
    )

    # Loop over 1D histograms and overplot truth values
    ndim = samples.shape[1]
    axes = np.array(fig.axes).reshape((ndim, ndim))
    """
    ndim = samples.shape[1]
    axes = np.array(fig.axes).reshape((ndim, ndim))
    if plot_truth:
        for i, (param_key, param_data) in enumerate(params_dict.items()):
            ax = axes[i, i] # select 1d hist
            p_mean, p_u, p_d = param_data["truth"] # Unpack mean +/-
            ax.axvspan(p_mean - p_u, p_mean + p_u, alpha=0.5, color='darkgrey', lw=0, zorder=0)
            ax.axvline(p_mean, color='grey')
    """
    # Turn grid lines off
    for ax in fig.axes:
        ax.grid(False)

    fig.suptitle(title, x=0.99, y=0.99, style='italic', ha="right")

    return fig, axes

def plot_exoplanet_WLC(dirpath, figsize):
    time, flux = np.load(f"{dirpath}/lc.npy").T
    summary = pd.read_csv(f"{dirpath}/summary.dat", sep=' ', index_col=0)
    lc_model_draws = pd.read_csv(f"{dirpath}/model_draws.dat", sep=' ')
    gp_pred_draws = pd.read_csv(f"{dirpath}/gp_pred_draws.dat", sep=' ')
    lc_model_med = pd.read_csv(f"{dirpath}/median_model.dat", sep=' ')["lc_model"]
    gp_pred_med = pd.read_csv(f"{dirpath}/median_gp_pred.dat", sep=' ')["gp_pred"]

    post_t0 = summary['mean']['t0']
    post_period = summary['mean']['period']
    post_f0 = summary['mean']['f0']
    post_sigma = summary['mean']['sigma']

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    ax = axes[0]
    #ax.plot(time, (lc_model_draws.T+gp_pred_draws.T), color='C0', alpha=0.01)
    model_flux = lc_model_med + gp_pred_med
    ax.plot(time, model_flux, label="systematics model")
    ax.plot(time, flux, 'o', mew=0, alpha=0.5, label="raw flux")
    ax.set_xlabel('Time (JD)')
    ax.set_ylabel('Normalized Flux')

    ax = axes[1]
    det_flux = flux-gp_pred_med-post_f0+1
    phase = ((time-post_t0) % post_period)/post_period
    phase[phase > 0.5] -= 1
    order = np.argsort(phase)
    ax.plot(
        phase[order], (lc_model_med-post_f0+1)[order], label="median model"
    )
    ax.errorbar(
        phase, det_flux, post_sigma,
        ls='', marker='o', mew=0, alpha=0.5, label="normalized flux"
    )
    ax.set_xlabel('Phase')
    ax.set_xlim(-0.05, 0.05)
    ax.set_ylim(0.9833, 1.004)

    for ax in axes:
        ax.legend(loc=3, fontsize=12)

    fig.tight_layout()
    return fig, axes

# PCA TOOLS:
def get_sigma(x):
    """
    This function returns the MAD-based standard-deviation.
    """
    median = np.median(x)
    mad = np.median(np.abs(x-median))
    return 1.4826*mad

def standarize_data(input_data):
    output_data = np.copy(input_data)
    averages = np.median(input_data,axis=1)
    for i in range(len(averages)):
        sigma = get_sigma(output_data[i,:])
        output_data[i,:] = output_data[i,:] - averages[i]
        output_data[i,:] = output_data[i,:]/sigma
    return output_data

def classic_PCA(Input_Data, standarize = True):
    """
    classic_PCA function
    Description
    This function performs the classic Principal Component Analysis on a given dataset.
    """
    if standarize:
        Data = standarize_data(Input_Data)
    else:
        Data = np.copy(Input_Data)
    eigenvectors_cols,eigenvalues,eigenvectors_rows = np.linalg.svd(np.cov(Data))
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx[::-1]]
    eigenvectors_cols = eigenvectors_cols[:,idx[::-1]]
    eigenvectors_rows = eigenvectors_rows[idx[::-1],:]
    # Return: V matrix, eigenvalues and the principal components.
    return eigenvectors_rows,eigenvalues,np.dot(eigenvectors_rows,Data)

def reverse_ld_coeffs(ld_law, q1, q2):
    if ld_law == 'quadratic':
        coeff1 = 2.*np.sqrt(q1)*q2
        coeff2 = np.sqrt(q1)*(1.-2.*q2)
    elif ld_law=='squareroot':
        coeff1 = np.sqrt(q1)*(1.-2.*q2)
        coeff2 = 2.*np.sqrt(q1)*q2
    elif ld_law=='logarithmic':
        coeff1 = 1.-np.sqrt(q1)*q2
        coeff2 = 1.-np.sqrt(q1)
    elif ld_law == 'linear':
        return q1,q2
    return coeff1,coeff2


# Make a batman model
def init_batman(t, law):
    """
    This function initializes the batman code.
    """
    params = batman.TransitParams()
    params.t0 = 0.
    params.per = 1.
    params.rp = 0.1
    params.a = 15.
    params.inc = 87.
    params.ecc = 0.
    params.w = 90.
    if law == 'linear':
        params.u = [0.5]
    else:
        params.u = [0.1, 0.3]
    params.limb_dark = law
    m = batman.TransitModel(params, t)
    return params, m

def get_transit_model(t, lc_params):
    # Load ld params and init model
    ld_law = lc_params['ld_law']
    params, m = init_batman(t, law=ld_law)
    q1 = lc_params['q1']
    if ld_law != 'linear':
        q2 = lc_params['q2']
        coeff1, coeff2 = reverse_ld_coeffs(ld_law, q1, q2)
        params.u = [coeff1, coeff2]
    else:
        params.u = [q1]

    # Load the rest of the transit params
    params.t0 = lc_params['t0']
    params.per = lc_params['P']
    params.rp = lc_params['p']
    params.a = lc_params['a']
    params.inc = lc_params['inc']
    return m.light_curve(params)

def weighted_mean_uneven_errors(k,k_up,k_low,model=1):
    """
    A function to calculate the weighted mean of multiple, concatenated,
    transmission spectra that have un-even (non-symmetric) uncertainties. This
    uses the models of Barlow 2003. Inputs: k - the concatenated Rp/Rs values
    k_up - the concatenated positive uncertainties in Rp/Rs k_low - the
    concatenated negative uncertainties in Rp/Rs model - the number of the
    model as given in Barlow 2003 (either 1 or 2) Returns: weighted mean Rp/Rs
    the uncertainties in the weighted mean Rp/Rs values
    """
    nvalues = len(k)
    sigma = {}
    alpha = {}
    V = {}
    b = {}
    w = {}
    x_numerator = 0
    x_denominator = 0
    e_numerator = 0
    e_denominator = 0
    for i in range(nvalues):
        sigma[i+1] = (k_up[i]+k_low[i])/2. # eqn 1
        alpha[i+1] = (k_up[i]-k_low[i])/2. # eqn 1
        if model == 1:
            V[i+1] = sigma[i+1]**2 + (1 - 2/np.pi)*alpha[i+1]**2 # eqn 18
            b[i+1] = (k_up[i]-k_low[i])/np.sqrt(2*np.pi) # eqn 17
        if model == 2:
            V[i+1] = sigma[i+1]**2 + 2*alpha[i+1]**2 # eqn 18
            b[i+1] = alpha[i+1] # eqn 17
        w[i+1] = 1/V[i+1]
        x_numerator += (w[i+1]*(k[i]-b[i+1])) # eqn 16
        x_denominator += (w[i+1])
        e_numerator += (w[i+1]**2)*V[i+1] # below eqn 17
        e_denominator += w[i+1]
    return x_numerator/x_denominator, np.sqrt(e_numerator/(e_denominator**2))

def get_Delta_lnZ(fpath, fpath_flat):
    # load in exoretreival pkls
    with open(fpath, 'rb') as f:
        post = pickle.load(f)
    with open(fpath_flat, 'rb') as f:
        post_flat = pickle.load(f)

    delta_ln_Z = post['lnZ'] - post_flat['lnZ']
    sM = post['lnZerr']
    sF = post_flat['lnZerr']
    delta_ln_Z_unc = np.sqrt(sM**2 + sF**2)
    return delta_ln_Z, delta_ln_Z_unc, post['lnZ'], sM

def get_retr_params(dirpath, model, dirpath_flat, model_flat):
    # extract data
    df = pd.DataFrame(post['samples'])
    params_dict = {}
    for k, dist in df.items():
        v = dist.median()
        v_u = dist.quantile(0.84) - dist.quantile(0.5)
        v_d = dist.quantile(0.5) - dist.quantile(0.16)
        params_dict[k] = f' {v:.2f}^{{+{v_u:.2f}}}_{{-{v_d:.2f}}} '
        #print(f"{k}: $ {v:.2f}^{{+{v_u:.2f}}}_{{-{v_d:.2f}}} $")
    return params_dict

def plot_model(
    ax,
    model,
    fill_kwargs=None,
    model_kwargs=None,
):
    """ Plots the retrieval model and 1-sigma region.

    Parameters
    ----------
    ax : matplotlib.axes
        Axis to plot onto.

    model : astropy.table.table.Table .
        The table is read from retr_model.txt and contains five columns:
        1) wav: wavelengths (Å), 2) flux: Transit depth (ppm),
        3/4) wav_d, wav_u: +/- on wav, 5) flux_err: +/- on flux.

    fill_kwargs : dict
        keyword arguments to pass to ax.fill_between().

    model_kwargs : dict
        keyword arguments to pass to ax.semilogx().
    """
    if fill_kwargs is None:
        fill_kwargs = {}
    if model_kwargs is None:
        model_kwargs = {}

    # Plot fill
    wav, flux_d, flux_u = model['wav'], model['flux_d'], model['flux_u']
    p = ax.fill_between(wav, flux_d, flux_u, **fill_kwargs)

    # Plot model
    flux_model = model['flux']
    c = p.get_facecolor()[0]
    p = ax.plot(wav, flux_model, c=c, **model_kwargs)

    return ax, p

def plot_instrument(
    ax,
    instr=None,
    instr_sampled=None,
    sampled=True,
    instr_kwargs=None,
    sampled_kwargs=None,
):
    """ Plot retrieved transmission spectrum.

    Parameters
    ----------
    ax : matplotlib.axes
        Axis to plot onto.

    instr : astropy.table.table.Table
        The file is read from retr_<instrument>.txt and contains five columns:
        1) wav: wavelengths (Å), 2) flux: Transit depth (ppm),
        3/4) wav_d, wav_u: +/- on wav, 5) flux_err: +/- on flux.

    instr_sampled : astropy.table.table.Table .
        The file is read from retr_model_sampled_<instrument>.txt and contains
        two columns: 1) wav - wavelengths (Å), 2) flux - Transit depth (ppm).

    sampled : bool (default=True)
        If True, this plots the sampled retrieval model data points from
        `instr_sampled` onto the axis.

    instr_kwargs : dict
        keyword arguments to pass to `ax.errorbar` for the whole plot.

    sampled_kwargs : dict
        keyword arguments to pass to `ax.semilogx` for the sampled plot.
    """
    if instr_kwargs is None:
        instr_kwargs = {}
    if sampled_kwargs is None:
        sampled_kwargs = {}

    # Plot instrument
    if instr is not None:
        wav, flux = instr['wav'], instr['flux']
        x_d, x_u, y_err = instr['wav_d'], instr['wav_u'], instr['flux_err']
        ax.errorbar(wav, flux,
                    #xerr=[x_d, x_u],
                    yerr=y_err, **instr_kwargs)

    # Plot model sampled
    if sampled:
        wav_sampled, flux_sampled = instr_sampled['wav'], instr_sampled['flux']
        ax.plot(wav_sampled, flux_sampled, **sampled_kwargs)

    return ax

def myparser(s):
    dt, day_frac = s.split('.')
    dt = datetime.strptime(dt, "%Y-%m-%d")
    ms = 86_400_000.0 * float(f".{day_frac}")
    ms = timedelta(milliseconds=int(ms))
    return dt + ms

def plot_fluxes(
    ax,
    data,
    normalize=False,
    oot=False,
    idx_oot=None,
    use_time=True,
    t0=0,
):
    targ_flux = data["oLC"]
    comp_fluxes = data["cLC"]
    cNames = data["cNames"]
    exptime = data["etimes"]

    if use_time:
        time = (data["t"] - t0) * 24.0
    else:
        time = range(len(targ_flux))


    if normalize:
        #comp_fluxes = comp_fluxes / np.max(targ_flux)
        comp_fluxes = comp_fluxes / np.median(comp_fluxes, axis=0)
        targ_flux = targ_flux / np.median(targ_flux)

    if oot:
        targ_flux = targ_flux[idx_oot]
        comp_fluxes = comp_fluxes[idx_oot, :]
        time = time[idx_oot]

    # Plot
    targ_flux /= exptime
    comp_fluxes = comp_fluxes / exptime[:, None]
    ax.plot(time, targ_flux, '.', label="target")
    for cName, comp in zip(cNames, comp_fluxes.T):
        if "7" not in cName:
            ax.plot(time, comp, '.', label=cName)

    plot_data = {
        "time":time,
        "targ_flux":targ_flux,
        "comp_fluxes":comp_fluxes,
        "cNames":cNames,
    }

    ax.legend(fontsize=12)
    return ax, plot_data

def plot_div_fluxes(ax, time, targ_flux, comp_fluxes, cNames):
    ax.plot(time, targ_flux/targ_flux, '.', label="target")
    fluxes = []
    for cName, comp in zip(cNames, comp_fluxes.T):
        if "7" not in cName:
            flux = targ_flux/comp
            ax.plot(time, flux, '.', label=cName)
            fluxes.append(flux)
    ax.legend(fontsize=12)
    return ax

def plot_inset(
    ax,
    species_slc=slice(5,10),
    box_lims=[0.38, 0.65, 0.2, 0.2],
    lims=(7657-10, 7697+10),
):
    axins = ax.inset_axes(box_lims)
    axins.axhline(mean_wlc_depth, color="darkgrey", zorder=0, ls='--')
    [axins.axvline(wav, ls='--', lw=0.5, color='grey', zorder=0)
            for name, wav in species.items()]
    p_in = axins.errorbar(
        wav[species_slc],
        tspec_combined[species_slc],
        yerr=tspec_combined_unc[species_slc],
        c='w',
        mec='k',
        fmt='o',
        zorder=10,
        ecolor='k',
        lw=4,
    )
    axins.set_xlim(lims)
    #axins.set_ylim(ax.get_ylim())
    ax.indicate_inset_zoom(axins, alpha=1.0, edgecolor='k')

def detrend_BMA_WLC(
    out_folder,
    ld_law="linear",
    eccmean=0.0,
    omegamean=90.0,
    pl=0.0,
    pu=1.0,
    JITTER=(200.0 * 1e-6)**2.0,
):
    # File paths
    lc_path = f"{out_folder}/lc.dat"
    BMA_path = f"{out_folder}/results.dat"
    comps_path = f"{out_folder}/comps.dat"
    eparams_path = f"{out_folder}/../eparams.dat"

    # Raw data
    tall, fall, f_index = np.genfromtxt(lc_path, unpack=True)
    idx = np.where(f_index == 0)[0]
    t, f = tall[idx], fall[idx]

    # External params
    data = np.genfromtxt(eparams_path)
    X = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Comp stars
    data = np.genfromtxt(comps_path)
    Xc = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    if len(Xc.shape) != 1:
        eigenvectors, eigenvalues, PC = classic_PCA(Xc.T)
        Xc = PC.T
    else:
        Xc = Xc[:, None]

    ########################
    # BMA transit model vals
    ########################
    BMA = pd.read_table(
        BMA_path,
        escapechar='#',
        sep="\s+",
        index_col=" Variable",
    )

    mmeani, t0, P, r1, r2, q1 = (
        BMA.at["mmean", "Value"],
        BMA.at["t0", "Value"],
        BMA.at["P", "Value"],
        BMA.at["r1", "Value"],
        BMA.at["r2", "Value"],
        BMA.at["q1", "Value"],
    )

    if "rho" in BMA.columns:
        rhos = BMA.at["rho", "Value"]
        aR = ((rhos * G * ((P * 24.0 * 3600.0) ** 2)) / (3.0 * np.pi)) ** (
            1.0 / 3.0
        )
    else:
        aR = BMA.at["aR", "Value"]

    Ar = (pu - pl) / (2.0 + pl + pu)
    if r1 > Ar:
        b, p = (
            (1 + pl) * (1.0 + (r1 - 1.0) / (1.0 - Ar)),
            (1 - r2) * pl + r2 * pu,
        )
    else:
        b, p = (
            (1.0 + pl) + np.sqrt(r1 / Ar) * r2 * (pu - pl),
            pu + (pl - pu) * np.sqrt(r1 / Ar) * (1.0 - r2),
        )

    ecc = eccmean
    omega = omegamean
    ecc_factor = (1.0 + ecc * np.sin(omega * np.pi / 180.0)) / (
        1.0 - ecc ** 2
    )
    inc_inv_factor = (b / aR) * ecc_factor
    inc = np.arccos(inc_inv_factor) * 180.0 / np.pi

    # Comp star model
    mmean = BMA.at["mmean", "Value"]
    xcs = [xci for xci in BMA.index if "xc" in xci]
    xc = np.array([BMA.at[f"{xci}", "Value"] for xci in xcs])
    comp_model = mmean + np.dot(Xc[idx, :], xc)

    ###############
    # Transit model
    ###############
    params, m = init_batman(t, law=ld_law)

    if ld_law != "linear":
        q2 = BMA.at["posterior_samples"]["q2", "Value"]
        coeff1, coeff2 = reverse_ld_coeffs(ld_law, q1, q2)
        params.u = [coeff1, coeff2]
    else:
        params.u = [q1]

    params.t0 = t0
    params.per = P
    params.rp = p
    params.a = aR
    params.inc = inc
    params.ecc = ecc
    params.w = omega

    lcmodel = m.light_curve(params)
    model = -2.51 * np.log10(lcmodel)

    #####
    # GP
    ####
    kernel = np.var(f) * george.kernels.Matern32Kernel(
        np.ones(X[idx, :].shape[1]),
        ndim=X[idx, :].shape[1],
        axes=list(range(X[idx, :].shape[1])),
    )

    jitter = george.modeling.ConstantModel(np.log(JITTER))
    ljitter = np.log(BMA.at["jitter", "Value"]**2)
    max_var = BMA.at["max_var", "Value"]
    alpha_names = [k for k in BMA.index if "alpha" in k]
    alphas = np.array([BMA.at[alpha, "Value"] for alpha in alpha_names])

    gp = george.GP(
        kernel,
        mean=0.0,
        fit_mean=False,
        white_noise=jitter,
        fit_white_noise=True,
    )
    gp.compute(X[idx, :])
    gp_vector = np.r_[ljitter, np.log(max_var), np.log(1.0 / alphas)]
    gp.set_parameter_vector(gp_vector)

    #############
    # Detrending
    ############
    residuals = f - (model + comp_model)
    pred_mean, pred_var = gp.predict(residuals, X, return_var=True)

    detrended_lc = f - (comp_model + pred_mean)

    LC_det = 10**(-detrended_lc/2.51)
    LC_det_err = np.sqrt(np.exp(ljitter))
    LC_transit_model = lcmodel
    LC_systematics_model = comp_model + pred_mean

    return {
       "LC_det":LC_det,
       "LC_det_err":LC_det_err,
       "LC_transit_model":LC_transit_model,
       "LC_systematics_model":LC_systematics_model,
       "comp_model":comp_model,
       "pred_mean":pred_mean,
       "t":t,
       "t0":t0,
       "P":P,
   }

def get_wlc_and_tspec_data(data_paths_glob, offset=True):
    # Expand data paths
    data_paths = sorted(glob.glob(data_paths_glob))

    # Assign a transit number to each
    data_path_dict = {
        f"Transit {i}":data_path
        for (i, data_path) in enumerate(data_paths, start=1)
    }

    # Hold WLC (depth, depth_u, depth_d) in each row
    Ntransits = len(data_path_dict)
    wlc_data = np.zeros((Ntransits, 3))

    # Hold tspec + binned unc for each night
    Nbins = max(
        (
            len(glob.glob(f"{transit_path}/wavelength/wbin*"))
            for transit_path in data_path_dict.values()
        )
    )
    wavs_data = np.zeros((Ntransits, Nbins))
    tspec_data = np.zeros((Ntransits, Nbins, 3))

    for i, (transit_name, transit_path) in enumerate(data_path_dict.items()):

        # Load tspec data
        df_tspec = pd.read_csv(f"{transit_path}/transpec.csv")

        # Wavelengths
        wavs = df_tspec[["Wav_d", "Wav_u"]].mean(axis=1).values

        # WLC
        df_wlc = pd.read_table(
            f"{transit_path}/white-light/results.dat",
            sep='\s+',
            escapechar='#',
            index_col=" Variable",
        )

        wlc_data[i, :] = df_wlc.loc["p"]**2 * 1e6
        wavs_data[i, 0:len(df_tspec)] = wavs
        tspec_data[i, 0:len(df_tspec), :] = df_tspec[
                ["Depth (ppm)", "Depthup (ppm)", "DepthDown (ppm)"]
        ].to_numpy()

    # Combine
    mean_wlc_depth, mean_wlc_unc = weighted_mean_uneven_errors(
        *wlc_data.T
    )
    if offset:
        wlc_offsets = wlc_data[:, 0] - mean_wlc_depth
        tspec_data[:, :, 0] -= wlc_offsets.reshape((Ntransits, 1))
    else:
        wlc_offsets = 0.0

    return wavs_data, tspec_data, wlc_data, wlc_offsets

def plot_tspec(ax, data_paths_glob, offset=True):
    wavs_data, tspec_data, wlc_data, wlc_offsets = get_wlc_and_tspec_data(
            data_paths_glob
    )
    for i, (wavs_i, tspec_i) in enumerate(zip(wavs_data, tspec_data)):
        ax.errorbar(
                wavs_i,
                tspec_i[:, 0],
                yerr=tspec_i[:, 1],
                fmt = 'o',
        )
    return ax
