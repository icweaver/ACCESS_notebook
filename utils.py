import glob as glob
import matplotlib as mpl
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns

import bz2
import corner
import json
import pathlib
import pickle
import warnings
import os

from astropy import constants as const
from astropy import units as uni
from astropy.io import ascii, fits
from astropy.time import Time
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import ImageGrid

warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
warnings.filterwarnings("ignore", r"Degrees of freedom <= 0 for slice")

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

def _bad_idxs(s):
    if s == "[]":
        return []
    else:
        # Merges indices/idxs specified in `s` into a single numpy array of
        # indices to omit
        s = s.strip("[]").split(",")
        bad_idxs = list(map(_to_arr, s))
        bad_idxs = np.concatenate(bad_idxs, axis=0)
        return bad_idxs


def _to_arr(idx_or_slc):
    # Converts str to 1d numpy array
    # or slice to numpy array of ints.
    # This format makes it easier for flattening multiple arrays in `_bad_idxs`
    if ":" in idx_or_slc:
        lower, upper = map(int, idx_or_slc.split(":"))
        return np.arange(lower, upper + 1)
    else:
        return np.array([int(idx_or_slc)])


def compress_pickle(fname_out, fpath_pickle):
    data = load_pickle(fpath_pickle)
    with bz2.BZ2File(f"{fname_out}.pbz2", "wb") as f:
        pickle.dump(data, f)


def decompress_pickle(fname):
    data = bz2.BZ2File(fname, "rb")
    return pickle.load(data)


def get_evidences(base_dir, relative_to_spot_only=False):
    fit_R0 = "fitR0" if "fit_R0" in base_dir else "NofitR0"

    species = ["Na", "K", "TiO", "Na_K", "Na_TiO", "K_TiO", "Na_K_TiO"]
    model_names_dict = {
        "clear": f"NoHet_FitP0_NoClouds_NoHaze_{fit_R0}",
        "clear+cloud": f"NoHet_FitP0_Clouds_NoHaze_{fit_R0}",
        "clear+haze": f"NoHet_FitP0_NoClouds_Haze_{fit_R0}",
        "clear+cloud+haze": f"NoHet_FitP0_Clouds_Haze_{fit_R0}",
        "clear+spot": f"Het_FitP0_NoClouds_NoHaze_{fit_R0}",
        "clear+spot+cloud": f"Het_FitP0_Clouds_NoHaze_{fit_R0}",
        "clear+spot+haze": f"Het_FitP0_NoClouds_Haze_{fit_R0}",
        "clear+spot+cloud+haze": f"Het_FitP0_Clouds_Haze_{fit_R0}",
    }

    data_dict = {
        sp: {
            model_name: load_pickle(f"{base_dir}/HATP23_E1_{model_id}_{sp}/retrieval.pkl")
            for (model_name, model_id) in model_names_dict.items()
        }
        for sp in species
    }

    lnZ = {}
    lnZ_err = {}
    for species_name, species_data in data_dict.items():
        lnZ[species_name] = {}
        lnZ_err[species_name] = {}
        for model_name, model_data in species_data.items():
            lnZ[species_name][model_name] = model_data["lnZ"]
            lnZ_err[species_name][model_name] = model_data["lnZerr"]

    df_lnZ = pd.DataFrame(lnZ)
    df_lnZ_err = pd.DataFrame(lnZ_err)

    # Get log evidence for spot-only model and compute relative to this instead
    if relative_to_spot_only:
        model_id = f"Het_FitP0_NoClouds_NoHaze_{fit_R0}_no_features"
        df_lnZ_min = load_pickle(f"{base_dir}/HATP23_E1_{model_id}/retrieval.pkl")
        # print(f"spot only lnZ: {df_lnZ_min['lnZ']} +/- {df_lnZ_min['lnZerr']}")
        species_min = "no_features"
        model_min = "spot only"
    else:
        species_min = df_lnZ.min().idxmin()
        model_min = df_lnZ[species_min].idxmin()
        df_lnZ_min = data_dict[species_min][model_min]

    df_Delta_lnZ = df_lnZ - df_lnZ_min["lnZ"]
    df_Delta_lnZ_err = np.sqrt(df_lnZ_err ** 2 + df_lnZ_min["lnZerr"] ** 2)

    return df_Delta_lnZ, df_Delta_lnZ_err, species_min, model_min, data_dict


def get_phases(t, P, t0):
    """
    Given input times, a period (or posterior dist of periods)
    and time of transit center (or posterior), returns the
    phase at each time t. From juliet =]
    """
    if type(t) is not float:
        phase = ((Time(t) - t0) / P) % 1
        ii = np.where(phase >= 0.5)[0]
        phase[ii] = phase[ii] - 1.0
    else:
        phase = ((t - np.median(t0)) / np.median(P)) % 1
        if phase >= 0.5:
            phase = phase - 1.0
    return phase


def get_result(fpath, key="t0", unc=True):
    data = np.genfromtxt(fpath, encoding=None, dtype=None)
    for line in data:
        if key in line:
            if unc:
                return line
            else:
                return line[1]

    print(f"{key} not found. Check results.dat file.")


def get_table_stats(df, ps=[0.16, 0.5, 0.84], columns=None):
    ps_strs = [f"{p*100:.0f}%" for p in ps]
    df_stats = df.describe(percentiles=ps).loc[ps_strs]
    df_latex = pd.DataFrame(columns=df.columns)
    df_latex.loc["p"] = df_stats.loc[ps_strs[1]]
    df_latex.loc["p_u"] = df_stats.loc[ps_strs[2]] - df_stats.loc[ps_strs[1]]
    df_latex.loc["p_d"] = df_stats.loc[ps_strs[1]] - df_stats.loc[ps_strs[0]]
    latex_strs = df_latex.apply(write_latex_row2, axis=0)
    return pd.DataFrame(latex_strs, columns=columns)


def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f, encoding="latin")  # Python 2 -> 3
    return data


def myparser(s):
    dt, day_frac = s.split(".")
    dt = datetime.strptime(dt, "%Y-%m-%d")
    ms = 86_400_000.0 * float(f".{day_frac}")
    ms = timedelta(milliseconds=int(ms))
    return dt + ms


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
    if plot_kwargs is None:
        plot_kwargs = {}
    if annotate_kwargs is None:
        annotate_kwargs = {}
    if annotate_rms_kwargs is None:
        annotate_rms_kwargs = {}

    offs = 0
    if idxs_used is None:
        idx_used = range
        slc = slice(0, len(fluxes.shape[0]) + 1)
    else:
        slc = idxs_used

    # fluxes = fluxes[slc, :]
    N = bins.shape[0]  # number of wavelength bins

    for i in range(N):
        wav_bin = [round(bins[i][j], 3) for j in range(2)]

        if utc:
            t_date = Time(time, format="jd")
            ax.plot_date(
                t_date.plot_date,
                fluxes[:, i] + offs,
                c=colors[i],
                label=wav_bin,
                **plot_kwargs,
            )
        else:
            ax.plot(
                idxs_used,
                fluxes[:, i] + offs,
                c=0.9 * colors[i],
                label=wav_bin,
                # mec=0.9*colors[i],
                **plot_kwargs,
            )
            if models is not None:
                ax.plot(idxs_used, models[:, i] + offs, c=0.6 * colors[i], lw=2)

        if annotate:
            # trans = transforms.blended_transform_factory(
            #            ax.transAxes, ax.transData
            #        )
            trans = transforms.blended_transform_factory(ax.transData, ax.transData)
            # Annotate wavelength bins
            ann = ax.annotate(
                wav_bin,
                # xy=(0, 1.004*(1 + offs)),
                xy=(idxs_used[-1], 1.002 * (1 + offs)),
                xycoords=trans,
                **annotate_kwargs,
            )
            rms = np.std(fluxes[:, i]) * 1e6
            ann_rms = ax.annotate(
                f"{int(rms)}",
                xy=(idxs_used[0], 1.002 * (1 + offs)),
                xycoords=trans,
                **annotate_rms_kwargs,
            )

            # Make annotations bold if bin is a species bin
            if bold_species:
                if species is None:
                    species = dict()
                for spec, spec_wav in species.items():
                    if wav_bin[0] <= spec_wav <= wav_bin[1]:
                        ann.set_text(f"{spec}\n{ann.get_text()}")
                        ann.set_weight("bold")

        offs += offset

    return ax

def get_1D_spec(data, x):
    aperture = slice(x-25, x+25)
    peak_fluxes = []
    for row in data[None:None:-1, aperture]:
        peak_fluxes.append(np.max(row))

    return peak_fluxes

def plot_aperture(fpath):
    header = fits.getheader(fpath)
    data = fits.getdata(fpath)
    print(data.shape)
    name = header['filename']
    ut_start = header['UT-TIME']
    ut_end = header['UT-END']

    fig, axes = plt.subplots(1, 3, figsize=(11, 5))

    im = axes[0].imshow(data, cmap='viridis', vmin=0, vmax=10_000)
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

def plot_chips(dirpath, fpathame, target="", vmin=0, vmax=2_000, spec_ap=0, sky_ap=0):
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
                    if len(splitted) == 0:
                        break
                    self.obj = np.append(self.obj, splitted[0])
                    self.chip = np.append(self.chip, splitted[1][-1])
                    self.x = np.append(self.x, splitted[2])
                    self.y = np.append(self.y, splitted[3])
            self.chip = self.chip.astype(np.int)
            self.x = self.x.astype(np.float)
            self.y = self.y.astype(np.float)

    coords_file = f"{dirpath}/{target}.coords"
    coords = CoordinateData(coords_file)

    def biassec(x):
        x = x.split(",")
        fpathum = ((x[0])[1:]).split(":")
        snum = ((x[1])[: len(x[1]) - 1]).split(":")
        fpathum[0] = int(fpathum[0])
        fpathum[1] = int(fpathum[1])
        snum[0] = int(snum[0])
        snum[1] = int(snum[1])
        return fpathum, snum

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
        # bias has no significant structure, so a single median suffices, I think
        # overscan = [0:49] [2097:2145]
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
        # overscan = np.median(d[:,2048:2112])
        # newdata = d[:4095,0:2048] - overscan
        if (c == "c5") or (c == "c6") or (c == "c7") or (c == "c8"):
            if otype == "iff":
                newdata = newdata[::-1, :]
            else:
                newdata = newdata[::-1, ::-1]

        return newdata

    ############
    # Plot chips
    ############
    image = fpathame
    otype = image[0:3]
    chips = ["c1", "c2", "c3", "c4", "c6", "c5", "c8", "c7"]
    order = [1, 2, 3, 4, 6, 5, 8, 7]
    ranges = [vmin, vmax]

    fig = plt.figure(figsize=(8, 8))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(2, 4),
        axes_pad=0.05,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="5%",
        cbar_pad=0.05,
        share_all=True,
    )

    for i, ax in enumerate(grid):
        # print('Chip number '+(chips[i])[1:]+' overscan:')
        # ax.plot(int('24'+str(i+1)))
        # print(image, chips)
        d, h = fits.getdata(f"{dirpath}/{image}{chips[i]}.fits", header=True)
        d = BiasTrim(d, chips[i], h, "ift")
        im = ax.imshow(d, vmin=ranges[0], vmax=ranges[1], cmap="viridis")
        ax.xaxis.set_ticks(np.arange(250, 1_000, 250))
        ax.yaxis.set_ticks(np.arange(250, 2_000, 250))
        ax.tick_params(
            axis="both",
            which="major",
            length=0,
            labelsize=10,
        )
        va = "bottom"
        if chips[i] in ["c6", "c5", "c8", "c7"]:
            va = "top"
        for c in range(len(coords.chip)):
            if order[i] == coords.chip[c]:
                col = "w"
                if target in coords.obj[c]:
                    col = "y"
                ax.axvline(coords.x[c] - spec_ap, color="g", lw=1)
                ax.axvline(coords.x[c] + spec_ap, color="g", lw=1)
                ax.axvline(coords.x[c] - sky_ap, color="g", lw=1)
                ax.axvline(coords.x[c] + sky_ap, color="g", lw=1)
                ax.plot(coords.x[c], coords.y[c], "X", c="#c44e52")
                txt = ax.text(
                    coords.x[c],
                    coords.y[c],
                    coords.obj[c],
                    color=col,
                    ha="right",
                    va=va,
                    fontsize=14,
                    rotation=90,
                )
                txt.set_path_effects(
                    [PathEffects.withStroke(linewidth=2, foreground="k")]
                )
        # print('Shape:',d.shape)
        y = 1.05
        if i > 3:
            y = -0.2
        ax.set_title("Chip " + (chips[i])[1:], fontsize=12, y=y)
        ax.grid(False)
        #date_time = f"{h['DATE-OBS']} -- {h['TIME-OBS']}"
        #date_time = f"{h['DATE-OBS']} -- {h['LC-TIME']}"

    plt.axis("off")
    fig.colorbar(im, cax=grid.cbar_axes[0])
    return fig, im


def plot_corner(
    samples,
    fpath_truths=None,
    title=None,
    fig=None,
    c="C0",
    plot_truth=True,
    weights=None,
    ranges=None,
    params=None,
    corner_kwargs=None,
    hist_kwargs=None,
):
    if corner_kwargs is None:
        corner_kwargs = {}
    if hist_kwargs is None:
        hist_kwargs = {}
    # Load data
    if fpath_truths is not None:
        with open(fpath_truths) as f:
            params_dict = json.load(f)
        labels = [p["symbol"] for p in params_dict.values()]
    else:
        labels = list(params.values())

    #############
    # Plot corner
    #############
    fig = corner.corner(
        samples,
        plot_datapoints=False,
        color=c,
        labels=labels,
        # show_titles=False,
        label_kwargs={"rotation": 45},
        fill_contours=True,
        # hist_kwargs={'histtype':'step', 'lw':2, 'density':True},
        hist_kwargs=hist_kwargs,
        # title_fmt='.4e',
        # title_fmt='.5f',
        title_kwargs={"ha": "left", "x": -0.03},
        # quantiles=[0.16, 0.5, 0.84],
        use_math_text=True,
        labelpad=0.7,
        fig=fig,
        weights=weights,
        range=ranges,
        # truths=truths,
        # truth_color="grey",
        **corner_kwargs,
    )

    # Loop over 1D histograms and overplot truth values
    ndim = samples.shape[1]
    axes = np.array(fig.axes).reshape((ndim, ndim))

    # Turn grid lines off
    for ax in fig.axes:
        ax.grid(False)

    fig.suptitle(title, x=0.99, y=0.99, style="italic", ha="right")

    return fig, axes


def plot_divided_wlcs(
    ax,
    data,
    t0=0,
    ferr=0,
    c="b",
    comps_to_use=None,
    div_kwargs=None,
    bad_div_kwargs=None,
    bad_idxs_user=None,
):
    # Create plotting config dictionaries if needed
    if div_kwargs is None:
        div_kwargs = {}

    # Unpack data
    flux_target = np.c_[data["oLC"]]
    flux_comps = data["cLC"]
    cNames = data["cNames"]  # Original unsorted order from tepspec
    time = data["t"]
    airmass = data["Z"]
    flux_comps = flux_comps[:, np.argsort(cNames)]  # Sort columns by cName
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
        bad_idxs = []
        for i in range(1, len(flux_div) - 1):
            diff_left = np.abs(flux_div[i] - flux_div[i - 1])
            diff_right = np.abs(flux_div[i] - flux_div[i + 1])
            if ((diff_left > 2 * ferr) and (diff_right > 2 * ferr)) or airmass[i] >= 2:
                bad_idxs.append(i)
        # print(cName, bad_idxs)

        # ax.plot(
        #    (time[bad_idxs] - t0) * 24.0,
        #    flux_div[bad_idxs],
        #    #yerr=ferr,
        #    "k.",
        #    ms=15,
        #    zorder=10,
        #    #**bad_div_kwargs,
        # )
        if bad_idxs_user is not None:
            if isinstance(bad_idxs_user, str):
                bad_idxs_user = _bad_idxs(bad_idxs_user)

            # print(bad_idxs_user)
            ax.errorbar(
                (time[bad_idxs_user] - t0) * 24.0,
                flux_div[bad_idxs_user],
                yerr=ferr,
                zorder=10,
                **bad_div_kwargs,
            )
            # bad_idxs = set(bad_idxs_user).union(set(bad_idxs))
            # bad_idxs = list(bad_idxs)

        idxs = np.arange(len(flux_div))
        flux_div_used = flux_div  # [idxs != bad_idxs_user]
        idxs_used = idxs  # [idxs != bad_idxs_user]
        ax.errorbar(
            (time[idxs_used] - t0) * 24.0,
            flux_div_used,
            yerr=ferr,
            label=cName,
            **div_kwargs,
        )

    return ax, bad_idxs


def plot_evidence_summary(ax, df):
    p = df.transpose().plot(
        ax=ax,
        kind="bar",
        width=0.85,
        lw=0,
        capsize=3,
        ecolor="grey",
        # yerr=df_Delta_lnZ_err.transpose(), # Very small, omitted for visibility
        legend=False,
        # xlabel="Species",
        # ylabel="Relative log-evidence",
        rot=45,
        fontsize=12,
    )
    return p


def plot_inset(
    ax,
    wav,
    tspec,
    tspec_unc,
    species,
    species_slc=slice(5, 10),
    box_lims=[0.38, 0.65, 0.2, 0.2],
    lims=(7657 - 10, 7697 + 10),
    mean_wlc_depth=0.0,
):
    axins = ax.inset_axes(box_lims)
    axins.axhline(mean_wlc_depth, color="darkgrey", zorder=0, ls="--")
    [
        axins.axvline(wav, ls="--", lw=0.5, color="grey", zorder=0)
        for name, wav in species.items()
    ]
    p_in = axins.errorbar(
        wav[species_slc],
        tspec[species_slc],
        yerr=tspec_unc[species_slc],
        c="w",
        mec="k",
        fmt="o",
        zorder=10,
        ecolor="k",
        lw=4,
    )
    axins.set_xlim(lims)
    # axins.set_ylim(ax.get_ylim())
    ax.indicate_inset_zoom(axins, alpha=1.0, edgecolor="k")


def plot_model(
    ax,
    model_dict,
    fill_kwargs=None,
    model_kwargs=None,
    sample=True,
    sample_kwargs=None,
):
    """Plots the retrieval model and 1-sigma region.

    Parameters
    ----------
    ax : matplotlib.axes
        Axis to plot onto.

    model_dict : Dict with the following keys:
        data : astropy.table.table.Table .
            The table is read from retr_model.txt and contains five columns:
            1) wav: wavelengths (Å), 2) flux: Transit depth (ppm),
            3/4) wav_d, wav_u: +/- on wav, 5) flux_err: +/- on flux.

        sampled_data : astropy.table.table.Table .
            sampled data at given wavelengths in transmission spectrum

    fill_kwargs : dict
        keyword arguments to pass to ax.fill_between().

    model_kwargs : dict
        keyword arguments to pass to ax.semilogx().
    """
    if fill_kwargs is None:
        fill_kwargs = {}
    if model_kwargs is None:
        model_kwargs = {}
    if sample_kwargs is None:
        sample_kwargs = {}

    # Plot fill
    model = model_dict["tspec"]
    wav, flux_d, flux_u = model["wav"], model["flux_d"], model["flux_u"]
    p = ax.fill_between(wav, flux_d, flux_u, **fill_kwargs)

    # Plot model
    flux_model = model["flux"]
    c = p.get_facecolor()[0][0:3]
    p = ax.plot(wav, flux_model, c=c, **model_kwargs)

    # Plot model sampled
    if sample:
        model_sampled = model_dict["tspec_sampled"]
        wav_sampled, flux_sampled = model_sampled["wav"], model_sampled["flux"]
        ax.plot(wav_sampled, flux_sampled, "s", mec=0.7 * c, c=c)

    return ax


def plot_instrument(ax, instrument_data=None, instr_kwargs=None):
    """Plot retrieved transmission spectrum.

    Parameters
    ----------
    ax : matplotlib.axes
        Axis to plot onto.

    instrument_data : Dict
        Dict with the following keys:
        data : Table
            The file is read from retr_<instrument>.txt and contains five
            columns: 1) wav: wavelengths (Å), 2) flux: Transit depth (ppm),
            3/4) wav_d, wav_u: +/- on wav, 5) flux_err: +/- on flux.

        plot_kwargs : Dict
            keyword arguments to pass to `ax.errorbar` for the whole plot.
    """
    # Plot instrument
    instr = instrument_data["data"]
    wav, flux = instr["wav"], instr["flux"]
    x_d, x_u, y_err = instr["wav_d"], instr["wav_u"], instr["flux_err"]
    ax.errorbar(
        wav,
        flux,
        # xerr=[x_d, x_u],
        yerr=y_err,
        **instrument_data["plot_kwargs"],
    )

    return ax


def plot_params():
    return {
        # xticks
        "xtick.top": False,
        "xtick.direction": "out",
        "xtick.major.size": 5,
        "xtick.minor.visible": False,
        # yticks
        "ytick.right": False,
        "ytick.direction": "out",
        "ytick.major.size": 5,
        "ytick.minor.visible": False,
        # tick labels
        "axes.formatter.useoffset": False,
        # pallete
        "axes.prop_cycle": mpl.cycler(
            color=[
                "#fdbf6f",  # Yellow
                "#ff7f00",  # Orange
                "#a6cee3",  # Cyan
                "#1f78b4",  # Blue
                "plum",
                "#956cb4",  # Purple
                "mediumaquamarine",
                "#029e73",  # Green
                "slategray",
            ]
        ),
    }

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

def plot_spec_file(
    ax,
    fpath=None,
    data=None,
    wavs=None,
    i=1,
    label=None,
    median_kwargs=None,
    fill_kwargs=None,
):
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

    if fill_kwargs is None:
        fill_kwargs = {}

    if fpath is not None:
        data = fits_data(fpath)
        specs = data[:, i, :]
        wavs = range(specs.shape[1])
    else:
        specs = data
        wavs = wavs

    specs[specs <= 0] = np.nan  # Sanitize data
    specs_med = np.nanmedian(specs, axis=0)
    specs_std = np.nanstd(specs, axis=0)
    factor = np.nanmedian(specs_med)
    specs_med /= factor
    specs_std /= factor

    # Empty plot for custom legend
    p = ax.plot([], "-o", label=label, **median_kwargs)

    # Plot normalized median and 1-sigma region
    specs_med /= np.nanmax(specs_med)
    specs_std /= np.nanmax(specs_med)
    c = p[0].get_color()
    ax.fill_between(
        wavs,
        specs_med - specs_std,
        specs_med + specs_std,
        color=c,
        alpha=0.25,
        lw=0,
        **fill_kwargs,
    )
    ax.plot(wavs, specs_med, lw=2, color=c)
    return ax, wavs, data


def plot_tspec_IMACS(ax, base_dir):
    data_dirs = sorted(glob.glob(f"{base_dir}/hp*"))
    data_dict = {
        f"Transit {i}": data_dir for (i, data_dir) in enumerate(data_dirs, start=1)
    }

    # Use first entry for wavelength
    transit_0, dirpath_0 = "Transit 1", data_dict["Transit 1"]
    fpath = f"{dirpath_0}/transpec.csv"
    df_wavs = pd.read_csv(fpath)[["Wav_d", "Wav_u"]]
    wav = np.mean(df_wavs, axis=1)

    depth_wlc_stats = []
    tspec_stats = []
    for transit, dirpath in data_dict.items():
        # WLCs
        fpath = f"{dirpath}/white-light/results.dat"
        results = pd.read_table(fpath, sep="\s+", escapechar="#", index_col="Variable")
        p, p_u, p_d = results.loc["p"]
        wlc_depth = p ** 2 * 1e6
        wlc_depth_u = p_u ** 2 * 1e6
        wlc_depth_d = p_d ** 2 * 1e6
        depth_wlc_stats.append([wlc_depth, wlc_depth_u, wlc_depth_d])

        # Tspec
        fpath = f"{dirpath}/transpec.csv"
        df_tspec = pd.read_csv(fpath)[["Depth (ppm)", "Depthup (ppm)", "DepthDown (ppm)"]]

        if (transit == "Transit 1" and "full" in dirpath) or ("species" in dirpath):
            tspec, tspec_u, tspec_d = df_tspec.values.T
        else:
            tspec, tspec_u, tspec_d = df_tspec.values[1:-1, :].T

        tspec_stats.append([tspec, tspec_u, tspec_d])

    # Compute offset
    depth_wlc_stats = np.array(depth_wlc_stats)
    depth_wlc, depth_wlc_u, depth_wlc_d = depth_wlc_stats.T
    mean_wlc_depth, mean_wlc_depth_unc = weighted_mean_uneven_errors(
        depth_wlc, depth_wlc_u, depth_wlc_d
    )

    wlc_offsets = depth_wlc - mean_wlc_depth
    # wlc_offsets *= 0
    print(f"offsets: {wlc_offsets}")
    print(f"offsets (% mean wlc depth): {wlc_offsets*100/mean_wlc_depth}")
    tspec_stats = np.array(tspec_stats)  # transits x (depth, u, d) x wavelength
    print(tspec_stats.shape)
    tspec_stats[:, 0, :] -= wlc_offsets[np.newaxis].T

    tspec_depths = tspec_stats[:, 0, :]
    tspec_us = tspec_stats[:, 1, :]
    tspec_ds = tspec_stats[:, 2, :]

    ######
    # Plot
    ######

    ax.axhline(mean_wlc_depth, color="darkgrey", zorder=0, ls="--")

    # Species
    species = {
        "Na I-D": 5892.9,
        # "Hα":6564.6,
        "K I_avg": 7682.0,
        "Na I-8200_avg": 8189.0,
    }
    [
        ax.axvline(wav, ls="--", lw=0.5, color="grey", zorder=0)
        for name, wav in species.items()
    ]

    tspec_tables = {}  # Each entry will hold Table(Transit, Depth, Up , Down)
    # For combining into latex later
    for transit, tspec, tspec_d, tspec_u in zip(
        data_dict.keys(), tspec_depths, tspec_ds, tspec_us
    ):
        ax.errorbar(
            wav,
            tspec,
            yerr=[tspec_d, tspec_u],
            fmt="o",
            alpha=1.0,
            mew=0,
            label=transit,
            barsabove=False,
        )

        data_i = {}
        data_i["Depth (ppm)"] = tspec
        data_i["Depthup (ppm)"] = tspec_u
        data_i["DepthDown (ppm)"] = tspec_d
        df = pd.DataFrame(data_i)
        tspec_tables[transit] = df

    tspec_combined = []
    tspec_combined_unc = []
    tspec_combined_max = []
    tspec_combined_unc_max = []
    for i in range(len(tspec_stats[0, 0, :])):
        tspec_comb, tspec_comb_unc = weighted_mean_uneven_errors(
            tspec_stats[:, 0, i], tspec_stats[:, 1, i], tspec_stats[:, 2, i]
        )
        tspec_combined.append(tspec_comb)
        tspec_combined_unc.append(tspec_comb_unc)

        # single errorbar way
        uncs_max = np.max([tspec_stats[:, 1, i], tspec_stats[:, 2, i]], axis=0)
        weights = 1 / uncs_max ** 2
        tspec_comb_max = np.average(tspec_stats[:, 0, i], weights=weights)
        tspec_comb_max_unc = weighted_err(uncs_max)
        tspec_combined_max.append(tspec_comb_max)
        tspec_combined_unc_max.append(tspec_comb_max_unc)

    # Combined
    tspec_combined = np.array(tspec_combined)
    tspec_combined_unc = np.array(tspec_combined_unc)
    p = ax.errorbar(
        wav,
        tspec_combined,
        yerr=tspec_combined_unc,
        c="w",
        mec="k",
        fmt="o",
        zorder=10,
        label="combined",
        ecolor="k",
        lw=4,
    )

    # Write to table
    tspec_table = pd.DataFrame()
    tspec_table["Wavelength (Å)"] = df_wavs.apply(write_latex_wav, axis=1)
    # Transmission spectra
    for transit, df_tspec in tspec_tables.items():
        tspec_table[transit] = df_tspec.apply(write_latex_row_sig_fig, axis=1)
    data = np.array([tspec_combined, tspec_combined_unc]).T
    df_combined = pd.DataFrame(data, columns=["Combined", "Unc"])
    tspec_table["Combined"] = df_combined.apply(write_latex_single_sig_fig, axis=1)
    # Save data
    table_suffix = "c" if "full" in dirpath else "s"
    out_path_tspec = f"{base_dir}/tspec_{table_suffix}.csv"
    print(f"Saving tspec to: {out_path_tspec}")
    tspec_table.to_csv(out_path_tspec, index=False)
    np.savetxt(
        f"{base_dir}/tspec_{table_suffix}.txt",
        np.c_[wav, data],
        header="wav(Å) depth(ppm) depth_unc(ppm)",
    ),
    np.savetxt(
        f"{base_dir}/mean_wlc_depth_{table_suffix}.txt",
        np.c_[mean_wlc_depth, mean_wlc_depth_unc],
        header="depth(ppm) depth_unc(ppm)",
    )

    ax.legend(loc=1, ncol=6, frameon=True, fontsize=11)
    # Inset plots
    if "species" in dirpath:
        margin = 10
        plot_inset(
            ax,
            wav,
            tspec_combined,
            tspec_combined_unc,
            species,
            species_slc=slice(0, 5),
            box_lims=[0.3, 0.15, 0.2, 0.2],
            lims=(5780.40 - margin, 6005.40 + margin),
            mean_wlc_depth=mean_wlc_depth,
        )
        plot_inset(
            ax,
            wav,
            tspec_combined,
            tspec_combined_unc,
            species,
            species_slc=slice(5, 10),
            box_lims=[0.38, 0.65, 0.2, 0.2],
            lims=(7657 - margin, 7707 + margin),
            mean_wlc_depth=mean_wlc_depth,
        )
        plot_inset(
            ax,
            wav,
            tspec_combined,
            tspec_combined_unc,
            species,
            species_slc=slice(10, 15),
            box_lims=[0.78, 0.15, 0.2, 0.2],
            lims=(8089 - margin, 8289 + margin),
            mean_wlc_depth=mean_wlc_depth,
        )

    # Save
    ax.set_xlim(5106.55, 9362.45)
    ax.set_ylim(9_500, 16_500)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel(r"Transit Depth (ppm)")

    print("mean WLC depth:", mean_wlc_depth, mean_wlc_depth_unc)
    Rs = 1.152 * uni.solRad
    Rp = np.sqrt(mean_wlc_depth * 1e-6 * Rs ** 2)
    Mp = 1.92 * uni.jupiterMass
    gp = const.G * Mp / Rp ** 2
    print("Rp (Rj):", Rp.to("Rjupiter"))
    print("Rs (Rsun):", Rs.to("Rsun"))
    print("gp (m/s^2):", gp.to("cm/s^2"))

    return ax


def savefig(fpath, **kwargs):
    pathlib.Path(fpath).parents[0].mkdir(parents=True, exist_ok=True)
    plt.savefig(fpath, bbox_inches="tight", **kwargs)


def wbin_num(fpath):
    # Extracts <num> from fpath = .../wbin<num>/...
    tokens = fpath.split("/")
    for token in tokens:
        if "wbin" in token:  # Do the extraction
            bin_str = token.split("wbin")[-1]
            bin_num = int(bin_str)
            return bin_num


def weighted_err(errs):
    weights = 1 / errs ** 2
    partials = weights / np.sum(weights, axis=0)
    deltas = np.sqrt(np.sum((partials * errs) ** 2, axis=0))
    return deltas


def weighted_mean_uneven_errors(k, k_up, k_low, model=1):
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
        sigma[i + 1] = (k_up[i] + k_low[i]) / 2.0  # eqn 1
        alpha[i + 1] = (k_up[i] - k_low[i]) / 2.0  # eqn 1
        if model == 1:
            V[i + 1] = sigma[i + 1] ** 2 + (1 - 2 / np.pi) * alpha[i + 1] ** 2  # eqn 18
            b[i + 1] = (k_up[i] - k_low[i]) / np.sqrt(2 * np.pi)  # eqn 17
        if model == 2:
            V[i + 1] = sigma[i + 1] ** 2 + 2 * alpha[i + 1] ** 2  # eqn 18
            b[i + 1] = alpha[i + 1]  # eqn 17
        w[i + 1] = 1 / V[i + 1]
        x_numerator += w[i + 1] * (k[i] - b[i + 1])  # eqn 16
        x_denominator += w[i + 1]
        e_numerator += (w[i + 1] ** 2) * V[i + 1]  # below eqn 17
        e_denominator += w[i + 1]
    return (
        x_numerator / x_denominator,
        np.sqrt(e_numerator / (e_denominator ** 2)),
    )


def write_latex_row(row):
    v, vu, vd = row
    return f"{v:.3f}^{{+{vu:.1e}}}_{{-{vd:.1e}}}"


def write_latex_row2(row):
    v, vu, vd = row
    return f"{v:.1f}^{{+{vu:.0f}}}_{{-{vd:.0f}}}"


def write_latex_row_sig_fig(row, x_digits=-2, xu_digits=-2, xd_digits=-2):
    x, xu, xd = row
    v = int(round(x, x_digits))
    vu = int(round(xu, xu_digits))
    vd = int(round(xd, xd_digits))
    return f"{v}^{{+{vu}}}_{{-{vd}}}"
    # return f"{v:.3g}^{{+{vu:.2g}}}_{{-{vd:.2g}}}"


def write_latex2(val, unc):
    return f"{val:.2f} \pm {unc:.2f}"


def write_latex_single(row):
    v, v_unc = row
    return f"{v:.3f} \pm {v_unc:.3f}"


def write_latex_single_sig_fig(row, x_digits=-2, x_unc_digits=-2):
    x, x_unc = row
    v = int(round(x, x_digits))
    v_unc = int(round(x_unc, x_unc_digits))
    return f"{v} \pm {v_unc}"


def write_latex_wav(row):
    wav_d, wav_u = row
    return f"{wav_d:.1f} - {wav_u:.1f}"

def plot_traces(dirpath, start=0, end=-2):
    XX = load_pickle(f'{dirpath}/XX.pkl')
    YY = load_pickle(f'{dirpath}/YY.pkl')

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

    #axcb = fig.colorbar(lines, cmap='Blues', label='index')
    fig.suptitle(dirpath, y=1.01)
    fig.tight_layout()

    return fig, axes, XX, YY

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
            sep = '\s+'
        elif ".txt" in fpath_lines_ref:
            col_name_pixel = "Pix"
            col_name_wav = "Wav"
            sep = '\t'
        else:
            col_name_pixel = "Pix"
            col_name_wav = "Wav"
            sep = ','

        df_ref = pd.read_csv(
            fpath_lines_ref,
            #usecols=["Wav", "Pix", "Chip"],
            escapechar='#',
            sep=sep,
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
    #path = pathlib.PurePath(fpath_arc_ref)
    #parents = path.parents
    #fname = path.name
    #title = f"{parents[1].name}/{parents[0].name}/{fname}"
    title = fpath_arc_ref
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
        ann_c="lime",
    )
    # Highlight wavelengths in top panel if already selected in bottom
    annotations = [
        child for child in ax_ref.get_children()
        if isinstance(child, plt.matplotlib.text.Annotation)
    ]
    for ann in annotations:
        if float(ann.get_text().split('\n')[1]) in wavs:
            ann.set_color("lime")
            ann.set_fontsize(12)

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
        wav = ann.get_text().split(',')[1]
        wavs.append(float(wav))
        # Annotes the wavelength value on bottom panel for ease of reference
        ax_lco.annotate(wav, xy=(x+1, y+0.02), fontsize=10, c='g')
        fig.canvas.draw()
        fig.canvas.mpl_disconnect(cid) # Only run once per `X` press on keyboard

    def on_key(event):
        # Display and record pixel location on 'X' key press on bottom panel
        if (event.key == 'x' or event.key == 'X') and (event.inaxes == ax_lco):
            pixel = event.xdata
            pixels.append(pixel)
            y = event.ydata
            ax_lco.axvline(pixel, c='r', ls='--', lw=1)
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
                f"{pixel:.1f}\n{wav:.4f}", (pixel, y),
                fontsize=12, picker=1, color=ann_c,
            )
            ax.axvline(pixel, lw=1, ls = '--', c='r')

    ax.set_xlabel("Pixel")
    ax.set_ylabel("Normalized flux")
    #ax.set_ylim(-0.01, 1.1)

    return ax

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
