import numpy as np

target = "WASP50"
filedir = "/home/mango/data/WASP50/wasp50_ut161211"
pklname = "WASP50_WLC_OTG.pkl"
coords_file = "WASP50.coords"
science_object = "w50is science"
first_frame = ""  # Not necessary, but you can use it to truncate the file list.
last_frame = ""
bad_comps = [
    ""
]  # Comparison stars you don't want to include in the detrending, e.g. 'comp2'.
bad_objs = [
    ""
]  # Comparison/chip combinations from coordinates file that you don't want to include, e.g. 'comp2_3'.
Interactive = False  # Set to false to save PDF.
n_to_add = np.inf  # Number of frames to add before stopping to display plot
# np.inf = only displays the plot after all existing frames are added.

slitwidth = 55  # roughly the slit width in pixels
basewindow = 45  # pixels used in median filter to obtain background
medfilter = 13  # points used in median filter for plot

ing_time = ["2016-12-12T02:08:00.0"]
mid_time = ["2016-12-12T03:03:00.0"]
egr_time = ["2016-12-12T03:58:00.0"]

p = 0.139  # rp/rs
p_err = 0.0006
source = "(--- et al.)"
