import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os, sys
import astropy.io.fits as pyfits
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from scipy.optimize import curve_fit

mpl.use("Qt5Agg")

##############################################
#Functions

def gauss(x, a, mu, sigma):
    return a*np.exp((-(x-mu)**2)/(2*sigma**2))

def lorentz(x, I, x_0, HWHM):
    return I * ( HWHM**2 / ( (x - x_0)**2 + HWHM**2 ) )

def InteractiveID(star, chip, init_wav, init_pix, all_lines, o, width=5, fitorder=2, binning=2):
    #loading data for this chip
    d = pyfits.getdata(star+'_'+str(chip)+'_arc.fits')
    if len(d.shape) > 1:
        d = d[1]
    flux = d - medfilt(d,101)
    ###IM HERE##
    pix = np.arange(len(d))
    f = interp1d(pix,flux)
    #making interactive plot
    #plt.ion()
    #fig = plt.figure('Star: '+star+' | Chip: '+str(chip))
    #ax = fig.add_subplot(1,1,1)
    fig, ax = plt.subplots(num=f"Star: {star} | Chip: {chip}", figsize=(11, 6))
    # plot calibrated lamp:
    plt.plot(pix,flux)
    # Predict lines...
    coeff = np.polyfit(init_wav,init_pix,fitorder)
    pix_lines = np.polyval(coeff,all_lines)
    idx = np.where((pix_lines>0)&(pix_lines<4095/binning))[0]
    pix_lines = pix_lines[idx]
    wav_lines = all_lines[idx]
    ax.bar(pix_lines,f(pix_lines),width=0.05,color='black',alpha=0.3)
    for i in range(len(pix_lines)):
        ax.text(pix_lines[i],f(pix_lines[i])+0.05,str(wav_lines[i]),alpha=0.3)

    #fitting procedure
    Done = False
    for i in range(len(wav_lines)):
        j = (i-1)%len(wav_lines)            # For identifying nearby lines
        k = (i+1)%len(wav_lines)
        line = wav_lines[i]
        pixel = pix_lines[i]
        UserVerified = False
        if Done:
            break
        while not UserVerified:
            pix_win = np.arange(int(pixel-12),int(pixel+12))
            flux_win = np.zeros(len(pix_win))
            for ii in range(len(pix_win)):
                if pix_win[ii] in range(0,2048):
                    flux_win[ii] = flux[pix_win[ii]]
            ax.set_xlim([np.min(pix_win), np.max(pix_win)])
            ax.set_ylim([np.min(flux_win),1.2*np.max(flux_win)])
            question = '\nIs prediction for '+str(line)+' near line center? [y: yes, n:no, s:skip line] '
            ax.set_title(question)
            fig.canvas.draw()
            plt.show(block=False)
            user_ds = input(question)
            if user_ds.startswith('s'):
                print('Skipping '+str(line))
                break
            elif (user_ds.startswith('y') or user_ds==''):
                PixelCenterIsGood = True
                print('Using pixel center: {:}'.format(pixel))
            elif user_ds=='q':
                Done = True
                break
            else:
                PixelCenterIsGood = False
            while not PixelCenterIsGood:
                question = '\nSuggest new line center:'
                answer = input(question)
                try:
                    pixel = float(answer)
                except:
                    print('\nNumber is not valid. Try again.')
                else:
                    print('Using new pixel center:',pixel)
                    PixelCenterIsGood = True
            print('Fitting profile to line.')
            Fitting = True
            l_width = np.min([width, np.abs( (pix_lines[i]-pix_lines[j])/2. )]) # Restrict fitting window if known line nearby
            r_width = np.min([width, np.abs( (pix_lines[k]-pix_lines[i])/2. )])
            x = np.arange(pixel-l_width, pixel+r_width, 0.1)
            try:
                y = f(x)
                y[np.where(x<0)] = 0
            except:
                print('> Line falls off chip.  Skipping '+str(line))
                Fitting = False
                UserVerified = True
            while Fitting:
#                 mu = np.sum(x * y)/np.sum(y)
#                 a = f(mu)
#                 mu = pixel
#                 sigma = np.sum(y * (x - mu)**2)/np.sum(y)
#                 p0 = [a, mu, sigma]
#                 try:
#                     popt, pcov = curve_fit(gauss, x, y, p0)
#                 except:
#                     popt = p0
#                 plt.plot(x, gauss(x, *popt), 'k-')
#                 ax.bar(popt[1],gauss(popt[1], *popt),width=0.01)
                x_0 = np.sum(x * y)/np.sum(y)
                I = f(x_0)
                HWHM = 0.8
                p0 = [I, x_0, HWHM]
                try:
                    popt, pcov = curve_fit(lorentz, x, y, p0)
                except:
                    popt = p0
                if popt[1] < np.min(x) or popt[1] > np.max(x):
                    popt[1] = pixel
                plt.plot(x, lorentz(x, *popt), 'k-')
                ax.bar(popt[1],lorentz(popt[1], *popt),width=0.05, color='black')
                question = 'Is center good? [y: yes, n: no, s:skip line] '
                ax.set_title(question)
                fig.canvas.draw()
                user_ds = input(question)
                Fitting = False
                if user_ds.startswith('y') or user_ds=='':
                    UserVerified = True
                    if line in o['wav']:
                        print('Overwriting saved position -> line: {:} pixel: {:}'.format(line, popt[1]))
                        o_idx = o['wav'].index(line)
                        o['wav'][o_idx] = line
                        o['pix'][o_idx] = popt[1]
                        o['chip'][o_idx] = chip
                    else:
                        print('Saving new position -> line: {:} pixel: {:}'.format(line, popt[1]))
                        o['wav'].append(line)
                        o['pix'].append(popt[1])
                        o['chip'].append(chip)
                elif user_ds.startswith('s'):
                    UserVerified = True
                    print('Skipping '+str(line))
                else:
                    Fitting = False
    plt.close(fig)
    print('Done with chip: '+str(chip)+'.')
    return o


##############################################
#Loading Data and Initial Guesses

print('''
     ###### Interactive Line Identification #####
                  return = yes
                       y = yes
                       n = no
                       q = quit
''')

all_lines = np.loadtxt('HeNeAr.dat',unpack=True)
all_lines.sort()

#all_files = glob.glob("../data/data_reductions/HATP23/ut180603_a15_25_noflat/arcs/*_arc.fits")
all_files = glob.glob('*_?_arc*.fits')
stars = []
for file in all_files:
    idx_underscore = file.find('_')
    star = file[0:idx_underscore]
    if star not in stars:
        stars.append(star)

string_to_show =  '\nStars with Arc Lamps:\n'
for i, star in enumerate(stars):
    string_to_show += '({:})\t{:}'.format(i, star)
    if os.path.exists(star+'_guesses.txt'):
        string_to_show += '\tguesses exist'
    if os.path.exists(star+'_lines_chips.txt'):
        string_to_show += '\tlines exist'
    string_to_show += '\n'
print(string_to_show)
user_ds = input('Choose a star: ')
try:
    star = stars[int(user_ds)]
except:
    sys.exit('Not a valid choice. Exiting...')

files = glob.glob(star+'_?_arc*.fits')
chips = []
for file in files:
    idx_underscore = file.find('_arc')
    chips.append(int(file[idx_underscore-1]))

filename = star+'_guesses.csv'
try:
    wav0, pix0, obj_chip0 = np.loadtxt(filename,unpack=True,delimiter=',',skiprows=1)
except:
    sys.exit('Initial guesses for '+str(star)+' not found. Exiting...')

##############################################
#Interactive Plot

o = {'wav':[], 'pix':[], 'chip':[]}

print('Found chips: ',chips)

for chip in chips:
    idx = np.where(obj_chip0==chip)[0]
    init_wav = wav0[idx]
    init_pix = pix0[idx]

    if len(idx)==0:
        sys.exit('No lines identified on '+star+'_'+str(chip)+'_arc.fits. Identify at least 2 lines per chip and retry.')

    print('> Working on chip',chip)
    TryAgain = True
    fitorder = 1
    while TryAgain==True:
        o = InteractiveID(star, chip, init_wav, init_pix, all_lines, o, fitorder=fitorder)
        user_ds = input('Would you like to recalculate the fit and find more lines? [y/n] ')
        if user_ds.startswith('y'):
            idx = np.where( np.array(o['chip'])==chip)
            if len(idx)>1:
                init_wav, init_pix = np.array(o['wav'])[idx], np.array(o['pix'])[idx]
            user_ds = input('New fit order?: ')
            if user_ds!='':
                try:
                    fitorder = int(user_ds)
                except:
                    print('Not an integer.  Using old fit order of:',fitorder)
        else:
            TryAgain = False

##############################################
#Saving Data

if len(o['wav'])>0:
    print('\nSaving data...\n')
    fname = star+'_lines_chips.txt'
    if os.path.exists(fname):
        print('> File exists:  '+fname)
        file_exists = True
        counter = 1
        while file_exists:
            fname = star+'_lines_chips_'+str(counter)+'.txt'
            if not os.path.exists(fname):
                file_exists = False
            else:
                counter +=1
    print('> Saved as:  '+fname)
    f = open(fname, 'w')
    f.write('#Wav\tPix\tChip\n')
    order = np.argsort(o['wav'])
    for i in range(len(o['wav'])):
        out_line = str(round(o['wav'][order[i]],4))+'\t'+str(round(o['pix'][order[i]],1))+'\t'+str(o['chip'][order[i]])+'\n'
        f.write(out_line)
    f.close()

print('\nExiting...\n')
