#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker      # to use Formatter and Locator
import matplotlib.patches as patches
import numpy as np
import librosa
import math
import seaborn as sns
import pandas as pd
#import soundfile as sf
#import scipy.io.wavfile as wio


font_scalings = {
    'xx-small' : 0.579,
    'x-small'  : 0.694,
    'small'    : 0.833,
    'medium'   : 1.0,
    'large'    : 1.200,
    'x-large'  : 1.440,
    'xx-large' : 1.728,
    'larger'   : 1.2,
    'smaller'  : 0.833,
    None       : 1.0
}

plot_type = {
    'plot',
    'boxplot',
    'scatter',
    'bar',
    'barth',
    'hist',
    'image',
    'confusion_matrix',
    None
}

class Figdata:
    def __init__(self, data, data2=[], type=None, labels=None, title=None, xlabel=None, ylabel=None, xticks=None, yticks=None, xlim=None, ylim=None, color=None, color2=None, mask=None, highlight_label=[]):
        self.data = data
        self.data2 = data2
        self.type = type
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xticks = xticks
        self.yticks = yticks
        self.xlim = xlim
        self.ylim = ylim
        self.color = color
        self.color2 = color2
        self.labels = labels
        self.highlight_label = highlight_label


def show_figs(*args, sup_title=None, sup_titlesize=None, dpi=100, width_mm=120, height_mm=30, 
              margin_top_mm=15, margin_bottom_mm=15, margin_left_mm=25, margin_right_mm=15, margin_middle_mm=15,
              fold_interval=0, export_path="fig.png", is_display_console=False):
    num_figs_x = math.ceil(len(args)/fold_interval)
    num_figs_y = fold_interval
    #print('In show_figure(*args, sup_title=None, dpi=100, width_mm=120, height_mm=30)')
    #print("args: %d" % (num_figs))
    #print("figsize: ", (plt.rcParams["figure.figsize"]))
    #print("dpi: ", (plt.rcParams["figure.dpi"]))
    #print("fontsize:", plt.rcParams["font.size"])
    #print("supsize:", plt.rcParams["figure.titlesize"])

    font_size = plt.rcParams["font.size"]
    font_scale = font_scalings[plt.rcParams["axes.titlesize"]]
    title_size = font_size * font_scale
    if sup_titlesize:
        #print(sup_titlesize)
        if sup_titlesize in font_scalings:
            font_scale = font_scalings[sup_titlesize]
            title_size = font_size * font_scale
        else:
            title_size = sup_titlesize
        #print(title_size)

    #print("title fontsize: ", font_size)
    #print("title font scale: ", font_scale)    
    #print("title size: ", title_size)
    
    plt.style.use('dark_background')
    #width_mm = 120
    #height_mm = 30    
    #margin_top_mm = 15
    #margin_bottom_mm = 15
    #margin_left_mm = 25
    #margin_right_mm = 10
    #margin_middle_mm = 15
    mm_per_inch = 25.4
    total_height_mm = margin_top_mm + margin_bottom_mm + (margin_middle_mm + height_mm)*(num_figs_y-1) + height_mm
    total_width_mm = width_mm + margin_left_mm + margin_right_mm + (margin_middle_mm + width_mm)*(num_figs_x-1)
    
    width_inch = width_mm / mm_per_inch
    height_inch = height_mm / mm_per_inch
    margin_top_inch = margin_top_mm / mm_per_inch
    margin_bottom_inch = margin_bottom_mm / mm_per_inch
    margin_left_inch = margin_left_mm / mm_per_inch
    margin_middle_inch = margin_middle_mm / mm_per_inch
    total_height_inch = total_height_mm / mm_per_inch
    total_width_inch = total_width_mm / mm_per_inch
    
    fig = plt.figure(figsize=(total_width_inch, total_height_inch), dpi=dpi)    
    ax = []
    for idx in range(len(args)):
        height = height_inch / total_height_inch
        width = width_inch / total_width_inch
        x0 = (margin_left_inch + (width_inch + margin_middle_inch)*(num_figs_x - 1 - idx//fold_interval)) / total_width_inch
        y0 = (margin_bottom_inch +  (height_inch + margin_middle_inch)*(num_figs_y - 1 - idx%fold_interval)) / total_height_inch
        ax.append(fig.add_axes((x0, y0, width, height)))
        if type(args[idx]) is Figdata:
            if args[idx].type == None or args[idx].type == 'plot':
                if args[idx].color:
                    ax[idx].plot(args[idx].data, color=args[idx].color)
                else:                
                    ax[idx].plot(args[idx].data)
                data2 = np.array(args[idx].data2)
                if len(data2.shape) != 0:
                    if len(data2.shape) == 2:
                        for d in data2:
                            if args[idx].color2:
                                ax[idx].plot(d, color=args[idx].color2)
                            else:
                                ax[idx].plot(d)
                    else:
                        if args[idx].color2:
                            ax[idx].plot(data2, color=args[idx].color2)
                        else:
                            ax[idx].plot(data2)
                if args[idx].labels:
                    ax[idx].legend(args[idx].labels)
            elif args[idx].type == 'boxplot':
                if len(np.shape(args[idx].data2)) > 1:
                    ax[idx].boxplot((args[idx].data, *args[idx].data2), vert=False, showmeans=True, widths=0.7, labels=args[idx].labels)
                else:
                    if len(args[idx].data2) != 0:
                        ax[idx].boxplot((args[idx].data, args[idx].data2), vert=False, showmeans=True, widths=0.7, labels=args[idx].labels)
                    else:
                        ax[idx].boxplot(args[idx].data, vert=False, showmeans=True, widths=0.7, labels=args[idx].labels)
            elif args[idx].type == 'image' :
                im = plt.imshow(args[idx].data)
                if args[idx].data.shape[2] == 1:
                    plt.colorbar(im)
            elif args[idx].type == 'confusion_matrix' :
                heatmap = sns.heatmap(
                    args[idx].data,
                    annot=True,
                    xticklabels=args[idx].xticks,
                    yticklabels=args[idx].yticks,
                    fmt="d"
                )
                for i in range(len(args[idx].highlight_label)):
                    if args[idx].highlight_label[i] >= 0:
                        heatmap.get_xticklabels()[args[idx].highlight_label[i]].set_weight("bold")
                        heatmap.get_xticklabels()[args[idx].highlight_label[i]].set_color("#00ff00")
                        heatmap.get_yticklabels()[i].set_weight("bold")
                        heatmap.get_yticklabels()[i].set_color("#00ff00")
                        heatmap.add_patch(patches.Rectangle(xy=(args[idx].highlight_label[i], i), width=1, height=1, ec='#00ff00', fill=False, linewidth=3))
            if args[idx].xlabel:
                ax[idx].set_xlabel(args[idx].xlabel)
            if args[idx].ylabel:
                ax[idx].set_ylabel(args[idx].ylabel)
            if args[idx].title:
                ax[idx].set_title(args[idx].title, loc='left')
            if args[idx].xlim:
                ax[idx].set_xlim(args[idx].xlim)
            if args[idx].ylim:
                ax[idx].set_ylim(args[idx].ylim)
        else:
            ax[idx].plot(args[idx])
    if sup_title:
        fig.suptitle(sup_title, y=(1 - 0.3 * margin_top_inch / total_height_inch), fontsize=title_size)
    if is_display_console:
        plt.show()
    plt.savefig(export_path)
    print("export fig -> {}".format(export_path))
    plt.close()
    return


@ticker.FuncFormatter
def major_formatter_khz(y, pos):
    return '{:.0f}'.format(y/1000)

if __name__ == '__main__':
    
    #print("figsize: ", (plt.rcParams["figure.figsize"]))
    #print("dpi: ", (plt.rcParams["figure.dpi"]))

    n_fft:int = 2048
    n_shift:int = 1024
    n_overlap = n_fft // n_shift
    
    x = np.linspace(0, 2*np.pi, 2048)
    vector1 = np.cos(x)
    vector2 = np.sin(x)
    
    filename = "./avemaria.wav"
    # y, sr = librosa.load(filename, sr=None, mono=False)   # read all
    y1, sr = librosa.load(filename, sr=None, mono=False, offset=3.0, duration=1.0)   # read for 3 seconds to 1 second
    y2, sr = librosa.load(filename, sr=None, mono=False, offset=6.5, duration=1.0)   # read for 6.5 seconds to 1 second
    # if stereo 2ch, split int Lch and Rch
    y1_l = y1[0, :]
    y1_r = y1[1, :]
    y2_l = y2[0, :]
    y2_r = y2[1, :]
    
    ymax = max(max(y1_l),max(y1_r),max(y2_l),max(y2_r))
    ymin = min(min(y1_l),min(y1_r),min(y2_l),min(y2_l))
    #ymax = max(abs(ymax),abs(ymin))
    #ymin = -ymax
    #ymax = 1
    #ymin = -1
    
    S = librosa.stft(y2_l, n_fft=n_fft, window='hamm')    
    f0y_0 = Figdata(np.abs(S)[200], data2=np.full(len(np.abs(S[200])), 0.1), xlabel="freq", ylabel="magnitude", color='g', title="FFT: abs")
    f0y_1 = Figdata(np.log(np.abs(S)[200]), xlabel="freq", ylabel="magnitude", color='r', title="FFT: log(abs)")
    f0y_2 = Figdata(np.angle(S)[200], data2='5', xlabel="freq", ylabel="magnitude", title="FFT: angle")
    f1y_l = Figdata(y1_l, xlabel="freq", ylabel="magnitude", title="Fig1", ylim=(ymin,ymax))
    f1y_r = Figdata(y1_r, xlabel="freq", ylabel="magnitude", title="Fig2", ylim=(ymin,ymax))
    f2y_l = Figdata(y2_l, data2=np.full(len(y2_l), 0.3), color2='yellow', xlabel="freq", ylabel="magnitude", title="Fig3", ylim=(ymin,ymax))
    f2y_r = Figdata(y2_r, xlabel="freq", ylabel="magnitude", title="Fig4", ylim=(ymin,ymax))
    s1 = Figdata(np.random.randn(100), data2=np.random.randn(200), type='boxplot', color='r', ylabel='condition', xlabel='anomaly score', title='baseline AE', labels=['anomaly', 'normal'])
    
    show_figs(vector1, f0y_0, f0y_1, f0y_2, f1y_l, s1, f1y_r, f2y_l, f2y_r, sup_title="test", sup_titlesize='xx-large', dpi=70)
    