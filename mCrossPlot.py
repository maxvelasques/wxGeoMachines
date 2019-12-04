#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2019 maxvelasques

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
import os
from matplotlib import cm
from matplotlib import colorbar
from matplotlib import colors
from matplotlib import figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimize 
import lasio
import scipy.signal as signal

import wx
import wx.lib.agw.aui as aui
import wx.lib.mixins.inspection as wit

from mutil import ReadLAS
from mutil import ReadASC
from mutil import emptyplotLog
from mutil import plotLogs

##############################################################################
# DRAW MAIN WINDOW;
##############################################################################
class CrossPlotFrame(wx.Frame):

    def __init__(self, parent,title,filename,welllogs,progressdialog,alllogs=False, *args, **kwargs):
        wx.Frame.__init__(self, parent, title='Crossplot - ' + title, size=wx.Size(1000,600), style=wx.DEFAULT_FRAME_STYLE|wx.STAY_ON_TOP)

        self.welllogs = {}
        self.filename = filename
        self.welllogs = welllogs
        self.alllogs = alllogs
        self.progressdialog = progressdialog
        self.InitUI()


    def InitUI(self):
        self.plotter  = CrossPlot(self, self.filename, self.welllogs, self.progressdialog, self.alllogs)


class CrossPlot(wx.Panel):
    def __init__(self, parent, filename,welllogs, progressdialog, alllogs, id=-1, dpi=None, **kwargs):
        wx.Panel.__init__(self, parent, id=id, **kwargs)
        self.filename = filename
        self.figure = figure.Figure(dpi=dpi, figsize=(20, 20))
        self.axes = []
        subplots = 1
        self.axes.append(self.figure.add_subplot(1,subplots,1))

        self.canvas = FigureCanvas(self, -1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas,  1, wx.EXPAND)
        sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        self.SetSizer(sizer)
        self.welllogs = welllogs
        self.progressdialog = progressdialog
        self.alllogs = alllogs
        
        self.xplot()
    
    def xplot(self):
        self.progressdialog.Update(10)
        ax = self.axes[0]
        ax.set_title('RHOB x DT')
        ax.set_xlabel('DT ($\mu$s/ft)')
        ax.set_ylabel('RHOB (g/cc)')
        ax.grid(which='both')

        cmap = cm.get_cmap('jet')     #CREATE COLORBAR
        normalize = colors.Normalize(vmin=7, vmax=20)
        color = [cmap(normalize(value)) for value in self.welllogs[self.filename]['CALI']]
        if self.alllogs:
            ax.set_xlim(35 , 180)
            ax.set_ylim(1.2, 4.5)
            n_itens = len(self.welllogs.keys())
            i = 1
            for key in self.welllogs.keys():
                self.progressdialog.Update(10+int(80*i/n_itens))
                i+=1
                color = [cmap(normalize(value)) for value in self.welllogs[key]['CALI']]
                ax.scatter(self.welllogs[key]['DT'], self.welllogs[key]['RHOB'], color=color,marker='.')     #PLOT SCATTER WITH COLORS
        else:
            self.progressdialog.Update(50)
            ax.set_xlim(35 , 180)
            ax.set_ylim(1.2, 3.4)
            color = [cmap(normalize(value)) for value in self.welllogs[self.filename]['CALI']]
            ax.scatter(self.welllogs[self.filename]['DT'], self.welllogs[self.filename]['RHOB'], color=color,marker='.')     #PLOT SCATTER WITH COLORS

        cax, _ = colorbar.make_axes(ax)
        cbar = colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
        cbar.ax.set_ylabel('CALIPER')
        self.progressdialog.Update(100)


