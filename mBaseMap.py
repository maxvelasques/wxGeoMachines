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
from mpl_toolkits.mplot3d import Axes3D

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
from mutil import basemapplot

##############################################################################
# DRAW MAIN WINDOW;
##############################################################################
class BaseMapFrame(wx.Frame):

    def __init__(self, parent,coords, well_logs, title='Base Map - Wells location', *args, **kwargs):
        wx.Frame.__init__(self, parent, title=title, size=wx.Size(1200,600), style=wx.DEFAULT_FRAME_STYLE|wx.STAY_ON_TOP)
        self.coords = coords
        self.well_logs = well_logs
        self.InitUI()

    def InitUI(self):
        self.plotter  = BaseMap(self, self.coords,self.well_logs)


class BaseMap(wx.Panel):
    def __init__(self, parent, coords, well_logs, id=-1, dpi=None, **kwargs):
        wx.Panel.__init__(self, parent, id=id, **kwargs)
        self.coords = coords
        self.well_logs = well_logs
        self.figure = figure.Figure(dpi=dpi, figsize=(20, 20))
        self.axes = []
        subplots = 1
        self.axes.append(self.figure.add_subplot(1,subplots,1, projection='3d'))

        self.canvas = FigureCanvas(self, -1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas,  1, wx.EXPAND)
        sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        self.SetSizer(sizer)
        
        basemapplot(self.figure ,self.axes,self.coords,self.well_logs)


