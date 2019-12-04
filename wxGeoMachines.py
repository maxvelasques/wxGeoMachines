#!/usr/bin/env pythonw
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

print('Loading python moduli...')
import sys
import os
import multiprocessing
try:
    print('Setting multiprocessing method to spawn...', end=" ")
    multiprocessing.set_start_method('spawn')
    print('Done')
except:
    print('Spawn multiprocessing method has already been set.')

try:
    from matplotlib import cm
    from matplotlib import colorbar
    from matplotlib import colors
    from matplotlib import figure
    from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
    from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
    import matplotlib.pyplot as plt
except:
    print('matplotlib not available.')
    exit(0)

try:
    import numpy as np
except:
    print('numpy not available.')
    exit(0)

try:
    import pandas as pd
except:
    print('pandas not available.')
    exit(0)

try:
    import scipy.optimize as optimize 
    import scipy.signal as signal
except:
    print('scipy not available.')
    exit(0)

try:
    import csv
except:
    print('csv not available.')
    exit(0)

try:
    import lasio
except:
    print('lasio not available.')
    exit(0)

try:
    from sklearn import cluster
    from sklearn.preprocessing import StandardScaler
except:
    print('Sklearn not available. Install sklearn before generating logs')

try:
    import wx
    import wx.lib.agw.aui as aui
    import wx.lib.mixins.inspection as wit
    import wx.adv
except:
    print('wxPython not available.')
    exit(0)

from mutil import ReadLAS
from mutil import ReadASC
from mutil import WriteLAS
from mutil import ReadCoords
from mutil import emptyplotLog
from mutil import plotLogs
from mutil import PrepareStringStatus
from mutil import PrepareStringError
from mutil import concatenate_pandas
from mutil import split_pandas
from mutil import dicts_are_equal
from mutil import ffnn_predict
from mutil import random_forest_predict
from mutil import gardner_predict
from mCrossPlot import CrossPlotFrame
from mBaseMap import BaseMapFrame


##############################################################################
# DRAW MAIN WINDOW;
##############################################################################
class mainFrame(wx.Frame):

    def __init__(self, *args, **kwargs):
        super(mainFrame, self).__init__(*args, **kwargs)
        self.InitUI()
        self.welllogs = {}
        self.welllogs_pre = {}
        self.welllogs_ffnn = {}
        self.welllogs_ffnn_error = {}
        self.welllogs_random_forest = {}
        self.welllogs_random_forest_error = {}
        self.welllogs_gardner = {}
        self.welllogs_gardner_error = {}
        self.preprocessing_pars = {}


    def InitUI(self):
        # Setting up the menu.
        datamenu = wx.Menu()
        self.Bind(wx.EVT_MENU, self.OnImport,    datamenu.Append(101, "Import well", "Import"))
        self.Bind(wx.EVT_MENU, self.OnBaseMap,   datamenu.Append(102, "Generate Base Map", "Base Map"))
        self.Bind(wx.EVT_MENU, self.OnExport,    datamenu.Append(103, "Export well", "Export"))
        self.Bind(wx.EVT_MENU, self.OnAbout,     datamenu.Append(wx.ID_ABOUT, "About GeoMachines", "About"))

        # datamenu.Append(wx.ID_ABOUT, "About","About")
        datamenu.Append(wx.ID_EXIT,"Exit","Close")
        
        preprosmenu = wx.Menu()
        self.Bind(wx.EVT_MENU, self.OnExportFlow,preprosmenu.Append(201, "Export workflow", "Export workflow"))
        self.Bind(wx.EVT_MENU, self.OnImportFlow,preprosmenu.Append(202, "Import and Apply workflow", "Import and Apply workflow"))
         
        # Creating the menubar.
        menuBar = wx.MenuBar()
        menuBar.Append(datamenu,   "Data")
        menuBar.Append(preprosmenu,"Pre-Processing")


        self.SetMenuBar(menuBar)  # Adding the MenuBar to the Frame content.
        self.SetTitle('GeoMachines')
        self.Centre()
        
        #CREATE A SPLITTER
        splitter = wx.SplitterWindow(self)
        self.BasicControls = BasicControls(splitter,self) ##LEFT SIDE
        self.plotter       = PlotNotebook(splitter) #RIGHT SIDE 
        splitter.SplitVertically(self.BasicControls, self.plotter)
        splitter.SetMinimumPaneSize(200)
        splitter.SetSashPosition(200)

        fig,axes = self.plotter.add('empty')
        emptyplotLog(fig,axes)

    def OnAbout(self, event):
        aboutInfo = wx.adv.AboutDialogInfo()
        aboutInfo.SetName("GeoMachines")
        aboutInfo.SetVersion("1.0")

        aboutInfo.SetDescription("""
Final project of CSCI-470 Machine Learning course at Colorado School
of Mines. Developed by GeoMachines group, this application is
capable of generating sonic logs using Machine Learning methods.
        """)

        aboutInfo.SetCopyright("(C) 2019-2019")
        aboutInfo.SetWebSite("https://github.com/maxvelasques/wxGeoMachines")
        aboutInfo.AddDeveloper("Max Velasques\nAndrea Damasceno\nAtilas Silva\nSamuel Chambers\nMeng Jia")

        wx.adv.AboutBox(aboutInfo)
        return

    def OnImport(self, e):
        dialog = wx.FileDialog(self,message="Select well logs",wildcard="LAS files (*.LAS; *.ASC)|*.las;*.LAS;*.asc;*.ASC",style=wx.FD_OPEN | wx.FD_MULTIPLE)
        if dialog.ShowModal() == wx.ID_CANCEL:
            return 0;
        progressdialog = wx.ProgressDialog('Progress dialog', message='Loading data...')

        filenames = dialog.GetFilenames()
        directory = dialog.GetDirectory()
        n_itens = len(filenames)
        i=1
        for filename in filenames:
            progressdialog.Update(int(100*i/n_itens))
            i+=1
            fullpath = os.path.join(directory,filename)
            if os.path.splitext(filename)[1] == '.LAS':
                print('\nReading LAS: "' + filename + '" ...')
                welllog = ReadLAS(fullpath)
                self.welllogs[filename] = welllog
                # self.welllogs_pre[filename] = welllog.copy()
                print(welllog.columns)
                fig,axes = self.plotter.add(filename)
                fig.gca()
                plotLogs(filename,fig,axes ,welllog)
            elif os.path.splitext(filename)[1] == '.ASC':
                print('\nReading ASC: "' + filename + '" ...')
                welllog = ReadASC(fullpath)
                self.welllogs[filename] = welllog
                # self.welllogs_pre[filename] = welllog.copy()
                print(welllog.columns)
                fig,axes = self.plotter.add(filename)
                fig.gca()
                plotLogs(filename,fig,axes ,welllog)
        progressdialog.Close()

    def OnExport(self, e):
        original_filename = self.plotter.nb.GetPageText(self.plotter.nb.GetSelection())
        if original_filename == 'empty':
            return wx.MessageBox('No file available.', "Warning", wx.OK | wx.ICON_WARNING)

        with wx.FileDialog(self,message="Save current well log",wildcard="LAS files (*.LAS)|*.LAS",style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dialog:
            if dialog.ShowModal() != wx.ID_CANCEL:
                fullpath = dialog.GetPath()
                if os.path.exists(fullpath):
                    print('Deleting "' + fullpath + '"...')
                    os.remove(fullpath)

                if original_filename in self.welllogs_pre:
                    well_log = self.welllogs_pre[original_filename]
                else:
                    well_log = self.welllogs[original_filename]

                well_log = well_log.copy()

                if original_filename in self.welllogs_ffnn:
                    well_log_ffnn = self.welllogs_ffnn[original_filename]
                    well_log['DT'] = well_log_ffnn['DT']

                WriteLAS(fullpath,well_log)
                # well_log.to_csv(fullpath + ".csv")

                return wx.MessageBox('LAS file exported successfully.', "Export LAS file", wx.OK | wx.ICON_INFORMATION)
        
    def OnBaseMap(self, e):
        dialog = wx.FileDialog(self,message="Select well coordinates",wildcard="XLSX files (*.XLS*)|*.xlsx;*.xls;*.XLSX;*.XLS",style=wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_CANCEL:
            return 0;
        coords_in = ReadCoords(dialog.GetPath())
        columns_names = list(coords_in.columns)
        coords_out = pd.DataFrame(columns=columns_names)
        for j in range(self.plotter.nb.GetPageCount()):
            filename = self.plotter.nb.GetPageText(j)
            if filename in coords_in.Filename.values:
                coords_out = coords_out.append(coords_in[coords_in.Filename == filename])
        frame = BaseMapFrame(self,coords_out,self.welllogs)
        frame.Show()


    def OnCrossPlot(self, e):
        dialog = wx.ProgressDialog('Progress dialog', message='Generating cross plot...')
        filename = self.plotter.nb.GetPageText(self.plotter.nb.GetSelection())
        title = self.BasicControls.QCPanel.choice.GetString(self.BasicControls.QCPanel.choice.GetCurrentSelection())
        if title == 'Current well - Original':
            frame = CrossPlotFrame(self,title,filename,self.welllogs, dialog, alllogs = False)
        elif title == 'Current well - Pre-processed':
            frame = CrossPlotFrame(self,title,filename,self.welllogs_pre, dialog, alllogs = False)
        elif title == 'All wells - Original':
            frame = CrossPlotFrame(self,title,filename,self.welllogs, dialog, alllogs = True)
        elif title == 'All wells - Pre-processed':
            frame = CrossPlotFrame(self,title,filename,self.welllogs_pre,dialog, alllogs = True)
        dialog.Close()
        frame.Show()

    def PreProcess(self,filename, welllogs, current_preprocessing_pars,progressdialog,alllogs=False):    
        filter_size = current_preprocessing_pars['filter_size']
        cut = current_preprocessing_pars['cut']
        if alllogs:
            n_itens = len(welllogs.keys())
            i = 1
            for key in welllogs.keys():
                progressdialog.Update(10+int(30*i/n_itens))
                i+=1
                well_tmp = welllogs[key].copy()
                #CUT THE BEGINING OF THE CURRENT WELLLOG
                if current_preprocessing_pars['cut_check']:
                    print('   Cutting log...')
                    mindepth = well_tmp.DEPT.min()
                    well_tmp = well_tmp[ well_tmp.DEPT > (mindepth + cut) ]

                #DESPIKE
                if current_preprocessing_pars['filter_check']:
                    print('   Despiking log...')
                    well_tmp.GR   = signal.medfilt(well_tmp['GR'].values,filter_size)
                    well_tmp.RHOB = signal.medfilt(well_tmp['RHOB'].values,filter_size)
                    well_tmp.NPHI = signal.medfilt(well_tmp['NPHI'].values,filter_size)
                    well_tmp.DT   = signal.medfilt(well_tmp['DT'].values,filter_size)
                    well_tmp.RESI = signal.medfilt(well_tmp['RESI'].values,filter_size)
                welllogs[key] = well_tmp

            #DBSCAN
            progressdialog.Update(50)
            print('All wells')
            if current_preprocessing_pars['dbscan_check']:
                print('   Applying DBSCAN...')
                ##############################################################
                merged_welllogs = concatenate_pandas(welllogs)
                rhob_dt_scaled = StandardScaler().fit_transform(merged_welllogs[['RHOB','DT']].values)
                progressdialog.Update(60)
                XX = cluster.DBSCAN(eps=current_preprocessing_pars['dbscan_eps'], min_samples=current_preprocessing_pars['dbscan_minneigh']).fit(rhob_dt_scaled)
                y_pred = XX.labels_
                y_pred = y_pred.astype(float)
                y_pred[y_pred == (-1)] = np.nan
                merged_welllogs['outliers'] = y_pred
                merged_welllogs = merged_welllogs.dropna()
                merged_welllogs = merged_welllogs.drop(columns=['outliers'])
                welllogs = split_pandas(merged_welllogs)
                ##############################################################
        else:
            progressdialog.Update(30)
            well_tmp = welllogs[filename].copy()
            #CUT THE BEGINING OF THE CURRENT WELLLOG
            print(filename)
            if current_preprocessing_pars['cut_check']:
                print('   Cutting log...')
                mindepth = well_tmp.DEPT.min()
                well_tmp = well_tmp[ well_tmp.DEPT > (mindepth + cut) ]

            progressdialog.Update(50)
            #DESPIKE
            if current_preprocessing_pars['filter_check']:
                print('   Despiking log...')
                well_tmp.GR   = signal.medfilt(well_tmp.GR.values,filter_size)
                well_tmp.RHOB = signal.medfilt(well_tmp.RHOB.values,filter_size)
                well_tmp.NPHI = signal.medfilt(well_tmp.NPHI.values,filter_size)
                well_tmp.DT   = signal.medfilt(well_tmp.DT.values,filter_size)
                well_tmp.RESI = signal.medfilt(well_tmp.RESI.values,filter_size)
            progressdialog.Update(80)
            #DBSCAN
            if current_preprocessing_pars['dbscan_check']:
                print('   Applying DBSCAN...')
                ##############################################################
                rhob_dt_scaled = StandardScaler().fit_transform(well_tmp[['RHOB','DT']].values)
                XX = cluster.DBSCAN(eps=current_preprocessing_pars['dbscan_eps'], min_samples=current_preprocessing_pars['dbscan_minneigh']).fit(rhob_dt_scaled)
                y_pred = XX.labels_
                y_pred = y_pred.astype(float)
                #y_pred[y_pred != (-1)] = (0)
                y_pred[y_pred == (-1)] = np.nan
                well_tmp['outliers'] = y_pred
                well_tmp = well_tmp.dropna()
                ##############################################################

            welllogs[filename] = well_tmp
        return welllogs

    def OnPrePross(self, e):
        dialog = wx.ProgressDialog('Progress dialog', message='Processing data...')

        filename = self.plotter.nb.GetPageText(self.plotter.nb.GetSelection())
        title = self.BasicControls.PreProsPanel.choice.GetString(self.BasicControls.PreProsPanel.choice.GetCurrentSelection())

        cut_check = int(self.BasicControls.PreProsPanel.checkbox_cut.IsChecked())
        cut = self.BasicControls.PreProsPanel.spinctrl_cut.GetValue()

        dbscan_check = int(self.BasicControls.PreProsPanel.checkbox_dbscan.IsChecked())
        dbscan_eps = self.BasicControls.PreProsPanel.spinctrl_eps.GetValue()
        dbscan_minneigh = self.BasicControls.PreProsPanel.spinctrl_minneigh.GetValue()

        filter_check = int(self.BasicControls.PreProsPanel.checkbox_median.IsChecked())
        filter_size = self.BasicControls.PreProsPanel.spinctrl_median.GetValue()

        if filter_size%2 == 0:
            filter_size+=1

        current_preprocessing_pars = {'cut_check': cut_check,
                                     'cut':cut,
                                     'dbscan_check':dbscan_check,
                                     'dbscan_eps':dbscan_eps,
                                     'dbscan_minneigh':dbscan_minneigh,
                                     'filter_check':filter_check,
                                     'filter_size':filter_size
                                     }

        preparamstr = PrepareStringStatus(current_preprocessing_pars)
        last_selection = self.plotter.nb.GetSelection()
        dialog.Update(10)
        if title == 'Current well - Original':
            self.welllogs_pre[filename] = self.welllogs[filename]
            self.preprocessing_pars[filename] = current_preprocessing_pars
            self.welllogs_pre = self.PreProcess(filename, self.welllogs_pre.copy(), self.preprocessing_pars[filename],dialog,alllogs=False)
            fig,axes = self.plotter.update(filename,self.plotter.nb.GetSelection())
            fig.gca()
            plotLogs(filename,fig,axes ,self.welllogs[filename],self.welllogs_pre[filename],paramstr=preparamstr)
            self.plotter.RefreshPlot(self.plotter.nb.GetSelection())
        elif title == 'All wells - Original':
            self.welllogs_pre = self.PreProcess(filename, self.welllogs.copy(), current_preprocessing_pars,dialog,alllogs=True)

            n_pages = self.plotter.nb.GetPageCount()
            filenames = []
            for i in range(n_pages):
                filenames.append(self.plotter.nb.GetPageText(i))
            
            dialog.Update(60)
            j = 1
            for i in reversed(range(n_pages)):
                dialog.Update(60+int(30*j/n_pages))
                j+=1
                filename = filenames[i]
                fig,axes = self.plotter.update(filename,i)
                fig.gca()
                plotLogs(filename,fig,axes ,self.welllogs[filename],self.welllogs_pre[filename],paramstr=preparamstr)
                self.plotter.RefreshPlot(i)

                self.preprocessing_pars[filename] = current_preprocessing_pars
        self.plotter.nb.SetSelection(last_selection)
        dialog.Update(100)
        dialog.Close()


    def OnImportFlow(self, e):
        dialog = wx.FileDialog(self,message="Import workflow",wildcard="Flow files (*.flow)|*.flow",style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() != wx.ID_CANCEL:
            progressdialog = wx.ProgressDialog('Progress dialog', message='Processing data...')

            fullpath = dialog.GetPath()
            readCSV = csv.reader(open(fullpath, "r"))
            for row in readCSV:
                filename = row[0]
                row.remove(filename)
                current_preprocessing_pars = {}
                for key, value in zip(row[::2],row[1::2]):
                    try:
                        current_preprocessing_pars[key] = int(value)
                    except ValueError:
                        current_preprocessing_pars[key] = float(value)

                n_pages = self.plotter.nb.GetPageCount()
                filenames = []
                for j in range(n_pages):
                    filenames.append(self.plotter.nb.GetPageText(j))
                for j in range(n_pages):
                    if filename == filenames[j]:
                        self.welllogs_pre[filename] = self.welllogs[filename]
                        self.preprocessing_pars[filename] = current_preprocessing_pars
                        break
            
            j = 0
            alllogs = True
            filename = ''
            for key in self.preprocessing_pars.keys():
                if j == 0:
                    reference_par = self.preprocessing_pars[key]
                    filename = key
                    j+=1
                else:
                    if dicts_are_equal(reference_par,self.preprocessing_pars[key]) == False:
                        alllogs = False
                        break
            if alllogs:
                self.welllogs_pre = self.PreProcess(filename, self.welllogs_pre.copy(), self.preprocessing_pars[filename],progressdialog,alllogs=True)

            for j in range(n_pages):
                filename = self.plotter.nb.GetPageText(j)
                preparamstr = PrepareStringStatus(self.preprocessing_pars[filename])

                #UPDATE
                if alllogs == False:
                    self.welllogs_pre = self.PreProcess(filename, self.welllogs_pre.copy(), self.preprocessing_pars[filename],progressdialog,alllogs=False)

                fig,axes = self.plotter.update(filename,j)
                fig.gca()
                plotLogs(filename,fig,axes ,self.welllogs[filename],self.welllogs_pre[filename],paramstr=preparamstr)
                self.plotter.RefreshPlot(j)
            
            progressdialog.Update(100)
            progressdialog.Close()

            # return wx.MessageBox('Pre-processing workflow imported successfully.', "Import workflow", wx.OK | wx.ICON_INFORMATION)


    def OnExportFlow(self, e):
        dialog = wx.FileDialog(self,message="Export workflow",wildcard="Flow files (*.flow)|*.flow",style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dialog.ShowModal() != wx.ID_CANCEL:
            fullpath = dialog.GetPath()
            w = csv.writer(open(fullpath, "w"))
            for key, vals in self.preprocessing_pars.items():
                row = []
                row.append(key)
                for subkey,subval in vals.items():
                    row.append(subkey)
                    row.append(subval)
                w.writerow(row)

            return wx.MessageBox('Pre-processing workflow exported successfully.', "Export workflow", wx.OK | wx.ICON_INFORMATION)

    def OnPredictLog(self, e):

        filename   = self.plotter.nb.GetPageText(self.plotter.nb.GetSelection())

        if filename in self.welllogs:
            progressdialog = wx.ProgressDialog('Progress dialog', message='Predicting sonic log...')
            progressdialog.Update(20)
            title      = self.BasicControls.MLPanel.choicewells.GetString(self.BasicControls.MLPanel.choicewells.GetCurrentSelection())
            model_name = self.BasicControls.MLPanel.choicemodel.GetString(self.BasicControls.MLPanel.choicemodel.GetCurrentSelection())
            welllog_orig    = self.welllogs[filename]
            welllog_pre     = None
            welllog_ffnn    = None
            welllog_ffnn_error = 0.0
            welllog_random_forest       = None
            welllog_random_forest_error = 0.0
            welllog_gardner = None
            welllog_gardner_error = 0.0
            preparamstr = ""
            errorstr    = ""

            preprocessed = False

            if title == 'Current well - Pre-processed':
                if filename in self.welllogs_pre:
                    welllog_pre = self.welllogs_pre[filename]
                    preprocessed = True
                else:
                    progressdialog.Update(100)
                    progressdialog.Close()
                    return wx.MessageBox('There is no pre-processed log for this well.', "Warning", wx.OK | wx.ICON_WARNING)

                preparamstr = PrepareStringStatus(self.preprocessing_pars[filename])

            if filename in self.welllogs_gardner:
                welllog_gardner = self.welllogs_gardner[filename]
                welllog_gardner_error = self.welllogs_gardner_error[filename]
            if filename in self.welllogs_ffnn:
                welllog_ffnn = self.welllogs_ffnn[filename]
                welllog_ffnn_error = self.welllogs_ffnn_error[filename]
            if filename in self.welllogs_random_forest:
                welllog_random_forest = self.welllogs_random_forest[filename]
                welllog_random_forest_error = self.welllogs_random_forest_error[filename]

            if 'FFNN' in model_name:
                progressdialog.Update(50)
                
                queue = multiprocessing.Queue()
                if preprocessed:
                    p1 = multiprocessing.Process(target=ffnn_predict, args=(queue,welllog_pre,)) 
                else:
                    p1 = multiprocessing.Process(target=ffnn_predict, args=(queue,welllog_orig,))                 
                p1.start() 
                result = queue.get()
                p1.join() 

                welllog_ffnn = result['log']
                welllog_ffnn_error = result['error']
                
                self.welllogs_ffnn[filename] = welllog_ffnn
                self.welllogs_ffnn_error[filename] = welllog_ffnn_error
                progressdialog.Update(80)
            elif 'Random Forest' in model_name:
                progressdialog.Update(50)

                if preprocessed:
                    welllog_random_forest,welllog_random_forest_error = random_forest_predict(welllog_pre)
                else:
                    welllog_random_forest,welllog_random_forest_error = random_forest_predict(welllog_orig)
                self.welllogs_random_forest[filename] = welllog_random_forest
                self.welllogs_random_forest_error[filename] = welllog_random_forest_error
                progressdialog.Update(80)
            elif 'Gardner' in model_name:
                progressdialog.Update(50)

                if preprocessed:
                    welllog_gardner,welllog_gardner_error = gardner_predict(welllog_pre)
                else:
                    welllog_gardner,welllog_gardner_error = gardner_predict(welllog_orig)
                self.welllogs_gardner[filename] = welllog_gardner
                self.welllogs_gardner_error[filename] = welllog_gardner_error
                progressdialog.Update(80)
            else:
                progressdialog.Update(100)
                progressdialog.Close()
                return wx.MessageBox('The selected model is not available.', "Warning", wx.OK | wx.ICON_WARNING)

            errorstr = PrepareStringError(welllog_ffnn_error,welllog_random_forest_error,welllog_gardner_error)
            fig,axes = self.plotter.update(filename,self.plotter.nb.GetSelection())
            fig.gca()
            plotLogs(filename,fig,axes ,welllog_orig, welllog_pre=welllog_pre, welllog_ffnn=welllog_ffnn,welllog_random_forest=welllog_random_forest,welllog_gardner=welllog_gardner,paramstr=preparamstr,errorstr=errorstr)
            self.plotter.RefreshPlot(self.plotter.nb.GetSelection())
            progressdialog.Update(100)
            progressdialog.Close()
            return


##############################################################################
# MAIN PANEL - RIGHT
##############################################################################

class Plot(wx.Panel):
    def __init__(self, parent, id=-1, dpi=None, **kwargs):
        wx.Panel.__init__(self, parent, id=id, **kwargs)
        self.figure = figure.Figure(dpi=dpi, figsize=(20, 20))
        self.axes = []
        subplots = 6
        for i in range(1,subplots+1):
            if i == 1:
                self.axes.append(self.figure.add_subplot(1,subplots,i))
            else:
                self.axes.append(self.figure.add_subplot(1,subplots,i,sharey=self.axes[0]))
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas  , 1, wx.EXPAND)
        sizer.Add(self.toolbar , 0, wx.LEFT | wx.EXPAND)
        self.SetSizer(sizer)

    def clear(self):
        self.figure.clf()
        self.axes = []
        subplots = 6
        for i in range(1,subplots+1):
            if i == 1:
                self.axes.append(self.figure.add_subplot(1,subplots,i))
            else:
                self.axes.append(self.figure.add_subplot(1,subplots,i,sharey=self.axes[0]))

class PlotNotebook(wx.Panel):
    def __init__(self, parent, id=-1):
        wx.Panel.__init__(self, parent, id=id)
        self.parent = parent
        self.nb = aui.AuiNotebook(self)
        sizer = wx.BoxSizer()
        sizer.Add(self.nb, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def add(self, name="plot"):
        page = Plot(self.nb)
        if self.nb.GetPageCount() > 0:
            if self.nb.GetPageText(0) == 'empty':
                self.nb.DeletePage(0)
        self.nb.AddPage(page, name)
        return page.figure,page.axes

    def update(self, name,page_number):
        page = self.nb.GetPage(page_number)
        page.clear()

        # self.nb.DeletePage(page_number)
        # page = Plot(self.nb)
        # self.nb.InsertPage(page_number, page, name)
        return page.figure,page.axes

    def RefreshPlot(self,page_number):
        page = self.nb.GetPage(page_number)
        page.canvas.draw()





##############################################################################
# LATERAL BAR - LEFT
##############################################################################
class BasicControls(wx.Panel):
    def __init__(self, parent, mainframe, id=-1):
        wx.Panel.__init__(self, parent, id=id)
        self.parent = parent
        font = wx.SystemSettings.GetFont(wx.SYS_SYSTEM_FONT)

        font.SetPointSize(9)
        
        #CREATE CONTROLS SUBPANELS
        vbox = wx.BoxSizer(wx.VERTICAL)

        vbox1 = wx.BoxSizer(wx.HORIZONTAL)
        vbox1.Add(ImportPanel(self,mainframe), flag=wx.EXPAND, border=2)
        vbox.Add(vbox1, flag=wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP, border=2)

        vbox.Add((-1, 7))
        
        vbox2 = wx.BoxSizer(wx.HORIZONTAL)
        self.QCPanel = QCPanel(self,mainframe)
        vbox2.Add(self.QCPanel, flag=wx.EXPAND, border=2)
        vbox.Add(vbox2, flag=wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP, border=2)

        vbox.Add((-1, 7))
        
        vbox3 = wx.BoxSizer(wx.HORIZONTAL)
        self.PreProsPanel = PreProsPanel(self,mainframe)
        vbox3.Add(self.PreProsPanel, flag=wx.EXPAND, border=2)
        vbox.Add(vbox3, flag=wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP, border=2)

        vbox.Add((-1, 7))
        
        vbox4 = wx.BoxSizer(wx.HORIZONTAL)
        self.MLPanel = MLPanel(self,mainframe)
        vbox4.Add(self.MLPanel, flag=wx.EXPAND, border=2)
        vbox.Add(vbox4, flag=wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP, border=2)

        vbox.Add((-1, 7))

        vbox5 = wx.BoxSizer(wx.HORIZONTAL)
        vbox5.Add(ExportPanel(self,mainframe), flag=wx.EXPAND, border=2)
        vbox.Add(vbox5, flag=wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP, border=2)

        self.SetSizer(vbox)

class ImportPanel(wx.Panel):
    def __init__(self, parent, mainframe, id=-1):
        wx.Panel.__init__(self, parent, id=id)
        self.parent = parent
        self.SetBackgroundColour('white')
        self.mainframe = mainframe

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.SetMinSize(200,20)
        sizer.AddSpacer(5)
        sizer.Add(wx.StaticText(self, label='- Data loading -'), flag=wx.RIGHT, border=2)
        sizer.AddSpacer(5)
        button = wx.Button(self, label='Import well logs...')
        button.Bind(wx.EVT_BUTTON, self.mainframe.OnImport)
        sizer.Add(button, proportion=1, flag=wx.ALIGN_RIGHT, border=2)
        sizer.AddSpacer(5)
        self.SetSizer(sizer)

class QCPanel(wx.Panel):
    def __init__(self, parent, mainframe, id=-1):
        wx.Panel.__init__(self, parent, id=id)
        self.parent = parent
        self.SetBackgroundColour('white')
        self.mainframe = mainframe

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.SetMinSize(200,20)

        sizer.AddSpacer(5)
        
        sizer.Add(wx.StaticText(self, label='- Quality Control -'), flag=wx.ALIGN_LEFT, border=2)
        
        sizer.AddSpacer(5)
        
        self.choice = wx.Choice(self, choices=['Current well - Original',
                                         'Current well - Pre-processed',
                                         'All wells - Original',
                                         'All wells - Pre-processed'])
        self.choice.SetSelection(0)
        sizer.Add(self.choice, proportion=1, flag=wx.ALIGN_RIGHT, border=2)

        sizer.AddSpacer(5)
    
        button = wx.Button(self, label='Generate CrossPlots')
        button.Bind(wx.EVT_BUTTON, self.mainframe.OnCrossPlot)
        sizer.Add(button, proportion=1, flag=wx.ALIGN_RIGHT, border=2)
        
        sizer.AddSpacer(5)
        self.SetSizer(sizer)

class PreProsPanel(wx.Panel):
    def __init__(self, parent, mainframe, id=-1):
        wx.Panel.__init__(self, parent, id=id)
        self.parent = parent
        self.SetBackgroundColour('white')
        self.mainframe = mainframe

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.SetMinSize(200,20)

        sizer.AddSpacer(5)
        sizer.Add(wx.StaticText(self, label='- Pre-processing -'), flag=wx.ALIGN_LEFT, border=2)
        sizer.AddSpacer(2)
        
        self.choice = wx.Choice(self, choices=['Current well - Original',
                                               'All wells - Original'])
        self.choice.SetSelection(0)
        sizer.Add(self.choice, proportion=1, flag=wx.ALIGN_LEFT, border=2)
        sizer.AddSpacer(5)

        self.checkbox_cut = wx.CheckBox(self, label='Cut log (in meters):')
        sizer.Add(self.checkbox_cut, flag=wx.ALIGN_LEFT, border=2)
        sizer.AddSpacer(2)     
        self.spinctrl_cut = wx.SpinCtrl(self, initial=0, min=0,max=1000)
        sizer.Add(self.spinctrl_cut, proportion=1, flag=wx.ALIGN_LEFT, border=2)
        sizer.AddSpacer(20)

        self.checkbox_median = wx.CheckBox(self, label='Median filter (in samples):')
        sizer.Add(self.checkbox_median, flag=wx.ALIGN_LEFT, border=2)
        sizer.AddSpacer(2)
        self.spinctrl_median = wx.SpinCtrl(self, initial=21, min=3,max=101)
        sizer.Add(self.spinctrl_median, proportion=1, flag=wx.ALIGN_LEFT, border=2)
        sizer.AddSpacer(20)

        self.checkbox_dbscan = wx.CheckBox(self, label='Outlier detector (DBScan):')
        sizer.Add(self.checkbox_dbscan, flag=wx.ALIGN_LEFT, border=2)
        sizer.AddSpacer(2)
        sizer.Add(wx.StaticText(self, label='EPS:'), flag=wx.ALIGN_LEFT, border=2)
        # self.spinctrl_eps = wx.SpinCtrlDouble(self, initial=0.15, min=0.10,max=10.00, inc=0.05)
        self.spinctrl_eps = wx.SpinCtrlDouble(self, initial=0.30, min=0.10,max=10.00, inc=0.05)
        sizer.Add(self.spinctrl_eps, proportion=1, flag=wx.ALIGN_LEFT, border=2)

        sizer.Add(wx.StaticText(self, label='Minimum neighbors:'), flag=wx.ALIGN_LEFT, border=2)
        # self.spinctrl_minneigh = wx.SpinCtrl(self, initial=64, min=3,max=100)
        self.spinctrl_minneigh = wx.SpinCtrl(self, initial=30, min=3,max=100)

        sizer.Add(self.spinctrl_minneigh, proportion=1, flag=wx.ALIGN_LEFT, border=2)
        sizer.AddSpacer(20)

        button = wx.Button(self, label='Apply')
        button.Bind(wx.EVT_BUTTON, self.mainframe.OnPrePross)
        sizer.Add(button, proportion=1, flag=wx.ALIGN_RIGHT, border=2)
        sizer.AddSpacer(5)

        self.SetSizer(sizer)

class MLPanel(wx.Panel):
     def __init__(self, parent, mainframe, id=-1):
        wx.Panel.__init__(self, parent, id=id)
        self.parent = parent
        self.SetBackgroundColour('white')
        self.mainframe = mainframe

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.SetMinSize(200,20)

        sizer.AddSpacer(5)
        
        sizer.Add(wx.StaticText(self, label='- Sonic log prediction -'), flag=wx.ALIGN_LEFT, border=2)
        
        sizer.AddSpacer(5)

        self.choicewells = wx.Choice(self, choices=['Current well - Original',
                                                    'Current well - Pre-processed',
                                                    ])
        self.choicewells.SetSelection(0)

        sizer.Add(self.choicewells, proportion=1, flag=wx.ALIGN_RIGHT, border=2)

        sizer.AddSpacer(5)
        sizer.Add(wx.StaticText(self, label='Select model:'), flag=wx.ALIGN_LEFT, border=2)
        sizer.AddSpacer(2)
        self.choicemodel = wx.Choice(self, choices=['Empirical (Gardner)',
                                                    'ML (FFNN)',
                                                    'ML (Random Forest)',
                                                    'ML (RNN)'
                                                   ])
        self.choicemodel.SetSelection(0)

        sizer.Add(self.choicemodel, proportion=1, flag=wx.ALIGN_RIGHT, border=2)

        sizer.AddSpacer(5)

        sizerh = wx.BoxSizer(wx.HORIZONTAL)
        predict_button = wx.Button(self, label='Predict log')
        predict_button.Bind(wx.EVT_BUTTON, self.mainframe.OnPredictLog)
        sizerh.Add(predict_button, proportion=1, flag=wx.ALIGN_RIGHT, border=2)

        sizer.Add(sizerh, proportion=1, flag=wx.ALIGN_RIGHT, border=2)

        sizer.AddSpacer(5)
        self.SetSizer(sizer)


class ExportPanel(wx.Panel):
    def __init__(self, parent, mainframe, id=-1):
        wx.Panel.__init__(self, parent, id=id)
        self.parent = parent
        self.SetBackgroundColour('white')
        self.mainframe = mainframe

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.SetMinSize(200,20)
        sizer.AddSpacer(5)
        sizer.Add(wx.StaticText(self, label='- Export Data -'), flag=wx.RIGHT, border=2)
        sizer.AddSpacer(5)
        button = wx.Button(self, label='Export well log...')
        button.Bind(wx.EVT_BUTTON, self.mainframe.OnExport)
        sizer.Add(button, proportion=1, flag=wx.ALIGN_RIGHT, border=2)
        sizer.AddSpacer(5)
        self.SetSizer(sizer)

##############################################################################
# MAIN FUNCTION;
##############################################################################
def main():
    # plt.close('all')
    # if os.path.exists('{}/PLOTS'.format(os.getcwd())) == False:
    #     os.mkdir('{}/PLOTS'.format(os.getcwd()))
    
    app = wx.App()
    app.SetAppName('GeoMachines')
    frame = mainFrame(None, -1, 'GeoMachines',size=wx.Size(1600,800))
    frame.Show()
    app.MainLoop()

if __name__ == "__main__":
    main()
