# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 17:38:08 2025

@author: fredd
"""

# trace generated using paraview version 5.12.0
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 12

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# -----------------------------------------------------------------------------
# File locations
# -----------------------------------------------------------------------------
import sys
import glob
import numpy as np
from pathlib import Path

# try to infer the VTU path from the active source
volume_source = GetActiveSource()
volume_vtu = ''
if volume_source is not None:
    file_prop = getattr(volume_source, 'FileName', '')
    if isinstance(file_prop, (list, tuple)):
        volume_vtu = file_prop[0]
    elif isinstance(file_prop, str):
        volume_vtu = file_prop

# fall back to command-line argument or glob search
if not volume_vtu:
    if len(sys.argv) > 1:
        volume_vtu = sys.argv[1]
    else:
        matches = glob.glob("volume_flow_datablade*Blade_*.vtu")
        volume_vtu = matches[0] if matches else "volume_flow_databladeVALIDATION_Blade_0.vtu"

volume_vtu = volume_vtu.replace('\\', '/')
vtu_dir, vtu_file = volume_vtu.rsplit('/', 1) if '/' in volume_vtu else ('', volume_vtu)
stem = vtu_file.split('.', 1)[0]
blade_id = stem.rsplit('_', 1)[-1]
history_csv_path = f"{vtu_dir}/history_databladeVALIDATION_Blade_{blade_id}.csv" if vtu_dir else f"history_databladeVALIDATION_Blade_{blade_id}.csv"

# Extract blade pitch from blade.databladeVALIDATION
def find_file_upwards(start_dir, relative_path):
    path = Path(start_dir) if start_dir else Path('.')
    for parent in [path] + list(path.parents):
        candidate = parent / relative_path
        if candidate.is_file():
            return candidate
    return None

# try locating blade.databladeVALIDATION near the VTU file
blade_file = find_file_upwards(vtu_dir, 'blade.databladeVALIDATION')

# final fallback: search the current tree for the blade file
if blade_file is None:
    matches = list(Path('.').rglob(f'Blade_{blade_id}/blade.databladeVALIDATION'))
    blade_file = matches[0] if matches else None

if blade_file is not None:
    with open(blade_file, 'r') as f:
        next(f)
        line = f.readline()
        pitch = np.float64(line.split()[4])
else:
    # try to read pitch from rerun.py colocated with the VTU file
    rerun_file = Path(vtu_dir) / 'rerun.py' if vtu_dir else Path('rerun.py')
    if rerun_file.is_file():
        try:
            with open(rerun_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('pitch ='):
                        pitch = np.float64(line.split('=', 1)[1].strip())
                        break
                else:
                    raise ValueError('pitch line not found')
        except Exception as e:
            pitch = 0.03295
            print(f"rerun.py found but failed to extract pitch ({e}); using default pitch {pitch}")
    else:
        pitch = 0.03295
        print(f"blade.databladeVALIDATION and rerun.py not found; using default pitch {pitch}")

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# get layout
layout1 = GetLayout()

# split cell
layout1.SplitHorizontal(0, 0.5)

# set active view
SetActiveView(None)

# split cell
layout1.SplitVertical(2, 0.5)

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView2 = CreateView('RenderView')
renderView2.AxesGrid = 'Grid Axes 3D Actor'
renderView2.StereoType = 'Crystal Eyes'
renderView2.CameraFocalDisk = 1.0
renderView2.LegendGrid = 'Legend Grid Actor'
renderView2.BackEnd = 'OSPRay raycaster'
renderView2.OSPRayMaterialLibrary = materialLibrary1

# assign view to a particular cell in the layout
AssignViewToLayout(view=renderView2, layout=layout1, hint=5)

# set active view
SetActiveView(None)

# Create a new 'Render View'
renderView3 = CreateView('RenderView')
renderView3.AxesGrid = 'Grid Axes 3D Actor'
renderView3.StereoType = 'Crystal Eyes'
renderView3.CameraFocalDisk = 1.0
renderView3.LegendGrid = 'Legend Grid Actor'
renderView3.BackEnd = 'OSPRay raycaster'
renderView3.OSPRayMaterialLibrary = materialLibrary1

# assign view to a particular cell in the layout
AssignViewToLayout(view=renderView3, layout=layout1, hint=6)

# set active view
SetActiveView(renderView2)

# find source
volume_flow_databladeVALIDATION_Blade_0vtu = volume_source if volume_source is not None else FindSource('volume_flow_databladeVALIDATION_Blade_0.vtu')

# create a new 'Append Datasets'
appendDatasets1 = AppendDatasets(registrationName='AppendDatasets1', Input=volume_flow_databladeVALIDATION_Blade_0vtu)

# show data in view
appendDatasets1Display = Show(appendDatasets1, renderView2, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
appendDatasets1Display.Representation = 'Surface'

# reset view to fit data
renderView2.ResetCamera(False, 0.9)

#changing interaction mode based on data extents
renderView2.InteractionMode = '2D'
renderView2.CameraPosition = [0.047614000737667084, -0.10536289773881435, 0.9570414148271084]
renderView2.CameraFocalPoint = [0.047614000737667084, -0.10536289773881435, 0.0]

# update the view to ensure updated data information
renderView2.Update()

# set active source
SetActiveSource(volume_flow_databladeVALIDATION_Blade_0vtu)

# create a new 'Append Datasets'
appendDatasets2 = AppendDatasets(registrationName='AppendDatasets2', Input=volume_flow_databladeVALIDATION_Blade_0vtu)

# show data in view
appendDatasets2Display = Show(appendDatasets2, renderView2, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
appendDatasets2Display.Representation = 'Surface'

# update the view to ensure updated data information
renderView2.Update()

# set active source
SetActiveSource(volume_flow_databladeVALIDATION_Blade_0vtu)

# create a new 'Append Datasets'
appendDatasets3 = AppendDatasets(registrationName='AppendDatasets3', Input=volume_flow_databladeVALIDATION_Blade_0vtu)

# show data in view
appendDatasets3Display = Show(appendDatasets3, renderView2, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
appendDatasets3Display.Representation = 'Surface'

# update the view to ensure updated data information
renderView2.Update()

# set active source
SetActiveSource(volume_flow_databladeVALIDATION_Blade_0vtu)

# create a new 'Append Datasets'
appendDatasets4 = AppendDatasets(registrationName='AppendDatasets4', Input=volume_flow_databladeVALIDATION_Blade_0vtu)

# show data in view
appendDatasets4Display = Show(appendDatasets4, renderView2, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
appendDatasets4Display.Representation = 'Surface'

# update the view to ensure updated data information
renderView2.Update()

# set active source
SetActiveSource(appendDatasets1)

# Properties modified on appendDatasets1Display
appendDatasets1Display.Position = [0.0, pitch, 0.0]

# Properties modified on appendDatasets1Display.DataAxesGrid
appendDatasets1Display.DataAxesGrid.Position = [0.0, pitch, 0.0]

# Properties modified on appendDatasets1Display.PolarAxes
appendDatasets1Display.PolarAxes.Translation = [0.0, pitch, 0.0]

# set active source
SetActiveSource(appendDatasets2)

# Properties modified on appendDatasets2Display
appendDatasets2Display.Position = [0.0, -pitch, 0.0]

# Properties modified on appendDatasets2Display.DataAxesGrid
appendDatasets2Display.DataAxesGrid.Position = [0.0, -pitch, 0.0]

# Properties modified on appendDatasets2Display.PolarAxes
appendDatasets2Display.PolarAxes.Translation = [0.0, -pitch, 0.0]

# set active source
SetActiveSource(appendDatasets3)

# Properties modified on appendDatasets3Display
appendDatasets3Display.Position = [0.0, 2*pitch, 0.0]

# Properties modified on appendDatasets3Display.DataAxesGrid
appendDatasets3Display.DataAxesGrid.Position = [0.0, 2*pitch, 0.0]

# Properties modified on appendDatasets3Display.PolarAxes
appendDatasets3Display.PolarAxes.Translation = [0.0, 2*pitch, 0.0]

# set active source
SetActiveSource(appendDatasets4)

# Properties modified on appendDatasets4Display
appendDatasets4Display.Position = [0.0, -2*pitch, 0.0]

# Properties modified on appendDatasets4Display.DataAxesGrid
appendDatasets4Display.DataAxesGrid.Position = [0.0, -2*pitch, 0.0]

# Properties modified on appendDatasets4Display.PolarAxes
appendDatasets4Display.PolarAxes.Translation = [0.0, -2*pitch, 0.0]

# set active source
SetActiveSource(volume_flow_databladeVALIDATION_Blade_0vtu)

# show data in view
volume_flow_databladeVALIDATION_Blade_0vtuDisplay = Show(volume_flow_databladeVALIDATION_Blade_0vtu, renderView2, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
volume_flow_databladeVALIDATION_Blade_0vtuDisplay.Representation = 'Surface'

# update the view to ensure updated data information
renderView2.Update()

# set scalar coloring
ColorBy(volume_flow_databladeVALIDATION_Blade_0vtuDisplay, ('POINTS', 'Mach'))

# rescale color and/or opacity maps used to include current data range
volume_flow_databladeVALIDATION_Blade_0vtuDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
volume_flow_databladeVALIDATION_Blade_0vtuDisplay.SetScalarBarVisibility(renderView2, True)

# get color transfer function/color map for 'Mach'
machLUT = GetColorTransferFunction('Mach')

# get opacity transfer function/opacity map for 'Mach'
machPWF = GetOpacityTransferFunction('Mach')

# get 2D transfer function for 'Mach'
machTF2D = GetTransferFunction2D('Mach')

# set active source
SetActiveSource(appendDatasets1)

# set scalar coloring
ColorBy(appendDatasets1Display, ('POINTS', 'Mach'))

# rescale color and/or opacity maps used to include current data range
appendDatasets1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
appendDatasets1Display.SetScalarBarVisibility(renderView2, True)

# hide data in view
Hide(appendDatasets1, renderView2)

# set active source
SetActiveSource(appendDatasets1)

# show data in view
appendDatasets1Display = Show(appendDatasets1, renderView2, 'UnstructuredGridRepresentation')

# show color bar/color legend
appendDatasets1Display.SetScalarBarVisibility(renderView2, True)

# update the view to ensure updated data information
renderView2.Update()

# set active source
SetActiveSource(appendDatasets3)

# set active source
SetActiveSource(appendDatasets1)

# set active source
SetActiveSource(appendDatasets2)

# set scalar coloring
ColorBy(appendDatasets2Display, ('POINTS', 'Mach'))

# rescale color and/or opacity maps used to include current data range
appendDatasets2Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
appendDatasets2Display.SetScalarBarVisibility(renderView2, True)

# set active source
SetActiveSource(appendDatasets3)

# set scalar coloring
ColorBy(appendDatasets3Display, ('POINTS', 'Mach'))

# rescale color and/or opacity maps used to include current data range
appendDatasets3Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
appendDatasets3Display.SetScalarBarVisibility(renderView2, True)

# set active source
SetActiveSource(appendDatasets4)

# set scalar coloring
ColorBy(appendDatasets4Display, ('POINTS', 'Mach'))

# rescale color and/or opacity maps used to include current data range
appendDatasets4Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
appendDatasets4Display.SetScalarBarVisibility(renderView2, True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
machLUT.ApplyPreset('Cool to Warm (Extended)', True)

# get color legend/bar for machLUT in view renderView2
machLUTColorBar = GetScalarBar(machLUT, renderView2)

# change scalar bar placement
machLUTColorBar.WindowLocation = 'Any Location'
machLUTColorBar.Position = [0.07035175879396982, 0.3224043715846995]
machLUTColorBar.ScalarBarLength = 0.32999999999999985

# set active view
SetActiveView(renderView3)

# set active source
SetActiveSource(volume_flow_databladeVALIDATION_Blade_0vtu)

# show data in view
volume_flow_databladeVALIDATION_Blade_0vtuDisplay_1 = Show(volume_flow_databladeVALIDATION_Blade_0vtu, renderView3, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
volume_flow_databladeVALIDATION_Blade_0vtuDisplay_1.Representation = 'Surface'

#changing interaction mode based on data extents
renderView3.InteractionMode = '2D'
renderView3.CameraPosition = [0.047614000737667084, -0.10536289773881435, 0.9570414148271084]
renderView3.CameraFocalPoint = [0.047614000737667084, -0.10536289773881435, 0.0]

# reset view to fit data
renderView3.ResetCamera(False, 0.9)

# update the view to ensure updated data information
renderView3.Update()

# set active source
SetActiveSource(appendDatasets1)

# show data in view
appendDatasets1Display_1 = Show(appendDatasets1, renderView3, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
appendDatasets1Display_1.Representation = 'Surface'

# update the view to ensure updated data information
renderView3.Update()

# set active source
SetActiveSource(appendDatasets2)

# show data in view
appendDatasets2Display_1 = Show(appendDatasets2, renderView3, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
appendDatasets2Display_1.Representation = 'Surface'

# update the view to ensure updated data information
renderView3.Update()

# set active source
SetActiveSource(appendDatasets3)

# show data in view
appendDatasets3Display_1 = Show(appendDatasets3, renderView3, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
appendDatasets3Display_1.Representation = 'Surface'

# update the view to ensure updated data information
renderView3.Update()

# set active source
SetActiveSource(appendDatasets4)

# show data in view
appendDatasets4Display_1 = Show(appendDatasets4, renderView3, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
appendDatasets4Display_1.Representation = 'Surface'

# update the view to ensure updated data information
renderView3.Update()

# set active source
SetActiveSource(appendDatasets1)

# Properties modified on appendDatasets1Display_1
appendDatasets1Display_1.Position = [0.0, pitch, 0.0]

# Properties modified on appendDatasets1Display_1.DataAxesGrid
appendDatasets1Display_1.DataAxesGrid.Position = [0.0, pitch, 0.0]

# Properties modified on appendDatasets1Display_1.PolarAxes
appendDatasets1Display_1.PolarAxes.Translation = [0.0, pitch, 0.0]

# set active source
SetActiveSource(appendDatasets2)

# Properties modified on appendDatasets2Display_1
appendDatasets2Display_1.Position = [0.0, -pitch, 0.0]

# Properties modified on appendDatasets2Display_1.DataAxesGrid
appendDatasets2Display_1.DataAxesGrid.Position = [0.0, -pitch, 0.0]

# Properties modified on appendDatasets2Display_1.PolarAxes
appendDatasets2Display_1.PolarAxes.Translation = [0.0, -pitch, 0.0]

# set active source
SetActiveSource(appendDatasets3)

# Properties modified on appendDatasets3Display_1
appendDatasets3Display_1.Position = [0.0, 2*pitch, 0.0]

# Properties modified on appendDatasets3Display_1.DataAxesGrid
appendDatasets3Display_1.DataAxesGrid.Position = [0.0, 2*pitch, 0.0]

# Properties modified on appendDatasets3Display_1.PolarAxes
appendDatasets3Display_1.PolarAxes.Translation = [0.0, 2*pitch, 0.0]

# set active source
SetActiveSource(appendDatasets4)

# Properties modified on appendDatasets4Display_1
appendDatasets4Display_1.Position = [0.0, -2*pitch, 0.0]

# Properties modified on appendDatasets4Display_1.DataAxesGrid
appendDatasets4Display_1.DataAxesGrid.Position = [0.0, -2*pitch, 0.0]

# Properties modified on appendDatasets4Display_1.PolarAxes
appendDatasets4Display_1.PolarAxes.Translation = [0.0, -2*pitch, 0.0]

# set active source
SetActiveSource(volume_flow_databladeVALIDATION_Blade_0vtu)

# set scalar coloring
ColorBy(volume_flow_databladeVALIDATION_Blade_0vtuDisplay_1, ('POINTS', 'Eddy_Viscosity'))

# rescale color and/or opacity maps used to include current data range
volume_flow_databladeVALIDATION_Blade_0vtuDisplay_1.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
volume_flow_databladeVALIDATION_Blade_0vtuDisplay_1.SetScalarBarVisibility(renderView3, True)

# get color transfer function/color map for 'Eddy_Viscosity'
eddy_ViscosityLUT = GetColorTransferFunction('Eddy_Viscosity')

# get opacity transfer function/opacity map for 'Eddy_Viscosity'
eddy_ViscosityPWF = GetOpacityTransferFunction('Eddy_Viscosity')

# get 2D transfer function for 'Eddy_Viscosity'
eddy_ViscosityTF2D = GetTransferFunction2D('Eddy_Viscosity')

# set active source
SetActiveSource(appendDatasets1)

# set scalar coloring
ColorBy(appendDatasets1Display_1, ('POINTS', 'Eddy_Viscosity'))

# rescale color and/or opacity maps used to include current data range
appendDatasets1Display_1.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
appendDatasets1Display_1.SetScalarBarVisibility(renderView3, True)

# set active source
SetActiveSource(appendDatasets2)

# set scalar coloring
ColorBy(appendDatasets2Display_1, ('POINTS', 'Eddy_Viscosity'))

# rescale color and/or opacity maps used to include current data range
appendDatasets2Display_1.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
appendDatasets2Display_1.SetScalarBarVisibility(renderView3, True)

# set active source
SetActiveSource(appendDatasets3)

# set scalar coloring
ColorBy(appendDatasets3Display_1, ('POINTS', 'Eddy_Viscosity'))

# rescale color and/or opacity maps used to include current data range
appendDatasets3Display_1.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
appendDatasets3Display_1.SetScalarBarVisibility(renderView3, True)

# set active source
SetActiveSource(appendDatasets4)

# set scalar coloring
ColorBy(appendDatasets4Display_1, ('POINTS', 'Eddy_Viscosity'))

# rescale color and/or opacity maps used to include current data range
appendDatasets4Display_1.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
appendDatasets4Display_1.SetScalarBarVisibility(renderView3, True)

# get color legend/bar for eddy_ViscosityLUT in view renderView3
eddy_ViscosityLUTColorBar = GetScalarBar(eddy_ViscosityLUT, renderView3)

# change scalar bar placement
eddy_ViscosityLUTColorBar.WindowLocation = 'Any Location'
eddy_ViscosityLUTColorBar.Position = [0.041876046901172526, 0.3032786885245901]
eddy_ViscosityLUTColorBar.ScalarBarLength = 0.3300000000000001

# set active view
SetActiveView(renderView2)

# change scalar bar placement
machLUTColorBar.Position = [0.03350083752093798, 0.3224043715846995]

# set active view
SetActiveView(renderView3)

# Properties modified on eddy_ViscosityLUTColorBar
eddy_ViscosityLUTColorBar.Title = ''

# set active view
SetActiveView(renderView2)

# Properties modified on machLUTColorBar
machLUTColorBar.Title = ''

# set active view
SetActiveView(renderView3)

# Properties modified on eddy_ViscosityLUTColorBar
eddy_ViscosityLUTColorBar.AutomaticLabelFormat = 0
eddy_ViscosityLUTColorBar.LabelFormat = '%-#6.3e'

# Properties modified on eddy_ViscosityLUTColorBar
eddy_ViscosityLUTColorBar.LabelFormat = '%-#6.1e'

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1196, 768)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.047614000737667084, -0.10536289773881435, 0.9570414148271084]
renderView1.CameraFocalPoint = [0.047614000737667084, -0.10536289773881435, 0.0]
renderView1.CameraParallelScale = 0.19187208995950336

# current camera placement for renderView2
renderView2.InteractionMode = '2D'
renderView2.CameraPosition = [0.013820974682304532, -0.013084401298166069, 0.9570414148271084]
renderView2.CameraFocalPoint = [0.013820974682304532, -0.013084401298166069, 0.0]
renderView2.CameraParallelScale = 0.03450988191746806

# current camera placement for renderView3
renderView3.InteractionMode = '2D'
renderView3.CameraPosition = [0.013824403255312596, -0.0069232336573539065, 0.741336828143775]
renderView3.CameraFocalPoint = [0.013824403255312596, -0.0069232336573539065, 0.0]
renderView3.CameraParallelScale = 0.03450988191746806


#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1796, 1079)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView2
renderView2.InteractionMode = '2D'
renderView2.CameraPosition = [0.565080912105909, 1.5407264776490739, 19.351978909705245]
renderView2.CameraFocalPoint = [0.565080912105909, 1.5407264776490739, 0.0]
renderView2.CameraParallelScale = 0.9008516529732421

# current camera placement for renderView3
renderView3.InteractionMode = '2D'
renderView3.CameraPosition = [0.5493781400960493, 1.5300506334542094, 19.351978909705245]
renderView3.CameraFocalPoint = [0.5493781400960493, 1.5300506334542094, 0.0]
renderView3.CameraParallelScale = 0.9008516529732421

##--------------------------------------------
## You may need to add some code at the end of this python script depending on your usage, eg:
#
## Render all views to see them appears
# RenderAllViews()
#
## Interact with the view, usefull when running from pvpython
# Interact()
#
## Save a screenshot of the active view
# SaveScreenshot("path/to/screenshot.png")
#
## Save a screenshot of a layout (multiple splitted view)
# SaveScreenshot("path/to/screenshot.png", GetLayout())
#
## Save all "Extractors" from the pipeline browser
# SaveExtracts()
#
## Save a animation of the current active view
# SaveAnimation()
#
## Please refer to the documentation of paraview.simple
## https://kitware.github.io/paraview-docs/latest/python/paraview.simple.html
##--------------------------------------------