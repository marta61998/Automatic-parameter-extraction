
"""

##### FUNCTIONS USED IN SCRIPT_2

@Alvaro's & @Marta's functions 

"""

import math as mt
from tkinter import *
from basefunctions import *
import numpy as np
import vtk
from vtk.util import numpy_support
from pymeshfix import _meshfix
from sklearn.neighbors import NearestNeighbors
import os


import sfepy, vtk,  vtk.util.numpy_support as numpy_support
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.discrete import (FieldVariable, Material, Integral, Function, Equation, Equations, Problem)
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.discrete.conditions import Conditions, EssentialBC

from sfepy.terms import Term

from vmtkcenterlines_2 import *


def readstl(filename):
    """Read VTK file"""
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def readvtk(filename, dataarrays=True):
    """Read image in vtk format."""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    if dataarrays == False:
        for i in range(reader.GetNumberOfPointArrays()):
            arrayname = reader.GetPointArrayName(i)
            reader.SetPointArrayStatus(arrayname, 0)
        for i in range(reader.GetNumberOfCellArrays()):
            arrayname = reader.GetCellArrayName(i)
            reader.SetPointArrayStatus(arrayname, 0)
        reader.Update()
    return reader.GetOutput()

def writevtk(mesh, filename):
    """
    Code to export a mesh
    :param mesh: Mesh file to export
    :param filename: Name of the file that will contain the mesh
    :return: -
    """
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(mesh)
    # writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()


def estimate_center(LAA_edge):
    """
    Estimate the center of the base
    :param LAA_edge: Edges of the LAA base
    :return: center
    """
    npoints = LAA_edge.GetNumberOfPoints()
    cent = []

    for i in range(npoints):
        points = LAA_edge.GetPoint(i)
        if i == 0:
            cent.append(points)
        else:
            cent.append(np.add(cent[-1], points))

    center = np.array([cent[-1][0]/npoints, cent[-1][1]/npoints, cent[-1][2]/npoints])

    return center

def estimate_center_2(LAA_edge_points):
    """
    Estimate the center of the base
    :param LAA_edge: Edges of the LAA base
    :return: center
    """
    npoints = np.shape(LAA_edge_points)
    npoints = npoints[0]
    cent = []

    for i in range(npoints):
        points = LAA_edge_points[i,:]
        if i == 0:
            cent.append(points)
        else:
            cent.append(np.add(cent[-1], points))

    center = np.array([cent[-1][0]/npoints, cent[-1][1]/npoints, cent[-1][2]/npoints])

    return center


def vtk_show3(renderer, mesh, centreline1,centreline2,heigth, width,filename, actor1=None, actor2=None, actor3=None, actor4=None, actor5=None, actor6=None):
    """
    Show an interactive window with the mesh and the actors
    :param renderer: Renderer
    :param mesh: Input mesh
    :param heigth: Heigth of the visualizer
    :param width: Width of the visualizer
    :param actor1: Add actors to the window
    :param actor2: Add actors to the window
    :param actor3: Add actors to the window
    :return: Visualizer with mesh and actors
    """
    # Create renderer and window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetWindowName("Measures")
    renderWindow.AddRenderer(renderer)

    renderWindow.SetSize(heigth, width)

    # Make it interactive
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renderWindow)

    # Mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)

    mapper2 = vtk.vtkPolyDataMapper()
    mapper2.SetInputData(centreline1)

    mapper3 = vtk.vtkPolyDataMapper()
    mapper3.SetInputData(centreline2)


    # Define actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.35)
    #actor.GetProperty().SetColor(1.0, 0.87, 0.725)

    actor2 = vtk.vtkActor()
    actor2.SetMapper(mapper2)
    actor2.GetProperty().SetOpacity(0.35)
    color2 = [1.0, 0.0, 0.0]
    actor2.GetProperty().SetColor(color2)

    actor3 = vtk.vtkActor()
    actor3.SetMapper(mapper3)
    actor3.GetProperty().SetOpacity(1)
    actor3.GetProperty().SetLineWidth(2)
    color3 = [0.0, 0.0, 0.0]
    actor3.GetProperty().SetColor(color3)

    # Assign actor
    renderer.AddActor(actor)
    renderer.SetBackground(1.0, 1.0, 1.0)

    if actor1 is not None:
        renderer.AddActor(actor1)
    if actor2 is not None:
        renderer.AddActor(actor2)
    if actor3 is not None:
        renderer.AddActor(actor3)
    if actor4 is not None:
        renderer.AddActor(actor4)
    if actor5 is not None:
        renderer.AddActor(actor5)
    if actor6 is not None:
        renderer.AddActor(actor6)


    renderWindow.Render()
    if filename == False:
        # Initialize
        iren.Initialize()
        
        iren.Start()
    else:
    
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(renderWindow)
        w2if.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputData(w2if.GetOutput())
        writer.Write() 


    return renderer

def stl2vtk(filename1, filename2):
    """
    Code to convert between .stl and .vtk
    :param filename1: Name of the .stl file
    :param filename2: Name of the .vtk file
    :return: -
    """
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename1)
    reader.Update()

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputConnection(reader.GetOutputPort())
    writer.SetFileTypeToASCII()
    writer.SetFileName(filename2)
    writer.Write()


def rotate_angle(clip, normal1, normal2, mean_p, filename2 = None):
    """
    Given the normal vectors of two planes find the rotation angle and rotate mesh
    :param clip: Mesh 
    :param normal1: Normal from the plane to rotate
    :param normal2: "Reference" normal
    :param mean_p: Center of the LAA
    :param filename2: Rotated mesh path to save (optional)
    :return: Rotated mesh
    """

    mesh=clip

    dotprd = np.dot(normal1, normal2)
    normA = np.linalg.norm(normal1)
    normB = np.linalg.norm(normal2)

    angl = mt.acos(dotprd/(normA*normB))
    angle_degr = mt.degrees(angl)
    angl_dgr = (180 / mt.pi)*angl
 
    direction = np.cross(normal1, normal2)

    toorigin = [0, 0, 0]
    toorigin[0] = -1 * mean_p[0]
    toorigin[1] = -1 * mean_p[1]
    toorigin[2] = -1 * mean_p[2]

    if normal1[2] > 0 and (direction[0] < 0 or direction[1] < 0 or direction[2] <= 0):

        t = vtk.vtkTransform()
        t.PostMultiply()
        t.Translate(toorigin)
        t.RotateWXYZ(angl_dgr, -direction[0], -direction[1], -direction[2])
        t.Translate(mean_p)
        t.Inverse()
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetTransform(t)
        tf.SetInputData(mesh)
        tf.Update()

    # Handle cases that are not orientated coherently
    else:
        t = vtk.vtkTransform()
        t.PostMultiply()
        t.Translate(toorigin)
        #t.RotateWXYZ(angl_dgr, -direction[0], -direction[1], -direction[2])
        t.RotateWXYZ(angl_dgr + 180, -direction[0], -direction[1], -direction[2])
        t.Translate(mean_p)
        t.Inverse()
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetTransform(t)
        tf.SetInputData(mesh)
        tf.Update()

    t.Update()
    matrix = t.GetMatrix()
    #print(matrix)

    mp= np.zeros(shape=(4, 4))
    for i in range(0, 4):
        for j in range(0, 4):
            mp[i,j] = matrix.GetElement(i, j)

    #print(mp)
    file_rotated = tf.GetOutput()
    if filename2:
        writevtk(file_rotated, filename2)

    return file_rotated, mp

def normvector(v1):
    return mt.sqrt(np.dot(v1, v1))

def createPolyData(points1):
     
    poly = vtk.vtkPolyData()   
    points = vtk.vtkPoints()
    verts = vtk.vtkPoints()
    verts.SetData(numpy_support.numpy_to_vtk(points1))
    poly.SetPoints(verts)

    nCoords = points1.shape[0]
    nElem = points1.shape[1]

    cells = vtk.vtkCellArray()
    scalars = None

    cells_npy = np.vstack([np.ones(nCoords,dtype=np.int64),
            np.arange(nCoords,dtype=np.int64)]).T.flatten()
    cells.SetCells(nCoords,numpy_support.numpy_to_vtkIdTypeArray(cells_npy))
   
    poly.SetVerts(cells)

 
    return poly

def vtk_show(renderer, mesh, heigth, width, filename = None, actor1=None, actor2=None, actor3=None, actor4=None, actor5=None, actor6=None, actor7=None):
    """
    Show an interactive window with the mesh and the actors
    :param renderer: Renderer
    :param mesh: Input mesh
    :param heigth: Heigth of the visualizer
    :param width: Width of the visualizer
    :param actor1: Add actors to the window
    :param actor2: Add actors to the window
    :param actor3: Add actors to the window
    :return: Visualizer with mesh and actors
    """
    # Create renderer and window
    print('debug1')
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetWindowName("Measures")
    renderWindow.AddRenderer(renderer)

    renderWindow.SetSize(heigth, width)

    # Make it interactive
    print('debug2')
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renderWindow)

    # Mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)

    # Define actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.35)
    #actor.GetProperty().SetColor(1.0, 0.87, 0.725)

    # Assign actor
    renderer.AddActor(actor)
    renderer.SetBackground(1.0, 1.0, 1.0)
    print('debug3')

    if actor1 is not None:
        renderer.AddActor(actor1)
    if actor2 is not None:
        renderer.AddActor(actor2)
    if actor3 is not None:
        renderer.AddActor(actor3)
    if actor4 is not None:
        renderer.AddActor(actor4)
    if actor5 is not None:
        renderer.AddActor(actor5)
    if actor6 is not None:
        renderer.AddActor(actor6)
    if actor7 is not None:
        renderer.AddActor(actor7)
    
    renderWindow.Render()
    if filename == False:
        # Initialize
        iren.Initialize()
        
        iren.Start()
    else:
        print('debug4')
    
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(renderWindow)
        w2if.Update()
        print('debug5')
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputData(w2if.GetOutput())
        writer.Write() 


    return renderer

def vtk_show4(renderer, mesh, centreline1,centreline2,centreline3,heigth, width, actor1=None, actor5=None, actor6=None):
    """
    Show an interactive window with the mesh and the actors
    :param renderer: Renderer
    :param mesh: Input mesh
    :param heigth: Heigth of the visualizer
    :param width: Width of the visualizer
    :param actor1: Add actors to the window
    :param actor2: Add actors to the window
    :param actor3: Add actors to the window
    :return: Visualizer with mesh and actors
    """
    # Create renderer and window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetWindowName("Measures")
    renderWindow.AddRenderer(renderer)

    renderWindow.SetSize(heigth, width)

    # Make it interactive
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renderWindow)

    # Mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)

    mapper2 = vtk.vtkPolyDataMapper()
    mapper2.SetInputData(centreline1)

    mapper3 = vtk.vtkPolyDataMapper()
    mapper3.SetInputData(centreline2)

    mapper4 = vtk.vtkPolyDataMapper()
    mapper4.SetInputData(centreline3)


    # Define actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.35)
    actor.GetProperty().SetColor(0.0, 0.0, 1.0)

    actor2 = vtk.vtkActor()
    actor2.SetMapper(mapper2)
    actor2.GetProperty().SetOpacity(0.35)
    color2 = [1.0, 0.0, 0.0]
    actor2.GetProperty().SetColor(color2)

    actor3 = vtk.vtkActor()
    actor3.SetMapper(mapper3)
    actor3.GetProperty().SetOpacity(0.35)
    color3 = [1.0, 1.0, 0.0]
    actor3.GetProperty().SetColor(color3)

    actor4 = vtk.vtkActor()
    actor4.SetMapper(mapper4)
    actor4.GetProperty().SetOpacity(0.35)
    color4 = [1.0, 0.0, 1.0]
    actor4.GetProperty().SetColor(color4)

    # Assign actor
    renderer.AddActor(actor)
    renderer.SetBackground(1.0, 1.0, 1.0)

    if actor1 is not None:
        renderer.AddActor(actor1)
    if actor2 is not None:
        renderer.AddActor(actor2)
    if actor3 is not None:
        renderer.AddActor(actor3)
    if actor4 is not None:
        renderer.AddActor(actor4)
    if actor5 is not None:
        renderer.AddActor(actor5)
    if actor6 is not None:
        renderer.AddActor(actor6)

    # Initialize
    iren.Initialize()
    renderWindow.Render()
    iren.Start()

    return renderer

def vtk_show_centreline(renderer, mesh, centreline, heigth, width, actor1=None, actor2=None, actor3=None, actor4=None, actor5=None, actor6=None):
    """
    Show an interactive window with the mesh and the actors
    :param renderer: Renderer
    :param mesh: Input mesh
    :param heigth: Heigth of the visualizer
    :param width: Width of the visualizer
    :param actor1: Add actors to the window
    :param actor2: Add actors to the window
    :param actor3: Add actors to the window
    :return: Visualizer with mesh and actors
    """
    # Create renderer and window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetWindowName("Measures")
    renderWindow.AddRenderer(renderer)

    renderWindow.SetSize(heigth, width)

    # Make it interactive
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renderWindow)

    # Mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)

    mapper2 = vtk.vtkPolyDataMapper()
    mapper2.SetInputData(centreline)

    # Define actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.35)
    #actor.GetProperty().SetColor(1.0, 0.87, 0.725)

    actor_c = vtk.vtkActor()
    actor_c.SetMapper(mapper2)
    actor_c.GetProperty().SetOpacity(0.35)
    color_c = [0.0, 0.0, 1.0]
    #color2 = [1.0, 0.0, 0.0]
    actor_c.GetProperty().SetColor(color_c)

    # Assign actor
    renderer.AddActor(actor)
    renderer.SetBackground(1.0, 1.0, 1.0)

     # Assign actor
    renderer.AddActor(actor_c)
   

    if actor1 is not None:
        renderer.AddActor(actor1)
    if actor2 is not None:
        renderer.AddActor(actor2)
    if actor3 is not None:
        renderer.AddActor(actor3)
    if actor4 is not None:
        renderer.AddActor(actor4)
    if actor5 is not None:
        renderer.AddActor(actor5)
    if actor6 is not None:
        renderer.AddActor(actor6)

    # Initialize
    iren.Initialize()
    renderWindow.Render()
    iren.Start()

    return renderer

def vtk_show_centreline_multi(renderer, mesh, centreline, heigth, width, actor_list=None):
    """
    Show an interactive window with the mesh and the actors
    :param renderer: Renderer
    :param mesh: Input mesh
    :param heigth: Heigth of the visualizer
    :param width: Width of the visualizer
    :param actor1: Add actors to the window
    :param actor2: Add actors to the window
    :param actor3: Add actors to the window
    :return: Visualizer with mesh and actors
    """
    # Create renderer and window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetWindowName("Measures")
    renderWindow.AddRenderer(renderer)

    renderWindow.SetSize(heigth, width)

    # Make it interactive
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renderWindow)

    # Mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)

    mapper2 = vtk.vtkPolyDataMapper()
    mapper2.SetInputData(centreline)

    # Define actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.35)
    #actor.GetProperty().SetColor(1.0, 0.87, 0.725)

    actor_c = vtk.vtkActor()
    actor_c.SetMapper(mapper2)
    actor_c.GetProperty().SetOpacity(1)
    color_c = [0.0, 0.0, 1.0]
    #color2 = [1.0, 0.0, 0.0]
    actor_c.GetProperty().SetColor(color_c)

    # Assign actor
    renderer.AddActor(actor)
    renderer.SetBackground(1.0, 1.0, 1.0)

     # Assign actor
    renderer.AddActor(actor_c)
   

    # Define actor
    for i in range(len(actor_list)):
        renderer.AddActor(actor_list[i])

    # Initialize
    iren.Initialize()
    renderWindow.Render()
    iren.Start()

    return renderer

def vtk_show_multi(renderer, mesh, heigth, width, actor_list=None):
    """
    Show an interactive window with the mesh and the actors
    :param renderer: Renderer
    :param mesh: Input mesh
    :param heigth: Heigth of the visualizer
    :param width: Width of the visualizer
    :param actor1: Add actors to the window
    :param actor2: Add actors to the window
    :param actor3: Add actors to the window
    :return: Visualizer with mesh and actors
    """
    # Create renderer and window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetWindowName("Measures")
    renderWindow.AddRenderer(renderer)

    renderWindow.SetSize(heigth, width)

    # Make it interactive
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renderWindow)

    # Mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)

    # Define actor
    for i in range(len(actor_list)):
        renderer.AddActor(actor_list[i])
    #actor.GetProperty().SetColor(1.0, 0.87, 0.725)

    # Define actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.35)
    #actor.GetProperty().SetColor(1.0, 0.87, 0.725)

    # Assign actor
    renderer.AddActor(actor)
    renderer.SetBackground(1.0, 1.0, 1.0)


    # Initialize
    iren.Initialize()
    renderWindow.Render()
    iren.Start()

    return renderer

def vtk_show_u(renderer, mesh, heigth, width, actor1=None, actor2=None, actor3=None, actor4=None, actor5=None, actor6=None):
    """
    Show an interactive window with the mesh and the actors
    :param renderer: Renderer
    :param mesh: Input mesh
    :param heigth: Heigth of the visualizer
    :param width: Width of the visualizer
    :param actor1: Add actors to the window
    :param actor2: Add actors to the window
    :param actor3: Add actors to the window
    :return: Visualizer with mesh and actors
    """
    # Create renderer and window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetWindowName("Measures")
    renderWindow.AddRenderer(renderer)

    renderWindow.SetSize(heigth, width)

    # Make it interactive
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renderWindow)

    # Mapper
    #fieldData = mesh.GetPointData()
    #print(fieldData)
    #drange = fieldData.GetScalars().GetRange()


    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(mesh)
    mapper.SetScalarVisibility(1)
    #mapper.SetScalarRange(drange)

    # Define actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.35)
    #actor.GetProperty().SetColor(1.0, 0.87, 0.725)

    # Assign actor
    renderer.AddActor(actor)
    renderer.SetBackground(1.0, 1.0, 1.0)

    if actor1 is not None:
        renderer.AddActor(actor1)
    if actor2 is not None:
        renderer.AddActor(actor2)
    if actor3 is not None:
        renderer.AddActor(actor3)
    if actor4 is not None:
        renderer.AddActor(actor4)
    if actor5 is not None:
        renderer.AddActor(actor5)
    if actor6 is not None:
        renderer.AddActor(actor6)

    # Initialize
    iren.Initialize()
    renderWindow.Render()
    iren.Start()

    return renderer

def addline_show(p1, p2, color=[0.0, 0.0, 1.0]):
    """
    Add a line to the interactive visualizer
    :param p1: Start of the line
    :param p2: End of the line
    :param color: color
    :return: Actor
    """
    line = vtk.vtkLineSource()
    line.SetPoint1(p1)
    line.SetPoint2(p2)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(line.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetLineWidth(1.5)

    return actor

def addpoint_show(p, radius=0.6, color=[0.0, 0.0, 1.0]):
    """
    Add a point to the visualizer
    :param p: Coordinates of the point
    :param radius: Radius of the point (considered as a sphere)
    :param color: color of the point
    :return: Actor
    """
    point = vtk.vtkSphereSource()
    point.SetCenter(p)
    point.SetRadius(radius)
    point.SetPhiResolution(100)
    point.SetThetaResolution(100)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(point.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)

    return actor

def extractlargestregion_edges_m2(polydata):
    # NOTE: extract LAA cut region with highest area as ostium 
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputData(polydata)
    surfer.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(surfer.GetOutputPort())
    cleaner.Update()

    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInputConnection(cleaner.GetOutputPort())
    connect.SetExtractionModeToAllRegions()
    connect.Update()
    n = connect.GetNumberOfExtractedRegions()

    LAA_error = False
    if n > 1:
        #print('WARNING, plane cuts the LAA twice!')	
        #print('Number of cut contours: ',n)
        #print('Extracting closest contour...')
        LAA_error = True
    sizes = connect.GetRegionSizes()

    connect.ColorRegionsOn()
    connect.Update()

    regions = []
    regions_center = []
    
    for i in range(n):
        connect_loop = vtk.vtkPolyDataConnectivityFilter()
        connect_loop.SetInputConnection(surfer.GetOutputPort())
        connect_loop.SetExtractionModeToSpecifiedRegions()
        connect_loop.AddSpecifiedRegion(i)
        connect_loop.Update()

        region_polydata = vtk.vtkPolyData()
        region_polydata.DeepCopy(connect_loop.GetOutput())
        
        region_array = []
        idList = vtk.vtkIdList()
        
        while(region_polydata.GetLines().GetNextCell(idList)):
            for pointId in range(idList.GetNumberOfIds()):
                point = region_polydata.GetPoint(idList.GetId(pointId))
                region_array.append(point)
        
        regions.append(region_array)
 
 
    list_ostium_numpy=[]
    if(n>1):
        for i in range(n):
            ostium_array = []
            #if(i == closest_index ):
            region = regions[i]
            for j in range(len(region)):
                    ostium_array.append(region[j])
        
            ostium_numpy = np.zeros((len(ostium_array),3))
            for i in range(len(ostium_array)):
                ostium_numpy[i] = ostium_array[i]
            list_ostium_numpy.append(ostium_numpy)
            center_region = estimate_center_2(ostium_numpy)
            regions_center.append(center_region)


    list_area = []
    list_delaunay = []
    list_poly = []
    #print('number of regions',n)
    if(n>1):
        for i in range(n):
            #print(len(list_ostium_numpy[i]))
            poly = createPolyData(list_ostium_numpy[i])
            list_poly.append(poly)

            surfer = vtk.vtkDataSetSurfaceFilter()
            surfer.SetInputData(poly)
            surfer.Update()

            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputConnection(surfer.GetOutputPort())
            cleaner.Update()

            connect = vtk.vtkPolyDataConnectivityFilter()
            connect.SetInputConnection(cleaner.GetOutputPort())
            connect.SetExtractionModeToAllRegions()
            connect.Update()

            delaunay = vtk.vtkDelaunay2D()
            delaunay.SetInputData(connect.GetOutput())
            delaunay.Update()  #quitar
                            
            polygonProperties = vtk.vtkMassProperties()
            polygonProperties.SetInputConnection(delaunay.GetOutputPort())
            polygonProperties.Update()
            list_area.append(polygonProperties.GetSurfaceArea())
            list_delaunay.append(delaunay.GetOutput())
        
        id_max = np.argmax(list_area)
        laa_edge = list_delaunay[id_max]
        area = list_area[id_max]
        poly = list_poly[id_max]
        
        return poly, area, laa_edge, LAA_error, regions_center
    else:
        
            delaunay = vtk.vtkDelaunay2D()
            delaunay.SetInputData(connect.GetOutput())
            delaunay.Update()  #quitar
                            
            polygonProperties = vtk.vtkMassProperties()
            polygonProperties.SetInputConnection(delaunay.GetOutputPort())
            polygonProperties.Update()

    return connect.GetOutput(), polygonProperties.GetSurfaceArea(), delaunay.GetOutput(), LAA_error, regions_center


def extractclosest_indexedregion_marta(polydata,ostium_centre,flag=False):
    # NOTE: preventive measures: clean before connectivity filter
    # to avoid artificial regionIds
    # It slices the surface down the middle
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputData(polydata)
    surfer.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(surfer.GetOutputPort())
    cleaner.Update()

    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInputConnection(cleaner.GetOutputPort())
    connect.SetExtractionModeToAllRegions()
    connect.Update()
    n = connect.GetNumberOfExtractedRegions()

    if n>1 :
        #print('WARNING.',n, 'connected regions')
        #print('Extracting connected region closest to ostium...')
        h=0
    
    sizes = connect.GetRegionSizes()
    
    #print('sizes',sizes)

    connect.ColorRegionsOn()
    connect.Update()

    regions = []
    regions_center = []
    
    list_centros = []
    list_edges_d = []
    list_areas = []
    for i in range(n):
        region = extractindexedregion(polydata,i)
        edges = extractboundaryedge(region)

        surface_area = vtk.vtkMassProperties()
        surface_area.SetInputData(region)
        area = surface_area.GetSurfaceArea()
            #print('region',i,'area',area)
        list_areas.append(area)
    
        try:
            center_region = estimate_center(edges)
            regions_center.append(center_region)
            actor_centro2  = addpoint_show(center_region, color=[0, 0, 1])
            list_centros.append(actor_centro2)
        except:
            print('ERROR in estimate_center, ignoring a region')
        #center_region = estimate_center(region)
        ##regions.append(region_array)
    
   
  
            print('ERROR in estimate_center, ignoring a region')
        #center_region = estimate_center(region)
        ##regions.append(region_array) 
        
    largest_index = list_areas.index(max(list_areas))
    clip = extractindexedregion(polydata,largest_index)
    #print('marta la mejor')
    #rend2 = vtk.vtkRenderer()
    #vtk_show(rend2, clip, 1000, 1000 )    

    return clip

def get_d1d2alt(LAA_edge, show_points=False):
    """
    Get d1, d2_min and d2_complete given a clipped file.
    D1 is defined as the ABSOLUTE MAXIMUM distance between two points of the orifice contour.
    D2 is defined as the sum of the minimum radius complementary and the ABSOLUTE minimum radius.
    The absolute minimum radius is the minimum distance between the centre of the LAA orifice and the contour.
    The complementary of the minimum radius is the line that has a bending of 180 degrees (approximately) with respect
    to the minimum absolute radius.
    The centre of the LAA orifice is estimated as the sum of all the points divided by the number of points of the
    orifice -> A better estimation can be obtained with the centreline script, but it is not implemented here.

    :param LAA_clip: Path of the LAA file in .vtk format CLIPPED
    :param filename2: Route in which the extracted edges will be saved
    :param LAA_id: ID of the file being considered
    :return: d1 (mm), min d2(mm), d2 complete(mm) and points to visualize the lines
    """

    #edges = extractboundaryedge(LAA_clip)
    ##LAA_edged,surface_area,LAA_edge = extractlargestregion_edges(edges)  #mostrar que es LAA_edge
    #edge_noDelaunay,surface_area, LAA_edge,LAA_error = extractclosestregion_edges(edges,origin)
    ##print('edge surface area', surface_area)
    
    # edge_f = r"C:\Users\U204585\Downloads\alvaro\VTK\files\LAA_edge.vtk"
    #writevtk(LAA_edge, edge_f)
    #writevtk(LAA_edge, filename2)
    bounds = LAA_edge.GetBounds()

    x_min = bounds[0]
    x_max = bounds[1]
    y_min = bounds[2]
    y_max = bounds[3]
    z_min = bounds[4]
    z_max = bounds[5]

    npoints = LAA_edge.GetNumberOfPoints()
    
    p0 = []
    p0_ids = []

    for i in range(npoints):
        #p0.append(edges.GetPoint(i))
        p0.append(LAA_edge.GetPoint(i))
    

    #PRUEBA CLIP IDS 
    #--
    #pointid = vtk.vtkIdFilter()
    #pointid.SetInputData(LAA_edge) # A vtkPolyData() mesh with 20525 points and 10000 cells
    #pointid.CellIdsOff()
    #pointid.PointIdsOn()
    #pointid.SetIdsArrayName('Point_IDs')
    #pointid.Update()

    #pointids = pointid.GetOutput().GetPointData().GetArray('Point_IDs')
    #print(pointids.GetValue(50))

    boundaryIds = 0
    #--

        
    
    #idFilter = vtk.vtkIdFilter()
    #idFilter.SetInputConnection(LAA_edge)
    #idFilter.SetIdsArrayName("ids")
    #idFilter.SetPointIds(True)
    #idFilter.SetCellIds(False)
    #idFilter.Update()

    #array = l2.GetPointData().GetArray("ids")
    #boundaryIds = []
    #for i in range(npoints):
    #    boundaryIds.append(array.GetValue(i))
    
    # Check boundaries 
    #print('points', len(p0))
    #index_delete0 = check_boundaries(p0, npoints, x_min, x_max, y_min, y_max, z_min, z_max)
    # print index_delete0[0]

    # Delete values that are out of bounds
    
    #for i in sorted(index_delete0, reverse=True):
    #    del p0[i]

    npoints_del = len(p0)
    #print('points del', npoints_del)
    m_distances = np.empty([npoints_del, npoints_del])
    #calcula la distancia de cada punto del contorno con el resto, la primera distancia que calcula (entre punto[0] y punto[1]) es dist_init, si la distancia calculada es mayor que la distancia anterior guardala en dist_init
    #y haz esyo con todos los puntos y guardalo en una matriz de distancias (m_distances)
  
    for i in range(npoints_del):
        point0 = p0[i]
        
        for j in range(npoints_del):
            if j >= 1:
                point1 = p0[j]
                dist = float(math.sqrt(math.pow(point0[0] - point1[0], 2) + math.pow(point0[1] - point1[1], 2) + math.pow(
                       point0[2] - point1[2], 2)))
                m_distances[i, j] = dist
                m_distances[j, i] = dist
                if j == 1 and i == 0:
                    dist_init = dist
                else:
                    if dist_init < dist:
                        dist_init = dist
                        #print "dist_0:", dist_init
                    else:
                        pass
    # Compute d1
    #dist_init siempre sera la maxima no?
    max_distance = np.max(m_distances)
    if max_distance != dist_init:
        max_distance = dist_init
    else:
        pass
    #busca entre que 2 puntos da la distancia maxima (el index que le pasas a p0)
    points_d1 = np.where(m_distances == max_distance)[0]
    id_d1_0 = points_d1[0]
    id_d1_1 = points_d1[1]
    p0_d1 = p0[id_d1_0]
    p1_d1 = p0[id_d1_1]

    # Obtain mean 
    
    
    mean_p = np.empty([3])  #x,y,z?
    mean_p[0] = (p0_d1[0] + p1_d1[0]) / 2
    mean_p[1] = (p0_d1[1] + p1_d1[1]) / 2
    mean_p[2] = (p0_d1[2] + p1_d1[2]) / 2

    n_dist = np.empty([npoints_del])

    for i in range(npoints_del):
        points = p0[i]
        dist_d2 = math.sqrt(math.pow(mean_p[0] - points[0], 2) + math.pow(mean_p[1] - points[1], 2) +
                            math.pow(mean_p[2] - points[2], 2))
        n_dist[i] = dist_d2

    min_distance = np.min(n_dist)
    points_d2 = np.where(n_dist == np.min(n_dist))[0]
    id_d2_0 = points_d2[0]
    p0_d2 = p0[id_d2_0]

    # D2 COMPLETE
    angle_vec = np.empty([npoints_del])
    dif = np.empty([npoints_del])
    vec = normalizevector(np.subtract(mean_p, p0_d2))

    for j in range(npoints_del):
        vec_2 = normalizevector(np.subtract(mean_p, p0[j]))
        angle_vec[j] = math.degrees(np.arccos(np.clip(np.dot(vec, vec_2), -1.0, 1.0)))
        dif[j] = 180 - angle_vec[j]

    indx_angle = np.where(dif == np.min(dif))[0]
    indx_j = indx_angle[0]

    dist_min = math.sqrt(math.pow(p0_d2[0] - p0[indx_j][0], 2) + math.pow(p0_d2[1] - p0[indx_j][1],
                                    2) + math.pow(p0_d2[2] - p0[indx_j][2], 2))

    points_d2_2 = p0[indx_j]

    show_points == False
    if show_points == True:

        list_actors=[]
        for i in range(len(p0)):
            actor = addpoint_show(p0[i], color=[0.70, 0.13, 0.13])
            list_actors.append(actor)

        rend5 = vtk.vtkRenderer()
        vtk_show_multi(rend5, LAA_edge, 1000, 1000, list_actors)
        vtk_show(rend5, LAA_edge, 1000, 1000 )


    return min_distance, max_distance, mean_p, p0_d2, p0_d1, p1_d1, dist_min, points_d2_2,LAA_edge,p0, boundaryIds


def extractclosest_indexedregion_2(polydata,ostium_centre,flag=False):
    # NOTE: extracts connected component (not edge) with highest cut diameter
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputData(polydata)
    surfer.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(surfer.GetOutputPort())
    cleaner.Update()

    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInputConnection(cleaner.GetOutputPort())
    connect.SetExtractionModeToAllRegions()
    connect.Update()
    n = connect.GetNumberOfExtractedRegions()

    # if n>1 :
    #     #print('WARNING.',n, 'connected regions')
    #     #print('Extracting connected region closest to ostium...')
    #     h=0
    
    
    #print('sizes',sizes)

    connect.ColorRegionsOn()
    connect.Update()

    regions_center = []
    
    list_centros = []
    list_areas = []
    list_distance = []
    list_regions = []
    list_d1 = []
    for i in range(n):
        region = extractindexedregion(polydata,i)
        edges = extractboundaryedge(region)

        surface_area = vtk.vtkMassProperties()
        surface_area.SetInputData(region)
        area = surface_area.GetSurfaceArea()
            #print('region',i,'area',area)
        list_areas.append(area)
        list_regions.append(region)
    
        d2_min, d1_complete, mean_p, p0d2, p0d1, p1d1, d2_complete, p1d2,LAA_edge,p0,boundaryIds = get_d1d2alt(edges, show_points=False) 
        list_d1.append(d1_complete)
        center_region = estimate_center(edges)
        regions_center.append(center_region)
        actor_centro2  = addpoint_show(center_region, color=[0, 0, 1])
        list_centros.append(actor_centro2)
        distance = calculateDistance(center_region,ostium_centre)
        list_distance.append(distance)


    largest_index = list_d1.index(max(list_d1))
    clip = extractindexedregion(polydata,largest_index)
   
    return clip, list_d1,list_regions


def extractclosest_indexedregion_3(polydata,centre_good,ostium_centre,regions_center_list):

    # NOTE: preventive measures: clean before connectivity filter
    # to avoid artificial regionIds
    # It slices the surface down the middle
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputData(polydata)
    surfer.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(surfer.GetOutputPort())
    cleaner.Update()

    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInputConnection(cleaner.GetOutputPort())
    connect.SetExtractionModeToAllRegions()
    connect.Update()


    n = connect.GetNumberOfExtractedRegions()

    # if n>1 :
    #     print('WARNING.',n, 'connected regions')
    #     print('Extracting connected region closest to ostium...')
    
    sizes = connect.GetRegionSizes()
    
    #print('sizes',sizes)

    connect.ColorRegionsOn()
    connect.Update()

    regions = []
    regions_center = []
    
    list_regions = []
    list_index = []
    distanceToCenter_array = []

    # list_2= []
    # for i in range(len(regions_center_list)):
    #     list_2.append(calculateDistance(regions_center_list[i],ostium_centre))
    # index = list_2.index(max(list_2))
    # new_centre = regions_center_list[index]

    dista = np.zeros((n,len(regions_center_list)))
    list_actors = []
    for i in range(n):
        region = extractindexedregion(polydata,i)
        # rend = vtk.vtkRenderer()
        # vtk_show(rend, region, 1000,1000)
        centre = estimate_center(region)
        if np.ceil(centre[0]).astype('int') == np.ceil(centre_good[0]).astype('int'):
            id_la = i
            dista[id_la,:] = 10000
            continue
        edges = extractboundaryedge(region)
        center_region = estimate_center(edges)
        
        #center_region = estimate_center(region)
        ##regions.append(region_array)
        regions_center.append(center_region)
        list_actors.append(addpoint_show(center_region, color = [1,0,0]))
        for j in range(len(regions_center_list)):
            dista[i,j] = calculateDistance(regions_center_list[j],center_region)
        #distanceToCenter_array.append(calculateDistance(center_region,new_centre))
    index_min = np.unravel_index(dista.argmin(), dista.shape)
    list_index.append(i)
    list_regions.append(region)

    
    
    distances = []
    furthestIndexes = []
    furthestDistances = []

    
    #esto estaba así
    # for i in range(len(regions_center)):
    #     point1 = regions_center[i]
    #     distanceToCenter_array.append(calculateDistance(point1,ostium_centre))
        
    clip = extractindexedregion(polydata,index_min[0])
    #return clip, list_regions, list_index
    return clip


def calculateDistance(point1,point2):
    return math.sqrt(math.pow(point1[0]-point2[0],2) + math.pow(point1[1]-point2[1],2) + math.pow(point1[2]-point2[2],2))

def extractclosest_indexedregion(polydata,ostium_centre,flag=False):
    # NOTE: preventive measures: clean before connectivity filter
    # to avoid artificial regionIds
    # It slices the surface down the middle
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputData(polydata)
    surfer.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(surfer.GetOutputPort())
    cleaner.Update()

    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInputConnection(cleaner.GetOutputPort())
    connect.SetExtractionModeToAllRegions()
    connect.Update()
    n = connect.GetNumberOfExtractedRegions()

    if n>1 :
        print('WARNING.',n, 'connected regions')
        print('Extracting connected region closest to ostium...')
    
    sizes = connect.GetRegionSizes()
    
    #print('sizes',sizes)

    connect.ColorRegionsOn()
    connect.Update()

    regions = []
    regions_center = []
    
    for i in range(n):
        region = extractindexedregion(polydata,i)
        edges = extractboundaryedge(region)
        center_region = estimate_center(edges)
        #center_region = estimate_center(region)
        ##regions.append(region_array)
        regions_center.append(center_region)
    
    
    distances = []
    furthestIndexes = []
    furthestDistances = []
    distanceToCenter_array = []
    
    for i in range(len(regions_center)):
        point1 = regions_center[i]
        distanceToCenter_array.append(calculateDistance(point1,ostium_centre))
        
    if flag == False:
        closest_index = np.argmin(distanceToCenter_array)

        clip = extractindexedregion(polydata,closest_index)
        return clip

    if flag == True:
        furthest_index = np.argmax(distanceToCenter_array)
        #del distanceToCenter_array[furthest_index]

        connectivity0 = vtk.vtkPolyDataConnectivityFilter()
        connectivity0.SetInputConnection(connect.GetOutputPort())
        connectivity0.SetExtractionModeToSpecifiedRegions()
        #for i in range(len(distanceToCenter_array)):
        #    if i != furthest_index:
        #        connectivity0.AddSpecifiedRegion(i)
        #connectivity0.Update()
        list_areas=[]

        for i in range(len(distanceToCenter_array)):
            connectivity0.AddSpecifiedRegion(i)

            surface_area = vtk.vtkMassProperties()
            surface_area.SetInputConnection(connectivity0.GetOutputPort())
            area = surface_area.GetSurfaceArea()
            #print('region',i,'area',area)
            list_areas.append(area)
            connectivity0 = vtk.vtkPolyDataConnectivityFilter()
            connectivity0.SetInputConnection(connect.GetOutputPort())
            connectivity0.SetExtractionModeToSpecifiedRegions()
        
        biggest_area = np.argmax(list_areas)
    
        connectivity1 = vtk.vtkPolyDataConnectivityFilter()
        connectivity1.SetInputConnection(connect.GetOutputPort())
        connectivity1.SetExtractionModeToSpecifiedRegions()
        for i in range(len(distanceToCenter_array)):
            if i != biggest_area:
                connectivity1.AddSpecifiedRegion(i)
        #connectivity1.DeleteSpecifiedRegion(biggest_area)
        connectivity1.Update()

        connectivity2 = vtk.vtkPolyDataConnectivityFilter() 
        connectivity2.SetInputConnection(connectivity1.GetOutputPort())
        connectivity2.SetExtractionModeToLargestRegion()
        connectivity2.Update()
       
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(connectivity2.GetOutputPort())
        cleaner.Update()
        return cleaner.GetOutput()
 

def return_edge(surface, origin,normal):

    LAA_clip = planeclip(surface, origin, normal, insideout=0)
    area_largest1 = extractlargestregion(LAA_clip)
    polygonProperties = vtk.vtkMassProperties()
    polygonProperties.SetInputData(area_largest1)
    polygonProperties.Update()
    area1 = polygonProperties.GetSurfaceArea()
    #print('area clip1', area1)

    LAA_clip_op = planeclip(surface, origin, normal, insideout=1)
    area_largest2= extractlargestregion(LAA_clip_op)
    polygonProperties = vtk.vtkMassProperties()
    polygonProperties.SetInputData(area_largest2)
    polygonProperties.Update()
    area2 = polygonProperties.GetSurfaceArea()

    #print('area clip2', area2)
  

    if area1 > area2:
        LAA_clip_n = LAA_clip_op
        LAA_clip_op_n = LAA_clip
    else:
        LAA_clip_n = LAA_clip
        LAA_clip_op_n = LAA_clip_op
 
    #rendc = vtk.vtkRenderer()
    
    #vtk_show(rendc, LAA_clip_n, 1000, 1000)
    #-------------------
    ##clip = extractclosest_indexedregion(LAA_clip_n,origin,flag=False)
    #---------------------
    #clip = extractclosest_indexedregion_marta(LAA_clip_n,origin,flag=False)

    # if case == 163:
    #     #extracts correct region by closest edge center
    #     clip, list_areas, list_regions = extractclosest_indexedregion_3(LAA_clip_n,origin,flag=False)
    # else:
    #extracts correct region by largest diameter
    #clip, list_areas, list_regions = extractclosest_indexedregion_2(LAA_clip_n,origin,flag=False)
    clip = extractclosest_indexedregion(LAA_clip_n,origin,flag=False)

    #if case 33 or 7
    #clip = extractlargestregion(LAA_clip_n)
        

    #print('extractclosest_indexedregion2')
    #rendc = vtk.vtkRenderer()
    #vtk_show(rendc, clip, 1000, 1000)

    #EXTRACT EDGES (AND DETECT IF THE LAA IS BADLY CLIPPED)
    edges = extractboundaryedge(clip)
    #if case == '214':
    #     edge_noDelaunay, surface_area, LAA_edge, LAA_error = extractclosestregion_edges(edges,origin)
    #else:
    edge_noDelaunay, surface_area, LAA_edge, LAA_error = extractlargestregion_edges_m2(edges)
    cent = estimate_center(surface)

    return  LAA_edge

def return_edge_2(surface, origin,normal):

    LAA_clip = planeclip(surface, origin, normal, insideout=0)
    area_largest1 = extractlargestregion(LAA_clip)
    polygonProperties = vtk.vtkMassProperties()
    polygonProperties.SetInputData(area_largest1)
    polygonProperties.Update()
    area1 = polygonProperties.GetSurfaceArea()
    #print('area clip1', area1)

    LAA_clip_op = planeclip(surface, origin, normal, insideout=1)
    area_largest2= extractlargestregion(LAA_clip_op)
    polygonProperties = vtk.vtkMassProperties()
    polygonProperties.SetInputData(area_largest2)
    polygonProperties.Update()
    area2 = polygonProperties.GetSurfaceArea()

    #print('area clip2', area2)
  

    if area1 > area2:
        LAA_clip_n = LAA_clip_op
        LAA_clip_op_n = LAA_clip
    else:
        LAA_clip_n = LAA_clip
        LAA_clip_op_n = LAA_clip_op
 

    #extracts correct region by largest diameter
    clip, list_areas, list_regions = extractclosest_indexedregion_2(LAA_clip_n,origin,flag=False)

    
    edges = extractboundaryedge(clip)
    #if case == '214':
    #     edge_noDelaunay, surface_area, LAA_edge, LAA_error = extractclosestregion_edges(edges,origin)
    #else:
    edge_noDelaunay, surface_area, LAA_edge, LAA_error, regions_center = extractlargestregion_edges_m2(edges)

    cent = estimate_center(surface)

    return  LAA_edge

def brute_force_perturbation(pred_surface,normal, centroid, angle, count, no = 20):

    #NOTE: small perturbations of centroid and normal angle, find cut with lowest diameter (ostium)
    
        np.random.seed(1234)
        if count == 2:
            centroid = np.array((centroid[0], centroid[1], centroid[2]+5))

        normal_proposals = np.random.uniform(-angle,angle,(no,3)) + normal
        normal_proposals = np.vstack((normal,normal_proposals))
        centroid_proposals = np.random.uniform(-1,1,(no,3)) + centroid
        centroid_proposals = np.vstack((centroid,centroid_proposals))
        min_diam = np.inf
        
        for proposal in range(no):

            #this function will cut the laa and get the ostium cut edge, not other structures (BASED ON LARGEST DIAMETER)
            LAA_edge = return_edge_2(pred_surface,centroid_proposals[proposal,:],normal_proposals[proposal,:])          
            
            d2_min, d1_complete, mean_p, p0d2, p0d1, p1d1, d2_complete, p1d2,LAA_edge,p0,boundaryIds = get_d1d2alt(LAA_edge, show_points=False) 

            if d1_complete < min_diam and d1_complete>10:
                min_diam = d1_complete
                normal_opt = normal_proposals[proposal,:]
                centroid_opt = centroid_proposals[proposal,:]
                edge = LAA_edge
              

     
        print("Min D1:", min_diam)
        th = 40
       
        if min_diam>th: 
            count = count + 1
           
            print("No good estimate found - retrying")
            if count == 1:
                normal_opt, centroid_opt = brute_force_perturbation(pred_surface,normal, centroid , 0.6, count,  no = 20)
            if count == 2:
                normal_opt, centroid_opt = brute_force_perturbation(pred_surface,normal, centroid , 1.0, count, no = 100)
            #elif count == 3:
            #    break

        return normal_opt, centroid_opt


def brute_force_perturbation_2(pred_surface,normal, centroid, angle, count,next_rotated, no = 20):

    #NOTE: small perturbations of centroid and normal angle, find cut with lowest diameter (ostium)
    #Here you know the next point in the centerline (next_rotated)

        flag = False
        np.random.seed(1234)
        if count == 0:
            centroid_ori = centroid

        if count == 2: #if after 2 tries the diam. is too big, move centroid point along the centerline
            centroid = next_rotated
            flag = True

        normal_proposals = np.random.uniform(-angle,angle,(no,3)) + normal
        normal_proposals = np.vstack((normal,normal_proposals))
        centroid_proposals = np.random.uniform(-1,1,(no,3)) + centroid
        centroid_proposals = np.vstack((centroid,centroid_proposals))
        min_diam = np.inf
        
        for proposal in range(no):
            
            LAA_edge = return_edge_2(pred_surface,centroid_proposals[proposal,:],normal_proposals[proposal,:]) 

            #this function computes the ostium diameter given the laa edge
            d2_min, d1_complete, mean_p, p0d2, p0d1, p1d1, d2_complete, p1d2,LAA_edge,p0,boundaryIds = get_d1d2alt(LAA_edge, show_points=False) 

            if d1_complete < min_diam and d1_complete>10:
                min_diam = d1_complete
                normal_opt = normal_proposals[proposal,:]
                centroid_opt = centroid_proposals[proposal,:]
                edge = LAA_edge
     
        print("Min D1:", min_diam)
        th = 40 #threshold for too big diameter

        if min_diam>th: 
            count = count + 1
            print("No good estimate found - retrying") #every try I increase the range of ostium normal angles
            if count == 1:
                normal_opt, centroid_opt, flag = brute_force_perturbation_2(pred_surface,normal, centroid , 0.6, count,next_rotated, no = 20)
            if count == 2:
                normal_opt, centroid_opt, flag = brute_force_perturbation_2(pred_surface,normal, centroid , 1.0, count,next_rotated, no = 100)
     

        return normal_opt, centroid_opt,flag


def reconstruct_LAA(clip,LAA_clip_op_n,origin):

    print('Reconstructing the LAA...')
    #find the closest region to LAA in the reverse clip
    clip_op = extractclosest_indexedregion(LAA_clip_op_n,origin,flag=True)
    appendm = vtk.vtkAppendPolyData()
    appendm.AddInputData(clip)
    appendm.AddInputData(clip_op)
    appendm.Update()   
    clip = appendm.GetOutput() 
    clip = extractlargestregion(clip)


    return clip


def reconstruct_LAA_2(clip,LAA_clip_op_n,centre_good,origin, regions_center):
    print('Reconstructing the LAA...')
    # find the adequate region in the good clip
    clip_op = extractclosest_indexedregion_3(LAA_clip_op_n,centre_good,origin,regions_center)
    appendm = vtk.vtkAppendPolyData()
    appendm.AddInputData(clip)
    appendm.AddInputData(clip_op)
    appendm.Update()   
    clip = appendm.GetOutput()
    clip = extractlargestregion(clip)
            
    return clip



def clip_LAA(file,origin,normal, viz = False):
    
    #NOTE: Clip the LAA, detect the LAA region, detect if the LAA has been cut twice, reconstruct the LAA

    #Detect which clip insideout has the LAA
    LAA_clip = planeclip(file, origin, normal, insideout=0)
    #area clip1
    area_largest1 = extractlargestregion(LAA_clip)
    polygonProperties = vtk.vtkMassProperties()
    polygonProperties.SetInputData(area_largest1)
    polygonProperties.Update()
    area1 = polygonProperties.GetSurfaceArea()
    LAA_clip_op = planeclip(file, origin, normal, insideout=1)


    #area clip2
    area_largest2= extractlargestregion(LAA_clip_op)
    polygonProperties = vtk.vtkMassProperties()
    polygonProperties.SetInputData(area_largest2)
    polygonProperties.Update()
    area2 = polygonProperties.GetSurfaceArea()

    if area1 > area2:
        LAA_clip_n = LAA_clip_op
        LAA_clip_op_n = LAA_clip
    else:
        LAA_clip_n = LAA_clip
        LAA_clip_op_n = LAA_clip_op
       
    clip = extractclosest_indexedregion(LAA_clip_n,origin,flag=False)
    edges = extractboundaryedge(clip)
    centre_good = estimate_center(clip)

    #clip_ori = clip
    #forma1
    npoints = edges.GetNumberOfPoints()
    edge_noDelaunay, surface_area, LAA_edge, LAA_error, regions_center = extractlargestregion_edges_m2(edges)
    cent = estimate_center(file)
    
    edges = extractboundaryedge(LAA_edge)
    npoints = edges.GetNumberOfPoints()
    p0 = []
    #p0_ids = []
    for i in range(npoints):
        p0.append(edges.GetPoint(i))

    if viz == True:
        list_actors=[]
        #ostium vidaa
        for point_i in range(len(p0)):
            actor = addpoint_show(p0[point_i], color=[0.13, 0.13, 0.7])
            list_actors.append(actor)  
        rend2 = vtk.vtkRenderer()
        vtk_show_multi(rend2, file, 1000, 1000, list_actors)

    if LAA_error == True:
        clip = reconstruct_LAA(clip,LAA_clip_op_n,origin)

    #If the LAA piece to reconstruct is in the same insideout clip as the LAA (not common)
    edge_noDelaunay, surface_area, LAA_edge, LAA_error, regions_center = extractlargestregion_edges_m2(edges)
    if LAA_error == True:
        print('Failed reconstruction')
        clip = reconstruct_LAA_2(clip,LAA_clip_n,centre_good,origin, regions_center)


    return clip, p0

def writestl(surface, filename, type='ascii'):
    """Write binary or ascii VTK file"""
    writer = vtk.vtkSTLWriter()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        writer.SetInputData(surface)
    else:
        writer.SetInput(surface)
    writer.SetFileName(filename)
    if type == 'ascii':
        writer.SetFileTypeToASCII()
    elif type == 'binary':
        writer.SetFileTypeToBinary()
    writer.Write()

def fix_mesh(path_clip_bueno_stl, path_clip_output): ##puedo intentar hacerlo 2 veces si no funciona bien
    tin = _meshfix.PyTMesh()
    tin.load_file(path_clip_bueno_stl)
    # Clean Mesh
    print("Cleaning mesh...")
    tin.select_intersecting_triangles()
    #tin.join_closest_components()
    tin.fill_small_boundaries()
    tin.clean(max_iters=100, inner_loops=2)
    tin.save_file(path_clip_output)
    repaired = readstl(path_clip_output)
    

    #segunda vez?
    tin = _meshfix.PyTMesh()
    tin.load_file(path_clip_output)
    
#    # Clean Mesh
    print("Cleaning mesh...")
    tin.select_intersecting_triangles()
    #tin.join_closest_components()
    tin.fill_small_boundaries()
    tin.clean(max_iters=100, inner_loops=2)
    tin.save_file(path_clip_output)
   

def skippoints(polydata, nskippoints):
    """Generate a single cell line from points in idlist."""
    # derive number of nodes
    numberofnodes = polydata.GetNumberOfPoints() - nskippoints

    # define points and line
    points = vtk.vtkPoints()
    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(numberofnodes)

    # assign id and x,y,z coordinates
    for i in range(nskippoints,polydata.GetNumberOfPoints()):
        pointid = i - nskippoints
        polyline.GetPointIds().SetId(pointid,pointid)
        point = polydata.GetPoint(i)
        points.InsertNextPoint(point)

    # define cell
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyline)

    # add to polydata
    polyout = vtk.vtkPolyData()
    polyout.SetPoints(points)
    polyout.SetLines(cells)
    #if not vtk.vtkVersion.GetVTKMajorVersion() > 5:
    #    polyout.Update()
    return polyout

######### heat equation functions

def detectOstium(meshFile,list_os,normal,origin):
 #código de Ainhoa modificado

 vertices = meshFile.points

 #find mesh vertices closest to ostium edge (list_os)
 nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(vertices)
 distances_2, indices_2 = nbrs.kneighbors(list_os)
           
 list_points = []
 for i in range(len(indices_2)):
    ver = indices_2[i]
    point = meshFile.points[ver]
    list_points.append(point[0])

 list_np = np.array(list_points)

 point = origin

 A = normal[0]
 B = normal[1]
 C = normal[2]
#define ostium plane equation
 D = -(A * point[0] + B * point[1] + C * point[2])

 ostium = np.zeros((len(vertices), 3))
 LAA_wall = np.zeros((len(vertices), 3)) 
 #se crea un numpy array donde irán los vertices, eso son los boundary conditions?? 

 sum_wall = 0
 sum_ostium = 0
 sumUnreducedOstium = 0
 x = 0
 y = 0
 z = 0
 indexes = []
 #para cada vértice.... es siempre la misma normal pero distinto origen del plano
 for i in range(len(vertices)):
  x0 = vertices[i, 0]
  y0 = vertices[i, 1]
  z0 = vertices[i, 2]
  d = abs(A * x0 + B * y0 + C * z0 + D) / math.sqrt(A ** 2 + B ** 2 + C ** 2)
  
  if d < 3:#d < 0.15:
   ostium[sum_ostium, :] = vertices[i, :] #solo guardas los mesh vertices a una dist. < 3 del ostium plane
   sum_ostium = sum_ostium + 1 #esto son contadores, para guardar valores
   sumUnreducedOstium = sumUnreducedOstium + 1

   x = x + vertices[i,0]
   y = y + vertices[i,1]
   z = z + vertices[i,2]
   indexes.append(i)
  else:
   #esto es lo importante, osea solo guarda si d >3
   LAA_wall[sum_wall, :] = vertices[i, :]
   sum_wall = sum_wall + 1

 #ostium son los vertices con plano a distancia < 3
 #LAA_wall son los vertices con plano a distancia > 3

 LAA_wall = LAA_wall[0:sum_wall, :]
 
 x = x / sumUnreducedOstium
 y = y / sumUnreducedOstium
 z = z / sumUnreducedOstium
 ostium_centre = [x,y,z]
 
 ostium_array = list_np
 #print("Ostium centre:",x,y,z)
 return ostium_array, LAA_wall, ostium_centre

def readUnstructuredGridVTK(filePath):
  reader = vtk.vtkUnstructuredGridReader()
  reader.SetFileName(filePath)
  reader.SetScalarsName('heat')
  reader.Update()
  return reader.GetOutput()


def defineBoundaryConditions(grid, ostium, LAA_wall):
 vol_points = grid.points

 nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(vol_points)
 distances, indices_ostium = nbrs.kneighbors(ostium)
 # quieres buscar los puntos de la malla mas cercanos al ostium_edge


 nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(vol_points)
 distances, indices_wall = nbrs.kneighbors(LAA_wall)
 # quieres buscar los puntos de la malla mas cercanos a los vertices lAA wall (CREO QUE LO DE LA D ES PARA ASEGURARS DE QUE NO COGE LA MALLA? ME GUSTARIA VISUALIZAR ESOS VERTICES)

 #ENTONCES LOS PUNTOS DE LA MALLA MAS CERCANOS AL OSTIUM EDGE Y AL LAA WALL SON LOS BOUNDARY POINTS
 boundaryPoints = np.zeros((len(indices_wall) + len(ostium), 3))
 boundaryConditions = np.zeros((len(indices_wall) + len(ostium), 1))
 # TODO ESTO ES PARA ENCONTRAR LOS BOUNDARY POINTS Y DECIR QUE ENE ESOS PUNTOS LOS CONDITIONS ES 1
 
 for i in range(len(indices_wall)):
  boundaryPoints[i, :] = vol_points[indices_wall[i]]

 sum_x = 0
 sum_y = 0
 sum_z = 0
 
 for i in range(len(indices_ostium)):
  index = i + len(indices_wall)
  boundaryPoints[index, :] = vol_points[indices_ostium[i]]
  
  sum_x = sum_x + vol_points[indices_ostium[i],0]
  sum_y = sum_y + vol_points[indices_ostium[i],1]
  sum_z = sum_z + vol_points[indices_ostium[i],2]
  boundaryConditions[index, :] = 1

 ostium_centre = [sum_x / len(indices_ostium), sum_y / len(indices_ostium), sum_z / len(indices_ostium)]
 
 return boundaryConditions, boundaryPoints

class ClosestPointStupid:
    def __init__(self, points, val, vtkMesh):
        self.points = points
        self.val = val
        self.j = 0
        self.locator = vtk.vtkCellLocator()
        self.locator.SetDataSet(vtkMesh)
        self.locator.BuildLocator()
        
    def findClosestPoint(self, p):
        subId = vtk.mutable(0) 
        meshPoint = np.zeros(3)
        cellId = vtk.mutable(0) 
        dist2 =  vtk.mutable(0.) 
        self.locator.FindClosestPoint(p, meshPoint, cellId, subId, dist2)
    
    def interpolate(self, coors):
        """
        Define in base of coordinates...

        See if interpolations are possible
        """
        i = np.argmin(np.linalg.norm(coors - self.points, axis = 1))
        return self.val[i]


def solveLaplaceEquationTetrahedral(mesh,meshVTK, boundaryPoints, boundaryConditions):
    """
    mesh: path to a 3D mesh / sfepy mesh
    
    """

    if isinstance(mesh, str):
        mesh = Mesh.from_file(mesh)
    
    # #Set domains  
    #1. Identify computational domain
    domain = FEDomain('domain', mesh)
    omega = domain.create_region('Omega', 'all')
    #2. Identify the boundary conditions
    boundary = domain.create_region('gamma', 'vertex  %s' % ','.join(map(str, range(meshVTK.GetNumberOfPoints()))), 'facet')

    #set fields
    field = Field.from_args('fu', np.float64, 1, omega, approx_order=1)
    u = FieldVariable('u', 'unknown', field)
    v = FieldVariable('v', 'test', field, primary_var_name='u')
    m = Material('m', val = [1.])

    #Define element integrals
    integral = Integral('i', order=3)

    #Equations defining 
    t1 = Term.new('dw_laplace( v, u )',
            integral, omega,v=v, u=u)
    eq = Equation('balance', t1)
    eqs = Equations([eq])
    
    heatBoundary = boundaryConditions
    points = boundaryPoints

    #Boundary conditions
    c = ClosestPointStupid(points,heatBoundary, meshVTK)

    def u_fun(ts, coors, bc=None, problem=None, c = c):
        c.distances = []
        v = np.zeros(len(coors))
        for i, p in enumerate(coors):
            #print(p)
            v[i] = c.interpolate(p)
            #c.findClosestPoint(p)
        return v

    bc_fun = Function('u_fun', u_fun)
    fix1 = EssentialBC('fix_u', boundary, {'u.all' : bc_fun})
    
    #Solve problem
    ls = ScipyDirect({})
    nls = Newton({}, lin_solver=ls)

    pb = Problem('heat', equations=eqs)
    pb.set_bcs(ebcs=Conditions([fix1]))
    
    pb.set_solver(nls)
    state = pb.solve(verbose = False)
    u_m = state()
    print(u_m.shape)
    print(u_m)
    #u = state.get_parts()['u']
    return u_m

def add_scalar(mesh, array, name = 'scalarField', domain = 'point'):
    scalars = vtk.vtkFloatArray()
    scalars.Initialize()
    scalars.SetName(name)
    scalars.SetNumberOfComponents(1)
    for i, v in enumerate(array):
        scalars.InsertNextValue(v)
    if domain == 'point':
        mesh.GetPointData().AddArray(scalars)
    else:
        mesh.GetCellData().AddArray(scalars)

    return mesh

def writeUnstructuredGridVTK(filePath, vtkUnstructuredGrid):
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(filePath)
    writer.SetInputData(vtkUnstructuredGrid)
    writer.SetFileTypeToASCII()
    writer.Update()
    writer.Write()
    
def detectEndPointOfCentreline(result1,grid):
  small = 0
  cont = 0
  index = 0
  result = np.gradient(result1)
  for i in range(len(result)):
    if(result[i] > 0 and result[i] < 1):
        if(cont == 0):
            small = result[i]
            cont = cont + 1
            index = 1
        else:
            if(result[i] < small):
                cont = cont + 1
                small = result[i]
                index = i
  print(cont)  
  #print('results',results[index] )   
  print("The end point of the centreline is...",grid.points[index])
  return grid.points[index]



def vmtkcenterlines(surface, sourcepoints, targetpoints, endpoints=0):
    #computer = vmtkscripts.vmtkCenterlines()
    computer = vmtkCenterlines()
    computer.Surface = surface
    computer.SeedSelectorName = 'pointlist'
    computer.SourcePoints = sourcepoints
    computer.TargetPoints = targetpoints
    computer.AppendEndPoints = endpoints
    computer.Execute()
    return computer.Centerlines

def vmtkcenterlinegeometry(centerline,iterations=100, factor=0.5):
    #computer = vmtkscripts.vmtkCenterlines()
    computer = vmtkCenterlineGeometry()
    computer.Centerlines = centerline
    computer.LineSmoothing = 1
    computer.NumberOfSmoothingIterations = iterations
    computer.SmoothingFactor = factor
    computer.Execute()
    return computer.Centerlines


def vmtkcenterlineresampling(centerline, length=.1):
    resampler = vmtkCenterlineResampling()
    resampler.Centerlines = centerline
    resampler.Length = length
    resampler.Execute()
    return resampler.Centerlines

def vmtkcenterlinesmoothing(centerline, iterations=100, factor=0.1):
    smoother = vmtkCenterlineSmoothing()
    smoother.Centerlines = centerline
    smoother.NumberOfSmoothingIterations = iterations
    smoother.SmoothingFactor = factor
    smoother.Execute()
    return smoother.Centerlines



def calculateDistanceToOstium(CL_points, ostium):
    distances = []
    for i in range(len(CL_points)):
        point1 = CL_points[i]
        distance = math.sqrt((point1[0] - ostium[0]) ** 2 +
                     (point1[1] - ostium[1]) ** 2 +
                     (point1[2] - ostium[2]) ** 2)
        distances.append(distance)
    
    return distances

def find_intersect(mesh, psource, ptarget):
    """
    Find the intersection between a mesh and a line defined by ptarget and psource
    :param filename1: Path of the input mesh
    :param psource: Origin of the "intersected" line
    :param ptarget: End point of the "intersected" line
    :return: Points in which the line and the mesh intersect and flag (1 intersected, 0 no intersection)
    """
 

    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(mesh)
    obbTree.BuildLocator()

    pointsVTKintersection = vtk.vtkPoints()
    flag_intersect = obbTree.IntersectWithLine(psource, ptarget, pointsVTKintersection, None)

    pointsVTKIntersectionData = pointsVTKintersection.GetData()
    noPointsVTKIntersection = pointsVTKIntersectionData.GetNumberOfTuples()
    pointsIntersection = []

    for idx in range(noPointsVTKIntersection):
        _tup = pointsVTKIntersectionData.GetTuple3(idx)
        pointsIntersection.append(_tup)

    return pointsIntersection, flag_intersect

def get_halt(mesh, ori):
    """
    Get H considering the LAA center in the base (Clipped)
    :param filename1: Path to read the .vtk file containing the LAA
    :param ori: origin of the LAA (center of the base)
    :return: H - distance from the LAA base to the surface (mm) and intersection with the surface
    """

    final = np.add(ori, [0, 0, 80])
    origin = np.add(ori, [0, 0, 0.2])
    # actor1 = addpoint_show(final, color=[0.70, 0.13, 0.13])
    # actor2 = addpoint_show(origin, color=[0.13, 0.7, 0.13])
    # rend = vtk.vtkRenderer()
    # rotated = readvtk(filename1)
    # vtk_show(rend, rotated, 1000, 1000,actor1,actor2)


    intersect, flag = find_intersect(mesh, origin, final)
    flagh = False
    if not intersect:
       origin = np.add(ori, [0, 0, 0.2])
       final = np.add(ori, [0, 0, -80])
       intersect, flag = find_intersect(mesh, origin, final)
       flagh = True
    else:
       pass

    h_comp = np.subtract(intersect[0], origin)
    #print "H comp:", h_comp
    h = math.sqrt(math.pow(h_comp[0], 2) + math.pow(h_comp[1], 2) + math.pow(h_comp[2], 2))

    return h, intersect, h_comp, origin, flagh

def get_ids_ordered_contour(cont):
    # Given a contour, get the order of the points following the connectivity of the lines (cells)
    lines = np.zeros((cont.GetNumberOfCells(), 2))  # [p1; p2]
    for i in range(cont.GetNumberOfCells()):
        ptids = cont.GetCell(i).GetPointIds()
        lines[i, :] = np.array([ptids.GetId(0), ptids.GetId(1)])
    matrix = lines.astype(int)
    ordered_cont_points = np.zeros(matrix.shape[0]) - 1  # initialize to -1 because 0 is a suitable point id

    # initialize first 2 points (first segment)
    ordered_cont_points[0] = matrix[0, 0]
    ordered_cont_points[1] = matrix[0, 1]
    first_col = matrix[:, 0]  # integers
    second_col = matrix[:, 1]
    reverse = False

    for i in range(1, matrix.shape[0] - 1):
        if reverse == False:
            pos = np.where(first_col == ordered_cont_points[i])[0]
            if pos.size == 0:  # there is no continuity in this direction, start reverse mode
                second_pos_options = np.where(second_col == ordered_cont_points[i])[0]  # only 2 options I think
                # check which ones I already have in ordered_cont_points
                if second_pos_options[0] in ordered_cont_points:
                    second_pos = second_pos_options[1]  # keep second value in the where().
                else:
                    second_pos = second_pos_options[0]  # keep first value in the where().
                ordered_cont_points[i + 1] = first_col[
                    second_pos]  # in this case, the next point is in the first column, then continue in reverse order
                reverse = True
            else:
                if pos.shape[0] > 1:  # loop possibility. One point has 2 continuities, pick smallest one the first time and second the other time...
                    # check if one point is already in the path (second time here).
                    option1 = pos[0]
                    option2 = pos[1]
                    if second_col[option1] in ordered_cont_points or second_col[option2] in ordered_cont_points:
                        ordered_cont_points[i + 1] = next_time_follow
                    else:  # first time here, need to determine which path follow
                        path1 = np.array([second_col[option1]])  # initialize with the first next in each path
                        path2 = np.array([second_col[option2]])
                        # el tema es que cuando llegue de nuevo al causante del fallo tendre el mismo problema -> dos continuaciones
                        mmm = 1
                        while path1[mmm - 1] != ordered_cont_points[i] and path2[mmm - 1] != ordered_cont_points[i]:  # distinto del causante del loop
                            next1 = np.array(second_col[np.where(first_col == path1[mmm - 1])[0]])
                            next2 = np.array(second_col[np.where(first_col == path2[mmm - 1])[0]])
                            path1 = np.concatenate((path1, next1))
                            path2 = np.concatenate((path2, next2))
                            mmm = mmm + 1
                        if ordered_cont_points[i] in path1:
                            ordered_cont_points[i + 1] = path1[0]
                            next_time_follow = path2[0]
                            # second_time = True
                        elif ordered_cont_points[i] in path2:
                            ordered_cont_points[i + 1] = path2[0]
                            next_time_follow = path1[0]
                            # second_time = True
                        else:
                            print ('Something is wrong, not sure which path should I follow.')

                else:  # easy case, only 1 next point (original pos or the pos after operation to manage loops)
                    ordered_cont_points[i + 1] = second_col[pos]
        else:  # reverse mode
            pos = np.where(second_col == ordered_cont_points[i])[0]
            if pos.size == 0:  # there is no continuity in this direction, come back to non reverse mode
                second_pos_options = np.where(first_col == ordered_cont_points[i])[0]
                if second_pos_options[0] in ordered_cont_points:
                    second_pos = second_pos_options[1]
                else:
                    second_pos = second_pos_options[0]
                ordered_cont_points[i + 1] = second_col[
                    second_pos]  # in this case, the next point is in the first column, then continue in reverse order
                reverse = False
            else:
                ordered_cont_points[i + 1] = first_col[pos]
    return ordered_cont_points


def get_ostium_perimeter(LAA_clip,origin):
    """
    Compute the perimeter of the ostium by ordering the index of the points and then summing all the distances.
    :param LAA_clip: Ostium clipped
    :return: Perimeter of the ostium
    """

    LAA_clip = extractclosest_indexedregion(LAA_clip, origin,flag=False)
    edges = extractboundaryedge(LAA_clip)
    cont = extractlargestregion(edges) 

    #npoints = cont.GetNumberOfPoints()
    
    #p0 = []
    ##p0_ids = []

    #for i in range(npoints):
    #    p0.append(cont.GetPoint(i))
    ##    p0.append(cont.GetPoint(i))
    
    #list_actors = []
    #for point_i in range(npoints):
    #    actor = addpoint_show(np.array(cont.GetPoint(point_i)), color=[0.70, 0.13, 0.13])
    #    list_actors.append(actor)
    #rend2 = vtk.vtkRenderer()
    #vtk_show_multi(rend2, LAA_clip, 1000, 1000, list_actors)
    #vtk_show(rend2, LAA_clip, 1000, 1000) 

    
    
    #vtk_show_multi(rend2, LAA_clip, 1000, 1000, list_actors)
    #vtk_show(rend2, LAA_clip, 1000, 1000) 
    
    
      # just in case... it may be some residual small hole... keep only biggest contour
    edge_cont_ids = get_ids_ordered_contour(cont).astype(int)  # get the ids of the contour ORDERED. Ids correspond to mesh contour, not complete mesh. Use locator if the complete mesh related ids are required

    npoints = cont.GetNumberOfPoints()
    
    p0 = []
    #p0_ids = []

    for i in range(npoints):
        p0.append(cont.GetPoint(i))
    #    p0.append(cont.GetPoint(i))
    
    list_actors = []
    for point_i in range(npoints):
        actor = addpoint_show(np.array(cont.GetPoint(edge_cont_ids[point_i])), color=[0.70, 0.13, 0.13])
        list_actors.append(actor)
    # rend2 = vtk.vtkRenderer()
    # vtk_show_multi(rend2, LAA_clip, 1000, 1000, list_actors)
    # vtk_show(rend2, LAA_clip, 1000, 1000) 

    peri = 0
    for i in range(edge_cont_ids.shape[0]-1):
        peri = peri + euclideandistance(cont.GetPoint(edge_cont_ids[i]), cont.GetPoint(edge_cont_ids[i+1]))

    #print 'Contour perimeter is: ', peri
    return peri


def get_hthetaalt(mesh, p0, h, cent, intersect):
    """
    Compute h_theta; distance from h/2 to the LAA apex (most distant point of the LAA)
    :param filename1: Path of the LAA file .vtk format
    :param h: H of the LAA being considered
    :return: h_theta (mm), most distant point (the one defining the line) and M (tortuosity)
    """

    f = mesh
    npoints = f.GetNumberOfPoints()
    array_distances = np.empty([1, npoints])
    p1 = []
    for j in range(npoints):
        p1.append(f.GetPoint(j))  # Get points of the file and compute the distance for all of them
        dist = math.sqrt(math.pow(p0[0] - p1[j][0], 2) + math.pow(p0[1] - p1[j][1], 2) + math.pow(p0[2] - p1[j][2], 2))
        array_distances[0, j] = dist
    



    h_theta = np.max(array_distances)  # Find the point that has the max distance
    indx = np.where(array_distances == np.max(array_distances))[1]
    #print indx
    h_point = p1[indx[0]]

    #find vector between p1 and p0
   


    # print "More distant point: ", distant_point, " Distance: ", H_theta
    # print H_theta
    # print H / (2 * H_theta + H)

    # Get measure M
    M = h/(h/2 + h_theta)



    return h_theta, h_point, M

def get_angle(v1, v2):

    return mt.acos(np.dot(v1, v2) / (normvector(v1) * normvector(v2)))
    
def get_bending (h_comp, p0, th_comp):
    """
    Estimate the bending of the appendage
    :param h_comp: Vector of the height
    :param p0: Initial position vector to compute H_theta
    :param th_comp: Vector of H_theta
    :return: Estimate of the bending
    """
    return mt.degrees(get_angle(h_comp, p0 - th_comp))

def get_bending_2 (point1, point2, point3):
    """
    Estimate the bending of the appendage
    :param h_comp: Vector of the height
    :param p0: Initial position vector to compute H_theta
    :param th_comp: Vector of H_theta
    :return: Estimate of the bending
    """
    return mt.degrees(get_angle(point1-point2, point3 - point2))

def calc_ostium_area(p0):
    ostium_numpy = np.zeros((len(p0),3))
    for i in range(len(p0)):
        ostium_numpy[i] = p0[i]

    poly = createPolyData(ostium_numpy)

    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputData(poly)
    surfer.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(surfer.GetOutputPort())
    cleaner.Update()

    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInputConnection(cleaner.GetOutputPort())
    connect.SetExtractionModeToAllRegions()
    connect.Update()

    delaunay = vtk.vtkDelaunay2D()
    delaunay.SetInputData(connect.GetOutput())
    delaunay.Update()
    #rend2 = vtk.vtkRenderer()
    #vtk_show(rend2, delaunay.GetOutput(), 1000, 1000) 

    polygonProperties = vtk.vtkMassProperties()
    polygonProperties.SetInputConnection(delaunay.GetOutputPort())
    polygonProperties.Update()
    edge_d = delaunay.GetOutput()
    writevtk(edge_d, 'edge.vtk')
    ostium_vtk = readvtk('edge.vtk')
    polygonProperties = vtk.vtkMassProperties()
    polygonProperties.SetInputData(ostium_vtk)
    polygonProperties.Update()
    ostium_area = polygonProperties.GetSurfaceArea()
    os.remove('edge.vtk')
    
    return ostium_area



def calc_h_param(edges,mesh_laa,normal, viz = False):
    cent = estimate_center(edges)
    #compute h and h_theta
    normal_align = np.array([0, 0, 1])  #eje z
    path_rotated = None
    rotated, matrix = rotate_angle(mesh_laa, normal, normal_align, cent, path_rotated)


    # actor = addpoint_show(cent, color=[0.70, 0.13, 0.13])
    # rend = vtk.vtkRenderer()
    # vtk_show(rend, rotated, 1000, 1000,actor )

    
    #print ("Computing h and h_theta...")
    h, intersect, h_comp, origin_h,flagh = get_halt(rotated, cent)  #Get H considering the LAA center in the base (Clipped), morphologicFunctions.py


    start = np.add(cent, intersect[0])
    p0 = np.array([start[0] / 2, start[1] / 2, start[2] / 2])  #punto medio entre centro ostium y techo

    h_theta, thpoint, M = get_hthetaalt(rotated, p0, h, cent, intersect[0]) 

       # #  # Show measures, descomenta este
    if viz == True:
        actor_h = addpoint_show(intersect[0], color=[0.0, 0.0, 0.0]) #punto en techo
        actor_oh = addpoint_show(origin_h, color=[0.0, 0.0, 0.0]) #punto centro del ostium
        actor_lh = addline_show(intersect[0],origin_h, color=[0.0, 0.0, 0.0]) 
        actor_p = addpoint_show(p0, color=[1.0, 0.0, 0.0])#punto cruce vectores
        actor_h2 = addpoint_show(thpoint, color=[0.0, 0.0, 0.0]) #punto endpoint
        actor_lh2 = addline_show(p0,thpoint, color=[0.0, 0.0, 0.0]) 


        renderer_h = vtk.vtkRenderer()
        vtk_show(renderer_h, rotated, 1000, 1000, actor_lh,actor_h, actor_oh, actor_p, actor_h2, actor_lh2)

    return h, h_theta, M
    








