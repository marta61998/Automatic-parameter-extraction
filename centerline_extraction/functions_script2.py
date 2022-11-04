
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

def vtk_show(renderer, mesh, heigth, width, actor1=None, actor2=None, actor3=None, actor4=None, actor5=None, actor6=None, actor7=None):
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
    if actor7 is not None:
        renderer.AddActor(actor7)

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
    actor_c.GetProperty().SetOpacity(1)
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

def extractclosestregion_edges_m2(polydata,ostium_centre):
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
        
        l = region_polydata.GetNumberOfPoints()
        region_array = []
        x = 0
        y = 0
        z = 0
        
        idList = vtk.vtkIdList()
        
        cont = 0
        while(region_polydata.GetLines().GetNextCell(idList)):
            for pointId in range(idList.GetNumberOfIds()):
                point = region_polydata.GetPoint(idList.GetId(pointId))
                region_array.append(point)
                x = point[0] + x
                y = point[1] + y
                z = point[2] + z
                cont = cont + 1
        
        #print('idlistedges',idList.GetNumberOfIds())
        if(l > 0 and idList.GetNumberOfIds() > 0 ):
            center = [x/cont, y/cont,z/cont]
        else:
            center = [0,0,0]
        
        regions.append(region_array)
        regions_center.append(center)
 
 
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
        #if case == '214':
        #       id_max = np.argmin(list_area)
        laa_edge = list_delaunay[id_max]
        area = list_area[id_max]
        poly = list_poly[id_max]

        
        return poly, area, laa_edge, LAA_error
    else:
        
            delaunay = vtk.vtkDelaunay2D()
            delaunay.SetInputData(connect.GetOutput())
            delaunay.Update()  #quitar
                            
            polygonProperties = vtk.vtkMassProperties()
            polygonProperties.SetInputConnection(delaunay.GetOutputPort())
            polygonProperties.Update()

    return connect.GetOutput(), polygonProperties.GetSurfaceArea(), delaunay.GetOutput(), LAA_error


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

def get_d1d2alt(LAA_edge, filename2, origin, show_points=False):
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

        #rend5 = vtk.vtkRenderer()
        #vtk_show_multi(rend5, LAA_edge, 1000, 1000, list_actors)
        #vtk_show(rend5, LAA_edge, 1000, 1000 )


    return min_distance, max_distance, mean_p, p0_d2, p0_d1, p1_d1, dist_min, points_d2_2,LAA_edge,p0, boundaryIds


def extractclosest_indexedregion_2(polydata,ostium_centre,flag=False):
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
    
        #try:
        edge_f = '/Users/martasaiz/Downloads/p.json'
        d2_min, d1_complete, mean_p, p0d2, p0d1, p1d1, d2_complete, p1d2,LAA_edge,p0,boundaryIds = get_d1d2alt(edges, edge_f,ostium_centre, show_points=False) 
        list_d1.append(d1_complete)
        center_region = estimate_center(edges)
        regions_center.append(center_region)
        actor_centro2  = addpoint_show(center_region, color=[0, 0, 1])
        list_centros.append(actor_centro2)
        distance = calculateDistance(center_region,ostium_centre)
        list_distance.append(distance)

        # except:
        #     print('ERROR in estimate_center, ignoring a region')
        #center_region = estimate_center(region)
        ##regions.append(region_array)
    
   
  
            #print('ERROR in estimate_center, ignoring a region')
        #center_region = estimate_center(region)
        ##regions.append(region_array) 
        
    largest_index = list_d1.index(max(list_d1))
    clip = extractindexedregion(polydata,largest_index)
    #print('marta la mejor')
    #rend2 = vtk.vtkRenderer()
    #vtk_show_multi(rend2, clip, 1000, 1000 )    

    return clip, list_d1,list_regions


def extractclosest_indexedregion_3(polydata,ostium_centre,flag=False):
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
    
        #try:
        edge_f = '/Users/martasaiz/Downloads/p.json'
        d2_min, d1_complete, mean_p, p0d2, p0d1, p1d1, d2_complete, p1d2,LAA_edge,p0,boundaryIds = get_d1d2alt(edges, edge_f,ostium_centre, show_points=False) 
        list_d1.append(d1_complete)
        center_region = estimate_center(edges)
        regions_center.append(center_region)
        actor_centro2  = addpoint_show(center_region, color=[0, 0, 1])
        list_centros.append(actor_centro2)
        distance = calculateDistance(center_region,ostium_centre)
        list_distance.append(distance)

        
    closest_index = list_distance.index(min(list_distance))
    clip = extractindexedregion(polydata,closest_index)
    #print('marta la mejor')
    #rend2 = vtk.vtkRenderer()
    #vtk_show_multi(rend2, clip, 1000, 1000 )    

    return clip, list_d1,list_regions

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
    clip, list_areas, list_regions = extractclosest_indexedregion_2(LAA_clip_n,origin,flag=False)

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
    edge_noDelaunay, surface_area, LAA_edge, LAA_error = extractclosestregion_edges_m2(edges,origin)
    cent = estimate_center(surface)

    return  LAA_edge, list_areas, list_regions



def brute_force_perturbation(pred_surface,normal, centroid, angle, count, no = 20):
        np.random.seed(1234)


        normal_proposals = np.random.uniform(-angle,angle,(no,3)) + normal
        normal_proposals = np.vstack((normal,normal_proposals))
        centroid_proposals = np.random.uniform(-1,1,(no,3)) + centroid
        centroid_proposals = np.vstack((centroid,centroid_proposals))
        min_area = np.inf
        
        for proposal in range(no):
            plane = vtk.vtkPlane()
            plane.SetOrigin(centroid_proposals[proposal,:])
            plane.SetNormal(normal_proposals[proposal,:])
            
            cutter = vtk.vtkCutter()
            cutter.SetInputData(pred_surface)
            cutter.SetCutFunction(plane)
            cutter.GenerateCutScalarsOff()
            
            stripper = vtk.vtkStripper()
            stripper.SetInputConnection(cutter.GetOutputPort())
            stripper.Update()
           
            
            if stripper.GetOutput().GetNumberOfCells()>1: 

                # Take care of cuts that consists of several segments
                MaxCutNumber = 0
                area = 0
                minDist = np.inf
                minIDX = -1
                minIDXarea = 0
                
                for i in range(stripper.GetOutput().GetNumberOfCells()):
                   
                    #print(stripper.GetOutput().GetCell(i).GetPoints())
                    pd = stripper.GetOutput().GetCell(i).GetPoints()
                    CM = estimate_center(pd)
                    d2 = vtk.vtkMath().Distance2BetweenPoints(CM,centroid)
                  
                    
                    if d2 < minDist: 
                        minDist = d2
                        minIDX = i
            
                # Final cut
                finalCut = vtk.vtkPolyData()
                finalCut.DeepCopy(stripper.GetOutput())
                final_line = finalCut.GetCell(minIDX)
                cells = vtk.vtkCellArray()
                cells.InsertNextCell(final_line)
                points = final_line.GetPoints()
                pd_finalCut = vtk.vtkPolyData()
                pd_finalCut.SetPoints(points)
                pd_finalCut.SetLines(cells)
                
            else:
                finalCut = vtk.vtkPolyData()
                finalCut.DeepCopy(stripper.GetOutput())
                final_line = finalCut.GetCell(0)
                cells = vtk.vtkCellArray()
                cells.InsertNextCell(final_line)
                points = final_line.GetPoints()
                pd_finalCut = vtk.vtkPolyData()
                pd_finalCut.SetPoints(points)
                pd_finalCut.SetLines(cells)
        
      
            #cut_area = polygonArea(pd_finalCut)
            #renderer_d1 = vtk.vtkRenderer()
            #vtk_show_centreline(renderer_d1,pred_surface, pd_finalCut, 1000, 1000)
    
            LAA_edge, list_areas, list_regions = return_edge(pred_surface,centroid_proposals[proposal,:],normal_proposals[proposal,:])          
            #renderer_d1 = vtk.vtkRenderer()
            #vtk_show_centreline(renderer_d1,pred_surface, LAA_edge, 1000, 1000)

        #rend1 = vtk.vtkRenderer()
        #vtk_show(rend1, real_clip, 1000, 1000)
            edge_f='/Users/martasaiz/Downloads/p.json'
            d2_min, d1_complete, mean_p, p0d2, p0d1, p1d1, d2_complete, p1d2,LAA_edge,p0,boundaryIds = get_d1d2alt(LAA_edge, edge_f,centroid_proposals[proposal,:], show_points=False) 
          

            if d1_complete < min_area and d1_complete>10:
                min_area = d1_complete
                normal_opt = normal_proposals[proposal,:]
                centroid_opt = centroid_proposals[proposal,:]
                edge = LAA_edge
                listi = list_areas
                listi2 = list_regions
            

     
        print("Min Area:", min_area)
        th = 40

        # if case_num2 == 176 or case_num2 == 214 or case_num2 == 29 or case_num2 == 7 or case_num2 == 182:
        #     th = 30

        # if case_num2 == 163:
        #      th = 15
        
         
        #if case == '165':
        #    th = 45
       
        if min_area>th: 
            count = count + 1
           
            print("No good estimate found - retrying")
            if count == 1:
                normal_opt, centroid_opt = brute_force_perturbation(pred_surface,normal, centroid , 0.6, count,  no = 20)
            if count == 2:
                normal_opt, centroid_opt = brute_force_perturbation(pred_surface,normal, centroid , 1.0, count, no = 100)
            #elif count == 3:
            #    break

        return normal_opt, centroid_opt


def reconstruct_LAA(clip,LAA_clip_op_n,origin):

    print('Reconstructing the LAA...')
    clip_op = extractclosest_indexedregion(LAA_clip_op_n,origin,flag=True)
    appendm = vtk.vtkAppendPolyData()
    appendm.AddInputData(clip)
    appendm.AddInputData(clip_op)
    appendm.Update()   
    clip = appendm.GetOutput() 

    return clip



def clip_LAA(file,origin,normal, viz = False):
    
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

    #if case_num2 == 163:
    #    clip = extractclosest_indexedregion(LAA_clip_op_n,origin,flag=False) #flag is False so by closest index
    #    edges = extractboundaryedge(clip)
       
    clip = extractclosest_indexedregion(LAA_clip_n,origin,flag=False)
    #edges1 = extractboundaryedge(LAA_clip_n) 
    edges = extractboundaryedge(clip)

    #clip_ori = clip
    #forma1
    npoints = edges.GetNumberOfPoints()
    edge_noDelaunay, surface_area, LAA_edge, LAA_error = extractclosestregion_edges_m2(edges,origin)
    cent = estimate_center(file)
    
    edges = extractboundaryedge(LAA_edge)
    npoints = edges.GetNumberOfPoints()
    p0 = []
    #p0_ids = []
    for i in range(npoints):
        p0.append(edges.GetPoint(i))

    print('forma1-new')
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

def fix_mesh(path_clip_bueno_stl): ##puedo intentar hacerlo 2 veces si no funciona bien
    tin = _meshfix.PyTMesh()
    tin.load_file(path_clip_bueno_stl)
    # Clean Mesh
    print("Cleaning mesh...")
    tin.select_intersecting_triangles()
    #tin.join_closest_components()
    tin.fill_small_boundaries()
    tin.clean(max_iters=100, inner_loops=2)
    tin.save_file(path_clip_bueno_stl)
    repaired = readstl(path_clip_bueno_stl)
    

    #segunda vez?
    tin = _meshfix.PyTMesh()
    tin.load_file(path_clip_bueno_stl)
    
#    # Clean Mesh
    print("Cleaning mesh...")
    tin.select_intersecting_triangles()
    #tin.join_closest_components()
    tin.fill_small_boundaries()
    tin.clean(max_iters=100, inner_loops=2)
    tin.save_file(path_clip_bueno_stl)
   


######### heat equation functions


def detectOstium(meshFile,list_os,normal,origin):
 vertices = meshFile.points

 #----mi codigo 

 #puntos parecidos
 nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(vertices)
 distances_2, indices_2 = nbrs.kneighbors(list_os)
           
 list_points = []
 for i in range(len(indices_2)):
    ver = indices_2[i]
    point = meshFile.points[ver]
    list_points.append(point[0])

 list_np = np.array(list_points)

 #-------------- coge el vértice con el que hace el plano mas grande (""""ostium""""")
 #point = vertices[index]
 point = origin
 #u = neighbors[0] - neighbors[1] #esto esta mal, esta cogiendo los ultimos neighbours de la lista
 #v = neighbors[2] - neighbors[1]

 #Cr = np.cross(u, v) #la normal del plano
 A = normal[0]
 B = normal[1]
 C = normal[2]

 #point = vertices[index]
 D = -(A * point[0] + B * point[1] + C * point[2])
 #esta definiendo la ecuacion del plano (la normal la coge mal) que pasa por el punto que le interesa, el que da mayor edge

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
  #no entiendo res, está calculando la distancia del plano al origen (d)
  #why? why d<3?
  if d < 3:#d < 0.15:
  
   ostium[sum_ostium, :] = vertices[i, :] #solo te guardarás el vertice que tenga (siempre el mismo plano), a una dist. < 3
   sum_ostium = sum_ostium + 1 #esto son contadores, para guardar valores
   sumUnreducedOstium = sumUnreducedOstium + 1

   x = x + vertices[i,0] #de que le sirve hacer esto si no lo vuelve a llamar!!!!!
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

 #AL FINAL: - ostium_array es el ostium edge mas grande (y de ahí se debería coger el más cercano al ostium_centre pero lo hace mal )
          # - LAA_wall representa los vertices que con el plano definido por la normal teóricamente con corte mas grande (la del ostium), está a un distancia del origen de d > 3
 print("Ostium centre:",x,y,z)

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