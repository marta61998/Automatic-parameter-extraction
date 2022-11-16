import vtk
import math as mt
import numpy as np
from vmtkcenterlines_2 import *
#from vtk.util import numpy_support


def readstl(filename):
    """Read VTK file"""
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

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

def extractboundaryedge(polydata):
    edge = vtk.vtkFeatureEdges()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        edge.SetInputData(polydata)
    else:
        edge.SetInput(polydata)
    edge.FeatureEdgesOff()
    edge.NonManifoldEdgesOff()
    edge.Update()
    return edge.GetOutput()


def rotate_angle(clip, filename2, normal1, normal2, mean_p, LAA_id):
    """
    Given the normal vectors of two planes find the rotation angle and rotate mesh
    :param filename1: Mesh path
    :param filename2: Rotated mesh path
    :param normal1: Normal from the plane to rotate
    :param normal2: "Reference" normal
    :param mean_p: Center of the LAA
    :param LAA_id: Id to handle exception
    :return: Rotated mesh
    """
    #mesh = readvtk(filename1)
    mesh=clip

    dotprd = np.dot(normal1, normal2)
    #print "dot prod: ", dotprd
    normA = np.linalg.norm(normal1)
    normB = np.linalg.norm(normal2)

    angl = mt.acos(dotprd/(normA*normB))
    #print "Rotation angle in radians: \n", angl

    angle_degr = mt.degrees(angl)
    angl_dgr = (180 / mt.pi)*angl
    #print "Rotation angle in degrees: \n", angle_degr
    #print "Rotation 2:", angl_dgr

    direction = np.cross(normal1, normal2)
    #print "direction", direction

    toorigin = [0, 0, 0]
    toorigin[0] = -1 * mean_p[0]
    toorigin[1] = -1 * mean_p[1]
    toorigin[2] = -1 * mean_p[2]

    if normal1[2] > 0 and (direction[0] < 0 or direction[1] < 0 or direction[2] <= 0) and LAA_id != "476":

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
    #writevtk2(file_rotated, filename2)
    return angle_degr, direction, file_rotated, mp


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
    actor_c.GetProperty().SetOpacity(0.35)

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

def dot(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))

def normvector(v):
    return mt.sqrt(dot(v, v))

def normalizevector(v):
    norm = normvector(v)
    return [v[0] / norm,
            v[1] / norm,
            v[2] / norm]