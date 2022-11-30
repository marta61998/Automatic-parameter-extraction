# from __future__ import division
from random import random
import time
import math
import csv
import vtk
import numpy as np
import json
# from colormaps import *
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


# from scipy import linalg as la
# from sklearn.decomposition import PCA
# import sklearn.utils
# import matplotlib.pyplot as plt
# from matplotlib import mpl

def read_214(path):
    with open(path) as f:
        data = json.load(f)
    return data

####### Marta
def numpy_to_vtk_M(nparray, name):
    vtkarray = vtk.vtkDoubleArray()
    vtkarray.SetName(name)
    vtkarray.SetNumberOfTuples(len(nparray))
    for j in range(len(nparray)):
        vtkarray.SetTuple1(j, nparray[j])
    return vtkarray


def polar2cart(radius, angle, center):
    angle_rad = grad2rad(angle)
    return [radius * math.cos(angle_rad) + center[0], radius * math.sin(angle_rad) + center[1]]


def grad2rad(angle):
    radians = (angle * 2 * math.pi) / 360
    return radians


def resample_ray(polydata,
                 m):  # take a ray and taking first and last point interpolates m values in the middle (returns new polydata)
    # me interesa que el primero siempre sea el mismo para que la direccion del rayo se mantenga.
    # Uno de los puntos siempre es el mismo, centro de la vena. No veo esto... dejo igual de momento.
    twoPoints = vtk.vtkPoints()
    aux_poly = vtk.vtkPolyData()
    numPoints = polydata.GetNumberOfPoints()
    # create the 2 points
    coord0 = polydata.GetPoint(0)
    x, y, z = coord0[:3]
    twoPoints.InsertPoint(0, x, y, z)

    coord1 = polydata.GetPoint(numPoints - 1)  # last
    x, y, z = coord1[:3]
    twoPoints.InsertPoint(1, x, y, z)
    aux_poly.SetPoints(twoPoints)

    # create the line
    aPolyLine = vtk.vtkPolyLine()
    aPolyLine.GetPointIds().SetNumberOfIds(2)

    aPolyLine.GetPointIds().SetId(0, 0)
    aPolyLine.GetPointIds().SetId(1, 1)

    aux_poly.Allocate(1, 1)
    aux_poly.InsertNextCell(aPolyLine.GetCellType(), aPolyLine.GetPointIds())

    # resample
    sf = vtk.vtkSplineFilter()
    sf.SetInput(aux_poly)
    sf.SetNumberOfSubdivisions(m - 1)  # m-1 subdivisions, m points
    sf.Update()
    output_poly = sf.GetOutput()
    return output_poly


def sample_surface(polydata_points, image):
    # probe filter
    probe = vtk.vtkProbeFilter()
    probe.SetInput(polydata_points)
    probe.SetSource(image)
    probe.Update()
    return probe.GetOutput()


def read_image(inputFilename):
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(inputFilename)
    reader.Update()
    image = vtk.vtkImageData()
    image.DeepCopy(reader.GetOutput())
    return image


def get_smoothed_mesh(inputFilename):

    reader = vtk.vtkDataSetReader()
    reader.SetFileName(inputFilename)
    reader.Update()
    # cast to float to have intermediate values
    cast = vtk.vtkImageCast()
    cast.SetInput(reader.GetOutput())
    cast.SetOutputScalarTypeToFloat()
    cast.Update()
    # Gaussian Smoothing
    smooth = vtk.vtkImageGaussianSmooth()
    smooth.SetDimensionality(3)
    smooth.SetStandardDeviation(2, 2)
    # smooth.SetRadiusFactor(3)
    smooth.SetInput(cast.GetOutput())
    smooth.Update()
    # Marching Cubes
    mc = vtk.vtkImageMarchingCubes()
    mc.SetInput(smooth.GetOutput())
    mc.SetNumberOfContours(1)
    mc.SetValue(0, 0.5)
    mc.Update()
    # write the mesh
    cadena = inputFilename[0:-4] + "_smoothed_mc.vtk"  # remove '.vtk' from the name and add '_smoothed_mc.vtk'
    writevtk(mc.GetOutput(), cadena)
    return mc.GetOutput()


def get_wall_int_map2LA(wall, mri, mesh_laendo):
    # 	wall = filename of the wall's mask (lawall.vtk). NOT MESH.
    #    	mri = filename, i.e. "lgemri.vtk"
    #    	mesh_laendo = mesh of LA mask. Here we'll include the scalar array 'color_wall' with the voxel intensity in the wall.
    mesh_wall_mask = get_smoothed_mesh(wall)
    im = read_image(mri)
    mesh_wall_int = sample_surface(mesh_wall_mask, im)
    transfer_array(mesh_wall_int, mesh_laendo, 'scalars', 'color_wall')
    return mesh_laendo


def round_scar(mesh, arrayname):
    ref_array = mesh.GetPointData().GetArray(arrayname)
    n = ref_array.GetNumberOfTuples()
    for i in range(n):
        value = ref_array.GetValue(i)
        value2 = np.around(value, decimals=0)
        ref_array.SetValue(i, value2)
    return mesh


def cut_vein_quadrants(polydata, pv_label):  # pv_labels = ['lpv_sup','lpv_inf','rpv_sup','rpv_inf']
    flatlabels = getflatregionslabels()
    vein_quadrants = cellthreshold(polydata, 'sumlabels', flatlabels[pv_label + '_q1'], flatlabels[pv_label + '_q4'])
    return vein_quadrants


def scar_threshold(polydata, arrayname, start=0, end=1):  # very similar to pointthreshold (Cata)
    threshold = vtk.vtkThreshold()
    threshold.SetInput(polydata)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, arrayname)

    threshold.ThresholdBetween(start, end)
    threshold.Update()

    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInput(threshold.GetOutput())
    surfer.Update()
    return surfer.GetOutput()


def detect_edges(polydata):
    boundaryEdges = vtk.vtkFeatureEdges()
    boundaryEdges.SetInput(polydata)
    boundaryEdges.BoundaryEdgesOn()
    boundaryEdges.ManifoldEdgesOff()
    boundaryEdges.FeatureEdgesOff()
    boundaryEdges.Update()
    return boundaryEdges.GetOutput()


def get_connected_edges(polydata):
    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInputData(polydata)
    connect.SetExtractionModeToAllRegions()  # extract all regions
    connect.ColorRegionsOn()  # creates scalar array RegionId differenciating between different contours
    connect.Update()
    return connect


def scar_segmentation_disk(polydata, inarrayname, outarrayname,
                           th):  # lee array inarrayname ('scalars'), aplica el threshold que sea y guarda resultado en outarrayname, binario (0-> healthy,  1-> scar)
    ori = vtk_to_numpy(polydata.GetPointData().GetArray(inarrayname))
    # create new array
    npoints = polydata.GetNumberOfPoints()
    newarray = vtk.vtkDoubleArray()
    newarray.SetName(outarrayname)
    newarray.SetNumberOfTuples(npoints)
    polydata.GetPointData().AddArray(newarray)
    for i in range(len(ori)):
        value = ori[i]
        if value > th:
            newarray.SetValue(i, 1)
        else:
            newarray.SetValue(i, 0)
    polydata.Update()
    return polydata


def scar_percentage_SUM(disk):
    scar_per = np.ones(24)
    for i in range(24):
        region = cellthreshold(disk, 'sumlabels', i + 1, i + 1)
        scar_array = vtk_to_numpy(region.GetPointData().GetArray('scar'))
        scar_per.put(i, (sum(scar_array) / len(scar_array)) * 100)
    return scar_per


def read_unstructured_grid(filename):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def convert_to_poly(unstructured_grid):
    # convert unstructured grid to polydata
    geometryFilter = vtk.vtkGeometryFilter()
    geometryFilter.SetInputData(unstructured_grid)
    geometryFilter.Update()
    return geometryFilter.GetOutput()


def create_mesh_with_manual_MV_clip(in_clipped_mesh_filename):
    # Read .ply mesh with the PVs already clipped and with the MV manually colored in red.
    # It creates the file '_clipped_mitral.vtk' needed for the registration to the template
    # example:
    # input: 420_mip_08P_segmented_scar_clipped_meshlab.ply <-- 420_mip_08P_segmented_scar_clipped.vtk file with the MV manually colored in red
    # output: 420_mip_08P_segmented_scar_clipped_mitral.vtk   <-- the file used in SUM_run_currents (it must be with this name)

    # read _clipped_meshlab.ply
    # convert to .vtk
    filename2 = in_clipped_mesh_filename[0:len(in_clipped_mesh_filename) - 4] + '.vtk'
    ply2vtk(in_clipped_mesh_filename, filename2)
    clipped_mesh = readvtk(filename2)
    # read RGB array convert to 'mitral' array
    matrix = vtk_to_numpy(clipped_mesh.GetPointData().GetArray('RGB'))
    color_array = np.zeros([1, clipped_mesh.GetNumberOfPoints()])  # zeros in the remaining parts of the mesh
    color1 = [255, 0, 0]  # I colored in red the MV
    npoints = clipped_mesh.GetNumberOfPoints()
    for i in range(npoints):
        mline = matrix[i, :]
        if (mline[0] == color1[0] and mline[1] == color1[1] and mline[2] == color1[2]):
            color_array[0, i] = 1

    color_vtkarray = numpy_to_vtk(color_array[0])
    color_vtkarray.SetName('mitral')
    # color_vtkarray.SetNumberOfTuples(clipped_mesh.GetNumberOfPoints())
    clipped_mesh.GetPointData().AddArray(color_vtkarray)

    # threshold (get unstructured)
    unstruct_mesh = pointthreshold(clipped_mesh, 'mitral', 0, 0)
    # convert to poly
    final_mesh = convert_to_poly(unstruct_mesh)

    # Transfer 'autolabels' from _clipped
    ori_name = in_clipped_mesh_filename[0:len(in_clipped_mesh_filename) - 12] + '.vtk'
    # example: '/home/marta/Desktop/DATA/Henry_KCL/420_mip_08P/SUM_with_preprocess/420_mip_08P_segmented_scar_clipped.vtk'
    ori = readvtk(ori_name)
    transfer_array(ori, final_mesh, 'autolabels', 'autolabels')

    # save 420_mip_08P_segmented_scar_clipped_mitral.vtk
    out_clipped_mesh_filename = ori_name[0:len(ori_name) - 4] + '_mitral.vtk'
    writevtk(final_mesh, out_clipped_mesh_filename)


def vtk2ply(filename1, filename2):
    """Read a vtk file and save as ply."""
    m = readvtk(filename1)
    writeply(m, filename2)


###########################################

# ------------------------------------------------------------------------------
# Linear algebra
# ------------------------------------------------------------------------------

def angle(v1, v2):
    return math.acos(dot(v1, v2) / (normvector(v1) * normvector(v2)))


def acumvectors(point1, point2):
    return [point1[0] + point2[0],
            point1[1] + point2[1],
            point1[2] + point2[2]]


def subtractvectors(point1, point2):
    return [point1[0] - point2[0],
            point1[1] - point2[1],
            point1[2] - point2[2]]


def dividevector(point, n):
    nr = float(n)
    return [point[0] / nr, point[1] / nr, point[2] / nr]


def multiplyvector(point, n):
    nr = float(n)
    return [nr * point[0], nr * point[1], nr * point[2]]


def sumvectors(vect1, scalar, vect2):
    return [vect1[0] + scalar * vect2[0],
            vect1[1] + scalar * vect2[1],
            vect1[2] + scalar * vect2[2]]


def cross(v1, v2):
    return [v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]]


def dot(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def euclideandistance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 +
                     (point1[1] - point2[1]) ** 2 +
                     (point1[2] - point2[2]) ** 2)


def normvector(v):
    return math.sqrt(dot(v, v))


def normalizevector(v):
    norm = normvector(v)
    return [v[0] / norm,
            v[1] / norm,
            v[2] / norm]


#def PCA_get_eigen(data, dims_rescaled_data=2):
    """
    returns: first n eigenvectors
    pass in: data as 2D NumPy array
    """

   # mn = np.mean(data, axis=0)
    # mean center the data
   # data -= mn
    # calculate the covariance matrix
   # C = np.cov(data.T)
    # calculate eigenvectors & eigenvalues of the covariance matrix
   # evals, evecs = la.eig(C)
    # sorted them by eigenvalue in decreasing order
   # idx = np.argsort(evals)[::-1]
   # evecs = evecs[:, idx]
   # evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
   # evecs = evecs[:, :dims_rescaled_data]

   #
   #return evecs


#def PCA_sklearn(data, n_components=2):
#    pca = PCA(n_components)
#    pca.fit(data)
#    PCA(copy=True, n_components=2, whiten=False)

#    return pca.components_, pca.explained_variance_ratio_


# ------------------------------------------------------------------------------
# VTK
# ------------------------------------------------------------------------------

def transfer_array(ref, target, arrayname, targetarrayname):
    # initiate point locator
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(ref)
    locator.BuildLocator()

    # get array from reference
    refarray = ref.GetPointData().GetArray(arrayname)

    # create new array
    numberofpoints = target.GetNumberOfPoints()
    newarray = vtk.vtkDoubleArray()
    newarray.SetName(targetarrayname)
    newarray.SetNumberOfTuples(numberofpoints)
    target.GetPointData().AddArray(newarray)

    # go through each point of target surface, determine closest point on surface,
    for i in range(target.GetNumberOfPoints()):
        point = target.GetPoint(i)
        closestpoint_id = locator.FindClosestPoint(point)
        value = refarray.GetValue(closestpoint_id)
        newarray.SetValue(i, value)
    return target


def transfer_labels(surface, ref, arrayname, value):
    # initiate point locator
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()

    # get array from surface
    array = surface.GetPointData().GetArray(arrayname)

    # go through each point of ref surface, determine closest point on surface,

    for i in range(ref.GetNumberOfPoints()):
        point = ref.GetPoint(i)
        closestpoint_id = locator.FindClosestPoint(point)
        array.SetValue(closestpoint_id, value)

    return surface


def transfer_cell_labels(surface, ref, arrayname, value):
    # get array from surface
    array = surface.GetCellData().GetArray(arrayname)

    locator = vtk.vtkCellLocator()
    genericcell = vtk.vtkGenericCell()
    cellid = vtk.mutable(0)
    point = [0., 0., 0.]
    closestpoint = [0., 0., 0.]
    pcoords = [0., 0., 0.]
    subid = vtk.mutable(0)
    distance2 = vtk.mutable(0)
    w = vtk.mutable(0)

    # build locator
    locator.SetDataSet(surface)
    locator.BuildLocator()

    # cell centers
    cellcenter = vtk.vtkCellCenters()
    cellcenter.SetInput(ref)
    cellcenter.VertexCellsOn()
    cellcenter.Update()

    nelem = cellcenter.GetOutput().GetNumberOfPoints()

    # transfer value of each cell
    for i in range(nelem):
        point = cellcenter.GetOutput().GetPoint(i)
        locator.FindClosestPoint(point, closestpoint, genericcell,
                                 cellid, subid, distance2)
        array.SetValue(cellid, value)

    return surface


def pointset_centreofmass(polydata):
    centre = [0, 0, 0]
    for i in range(polydata.GetNumberOfPoints()):
        point = [polydata.GetPoints().GetPoint(i)[0],
                 polydata.GetPoints().GetPoint(i)[1],
                 polydata.GetPoints().GetPoint(i)[2]]
        centre = acumvectors(centre, point)
    return dividevector(centre, polydata.GetNumberOfPoints())


def pointset_normal(polydata):
    # estimate the normal for a given set of points which are supposedly in a plane
    com = pointset_centreofmass(polydata)
    normal = [0, 0, 0]
    n = 0
    for i in range(polydata.GetNumberOfPoints()):
        point = [polydata.GetPoints().GetPoint(i)[0],
                 polydata.GetPoints().GetPoint(i)[1],
                 polydata.GetPoints().GetPoint(i)[2]]
        if n == 0:
            vect = subtractvectors(point, com)
        else:
            vect2 = subtractvectors(point, com)
            crossprod = cross(vect, vect2)
            crossprod = normalizevector(crossprod)
            # make sure each normal is oriented coherently ...
            if n == 1:
                normal2 = crossprod
            else:
                if dot(crossprod, normal2) < 0:
                    crossprod = [-crossprod[0], -crossprod[1], -crossprod[2]]
            normal = acumvectors(normal, crossprod)
        n += 1
    return normalizevector(normal)


def pointset_projectplane(com, normal, polydata):
    n = polydata.GetNumberOfPoints()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    polygon = vtk.vtkPolyData()
    points.SetNumberOfPoints(n)
    lines.InsertNextCell(n + 1)
    polygon.SetPoints(points)
    polygon.SetLines(lines)
    maxleng = 0.0
    for i in range(n):
        point = [polydata.GetPoints().GetPoint(i)[0],
                 polydata.GetPoints().GetPoint(i)[1],
                 polydata.GetPoints().GetPoint(i)[2]]
        vect = subtractvectors(point, com)
        leng = dot(vect, normal)
        if abs(leng) > maxleng:
            maxleng = leng
        point = sumvectors(point, -leng, normal)
        points.SetPoint(i, point)
        lines.InsertCellPoint(i)
    lines.InsertCellPoint(0)
    return polygon, maxleng, points


def find_globalid(subpoly, polydata):
    ids = []
    for i in range(subpoly.GetNumberOfPoints()):
        point = [subpoly.GetPoints().GetPoint(i)[0],
                 subpoly.GetPoints().GetPoint(i)[1],
                 subpoly.GetPoints().GetPoint(i)[2]]
        ids.append(int(polydata.FindPoint(point)))
    return ids


def modify_pointdata(polydata, ids, points):
    n = 0
    for i in ids:
        point = [points.GetPoint(n)[0], points.GetPoint(n)[1], points.GetPoint(n)[2]]
        polydata.GetPoints().SetPoint(i, point)
        n += 1
    polydata.Modified()
    return polydata


def remove_point_array(polydata, arrayname):
    polydata.GetPointData().RemoveArray(arrayname)
    return polydata


def remove_cell_array(polydata, arrayname):
    polydata.GetCellData().RemoveArray(arrayname)
    return polydata


def flatten_boundaryedges(polydata, boundaryedges, steplength):
    # ultimate laziness function! takes a boundary ring and projects them onto a plane
    com = pointset_centreofmass(boundaryedges)
    normal = pointset_normal(boundaryedges)
    com = sumvectors(com, steplength, normal)
    shifted, step_dist, projected_points = pointset_projectplane(com, normal, boundaryedges)
    ids = find_globalid(boundaryedges, polydata)
    polydata = modify_pointdata(polydata, ids, projected_points)
    return polydata


def find_neighbours(pt, pt_set, r):
    """ Function that returns the indeces of the neghbouring points of pt from pt_set that are within a
    sphere of radius r """
    a = np.repeat(pt, pt_set.shape[0]).reshape(pt_set.T.shape).T
    return np.where(np.sum((pt_set - a) * (pt_set - a), axis=1) <= r * r)[0]


def regularise_points(pt_set):
    """ This function keeps points that are more than 10mm far apart one to each other """
    ioo = np.ones(pt_set.shape[0], dtype=int)

    for n, i in enumerate(ioo):
        if i:
            neigh = find_neighbours(pt_set[n, :], pt_set, 10)
            ioo[neigh] = 0
            ioo[n] = 1
    return pt_set[np.where(ioo == 1)[0], :], ioo


def delaunay2D(polydata):
    delny = vtk.vtkDelaunay2D()
    delny.SetInput(polydata)
    delny.SetTolerance(0.01)
    # delny.SetAlpha(1.0)
    # delny.BoundingTriangulationOff()
    delny.Update()
    geometry = vtk.vtkGeometryFilter()  # unstructured grid to polydata
    geometry.SetInput(delny.GetOutput())
    return geometry.GetOutput()


def delaunay3D(polydata):
    delny = vtk.vtkDelaunay3D()
    delny.SetInput(polydata)
    delny.SetTolerance(0.01)
    # delny.SetAlpha(1.0)
    # delny.BoundingTriangulationOff()
    delny.Update()
    geometry = vtk.vtkGeometryFilter()  # unstructured grid to polydata
    geometry.SetInput(delny.GetOutput())
    return geometry.GetOutput()


def smooth(polydata, iterations, factor):
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInput(polydata)
    smoother.SetNumberOfIterations(iterations)
    smoother.FeatureEdgeSmoothingOn()
    smoother.SetRelaxationFactor(factor)
    smoother.Update()
    return smoother.GetOutput()


def smooth_taubin(polydata, iterations=15, angle=120, passband=0.001):
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInput(polydata)
    smoother.SetNumberOfIterations(iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(angle)
    smoother.SetPassBand(passband)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return smoother.GetOutput()


def boolean_subtract(polydata1, polydata2):
    boolOp = vtk.vtkBooleanOperationPolyDataFilter()
    boolOp.SetOperation(vtk.VTK_DIFFERENCE)
    boolOp.SetInputConnection(0, polydata1)
    boolOp.SetInputConnection(1, polydata2)
    boolOp.Update()
    return boolOp.GetOutput()


def add_cell_array(polydata, name, value):
    # create array and add a label
    array = vtk.vtkDoubleArray()
    array.SetName(name)
    array.SetNumberOfTuples(polydata.GetNumberOfCells())
    polydata.GetCellData().AddArray(array)

    for i in range(polydata.GetNumberOfCells()):
        array.SetValue(i, value)

    return polydata


def add_point_array(polydata, name, value):
    # create array and add a label
    array = vtk.vtkDoubleArray()
    array.SetName(name)
    array.SetNumberOfTuples(polydata.GetNumberOfPoints())
    polydata.GetPointData().AddArray(array)

    for i in range(polydata.GetNumberOfPoints()):
        array.SetValue(i, value)

    return polydata


def append(polydata1, polydata2):
    appender = vtk.vtkAppendPolyData()
    appender.AddInput(polydata1)
    appender.AddInput(polydata2)
    appender.Update()
    return appender.GetOutput()


def areaweightedmean_scalarfield(polydata, scalarfield):
    surfacecell = point2cell(polydata)
    sumarea = 0
    scalar = 0
    sumscalar = 0
    for i in range(surfacecell.GetNumberOfCells()):
        area = surfacecell.GetCell(i).ComputeArea()
        scalarfieldarray = surfacecell.GetCellData().GetArray(scalarfield)
        scalar = scalarfieldarray.GetComponent(i, 0)
        sumarea += area
        sumscalar += area * scalar
    meanscalar = sumscalar / sumarea
    return meanscalar


def areaweightedmean_vectorfield(polydata, vectorfield):
    surfacecell = cellnormals(polydata)  # add normals as celldata
    sumarea = 0
    sumvector = [0] * 3
    for i in range(surfacecell.GetNumberOfCells()):
        vectorfieldarray = surfacecell.GetCellData().GetArray(vectorfield)
        vector = [vectorfieldarray.GetComponent(i, 0),
                  vectorfieldarray.GetComponent(i, 1),
                  vectorfieldarray.GetComponent(i, 2)]
        area = surfacecell.GetCell(i).ComputeArea()
        sumarea += area  # total area
        sumvector = [sumvector[0] + area * vector[0],  # weighted sum
                     sumvector[1] + area * vector[1],
                     sumvector[2] + area * vector[2]]
    meanvector = normalizevector([sumvector[0] / sumarea,  # weighted mean
                                  sumvector[1] / sumarea,
                                  sumvector[2] / sumarea])
    return meanvector


def cellnormals(polydata):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.ComputePointNormalsOff()
    normals.ComputeCellNormalsOn()
    normals.Update()
    return normals.GetOutput()


def celldatatopointdata(polydata):
    converter = vtk.vtkCellDataToPointData()
    converter.SetInputData(polydata)
    converter.Update()
    return converter.GetOutput()


def cellthreshold(polydata, arrayname, start=0, end=1):
    threshold = vtk.vtkThreshold()
    threshold.SetInput(polydata)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, arrayname)
    threshold.ThresholdBetween(start, end)
    threshold.Update()

    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInput(threshold.GetOutput())
    surfer.Update()
    return surfer.GetOutput()


def centerofmass(polydata):
    surface = cellnormals(polydata)  # add normals as celldata
    sumarea = 0
    sumpoint = [0] * 3
    # calculate for each cell the area and center of mass
    for i in range(surface.GetNumberOfCells()):
        area = surface.GetCell(i).ComputeArea()
        point0 = [surface.GetCell(i).GetPoints().GetPoint(0)[0],
                  surface.GetCell(i).GetPoints().GetPoint(0)[1],
                  surface.GetCell(i).GetPoints().GetPoint(0)[2]]
        point1 = [surface.GetCell(i).GetPoints().GetPoint(1)[0],
                  surface.GetCell(i).GetPoints().GetPoint(1)[1],
                  surface.GetCell(i).GetPoints().GetPoint(1)[2]]
        point2 = [surface.GetCell(i).GetPoints().GetPoint(2)[0],
                  surface.GetCell(i).GetPoints().GetPoint(2)[1],
                  surface.GetCell(i).GetPoints().GetPoint(2)[2]]
        point = [(point0[0] + point1[0] + point2[0]) / 3.0,
                 (point0[1] + point1[1] + point2[1]) / 3.0,
                 (point0[2] + point1[2] + point2[2]) / 3.0]
        sumarea += area  # total area
        sumpoint = [sumpoint[0] + area * point[0],
                    sumpoint[1] + area * point[1],
                    sumpoint[2] + area * point[2]]  # weighted sum
    meanpoint = [sumpoint[0] / sumarea,
                 sumpoint[1] / sumarea,
                 sumpoint[2] / sumarea]  # weighted mean
    return meanpoint


def centroidofcentroids(edges):
    # compute centroids of each edge
    # find average point
    acumvector = [0, 0, 0]
    rn = countregions(edges)
    # print "found",rn,'edges'
    for r in range(rn):
        oneedge = extractconnectedregion(edges, r)
        onecentroid = pointset_centreofmass(oneedge)
        acumvector = acumvectors(acumvector, onecentroid)
        # print acumvector
    finalcentroid = dividevector(acumvector, rn)
    return finalcentroid


def centroidofcentroidsrefdist(edges, refpoint, refdist=1000):
    # compute centroids of each edge
    # find average point
    acumvector = [0, 0, 0]
    rn = countregions(edges)
    rncnt = rn
    # print "found",rncnt,'edges'
    for r in range(rn):
        oneedge = extractconnectedregion(edges, r)
        onecentroid = pointset_centreofmass(oneedge)
        dist = euclideandistance(refpoint, onecentroid)
        print
        dist
        if dist < refdist:
            acumvector = acumvectors(acumvector, onecentroid)
        else:
            rncnt = rncnt - 1
            print
            "skiping this edge", dist, 'total edges', rncnt
            # print acumvector
    finalcentroid = dividevector(acumvector, rncnt)
    return finalcentroid


def cleanpolydata(polydata):
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInput(polydata)
    cleaner.Update()
    return cleaner.GetOutput()


def computecurvature(surface):
    # initialise filter
    curvaturesfilter = vtk.vtkCurvatures()
    curvaturesfilter.SetInput(surface)
    curvaturesfilter.SetCurvatureTypeToGaussian()
    curvaturesfilter.Update()

    return curvaturesfilter.GetOutput()


def computesphericity(surface):
    # initialise mass filter
    mass = vtk.vtkMassProperties()
    mass.SetInput(surface)
    mass.Update()
    # mass properties already calcualtes the NSI
    return mass.GetNormalizedShapeIndex()


def computeclippointnormal(clenabled, clids, factor):
    # compute clippoint
    numberofcl = len(clids)
    acumpoint = [0, 0, 0]
    clippoint = [0, 0, 0]
    currentpoint = [0, 0, 0]
    normpoint1 = [0, 0, 0]
    averageshpere = 0

    for j in clids:
        branch = cellthreshold(clenabled, 'CenterlineIds', j, j)
        sphere_array = clenabled.GetPointData().GetArray('MaximumInscribedSphereRadius')
        abscissas_array = clenabled.GetPointData().GetArray('Abscissas')
        # tractId = 0 is closest to seed
        tract = cellthreshold(branch, 'TractIds', 0, 0)
        # print tract
        startid = int(tract.GetNumberOfPoints() - 1)
        # print startid

        # find point 'factor times' inscribed sphere downstream
        currentid = startid
        currentabscissa = 0
        currentsphere = 0
        startsphere = sphere_array.GetValue(startid)
        diff = 0
        plotable_array = []

        for s in range(0, startid):
            averageshpere = averageshpere + sphere_array.GetValue(s)

        averageshpere = averageshpere / tract.GetNumberOfPoints()

        print
        "startsphere", sphere_array.GetValue(startid), "averageshpere", averageshpere
        # if startsphere is smaller than the average sphere, we are probably already inside the vein
        if startsphere > 1.3 * averageshpere:
            newfactor = startsphere / averageshpere
        else:
            newfactor = factor

        startabscissa = abscissas_array.GetValue(startid)
        targetsphere = newfactor * startsphere
        print
        "target", targetsphere, "averageshpere", averageshpere
        # print 'start',startabscissa,'current',currentabscissa,'sphere',startsphere
        while (currentabscissa <= targetsphere):
            currentid -= 1
            currentabscissa = startabscissa - abscissas_array.GetValue(currentid)
            print
            'start', startabscissa, 'current', currentabscissa, 'target', targetsphere

        currentpoint = tract.GetPoint(currentid + 1)
        acumpoint = acumvectors(acumpoint, currentpoint)
        currentpoint = tract.GetPoint(currentid - 20)  # point 2mm downstream
        normpoint1 = acumvectors(normpoint1, currentpoint)

    clippoint[0] = acumpoint[0] / numberofcl
    clippoint[1] = acumpoint[1] / numberofcl
    clippoint[2] = acumpoint[2] / numberofcl

    normpoint1[0] = normpoint1[0] / numberofcl
    normpoint1[1] = normpoint1[1] / numberofcl
    normpoint1[2] = normpoint1[2] / numberofcl

    # print startpoint,clippoint
    clipnormal = normalizevector([normpoint1[0] - clippoint[0],
                                  normpoint1[1] - clippoint[1],
                                  normpoint1[2] - clippoint[2]])
    return clippoint, clipnormal


def computelengthalongvector(polydata, refpoint, vector):
    # polydata should be a closed surface

    # intersect with line
    point1 = refpoint
    point2 = sumvectors(refpoint, 1000, vector)  # far away point
    intersectpoints = intersectwithline(polydata, point1, point2)
    furthestpoint1 = furthest_point_to_polydata(intersectpoints, refpoint)

    # intersect with line the other way
    point1 = refpoint
    point2 = sumvectors(refpoint, -1000, vector)  # far away point
    intersectpoints = intersectwithline(polydata, point1, point2)
    furthestpoint2 = furthest_point_to_polydata(intersectpoints, furthestpoint1)

    length = euclideandistance(furthestpoint1, furthestpoint2)
    return length


def countregions(polydata):
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
    connect.Update()
    return connect.GetNumberOfExtractedRegions()


def cutdataset(dataset, point, normal):
    cutplane = vtk.vtkPlane()
    cutplane.SetOrigin(point)
    cutplane.SetNormal(normal)
    cutter = vtk.vtkCutter()
    cutter.SetInputData(dataset)
    cutter.SetCutFunction(cutplane)
    cutter.Update()
    return cutter.GetOutput()


def cylinderclip(dataset, point0, point1, normal, radius):
    """Define cylinder. The cylinder is infinite in extent. We therefore have
    to truncate the cylinder using vtkImplicitBoolean in combination with
    2 clipping planes located at point0 and point1. The radius of the
    cylinder is set to be slightly larger than 'maxradius'."""

    rotationaxis = cross([0, 1, 0], normal)
    rotationangle = (180 / math.pi) * angle([0, 1, 0], normal)

    transform = vtk.vtkTransform()
    transform.Translate(point0)
    transform.RotateWXYZ(rotationangle, rotationaxis)
    transform.Inverse()

    cylinder = vtk.vtkCylinder()
    cylinder.SetRadius(radius)
    cylinder.SetTransform(transform)

    plane0 = vtk.vtkPlane()
    plane0.SetOrigin(point0)
    plane0.SetNormal([-x for x in normal])
    plane1 = vtk.vtkPlane()
    plane1.SetOrigin(point1)
    plane1.SetNormal(normal)

    clipfunction = vtk.vtkImplicitBoolean()
    clipfunction.SetOperationTypeToIntersection()
    clipfunction.AddFunction(cylinder)
    clipfunction.AddFunction(plane0)
    clipfunction.AddFunction(plane1)

    clipper = vtk.vtkClipPolyData()
    clipper.SetInput(dataset)
    clipper.SetClipFunction(clipfunction)
    clipper.Update()

    return extractlargestregion(clipper.GetOutput())


def clip_disk_with_circle(diskpolydata, normal, center, rad, scale, inside=0):
    # Translate and rotate the cylinder. By default, the cylinder is centered
    # at [0, 0, 0] and the axes of rotation is along the y-axis. Note: The
    # transformation transforms a point into the space of the implicit
    # function. To transform the implicit model into world coordinates, we
    # use the inverse of the transformation.
    rotationaxis = cross([0, 1, 0], normal)
    rotationangle = (180 / math.pi) * angle([0, 1, 0], normal)

    print
    rotationaxis, rotationangle
    transform = vtk.vtkTransform()
    transform.Translate(center)
    transform.RotateWXYZ(rotationangle, rotationaxis)
    transform.Scale(scale)
    transform.Inverse()

    # Define cylinder
    cylinder = vtk.vtkCylinder()
    cylinder.SetRadius(rad)
    cylinder.SetTransform(transform)
    clipfunction = vtk.vtkImplicitBoolean()
    clipfunction.SetOperationTypeToIntersection()
    clipfunction.AddFunction(cylinder)

    # Clip disk with cylinder. Also generate the clipped output ('inside').
    clipper = vtk.vtkClipPolyData()
    clipper.SetInput(diskpolydata)
    clipper.SetClipFunction(clipfunction)
    clipper.GenerateClippedOutputOn()
    clipper.Update()

    # Grab output
    if inside:
        output = clipper.GetClippedOutput()
    else:
        output = clipper.GetOutput()

    # ALWAYS AFTER CLIPING (clean and surf)
    outputcl = cleanpolydata(output)
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInput(outputcl)
    surfer.Update()

    return surfer.GetOutput()


def clip_disk_with_ellipse(diskpolydata, normal, center, rad, scale, inplaneangle, inside=0):
    # Translate and rotate the cylinder. By default, the cylinder is centered
    # at [0, 0, 0] and the axes of rotation is along the y-axis. Note: The
    # transformation transforms a point into the space of the implicit
    # function. To transform the implicit model into world coordinates, we
    # use the inverse of the transformation.
    rotationaxis = cross([0, 1, 0], normal)
    rotationangle = (180 / math.pi) * angle([0, 1, 0], normal)

    transform = vtk.vtkTransform()
    transform.Translate(center)
    transform.RotateWXYZ(rotationangle, rotationaxis)
    transform.RotateWXYZ(inplaneangle, [0, 1, 0])
    transform.Scale(scale)
    transform.Inverse()

    # Define cylinder
    cylinder = vtk.vtkCylinder()
    cylinder.SetRadius(rad)
    cylinder.SetTransform(transform)
    clipfunction = vtk.vtkImplicitBoolean()
    clipfunction.SetOperationTypeToIntersection()
    clipfunction.AddFunction(cylinder)

    # Clip disk with cylinder. Also generate the clipped output ('inside').
    clipper = vtk.vtkClipPolyData()
    clipper.SetInput(diskpolydata)
    clipper.SetClipFunction(clipfunction)
    clipper.GenerateClippedOutputOn()
    clipper.Update()

    # Grab output
    if inside:
        output = clipper.GetClippedOutput()
    else:
        output = clipper.GetOutput()

    # ALWAYS AFTER CLIPING (clean and surf)
    outputcl = cleanpolydata(output)
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInput(outputcl)
    surfer.Update()

    return surfer.GetOutput()


def clip_disk_with_box(diskpolydata, bounds, inside=0):
    # Define implicit box
    implicitCube = vtk.vtkBox()
    implicitCube.SetBounds(bounds)

    clipfunction = vtk.vtkImplicitBoolean()
    clipfunction.SetOperationTypeToIntersection()
    clipfunction.AddFunction(implicitCube)

    # Clip disk with line. Also generate the clipped output ('inside').
    clipper = vtk.vtkClipPolyData()
    clipper.SetInput(diskpolydata)
    clipper.SetClipFunction(clipfunction)
    clipper.GenerateClippedOutputOn()
    clipper.Update()

    # Grab output
    if inside:
        output = clipper.GetClippedOutput()
    else:
        output = clipper.GetOutput()

    # ALWAYS AFTER CLIPING (clean and surf)
    outputcl = cleanpolydata(output)
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInput(outputcl)
    surfer.Update()

    return surfer.GetOutput()


def clip_disk_with_box_rot(diskpolydata, lengthX, lengthY, center,
                           angle, inside=0):
    # rotate box?
    transform = vtk.vtkTransform()
    transform.Translate(center)
    transform.RotateWXYZ(angle, [0, 0, 1])
    transform.Inverse()

    # Define cube
    cube = vtk.vtkCubeSource()
    cube.SetXLength(lengthX)
    cube.SetYLength(lengthY)
    cube.SetZLength(lengthX)
    cube.Update()

    # Copy to implicit box
    implicitCube = vtk.vtkBox()
    implicitCube.SetBounds(cube.GetOutput().GetBounds())
    implicitCube.SetTransform(transform)

    clipfunction = vtk.vtkImplicitBoolean()
    clipfunction.SetOperationTypeToIntersection()
    clipfunction.AddFunction(implicitCube)

    # Clip disk with line. Also generate the clipped output ('inside').
    clipper = vtk.vtkClipPolyData()
    clipper.SetInput(diskpolydata)
    clipper.SetClipFunction(clipfunction)
    clipper.GenerateClippedOutputOn()
    clipper.Update()

    # Grab output
    if inside:
        output = clipper.GetClippedOutput()
    else:
        output = clipper.GetOutput()

    # ALWAYS AFTER CLIPING (clean and surf)
    outputcl = cleanpolydata(output)
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInput(outputcl)
    surfer.Update()

    return surfer.GetOutput()


def tooclose_to_polydata(polydata, point, refdist):
    # print "Is it too close?", polydata.GetNumberOfPoints()
    # check every point in polydata
    # if anypoint is below the reference distance --> flag true
    tooclose = 0
    for i in range(polydata.GetNumberOfPoints()):
        # print polydata.GetPoint(i)
        # print point
        dist = euclideandistance(polydata.GetPoint(i), point)
        # print dist
        if dist < refdist:
            tooclose = 1
    return tooclose


def furthest_point_to_polydata(pointset, refpoint):
    # visist each point in pointset
    # selecte point furthest from reference point
    refdist = 0
    for i in range(pointset.GetNumberOfPoints()):
        # print pointset.GetPoint(i)

        dist = euclideandistance(pointset.GetPoint(i), refpoint)
        # print dist
        if dist > refdist:
            refdist = dist
            selectedpointid = i
    return pointset.GetPoint(selectedpointid)


def closest_point_to_polydata(pointset, refpoint):
    # visist each point in pointset
    # selecte point closest from reference point
    refdist = 100000
    for i in range(pointset.GetNumberOfPoints()):
        # print pointset.GetPoint(i)

        dist = euclideandistance(pointset.GetPoint(i), refpoint)
        # print dist
        if dist < refdist:
            refdist = dist
            selectedpointid = i
    return pointset.GetPoint(selectedpointid)


def transfer_array_by_pointid(ref, target, arrayname, targetarrayname):
    # get array from reference
    refarray = ref.GetPointData().GetArray(arrayname)

    # create new array
    numberofpoints = target.GetNumberOfPoints()
    newarray = vtk.vtkDoubleArray()
    newarray.SetName(targetarrayname)
    newarray.SetNumberOfTuples(numberofpoints)

    target.GetPointData().AddArray(newarray)

    # go through each point of target surface,
    for i in range(target.GetNumberOfPoints()):
        value = refarray.GetValue(i)
        newarray.SetValue(i, value)

    return target


def distance_to_polydata(polydata, point):
    refdist = 0
    for i in range(polydata.GetNumberOfPoints()):
        # print polydata.GetPoint(i)
        # print point
        dist = euclideandistance(polydata.GetPoint(i), point)
        # print dist
        if dist > refdist:
            refdist = dist
    return refdist


def dice_metric(manual, auto):
    Aa = 0.0  # area automatic
    Am = 0.0  # area manual
    Aam = 0.0  # overlap

    # for manual
    numberofpixels = manual.GetNumberOfPoints()
    for i in range(numberofpixels):
        pixelm = manual.GetPointData().GetScalars().GetTuple1(i)
        if pixelm != 0:
            Am = Am + 1
        pixela = auto.GetPointData().GetScalars().GetTuple1(i)
        if pixela != 0:
            Aa = Aa + 1
        if (pixelm != 0) and (pixela != 0):
            Aam = Aam + 1
    # print Aam, Aa, Am
    if (Aa + Am) > 0:
        DM = (2.0 * Aam) / (Aa + Am)
    else:
        DM = 0
    print
    'Dice metric', DM
    return DM


def extractboundaryedge(polydata):
    edge = vtk.vtkFeatureEdges()
    edge.SetInputData(polydata)
    edge.FeatureEdgesOff()
    edge.NonManifoldEdgesOff()
    edge.Update()
    return edge.GetOutput()


def extractcells(polydata, idlist):
    """Extract cells from polydata whose cellid is in idlist."""
    cellids = vtk.vtkIdList()  # specify cellids
    cellids.Initialize()
    for i in idlist:
        cellids.InsertNextId(i)

    extract = vtk.vtkExtractCells()  # extract cells with specified cellids
    extract.SetInput(polydata)
    extract.AddCellList(cellids)
    extraction = extract.GetOutput()

    geometry = vtk.vtkGeometryFilter()  # unstructured grid to polydata
    geometry.SetInput(extraction)
    geometry.Update()
    return geometry.GetOutput()


def skippoints(polydata, nskippoints):
    """Generate a single cell line from points in idlist."""

    # derive number of nodes
    numberofnodes = polydata.GetNumberOfPoints() - nskippoints

    # define points and line
    points = vtk.vtkPoints()
    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(numberofnodes)

    # assign id and x,y,z coordinates
    for i in range(nskippoints, polydata.GetNumberOfPoints()):
        pointid = i - nskippoints
        polyline.GetPointIds().SetId(pointid, pointid)
        point = polydata.GetPoint(i)
        points.InsertNextPoint(point)

    # define cell
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyline)

    # add to polydata
    polyout = vtk.vtkPolyData()
    polyout.SetPoints(points)
    polyout.SetLines(cells)
    #polyout.Update()

    return polyout


def extractclosestpointregion(polydata, point=[0, 0, 0]):
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
    connect.SetExtractionModeToClosestPointRegion()
    connect.SetClosestPoint(point)
    connect.Update()
    return connect.GetOutput()


def extractconnectedregion(polydata, regionid):
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
    connect.ColorRegionsOn()
    connect.Update()
    surface = pointthreshold(connect.GetOutput(), 'RegionId', float(regionid), float(regionid))
    return surface


def extractlargestregion(polydata):
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
    connect.SetExtractionModeToLargestRegion()
    connect.Update()

    # leaves phantom points ....
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(connect.GetOutputPort())
    cleaner.Update()
    return cleaner.GetOutput()

def extractsmallestregion(polydata):
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
    connect.SetExtractionModeToSmallestRegion()
    connect.Update()

    # leaves phantom points ....
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(connect.GetOutputPort())
    cleaner.Update()
    return cleaner.GetOutput()

def extractlargestregion_edges(polydata):
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
    connect.SetExtractionModeToLargestRegion()
    connect.Update()

    # leaves phantom points ....
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(connect.GetOutputPort())
    cleaner.Update()

    delaunay = vtk.vtkDelaunay2D()
    delaunay.SetInputData(cleaner.GetOutput())
    delaunay.Update()  #quitar
                      
    polygonProperties = vtk.vtkMassProperties()
    polygonProperties.SetInputConnection(delaunay.GetOutputPort())
    polygonProperties.Update()
    

    return cleaner.GetOutput(), polygonProperties.GetSurfaceArea(), delaunay.GetOutput()

def extractindexedregion(polydata,index):
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
    
    reg=0

    
    if reg == 1:
        connect.AddSpecifiedRegion(0) 
        connect.SetExtractionModeToLargestRegion()
    else:
        connect.SetExtractionModeToSpecifiedRegions() 
        connect.Update()
        m = connect.GetNumberOfExtractedRegions()
        connect.AddSpecifiedRegion(index) #Manually increment from 0 up to filt.GetNumberOfExtractedRegions()
 
    connect.Update()

   

    # leaves phantom points ....
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(connect.GetOutputPort())
    cleaner.Update()
    return cleaner.GetOutput()





def discardsmallregions(polydata, th):
    numberofpossibleregions = countregions(polydata)
    largest = extractlargestregion(polydata)
    arealargest = surfacearea(largest)

    appender = vtk.vtkAppendPolyData()

    for r in range(0, numberofpossibleregions + 1):
        possibleregion = extractconnectedregion(polydata, r)
        possiblearea = surfacearea(possibleregion)
        if possiblearea > th * arealargest:
            appender.AddInput(possibleregion)
            appender.Update()

    return appender.GetOutput()


def extractnonregion(polydata, ref, refdist):
    numberofpossibleregions = countregions(polydata)

    # locator
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(ref)
    locator.BuildLocator()

    # appender
    appender = vtk.vtkAppendPolyData()

    # find the regionid closest to ref
    acumdist = 0
    mindist = 100000

    for r in range(numberofpossibleregions):
        acumdist = 0
        possibleregion = extractconnectedregion(polydata, r)
        for i in range(possibleregion.GetNumberOfPoints()):
            point0 = possibleregion.GetPoint(i)
            closestpoint_id = locator.FindClosestPoint(point0)
            point1 = ref.GetPoint(closestpoint_id)
            acumdist = acumdist + euclideandistance(point0, point1)
        dist = acumdist / possibleregion.GetNumberOfPoints()
        print
        "for region", r, "dist", dist
        if dist < mindist:
            mindist = dist
            selectedr = r
            print
            "selected region", selectedr, "distance", mindist

    # keep regions further than a certain distance

    for r in range(numberofpossibleregions):
        if r != selectedr:
            acumdist = 0
            possibleregion = extractconnectedregion(polydata, r)
            for i in range(possibleregion.GetNumberOfPoints()):
                point0 = possibleregion.GetPoint(i)
                closestpoint_id = locator.FindClosestPoint(point0)
                point1 = ref.GetPoint(closestpoint_id)
                acumdist = acumdist + euclideandistance(point0, point1)
            dist = acumdist / possibleregion.GetNumberOfPoints()
            print
            "for region", r, "dist", dist, refdist
            if dist > refdist:
                print
                "keeping region", r
                appender.AddInput(possibleregion)
                appender.Update()

    return appender.GetOutput()


def extrude(polydata, scale, flip=''):
    """Extrude surface along normal using scale. Flip on extrudes surface inwards."""
    polydatanormals = pointnormals(polydata, flip)

    extrusion = vtk.vtkLinearExtrusionFilter()
    extrusion.SetInput(polydatanormals)
    extrusion.SetExtrusionTypeToNormalExtrusion()
    extrusion.SetScaleFactor(scale)

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInput(extrusion.GetOutput())
    cleaner.Update()

    return cleaner.GetOutput()


def pointnormals(polydata, flip=''):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.SplittingOff()

    if flip:
        normals.FlipNormalsOn()

    normals.Update()
    return normals.GetOutput()


def extractsurface(polydata):
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputData(polydata)
    surfer.Update()

    return surfer.GetOutput()


def mindistancetopolydata(reference, polydata):
    """Compute minimum distance between two polydata."""
    refdist = 1000000

    # initiate point locator
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(reference)
    locator.BuildLocator()

    # go through each point of polydata
    for i in range(polydata.GetNumberOfPoints()):
        point = polydata.GetPoint(i)
        # determine closest point on target.
        closestpointid = locator.FindClosestPoint(point)
        dist = euclideandistance(point,
                                 reference.GetPoint(closestpointid))
        if dist < refdist:
            refdist = dist
    return refdist


def findadjoiningregionid(reference, target):
    """Find the regionid in target closest to any reference region."""
    nregions = getregionsrange(target)
    smallestdist = 1000000
    smallestdistr = 0

    print
    "target has", nregions
    # iterate over regions to find the adjoining regions
    if nregions > 0:
        for r in range(int(nregions[1]) + 1):
            print
            "checking region", r
            # taking centroid of region
            smallregion = extractconnectedregion(target, r)
            # find region closest to reference
            currentdist = mindistancetopolydata(reference, smallregion)
            print
            "currentdist", currentdist, "smallestdist", smallestdist
            if currentdist < smallestdist:
                smallestdist = currentdist
                smallestdistr = r
            print
            "smallest region", smallestdistr
    return smallestdistr


def findadjoiningarrayvalue(reference, target, arrayname):
    """Find the regionid in target closest to any reference region."""
    nregions = getregionsrange(target)
    smallestdist = 1000000
    smallestdistr = 0

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(reference)
    locator.BuildLocator()

    targetarray = target.GetPointData().GetArray(arrayname)
    refarray = reference.GetPointData().GetArray(arrayname)

    print
    "target has", nregions
    # iterate over regions to find the adjoining regions
    if nregions > 0:
        for r in range(int(nregions[1]) + 1):
            print
            "checking region", r
            # taking centroid of region
            smallregion = extractconnectedregion(target, r)
            # go through each point
            for p in range(smallregion.GetNumberOfPoints()):
                # find closest point
                point = smallregion.GetPoint(p)
                closestpoint_id = locator.FindClosestPoint(point)
                value = refarray.GetValue(closestpoint_id)
                targetarray.SetValue(p, value)

    return target


def findadjoiningcellregionid(reference, target, arrayname):
    """Find the array value in target closest to any reference region."""
    regions = target.GetCellData().GetArray(arrayname)
    nregions = regions.GetRange()[1]

    smallestdist = 1000000
    smallestdistr = 0

    print
    "target has", nregions
    # iterate over regions to find the adjoining regions
    if nregions > 0:
        for r in range(int(nregions) + 1):
            print
            "checking region", r
            # taking centroid of region
            smallregion = cellthreshold(target, arrayname, r, r)
            # find region closest to reference
            currentdist = mindistancetopolydata(reference, smallregion)
            print
            "currentdist", currentdist, "smallestdist", smallestdist
            if currentdist < smallestdist:
                smallestdist = currentdist
                smallestdistr = r
            print
            "smallest region", smallestdistr
    return smallestdistr


def findlargestregionid(polydata):
    # NOTE: preventive measures: clean before connectivity filter
    # to avoid artificial regionIds
    # It slices the surface down the middle
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInput(polydata)
    surfer.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInput(surfer.GetOutput())
    cleaner.Update()

    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInput(cleaner.GetOutput())
    connect.SetExtractionModeToAllRegions()
    connect.ColorRegionsOn()
    connect.Update()

    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInput(connect.GetOutput())
    surfer.Update()

    regions = surfer.GetOutput().GetPointData().GetArray('RegionId')
    regionsrange = regions.GetRange()
    maxpoints = 0
    largestregionid = regionsrange[0]
    # print regionsrange, largestregionid
    if (regionsrange[1] > 0.0):  # more than one region
        for j in range(int(regionsrange[0]), int(regionsrange[1]) + 1):
            outsurf = pointthreshold(surfer.GetOutput(), 'RegionId', j, j)
            numberofpoints = outsurf.GetNumberOfPoints()
            # print 'numberofpoints',numberofpoints, 'maxpoints',maxpoints
            if (numberofpoints > maxpoints):
                maxpoints = numberofpoints
                largestregionid = j
                # print "Largest region id", largestregionid
    return largestregionid


def findlargestarearegionid(polydata):
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
    connect.ColorRegionsOn()
    connect.Update()

    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputConnection(connect.GetOutputPort())
    surfer.Update()

    regions = surfer.GetOutput().GetPointData().GetArray('RegionId')
    regionsrange = regions.GetRange()
    maxarea = 0
    largestregionid = regionsrange[0]

    # print regionsrange, largestregionid
    if (regionsrange[1] > 0.0):  # more than one region
        for j in range(int(regionsrange[0]), int(regionsrange[1]) + 1):
            outsurf = pointthreshold(surfer.GetOutput(), 'RegionId', j, j)
            area = surfacearea(outsurf)
            # print 'numberofpoints',numberofpoints, 'maxpoints',maxpoints
            if (area > maxarea):
                maxarea = area
                largestregionid = j
                # print "Largest region id", largestregionid
    return largestregionid


def findlongestregionid(polydata):
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
    connect.ColorRegionsOn()
    connect.Update()

    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputConnection(connect.GetOutputPort())
    surfer.Update()

    regions = surfer.GetOutput().GetPointData().GetArray('RegionId')
    regionsrange = regions.GetRange()
    maxlength = 0
    largestregionid = regionsrange[0]

    # print regionsrange, largestregionid
    if (regionsrange[1] > 0.0):  # more than one region
        for j in range(int(regionsrange[0]), int(regionsrange[1]) + 1):
            outsurf = pointthreshold(surfer.GetOutput(), 'RegionId', j, j)
            length = outsurf.GetLength()
            # print 'numberofpoints',numberofpoints, 'maxpoints',maxpoints
            if (length > maxlength):
                maxlength = length
                largestregionid = j
                # print "Largest region id", largestregionid
    return largestregionid


def findkmeans(data, k):
    # init means and data to random values
    # use real data in your code
    means = [random() for i in range(k)]

    param = 0.01  # bigger numbers make the means change faster
    # must be between 0 and 1

    for x in data:
        closest_k = 0;
        smallest_error = 9999;  # this should really be positive infinity
        for k in enumerate(means):
            error = abs(x - k[1])
            if error < smallest_error:
                smallest_error = error
                closest_k = k[0]
            means[closest_k] = means[closest_k] * (1 - param) + x * (param)
    return means


def fillholes(polydata, size, id_LAA):
    filler = vtk.vtkFillHolesFilter()
    filler.SetInputData(polydata)
    filler.SetHoleSize(size)
    filler.Update()
    path = "/Users/alvaro/Documents/CBE_Master/Thesis/Morphologic_descriptors/LAA files_processed/Files_thrombus/" \
           "APP_original_smooth_vtk/Filled/"+ str(id_LAA) + "_filled.vtk"
    filled = filler.GetOutput()
    writevtk(filled, path)
    #return filler.GetOutput()


def getconnectedvertices(mesh, pointid):
    connectedVertices = vtk.vtkIdList()

    # get all cells that vertex 'id' is a part of
    cellIdList = vtk.vtkIdList()
    mesh.GetPointCells(pointid, cellIdList)

    for i in range(0, cellIdList.GetNumberOfIds()):
        pointIdList = vtk.vtkIdList()
        mesh.GetCellPoints(cellIdList.GetId(i), pointIdList)

    print
    "End points are ", pointIdList.GetId(0), " and ", pointIdList.GetId(1)

    if pointIdList.GetId(0) != pointid:
        print
        "Connected to ", pointIdList.GetId(0)
        connectedVertices.InsertNextId(pointIdList.GetId(0))
    else:
        print
        "Connected to ", pointIdList.GetId(1)
        connectedVertices.InsertNextId(pointIdList.GetId(1))

    return connectedVertices


def getregionsrange(polydata):
    # to avoid artificial regionIds
    # It slices the surface down the middle
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInput(polydata)
    surfer.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInput(surfer.GetOutput())
    cleaner.Update()

    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInput(cleaner.GetOutput())
    connect.SetExtractionModeToAllRegions()
    connect.ColorRegionsOn()
    connect.Update()

    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInput(connect.GetOutput())
    surfer.Update()

    regions = surfer.GetOutput().GetPointData().GetArray('RegionId')
    regionsrange = regions.GetRange()
    print
    regionsrange
    return regionsrange


def getregionslabels():
    """Return dictionary linking regionids to anatomical locations."""
    regionslabels = {'body': 36,
                     'laa': 37,
                     'pv2': 76,
                     'pv1': 77,
                     'pv3': 78,
                     'pv4': 79}
    return regionslabels


def getflatregionslabels():
    """Return dictionary linking SUM regionids to anatomical locations."""
    regionslabels = {'ant': 1,
                     'lat': 2,
                     'lattop': 2,
                     'laa_around': 3,
                     'laa_bridge_left': 3,
                     'laa_bridge_right': 3,
                     'roof': 4,
                     'roof_addon': 4,
                     'post': 5,
                     'isthmus': 6,
                     'floor': 7,
                     'floor_addon': 7,
                     'septum': 8,
                     'lpv_sup_q1': 9,
                     'lpv_sup_q2': 10,
                     'lpv_sup_q3': 11,
                     'lpv_sup_q4': 12,
                     'lpv_inf_q1': 13,
                     'lpv_inf_q2': 14,
                     'lpv_inf_q3': 15,
                     'lpv_inf_q4': 16,
                     'rpv_sup_q1': 17,
                     'rpv_sup_q2': 18,
                     'rpv_sup_q3': 19,
                     'rpv_sup_q4': 20,
                     'rpv_inf_q1': 21,
                     'rpv_inf_q2': 22,
                     'rpv_inf_q3': 23,
                     'rpv_inf_q4': 24}

    return regionslabels


def gradient(surface, inputarray, outputarray):
    gradients = vtk.vtkGradientFilter()
    gradients.SetInput(surface)
    gradients.SetInputScalars(0, inputarray)
    gradients.SetResultArrayName(outputarray)
    gradients.Update()
    return gradients.GetOutput()


def generateglyph(polyIn, scalefactor=2):
    vertexGlyphFilter = vtk.vtkGlyph3D()
    sphereSource = vtk.vtkSphereSource()
    vertexGlyphFilter.SetSource(sphereSource.GetOutput())
    vertexGlyphFilter.SetInput(polyIn)
    vertexGlyphFilter.SetColorModeToColorByScalar()
    vertexGlyphFilter.SetSourceConnection(sphereSource.GetOutputPort())
    vertexGlyphFilter.ScalingOn()
    vertexGlyphFilter.SetScaleFactor(scalefactor)
    vertexGlyphFilter.Update()
    return vertexGlyphFilter.GetOutput()


def generateglypharrow(polyIn):
    vertexGlyphFilter = vtk.vtkGlyph3D()
    source = vtk.vtkArrowSource()
    vertexGlyphFilter.SetSource(source.GetOutput())
    vertexGlyphFilter.SetInput(polyIn)
    vertexGlyphFilter.SetColorModeToColorByScalar()
    vertexGlyphFilter.SetSourceConnection(source.GetOutputPort())
    vertexGlyphFilter.ScalingOn()
    vertexGlyphFilter.SetScaleFactor(2)
    vertexGlyphFilter.Update()
    return vertexGlyphFilter.GetOutput()


def glyph_to_point(seeds, labels, value):
    # extract seeds belonging to labels
    seeds = pointthreshold(seeds, labels, value, value)
    numberofpoints = seeds.GetNumberOfPoints()
    print
    numberofpoints
    acumcenter = [0, 0, 0]
    seed = [0, 0, 0]

    # opening from a "glyph" so seed is center of glyph (bad previous implementation)
    for j in range(numberofpoints):
        currentpoint = seeds.GetPoint(j)
        acumcenter = acumvectors(acumcenter, currentpoint)

    seed[0] = acumcenter[0] / numberofpoints
    seed[1] = acumcenter[1] / numberofpoints
    seed[2] = acumcenter[2] / numberofpoints

    return seed


def intersectwithline(surface, p1, p2):
    # Create the locator
    tree = vtk.vtkOBBTree()
    tree.SetDataSet(surface)
    tree.BuildLocator()

    intersectPoints = vtk.vtkPoints()
    intersectCells = vtk.vtkIdList()

    tolerance = 1.e-3
    tree.SetTolerance(tolerance)
    tree.IntersectWithLine(p1, p2, intersectPoints, intersectCells)

    return intersectPoints


def linesource(p1, p2):
    source = vtk.vtkLineSource()
    source.SetPoint1(p1[0], p1[1], p1[2])
    source.SetPoint2(p2[0], p2[1], p2[2])

    return source.GetOutput()


def planeclip(surface, point, normal, insideout=1):

    clipplane = vtk.vtkPlane()
    clipplane.SetOrigin(point)
    clipplane.SetNormal(normal)
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(surface)
    clipper.SetClipFunction(clipplane)

    if insideout == 1:
        # print 'insideout ON'
        clipper.InsideOutOn()
    else:
        # print 'insideout OFF'
        clipper.InsideOutOff()
    clipper.Update()
    return clipper.GetOutput()


def point2cell(polydata):
    interpolator = vtk.vtkPointDataToCellData()
    interpolator.SetInput(polydata)
    interpolator.Update()
    return interpolator.GetOutput()


def point2vertexglyph(point):
    points = vtk.vtkPoints()
    points.InsertNextPoint(point[0], point[1], point[2])

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)

    glyph = vtk.vtkVertexGlyphFilter()
    glyph.SetInputConnection(poly.GetProducerPort())
    glyph.Update()
    return glyph.GetOutput()


def pointthreshold(polydata, arrayname, start=0, end=1, alloff=0):
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(polydata)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, arrayname)
    threshold.ThresholdBetween(start, end)
    if (alloff):
        threshold.AllScalarsOff()
    threshold.Update()

    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputConnection(threshold.GetOutputPort())
    surfer.Update()
    return surfer.GetOutput()


def rotatepolydata(polydata, rotationangle, axis):
    surfacecenter = polydata.GetCenter()
    toorigin = [0, 0, 0]
    toorigin[0] = -1 * surfacecenter[0]
    toorigin[1] = -1 * surfacecenter[1]
    toorigin[2] = -1 * surfacecenter[2]
    print
    toorigin

    # bring to origin + rotate + bring back

    transform = vtk.vtkTransform()
    transform.PostMultiply()
    transform.Translate(toorigin)
    transform.RotateWXYZ(rotationangle, axis[0], axis[1], axis[2])
    transform.Translate(surfacecenter)

    transformfilter = vtk.vtkTransformFilter()
    transformfilter.SetTransform(transform)
    transformfilter.SetInputData(polydata)
    transformfilter.Update()

    return transformfilter.GetOutput()


def scalepolydata(polydata, scalefactor):
    surfacecenter = polydata.GetCenter()
    toorigin = [0, 0, 0]
    toorigin[0] = -1 * surfacecenter[0]
    toorigin[1] = -1 * surfacecenter[1]
    toorigin[2] = -1 * surfacecenter[2]
    print
    toorigin

    # bring to origin + rotate + bring back

    transform = vtk.vtkTransform()
    transform.PostMultiply()
    transform.Translate(toorigin)
    transform.Scale(scalefactor, scalefactor, scalefactor)
    transform.Translate(surfacecenter)

    transformfilter = vtk.vtkTransformFilter()
    transformfilter.SetTransform(transform)
    transformfilter.SetInput(polydata)
    transformfilter.Update()

    return transformfilter.GetOutput()


def sphereclip(polydata, center, radius, insideout):
    transform = vtk.vtkTransform()
    transform.Translate(center)
    transform.Inverse()

    cylinder = vtk.vtkSphere()
    cylinder.SetRadius(radius)
    cylinder.SetTransform(transform)
    clipfunction = vtk.vtkImplicitBoolean()
    clipfunction.SetOperationTypeToIntersection()
    clipfunction.AddFunction(cylinder)

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(polydata)
    clipper.SetClipFunction(clipfunction)

    if insideout == 1:
        # print 'insideout ON'
        clipper.InsideOutOn()
    else:
        # print 'insideout OFF'
        clipper.InsideOutOff()

    clipper.Update()
    return clipper.GetOutput()


def spheresource(center, radius):
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetCenter(center)
    sphere.Update()

    return sphere.GetOutput()


def surfacearea(polydata):
    properties = vtk.vtkMassProperties()
    properties.SetInput(polydata)
    properties.Update()
    return properties.GetSurfaceArea()


def surface2surfacedistance(ref, target, arrayname):
    """Compute distance between two surfaces. Output is added as point array."""
    # adapted from vtkvmtkSurfaceDistance.cxx
    # initialise
    locator = vtk.vtkCellLocator()
    genericcell = vtk.vtkGenericCell()
    cell = vtk.vtkCell()
    cellid = vtk.vtkIdType()
    point = [0., 0., 0.]
    closestpoint = [0., 0., 0.]
    subid = 0
    distance2 = 0

    # create array
    distarray = vtk.vtkDoubleArray()
    distarray.SetName(arrayname)
    distarray.SetNumberOfTuples(target.GetNumberOfPoints())
    target.GetPointData().AddArray(distarray)

    # build locator
    locator.SetDataSet(ref)
    locator.BuildLocator()

    # compute distance
    for i in range(target.GetNumberOfPoints()):
        point = target.GetPoint(i)
        locator.FindClosestPoint(point, closestpoint, genericcell, cellid,
                                 subid, distance2)
        distance = math.sqrt(distance2)
        # add value to array
        distarray.SetValue(i, distance)

    return target


def triangulate(polydata):
    trianglefilter = vtk.vtkTriangleFilter()
    trianglefilter.SetInput(polydata)
    trianglefilter.Update()
    return trianglefilter.GetOutput()


def transform_lmk(sourcepoints, targetpoints, surface, affineon=0):
    lmktransform = vtk.vtkLandmarkTransform()
    lmktransform.SetSourceLandmarks(sourcepoints)
    lmktransform.SetTargetLandmarks(targetpoints)
    if affineon == 1:
        print
        "affine"
        lmktransform.SetModeToAffine()
    else:
        print
        "similarity"
        lmktransform.SetModeToSimilarity()
    lmktransform.Update()

    transformfilter = vtk.vtkTransformPolyDataFilter()
    transformfilter.SetInput(surface)
    transformfilter.SetTransform(lmktransform)
    transformfilter.Update()

    return transformfilter.GetOutput()


def vectormagnitude(surface, inputarray, outputarray):
    calc = vtk.vtkArrayCalculator()
    calc.SetInput(surface)
    calc.AddVectorArrayName(inputarray, 0, 1, 2)
    calc.SetFunction('mag(%s)' % inputarray)
    calc.SetResultArrayName(outputarray)
    calc.Update()
    return calc.GetOutput()


def vectorofcentroids(edges, refpoint):
    """Compute average vector from the centroid of each edge to refpoint."""
    # compute vectors from of each edge
    acumvector = [0, 0, 0]
    rn = countregions(edges)
    for r in range(rn):
        oneedge = extractconnectedregion(edges, r)
        onecentroid = pointset_centreofmass(oneedge)
        onevector = subtractvectors(onecentroid, refpoint)
        acumvector = acumvectors(acumvector, onevector)
    # find average vector
    finalvector = normalizevector(acumvector)
    return finalvector


def visualise_slice_plus_mesh(imagefile, surface, point, size, position, focalpoint,
                              scale, arrayname, filename, thick=3, orientation=2, interact=1):
    """""Iamge is visualised in greay scale. Mesh is intersected by the image plane
    to generate contours. Mesh is colour mapped according to 'arrayname'."""

    # SLICE_ORIENTATION_YZ = 0, SLICE_ORIENTATION_XZ = 1, SLICE_ORIENTATION_XY = 2
    colors = getcolors('la_mesh')

    # colormap
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(255)
    lut.SetValueRange(0, 255)

    for c in range(len(colors)):
        this = colors[c]
        lut.SetTableValue(this[0], this[1], this[2], this[3])
    lut.Build()

    # local read image (need inputconnection)
    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(imagefile)
    reader.Update()

    # find image settings
    imagedata = reader.GetOutput()
    imagearray = vtk_to_numpy(imagedata.GetPointData().GetScalars())
    mini = np.min(imagearray)
    maxi = np.percentile(imagearray, 99.99)
    level = (maxi - mini) / 2
    window = maxi - mini

    spacing = imagedata.GetSpacing()
    origin = imagedata.GetOrigin()
    # finding plane
    pointsl = [math.ceil((point[0] - origin[0]) / spacing[0]),
               math.ceil((point[1] - origin[1]) / spacing[1]),
               math.ceil((point[2] - origin[2]) / spacing[2])]
    # image viewer
    rangi = [0., 0.]
    imageViewer = vtk.vtkImageViewer2()
    imageViewer.SetInputConnection(reader.GetOutputPort())
    imageViewer.SetSliceOrientation(orientation)
    imageViewer.SetSlice(int(pointsl[orientation]))
    imageViewer.SetColorWindow(window)  # width
    imageViewer.SetColorLevel(level)  # center
    # imageViewer.GetSliceRange(rangi)
    # print "range",rangi
    imageViewer.Render()
    # print "slice",imageViewer.GetSlice(),"should be",int(pointsl[orientation])

    # cutting surface
    # get countour of mesh at slice location
    normal = [0., 0., 0.]
    normal[orientation] = 1.
    # this position doubled checked by increasing/dcreasing in steps of spacing/2
    pointclip = [origin[0] + pointsl[0] * spacing[0],
                 origin[1] + pointsl[1] * spacing[1],
                 origin[2] + pointsl[2] * spacing[2]]

    print
    "cutting at", pointclip, normal
    surfacecut = cutdataset(surface, pointclip, normal)
    # surfacecut = planeclip(surface,pointclip,normal)
    surfacecutcl = round_labels_array(surfacecut, arrayname, [36, 37, 76, 77, 78, 79])

    # add actor
    surfacemapper = vtk.vtkPolyDataMapper()
    surfacemapper.SetInput(surfacecutcl)
    surfacemapper.SetScalarModeToUsePointFieldData()
    surfacemapper.SelectColorArray(arrayname)
    surfacemapper.SetLookupTable(lut)
    surfacemapper.SetScalarRange(0, 255)
    surfacemapper.InterpolateScalarsBeforeMappingOn()
    surfaceactor = vtk.vtkActor()
    surfaceactor.SetMapper(surfacemapper)

    surfaceactor.GetProperty().SetRepresentationToWireframe()
    surfaceactor.GetProperty().SetLineWidth(thick)

    imageViewer.GetRenderer().AddActor(surfaceactor)

    aCamera = vtk.vtkCamera()
    aCamera.SetPosition(position[0], position[1], position[2])
    aCamera.SetFocalPoint(focalpoint[0], focalpoint[1], focalpoint[2])
    aCamera.SetParallelScale(scale)

    if orientation == 1:
        aCamera.SetViewUp(0, 0, 1)  # z up?
    if orientation == 2:
        aCamera.SetViewUp(0, 1, 0)  # y up?
    aCamera.Zoom(3)

    # render interactor
    imageViewer.SetSize(size[0], size[1])
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    imageViewer.SetupInteractor(renderWindowInteractor)

    imageViewer.Render()
    # reset camera before modying position
    imageViewer.GetRenderer().ResetCamera()
    # imageViewer.GetRenderer().GetActiveCamera().Zoom(3)
    imageViewer.GetRenderer().SetActiveCamera(aCamera)

    # Remove existing lights and add lightkit lights
    imageViewer.GetRenderer().RemoveAllLights()
    lightkit = vtk.vtkLightKit()
    lightkit.AddLightsToRenderer(imageViewer.GetRenderer())

    # enable user interface interactor
    if interact == 1:
        imageViewer.Render()
        renderWindowInteractor.Start()
        outcam = imageViewer.GetRenderer().GetActiveCamera()

    else:
        # save as png
        ## Screenshot
        windowToImageFilter = vtk.vtkWindowToImageFilter()
        windowToImageFilter.SetInput(imageViewer.GetRenderWindow())
        windowToImageFilter.SetMagnification(
            1)  # set the resolution of the output image (3 times the current resolution of vtk render window)
        windowToImageFilter.SetInputBufferTypeToRGBA()  # also record the alpha (transparency) channel
        windowToImageFilter.Update()

        pngwriter = vtk.vtkPNGWriter()
        pngwriter.SetFileName(filename)
        pngwriter.SetInputConnection(windowToImageFilter.GetOutputPort())
        pngwriter.Write()


def round_labels_array(surface, arrayname, labels):
    """Any value that is not part of the labels is rounded to minvalue."""
    # threshold range step = 1
    minval = min(labels)
    maxval = max(labels)
    dif = np.zeros(len(labels))
    for val in range(minval, maxval + 1):
        mindif = 10000
        closestlabel = 0
        # print 'label',val
        patch = pointthreshold(surface, arrayname, val - 0.5, val + 0.5, 1)  # all off
        if patch.GetNumberOfPoints() > 0:
            for l in range(0, len(labels)):
                dif[l] = val - labels[l]
            mindif = min(abs(dif))
            # print 'mindif',mindif
            if (mindif > 0.01):
                # found points to round
                # print 'updating value in surface'
                transfer_labels(surface, patch, arrayname, minval)
    return surface


def visualise(surface, ref, case, arrayname, mini, maxi, viewup, position, focalpoint, clipping, scale, size,
              interact=1, filename='out.png'):
    """Visualise surface setting camera view."""
    # Create a lookup table to map cell data to colors
    # print "Colormap from ", mini, "to", maxi
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(255)
    lut.SetValueRange(0, 255)

    # qualitative data from colorbrewer  --> matching qualitative colormap of Paraview
    lut.SetTableValue(0, 0, 0, 0, 1)  # Black
    lut.SetTableValue(mini, 1, 1, 1, 1)  # white
    lut.SetTableValue(mini + 1, 77 / 255., 175 / 255., 74 / 255., 1)  # green
    lut.SetTableValue(maxi - 3, 152 / 255., 78 / 255., 163 / 255., 1)  # purple
    lut.SetTableValue(maxi - 2, 255 / 255., 127 / 255., 0., 1)  # orange
    lut.SetTableValue(maxi - 1, 55 / 255., 126 / 255., 184 / 255., 1)  # blue
    lut.SetTableValue(maxi, 166 / 255., 86 / 255., 40 / 255., 1)  # brown
    lut.Build()

    # create a text actor
    txt = vtk.vtkTextActor()
    txt.SetInput(case)
    txtprop = txt.GetTextProperty()
    txtprop.SetFontFamilyToArial()
    txtprop.SetFontSize(40)
    txtprop.SetColor(0, 0, 0)
    txt.SetDisplayPosition(20, 30)

    # create a rendering window, renderer, and renderwindowinteractor
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    # for GIMIAS interaction style
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)
    iren.SetRenderWindow(renWin)

    # surface mapper and actor
    surfacemapper = vtk.vtkPolyDataMapper()
    surfacemapper.SetInput(surface)
    surfacemapper.SetScalarModeToUsePointFieldData()
    surfacemapper.SelectColorArray(arrayname)
    surfacemapper.SetLookupTable(lut)
    surfacemapper.SetScalarRange(0, 255)
    surfaceactor = vtk.vtkActor()
    # surfaceactor.GetProperty().SetOpacity(0)
    # surfaceactor.GetProperty().SetColor(1, 1, 1)
    surfaceactor.SetMapper(surfacemapper)

    # refsurface mapper and actor
    refmapper = vtk.vtkPolyDataMapper()
    refmapper.SetInput(ref)
    refmapper.SetScalarVisibility(0)
    refactor = vtk.vtkActor()
    refactor.GetProperty().SetOpacity(0.4)
    refactor.GetProperty().SetColor(1, 1, 1)
    refactor.SetMapper(refmapper)

    # It is convenient to create an initial view of the data. The FocalPoint
    # and Position form a vector direction. Later on (ResetCamera() method)
    # this vector is used to position the camera to look at the data in
    # this direction.
    aCamera = vtk.vtkCamera()
    # print "view", viewup[0], viewup[1], viewup[2]
    aCamera.SetViewUp(viewup[0], viewup[1], viewup[2])
    # print "position", position[0], position[1], position[2]
    aCamera.SetPosition(position[0], position[1], position[2])
    # print "focal point", focalpoint[0], focalpoint[1], focalpoint[2]
    aCamera.SetFocalPoint(focalpoint[0], focalpoint[1], focalpoint[2])
    # print "clipping", clipping[0],clipping[1]
    aCamera.SetClippingRange(clipping[0], clipping[1])
    # print "scale",scale
    aCamera.SetParallelScale(scale)

    # assign actors to the renderer
    ren.AddActor(refactor)
    ren.AddActor(surfaceactor)
    ren.AddActor(txt)
    ren.SetActiveCamera(aCamera)

    # set the background and size; zoom in; and render
    ren.SetBackground(1, 1, 1)
    # print "size",size[0],size[1]
    renWin.SetSize(size[0], size[1])
    renWin.Render()

    # Remove existing lights and add lightkit lights
    ren.RemoveAllLights()
    lightkit = vtk.vtkLightKit()
    lightkit.AddLightsToRenderer(ren)
    # ren.GetActiveCamera().Zoom(100/scale)


    # enable user interface interactor
    if interact == 1:
        iren.Initialize()
        renWin.Render()
        iren.Start()
        outcam = ren.GetActiveCamera()
        print
        "position", outcam.GetPosition()
        print
        "focalpoint", outcam.GetFocalPoint()
        print
        "parallel scale", outcam.GetParallelScale()
        print
        "viewup", outcam.GetViewUp()
        print
        "clip", outcam.GetClippingRange()

    else:
        # save as png
        ## Screenshot
        windowToImageFilter = vtk.vtkWindowToImageFilter()

        windowToImageFilter.SetInput(renWin)
        windowToImageFilter.SetMagnification(
            3)  # set the resolution of the output image (3 times the current resolution of vtk render window)
        windowToImageFilter.SetInputBufferTypeToRGBA()  # also record the alpha (transparency) channel
        windowToImageFilter.Update()

        pngwriter = vtk.vtkPNGWriter()
        pngwriter.SetFileName(filename)
        pngwriter.SetInputConnection(windowToImageFilter.GetOutputPort())
        pngwriter.Write()


def visualise_default(surface, ref, case, arrayname, mini, maxi):
    """Visualise surface with a default parameters."""

    # Create a lookup table to map cell data to colors
    print
    "Colormap from ", mini, "to", maxi
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(255)
    lut.SetValueRange(0, 255)

    # qualitative data from colorbrewer  --> matching qualitative colormap of Paraview
    lut.SetTableValue(0, 0, 0, 0, 1)  # Black
    lut.SetTableValue(mini, 1, 1, 1, 1)  # white
    lut.SetTableValue(mini + 1, 77 / 255., 175 / 255., 74 / 255., 1)  # green
    lut.SetTableValue(maxi - 3, 152 / 255., 78 / 255., 163 / 255., 1)  # purple
    lut.SetTableValue(maxi - 2, 255 / 255., 127 / 255., 0., 1)  # orange
    lut.SetTableValue(maxi - 1, 55 / 255., 126 / 255., 184 / 255., 1)  # blue
    lut.SetTableValue(maxi, 166 / 255., 86 / 255., 40 / 255., 1)  # brown
    lut.Build()

    # create a text actor
    txt = vtk.vtkTextActor()
    txt.SetInput(case)
    txtprop = txt.GetTextProperty()
    txtprop.SetFontFamilyToArial()
    txtprop.SetFontSize(18)
    txtprop.SetColor(0, 0, 0)
    txt.SetDisplayPosition(20, 30)

    # create a rendering window, renderer, and renderwindowinteractor
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    # for GIMIAS interaction style
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)
    iren.SetRenderWindow(renWin)

    # surface mapper and actor
    surfacemapper = vtk.vtkPolyDataMapper()
    surfacemapper.SetInput(surface)
    surfacemapper.SetScalarModeToUsePointFieldData()
    surfacemapper.SelectColorArray(arrayname)
    surfacemapper.SetLookupTable(lut)
    surfacemapper.SetScalarRange(0, 255)
    surfaceactor = vtk.vtkActor()
    # surfaceactor.GetProperty().SetOpacity(0)
    # surfaceactor.GetProperty().SetColor(1, 1, 1)
    surfaceactor.SetMapper(surfacemapper)

    # refsurface mapper and actor
    refmapper = vtk.vtkPolyDataMapper()
    refmapper.SetInput(ref)
    refmapper.SetScalarModeToUsePointFieldData()
    refmapper.SelectColorArray(arrayname)
    refmapper.SetLookupTable(lut)
    refmapper.SetScalarRange(0, 255)
    refactor = vtk.vtkActor()
    refactor.GetProperty().SetOpacity(0.5)
    # refactor.GetProperty().SetColor(1, 1, 1)
    refactor.SetMapper(refmapper)

    # assign actors to the renderer
    ren.AddActor(refactor)
    ren.AddActor(surfaceactor)
    ren.AddActor(txt)

    # set the background and size; zoom in; and render
    ren.SetBackground(1, 1, 1)
    renWin.SetSize(1280, 960)
    ren.ResetCamera()
    ren.GetActiveCamera().Zoom(1)

    # before
    print
    "before", ren.GetActiveCamera().GetViewUp()

    # enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()

    outcam = ren.GetActiveCamera()
    print
    "after", outcam.GetViewUp()


def visualise_default_continuous(surface, overlay, case, arrayname, edges=0, flip=0, colormap='BlYl',
                                 interact=1, filename='./screenshot.png', LegendTitle='', mini='', maxi='', mag=2):
    """Visualise surface with a continuos colormap according to 'arrayname'."""

    # surface mapper and actor
    surfacemapper = vtk.vtkPolyDataMapper()
    surfacemapper.SetInput(surface)
    surfacemapper.SelectColorArray(arrayname)

    # Create a lookup table to map cell data to colors
    if surface.GetPointData().GetArray(arrayname):
        print
        "Point data"
        array = vtk_to_numpy(surface.GetPointData().GetArray(arrayname))
        surfacemapper.SetScalarModeToUsePointFieldData()
    else:
        print
        "Cell data"
        array = vtk_to_numpy(surface.GetCellData().GetArray(arrayname))
        surfacemapper.SetScalarModeToUseCellFieldData()

    if not maxi:
        maxi = np.nanmax(array)

    if not mini:
        # mini might be zero, and is true
        if mini == 0:
            mini = mini
        else:
            mini = np.nanmin(array)

    print
    "From", mini, "to", maxi, "colormap", colormap

    colors = getcolors(colormap)
    numcolors = int(len(colors))

    if colormap == '24_regions':
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(24)

        # don't interpolate
        for c in range(len(colors)):
            this = colors[c]
            lut.SetTableValue(this[0], this[1], this[2], this[3])
        lut.Build()
        surfacemapper.SetLookupTable(lut)
        surfacemapper.SetScalarRange(1, 24)
        surfacemapper.InterpolateScalarsBeforeMappingOn()
    else:
        lut = vtk.vtkColorTransferFunction()
        lut.SetColorSpaceToHSV()
        lut.SetNanColor(0.5, 0.5, 0.5)
        for c in range(len(colors)):
            cmin = colors[0][0]
            cmax = colors[numcolors - 1][0]
            this = colors[c]
            rat = (this[0] - cmin) / (cmax - cmin)
            t = (maxi - mini) * rat + mini
            lut.AddRGBPoint(t, this[1], this[2], this[3])
        lut.Build()
        surfacemapper.SetLookupTable(lut)
        surfacemapper.SetScalarRange(mini, maxi)
        surfacemapper.InterpolateScalarsBeforeMappingOn()

    # create a text actor
    txt = vtk.vtkTextActor()
    txt.SetInput(case)
    txtprop = txt.GetTextProperty()
    txtprop.SetFontFamilyToArial()
    txtprop.SetFontSize(30)
    txtprop.SetColor(0, 0, 0)
    txt.SetDisplayPosition(20, 30)

    # create a rendering window, renderer, and renderwindowinteractor
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    # for GIMIAS interaction style
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)
    iren.SetRenderWindow(renWin)

    surfaceactor = vtk.vtkActor()
    surfaceactor.SetMapper(surfacemapper)

    # overlay mapper and actor
    refmapper = vtk.vtkPolyDataMapper()
    refmapper.SetInput(overlay)
    refmapper.SetScalarModeToUsePointFieldData()

    if edges == 1:
        refactor = vtk.vtkActor()
        refactor.GetProperty().SetOpacity(1)
        if colormap == 'erdc_rainbow_grey':
            refactor.GetProperty().SetColor(1, 1, 1)
        else:
            refactor.GetProperty().SetColor(0, 0, 0)
        refactor.GetProperty().SetRepresentationToWireframe()
        refactor.GetProperty().SetLineWidth(6.)
        refactor.SetMapper(refmapper)
    else:
        refactor = vtk.vtkActor()
        refactor.GetProperty().SetOpacity(0.5)
        refactor.GetProperty().SetColor(1, 0, 0)
        refactor.SetMapper(refmapper)

    ren.AddActor(surfaceactor)
    ren.AddActor(refactor)

    if LegendTitle:
        ScalarBarActor = vtk.vtkScalarBarActor()
        # colormap
        ScalarBarActor.SetLookupTable(surfaceactor.GetMapper().GetLookupTable())

        # labels format
        ScalarBarActor.GetLabelTextProperty().ItalicOff()
        ScalarBarActor.GetLabelTextProperty().BoldOn()
        ScalarBarActor.GetLabelTextProperty().ShadowOff()
        ScalarBarActor.GetLabelTextProperty().SetFontFamilyToArial()
        ScalarBarActor.GetLabelTextProperty().SetFontSize(100)
        ScalarBarActor.GetLabelTextProperty().SetColor(0., 0., 0.)
        ScalarBarActor.SetLabelFormat('%.0f')

        if colormap == '24_regions':
            ScalarBarActor.GetLabelTextProperty().BoldOff()
            ScalarBarActor.GetLabelTextProperty().SetFontSize(100)
            ScalarBarActor.SetNumberOfLabels(24)
            ScalarBarActor.SetLabelFormat('%.0f')

        # orientation
        # ScalarBarActor.SetOrientationToHorizontal()
        ScalarBarActor.SetMaximumWidthInPixels(175)
        # ScalarBarActor.SetAnnotationTextScaling(1)
        ScalarBarActor.SetPosition(0.90, 0.15)
        ren.AddActor(ScalarBarActor)
    else:
        ren.AddActor(txt)

    # set the background and size; zoom in; and render
    ren.SetBackground(1, 1, 1)
    renWin.SetSize(875, 800)
    ren.ResetCamera()

    if flip == 1:
        print
        "flip to foot to head position"
        # default values from paraview
        aCamera = vtk.vtkCamera()
        aCamera.SetViewUp(0, 1, 0)
        aCamera.SetPosition(0, 0, -3.17)
        aCamera.SetFocalPoint(0, 0, 0)
        aCamera.SetClippingRange(3.14, 3.22)
        aCamera.SetParallelScale(0.83)
        ren.SetActiveCamera(aCamera)

    ren.GetActiveCamera().Zoom(1.4)
    iren.Initialize()
    renWin.Render()

    # Remove existing lights and add lightkit lights
    ren.RemoveAllLights()
    lightkit = vtk.vtkLightKit()
    lightkit.AddLightsToRenderer(ren)

    # enable user interface interactor
    if interact == 1:

        iren.Start()
    else:
        # save as png
        ## Screenshot
        windowToImageFilter = vtk.vtkWindowToImageFilter()

        windowToImageFilter.SetInput(renWin)
        windowToImageFilter.SetMagnification(
            mag)  # set the resolution of the output image (3 times the current resolution of vtk render window)
        windowToImageFilter.SetInputBufferTypeToRGBA()  # also record the alpha (transparency) channel
        windowToImageFilter.Update()

        pngwriter = vtk.vtkPNGWriter()
        pngwriter.SetFileName(filename)
        pngwriter.SetInputConnection(windowToImageFilter.GetOutputPort())
        pngwriter.Write()


def roundpointarray(polydata, name):
    """Round values in point array."""
    # get original array
    array = polydata.GetPointData().GetArray(name)

    # round labels
    for i in range(polydata.GetNumberOfPoints()):
        value = array.GetValue(i)
        array.SetValue(i, round(value))
    return polydata


def save_colormap(name, outfile):
    """Save colormap as xml."""
    colors = getcolors(name)
    numcolors = int(len(colors))

    f = open(outfile, 'w')
    header = '<ColorMap name="' + name + '" space="HSV" indexedLookup="true">\n'
    f.write(header)

    for c in range(len(colors)):
        this = colors[c]
        line = '  <Point x="' + str(c) + '" o="1" '
        line += 'r="' + str(this[0]) + '" '
        line += 'g="' + str(this[1]) + '" '
        line += 'b="' + str(this[2]) + '"/>\n'
        f.write(line)
    line = '<NaN r="1" g="1" b="1"/>\n'
    f.write(line)
    line = '</ColorMap>\n'
    f.write(line)
    f.close()


def visualise_nan_glyph(surface, glyph, overlay, case, arrayname, edges=0, flip=0, colormap='BlYl',
                        interact=1, filename='./screenshot.png', LegendTitle='', mini='', maxi=''):
    """Visualise surface with glyphs colormapped according to 'arrayname'."""

    # glyph mapper
    glyphmapper = vtk.vtkPolyDataMapper()
    glyphmapper.SetInput(glyph)
    glyphmapper.SelectColorArray(arrayname)

    # only pointdata
    array = vtk_to_numpy(surface.GetPointData().GetArray(arrayname))
    glyphmapper.SetScalarModeToUsePointFieldData()

    if not maxi:
        maxi = np.nanmax(array)

    if not mini:
        # mini might be zero, and is true
        if mini == 0:
            mini = mini
        else:
            mini = np.nanmin(array)

    print
    "From", mini, "to", maxi

    colors = getcolors(colormap)
    numcolors = int(len(colors))

    if colormap == '24_regions':
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(24)

        # don't interpolate
        for c in range(len(colors)):
            this = colors[c]
            lut.SetTableValue(this[0], this[1], this[2], this[3])
        lut.Build()
        glyphmapper.SetLookupTable(lut)
        glyphmapper.SetScalarRange(1, 24)
        glyphmapper.InterpolateScalarsBeforeMappingOn()
    else:
        lut = vtk.vtkColorTransferFunction()
        lut.SetColorSpaceToHSV()
        lut.SetNanColor(0.5, 0.5, 0.5)
        for c in range(len(colors)):
            cmin = colors[0][0]
            cmax = colors[numcolors - 1][0]
            this = colors[c]
            rat = (this[0] - cmin) / (cmax - cmin)
            t = (maxi - mini) * rat + mini
            lut.AddRGBPoint(t, this[1], this[2], this[3])
        lut.Build()
        glyphmapper.SetLookupTable(lut)
        glyphmapper.SetScalarRange(mini, maxi)
        glyphmapper.InterpolateScalarsBeforeMappingOn()

    # create a text actor
    txt = vtk.vtkTextActor()
    txt.SetInput(case)
    txtprop = txt.GetTextProperty()
    txtprop.SetFontFamilyToArial()
    txtprop.SetFontSize(30)
    txtprop.SetColor(0, 0, 0)
    txt.SetDisplayPosition(20, 30)

    # create a rendering window, renderer, and renderwindowinteractor
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)
    iren.SetRenderWindow(renWin)

    # surface in solid color
    # surface mapper
    surfacemapper = vtk.vtkPolyDataMapper()
    surfacemapper.SetInput(surface)
    surfaceactor = vtk.vtkActor()
    surfaceactor.SetMapper(surfacemapper)
    surfaceactor.GetProperty().SetOpacity(0.5)
    surfaceactor.GetProperty().SetColor(0.5, 0.5, 0.5)

    # glyph in scalar colors

    glyphactor = vtk.vtkActor()
    glyphactor.SetMapper(glyphmapper)

    # overlay mapper and actor
    refmapper = vtk.vtkPolyDataMapper()
    refmapper.SetInput(overlay)
    refmapper.SetScalarModeToUsePointFieldData()

    if edges == 1:
        refactor = vtk.vtkActor()
        refactor.GetProperty().SetOpacity(1)
        refactor.GetProperty().SetColor(0, 0, 0)
        refactor.GetProperty().SetRepresentationToWireframe()
        refactor.GetProperty().SetLineWidth(6.)
        refactor.SetMapper(refmapper)
    else:
        refactor = vtk.vtkActor()
        refactor.GetProperty().SetOpacity(0.5)
        refactor.GetProperty().SetColor(1, 0, 0)
        refactor.SetMapper(refmapper)

    ren.AddActor(surfaceactor)
    ren.AddActor(refactor)
    ren.AddActor(glyphactor)

    if LegendTitle:
        ScalarBarActor = vtk.vtkScalarBarActor()
        # colormap
        ScalarBarActor.SetLookupTable(glyphactor.GetMapper().GetLookupTable())

        # labels format
        ScalarBarActor.GetLabelTextProperty().ItalicOff()
        ScalarBarActor.GetLabelTextProperty().BoldOn()
        ScalarBarActor.GetLabelTextProperty().ShadowOff()
        ScalarBarActor.GetLabelTextProperty().SetFontFamilyToArial()
        ScalarBarActor.GetLabelTextProperty().SetFontSize(100)
        ScalarBarActor.GetLabelTextProperty().SetColor(0., 0., 0.)
        ScalarBarActor.SetLabelFormat('%.2f')

        if colormap == '24_regions':
            ScalarBarActor.GetLabelTextProperty().BoldOff()
            ScalarBarActor.GetLabelTextProperty().SetFontSize(100)
            ScalarBarActor.SetNumberOfLabels(24)
            ScalarBarActor.SetLabelFormat('%.0f')

        # orientation
        # ScalarBarActor.SetOrientationToHorizontal()
        ScalarBarActor.SetMaximumWidthInPixels(200)
        # ScalarBarActor.SetAnnotationTextScaling(1)
        ScalarBarActor.SetPosition(0.90, 0.15)
        ren.AddActor(ScalarBarActor)
    else:
        ren.AddActor(txt)

    # set the background and size; zoom in; and render
    ren.SetBackground(1, 1, 1)
    renWin.SetSize(875, 800)
    ren.ResetCamera()

    if flip == 1:
        print
        "flip to foot to head position"
        # default values from paraview
        aCamera = vtk.vtkCamera()
        aCamera.SetViewUp(0, 1, 0)
        aCamera.SetPosition(0, 0, -3.17)
        aCamera.SetFocalPoint(0, 0, 0)
        aCamera.SetClippingRange(3.14, 3.22)
        aCamera.SetParallelScale(0.83)
        ren.SetActiveCamera(aCamera)

    ren.GetActiveCamera().Zoom(1.4)
    iren.Initialize()
    renWin.Render()

    # Remove existing lights and add lightkit lights
    ren.RemoveAllLights()
    lightkit = vtk.vtkLightKit()
    lightkit.AddLightsToRenderer(ren)

    # enable user interface interactor
    if interact == 1:

        iren.Start()
    else:
        # save as png
        ## Screenshot
        windowToImageFilter = vtk.vtkWindowToImageFilter()

        windowToImageFilter.SetInput(renWin)
        windowToImageFilter.SetMagnification(
            3)  # set the resolution of the output image (3 times the current resolution of vtk render window)
        windowToImageFilter.SetInputBufferTypeToRGBA()  # also record the alpha (transparency) channel
        windowToImageFilter.Update()

        pngwriter = vtk.vtkPNGWriter()
        pngwriter.SetFileName(filename)
        pngwriter.SetInputConnection(windowToImageFilter.GetOutputPort())
        pngwriter.Write()


def visualise_cell(surface, ref, case, arrayname, mini, maxi):
    """Visualise surface colormapping cell fielddata in 'arrayname'."""
    # Create a lookup table to map cell data to colors
    print
    "Colormap from ", mini, "to", maxi
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(255)
    lut.SetValueRange(0, 255)

    # qualitative data from colorbrewer  --> matching qualitative colormap of Paraview
    lut.SetTableValue(0, 0, 0, 0, 1)  # Black
    lut.SetTableValue(maxi, 0.894118, 0.101961, 0.109804, 1)  # red
    lut.SetTableValue(maxi - 1, 0.215686, 0.494118, 0.721569, 1)  # blue
    lut.SetTableValue(maxi - 2, 0.301961, 0.686275, 0.290196, 1)  # green
    lut.SetTableValue(maxi - 3, 0.596078, 0.305882, 0.639216, 1)  # pruple
    lut.SetTableValue(mini, 1, 0.498039, 0, 1)  # Orangie
    lut.SetTableValue(mini + 1, 1, 1, 0.498039, 1)  # yellow
    lut.Build()
    # print lut

    # create a text actor
    txt = vtk.vtkTextActor()
    txt.SetInput(case)
    txtprop = txt.GetTextProperty()
    txtprop.SetFontFamilyToArial()
    txtprop.SetFontSize(18)
    txtprop.SetColor(0, 0, 0)
    txt.SetDisplayPosition(20, 30)

    # create a rendering window, renderer, and renderwindowinteractor
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    # for GIMIAS interaction style
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)
    iren.SetRenderWindow(renWin)

    # surface mapper and actor
    surfacemapper = vtk.vtkPolyDataMapper()
    surfacemapper.SetInput(surface)
    surfacemapper.SetScalarModeToUseCellFieldData()
    surfacemapper.SelectColorArray(arrayname)
    surfacemapper.SetLookupTable(lut)
    surfacemapper.SetScalarRange(0, 255)
    surfaceactor = vtk.vtkActor()
    # surfaceactor.GetProperty().SetOpacity(0)
    # surfaceactor.GetProperty().SetColor(1, 1, 1)
    surfaceactor.SetMapper(surfacemapper)

    # refsurface mapper and actor
    refmapper = vtk.vtkPolyDataMapper()
    refmapper.SetInput(ref)
    refmapper.SetScalarModeToUseCellFieldData()
    refmapper.SelectColorArray(arrayname)
    refmapper.SetLookupTable(lut)
    refmapper.SetScalarRange(0, 255)
    refactor = vtk.vtkActor()
    refactor.GetProperty().SetOpacity(0.5)
    # refactor.GetProperty().SetColor(1, 1, 1)
    refactor.SetMapper(refmapper)

    # assign actors to the renderer
    ren.AddActor(refactor)
    ren.AddActor(surfaceactor)
    ren.AddActor(txt)

    # set the background and size; zoom in; and render
    ren.SetBackground(1, 1, 1)
    renWin.SetSize(1280, 960)
    ren.ResetCamera()
    ren.GetActiveCamera().Zoom(1)

    # enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()


def visualise_color(surface, ref, case):
    """Visualise surface in solid color and 'ref' in trasparent."""
    # create a text actor
    txt = vtk.vtkTextActor()
    txt.SetInput(case)
    txtprop = txt.GetTextProperty()
    txtprop.SetFontFamilyToArial()
    txtprop.SetFontSize(18)
    txtprop.SetColor(0, 0, 0)
    txt.SetDisplayPosition(20, 30)

    # create a rendering window, renderer, and renderwindowinteractor
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    # for GIMIAS interaction style
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)
    iren.SetRenderWindow(renWin)

    # surface mapper and actor
    surfacemapper = vtk.vtkPolyDataMapper()
    surfacemapper.SetInput(surface)
    surfacemapper.SetScalarModeToUsePointFieldData()
    surfaceactor = vtk.vtkActor()
    # surfaceactor.GetProperty().SetOpacity(0)
    surfaceactor.GetProperty().SetColor(288 / 255, 26 / 255, 28 / 255)
    surfaceactor.SetMapper(surfacemapper)

    # refsurface mapper and actor
    refmapper = vtk.vtkPolyDataMapper()
    refmapper.SetInput(ref)
    refmapper.SetScalarModeToUsePointFieldData()

    refactor = vtk.vtkActor()
    refactor.GetProperty().SetOpacity(0.5)
    refactor.GetProperty().SetColor(1, 1, 1)
    refactor.SetMapper(refmapper)

    # assign actors to the renderer
    # ren.AddActor(refactor)
    ren.AddActor(surfaceactor)
    ren.AddActor(refactor)
    ren.AddActor(txt)

    # set the background and size; zoom in; and render
    ren.SetBackground(1, 1, 1)
    renWin.SetSize(800, 800)
    ren.ResetCamera()
    ren.GetActiveCamera().Zoom(1)

    # enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()


def volume(polydata):
    """Compute volume in polydata."""
    properties = vtk.vtkMassProperties()
    properties.SetInput(polydata)
    properties.Update()
    return properties.GetVolume()


# ------------------------------------------------------------------------------
# Input/Output
# ------------------------------------------------------------------------------

def csv2list(csvfile):
    """Read csv file and return list."""
    ifile = open(csvfile, 'rb')
    reader = csv.reader(ifile)
    csvlist = []
    for row in reader:
        csvlist.append(row)
    ifile.close
    return csvlist


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


def readvtp(filename, dataarrays=True):
    """Read polydata in vtp format."""
    reader = vtk.vtkXMLPolyDataReader()
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


def readpolydatavtk(filename, dataarrays=True):
    """Read polydata in vtk format."""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def replacestring(lines, tag, value):
    """Replace a string in 'tag' with 'value'."""
    output = []
    for line in lines:
        line = line.replace(tag, value)
        output.append(line)
    return output


def round_point_array(polydata, name):
    """Round all componenets of data array."""
    # get original array
    array = polydata.GetPointData().GetArray(name)

    # round labels
    for i in range(polydata.GetNumberOfPoints()):
        value = array.GetValue(i)
        array.SetValue(i, round(value))

    return polydata


def set_point_array_value(polydata, name, value):
    """Set all components of data array to 'value'."""
    # get original array
    array = polydata.GetPointData().GetArray(name)

    # round labels
    for i in range(polydata.GetNumberOfPoints()):
        array.SetValue(i, round(value))

    return polydata


def vtk2vtp(inputfile, outputfile):
    """Read a vtk polydata and save as vtp."""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(inputfile)
    reader.Update()
    surface = reader.GetOutput()
    writevtp(surface, outputfile)


def ply2vtk(filename1, filename2):
    """Read a ply file and save as vtk ascii."""
    reader = vtk.vtkPLYReader()
    reader.SetFileName(filename1)
    reader.Update()

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputConnection(reader.GetOutputPort())
    writer.SetFileTypeToASCII()
    writer.SetFileName(filename2)
    writer.Write()


def vtk2vtk(inputfile, outputfile):
    """Read a vtk polydata and save as vtk in ascii format."""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(inputfile)
    reader.Update()

    writer = vtk.vtkPolyDataWriter()
    writer.SetInput(surface)  # M: surface??? reader.GetOutput()
    writer.SetFileName(outputfile)
    writer.SetFileTypeToASCII()
    writer.Write()

def vtk2stl(fn_in, fn_out):

    reader = vtk.vtkDataSetReader()
    reader.SetFileName(fn_in)
    reader.Update()

    gfilter = vtk.vtkGeometryFilter()
    gfilter.SetInputData(reader.GetOutput())

    writer = vtk.vtkSTLWriter()
    writer.SetFileName(fn_out)
    writer.SetInputConnection(gfilter.GetOutputPort())
    writer.Write()  


def vtp2vtk(inputfile, outputfile):
    """Read a vtp polydata and write as vtk."""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(inputfile)
    reader.Update()
    surface = reader.GetOutput()

    writer = vtk.vtkPolyDataWriter()
    writer.SetInput(surface)
    writer.SetFileName(outputfile)
    writer.Write()


def writepolydataarray2csv(polydata, outputfile, regionsname='RegionId', distancearrayname='Distance'):
    """Write a scalar array as csv."""
    array = polydata.GetPointData().GetArray(distancearrayname)
    regions = polydata.GetPointData().GetArray(regionsname)
    # print array,regions

    f = open(outputfile, 'wb')
    # write values of arrays row by row
    for i in range(polydata.GetNumberOfPoints()):
        line = str(regions.GetValue(i)) + ', ' + str(array.GetValue(i)) + '\n'
        f.write(line)
    f.close()


def writeply(surface, filename):
    """Write mesh as ply file."""
    writer = vtk.vtkPLYWriter()
    writer.SetInput(surface)
    # writer.SetFileTypeToASCII()
    writer.SetFileName(filename)
    writer.SetFileTypeToBinary()  # Marta
    writer.Write()


def writevti(image, filename):
    """Write vti file."""
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetInput(image)
    writer.SetFileName(filename)
    writer.Write()


def writevtk(surface, filename):  #ESTO ESTA MAL. LO GUARDA EN 5.1 VERSION (YO QUIERO EN 4.2 PARA PODER ABRIRLO)
#PASA POR LA VERSION DE VTK QUE TENGO (9), NO PUEDO DOWNGRADE (8.2) CON LA VERSION DE PYTHON QUE TENGO (3)
    """Write vtk polydata file."""
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(surface)
    writer.SetFileTypeToASCII()
    writer.SetFileName(filename)
 
    
    # writer.SetDataModeToBinary()
    writer.Write()


def writevtp(surface, filename):
    """Write vtp polydata file."""
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(surface)
    writer.SetFileName(filename)
    #    writer.SetDataModeToBinary()
    writer.Write()


def writematrix2csv(array, ofile):
    """Write matrix to csv."""
    f = open(ofile, 'wb')
    arraysize = array.shape
    print
    arraysize

    line = ''
    for r in range(arraysize[0]):
        for c in range(arraysize[1]):
            line = line + str(array[r, c]) + ', '
        line = line + '\n'
        f.write(line)
        line = ''

    f.close()


def writearray2csv(array, ofile, label=''):
    """Write array to csv."""
    f = open(ofile, 'wb')
    for i in range(len(array)):
        if label:
            line = str(label[i]) + ', ' + str(array[i]) + '\n'
        else:
            line = str(array[i]) + '\n'
        f.write(line)
    f.close()
