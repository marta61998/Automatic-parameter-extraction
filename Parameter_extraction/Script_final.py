import pandas as pd
import numpy as np
import pandas as pd
from sympy import Q
from functions_script2 import *
import pyvista as pv
import tetgen
import time
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input_mesh', type=str, help='path to input mesh')
parser.add_argument('--input_excel', type=str, help='path to input excel')
parser.add_argument('--output', type=str, help='path to output')

args = parser.parse_args()

input_path = args.input_mesh
input_path_excel = args.input_excel

output_path = args.output
cur_dir = output_path
print(cur_dir)

#---------------------------------------------  SCRIPT 2-----------------------------------------------------------

#Read path to ostium coordinate prediction and LA mesh
if not os.path.exists(cur_dir):
    os.mkdir(cur_dir)

frames = []
list_times = []
df4 = pd.read_excel(input_path_excel, engine='openpyxl',)

tic = time.perf_counter()
mesh = readstl(input_path)
mesh = extractlargestregion(mesh)
center_mesh = estimate_center(mesh)
ostium_coord_pred = np.array((df4.ostium_pred_x[0],df4.ostium_pred_y[0],df4.ostium_pred_z[0]))

# ################################ 1. CUT LAA RAW ##################################

#find initial normal vector
AB = np.subtract(ostium_coord_pred,center_mesh)
AB2 = normalizevector(AB)
AB2 = np.array(AB2)
n=100
P4 = center_mesh + n * AB2

normal = AB2
origin = ostium_coord_pred

#rotate to z axis
normal_align = np.array([0, 0, 1])  #eje z
path_rotated = None
rotated, matrix = rotate_angle(mesh, normal, normal_align, center_mesh, path_rotated)
cent2 = np.array([origin[0],origin[1],origin[2],1])
cent_rotated = matrix.dot(cent2)
origin_rotated = cent_rotated[0:3]
inv = np.linalg.inv(matrix)

#Raw clip of LAA (for heat equation)
print('Rotated')
#Apply random perturbations to raw normal
normal_opt, centroid_opt = brute_force_perturbation(rotated,normal_align, origin_rotated,0.4,0, no=20)

origin = centroid_opt
normal = normal_opt
clip, p0 = clip_LAA(rotated, origin, normal, viz = False)

#Rotate to original axis
n=100
P2 = origin + n * normal

origin2= np.array([origin[0],origin[1],origin[2],1])
origin_ori = inv.dot(origin2)
origin_ori2 = origin_ori[0:3]

P2= np.array([P2[0],P2[1],P2[2],1])
P2 = inv.dot(P2)
P2_ori2 = P2[0:3]

normal_ori = np.subtract(P2_ori2,origin_ori2)
normal_ori2 = normalizevector(normal_ori)

origin = np.array(origin_ori2)
normal = np.array(normal_ori2)

T = vtk.vtkTransform()
matrix = vtk.vtkMatrix4x4()
for i in range(0, 4):
    for j in range(0, 4):
        matrix.SetElement(i, j, inv[i, j])
T.SetMatrix(matrix)

tf = vtk.vtkTransformPolyDataFilter()
tf.SetTransform(T)
tf.SetInputData(clip)
tf.Update()
clip_ori = tf.GetOutput()

edges_ori = extractboundaryedge(clip_ori)

if not os.path.exists(cur_dir + '/cut_raw'):
    os.mkdir(cur_dir + '/cut_raw')
path_clip_bueno_stl= cur_dir + '/cut_raw/LAA.stl'
path_clip_output= cur_dir + '/cut_raw/LAA_rep.stl'
writestl(clip_ori, path_clip_bueno_stl)
#Fix mesh
fix_mesh(path_clip_bueno_stl,path_clip_output) 

repaired = readstl(path_clip_bueno_stl)

list_ends = []
list_dist = []

#Detect contour points of the ostium (needed in heat equation)
edge_noDelaunay, surface_area, LAA_edge, LAA_error, regions_center = extractlargestregion_edges_m2(edges_ori)
npoints = edge_noDelaunay.GetNumberOfPoints()
ostium_array = []
for i in range(npoints):
    ostium_array.append(np.array(edge_noDelaunay.GetPoint(i)))
p0_np = np.array(ostium_array)

# ################################ 2. DETECT LAA ENDPOINT HEAT ##################################

target_reduction_list = [0.4,0.5,0.6,0.7] #try heat equation with different decimation indices (depends on the geometry)
list_mesh_scalar = []
if not os.path.exists(cur_dir + '/vol_processed'):
    os.mkdir(cur_dir + '/vol_processed')
print('hola')
for m in target_reduction_list:

    mesh_LAA = pv.read(path_clip_bueno_stl)
    smooth = mesh_LAA.smooth(n_iter=1)
    decimated = mesh_LAA.decimate(m)
    decimated = decimated.decimate(m)
    print("Constructing volumetric mesh...")
    tet = tetgen.TetGen(decimated)
    tet.make_manifold()
    tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
    grid = tet.grid

    ostium, LAA_wall, ostium_centre = detectOstium(decimated,p0_np,normal,origin)

    print("Defining boundary conditions")
    boundaryConditions, boundaryPoints = defineBoundaryConditions(grid, ostium, LAA_wall)
    volumetricMeshPath = cur_dir + '/vol_processed/LA_processed_vol.vtk'
    grid.save(volumetricMeshPath)
            
    meshVTK = readUnstructuredGridVTK(volumetricMeshPath)
    print("Calculating the end point of the centreline, solving the Laplace Equation...")
    result = solveLaplaceEquationTetrahedral(volumetricMeshPath,decimated , boundaryPoints, boundaryConditions[:,0])
    #save mesh with heat equation results
    mesh_with_scalar = add_scalar(meshVTK, result, name = 'heat', domain = 'point')
    writeUnstructuredGridVTK(volumetricMeshPath,mesh_with_scalar)

    end_point = detectEndPointOfCentreline(result,grid)
    print(end_point)
    list_ends.append(end_point)  
  
    dist = calculateDistance(origin,end_point)
    list_dist.append(dist)
    list_mesh_scalar.append(mesh_with_scalar)

#select the endpoint further away from ostium
ind = list_dist.index(max(list_dist))
end_point = list_ends[ind]
writeUnstructuredGridVTK(volumetricMeshPath,list_mesh_scalar[ind])
actor = addpoint_show(end_point, color = [1,0,0])
rend = vtk.vtkRenderer()
#if not os.path.exists(cur_dir + '/image_out'):
#    os.mkdir(cur_dir + '/image_out')
#filename = cur_dir + '/image_out/screenshot1.png' #save output renderer and not show
#vtk_show(rend, clip_ori,1000,1000,filename, actor)



#Save endpoint detected results
print("Writing results...")
raw_data = {'Patient_ID': [int(0)], 'end_point_x': [end_point[0]], 'end_point_y': [end_point[1]],'end_point_z': [end_point[2]],'ost_pos_x': [origin[0]], 'ost_pos_y': [origin[1]],'ost_pos_z': [origin[2]],'ost_or_x': [normal[0]], 'ost_or_y': [normal[1]],'ost_or_z': [normal[2]]}
df = pd.DataFrame(raw_data, columns=['Patient_ID', 'end_point_x', 'end_point_y','end_point_z', 'ost_pos_x', 'ost_pos_y','ost_pos_z', 'ost_or_x', 'ost_or_y','ost_or_z'])

frames.append(df)
#print(case)

result = pd.concat(frames)

out_fcsv = cur_dir + '/endpoint_script2.xlsx'  
result.to_excel(out_fcsv)

print('time script2',time)

    ################################ 3. MAKE CENTERLINE ##################################

centreline = vmtkcenterlines(mesh, center_mesh, end_point, 0)
nskippoints = 0

clspacing = 0.4
cl = vmtkcenterlineresampling(centreline, clspacing)

if not os.path.exists(cur_dir + '/new_centrelines'):
    os.mkdir(cur_dir + '/new_centrelines')
writevtp(cl, cur_dir + '/new_centrelines/centreline.vtp')

#-------------------------------------------------- SCRIPT 3--------------------------------------------------------------

    ################################ 1. GET FINAL OSTIUM NORMAL ##################################
#find definitive ostium plane normal from centerline
npoints = cl.GetNumberOfPoints()
CL_points = []
for i in range(npoints):
    CL_points.append(cl.GetPoint(i))
distances = calculateDistanceToOstium(CL_points, origin)  #calcula la distancia de cada punto del centreline a la coordenada de Daniel, entonces una distancia debe de ser 0 no?
start = np.argmin(distances) 
origin_cl = CL_points[start]
last = CL_points[0]
                
previous_point = CL_points[start-5]
next_point = CL_points[start+5]

vector_x = next_point[0] - previous_point[0]
vector_y = next_point[1] - previous_point[1]
vector_z = next_point[2] - previous_point[2]
module = math.sqrt(math.pow(vector_x, 2) + math.pow(vector_y, 2) + math.pow(vector_z, 2))

normal = [vector_x/module,vector_y/module,vector_z/module]

#rotate to z axis
normal_align = np.array([0, 0, 1])  #eje z
path_rotated = None
rotated, matrix = rotate_angle(mesh, normal, normal_align, center_mesh, path_rotated)
cent2 = np.array([origin[0],origin[1],origin[2],1])
cent_rotated = matrix.dot(cent2)
origin_rotated = cent_rotated[0:3]
#Rotate the 'next_point' (+5) centerline point of the origin for new ostium plane search
next2 = np.array([next_point[0],next_point[1],next_point[2],1])
next_rotated = matrix.dot(next2)
next_rotated = next_rotated[0:3]


print('Rotated')
normal_opt, centroid_opt,flag = brute_force_perturbation_2(rotated,normal_align, origin_rotated,0.4,0,next_rotated, no=20)

#Rotate back 
origin = centroid_opt
normal = normal_opt
#file = rotated

clip, p0 = clip_LAA(rotated, origin, normal, viz = False)
inv = np.linalg.inv(matrix)
n=100
P2 = origin + n * normal

origin2= np.array([origin[0],origin[1],origin[2],1])
origin_ori = inv.dot(origin2)
origin_ori2 = origin_ori[0:3]

P2= np.array([P2[0],P2[1],P2[2],1])
P2 = inv.dot(P2)
P2_ori2 = P2[0:3]
normal_ori = np.subtract(P2_ori2,origin_ori2)
normal_ori2 = normalizevector(normal_ori)
print(normal_ori2)

origin = np.array(origin_ori2)
normal = np.array(normal_ori2)

T = vtk.vtkTransform()
matrix = vtk.vtkMatrix4x4()
for i in range(0, 4):
    for j in range(0, 4):
        matrix.SetElement(i, j, inv[i, j])
T.SetMatrix(matrix)

tf = vtk.vtkTransformPolyDataFilter()
tf.SetTransform(T)
tf.SetInputData(clip)
tf.Update()
clip_ori = tf.GetOutput()

#save mesh, .stl, este es el LAA bueno
if not os.path.exists(cur_dir + '/LAA_clip_final'):
    os.mkdir(cur_dir + '/LAA_clip_final')
path_clip_bueno_stl= cur_dir + '/LAA_clip_final/LAA.stl'
writestl(clip_ori, path_clip_bueno_stl)

#rend1 = vtk.vtkRenderer()
#actor_p = addpoint_show(previous_point, color = [1,0,0])
#actor_a = addpoint_show(next_point, color = [0,0,1])
#filename = cur_dir + '/image_out/screenshot2.png'
#vtk_show3(rend1,mesh,clip_ori,cl,1000,1000,filename, actor_p, actor_a)

#compute centerline id close to ostium 
npoints = cl.GetNumberOfPoints()
CL_points_rotated = []
for i in range(npoints):
    CL_points_rotated.append(cl.GetPoint(i))
distances = calculateDistanceToOstium(CL_points_rotated, origin)  #calcula la distancia de cada punto del centreline a la coordenada de Daniel, entonces una distancia debe de ser 0 no?
start = np.argmin(distances) 

#compute LA and LAA centerline lengths
length_la = 0
length_laa = 0
for i in range(len(CL_points_rotated)-1):

    previous_point = CL_points_rotated[i]
    next_point = CL_points_rotated[i+1]

    vector_x = next_point[0] - previous_point[0]
    vector_y = next_point[1] - previous_point[1]
    vector_z = next_point[2] - previous_point[2]
    module = math.sqrt(math.pow(vector_x, 2) + math.pow(vector_y, 2) + math.pow(vector_z, 2))

    length_la = length_la + module
    
for i in range(start-1):
    #if i == start:
        previous_point = CL_points_rotated[i]
        next_point = CL_points_rotated[i+1]

        vector_x = next_point[0] - previous_point[0]
        vector_y = next_point[1] - previous_point[1]
        vector_z = next_point[2] - previous_point[2]
        module = math.sqrt(math.pow(vector_x, 2) + math.pow(vector_y, 2) + math.pow(vector_z, 2))

        length_laa = length_laa + module

#Find bending angle by detecting maximum curvature of centerline
nskippoints = round(start)
cl_ori = skippoints(cl, int(nskippoints))
clspacing = 0.4
cl_res = vmtkcenterlineresampling(cl_ori, clspacing) 
centerline_geom = vmtkcenterlinegeometry(cl_res,iterations=100, factor=0.5)
pointids = centerline_geom.GetPointData().GetArray('Curvature')

list_curvature = []
for i in range(pointids.GetSize()):
    list_curvature.append(pointids.GetValue(i))

#list_laa = list_curvature[start:len(list_curvature)]
factor = round(len(list_curvature) / 4)

#list_curvature_2 = list_curvature[start:len(list_curvature)- factor]
list_curvature_2 = list_curvature[0:len(list_curvature)- factor]

point_max = list_curvature.index(max(list_curvature_2))
point_cent = cl_res.GetPoint(point_max)


point1 = np.array((CL_points_rotated[start][0],CL_points_rotated[start][1],CL_points_rotated[start][2]))
point2 = point_cent
point3 = np.array((CL_points_rotated[len(CL_points_rotated)-1][0],CL_points_rotated[len(CL_points_rotated)-1][1],CL_points_rotated[len(CL_points_rotated)-1][2]))


bending = get_bending_2(point1,point_cent,point3)
print('bending', bending)

#-------------------------------------------------SCRIPT 4 ---------------------------------------------------------------

#Compute final ostium diameter
edges = extractboundaryedge(clip_ori)
edge_noDelaunay, surface_area, LAA_edge, LAA_error, regions_center = extractlargestregion_edges_m2(edges)
d2_min, d1_complete, mean_p, p0d2, p0d1, p1d1, d2_complete, p1d2,LAA_edge,p0,boundaryIds = get_d1d2alt(edge_noDelaunay,show_points=False) 


ostium_area = calc_ostium_area(p0)

h, h_theta, M = calc_h_param(edge_noDelaunay, clip_ori, normal,viz = False)


# COMPUTE VOLUME/AREA LA 
polygonProperties = vtk.vtkMassProperties()
polygonProperties.SetInputData(mesh)
polygonProperties.Update()
LA_volume_mm = polygonProperties.GetVolume()
LA_volume_ml = LA_volume_mm / 1000
LA_area = polygonProperties.GetSurfaceArea()

path_clip_output= cur_dir + '/LAA_clip_final/LAA_rep.stl'
fix_mesh(path_clip_bueno_stl,path_clip_output)
    #    print('repaired clip')
repaired = readstl(path_clip_bueno_stl)

# COMPUTE VOLUME/AREA LAA
polygonProperties = vtk.vtkMassProperties()
polygonProperties.SetInputData(repaired)
polygonProperties.Update()
LAA_volume_mm = polygonProperties.GetVolume()
LAA_volume_ml = LAA_volume_mm / 1000
LAA_area = polygonProperties.GetSurfaceArea()

#compute ostium perimeter
ostium_perimeter = get_ostium_perimeter(clip_ori,origin) 
# COMPUTE IRREGULARITY
area_theoric = math.pi * d2_complete * d1_complete/4
irregularity = abs(1- area_theoric/ostium_area)

print ("Irregularity: ", irregularity)

ost_eccentricity=abs(1-(d2_complete/d1_complete)) #Ostium eccentricity

ost_mean_diameter=(d2_complete+d1_complete)/2 #Ostium mean diameter

frames = []
raw_data = {'Patient_ID': 1, 'D1_ostium':[d1_complete], 'D2_ostium': [d2_complete],'H(mm)': [h], 'H_theta(mm)': [h_theta], 'M': [M],'LA_area(mm^2)':[LA_area],'LA_Vol(mm^3)': [LA_volume_mm], 'LA_Vol(ml^3)': [LA_volume_ml], 'LAA_Vol(mm^3)': [LAA_volume_mm], 'LAA_Vol(ml^3)': [LAA_volume_ml], 'LAA area(mm^2)': [LAA_area],'Ostium perimeter': [ostium_perimeter], 'Ostium area': [ostium_area], 'Ost_eccentricity': [ost_eccentricity],'Ost_mean_diameter': [ost_mean_diameter],'Irregularity': [irregularity],'LA_centreline_length': [length_la], 'LAA_centreline_length': [length_laa], 'bending': [bending]}
#columns=['Patient_ID', 'D1_ostium','D2_ostium','LA_centreline_length','LAA_centreline_length']
df = pd.DataFrame(raw_data, columns=['Patient_ID', 'D1_ostium','D2_ostium','H(mm)','H_theta(mm)','M','LA_area(mm^2)','LA_Vol(mm^3)','LA_Vol(ml^3)','LAA_Vol(mm^3)','LAA_Vol(ml^3)','LAA area(mm^2)','Ostium perimeter','Ostium area','Ost_eccentricity','Ost_mean_diameter','Irregularity','LA_centreline_length','LAA_centreline_length','bending'])
frames.append(df)

#print(case)

result = pd.concat(frames)
##        # All results are stored in this path
out_fcsv = cur_dir+'/parametros_extraidos_final.xlsx'         
result.to_excel(out_fcsv)
toc = time.perf_counter()
print('Time: ', toc-tic)
# rend = vtk.vtkRenderer()
# vtk_show_centreline(rend, mesh, clip_ori, 1000,1000)