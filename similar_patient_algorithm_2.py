import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from functions_similar import *


#Load morphological parameters 
df1 = pd.read_excel('/Users/martasaiz/Documents/Carpetas_Marta/TESIS/FASE1/Extraccion_param/Codigos_Marta/excels/parametros.xlsx',
     engine='openpyxl',
)

LAA_id = df1.Patient_ID
list_LAA_id = []
for i in LAA_id: 
    list_LAA_id.append(i)

#Select input patient
case_input = input('Select input patient [number]: ')

if int(case_input) not in list_LAA_id: print('Incorrect patient number, try again')
while int(case_input) not in list_LAA_id:
    case_input = input('Select input patient [number]: ')


case_number = int(case_input)
id = list_LAA_id.index(case_number)

#info of selected input patient (Systole/Diastole and RAS2LPS)
flag_systole = df1.Systole[id]
ras2lps_input = df1.RAS2LPS[id]

print('Selected input patient', str('Bordeaux_Case'+str(case_number)))

list_case = []
list_vol_la_mm=[]
list_vol_laa_path_mm=[]
list_vol_laa_fill_mm=[]

list_d1=[]
list_d2=[]
list_centreline_laa=[]
list_centreline_la=[]
list_curvature=[]
list_bending = []
list_gi =[] 
big = 1000000000000

for i in range(len(LAA_id)):
    #only select patients with same state (systole/diastole) 
    if df1.Systole[i] == flag_systole:

        if df1.Patient_ID[i] == case_number: 
            case_i = i
            print('-------------')
            if df1.Systole[i] == 1:
                m = 'Sytole'
            else:
                m = 'Diastole'
            print('Input patient data: Sys/Dias:',m,'; LA_volume_mm',df1.LA_Vol_mm[i],';  LAA_volume_mm',  df1.LAA_Vol_mm[i],';  ostium_mean_D',  (df1.D1[i] + df1.D2[i]) / 2,';  LAA_centreline_length',  df1.LAA_centreline_length[i],';  LA_centreline_length',  df1.LA_centreline_length[i], '; mean_curvature:', df1.mean_curvature[i],'; Bending angle (deg):', df1.Bending[i],'; Gyrification index (GI):', df1.gi_area[i]   )
            
            #make sure the most similar patient is not the input patient
            list_vol_la_mm.append(big)
            list_vol_laa_fill_mm.append(big)
            list_centreline_laa.append(big)
            list_centreline_la.append(big)
            list_curvature.append(big)
            list_bending.append(big)
            list_gi.append(big)
            list_case.append(df1.Patient_ID[i])
        
        else:

            list_case.append(df1.Patient_ID[i])
            list_vol_la_mm.append(df1.LA_Vol_mm[i])
            list_vol_laa_fill_mm.append(df1.LAA_Vol_mm[i])
            list_d1.append(df1.D1[i])
            list_d2.append(df1.D2[i])
            list_centreline_laa.append(df1.LAA_centreline_length[i])
            list_centreline_la.append(df1.LA_centreline_length[i])
            list_curvature.append(df1.mean_curvature[i])
            list_bending.append(df1.Bending[i])
            list_gi.append(df1.gi_area[i])

print('-------------')
print('Computing closest patient ...')
#LA parameter
dif_la = abs(list_vol_la_mm - df1.LA_Vol_mm[case_i]) 
dif_la_sort = np.array(dif_la)
dif_la = np.array(dif_la)
dif_la_sort.sort()
norm_dif_la = (dif_la- np.min(dif_la))/(dif_la_sort[-2]-np.min(dif_la))

#LAA parameter
dif_laa_fill = abs(list_vol_laa_fill_mm - df1.LAA_Vol_mm[case_i])
dif_la_fill_sort = np.array(dif_laa_fill)
dif_la_fill = np.array(dif_laa_fill)
dif_la_fill_sort.sort()
norm_dif_laa_fill= (dif_laa_fill - np.min(dif_laa_fill))/(dif_la_fill_sort[-2]-np.min(dif_laa_fill))

#curvature parameter
dif_curv = abs(list_curvature - df1.mean_curvature[case_i])
dif_curv_sort = np.array(dif_curv)
dif_curv = np.array(dif_curv)
dif_curv_sort.sort()
norm_dif_curv= (dif_curv - np.min(dif_curv))/(dif_curv_sort[-2]-np.min(dif_curv))

#bending parameter
dif_bend = abs(list_bending - df1.Bending[case_i])
dif_bend_sort = np.array(dif_bend)
dif_bend = np.array(dif_bend)
dif_bend_sort.sort()
norm_dif_bend= (dif_bend - np.min(dif_bend))/(dif_bend_sort[-2]-np.min(dif_bend))

#gi parameter
dif_gi = abs(list_gi - df1.gi_area[case_i])
dif_gi_sort = np.array(dif_gi)
dif_gi = np.array(dif_gi)
dif_gi_sort.sort()
norm_dif_gi= (dif_gi - np.min(dif_gi))/(dif_gi_sort[-2]-np.min(dif_gi))

#ostium diameter parameter
list_diferencias_meand = []
media_1 = (df1.D1[case_i] + df1.D2[case_i]) / 2
for i in range(len(LAA_id)):
    if df1.Systole[i] == flag_systole:
        if str(df1.Patient_ID[i]) == str(case_number): 
            list_diferencias_meand.append(10000000000000)
            continue
        media_2  = (df1.D1[i] + df1.D2[i]) / 2
        dif_d = abs(media_1 - media_2)
        list_diferencias_meand.append(dif_d)

norm_dif_d = np.array(list_diferencias_meand)
norm_dif_d_sort = np.array(list_diferencias_meand)
norm_dif_d_sort.sort()
norm_dif_d2 = (norm_dif_d  - np.min(norm_dif_d ))/(norm_dif_d_sort[-2] - np.min(norm_dif_d))

#length laa centerline parameter
dif_laa_centreline = abs(list_centreline_laa - df1.LAA_centreline_length[case_i]) 
dif_laa_centreline = np.array(dif_laa_centreline)
dif_laa_centreline_sort = np.array(dif_laa_centreline)
dif_laa_centreline_sort.sort()
norm_dif_laa_centreline = (dif_laa_centreline - np.min(dif_laa_centreline))/(dif_laa_centreline_sort[-2]-np.min(dif_laa_centreline))

#length la centreline parameter
dif_la_centreline = abs(list_centreline_la - df1.LA_centreline_length[case_i]) 
dif_la_centreline = np.array(dif_la_centreline)
dif_la_centreline_sort = np.array(dif_la_centreline)
dif_la_centreline_sort.sort()
norm_dif_la_centreline = (dif_la_centreline - np.min(dif_la_centreline))/(dif_la_centreline_sort[-2]-np.min(dif_la_centreline))


#Ecuacion modelo --------------------------
norm_suma_la_laa_fill =  norm_dif_laa_centreline + norm_dif_d2 + norm_dif_laa_fill + norm_dif_la + norm_dif_la_centreline +  norm_dif_bend + norm_dif_curv + norm_dif_gi
#----------------------------------------------------------

suma_list2 = norm_suma_la_laa_fill.tolist()
mini2 = min(suma_list2)

index_mini2 = suma_list2.index(mini2)
case_selected = list_case[index_mini2]
id_selected = list_LAA_id.index(case_selected)

ras2lps_output = df1.RAS2LPS[id_selected]

print('-------------')
print('Most similar case: ', str(int(case_selected)),'. Sys/Dias:',m, 'LA_volume_mm',list_vol_la_mm[index_mini2],';  LAA_volume_mm',  list_vol_laa_fill_mm[index_mini2],';  ostium_mean_D',   (df1.D1[id_selected] + df1.D2[id_selected]) / 2,';  LAA_centreline_length',  df1.LAA_centreline_length[id_selected],';  LA_centreline_length',  df1.LA_centreline_length[id_selected],'; mean_curvature',  df1.mean_curvature[id_selected],'; Bending angle (deg)',  df1.Bending[id_selected],'; Gyrification index (GI):', df1.gi_area[id_selected] )
print('-------------')

#Visualize the results

##LA meshes viz---------

input_LA_mesh = readstl('/Users/martasaiz/Downloads/casos_remesh_f/case_'+str(case_number)+'.stl')
similar_LA_mesh = readstl('/Users/martasaiz/Downloads/casos_remesh_f/case_' + str(int(case_selected))+'.stl')

cent1 = estimate_center(input_LA_mesh)
cent2 = estimate_center(similar_LA_mesh)

#translate similar mesh
t = vtk.vtkTransform()
t.PostMultiply()
t.Translate(cent1-cent2)
tf = vtk.vtkTransformPolyDataFilter()
tf.SetTransform(t)
tf.SetInputData(similar_LA_mesh)
tf.Update()
ts_mesh2 = tf.GetOutput()

#apply alignement 

sourceOBBTree = vtk.vtkOBBTree()
sourceOBBTree.SetDataSet(input_LA_mesh)
sourceOBBTree.SetMaxLevel(1)
sourceOBBTree.BuildLocator()

sourceLandmarks = vtk.vtkPolyData()
sourceOBBTree.GenerateRepresentation(0, sourceLandmarks)

targetOBBTree = vtk.vtkOBBTree()
targetOBBTree.SetDataSet(ts_mesh2)
targetOBBTree.SetMaxLevel(1)
targetOBBTree.BuildLocator()

targetLandmarks = vtk.vtkPolyData()
targetOBBTree.GenerateRepresentation(0, targetLandmarks)

print('Computing...')
angle_step = 10
angle_int = int(360/angle_step)
list_dist_final = []
list_align_rot = []
list_align_centreline = []
list_final_point_align = []
list_rotate_angle = []
for n in range(angle_int):
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.Translate(-cent1)
    t.RotateZ(n*10)
    list_rotate_angle.append(n*10)
    #print('Rotation angle', n*10)
    t.Translate(cent1)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetTransform(t)
    tf.SetInputData(ts_mesh2)
    tf.Update()
    aligned_mesh3 = tf.GetOutput()
    list_align_rot.append(aligned_mesh3)
    distance = vtk.vtkHausdorffDistancePointSetFilter()
    distance.SetInputData(0, input_LA_mesh)
    distance.SetInputData(1, aligned_mesh3)
    distance.Update()
    distanceMesh_cent = distance.GetOutput(0).GetFieldData().GetArray('HausdorffDistance').GetComponent(0, 0)
    list_dist_final.append(distanceMesh_cent)


index_min = list_dist_final.index(min(list_dist_final))
mesh_min = list_align_rot[index_min]
print('LA HD Align', min(list_dist_final))
rend1 = vtk.vtkRenderer()
vtk_show_centreline(rend1, input_LA_mesh, mesh_min, 1000, 1000)


##LAA meshes viz-------------------- (align to z axis by ostium plane)
list_ids = [id, id_selected]
input_LAA_mesh = readstl('/Users/martasaiz/Downloads/LAA_mesh_stl/LAA_'+str(case_number)+'.stl')
similar_LAA_mesh = readstl('/Users/martasaiz/Downloads/LAA_mesh_stl/LAA_'+str(int(case_selected))+'.stl')
edge1 = extractboundaryedge(input_LAA_mesh)
edge2 = extractboundaryedge(similar_LAA_mesh)
cent_1 = estimate_center(edge1)
cent_2 = estimate_center(edge2)

list_mesh = [input_LAA_mesh,similar_LAA_mesh]
list_centro = [cent_1, cent_2]
list_edges = [edge1, edge2]

#align both meshes to z axis 
list_aligned = []
list_cent = []
list_endpoint_rot = []
for i in range(2):

    point1 = list_centro[i]
    point2 = list_edges[i].GetPoint(10)
    point3 = list_edges[i].GetPoint(100)

    AB = np.subtract(point1, point2)
    AB2 = normalizevector(AB)
    AC = np.subtract(point1, point3)
    AC2 = normalizevector(AC)

    normal = np.cross(AB2, AC2)
    endpoint = np.array((df1.end_point_x[list_ids[i]],df1.end_point_y[list_ids[i]],df1.end_point_z[list_ids[i]]))
    
    normal_align = np.array([0,0,1])
    path_rotated ='/Users/martasaiz/Documents/Carpetas_Marta/TESIS/FASE1/Extraccion_param/alvaro/clip_z_align/LAA_'+str(LAA_id[list_ids[i]])+'.vtk'
    angl_rotated, direction, rotated, matrix = rotate_angle(list_mesh[i], path_rotated, normal, normal_align, list_centro[i], int(LAA_id[list_ids[i]]))
    endpoint2 = np.array([endpoint[0],endpoint[1],endpoint[2],1])
    endpoint_rotated = matrix.dot(endpoint2)
    endpoint_rotated = endpoint_rotated[0:3]

    cent2 = np.array([list_centro[i][0],list_centro[i][1],list_centro[i][2],1])
    cent_rotated = matrix.dot(cent2)
    cent_rotated = cent_rotated[0:3]

    list_aligned.append(rotated)
    list_cent.append(cent_rotated)
    list_endpoint_rot.append(endpoint_rotated)

#find best alignement by rotating around z axis 
t = vtk.vtkTransform()
t.PostMultiply()
t.Translate(list_cent[0]-list_cent[1])
tf = vtk.vtkTransformPolyDataFilter()
tf.SetTransform(t)
tf.SetInputData(list_aligned[1])
tf.Update()
aligned_mesh2 = tf.GetOutput()


angle_step = 10
angle_int = int(360/angle_step)
list_dist_final = []
list_align_rot = []
list_align_centreline = []
list_final_point_align = []
print('performing LAA align...')
list_rotate_angle = []
for n in range(angle_int):
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.Translate(-list_cent[0])
    t.RotateZ(n*10)
    list_rotate_angle.append(n*10)
    #print('Rotation angle', n*10)
    t.Translate(list_cent[0])
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetTransform(t)
    tf.SetInputData(aligned_mesh2)
    tf.Update()
    aligned_mesh3 = tf.GetOutput()
    list_align_rot.append(aligned_mesh3)

    t.Update()
    matrix = t.GetMatrix()
    #print(matrix)
    mp= np.zeros(shape=(4, 4))
    for i in range(0, 4):
        for j in range(0, 4):
            mp[i,j] = matrix.GetElement(i, j)

    endpoint_a2 = np.array([list_endpoint_rot[1][0],list_endpoint_rot[1][1],list_endpoint_rot[1][2],1])
    endpoint_a = mp.dot(endpoint_a2)
    endpoint_a = endpoint_a[0:3]

    dist=np.linalg.norm(np.array(list_endpoint_rot[0])-np.array(endpoint_a))
    list_dist_final.append(dist)

index_min = list_dist_final.index(min(list_dist_final))
mesh_min = list_align_rot[index_min]

rend1 = vtk.vtkRenderer()
vtk_show_centreline(rend1,list_aligned[0],mesh_min,1000,1000)


#compute Haussdorf distance
distance = vtk.vtkHausdorffDistancePointSetFilter()
distance.SetInputData(0, list_aligned[0])
distance.SetInputData(1, mesh_min)
distance.Update()
distanceMesh = distance.GetOutput(0).GetFieldData().GetArray('HausdorffDistance').GetComponent(0, 0)
print('Haussdorf distance LAA meshes:',distanceMesh)