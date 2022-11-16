import pandas as pd
import numpy as np
import pandas as pd
from sympy import Q
from functions_script2 import *
import pyvista as pv
import tetgen
import time

#extract morphological parameters from mesh and cut LAA good


frames = []
list_times = []
#cargo el excel generado en Script2.py (necesito el endpoint y el ostium_pred_rotado)
input_excel ='/Users/martasaiz/Documents/Carpetas_Marta/TESIS/FASE1/Extraccion_param/Codigos_Marta/excels/endpoint_script2_prueba4.xlsx'         
input_excel_2 = '/Users/martasaiz/Documents/Carpetas_Marta/TESIS/FASE1/Extraccion_param/Codigos_Marta/excels/ostium_plane.xlsx'    


df4 = pd.read_excel(input_excel, engine='openpyxl',)
df5 = pd.read_excel(input_excel_2, engine='openpyxl',)

list_caso_n4 = []
for caso in df4.Patient_ID:
    list_caso_n4.append(int(caso))

list_caso_n5 = []
for caso in df5.Patient_ID:
    list_caso_n5.append(int(caso))


for n,case in zip(range(len(list_caso_n4)),list_caso_n4):

    case_num2 = int(case)

    print('Case',case_num2)
    if case_num2 == 1:
        tic = time.perf_counter()
        case_str = 'case_'+str(case_num2)+'.stl'
        #load LA mesh
        input_path = '/Users/martasaiz/Downloads/rotated_raw3/LA_' + str(case_num2)+ '.stl'
        mesh = readstl(input_path)
        cent = estimate_center(mesh)
        #load LAA mesh
        input_path = '/Users/martasaiz/Downloads/LAA_good_' + str(case_num2)+ '.stl'
        mesh_laa = readstl(input_path)
        #load good ostium normal and origin
        ind = list_caso_n5.index(case_num2)
        origin = np.array((df5.ost_pos_x[ind],df5.ost_pos_y[ind],df5.ost_pos_z[ind]))
        normal = np.array((df5.ost_or_x[ind],df5.ost_or_y[ind],df5.ost_or_z[ind]))
        #load centreline
        input_centerline = '/Users/martasaiz/Downloads/centerline_rotate_'+str(case_num2)+'.vtk'
        centreline = readvtk(input_centerline)

        edges = extractboundaryedge(mesh_laa)

        #compute ostium diameters
        edge_f = '/Users/martasaiz/Downloads/p.json'
        d2_min, d1_complete, mean_p, p0d2, p0d1, p1d1, d2_complete, p1d2,LAA_edge,p0,boundaryIds = get_d1d2alt(edges, edge_f,origin, show_points=False) 

        
        cent = estimate_center(edges)
        #compute h and h_theta
        normal_align = np.array([0, 0, 1])  #eje z
        path_rotated = '/Users/martasaiz/Downloads/rotate2_'+str(case_num2)+'.vtk'
        rotated, matrix = rotate_angle(mesh_laa, normal, normal_align, cent, path_rotated)
        
        
        #print ("Computing h and h_theta...")
        h, intersect, h_comp = get_halt(path_rotated, cent)  #Get H considering the LAA center in the base (Clipped), morphologicFunctions.py

        print ("H (mm) :", h)

        start = np.add(cent, intersect[0])
        p0 = np.array([start[0] / 2, start[1] / 2, start[2] / 2])

        h_theta, thpoint, M = get_hthetaalt(path_rotated, p0, h, cent, intersect[0]) 
        
        print ("H_theta (mm): ", h_theta)
        print( "M (tortuosity): ", M)
    
        #renderer_h = vtk.vtkRenderer()
        #vtk_show(renderer_h, rotated, 1000, 1000, actor_h, actor_inter, actor_ori, actor_thetaori, actor_htheta,actor_thetainter)
        #vtk_show(renderer_h, rotated, 1000, 1000, actor_thetainter)


        # COMPUTE VOLUME/AREA LA 
        polygonProperties = vtk.vtkMassProperties()
        polygonProperties.SetInputData(mesh)
        polygonProperties.Update()
        LA_volume_mm = polygonProperties.GetVolume()
        LA_volume_ml = LA_volume_mm / 1000
        LA_area = polygonProperties.GetSurfaceArea()

        # COMPUTE VOLUME/AREA LAA
        polygonProperties = vtk.vtkMassProperties()
        polygonProperties.SetInputData(mesh_laa)
        polygonProperties.Update()
        LAA_volume_mm = polygonProperties.GetVolume()
        LAA_volume_ml = LAA_volume_mm / 1000
        LAA_area = polygonProperties.GetSurfaceArea()


        length_la = df5.length_la[ind]
        length_laa = df5.length_laa[ind]
        bending = df5.bending[ind]

        raw_data = {'Patient_ID': case_num2, 'D1_ostium':[d1_complete], 'D2_ostium': [d2_complete],'LA_centreline_length': [length_la], 'LAA_centreline_length': [length_laa], 'bending': [bending], 'LA_vol': [LA_volume_ml], 'LAA_vol': [LAA_volume_ml]}
        columns=['Case', 'D1_ostium','D2_ostium','LA_centreline_length','LAA_centreline_length']
        df = pd.DataFrame(raw_data, columns=['Case', 'D1_ostium','D2_ostium','LA_centreline_length','LAA_centreline_length','bending','LA_vol','LAA_vol'], )
        frames.append(df)
        
        #print(case)

        result = pd.concat(frames)
        ##        # All results are stored in this path
        out_fcsv = '/Users/martasaiz/Documents/Carpetas_Marta/TESIS/FASE1/Extraccion_param/Codigos_Marta/excels/parametros_extraidos.xlsx'         
        result.to_excel(out_fcsv)