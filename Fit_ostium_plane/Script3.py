import pandas as pd
import numpy as np
import pandas as pd
from sympy import Q
from functions_script2 import *
import pyvista as pv
import tetgen
import time

#En este script corto bien el LAA, despues de estimar el ostium_plane con el centerline y el ostium prediction

frames = []
list_times = []
#cargo el excel generado en Script2.py (necesito el endpoint y el ostium_pred_rotado)
input_excel ='/Users/martasaiz/Documents/Carpetas_Marta/TESIS/FASE1/Extraccion_param/Codigos_Marta/excels/endpoint_script2_prueba4.xlsx'         

df4 = pd.read_excel(input_excel, engine='openpyxl',)

list_caso_n4 = []
for caso in df4.Patient_ID:
    list_caso_n4.append(int(caso))

for n,case in zip(range(len(list_caso_n4)),list_caso_n4):

    case_num2 = int(case)

    print('Case',case_num2)
    if case_num2 == 1:
        tic = time.perf_counter()
        case_str = 'case_'+str(case_num2)+'.stl'

        #cargo la malla rotada
        input_path = '/Users/martasaiz/Downloads/rotated_raw3/LA_' + str(case_num2)+ '.stl'
        mesh = readstl(input_path)
        file = extractlargestregion(mesh) #in case errors of segmentation
        center_mesh = estimate_center(mesh)

        #cargo el centerline
        input_centerline = '/Users/martasaiz/Downloads/centreline.vtp'
        centreline = readvtp(input_centerline)

        rend = vtk.vtkRenderer()
        vtk_show_centreline(rend,mesh, centreline, 1000,1000)

        ostium_coord_pred = np.array((df4.rotpred_point_x[n],df4.rotpred_point_y[n],df4.rotpred_point_z[n]))
        endpoint = np.array((df4.end_point_x[n],df4.end_point_y[n],df4.end_point_z[n]))

        ## GET NORMAL FROM AUTOMATIC CENTRELINE
        npoints = centreline.GetNumberOfPoints()
        CL_points = []
        for i in range(npoints):
            CL_points.append(centreline.GetPoint(i))
        distances = calculateDistanceToOstium(CL_points, ostium_coord_pred)  #calcula la distancia de cada punto del centreline a la coordenada de Daniel, entonces una distancia debe de ser 0 no?
        start = np.argmin(distances) 
        origin = CL_points[start]
        last = CL_points[0]

        try:                    
            previous_point = CL_points[start-5]
            next_point = CL_points[start+5]
        except:
            try:
                ult = len(CL_points)
                dif = ult-start
                previous_point = CL_points[start-(dif-1)]
                next_point = CL_points[start+(dif+1)]
            except:
                previous_point = CL_points[start-5]
                next_point = CL_points[start]

        vector_x = next_point[0] - previous_point[0]
        vector_y = next_point[1] - previous_point[1]
        vector_z = next_point[2] - previous_point[2]
        module = math.sqrt(math.pow(vector_x, 2) + math.pow(vector_y, 2) + math.pow(vector_z, 2))
        
        normal = [vector_x/module,vector_y/module,vector_z/module]
        origin = ostium_coord_pred

        #rotate to z axis
        normal_align = np.array([0, 0, 1])  #eje z
        path_rotated = '/Users/martasaiz/Downloads/rotate_'+str(case_num2)+'.vtk'
        rotated, matrix = rotate_angle(file, normal, normal_align, center_mesh, path_rotated)
        cent2 = np.array([origin[0],origin[1],origin[2],1])
        cent_rotated = matrix.dot(cent2)
        origin_rotated = cent_rotated[0:3]

        #also rotate centerline to z axis 
        path_rotated = '/Users/martasaiz/Downloads/centerline_rotate_'+str(case_num2)+'.vtk'
        rotated_c, matrix_c = rotate_angle(centreline, normal, normal_align, center_mesh, path_rotated)

        print('Rotated')
        normal_opt, centroid_opt = brute_force_perturbation(rotated,normal_align, origin_rotated,0.4,0, no=20)

        origin = centroid_opt
        normal = normal_opt
        file = rotated

        clip, p0 = clip_LAA(file, origin, normal, viz = False)

         #save mesh, .stl, este es el LAA bueno
        path_clip_bueno_stl= '/Users/martasaiz/Downloads/LAA_good_' + str(case_num2)+ '.stl'
        writestl(clip, path_clip_bueno_stl)

        rend1 = vtk.vtkRenderer()
        vtk_show(rend1, clip,1000,1000)


        #compute centerline id close to ostium 
        npoints = rotated_c.GetNumberOfPoints()
        CL_points_rotated = []
        for i in range(npoints):
            CL_points_rotated.append(rotated_c.GetPoint(i))
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


        #save final ostium plane origin and normal, and centerline lengths
        print("Writing results...")
        raw_data = {'Patient_ID': [case_num2], 'ost_pos_x': [origin[0]], 'ost_pos_y': [origin[1]],'ost_pos_z': [origin[2]], 'ost_or_x': [normal[0]], 'ost_or_y': [normal[1]],'ost_or_z': [normal[2]], 'length_la': [length_la],'length_laa': [length_laa]}
        df = pd.DataFrame(raw_data, columns=['Patient_ID', 'ost_pos_x', 'ost_pos_y','ost_pos_z', 'ost_or_x','ost_or_y','ost_or_z','length_la', 'length_laa'])

        frames.append(df)
        #print(case)

        result = pd.concat(frames)
        ##        # All results are stored in this path
        out_fcsv = '/Users/martasaiz/Documents/Carpetas_Marta/TESIS/FASE1/Extraccion_param/Codigos_Marta/excels/ostium_plane.xlsx'         
        result.to_excel(out_fcsv)