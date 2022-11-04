import pandas as pd
import numpy as np
import pandas as pd
from sympy import Q
from functions_script2 import *
import pyvista as pv
import tetgen
import time

#A partir de las predicciones de ostium (RL) hago un raw cut y aplico heat equation para calcular endpoint. Con el endpoint calculo el centerline. 

#input: stl LA mesh 
#input_excel: ostium position world coordinates
#casos con excepciones: 195, 177, 92,176, 214,29,7,182, 163, 169,172, 174, 176, 178, 214, 182,29,33, 46

frames = []
list_times = []
input_excel = '/Users/martasaiz/Documents/Carpetas_Marta/TESIS/FASE1/Extraccion_param/Codigos_Marta/excels/ostium_pred_RL_good.xlsx'
df4 = pd.read_excel(input_excel, engine='openpyxl',)

list_caso_n4 = []
for caso in df4.Patient_ID:
    list_caso_n4.append(int(caso))



target_reduction_list = [0.4, 0.5, 0.6, 0.7]


for n,case in zip(range(len(list_caso_n4)),list_caso_n4):

    case_num2 = int(case)

    print('Case',case_num2)
    #voy a ignorar estos casos porque el ostium_prediction es malo...
    #if case_num2 == 195 or case_num2 == 177 or case_num2 == 92: 
    #        continue

    if case_num2 == 1:
        tic = time.perf_counter()

        case_str = 'case_'+str(case_num2)+'.stl'
        input_path = '/Users/martasaiz/Downloads/casos_remesh_f/' + case_str

        mesh = readstl(input_path)
        file = extractlargestregion(mesh) #in case errors of segmentation
        center_mesh = estimate_center(mesh)

        ostium_coord_pred = np.array((df4.ostium_pos_x[n],df4.ostium_pos_y[n],df4.ostium_pos_z[n]))
        #print('ostium_coord_pred', ostium_coord_pred)
        #actor_p5 = addpoint_show(ostium_coord_pred, color=[1., 0., 0.])
        #rend1 = vtk.vtkRenderer()
        #vtk_show(rend1, mesh, 1000, 1000, actor_p5)

        ################################ 1. CUT LAA RAW ##################################

        #find initial normal vector
        AB = np.subtract(ostium_coord_pred,center_mesh)
        AB2 = normalizevector(AB)
        AB2 = np.array(AB2)
        n=100
        P4 = center_mesh + n * AB2

        normal = AB2
        origin = ostium_coord_pred

        #actor_p5 = addline_show(center_mesh, P4, color=[1., 0., 0.])
        #rend1 = vtk.vtkRenderer()
        #vtk_show(rend1, file, 1000, 1000, actor_p5)

        #rotate to z axis
        normal_align = np.array([0, 0, 1])  #eje z
        path_rotated = None
        rotated, matrix = rotate_angle(file, normal, normal_align, center_mesh, path_rotated)
        cent2 = np.array([origin[0],origin[1],origin[2],1])
        cent_rotated = matrix.dot(cent2)
        origin_rotated = cent_rotated[0:3]

        writestl(rotated,'/Users/martasaiz/Downloads/rotated_raw3/LA_' + str(case_num2)+ '.stl')

        #find normal and clip
        print('Rotated')
        normal_opt, centroid_opt = brute_force_perturbation(rotated,normal_align, origin_rotated,0.4,0, no=20)
    

        origin = centroid_opt
        normal = normal_opt
        file = rotated

        clip, p0 = clip_LAA(file, origin, normal, viz = False)

        # print('raw clip')
        # rendm = vtk.vtkRenderer()
        # vtk_show(rendm,clip, 1000, 1000)   

        #save mesh, .stl 
        path_clip_bueno_stl= '/Users/martasaiz/Downloads/cut_raw3/LAA_ost_' + str(case_num2)+ '.stl'
        writestl(clip, path_clip_bueno_stl)

        #Fix mesh

        fix_mesh(path_clip_bueno_stl)
        #    print('repaired clip')
        
        repaired = readstl(path_clip_bueno_stl)


        # ################################ 2. APPLY HEAT EQUATION FOR ENDPOINT DETECTION ##################################


        #lista_excep = ['169','174', '176', '178', '214', '182','29', '33', '46']

        #if str(case_num2) in lista_excep:
        #        m = 0.7
        #else:
        m = 0.6

        #if case_num2 == 172:
        #    m = 0.4

        mesh_LAA = pv.read(path_clip_bueno_stl)
        smooth = mesh_LAA.smooth(n_iter=1)
        decimated = mesh_LAA.decimate(m)
        decimated = decimated.decimate(m)
        print("Constructing volumetric mesh...")
        tet = tetgen.TetGen(decimated)
        tet.make_manifold()
        tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
        grid = tet.grid

        p0_np = np.array(p0)
        print(p0_np)

        ostium, LAA_wall, ostium_centre = detectOstium(decimated,p0_np,normal,origin)
        print("Defining boundary conditions")
        boundaryConditions, boundaryPoints = defineBoundaryConditions(grid, ostium, LAA_wall)
        #volumetricMeshPath = '/Users/martasaiz/Downloads/processed_vol/LA_' + str(case_num2)+'_processed_vol.vtk'
        volumetricMeshPath = '/Users/martasaiz/Downloads/vol_processed3/LA_' + str(case_num2)+'_processed_vol.vtk'
        grid.save(volumetricMeshPath)
                
        meshVTK = readUnstructuredGridVTK(volumetricMeshPath)
        print("Calculating the end point of the centreline, solving the Laplace Equation...")
        result = solveLaplaceEquationTetrahedral(volumetricMeshPath,decimated , boundaryPoints, boundaryConditions[:,0])
        #save mesh with heat equation results
        mesh_with_scalar = add_scalar(meshVTK, result, name = 'heat', domain = 'point')
        writeUnstructuredGridVTK(volumetricMeshPath,mesh_with_scalar)

        end_point = detectEndPointOfCentreline(result,grid)
  
            

        toc = time.perf_counter()
        print('Time: ', toc-tic)
        tiempo = toc-tic
        list_times.append(tiempo)

        print("Writing results...")
        raw_data = {'Patient_ID': [case_num2], 'end_point_x': [end_point[0]], 'end_point_y': [end_point[1]],'end_point_z': [end_point[2]], 'rotpred_point_x': [origin_rotated[0]], 'rotpred_point_y': [origin_rotated[1]],'rotpred_point_z': [origin_rotated[2]]}
        df = pd.DataFrame(raw_data, columns=['Patient_ID', 'end_point_x', 'end_point_y','end_point_z', 'rotpred_point_x','rotpred_point_y','rotpred_point_z'])

        frames.append(df)
        #print(case)

        result = pd.concat(frames)
        ##        # All results are stored in this path
        #out_fcsv = '/Users/martasaiz/Documents/Carpetas_Marta/TESIS/FASE1/Extraccion_param/Codigos_Marta/excels/endpoint_script2_prueba4.xlsx'         
        #result.to_excel(out_fcsv)


        print(list_times)
        print('mean time',np.mean(np.array(list_times)))

            ################################ 3. MAKE CENTERLINE ##################################

        centreline = vmtkcenterlines(mesh, np.array(center_mesh), np.array(end_point), 1)
        clspacing = 0.4
        cl = vmtkcenterlineresampling(centreline, clspacing)

        rend1 = vtk.vtkRenderer()
        vtk_show_centreline(rend1, mesh,cl,1000, 1000)
        writevtp(cl, '/Users/martasaiz/Documents/Carpetas_Marta/TESIS/FASE1/Extraccion_param/alvaro/cut/processed/centreline.vtp')
       










