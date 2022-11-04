#Visualizar prediccion en CT, luego convertir voxel a world coordinates (CUIDADO CON SISTEMAS RAS Y LPS)


# Algunos casos de Burdeos están en LPS y otros en RAS (tengo que cargar el excel df4 para coger la info.)


import os
import nibabel as nib
import numpy as np 
import pandas as pd 
import SimpleITK as sitk
import matplotlib.pyplot as plt


df3 = pd.read_excel('/content/drive/MyDrive/Automatic_ostium_detection/results_csv_tot.xlsx',
     engine='openpyxl',
)

df4 = pd.read_excel('/content/drive/MyDrive/Automatic_ostium_detection/Marta_Excel_final_final.xlsx',
     engine='openpyxl',
)

frames=[]
list_id = []

print(df3.filename)
for case in df3.filename[0:131]:

  if case == 'nan':
    continue
  else:
    num = int(case.split('_')[1])
    list_id.append(num)

list_pred_xl = []
for caso in df4.Patient_ID:
    list_pred_xl.append(int(caso))


for case,n in zip(os.listdir('/content/drive/MyDrive/Automatic_ostium_detection/RL_landmark_detection_for_cardiac_applications/Images_segmentation_resampled/'), range(len(os.listdir('/content/drive/MyDrive/Automatic_ostium_detection/RL_landmark_detection_for_cardiac_applications/Images_segmentation_resampled/')))):


    case_num = case.split('.')[0]
    case_num2 = int(case_num.split('_')[1])
    #print('case_num2', case_num2)

    ind = list_id.index(case_num2)
    epi_img = nib.load('/content/drive/MyDrive/Automatic_ostium_detection/RL_landmark_detection_for_cardiac_applications/Images_segmentation_resampled/'+case)
    matrix_transf = epi_img.affine

    ind_pred_xl = list_pred_xl.index(int(case_num2))
    #print(df4.Patient_ID[ind_pred_xl])

    pixel_spacing = np.array(matrix_transf).diagonal()
    pixel_spacing  = abs(pixel_spacing[0:3])

    origin_ima = matrix_transf[:,3]
    origin_ima  = np.array((-origin_ima[0], -origin_ima[1], origin_ima[2]))

    image = sitk.Image([256, 256, 305], sitk.sitkVectorFloat32)
    image.SetOrigin((origin_ima[0],origin_ima[1],origin_ima[2]))
    image.SetSpacing((pixel_spacing[0], pixel_spacing[1], pixel_spacing[2]))

    pred_x = df3.final_coordinates_x[ind]
    pred_y = df3.final_coordinates_y[ind]
    pred_z = df3.final_coordinates_z[ind]

    target_x = df3.target_x[ind]
    target_y = df3.target_y[ind]
    target_z = df3.target_z[ind]


    pred = np.array((float(pred_x),float(-(256-pred_y)),float(pred_z)))
    xyz2_pred = image.TransformContinuousIndexToPhysicalPoint(pred)
    print('xyz_transformed_pred',xyz2_pred)

    target = np.array((float(target_x),float(-(256-target_y)),float(target_z)))
    target_pred = image.TransformContinuousIndexToPhysicalPoint(target)

    if df4.RAS2LPS[ind_pred_xl] ==  1:
        print('yes ras2lps')
        xyz2_pred = np.array((-xyz2_pred[0],-xyz2_pred[1],xyz2_pred[2]))
        print('xyz_transformed_pred2',xyz2_pred)
        target_pred2 = np.array((-target_pred[0],-target_pred[1],target_pred[2]))
        print(target_pred2)


    raw_data = {'Case': case, 'ostium_pred_x':[xyz2_pred[0]], 'ostium_pred_y': [xyz2_pred[1]],'ostium_pred_z': [xyz2_pred[2]]}
    columns=['Case', 'ostium_pred_x','ostium_pred_y','ostium_pred_z']
    df = pd.DataFrame(raw_data, columns=['Case', 'ostium_pred_x','ostium_pred_y','ostium_pred_z'])
    frames.append(df)

    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName('/content/drive/MyDrive/Automatic_ostium_detection/RL_landmark_detection_for_cardiac_applications/Images_segmentation_resampled/'+case)
    image = reader.Execute()

    #pred
    image2 = image[:,:,int(pred_z)]
    image3  = sitk.GetArrayViewFromImage(image2)
    image4 = np.flipud(image3)

    plt.imshow(image4)
    plt.plot(abs(int(pred_x)),abs(int(pred_y)),'r+')
    plt.show()

    #image4 = np.flipud(image3)

    image2 = image[:,:,int(target_z)]
    image3  = sitk.GetArrayViewFromImage(image2)
    image4 = np.flipud(image3)
    plt.imshow(image4)
    plt.plot(int(target_x),int(target_y),'r+')
    #plt.plot(abs(int(pred_x[0])),256-abs(int(pred_y[1])),'b+')
    plt.show()


    #if n == 130:
    # result = pd.concat(frames)
                # All results are stored in this path
    # out_fcsv = '/content/drive/MyDrive/Automatic_ostium_detection/ostium_pred_RL_2.xlsx'
                
    # result.to_excel(out_fcsv)

    # print ("Finishing...")
    #else:
    #     pass