#Visualizar prediccion en CT, luego convertir voxel a world coordinates (CUIDADO CON SISTEMAS RAS Y LPS)


# Algunos casos de Burdeos están en LPS y otros en RAS (tengo que cargar el excel df4 para coger la info.)


import os
import nibabel as nib
import numpy as np 
import pandas as pd 
import SimpleITK as sitk
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input_image', type=str, help='path image_noflip nii.gz')
parser.add_argument('--input_csv', type=str, help='path folder csv')
parser.add_argument('--output', type=str, help='path output folder')
parser.add_argument('--RAS2LPS', type=int, help='1 if image in RAS, 0 if in LPS',
                        default=0)

args = parser.parse_args()
csv_path = args.input_csv
input_dir_path = args.input_image
RAS2LPS = args.RAS2LPS
output_path = args.output


df3 = pd.read_csv(csv_path,)

#df3 = pd.read_excel('/Users/martasaiz/Downloads/results_csv_tot.xlsx',engine = 'openpyxl'  
# )

#image_nf_path = '/Users/martasaiz/Downloads/case_305/image_noflip'
#image_nf_path = '/Users/martasaiz/Downloads/burdeos_101_0000.nii.gz'

frames=[]
list_id = []

#dirs = os.listdir(image_nf_path)

#epi_img = nib.load(image_nf_path + '/'+ dirs[0])

dirs = os.listdir(input_dir_path)
image_nf_path = input_dir_path + '/' + dirs[0]

epi_img = nib.load(image_nf_path)
matrix_transf = epi_img.affine

pixel_spacing = np.array(matrix_transf).diagonal()
pixel_spacing  = abs(pixel_spacing[0:3])

origin_ima = matrix_transf[:,3]
origin_ima  = np.array((-origin_ima[0], -origin_ima[1], origin_ima[2]))
#origin_ima  = np.array((origin_ima[0], origin_ima[1], origin_ima[2]))
print(origin_ima)
print(pixel_spacing)

image = sitk.Image([256, 256, 305], sitk.sitkVectorFloat32)
image.SetOrigin((origin_ima[0],origin_ima[1],origin_ima[2]))
image.SetSpacing((pixel_spacing[0], pixel_spacing[1], pixel_spacing[2]))

pred_x = df3.final_coordinates_x[0]
pred_y = df3.final_coordinates_y[0]
pred_z = df3.final_coordinates_z[0]

print(df3.filename[0])

pred = np.array((float(pred_x),float(-(256-pred_y)),float(pred_z))) #ESTE ES EL BUENO
#pred = np.array((float(pred_x),float(-(256-pred_y)),float(-(305-pred_z))))
xyz2_pred = image.TransformContinuousIndexToPhysicalPoint(pred)


if RAS2LPS ==  1:
  print('yes ras2lps')
  xyz2_pred = np.array((-xyz2_pred[0],-xyz2_pred[1],xyz2_pred[2]))

print('xyz_transformed_pred2',xyz2_pred)


raw_data = {'Case': 1, 'ostium_pred_x':[xyz2_pred[0]], 'ostium_pred_y': [xyz2_pred[1]],'ostium_pred_z': [xyz2_pred[2]]}
columns=['Case', 'ostium_pred_x','ostium_pred_y','ostium_pred_z']
df = pd.DataFrame(raw_data, columns=['Case', 'ostium_pred_x','ostium_pred_y','ostium_pred_z'])
frames.append(df)

if not os.path.exists(output_path):
  os.mkdir(output_path)

reader = sitk.ImageFileReader()
reader.SetImageIO("NiftiImageIO")
#reader.SetFileName(image_nf_path + '/'+ dirs[0])
reader.SetFileName(image_nf_path)
image = reader.Execute()

#pred
image2 = image[:,:,int(pred_z)]
image3  = sitk.GetArrayViewFromImage(image2)
image4 = np.flipud(image3)

fig = plt.figure()
plt.imshow(image4)
plt.plot(abs(int(pred_x)),abs(int(pred_y)),'r+')
plt.savefig(output_path + '/pred_plot.png')
plt.close(fig)

#image4 = np.flipud(image3)

# image2 = image[:,:,int(target_z)]
# image3  = sitk.GetArrayViewFromImage(image2)
# image4 = np.flipud(image3)
# plt.imshow(image4)
# plt.plot(int(target_x),int(target_y),'r+')
# #plt.plot(abs(int(pred_x[0])),256-abs(int(pred_y[1])),'b+')
# plt.show()


#if n == 130:
result = pd.concat(frames)
            # All results are stored in this path
out_fcsv = output_path + '/ostium_pred.xlsx'
            
result.to_excel(out_fcsv)

# print ("Finishing...")
#else:
#     pass