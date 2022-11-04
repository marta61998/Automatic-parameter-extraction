import os
import pandas as pd
import shutil
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import shutil
import argparse

from Functions import *

#----
parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str, help='path input image')
parser.add_argument('--output', type=str, help='path output preprocessed image')
parser.add_argument('--landmark_GT', type=int, help='1 if we have ground-truth landmarks, 0 if we do not',
                        default=0)



args = parser.parse_args()

input_path = args.input
output_path = args.output
cur_dir = os.path.dirname(input_path) + '/'


#-------Copy non-repeated CT cases from nifti/pre to /Images_segmentation folder. Comment this if not needed.

#files = os.listdir(cur_dir+'/pre') #folder with repeated patients 

#df = pd.read_excel('/Users/martasaiz/Documents/Carpetas_Marta/TESIS/FASE1/Extraccion_param/Codigos_Marta/excels/Marta_Excel_final_final.xlsx', engine = 'openpyxl')
#list_ID = []

#patients = np.array(df['Patient_ID'])
#new_patients = patients[np.logical_not(np.isnan(patients))]

#for i in new_patients:
#    list_ID.append(int(i))

#if not os.path.exists(cur_dir+'/pre_segmentation'):
#    os.mkdir(cur_dir+'/pre_segmentation')

#for f in files:
#    case1=f.split('.')[0]
#    case2 = int(case1.split('_')[1])
#    if case2 in list_ID:
#        shutil.copy(cur_dir+'/pre/'+f, cur_dir+'/pre_segmentation/'+f )
#------


#Apply resampling function 



#dir_with_img = os.path.join(cur_dir, 'pre_segmentation')
dir = os.listdir(input_path)

list_mal=[]
for sub_dir in dir:
    if sub_dir == '.DS_Store':
        continue

    if sub_dir.startswith('d'):
        continue
    case = int(sub_dir.split('_')[1])
    path_to_sitk_image = os.path.join(input_path,sub_dir)
    print('Reading Image at:', path_to_sitk_image)

    image = sitk.ReadImage(path_to_sitk_image)
    # Calculating new spacing
    img_size = np.asarray(image.GetSize())
    img_spacing = np.asarray(image.GetSpacing())
    my_image_size = np.asarray((256,256,305))
    new_spacing = img_spacing*(img_size/my_image_size)
    print(f'Resampling from {img_size} to {my_image_size}.\nPrevious Spacing {img_spacing}\tNew Spacing {new_spacing}')
    newimage  = resample_image(image, out_spacing=new_spacing, is_label=False)

    #Perform flip 
    newimage2 = newimage[:, ::-1, :]

    if not os.path.exists(cur_dir+'/out1'):
        os.mkdir(cur_dir+'/out1')
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    dest_dir_img = os.path.join(cur_dir, 'out1')
    dest_dir_img2 = os.path.join(output_path)

    #I save the non-flipped image for world to voxel conversion, after I will delete the folder
    outputImageFileName = os.path.join(dest_dir_img,sub_dir)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(outputImageFileName)
    #writer.SetUseCompression(True)
    writer.Execute(newimage)

    #flipped images, input to the network
    outputImageFileName2 = os.path.join(dest_dir_img2,sub_dir)
    writer2 = sitk.ImageFileWriter()
    writer2.SetFileName(outputImageFileName2)
    print(f'Writing to : {outputImageFileName}\n')
    writer2.Execute(newimage2)

name_landmarks = 'landmarks2'
if not os.path.exists(cur_dir+ name_landmarks):
    os.mkdir(cur_dir+ name_landmarks)

if (args.landmark_GT)==1:  #if we have the landmarks GT, i.e., in an excel
    try:
        df = pd.read_excel('/Users/martasaiz/Documents/Carpetas_Marta/TESIS/FASE1/Extraccion_param/Codigos_Marta/excels/Marta_Excel_final_final.xlsx', engine = 'openpyxl')
    except:
        print('ERROR: upload landmarks in excel format')
    list_ID = []
    #print(np.isnan(np.array(df['Patient_ID'])))
    patients = np.array(df['Patient_ID'])
    new_patients = patients[np.logical_not(np.isnan(patients))]

    #df['Patient_ID'].dropna()
    #print(df)
    for i in new_patients:
        list_ID.append(int(i))

    #Create landmarks folder, save voxel landmarks 
  
    #convert ostium world coordinates to voxel 
    image_path = cur_dir + 'out1' #non flipped
    list_x2, list_y2, list_z2, list_num = world2voxel(list_ID,image_path, df) #df is the excel where the gt is stored


else:
    list_x2 = []
    list_y2 = []
    list_z2 = []
    list_num =[] 
    for case,i in zip(os.listdir(output_path),range(len(os.listdir(output_path)))):
        list_x2.append(0)
        list_y2.append(0)
        list_z2.append(0)
        casito = case.split('.')[0]
        case_num = casito.split('_')[1]
        case_num = int(case_num)
        list_num.append(case_num)

for i in range(len(list_num)):
    with open(cur_dir + name_landmarks + '/burdeos_' +str(list_num[i]) + '.txt','w') as f:
        f.write('%d' % list_x2[i] +','+'%d' % list_y2[i] +','+'%d' % list_z2[i])


#remove non-flipped image folder
shutil.rmtree(cur_dir + 'out1') 


#save image and landmarks path in txt
with open(cur_dir+'images_2.txt', 'w') as fi:
    for case in os.listdir(output_path):
        fi.write(output_path + '/'+case + '\n')

with open(cur_dir+'landmark_2.txt', 'w') as fi:
    for i in range(len(list_num)):
        fi.write(cur_dir + name_landmarks + '/burdeos_' +str(list_num[i]) + '.txt' + '\n')
