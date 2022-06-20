import numpy as np
import os
import pydicom
import shutil
import yaml


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

with open('./config.yaml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

prediction_file = cfg['post_processing']['prediction_file']
root_dicom_folder = cfg['preprocessing']['root_dicom_folder']


base_dir = os.path.dirname(os.path.normpath(root_dicom_folder))
structured_dicom_folder = os.path.join(base_dir,'DICOM_STRUCTURED')
root_out_folder = os.path.join(base_dir, 'DICOM_SORTED')

os.makedirs(root_out_folder, exist_ok=True)

predictions = np.loadtxt(prediction_file, dtype=np.str)

prediction_names = ['T1', 'T1GD', 'T2', 'PD', 'FLAIR', 'DWI_DWI', 'DERIVED', 'PWI_DSC', 'UNKNOWN']
orientation_names = ['3D', 'Ax', 'Cor', 'Sag', 'Obl', '4D', 'UNKNOWN']

prediction_file_names = predictions[:, 0]
prediction_results = predictions[:, 1].astype(np.int)##################original=>prediction_results = predictions[:, 1].astype(np.int)
#print(prediction_results)##################

file_names = [i_file_name.split(os.sep)[-1] for i_file_name in prediction_file_names]


def get_image_orientation(dicom_slice):
    """
    This function will determine the orientation of MRI from DICOM headers.
    """

    # First check the Acquisition type, if it's 3D or 4D there's no orientation
    try:
        acquistion_type = dicom_slice[0x18, 0x23].value
    except KeyError:
        return 6

    if acquistion_type == '3D':
        return 0
    elif acquistion_type == '4D':
        return 5
    else:
        # Determine orientation from direction cosines
        if (0x20, 0x37) in dicom_slice:
            orientation = dicom_slice[0x20, 0x37].value
        else:
            return 6

        X_vector = np.abs(np.array(orientation[0:3]))
        Y_vector = np.abs(np.array(orientation[3:6]))

        X_index = np.argmax(X_vector)
        Y_index = np.argmax(Y_vector)

        if X_index == 0 and Y_index == 1:
            orientation = 1
        elif X_index == 0 and Y_index == 2:
            orientation = 2
        elif X_index == 1 and Y_index == 2:
            orientation = 3
        else:
            orientation = 4
    return orientation


unique_names = np.unique(file_names)
unique_predictions = np.zeros([len(unique_names), 1])

#print(file_names)

for root, dir, files in os.walk(structured_dicom_folder):
    if len(files) > 0 and not files[0].endswith('.DS_Store'):########(1)to get dirs containing dcm files;(2)To get rid of mac generated .DS_Store files in os.walk()
        #print(files)
        #b = root.split(structured_dicom_folder)
        #print(b)
        dir_name = root.split(structured_dicom_folder)[1][1:]#(head,tail):(here head is  structured_dicom_folder, [1:] is to exclue sign(/))
        #print(dir_name)###########prints directory pathes contatining dcm files
        #patient_ID = dir_name.split('/')[0]
        #~~~~~~~~~~~~~~~~~~1~at the end of wf, put a condition to empty data folder and it by product (dcm stuctured folfer, nifti folder, etc.) and delete .csv prediction file exist (from the current run), to go to the next run
        
        idtype = dir_name.split('/')[0]
        print(idtype)
        ID = idtype.split('-')[0]
        #print(ID)
        scanType = idtype.split('-')[1]
        print(scanType)

        sub_dir_name = dir_name.split('/')[1:len(dir_name.split('/'))-1]#---------------
        #print(sub_dir_name)
        #scan_label = dir_name.split('/')[-1]#scan_label is the folder containg dcm files
        #print(scan_label)
        dir_name = dir_name.replace('/', '__')
        indices = np.argwhere([dir_name in i_result_name for i_result_name in file_names])# finds indices (rows) of prediction .csv file containing dir_name
        #print(indices)
        if len(indices) != 0:
            predictions = prediction_results[indices].ravel()#finds prediction results (class) for the specific rows of prediction .csv file shown by indices 
            #print(predictions)
            i_prediction = np.bincount(predictions).argmax() - 1 #The np. bincount() is a numpy library method used to obtain the frequency of each element provided inside a numpy array
            #print(i_prediction)
        else:
            i_prediction = -1

        scanType_toRename = prediction_names[i_prediction]#~~~~~4~~~~~~~~~~~~~~
        print(scanType_toRename)
        #~~~~~~~~now rename the scan type on xnat#~~~~~~~~~~~~~~~~
        dicom = pydicom.read_file(os.path.join(root, files[0]))   ################### dicom = pydicom.read_file(os.path.join(root, files[0]), force=True)

        orientation = get_image_orientation(dicom)

        if len(sub_dir_name) > 1:
            sub_dir_name = '/'.join(sub_dir_name)
        else:
            sub_dir_name = sub_dir_name[0]

        temp_new_dir_name = os.path.join(root_out_folder, patient_ID, sub_dir_name, prediction_names[i_prediction] + '_' + orientation_names[orientation])


        new_dir_name = temp_new_dir_name
        i_new_dir_counter = 1
        while os.path.exists(new_dir_name):
            new_dir_name = temp_new_dir_name + '_' + str(i_new_dir_counter)
            i_new_dir_counter += 1

        shutil.copytree(root, new_dir_name)

