from pydra import mark
import xnat

import numpy as np
import os, shutil
import pydicom
import yaml

#@mark.task#-------------------------------------------------
def rename_on_xnat(server_url: str,
                   project_id: str,
                   experiment_label: str,
                   username: str,
                   password: str,
                   configymlfile: str):  #path to config.yaml file (./config.yaml)
    
    with xnat.connect(server=server_url, user=username, password=password) as xlogin:
        xproject = xlogin.projects[project_id]
        xsession = xproject.experiments[experiment_label]
    
        # Rename "scan types" of sessions

        with open(configymlfile, 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)

        prediction_file = cfg['post_processing']['prediction_file']
        root_dicom_folder = cfg['preprocessing']['root_dicom_folder']


        base_dir = os.path.dirname(os.path.normpath(root_dicom_folder))
        structured_dicom_folder = os.path.join(base_dir,'DICOM_STRUCTURED')
        root_out_folder = os.path.join(base_dir, 'DICOM_SORTED')

        os.makedirs(root_out_folder, exist_ok=True)

        predictions = np.loadtxt(prediction_file, dtype=np.str)

        prediction_names = ['T1', 'T1GD', 'T2', 'PD', 'FLAIR', 'DWI_DWI', 'DERIVED', 'PWI_DSC', 'UNKNOWN']

        prediction_file_names = predictions[:, 0]
        prediction_results = predictions[:, 1].astype(np.int)
       

        file_names = [i_file_name.split(os.sep)[-1] for i_file_name in prediction_file_names]

        unique_names = np.unique(file_names)
        unique_predictions = np.zeros([len(unique_names), 1])

        for root, dir, files in os.walk(structured_dicom_folder):
            
            if len(files) > 0 and not files[0].endswith('.DS_Store'):########(1)to get dirs containing dcm files;(2)To get rid of mac generated .DS_Store files in os.walk()
                dir_name = root.split(structured_dicom_folder)[1][1:]#(head,tail):(here head is  structured_dicom_folder, [1:] is to exclue sign(/))
                idtype = dir_name.split('/')[0]
                print(idtype)
                scan_id = idtype.split('-')[0]
                #print(scan_id)
                unpredicted_scan_type = idtype.split('-')[1]
                #print(unpredicted_scan_type)

                dir_name = dir_name.replace('/', '__')
                indices = np.argwhere([dir_name in i_result_name for i_result_name in file_names])# finds indices (rows) of prediction .csv file containing dir_name
                
                if len(indices) != 0:
                    predictions = prediction_results[indices].ravel()#finds prediction results (class) for the specific rows of prediction .csv file shown by indices 
                    i_prediction = np.bincount(predictions).argmax() - 1 #The np. bincount() is a numpy library method used to obtain the frequency of each element provided inside a numpy array
                else:
                    i_prediction = -1

                predicted_scan_type = prediction_names[i_prediction]
                print(predicted_scan_type)

                scan = xsession.scans[scan_id]
                scan.type = prediction_names[i_prediction]


#rename_on_xnat('http://localhost:8080/','20220609122023','timepoint1','admin','admin','/Users/mahdieh/git/workflows/deepdicomsort4xnat/deepdicomsort4xnat/DDS_main/config.yaml')