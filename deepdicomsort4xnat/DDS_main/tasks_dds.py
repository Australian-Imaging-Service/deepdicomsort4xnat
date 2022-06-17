from pathlib import Path
from pydra import Workflow
from pydra import mark
import numpy as np
import os, shutil, glob
import pydicom
import yaml
from tensorflow.keras.models import load_model
import Tools.data_IO as data_IO
import tensorflow as tf
from Preprocessing import DICOM_preparation_functions as DPF
from Preprocessing import NIFTI_preparation_functions as NPF
import time
from arcana.core.data.row import DataRow



@mark.task #---task #1: tested ok----------------------------------------------
def download_from_xnat(row: DataRow) -> Path:

    download_dir = Path.cwd() / 'download'

    store = row.dataset.store

    with store:
        xproject = store.login[row.dataset.id]
        xsession = xproject.experiments[row.id]
        xsession.download_dir(str(download_dir)) #download the data to the local machine

    return download_dir



#task1 = download_from_xnat('http://localhost:8080/','20220609122023','timepoint1','admin','admin','/Users/mahdieh/git/workflows/deepdicomsort4xnat/deepdicomsort4xnat/DDS_main/config.yaml')
#task1()

@mark.task #---task #2: tested ok----------------------------------------------
@mark.annotate({
    "return": {
        "label_file": Path,
        "nifti_dir": Path
    }
})
def preprocessing(dicom_dir: Path,
                  dcm2niix_bin: str,
                  fslval_bin: str,
                  x_image_size: int=256,
                  y_image_size: int=256,
                  z_image_size: int=25):
    #config_ymlfile = '/Users/mahdieh/git/workflows/deepdicomsort4xnat/deepdicomsort4xnat/DDS_main/config.yaml'

    start_time = time.time()
    # with open(config_ymlfile, 'r') as ymlfile:
    #     cfg = yaml.safe_load(ymlfile)

    # x_image_size = cfg['data_preparation']['image_size_x']
    # y_image_size = cfg['data_preparation']['image_size_y']
    # z_image_size = cfg['data_preparation']['image_size_z']
    # DICOM_FOLDER = cfg['preprocessing']['root_dicom_folder']
    # DCM2NIIX_BIN = cfg['preprocessing']['dcm2niix_bin']
    # FSLREORIENT_BIN = cfg['preprocessing']['fslreorient2std_bin']
    # FSLVAL_BIN = cfg['preprocessing']['fslval_bin']


    DEFAULT_SIZE = [x_image_size, y_image_size, z_image_size]


    def create_directory(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)


    def is_odd(number):
        return number % 2 != 0


    print('Sorting DICOM to structured folders....')
    structured_dicom_folder = DPF.sort_DICOM_to_structured_folders(dicom_dir)

    # Turn the following step on if you have problems running the pipeline
    # It will replaces spaces in the path names, which can sometimes
    # Cause errors with some tools
    # print('Removing spaces from filepaths....')
    # DPF.make_filepaths_safe_for_linux(structured_dicom_folder)
    #
    print('Checking and splitting for double scans in folders....')
    DPF.split_in_series(structured_dicom_folder)

    print('Converting DICOMs to NIFTI....')
    nifti_folder = NPF.convert_DICOM_to_NIFTI(structured_dicom_folder, dcm2niix_bin)

    print('Moving RGB valued images.....')
    NPF.move_RGB_images(nifti_folder, fslval_bin)

    print('Extracting single point from 4D images....')
    images_4D_file = NPF.extract_4D_images(nifti_folder)

    # print('Reorient to standard space....')
    # NPF.reorient_to_std(nifti_folder, FSLREORIENT_BIN)

    print('Resampling images....')
    nifti_resampled_folder = NPF.resample_images(nifti_folder, DEFAULT_SIZE)#resample to default size

    print('Extracting slices from images...')
    nifti_slices_folder = NPF.slice_images(nifti_resampled_folder)

    print('Rescaling image intensity....')
    NPF.rescale_image_intensity(nifti_slices_folder)

    print('Creating label file....')
    NPF.create_label_file(nifti_slices_folder, images_4D_file)

    elapsed_time = time.time() - start_time

    print(elapsed_time)

    return label_file, nifti_dir

#preprocessing('/Users/mahdieh/git/workflows/deepdicomsort4xnat/deepdicomsort4xnat/DDS_main/config.yaml')

@mark.task #---task #3: tested ok----------------------------------------------
def predict_from_CNN(model_file: str, config_ymlfile: str):
    #model_file = './Trained_Models/model_all_brain_tumor_data.hdf5'
    #config_ymlfile = '/Users/mahdieh/git/workflows/deepdicomsort4xnat/deepdicomsort4xnat/DDS_main/config.yaml'

    batch_size = 1

    with open(config_ymlfile, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    test_label_file = cfg['testing']['test_label_file']
    x_image_size = cfg['data_preparation']['image_size_x']
    y_image_size = cfg['data_preparation']['image_size_y']

    output_folder = cfg['testing']['output_folder']
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_name = os.path.basename(os.path.normpath(model_file)).split('.hdf5')[0]
    out_file = os.path.join(output_folder, 'Predictions_' + model_name + '.csv')


    def load_labels(label_file):
        labels = np.genfromtxt(label_file, dtype='str')
        label_IDs = labels[:, 0]
        label_IDs = np.asarray(label_IDs)
        label_values = labels[:, 1].astype(np.int)
        extra_inputs = labels[:, 2:].astype(np.float)
        np.round(extra_inputs, 2)

        N_classes = len(np.unique(label_values))

        # Make sure that minimum of labels is 0
        label_values = label_values - np.min(label_values)

        return label_IDs, label_values, N_classes, extra_inputs


    test_image_IDs, test_image_labels, _, extra_inputs = load_labels(test_label_file)


    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, amsgrad=False)

    model = load_model(model_file)
    model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['categorical_accuracy']
    )

    NiftiGenerator_test = data_IO.NiftiGenerator2D_ExtraInput(batch_size,
                                                            test_image_IDs,
                                                            test_image_labels,
                                                            [x_image_size, y_image_size],
                                                            extra_inputs)

    with open(out_file, 'w') as the_file:
        for i_file, i_label, i_extra_input in zip(test_image_IDs, test_image_labels, extra_inputs):
            print(i_file)

            image = NiftiGenerator_test.get_single_image(i_file)

            supplied_extra_input = np.zeros([1, 1])
            supplied_extra_input[0, :] = i_extra_input
            prediction = model.predict([image, supplied_extra_input])
            the_file.write(i_file + '\t' + str(np.argmax(prediction) + 1) + '\t' + str(i_label) + '\n')

#predict_from_CNN('./Trained_Models/model_all_brain_tumor_data.hdf5', '/Users/mahdieh/git/workflows/deepdicomsort4xnat/deepdicomsort4xnat/DDS_main/config.yaml')

@mark.task #---task #4: tested ok----------------------------------------------
def rename_on_xnat(server_url: str,
                   project_id: str,
                   experiment_label: str,
                   username: str,
                   password: str,
                   config_ymlfile: str):  #path to config.yaml file (./config.yaml)
    
    with xnat.connect(server=server_url, user=username, password=password) as xlogin:
        xproject = xlogin.projects[project_id]
        xsession = xproject.experiments[experiment_label]
    
        # Rename "scan types" of sessions

        with open(config_ymlfile, 'r') as ymlfile:
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

@mark.task #---task #5: tested ok----------------------------------------------
def cleanup(config_ymlfile: str):
    
    with open(config_ymlfile, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    data_dir = cfg['preprocessing']['root_dicom_folder']
    base_dir = os.path.dirname(os.path.normpath(data_dir)) 
    
    output_dir = os.path.join(base_dir, 'Output')
    structured_dir = os.path.join(base_dir, 'DICOM_STRUCTURED')
    NIFTI_dir  = os.path.join(base_dir, 'NIFTI')
    NIFTI_RESAMPLED_dir = os.path.join(base_dir, 'NIFTI_RESAMPLED')
    NIFITI_SLICES_dir = os.path.join(base_dir, 'NIFTI_SLICES')

    shutil.rmtree(output_dir)
    shutil.rmtree(data_dir)#remove data folder
    shutil.rmtree(structured_dir)
    shutil.rmtree(NIFTI_dir)
    shutil.rmtree(NIFTI_RESAMPLED_dir)
    shutil.rmtree(NIFITI_SLICES_dir)
    

############Building workflow###########################################

def dds4xnat_workflow(name): #dds_wf
    

    config_ymlfile = '/Users/mahdieh/git/workflows/deepdicomsort4xnat/deepdicomsort4xnat/DDS_main/config.yaml'
    server_url= 'http://localhost:8080/'
    project_id= '20220609122023'
    experiment_label= 'timepoint1'
    username= 'admin'
    password= 'admin'
    
    wf = Workflow(name=name, input_spec=["x"], x = [server_url,project_id,experiment_label,username,password,config_ymlfile])

    wf.add(download_from_xnat(name="Download", x=wf.lzin.x))
    

    #return wf

dds4xnat_workflow('dds_wf')

