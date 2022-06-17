from pydra import Workflow
from .tasks import download_from_xnat, preprocessing, predict_from_CNN

def dds4xnat_workflow(name):
    
    wf = Workflow(name=name, input_spec=['mr_session',
                                         'dcm2niix_bin',
                                         'fslval_bin'])

    # Build workflow here
    wf.add(
        download_from_xnat(
            name='download',
            row=wf.lzin.mr_session
        )
    )

    wf.add(
        preprocessing(
            name='preprocessing',
            dicom_dir=wf.download.lzout.out,
            dcm2niix_bin=wf.lzin.dcm2niix_bin,
            fslval_bin=wf.lzin.fslval_bin
        )
    )

    wf.add(
        predict_from_CNN(
            name='predict',
            label_file=wf.preprocessing.lzout.label_file
        )
    )

    return wf