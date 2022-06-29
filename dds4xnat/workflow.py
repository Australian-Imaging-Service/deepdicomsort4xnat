from pydra import Workflow
from .tasks import *

WORKFLOW_INPUTS = [
    'mr_session', 'dcm2niix_bin', 'fslval_bin',
    'model_file', 'output_folder']


def dds4xnat_workflow(name, **kwargs):
    
    wf = Workflow(name=name, input_spec=WORKFLOW_INPUTS, **kwargs)

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
            download_dir=wf.download.lzout.out,
            dcm2niix_bin=wf.lzin.dcm2niix_bin,
            fslval_bin=wf.lzin.fslval_bin
        )
    )

    wf.add(
        predict_from_CNN(
            name='predict',
            label_file = wf.preprocessing.lzout.label_file, 
            model_file = wf.lzin.model_file,
            output_folder = wf.lzin.output_folder
        )
    )

    wf.add(
        rename_on_xnat(
            name='rename',
            out_file = wf.predict.lzout.out_file,
            output_folder=wf.lzin.output_folder,
            download_dir=wf.download.lzout.out,
            row=wf.lzin.mr_session
        )
    )

    wf.set_output([("rename_log", wf.rename.lzout.info_file)])

    return wf