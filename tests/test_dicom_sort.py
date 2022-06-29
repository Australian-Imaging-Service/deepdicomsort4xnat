from pathlib import Path
from dds4xnat import dds4xnat_workflow
from arcana.core.data.row import DataRow



def test_dicom_sort(mr_session: DataRow, trained_model_file: Path, work_dir: Path):
    
    output_dir = work_dir / 'outputs'
    output_dir.mkdir()

    wf = dds4xnat_workflow(
        name='test_workflow',
        mr_session=mr_session,
        dcm2niix_bin='/usr/local/bin/dcm2niix',
        fslval_bin='/usr/local/fsl/bin/fslval',
        model_file=trained_model_file,
        output_folder=output_dir)

    # run workflow
    wf(plugin='serial')

