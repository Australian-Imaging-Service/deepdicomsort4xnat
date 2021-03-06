import os
from pathlib import Path
import tempfile
from datetime import datetime
import pytest
import xnat4tests.config
from arcana.data.stores.medimage import Xnat


TEST_SUBJECT_LABEL = 'TESTSUBJ'
TEST_SESSION_LABEL = 'TESTSUBJ_01'

trained_models_dir = Path(__file__).parent / 'dds4xnat' / 'trained_models'

test_data_dir = Path(__file__).parent / 'tests' / 'data'

test_mr_session_names = [str(p.stem) for p in test_data_dir.iterdir()
                         if not p.name.startswith('.')]

@pytest.fixture(scope='session')
def trained_model_file():
    return trained_models_dir / 'model_all_brain_tumor_data.hdf5'

@pytest.fixture()
def work_dir():
    return Path(tempfile.mkdtemp())

@pytest.fixture(scope='session')
def xnat_project(timestamp):
    xnat4tests.launch_xnat()
    with xnat4tests.connect() as xlogin:
        xlogin.put(f'/data/archive/projects/{timestamp}')
    with xnat4tests.connect() as xlogin:
        yield xlogin.projects[timestamp]
    #xnat4tests.stop_xnat()


@pytest.fixture(scope='session')
def timestamp():
    "A datetime string used to avoid stale data left over from previous tests"
    return datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')


@pytest.fixture(params=test_mr_session_names)
def mr_session(xnat_project, work_dir, request):
    session_label = request.param
    # Upload test data to test XNAT
    upload_test_dataset(xnat_project, session_label,
                        test_data_dir / session_label)
    # Create store and dataset objects to access data
    store = Xnat(server=xnat4tests.config.XNAT_URI,
                   user=xnat4tests.config.XNAT_USER,
                   password=xnat4tests.config.XNAT_PASSWORD,
                   cache_dir=work_dir / 'cache')
    dataset = store.new_dataset(xnat_project.id)
    # Access single row of dataset
    return dataset.row(id=session_label, row_frequency='session')
    

def make_project_name(dataset_name: str, run_prefix: str=None):
    return (run_prefix if run_prefix else '') + dataset_name


def upload_test_dataset(xnat_project: str, session_label: str,
                        source_data_dir: Path):
    """
    Creates dataset for each entry in dataset_structures
    """
    xlogin = xnat_project.xnat_session
    xclasses = xlogin.classes
    xsubject = xclasses.SubjectData(label=session_label + '_subj',
                                    parent=xnat_project)
    xsession = xclasses.MrSessionData(label=session_label,
                                      parent=xsubject)
    for scan_path in source_data_dir.iterdir():
        if scan_path.name.startswith('.'):
            continue
        # Create scan
        xscan = xclasses.MrScanData(id=scan_path.stem, type=scan_path.stem,
                                    parent=xsession)
        
        for resource_path in scan_path.iterdir():

            # Create the resource
            xresource = xscan.create_resource(resource_path.stem)
            # Create the dummy files
            xresource.upload_dir(resource_path, method='tar_file')
    xlogin.put(f'/data/experiments/{xsession.id}?pullDataFromHeaders=true')

# For debugging in IDE's don't catch raised exceptions and let the IDE
# break at it
if os.getenv('_PYTEST_RAISE', "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value

    catch_cli_exceptions = False
else:
    catch_cli_exceptions = True    