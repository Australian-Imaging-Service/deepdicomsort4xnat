from pydra import mark
import xnat


@mark.task
def rename_on_xnat(server_url: str,
                   project_id: str,
                   session_id: str,
                   username: str,
                   password: str,
                   name_mapping_file: str):
    
    with xnat.connect(server=server_url, user=username, password=password) as xlogin:
        xproject = xlogin.projects[project_id]
        xsession = xproject.experiments[session_id]
    
        # Rename "scan types" of sessions

"""
def create_label_file(root_dir, images_4D_file):
    base_dir = os.path.dirname(os.path.normpath(root_dir))
    data_dir = os.path.join(base_dir, 'DATA')

    label_file = os.path.join(data_dir, 'labels.txt')

    images_4D = np.genfromtxt(images_4D_file, dtype='str')

    with open(label_file, 'w') as the_file:
        for root, dirs, files in os.walk(root_dir):
            for i_file in files:
                if '.nii.gz' in i_file:
                    file_name = i_file.split('.nii.gz')[0].split('_')[0:-1]
                    file_name = '_'.join(file_name)
                    if file_name in images_4D:
                        is_4D = '1'
                    else:
                        is_4D = '0'

                    file_location = os.path.join(root, i_file)

                    out_elements = [file_location, '0', is_4D]

                    the_file.write('\t'.join(out_elements) + '\n')

    return label_file