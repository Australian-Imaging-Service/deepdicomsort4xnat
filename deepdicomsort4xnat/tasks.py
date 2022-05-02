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
