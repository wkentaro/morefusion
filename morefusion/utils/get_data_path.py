import path


def get_data_path(name):
    home = path.Path('~').expanduser()
    return home / 'data' / name
