import glob
import os
import tarfile
from typing import List


def open_tar(tar_name: str):
    file = tarfile.open(tar_name)
    file.extractall('./data')
    file.close()


def get_path_files(starting_directory: str, target_directory_name: str) -> List[str]:
    file_paths = []
    if starting_directory != target_directory_name:
        directories = glob.glob(f'{starting_directory}/**/{target_directory_name}', recursive=True)
    else:
        directories = [starting_directory]

    for directory in directories:
        print(f"Folder found: {directory}")
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                file_paths.append(file_path)

    return file_paths
