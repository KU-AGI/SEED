import tarfile
import os
import json
from tqdm import tqdm

def get_id(image_name):
    return image_name.split('.')[0]

def create_image_tar_index(tar_files, index_path):
    """
    Create an index mapping image names to their containing tar files.

    :param tar_files: List of paths to tar files.
    :param index_path: Path to save the index JSON file.
    """
    index = {}
    for tar_file in tqdm(tar_files):
        print(f"Processing {tar_file}")
        with tarfile.open(tar_file, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith('.jpg'):
                    index[get_id(member.name)] = tar_file
    with open(index_path, 'w') as f:
        json.dump(index, f)

# Example usage
tar_files = [f'/home/zheedong/Projects/SEED/data/cc3m/{i:05d}.tar' for i in range(332)]  # Adjust the range as needed
index_path = 'cc3m_index_v2.json'
create_image_tar_index(tar_files, index_path)
