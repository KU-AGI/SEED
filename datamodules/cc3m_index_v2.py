import tarfile
import json

def process_tar_file(tar_file):
    """
    Process a single tar file to extract image names and map them to the tar file.
    """
    image_tar_mapping = {}
    try:
        with tarfile.open(tar_file, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile():
                    # Map image name to tar file
                    image_tar_mapping[member.name] = tar_file
    except Exception as e:
        print(f"Error processing {tar_file}: {e}")
    return image_tar_mapping

from dask import compute
import dask

def create_image_tar_index_parallel(tar_files, index_path):
    """
    Create an index mapping image names to their containing tar files using Dask for parallel processing.
    """
    # Create delayed tasks for processing each tar file
    tasks = [dask.delayed(process_tar_file)(tar_file) for tar_file in tar_files]
    
    # Use Dask to execute tasks in parallel and gather results
    results = compute(*tasks, scheduler='processes')[0]
    
    # Combine results from all tasks into a single index
    index = {}
    for result in results:  # results is a tuple of results, so we take the first element
        index.update(result)
    
    # Save the index to a JSON file
    with open(index_path, 'w') as f:
        json.dump(index, f)

# Example usage
if __name__ == "__main__":
    # List of tar files to process
    tar_files = [f'/home/zheedong/Projects/SEED/data/cc3m/{i:05d}.tar' for i in range(332)]  # Adjust the range as needed
    # Path to save the index file
    index_path = 'cc3m_index_v2.json'
    create_image_tar_index_parallel(tar_files, index_path)
