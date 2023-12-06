import json
import os
from tqdm import tqdm

karpathy_val = '/home/zheedong/Projects/SEED/coco/annotations/karpathy/dataset_coco_val.json'
id2file_name = {}
with open(karpathy_val, 'r') as f:
    data = json.load(f)
    # Make {id : file_name} dict
    for img in data['images']:
        id2file_name[img['id']] = img['file_name']
    
img_path = '/home/zheedong/Projects/SEED/b64_2ea_iter50000_copy'
# Rename images
for img in tqdm(sorted(os.listdir(img_path))):
    img_id = img.split('.')[0]
    file_name = id2file_name[int(img_id)]
    os.rename(os.path.join(img_path, img), os.path.join(img_path, file_name))