import json
path = "/home/byeongguk/projects/magvlt2/MAGVLT2/results/seed_recon/epoch2.json"
#path = '/home/byeongguk/projects/magvlt2/MAGVLT2/results/seed_ver2/epoch1_iter20000.json'

with open(path, 'r') as f:
    data = json.load(f)
print(data[0])