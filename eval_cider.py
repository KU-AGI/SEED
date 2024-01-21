import json
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import argparse

def coco_results_processor(result_file, annotation_file):

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(result_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")
    
    return coco_eval

def metric_engine(result_file, annotation_file):
    # process results
    with open(result_file, "r") as f:
        result_file = json.load(f)

    metric = None
    metric = coco_results_processor(result_file, annotation_file)
    
    return metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--data_split", type=str, default=None)

    args = parser.parse_args()
    if args.data_split == "test":
        coco_gt_file = "/ssd0/data/coco/annotations/karpathy/dataset_coco_test.json"
    elif args.data_split == "val":
        coco_gt_file = "/ssd0/data/coco/annotations/karpathy/dataset_coco_val.json"
    else:
        print("Invalid data split")
        exit()

    metric_engine(args.result_file, coco_gt_file)