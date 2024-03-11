from PythonEvaluationTools.PythonHelperTools.vqaTools.vqa import VQA
from PythonEvaluationTools.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval
import argparse


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_file_path", type=str, default="/data/public/rw/team-mmu/dataset/VQAv2/v2_mscoco_val2014_annotations.json", help="the reference file to evaluate")
    parser.add_argument("--ques_file_path", type=str, default="/data/public/rw/team-mmu/dataset/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json", help="the question file to evaluate")
    parser.add_argument("--result_file_path", type=str, help="the result file to evaluate")
    return parser.parse_args()


def vqa_eval(vqa, result_file, test_ques_path):
    vqaRes = vqa.loadRes(result_file, test_ques_path)
    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2
    # evaluate results
    vqaEval.evaluate()

    # print accuracies
    print("\n")
    print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
    print("Per Answer Type Accuracy is the following:")
    for ansType in vqaEval.accuracy['perAnswerType']:
        print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
    print("\n")

    return vqaEval


if __name__ == '__main__':
    args = parse_config()
    args.result_file_path = "/data/private/IT2IT/data/results/vqav2-AR-439M-vqgan-clipbpe-16x16-batch512-LS-all-beam3.json"
    vqa = VQA(args.anno_file_path, args.ques_file_path)
    vqa_eval(vqa, args.result_file_path, args.ques_file_path)

