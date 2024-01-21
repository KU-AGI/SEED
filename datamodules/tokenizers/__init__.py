import os
import re
import torch
import numpy as np
from functools import partial
from tokenizers import CharBPETokenizer
from transformers import GPT2Tokenizer, CLIPTokenizer
from datamodules.datasets.dataclass import TextInputItem

root_dir = os.path.dirname(os.path.abspath(__file__))

TOKENIZERS = {
    'CharBPE': partial(CharBPETokenizer.from_file,
                       vocab_filename=os.path.join(root_dir, "bpe-16k-vocab.json"),
                       merges_filename=os.path.join(root_dir, "bpe-16k-merges.txt"),
                       unk_token="[UNK]"),
    'GPT2': partial(GPT2Tokenizer,
                    os.path.join(root_dir, "gpt2-medium-vocab.json"),
                    os.path.join(root_dir, "gpt2-medium-merges.txt"),
                    ),
    'CLIP': partial(CLIPTokenizer,
                    os.path.join(root_dir, "clip-vocab.json"),
                    os.path.join(root_dir, "clip-merges.txt"),
                    ),
    'ClipBPE': partial(CharBPETokenizer.from_file,
                       vocab_filename=os.path.join(root_dir, "clip-vocab.json"),
                       merges_filename=os.path.join(root_dir, "clip-merges.txt"),
                       unk_token="[UNK]"),
    'GPT2BPE': partial(CharBPETokenizer.from_file,
                       vocab_filename=os.path.join(root_dir, "gpt2-medium-vocab.json"),
                       merges_filename=os.path.join(root_dir, "gpt2-medium-merges.txt"),
                       unk_token="[UNK]"),
}


def build_tokenizer(tokenizer_type: str,
                    context_length: int,
                     *args,
                     **kwargs):
    tokenizer = TOKENIZERS[tokenizer_type](*args, **kwargs)
    if tokenizer_type.startswith('CharBPE'):
        tokenizer.add_special_tokens(["[PAD]"])
        tokenizer.enable_padding(length=context_length,
                                 pad_id=tokenizer.token_to_id("[PAD]"))
        tokenizer.enable_truncation(max_length=context_length)
    elif tokenizer_type == 'GPT2':
        tokenizer.add_special_tokens({
            'pad_token': '[PAD]',
            'unk_token': '[UNK]'
        })
    elif tokenizer_type in ['ClipBPE', 'GPT2BPE']:
        # tokenizer.add_special_tokens(["[PAD]"])
        # pad_token_id = tokenizer.token_to_id("[PAD]")
        tokenizer.eos_token_id = tokenizer.token_to_id("<|endoftext|>")
        tokenizer.pad_token_id = tokenizer.token_to_id("<|endoftext|>")
        tokenizer.enable_padding(length=context_length,
                                 pad_id=tokenizer.pad_token_id)
        tokenizer.enable_truncation(max_length=context_length)

    return tokenizer


class TokenizerUtils:
    def build_tokenizer(self, tokenizer_type, text_ctx, lowercase=True, dropout=0.0, sep_token=None):
        self.text_ctx = text_ctx
        self.tokenizer = build_tokenizer(tokenizer_type, text_ctx, lowercase=lowercase, dropout=dropout)

        self.sep_token = sep_token
        self.sep_token_id = None
        if sep_token is not None:
            ids, _ = self.get_token_ids(sep_token)
            sep_id = ids[:ids.index(self.tokenizer.eos_token_id)]
            self.sep_token_id = sep_id[0]

    def get_n_txt(self, ids):
        n_txt = 0
        for _id in ids:
            if _id == self.tokenizer.eos_token_id:
                break
            n_txt += 1
        return n_txt

    def get_token_ids(self, txt, pre_proc=None):
        if pre_proc is not None:
            txt = pre_proc(txt)
        if callable(self.tokenizer):
            output = self.tokenizer(txt, padding="max_length", max_length=self.text_ctx)
            ids = output.input_ids
            if len(ids) > self.text_ctx:
                ids = ids[:self.text_ctx]
        else:
            output = self.tokenizer.encode(txt)
            ids = output.ids

        n_txt = self.get_n_txt(ids) + 1  # including <eos>

        return ids, n_txt

    def get_input(self, txt, pre_proc=None) -> TextInputItem:
        try:
            input, n_txt = self.get_token_ids(txt, pre_proc=pre_proc)
        except:
            txt = txt.encode("ascii", "ignore").decode()
            input, n_txt = self.get_token_ids(txt, pre_proc=pre_proc)

        input_mask = np.ones(len(input))
        input_mask[n_txt:] = 0

        item = TextInputItem(input, input_mask)
        return item

    def get_QA_input(self, question, answer=None, max_ques_len=30) -> TextInputItem:
        ques_ids, n_ques = self.get_token_ids(self.pre_question(question))
        ques_ids = ques_ids[:max_ques_len] + [self.sep_token_id]
        input = ques_ids

        ques_input_mask = np.ones(len(ques_ids))
        ques_input_mask[n_ques:] = 0
        ques_input_mask[-1] = 1 # sep_token_index
        input_mask = ques_input_mask
        target_mask = None
        if answer is not None:
            ans_ids, n_ans = self.get_token_ids(self.pre_answer(answer))
            ans_ids = ans_ids[:len(ans_ids) - self.max_ques_len]
            input += ans_ids
            ans_input_mask = np.ones(len(ans_ids))
            ans_input_mask[n_ans:] = 0
            input_mask = np.concatenate([input_mask, ans_input_mask], axis=0)
            target_mask = np.zeros(len(input_mask))
            boa_idx = len(ques_input_mask) - 1
            target_mask[boa_idx:boa_idx + n_ans] = 1

        item = TextInputItem(input, input_mask, target_mask=target_mask)
        return item

    def get_vocab_size(self):
        if callable(self.tokenizer):
            return len(self.tokenizer)
        else:
            return self.tokenizer.get_vocab_size()

    @staticmethod
    def pre_caption(caption):
        caption = caption.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ').replace('<person>', 'person').replace('< person >', 'person')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        return caption

    @staticmethod
    def post_caption(caption):
        caption = TokenizerUtils.pre_caption(caption)
        caption = caption.replace('<', '').replace('>', '')
        return caption

    @staticmethod
    def pre_answer(answer):
        answer = answer.replace('’', '\'')
        return answer

    @staticmethod
    def pre_question(question, max_ques_words=None):
        question = re.sub(
            r"([,.'!?\"()*#:;~])",
            '',
            question.lower(),
        ).replace('-', ' ').replace('/', ' ')
        question = question.rstrip(' ')

        #truncate question
        question_words = question.split(' ')
        if max_ques_words is not None and len(question_words) > max_ques_words:
            question = ' '.join(question_words[:max_ques_words])

        return question


if __name__ == "__main__":
    tokenizer = build_tokenizer("ClipBPE", 64, lowercase=True, dropout=0.1)

    sent = 'Run Like a Girl Baseball 3⁄4 Sleeve T-Shirt'
    sent = '<PERSON> wins sword of honour from Sandhurst Academy A moment of great pride for all <PERSON> It is a moment of great pride for Pakistan when young <PERSON> wins a sword of honour from Royal Military Academy of ... Continue reading → Pakistan Today, Pakistan Army, Royal Military Academy Sandhurst, Pakistan Armed Forces, Royal Marines, War Machine, Canada Goose Jackets, Armies, <PERSON>'
    sent = 'All Inclusive package Airport -- Shuttle Bus & Cruise Ship from €175 Airport Shuttle Transfer Cruise Ship 1200x480'
    sent = 'what is the color? yellow'
    sent = '<person>'

    out = tokenizer.encode(sent)
    print(out.ids)

    # import json
    # data = json.load(open('/data/public/rw/datasets/coco/annotations/captions_test2014.json', 'r'))
    # data = data['annotations']
    # for ann in data:
    #     caption = ann['caption']
    #     clean_cap = TokenizerUtils.pre_caption(caption)
    #     if caption.lower() != clean_cap:
    #         print(caption, clean_cap)

    # import json
    # import numpy as np
    # from tqdm import tqdm
    # # sents = json.load(open('/data/private/IT2IT/data/VQA/answer_list.json', 'r'))
    # sents = []
    # data = json.load(open('/data/private/IT2IT/data/VQA/vqa_train.json', 'r'))
    # for dt in tqdm(data):
    #     answers = dt['answer']
    #     for answer in answers:
    #         sents.append(TokenizerUtils.pre_question(answer))
    #
    # lengths = []
    # for sent in tqdm(sents):
    #     sent = sent.replace('’', '\'')
    #     try:
    #         out = tokenizer.encode(sent)
    #     except:
    #         sent_ = sent.encode("ascii", "ignore").decode()
    #         print(sent, sent_)
    #     ids = out.ids
    #     ids = ids[:ids.index(tokenizer.eos_token_id)]
    #     lengths.append(len(ids))
    #
    # lengths = np.array(lengths)
    #
    # print(lengths.mean(), lengths.max(), lengths.min())

    # print(tokenizer)
    # if callable(tokenizer):
    #     vocab = tokenizer.get_vocab()
    #     print(vocab["<|endoftext|>"])
    #     # print(vocab["[UNK]"])
    #     # print(vocab["[PAD]"])
    #     out = tokenizer.encode(sent)
    #     txt = tokenizer.decode(out)
    #     print(out)
    #     print(txt)
    # else:
    #     print(tokenizer.token_to_id("<|endoftext|>"))
    #     print(tokenizer.token_to_id("[UNK]"))
    #     print(tokenizer.token_to_id("[PAD]"))
    #     try:
    #         txt = sent.encode("ascii", "ignore").decode()
    #         print(txt)
    #         out = tokenizer.encode(txt)
    #         ids = out.ids
    #     except:
    #         print(sent)
    #         pass
    #
    #     print(ids)
    #     ids = ids[:ids.index(tokenizer.eos_token_id)]
    #     print(ids)
    #     print(ids[:ids.index(tokenizer.eos_token_id)])