import json
import re
from collections import Counter


class TokenizerS(object):
    def __init__(self, args):
        self.ann_path = args.ann_path # annotation.json
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name # mimic_cxr
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        else:
            self.clean_report = self.clean_report_mimic_cxr
        self.ann = json.loads(open(self.ann_path, 'r').read()) # load annotation.json
        self.token2idx, self.idx2token = self.create_vocabulary() # create vocabulary to "split" : "train"

    def create_vocabulary(self):
        total_tokens = []

        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            # print(tokens)
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        # print(vocab)
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            # print(token2idx[token])
            # print(token)
            idx2token[idx + 1] = token
        # print(token2idx)
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        # print("ids:", ids)
        # print("Type of ids:", type(ids))
        # print("Shape of ids:", len(ids))

        # txt = ''
        # for i, idx in enumerate(ids):
        #     if idx > 0:
        #         if i >= 1:
        #             txt += ' '
        #         txt += self.idx2token[idx]
        #     else:
        #         break
        txt = ''
        zero_encountered = False
        for i, idx in enumerate(ids):
            if idx == 0:
                if zero_encountered:
                    break
                zero_encountered = True
            else:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
