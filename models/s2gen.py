import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn

from modulesS.alignment import Align
from modulesS.encoder_decoder import EncoderDecoder

def decode_and_split_reports(report_ids, tokenizer):
    decoded_reports = []

    for report in report_ids:
        decoded_report = tokenizer.decode(report.tolist())
        decoded_reports.append(decoded_report)

    all_sentence_lists = []

    for report in decoded_reports:
        tokens = report.split()
        sentence_list = []
        current_sentence = []

        for token in tokens:
            current_sentence.append(token)
            if token == '.':
                sentence = ' '.join(current_sentence)
                sentence_list.append(sentence)
                current_sentence = []

        all_sentence_lists.append(sentence_list)

    return all_sentence_lists, decoded_reports

class S2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(S2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer

        self.align = Align(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        # forward:
        self.forward = self.extract_aligned_features

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def extract_aligned_features(self, images, box_regions, targets=None, mode='train'):
        if mode == 'train':
            text_list, texts = decode_and_split_reports(targets, self.tokenizer)
            att_feats, fc_feats, one_loss, two_loss = self.align(images, box_regions, texts, text_list, mode=mode)
        elif mode == 'sample':
            text_list, texts = None, None
            att_feats, fc_feats, one_loss, two_loss = self.align(images, box_regions, texts, text_list, mode=mode)
        else:
            raise ValueError

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output, one_loss, two_loss