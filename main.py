import torch
import argparse
import numpy as np
from modulesS.tokenizersS import TokenizerS
from modulesS.dataloaders import R2DataLoader
from modulesS.metrics import compute_scores
from modulesS.optimizers import build_optimizer, build_lr_scheduler
from modulesS.trainer import Trainer
from modulesS.loss import compute_loss
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.s2gen import S2GenModel

def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='/root/autodl-tmp/R2Gen-main/data/mimic_cxr/images', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='/root/autodl-tmp/R2Gen-main/data/mimic_cxr/annotation.json', help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=100, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=10, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=32, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    # parser.add_argument('--visual_extractor', type=str, default='vit_base_patch32_224', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')


    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=1024, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    # for Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=50, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/mimic_cxr', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    # ViT 最后两层的学习率
    parser.add_argument('--lr_vit_last', type=float, default=1e-4, help='Learning rate for the last two layers of ViT (encoder.layer.11 and encoder.layer.12).')
    parser.add_argument('--lr_ve', type=float, default=1e-4, help='the learning rate for the visual extractor.')
    #parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--lr_te', type=float, default=1e-4, help='Learning rate for text extractor')
    parser.add_argument('--lr_tle', type=float, default=1e-4, help='Learning rate for text list extractor')
    parser.add_argument('--lr_other', type=float, default=1e-4, help='Learning rate for other model parameters')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR',
                        help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=6,
                        help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='the gamma of the learning rate scheduler.')
    parser.add_argument('--T_max', type=int, default=25, help='the number of epochs in the cosine annealing cycle.')
    parser.add_argument('--eta_min', type=float, default=1e-6, help='Minimum learning rate.')

    # Others
    parser.add_argument('--seed', type=int, default=456789, help='.')
    parser.add_argument('--resume', type=str,help='whether to resume the training from existing checkpoints.')


    # add-text
    parser.add_argument('--embed_size', default=1024, type=int, help='Dimensionality of the joint embedding.')
    # parser.add_argument('--no_txtnorm', action='store_true',help='Do not normalize the text embeddings.')
    parser.add_argument('--no_txtnorm', default=False, action='store_true', help='Do not normalize the text embeddings.')

    # add-align-img
    parser.add_argument('--alpha', default=2.0, type=float, help='Initial penalty parameter.')
    parser.add_argument('--precomp_enc_type', default="basic", help='basic|weight_norm')
    parser.add_argument('--thres', default=0, type=float,help='Optimal learning  boundary.')
    parser.add_argument('--lambda_softmax', default=20., type=float, help='Attention softmax temperature.')
    parser.add_argument('--l2_lambda', default=0.01, type=float, help='L2 regularization parameter.')
    parser.add_argument('--margin_ratio', default=0.1, type=float, help='Margin ratio parameter.')
    parser.add_argument('--margin', default=0.2, type=float, help='Rank loss margin.')
    parser.add_argument('--no_imgnorm', action='store_true', help='Do not normalize the image embeddings.')

    parser.add_argument('--test_only', action='store_true', help='If true, only generate reports without training.')

    args = parser.parse_args()
    return args

def main():
    # parse arguments
    args = parse_agrs()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = TokenizerS(args)

    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=False)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    model = S2GenModel(args, tokenizer)

    # freeze ViT layers
    # freeze_vit_layers(model)

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    if args.lr_scheduler == 'CosineAnnealingLR':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max)
    else:
        lr_scheduler = build_lr_scheduler(args, optimizer)

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()