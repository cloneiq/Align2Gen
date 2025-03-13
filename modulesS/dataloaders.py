import torch
import numpy as np
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset

def calculate_image_mean_std(region):
    mean = np.mean(region, axis=(0, 1)) / 255.0
    std = np.std(region, axis=(0, 1)) / 255.0

    std = np.maximum(std, 0.01)
    return mean, std

def get_padding(image_size):
    h, w = image_size
    max_side = max(h, w)
    pad_top = (max_side - h) // 2
    pad_bottom = max_side - h - pad_top
    pad_left = (max_side - w) // 2
    pad_right = max_side - w - pad_left
    return (pad_left, pad_top, pad_right, pad_bottom)


class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':

            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(384),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        # image_id, image, report_ids, report_masks, seq_length, boxes, scores
        images_id, images, reports_ids, reports_masks, seq_lengths, boxess, scoress, height, width, box_images = zip(*data)
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        cropped_regions = []
        for i, (boxes, new_img, orig_img) in enumerate(zip(boxess, images, box_images)):
            img_regions = []
            orig_h, orig_w = height[i], width[i]
            new_h, new_w = new_img.size(1), new_img.size(2)

            scale_x = new_w / orig_w
            scale_y = new_h / orig_h

            for box in boxes:
                if np.array_equal(box, [-1, -1, -1, -1]):
                    img_regions.append(torch.full((3, 384, 384), 1e-8))
                    continue


                x_min = int(box[0] * scale_x)
                y_min = int(box[1] * scale_y)
                x_max = int((box[0] + box[2]) * scale_x)
                y_max = int((box[1] + box[3]) * scale_y)


                if x_max > x_min and y_max > y_min:
                    region = orig_img.crop((x_min, y_min, x_max, y_max))
                    padding = get_padding(region.size)

                    mean, std = calculate_image_mean_std(region)

                    if np.allclose(mean, 0, atol=1e-2):
                        default_mean = [0.485, 0.456, 0.406]
                        default_std = [0.229, 0.224, 0.225]
                        mean, std = default_mean, default_std

                        transform = transforms.Compose([
                            transforms.Pad(padding=padding, fill=0),
                            transforms.Resize((384, 384)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
                    else:
                        transform = transforms.Compose([
                            transforms.Pad(padding=padding, fill=0),
                            transforms.Resize((384, 384)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean.tolist(), std.tolist())
                        ])
                    region = transform(region)
                    img_regions.append(region)
                else:
                    img_regions.append(torch.full((3, 384, 384), 1e-8))

            cropped_regions.append(torch.stack(img_regions))
        cropped_regions = torch.stack(cropped_regions)

        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks), cropped_regions

