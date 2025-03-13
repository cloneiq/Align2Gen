import torch
import torch.nn as nn
import torchvision.models as models
from .LERA import LERA
from .CPCA import CPCA

class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        # visual extractor ResNet101
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        self.avg_fnt = nn.AdaptiveAvgPool2d((1, 1))
        self.cpca = CPCA(2048, 2048)

        self.classifier = nn.Linear(2048, args.num_labels)

        self.img_dim = model.fc.in_features
        self.embed_size = args.embed_size

        self.local_embedder = nn.Conv2d(
            self.img_dim,
            self.embed_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, images):
        outputs = self.model(images)

        patch_feats = self.cpca(outputs)

        patch_feats = self.local_embedder(patch_feats)

        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))

        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)

        return patch_feats, avg_feats

class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=0)
        self.bn5 = nn.BatchNorm2d(1024)
        self.relu5 = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x

class MultiTargetExtractor(nn.Module):
    def __init__(self, args):
        super(MultiTargetExtractor, self).__init__()
        self.simple = SimpleConvNet()
        self.visual_extractor = VisualExtractor(args)
        self.region_visual_extractor = LERA(in_features=3, filters=3)

    def forward(self, images, box_regions):
        batch_size, num_boxes, C, H, W = box_regions.size()

        region_features = []

        for i in range(batch_size):
            local_features = []

            for j in range(num_boxes):
                region = box_regions[i, j]

                if torch.all(region == 1e-8):
                    placeholder_feature = torch.full((1, self.visual_extractor.embed_size), 1e-8, device=images.device)
                    local_features.append(placeholder_feature)
                    continue

                region = region.unsqueeze(0)

                region = self.region_visual_extractor(region)
                encoded_region = self.simple(region)
                local_features.append(encoded_region)


            if local_features:
                local_features_tensor = torch.cat(local_features, dim=0)  # (num_boxes, embed_size)
                region_features.append(local_features_tensor.unsqueeze(0))  # (1, num_boxes, embed_size)


        if region_features:
            region_features = torch.cat(region_features, dim=0)  # (batch_size, num_boxes, embed_size)
        else:
            region_features = torch.full((batch_size, num_boxes, self.visual_extractor.embed_size), 1e-8, device=images.device)

        patch_feats_mapped, avg_feats_mapped = self.visual_extractor(images)

        return region_features, avg_feats_mapped, patch_feats_mapped

