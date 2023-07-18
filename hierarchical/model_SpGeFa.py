import torch
from torch import nn, hub

# 自定義 ResNet50 分類器
class ResNet50MothClassifier(nn.Module):
    def __init__(self, num_of_species=2460, num_of_genus=1281, num_of_family=54):
        super(ResNet50MothClassifier, self).__init__()
        # 載入 torchvision 裡面訓練好的 resnet50, 以此為基礎做 transfer learning
        # 建議有空可以讀一下 ResNet 的論文，了解一下 Residual Network
        # 如果在 AS GPU Cloud, 這邊改成 v0.3.0
        model = hub.load('pytorch/vision:v0.3.0', 'resnet50', pretrained=True)
        self.instancenorm = nn.InstanceNorm2d(3)

        # 移除原本 1000 個類別的 predictor, 留下 feature extractor
        feature_extractor_modules = list(model.children())[:-2]
        self.resnet_feature_extractor = nn.Sequential(*feature_extractor_modules)

        # 自製 predictor, 如果不知道怎麼寫，可以看原本 resnet model 的最後兩層 list(model.children())[-2:]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_sp = nn.Linear(2048, num_of_species)
        self.output_ge = nn.Linear(num_of_species, num_of_genus)
        self.output_fa = nn.Linear(num_of_genus, num_of_family)
        
    
    # 模型的實際執行流程
    # x 的形狀會是 [BatchSize, NumOfChannels, ImgHeight, ImgWidth]
    # 可以用 shape attr in numpy array, size() method in pytorch tensor 看形狀
    def forward(self, x):
        # 如果使用 ImageNet 訓練好的模型，會需要用對應參數 (mean and std) 對 input (R,G,B) 進行標準化，會得到較好的結果
        # 蛾類標本的 R,G,B 分布與 ImageNet 差有點多， 這邊導入 instance normalization 取代之
        x = self.instancenorm(x)
        x = self.resnet_feature_extractor(x)
        # output 的形狀會是 [BatchSize, NumOfClasses]
        x = self.avgpool(x)
        print("x_size = ", x.size())
        output_sp = self.output_sp(x.view(-1, 2048))
        output_ge = self.output_ge(output_sp.view(-1, 2460))
        output_fa = self.output_fa(output_ge.view(-1, 1281))
        return x, output_sp, output_ge, output_fa
    
class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
