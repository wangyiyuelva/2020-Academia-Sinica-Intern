import torch
from torch import nn, hub

# 自定義 ResNet50 分類器
class ResNet50MothClassifier(nn.Module):
    def __init__(self, num_of_species, num_of_genus, num_of_family):
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
        #self.predictor = nn.Sequential(
        #    nn.AdaptiveAvgPool2d((1, 1)),
        #    nn.Linear(2048, 1000)
        #    # put some layers inside
        #)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.leaky_relu = nn.LeakyReLU()
        self.fc_species = nn.Linear(2048, num_of_species)
        self.fc_genus = nn.Linear(num_of_species, num_of_genus)
        self.fc_family = nn.Linear(num_of_genus, num_of_family)
    
    # 模型的實際執行流程
    # x 的形狀會是 [BatchSize, NumOfChannels, ImgHeight, ImgWidth]
    # 可以用 shape attr in numpy array, size() method in pytorch tensor 看形狀
    def forward(self, x):
        # 如果使用 ImageNet 訓練好的模型，會需要用對應參數 (mean and std) 對 input (R,G,B) 進行標準化，會得到較好的結果
        # 蛾類標本的 R,G,B 分布與 ImageNet 差有點多， 這邊導入 instance normalization 取代之
        x = self.instancenorm(x)
        x = self.resnet_feature_extractor(x)
        # output 的形狀會是 [BatchSize, NumOfClasses]
        # output = self.predictor(x)
        feat = self.avgpool(x)
        # print(x.size())
        species_output = self.fc_species(feat.view(-1, 2048))
        #species_output = x
        genus_output = self.fc_genus(self.leaky_relu(species_output))
        #genus_output = x
        family_output = self.fc_family(self.leaky_relu(genus_output))
        #family_output = x
        return feat, species_output, genus_output, family_output
