import torch_fidelity
import torch.nn.functional as F
import torch
import lpips
import torch.nn as nn
import torchvision.models as models


class loss_fn:
    def __init__(self, lam=0.1):
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
        self.lam = lam
        # self.feature_extractor = torch_fidelity.feature_extractor.InceptionV3FeatureExtractor()
        # 1️⃣ 加载 Inception v3 预训练模型（去掉分类头）
        # self.inception = models.inception_v3(pretrained=True)
        # self.inception.fc = nn.Identity()  # 让 Inception v3 直接输出 2048 维特征
        # self.inception.cuda()
        # self.inception.eval()  # 进入评估模式（如果训练时需 fine-tune，可改为 `train()`）
    @torch.enable_grad()
    def get_loss(self, pred, gt):
        loss_mse = torch.mean((pred - gt)**2)
        loss_lpips = self.loss_fn_vgg(pred, gt).mean()
        # pred_resized = F.interpolate(pred, size=(299, 299), mode='bilinear', align_corners=False)
        # gt_resized = F.interpolate(gt, size=(299, 299), mode='bilinear', align_corners=False)
        # features1 = self.inception(pred_resized)
        # features2 = self.inception(gt_resized)
        # loss_inception = (torch.sum((features1 - features2) ** 2, dim=1)).mean()
        # loss = loss_mse + loss_lpips * 0.1 + loss_inception * 0.001
        loss = loss_mse + loss_lpips * self.lam
        # print(loss_mse)
        # print(loss_lpips)
        # print(loss_inception)
        return {'loss': loss, 'loss_mse': loss_mse, 'loss_lpips': loss_lpips, 'loss_inception': torch.tensor(0.0)}
