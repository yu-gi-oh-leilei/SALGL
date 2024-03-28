import os
import math
import torch

import numpy as np
import torch.nn as nn
import os.path as osp
import torchvision
import torch.distributed as dist

from collections import OrderedDict
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter
from utils.misc import is_dist_avail_and_initialized, get_world_size
from utils.util import cal_gpu

try:
    from salgl_utils import EntropyLoss, GatedGNN, LowRankBilinearAttention, Element_Wise_Layer, TransformerEncoder, build_position_encoding
except ImportError:
    from .salgl_utils import EntropyLoss, GatedGNN, LowRankBilinearAttention, Element_Wise_Layer, TransformerEncoder, build_position_encoding



class SALGL(nn.Module):
    def __init__(self, cfg, feat_dim=2048, att_dim=1024):
        super(SALGL, self).__init__()

        self.cfg = cfg
        self.num_classes = cfg.DATA.num_class 
        self.num_scenes = cfg.MODEL.SALGL.num_scenes
        self.embed_type = cfg.MODEL.SALGL.embed_type
        self.embed_path = cfg.MODEL.SALGL.embed_path
        self.num_steps = cfg.MODEL.SALGL.num_steps
        self.outmess = cfg.MODEL.SALGL.outmess
        self.ignore_self = cfg.MODEL.SALGL.ignore_self
        self.normalize = cfg.MODEL.SALGL.normalize
        self.pos = cfg.MODEL.SALGL.pos
        self.soft = cfg.MODEL.SALGL.soft
        self.img_size = cfg.DATA.TRANSFORM.img_size
        self.arch_backbone = cfg.MODEL.BACKBONE.backbone
        self.distributed = cfg.MODEL.SALGL.distributed
        self.training = True
        self.pos = False

        model = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
        del model.avgpool
        del model.fc

        self.backbone = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

        self.tb = TransformerEncoder(d_model=feat_dim, nhead=8, num_layers=1)
        if self.pos:
            self.position_embedding = build_position_encoding(feat_dim, self.arch_backbone, 'sine', self.img_size)
        if self.embed_type == 'random':
            self.embeddings = nn.Parameter(torch.empty((self.num_classes, feat_dim)))
            nn.init.normal_(self.embeddings)
        else:
            self.register_buffer('embeddings', torch.from_numpy(np.load(self.embed_path)).float())
        embed_dim = self.embeddings.shape[-1]

        self.entropy = EntropyLoss()
        self.max_en = self.entropy(torch.tensor([1 / self.num_scenes] * self.num_scenes).cuda())

        self.scene_linear = nn.Linear(feat_dim, self.num_scenes, bias=False)
        self.ggnn = GatedGNN(feat_dim, steps=self.num_steps, outmess=self.outmess)
        self.register_buffer('comatrix', torch.zeros((self.num_scenes, self.num_classes, self.num_classes)))

        self.attention = LowRankBilinearAttention(feat_dim, embed_dim, att_dim)
        _feat_dim = feat_dim * 2 if self.num_steps > 0 else feat_dim
        self.fc = nn.Sequential(
            nn.Linear(_feat_dim, feat_dim),
            nn.Tanh()
        )
        self.classifier = Element_Wise_Layer(self.num_classes, feat_dim)

    def switch_mode_train(self):
        self.training = True
        self.train()

    def switch_mode_eval(self):
        self.training = False
        self.eval()

    def comatrix2prob(self):
        # comat: [bs, nc, nc]
        comat = torch.transpose(self.comatrix, dim0=1, dim1=2)
        temp = torch.diagonal(comat, dim1=1, dim2=2).unsqueeze(-1)  # [ns, nc, 1]
        comat = comat / (temp + 1e-8)  # divide diagonal

        if self.ignore_self:  # make the diagonal zeros
            mask = torch.eye(self.num_classes).cuda().unsqueeze(0)  # [1, nc, nc]
            masks = torch.cat([mask for _ in range(comat.shape[0])], dim=0)  # [ns, nc, nc]
            comat = comat * (1 - masks)

        if self.normalize:  # divide summation
            temp = torch.sum(comat, dim=-1).unsqueeze(-1)  # [ns, nc, 1]
            comat = comat / (temp + 1e-8)

        return comat

    def forward(self, x, y=None):
        if self.training and self.soft:
            self.comatrix.detach_()

        img_feats = self.backbone(x)
        if len(img_feats.size()) > 3:
            img_feats = img_feats.flatten(2).permute(2, 0, 1) 
        pos = None
        if self.pos:
            pos = self.position_embedding(x)
            pos = torch.flatten(pos, 2).transpose(1, 2)
        
        img_feats = img_feats.permute(1, 0, 2) 
        img_feats = self.tb(img_feats, pos=pos)

        img_contexts = torch.mean(img_feats, dim=1)
        scene_scores = self.scene_linear(img_contexts)

        if self.training:
            _scene_scores = scene_scores
            _scene_probs = F.softmax(_scene_scores, dim=-1)
            # print(_scene_probs.shape)
            batch_comats = torch.bmm(y.unsqueeze(2), y.unsqueeze(1))  # [bs, nc, nc]
            if self.distributed:
                _scene_probs = concat_all_gather(_scene_probs)
                batch_comats = concat_all_gather(batch_comats)
            for i in range(_scene_probs.shape[0]):
                if self.soft:
                    prob = _scene_probs[i].unsqueeze(-1).unsqueeze(-1)  # [num_scenes, 1, 1]
                    comat = batch_comats[i].unsqueeze(0)  # [1, num_classes, num_clasees]
                    self.comatrix += prob * comat
                else:
                    maxsid = torch.argmax(_scene_probs[i])
                    self.comatrix[maxsid] += batch_comats[i]
                

        scene_probs = F.softmax(scene_scores, dim=-1)
        sample_en = self.entropy(scene_probs)
        _scene_probs = torch.mean(scene_probs, dim=0)
        batch_en = (self.max_en - self.entropy(_scene_probs)) * 100

        # label vector embedding
        label_feats = self.embeddings.unsqueeze(0).repeat(x.shape[0], 1, 1)

        # compute visual representation of label
        label_feats, alphas = self.attention(img_feats, label_feats)

        # graph propagation
        if not self.soft:
            comats = self.comatrix2prob()
            indices = torch.argmax(scene_probs, dim=-1)
            comats = torch.index_select(comats, dim=0, index=indices)
        else:
            _scene_probs = scene_probs.unsqueeze(-1).unsqueeze(-1)  # [bs, num_scenes, 1, 1]
            comats = self.comatrix2prob().unsqueeze(0)  # [1, num_scenes, nc, nc]
            comats = _scene_probs * comats  # [bs, num_scenes, nc, nc]
            comats = torch.sum(comats, dim=1)  # [bs nc, nc]

        output = self.ggnn(label_feats, comats)

        if self.num_steps > 0:
            output = torch.cat([label_feats, output], dim=-1)
        output = self.fc(output)
        logits = self.classifier(output)
        # print('sample_en.shape', sample_en.shape, 'batch_en.shape', batch_en.shape, 'alphas.shape', alphas.shape, 'comats.shape', comats.shape, 'logits.shape', logits.shape, 'scene_probs.shape', scene_probs.shape)
        return {
            'logit': logits,
            'scene_probs': scene_probs,
            'sample_en': sample_en,
            'batch_en': batch_en,
            'att_weights': alphas,
            'comat': comats
        }


def do_forward_and_criterion_train(cfg, data_full, model, criterion, is_val):

    image, label = data_full['image'], data_full['target']
    targets = {'label': label}

    outputs = model(x=image, y=label)

    loss_dict = criterion(outputs, targets, is_val=False)
    weight_dict = criterion.weight_dict
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

    return outputs['logit'], losses, loss_dict, weight_dict

@torch.no_grad()
def do_forward_and_criterion_test(cfg, data_full, model, criterion, is_val):

    image, label = data_full['image'], data_full['target']
    targets = {'label': label}

    outputs = model(x=image, y=label)

    loss_dict = criterion(outputs, targets, is_val=True)
    weight_dict = criterion.weight_dict
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

    return outputs['logit'], losses, loss_dict, weight_dict


def do_forward_and_criterion(cfg, data_full, model, criterion, is_val):
    if not is_val:
        return do_forward_and_criterion_train(cfg, data_full, model, criterion, is_val)
    else:
        return do_forward_and_criterion_test(cfg, data_full, model, criterion, is_val)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def build_MLIC_SALGL(cfg):
    classnames = cfg.DATA.classnames
    
    model = SALGL(cfg)

    return model



class SetCriterion(nn.Module):

    def __init__(self, weight_dict, losses, cfg):
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.cls_loss = None
        # losses = ['cls', 'sample_en', 'batch_en']
        # losses = ['los_cls', 'loss_sample_en', 'loss_batch_en']

    def loss_cls(self, outputs, targets=None, **kwargs):

        cls = self.cls_loss(outputs['logit'], targets['label'])
        losses = {"los_cls": cls}

        return losses

    def loss_sample_en(self, outputs, targets=None, **kwargs):

        sample_en_loss = outputs['sample_en']
        losses = {"loss_sample_en": sample_en_loss}
        return losses
    
    def loss_batch_en(self, outputs, targets=None, **kwargs):

        batch_en_loss =  outputs['batch_en']
        losses = {"loss_batch_en": batch_en_loss}

        return losses


    def get_loss_train(self, loss, outputs, targets, **kwargs):

        loss_map = {
            'los_cls': self.loss_cls,
            'loss_sample_en': self.loss_sample_en,
            'loss_batch_en': self.loss_batch_en
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, **kwargs)


    def forward(self, outputs, targets=None, is_val=None, **kwargs):
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss_train(loss, outputs, targets, **kwargs))
        
        return losses




    # # model.eval()
    # for name, param in model.named_parameters():
    #     if "classifier" in name:
    #         print(name)
    #         param.requires_grad_(True)
    #     else:
    #         param.requires_grad_(False)

    # return model