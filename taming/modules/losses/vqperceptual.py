import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.modules.losses.lpips import LPIPS


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()

def image_loss(imgs, pred):     # 来自MAE
    """
    imgs: [N, 3, H, W]
    pred: [N, L, p*p*3]
    """
    # target = self.patchify(imgs)
    # if self.norm_pix_loss:
    #     mean = target.mean(dim=-1, keepdim=True)
    #     var = target.var(dim=-1, keepdim=True)
    #     target = (target - mean) / (var + 1.e-6)**.5

    loss = (pred - imgs) ** 2
    loss = loss.mean#(dim=-1)  # [N, L], mean loss per patch

    return loss

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

def sequence_mask(X, valid_lens, value=0):
    '''在序列中屏蔽不相关项'''
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_lens
    try:
        X[~mask] = value
    except Exception as e:
        print(e)
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)

        unweighted_loss = super().forward(pred.permute(0, 2, 1), label.long())
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss
    
class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, codebook_weight=1.0, perceptual_weight=1.0, disc_conditional=False):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight


    def forward(self, codebook_loss, inputs, reconstructions, i2t_rec, global_step,
                text_input, t2i_rec, text_rec, text_q_loss, i_cross_q_loss, 
                t_cross_q_loss, valid_lens, split="train"):
        # image part
        i2i_rec_loss = ((inputs - reconstructions) ** 2).mean()
        t2i_rec_loss = ((inputs - t2i_rec) ** 2).mean()

        if self.perceptual_weight > 0:
            i2i_p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())    #图像->图像  [8, 1, 1, 1]
            t2i_p_loss = self.perceptual_loss(inputs.contiguous(), t2i_rec.contiguous())        #文本->图像  [8, 1, 1, 1]

            i2i_rec_loss = i2i_rec_loss + self.perceptual_weight * i2i_p_loss                               #图像->图像  [8, 3, 256, 256])
            t2i_rec_loss = t2i_rec_loss + self.perceptual_weight * t2i_p_loss                   #文本->图像  [8, 3, 256, 256])
        else:
            p_loss = torch.tensor([0.0])

        i2i_nll_loss = i2i_rec_loss
        t2i_nll_loss = t2i_rec_loss

        i2i_nll_loss = torch.mean(i2i_nll_loss)
        t2i_nll_loss = torch.mean(t2i_nll_loss)

        # text part
        loss = MaskedSoftmaxCELoss()
           
        t2t_rec_loss = loss(text_rec[:, :49], text_input[:, :49], valid_lens).mean()
        i2t_rec_loss = loss(i2t_rec[:, :49], text_input[:, :49], valid_lens).mean()

        i_loss = i2i_nll_loss + t2i_nll_loss
        t_loss = t2t_rec_loss + i2t_rec_loss
        loss = i_loss + t_loss + self.codebook_weight * codebook_loss.mean() + text_q_loss.mean() + i_cross_q_loss + t_cross_q_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                
                "{}/i2i_nll_loss".format(split): i2i_nll_loss.detach().mean(),
                "{}/i2i_rec_loss".format(split): torch.mean(i2i_rec_loss-self.perceptual_weight * i2i_p_loss).detach().mean(),
                "{}/i2i_p_loss".format(split): i2i_p_loss.detach().mean(),
                "{}/t2i_nll_loss".format(split): t2i_nll_loss.detach().mean(),
                "{}/t2i_rec_loss".format(split): torch.mean(t2i_rec_loss-self.perceptual_weight * t2i_p_loss).detach().mean(),
                "{}/t2i_p_loss".format(split): t2i_p_loss.detach().mean(),

                "{}/image_quant_loss".format(split): codebook_loss.detach().mean(),
                "{}/text_quant_loss".format(split): text_q_loss.detach().mean(),
                "{}/t2t_rec_loss".format(split): t2t_rec_loss.detach().mean(),
                "{}/i2t_rec_loss".format(split): i2t_rec_loss.detach().mean(),
                "{}/image_cross_quant_loss".format(split): i_cross_q_loss.detach().mean(),
                "{}/text_cross_quant_loss".format(split): t_cross_q_loss.detach().mean()
                }

        return loss, log