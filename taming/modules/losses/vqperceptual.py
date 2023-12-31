import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.modules.losses.lpips import LPIPS
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

def sequence_mask(X, valid_lens, value=0):
    '''在序列中屏蔽不相关项'''
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_lens[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label.long())
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss
    
class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight                     #那个形而上学的λ

    def forward(self, inputs, text_input, i2t_rec, t2i_rec, codebook_loss, text_q_loss, valid_lens, c_loss,
                optimizer_idx, global_step, last_layer=None, cond=None, split="train"):
        # image part
        t2i_rec_loss = torch.abs(inputs.contiguous() - t2i_rec.contiguous())                    #文本->图像  [8, 3, 256, 256])

        if self.perceptual_weight > 0:      #1
            t2i_p_loss = self.perceptual_loss(inputs.contiguous(), t2i_rec.contiguous())        #文本->图像  [8, 1, 1, 1]
            t2i_rec_loss = t2i_rec_loss + self.perceptual_weight * t2i_p_loss                   #文本->图像  [8, 3, 256, 256])
        else:
            p_loss = torch.tensor([0.0])

        t2i_nll_loss = torch.mean(t2i_rec_loss)

        # text part
        loss = MaskedSoftmaxCELoss()
        i2t_rec_loss = loss(i2t_rec, text_input, valid_lens).mean()

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                t2i_logits_fake = self.discriminator(t2i_rec.contiguous())          #文本->图像  [8, 1, 30, 30]
            else:
                assert self.disc_conditional
                t2i_logits_fake = self.discriminator(torch.cat((t2i_rec.contiguous(), cond), dim=1))        #文本->图像
            t2i_g_loss = -torch.mean(t2i_logits_fake)       #文本->图像

            try:
                d_weight = self.calculate_adaptive_weight(t2i_nll_loss, t2i_g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            loss = t2i_nll_loss + d_weight * disc_factor * t2i_g_loss + i2t_rec_loss + self.codebook_weight * codebook_loss.mean() + text_q_loss.mean() + c_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/image_quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/t2i_nll_loss".format(split): t2i_nll_loss.detach().mean(),
                   "{}/t2i_rec_loss".format(split): torch.mean(t2i_rec_loss-self.perceptual_weight * t2i_p_loss).detach().mean(),
                   "{}/t2i_p_loss".format(split): t2i_p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/t2i_g_loss".format(split): t2i_g_loss.detach().mean(),
                   "{}/text_quant_loss".format(split): text_q_loss.detach().mean(),
                   "{}/i2t_rec_loss".format(split): i2t_rec_loss.detach().mean(),
                   "{}/constrain_loss".format(split): c_loss.detach().mean()
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())              # [8, 1, 30, 30]
                t2i_logits_fake = self.discriminator(t2i_rec.contiguous().detach())         # [8, 1, 30, 30]
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                t2i_logits_fake = self.discriminator(torch.cat((t2i_rec.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            t2i_d_loss = disc_factor * self.disc_loss(logits_real, t2i_logits_fake)

            log = {"{}/disc_t2i_loss".format(split): t2i_d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/t2i_logits_fake".format(split): t2i_logits_fake.detach().mean()
                   }
            return t2i_d_loss, log