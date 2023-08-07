import hashlib
import logging
import math
from typing import Union
import warnings
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
import urllib

from main import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder, Text_Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer
from taming.modules.transformer.cliptransformer import TextTransformer, Attention
from taming.modules.vqvae.quantize import Cluster
from taming.data.base import tokenize
from taming.modules.tokenizer.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch.optim.lr_scheduler import StepLR
import os

def download_pretrained_from_url(
        url: str,
        cache_dir: Union[str, None] = None,
):
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)

    if 'openaipublic' in url:
        expected_sha256 = url.split("/")[-2]
    elif 'mlfoundations' in url:
        expected_sha256 = os.path.splitext(filename)[0].split("-")[-1]
    else:
        expected_sha256 = ''

    download_target = os.path.join(cache_dir, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if expected_sha256:
            if hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(expected_sha256):
                return download_target
            else:
                warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")
        else:
            return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.headers.get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if expected_sha256 and not hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(expected_sha256):
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict

# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict

def resize_pos_embed(state_dict, pos_embed_width):
    # Resize the shape of position embeddings when loading from state_dict
    pos_emb = state_dict.get('positional_embedding', None)
    pos_emb = pos_emb.unsqueeze(0).permute(0, 2, 1)
    pos_emb = F.interpolate(pos_emb, size=pos_embed_width)
    pos_emb = pos_emb.permute(0, 2, 1).squeeze()
    state_dict['positional_embedding'] = pos_emb


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 ctconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ct_ckpt_dir=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        # 图像侧
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)  # 图像与文本侧解码器部分共用
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        # 文本侧
        self.text_encoder = TextTransformer(**ctconfig)
        self.text_decoder = Text_Decoder(ctconfig["vocab_size"])  # 图像与文本侧解码器部分共用
        self.quant_linear = nn.Linear(ctconfig["width"], embed_dim)       # 从文本侧的宽度映射到coodbook的宽度
        self.post_quant_tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=256, nhead=8), num_layers=2)
        if ct_ckpt_dir is not None:
            self.init_from_ct_ckpt(ct_ckpt_dir, self.text_encoder, ctconfig["context_length"], ignore_keys=ignore_keys)
        #图像与文本交叉量化
        self.i_t_cluster = Cluster(embed_dim)

        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_from_ct_ckpt(self, path, model, pos_embed_width, ignore_keys=list()):
        url = 'https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt'
        cache_dir = path
        checkpoint_path = download_pretrained_from_url(url, cache_dir=cache_dir)
        state_dict = load_state_dict(checkpoint_path)
        # detect old format and make compatible with new format
        if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
            state_dict = convert_to_custom_text_state_dict(state_dict)
        resize_pos_embed(state_dict, pos_embed_width)
        incompatible_keys = model.load_state_dict(state_dict, strict=False)
        return incompatible_keys        
        

    # 图像侧
    def encode(self, x):
        h = self.encoder(x)                 # torch.Size([2, 256, 16, 16])
        encode_image = self.quant_conv(h)   # torch.Size([2, 256, 16, 16])
        quant, emb_loss, (perplexity, min_encodings, image_quant_indices) = self.quantize(encode_image, key='image')
        return encode_image, quant, emb_loss, image_quant_indices

    def decode(self, quant):
        quant = self.post_quant_conv(quant)     # [2, 256, 16, 16]
        s0, s1 = quant.size(0), quant.size(1)
        quant = quant.reshape(s0, s1, -1).permute(0, 2, 1)  # [2, 256, 256]
        i2t_rec = self.text_decoder(quant)     # [8, 256, 49408]
        return i2t_rec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    #文本侧
    def text_encode(self, x, valid_lens):
        h, mask = self.text_encoder(x, valid_lens)              #[bs, l, d]  [8, 256, 256], mask在图像替换时使用
        encode_text = self.quant_linear(h)
        quant, q_loss, text_quant_indices = self.quantize(encode_text, key='text')     #quant: [8, 256, 256]
        return encode_text, quant, q_loss, text_quant_indices, mask

    def text_decode(self, quant, valid_lens):             # [bs, l, d]  [8, 256, 256]
        quant = quant.permute(1, 0, 2)                    # [l, bs, d]
        quant = self.post_quant_tf(quant)                 # self-attention
        quant = quant.permute(1, 2, 0)                    # [8, 256, 256], [bs, d, l]
        quant = quant.reshape(quant.size(0), quant.size(1), 16, 16) # [8, 256, 16, 16]
        t2i_rec = self.decoder(quant)
        return t2i_rec
    
    def constrain_loss(self, encode_image, encode_text, image_quant_indices, text_quant_indices):
        '''对编码器输出结果施加约束'''
        bs, lenth, dim = encode_text.shape
        constrain_loss = torch.abs(encode_image.reshape(bs, dim, -1).permute(0, 2, 1).contiguous() - encode_text.contiguous())
        constrain_loss = torch.mean(constrain_loss, dim=(1, 2), keepdim=True).squeeze(2)
        coef = []
        for i in range(bs):
            list_i = image_quant_indices.tolist()[lenth * i: lenth * (i +1)]
            list_t = text_quant_indices.tolist()[lenth * i: lenth * (i +1)]
            list_all = list_i + list_t
            lenth_i = len(set(list_i))
            lenth_t = len(set(list_t))
            lenth_all = len(set(list_all))
            coef.append(lenth_all /(lenth_i + lenth_t))
        coef = torch.tensor(coef).to(encode_image.device)
        c_loss = torch.matmul(coef, constrain_loss).to(encode_image.device)
        return c_loss

    def forward(self, image, text, valid_lens):
        encode_image, image_quant, image_q_loss, image_quant_indices = self.encode(image)
        encode_text, text_quant,  text_q_loss, text_quant_indices, mask = self.text_encode(text, valid_lens)

        c_loss = self.constrain_loss(encode_image, encode_text, image_quant_indices, text_quant_indices)

        i2t_rec = self.decode(image_quant)                # [8, 3, 256, 256],   [8, 256, 49408]
        t2i_rec  = self.text_decode(text_quant, valid_lens)        #不是整数, [8, 256, 49408],   [8, 3, 256, 256]
        return i2t_rec, t2i_rec, image_q_loss, text_q_loss, c_loss

    def get_input(self, batch, k):
        image = batch['image']                                #[8, 256, 256, 3]
        text, valid_lens = tokenize(list(batch['caption'][0]), context_length=256)  #[8, 1, 256]
        text = text.to(image.device)
        valid_lens = valid_lens.to(image.device)
        if len(text.shape) == 3:
            text = torch.squeeze(text, 1)
        if len(image.shape) == 3:
            image = image[..., None]
        image = image.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return image.float(), text, valid_lens                            #[8, 3, 256, 256], [8, 256]

    def training_step(self, batch, batch_idx, optimizer_idx):
        image, text, valid_lens = self.get_input(batch, self.image_key)
        i2t_rec, t2i_rec, image_q_loss, text_q_loss, c_loss = self(image, text, valid_lens)

        if optimizer_idx == 0:
            # autoencode
            loss, log_dict = self.loss(image, text, i2t_rec, t2i_rec, image_q_loss, text_q_loss, valid_lens, c_loss, optimizer_idx, self.global_step,   #返回loss和日志形式的字典
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return loss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(image, text, i2t_rec, t2i_rec, image_q_loss, text_q_loss, valid_lens, c_loss, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        image, text, valid_lens = self.get_input(batch, self.image_key) #torch.Size([3, 3, 256, 256]), torch.Size([3, 1, 77])
        i2t_rec, t2i_rec, image_q_loss, text_q_loss, c_loss  = self(image, text, valid_lens)                           #AE重构图，codebook损失

        loss, log_dict = self.loss(image, text, i2t_rec, t2i_rec, image_q_loss, text_q_loss, valid_lens, c_loss, 0, self.global_step,       #生成器的验证损失，以及字典形式的损失日志
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(image, text, i2t_rec, t2i_rec, image_q_loss, text_q_loss, valid_lens, c_loss, 1, self.global_step,   #判别器的验证损失日志
                                            last_layer=self.get_last_layer(), split="val")
        
        #log：像是TensorBoard等log记录器，对于每个log的标量，都会有一个相对应的横坐标，它可能是batch number或epoch number
        #on_step就表示把这个log出去的量的横坐标表示为当前batch，而on_epoch则表示将log的量在整个epoch上进行累积后log，横坐标为当前epoch
        self.log("val/loss", loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/disc_total_loss", discloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)             
        self.log_dict(log_dict)              #log_dict：和log函数唯一的区别就是，name和value变量由一个字典替换。表示同时log多个值
        self.log_dict(log_dict_disc)
        return self.log_dict                    #没有返回值限制，不一定非要输出一个val_loss    

    @staticmethod
    def set_bn_eval(module):
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    def freeze(self, module):
        for name, param in module.named_parameters():
            param.requires_grad = False
            
            
    def configure_optimizers(self):
        lr = self.learning_rate
        # self.freeze(self.encoder)
        # self.encoder.apply(self.set_bn_eval)

        # self.freeze(self.decoder)
        # self.decoder.apply(self.set_bn_eval)
        # for name, param in self.decoder.conv_out.named_parameters():
        #     param.requires_grad = True     

        # self.freeze(self.quantize)
        # self.quantize.apply(self.set_bn_eval)

        # self.freeze(self.quant_conv)
        # self.quant_conv.apply(self.set_bn_eval)

        # self.freeze(self.post_quant_conv)
        # self.post_quant_conv.apply(self.set_bn_eval)

        # self.freeze(self.text_encoder)
        # self.text_encoder.apply(self.set_bn_eval)

        # 获取需要更新的参数
        opt_vq_params = []
        opt_disc_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                # 将参数分别添加到对应的优化器参数列表中
                if name.startswith('loss.discriminator'):
                    opt_disc_params.append(param)
                else:
                    opt_vq_params.append(param)

        opt_vq = torch.optim.Adam(opt_vq_params,lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(opt_disc_params,lr=lr, betas=(0.5, 0.9))
        
        return [opt_vq, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        ori_text = []

        for i in range(len(batch['caption'][0])):
            ori_text.append(batch['caption'][0][i])
        image, text, valid_lens = self.get_input(batch, self.image_key)       #tuple(image, text)
        image = image.to(self.device)
        text = text.to(self.device)

        i2t_rec, t2i_rec, image_q_loss, text_q_loss, c_loss = self(image, text, valid_lens)

        encode_image, image_quant, image_q_loss, image_quant_indices = self.encode(image)
        encode_text, text_quant,  text_q_loss, text_quant_indices, mask = self.text_encode(text, valid_lens)
        i2t_without_quant = self.decode(encode_image)                # [8, 3, 256, 256],   [8, 256, 49408]
        t2i_without_quant  = self.text_decode(encode_text, valid_lens) 

        i2t_rec = self.decode(image_quant)                # [8, 3, 256, 256],   [8, 256, 49408]
        t2i_rec  = self.text_decode(text_quant, valid_lens)
        if image.shape[1] > 3:
            # colorize with random projection
            assert i2i_rec.shape[1] > 3
            image = self.to_rgb(image)              #图像的原始输入
            t2i_rec = self.to_rgb(t2i_rec)          #文本重建的图像
            
        # 文本 恢复成自然语言
        i2t_rec = torch.max(i2t_rec, 2)[1]          #返回最大值的索引
        i2t_without_quant = torch.max(i2t_without_quant, 2)[1]
        _tokenizer = _Tokenizer()
        i2t_rec = i2t_rec.cpu().numpy().squeeze().tolist()
        i2t_without_quant = i2t_without_quant.cpu().numpy().squeeze().tolist()
        i2t = []
        i2t_no_cq = []
        for i in range(len(i2t_rec)):
            i2t.append(_tokenizer.decode(i2t_rec[i][:int(valid_lens[i])]))
            i2t_no_cq.append(_tokenizer.decode(i2t_without_quant[i][:int(valid_lens[i])]))

        log["image_inputs"] = image                       #[n,3,256,256]
        log["text_inputs"] = ori_text
        log['t2i_reconstructions'] = t2i_rec
        log['t2i_without_quant'] = t2i_without_quant
        log['i2t_reconstructions'] = i2t
        log['i2t_without_quant'] = i2t_no_cq

        return log


class VQSegmentationModel(VQModel):
    def __init__(self, n_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        total_loss = log_dict_ae["val/total_loss"]
        self.log("val/total_loss", total_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return aeloss

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            # convert logits to indices
            xrec = torch.argmax(xrec, dim=1, keepdim=True)
            xrec = F.one_hot(xrec, num_classes=x.shape[1])
            xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


class VQNoDiscModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None
                 ):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="train")
        output = pl.TrainResult(minimize=aeloss)
        output.log("train/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        output = pl.EvalResult(checkpoint_on=rec_loss)
        output.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.learning_rate, betas=(0.5, 0.9))
        return optimizer


class GumbelVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 temperature_scheduler_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 ):

        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )

        self.loss.n_classes = n_embed
        self.vocab_size = n_embed

        self.quantize = GumbelQuantize(z_channels, embed_dim,
                                       n_embed=n_embed,
                                       kl_weight=kl_weight, temp_init=1.0,
                                       remap=remap)

        self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)   # annealing of temp

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def temperature_scheduling(self):
        self.quantize.temperature = self.temperature_scheduler(self.global_step)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.temperature_scheduling()
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        # encode
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, _, _ = self.quantize(h)
        # decode
        x_rec = self.decode(quant)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log


class EMAVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )
        self.quantize = EMAVectorQuantizer(n_embed=n_embed,
                                           embedding_dim=embed_dim,
                                           beta=0.25,
                                           remap=remap)
    def configure_optimizers(self):
        lr = self.learning_rate
        #Remove self.quantize from parameter list since it is updated via EMA
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []                                           