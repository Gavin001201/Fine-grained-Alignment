import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer
from taming.modules.transformer.cliptransformer import TextTransformer
from taming.modules.vqvae.quantize import Cluster
from taming.data.base import tokenize

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 ctconfig,
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
        self.quant_W = nn.Sequential(nn.Linear(ctconfig["width"], embed_dim, 1),       # 从文本侧的宽度映射到coodbook的宽度
                                     nn.ReLU())
        self.post_quant_W = nn.Sequential(nn.Linear(embed_dim, 2*embed_dim),           # 全连接以实现self-attention
                                          nn.ReLU(),
                                          nn.Linear(2*embed_dim, embed_dim),
                                          nn.ReLU())
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

    # 图像侧
    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h, key='image')
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        #图像->图像
        i2i_rec = self.decoder(quant)                      # [8, 3, 256, 256]
        #图像->文本
        i2t_rec = self.decoder(quant, key='text')     # [8, 256, 49408])
        return i2i_rec, i2t_rec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    #文本侧
    def text_encode(self, x):
        h, mask = self.text_encoder(x)              #[bs, l, d]  [8, 256, 256], mask在图像替换时使用
        h = self.quant_W(h)
        quant, q_loss, codebook_indices = self.quantize(h, key='text')     #quant: [8, 256, 256]
        return quant, q_loss, codebook_indices, mask

    def text_decode(self, quant):             # [bs, l, d]  [8, 256, 256]
        quant = quant.permute(0, 2, 1)        # [bs, d, l]  [8, 256, 256]
        quant = self.post_quant_W(quant)      # 这里以全连接实现 self-attention
        quant = quant.reshape(quant.size(0), quant.size(1), 16, 16)             #[bs, d, H, W],   [8, 256, 16, 16]

        #文本->文本
        t2t_rec = self.decoder(quant, key='text')                              # [8, 256, 49408]

        #文本->图像
        t2i_rec = self.decoder(quant)
        return t2t_rec, t2i_rec
    
    def forward(self, image, text):
        image_quant, image_q_loss, _ = self.encode(image)
        text_quant,  text_q_loss, _, mask = self.text_encode(text)

        # image_quant, text_quant = self.i_t_cluster(image_quant, text_quant, mask)           # [8, 256, 16, 16],  [8, 256, 256]

        i2i_rec, i2t_rec = self.decode(image_quant)              # [8, 3, 256, 256],   [8, 256, 49408]
        t2t_rec, t2i_rec  = self.text_decode(text_quant)        #不是整数, [8, 256, 49408],   [8, 3, 256, 256]
        return i2i_rec, i2t_rec, image_q_loss,    t2t_rec, t2i_rec, text_q_loss         # q_loss 是文本与图像相互替换时的损失

    def get_input(self, batch, k):
        image = batch['image']                                #[8, 256, 256, 3]
        text = tokenize(list(batch['caption'][0]), context_length=256).to(image.device)                               #[8, 1, 256]
        if len(text.shape) == 3:
            text = torch.squeeze(text, 1)
        if len(image.shape) == 3:
            image = image[..., None]
        image = image.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return image.float(), text                            #[8, 3, 256, 256], [8, 256]

    def training_step(self, batch, batch_idx, optimizer_idx):
        image, text = self.get_input(batch, self.image_key)
        i2i_rec, i2t_rec, image_q_loss, t2t_rec, t2i_rec, text_q_loss = self(image, text)

        if optimizer_idx == 0:
            # autoencode
            loss, log_dict = self.loss(image_q_loss, image, i2i_rec, i2t_rec, optimizer_idx, self.global_step, text, t2i_rec, t2t_rec, text_q_loss,  #返回loss和日志形式的字典
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return loss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(image_q_loss, image, i2i_rec, i2t_rec, optimizer_idx, self.global_step, text, t2i_rec, t2t_rec, text_q_loss,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        image, text = self.get_input(batch, self.image_key) #torch.Size([3, 3, 256, 256]), torch.Size([3, 1, 77])
        i2i_rec, i2t_rec, image_q_loss, t2t_rec, t2i_rec, text_q_loss = self(image, text)                           #AE重构图，codebook损失

        loss, log_dict = self.loss(image_q_loss, image, i2i_rec, i2t_rec, 0, self.global_step, text, t2i_rec, t2t_rec, text_q_loss,       #生成器的验证损失，以及字典形式的损失日志
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(image_q_loss, image, i2i_rec, i2t_rec, 1, self.global_step, text, t2i_rec, t2t_rec, text_q_loss,  #判别器的验证损失日志
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

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_vq = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters())+
                                  list(self.text_encoder.parameters())+
                                  list(self.quant_W.parameters())+
                                  list(self.post_quant_W.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))        
        return [opt_vq, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        image, text = self.get_input(batch, self.image_key)       #tuple(image, text)
        image = image.to(self.device)
        text = text.to(self.device)
        i2i_rec, i2t_rec, image_q_loss , t2t_rec, t2i_rec, text_q_loss= self(image, text)
        if image.shape[1] > 3:
            # colorize with random projection
            assert i2i_rec.shape[1] > 3
            image = self.to_rgb(image)              #图像的原始输入
            i2i_rec = self.to_rgb(i2i_rec)          #图像重建的图像
            t2i_rec = self.to_rgb(t2i_rec)          #文本重建的图像
        log["inputs"] = image                       #[n,3,256,256]
        log["i2i_reconstructions"] = i2i_rec        #[n,3,256,256]
        log['t2i_reconstructions'] = t2i_rec
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