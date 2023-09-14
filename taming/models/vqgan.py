import logging
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm

from main import instantiate_from_config

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer
from taming.modules.vqvae.quantize import Coquant
from taming.data.base import tokenize
from taming.modules.tokenizer.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch.optim.lr_scheduler import StepLR
from taming.lr_scheduler import LambdaWarmUpCosineScheduler
from open_clip.src.open_clip.factory import create_model_and_transforms
from taming.modules.lightdecoder import ImageDecoder, TextDecoder
from torch.optim.lr_scheduler import LambdaLR

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
                 lock_image=False,
                 lock_image_unlocked_groups=None,
                 lock_image_freeze_bn_stats=False,
                 lock_text=False,
                 lock_text_unlocked_layers=None,
                 lock_text_freeze_layer_norm=False,
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.clip, _, _ = create_model_and_transforms(
            'ViT-B-32', pretrained='laion400m_e32', output_dict=True)
        if lock_image:
            self.clip.lock_image_tower(
            unlocked_groups=lock_image_unlocked_groups,
            freeze_bn_stats=lock_image_freeze_bn_stats)
        if lock_text:
            self.clip.lock_text_tower(
            unlocked_layers=lock_text_unlocked_layers,
            freeze_layer_norm=lock_text_freeze_layer_norm)
        
        self.imagedecoder = ImageDecoder(**ddconfig)
        self.textdecoder = TextDecoder()
        self.text_former_linear = nn.Linear(512, 256)
        self.text_latter_linear = nn.Linear(256, 512)
        self.image_former_linear = nn.Linear(512, 256)
        self.image_latter_linear = nn.Linear(256, 512)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
            remap=remap, sane_index_shape=sane_index_shape)

        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            keys = list(sd.keys())
            for k in keys:
                if k != 'quantize.embedding.weight':
                    del sd[k]
            self.load_state_dict(sd, strict=False)

        self.coquant = Coquant(embed_dim)
        self.loss = instantiate_from_config(lossconfig)

        if monitor is not None:
            self.monitor = monitor

    def get_input(self, batch):
        images, texts, valid_lens = batch
        images = images.to(device=self.device, non_blocking=True)
        texts = texts.to(device=self.device, non_blocking=True)

        return images, texts, valid_lens

    def encode(self, images, texts):
        clip_out = self.clip(images, texts)
        image_features = clip_out["image_features"]     # [4, 49, 512]
        text_features = clip_out["text_features"]       # [4, 77, 512]

        return image_features, text_features

    def image_decode(self, image_quant, image_coquant):
        #图像->图像
        i2i_rec = self.imagedecoder(image_quant)        # [4, 49, 32*32*3]
        i2i_rec = self.unpatchify(i2i_rec)
        #文本->图像
        t2i_rec = self.imagedecoder(image_coquant)
        t2i_rec = self.unpatchify(t2i_rec)

        return i2i_rec, t2i_rec

    def text_decode(self, text_quant, text_coquant):
        #文本->文本
        t2t_rec = self.textdecoder(text_quant)
        #图像->文本
        i2t_rec = self.textdecoder(text_coquant)                              # [8, 256, 49408]

        return t2t_rec, i2t_rec
    
    def forward(self, images, texts):
        image_features, text_features = self.encode(images, texts)

        image_features = self.image_former_linear(image_features)
        text_features = self.text_former_linear(text_features)

        image_quant, image_quant_loss, image_codebook_indices = self.quantize(image_features, key='image')
        text_quant, text_quant_loss, text_codebook_indices = self.quantize(text_features, key='text')

        image_coquant, text_coquant, i_coquant_loss, t_coquant_loss = self.coquant(image_quant, text_quant)
     
        image_quant = self.image_latter_linear(image_quant)
        text_quant = self.text_latter_linear(text_quant)
        image_coquant = self.image_latter_linear(image_coquant)
        text_coquant = self.text_latter_linear(text_coquant)

        i2i_rec, t2i_rec = self.image_decode(image_quant, image_coquant)
        t2t_rec, i2t_rec = self.text_decode(text_quant, text_coquant)

        return i2i_rec, t2i_rec, t2t_rec, i2t_rec, image_quant_loss, text_quant_loss, i_coquant_loss, t_coquant_loss


    def training_step(self, batch, batch_idx):
        images, texts, valid_lens = self.get_input(batch)
        i2i_rec, t2i_rec, t2t_rec, i2t_rec, image_quant_loss, text_quant_loss, i_coquant_loss, t_coquant_loss = self(images, texts)

        loss, log_dict = self.loss(image_quant_loss, images, i2i_rec, i2t_rec, self.global_step, texts, t2i_rec, 
            t2t_rec, text_quant_loss, i_coquant_loss, t_coquant_loss, valid_lens, split="train")

        log_dict['lr'] = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, texts, valid_lens = self.get_input(batch)
        i2i_rec, t2i_rec, t2t_rec, i2t_rec, image_quant_loss, text_quant_loss, i_coquant_loss, t_coquant_loss = self(images, texts)

        loss, log_dict = self.loss(image_quant_loss, images, i2i_rec, i2t_rec, self.global_step, texts, t2i_rec, 
            t2t_rec, text_quant_loss, i_coquant_loss, t_coquant_loss, valid_lens, split="val")
        
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)           
        self.log_dict(log_dict)

        return self.log_dict  

    @staticmethod
    def set_bn_eval(module):
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    def freeze(self, module):
        for name, param in module.named_parameters():
            param.requires_grad = False
            

    def configure_optimizers(self):
        lr = self.learning_rate
        # 获取需要更新的参数
        opt_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                # 将参数添加到优化器参数列表中
                opt_params.append(param)
        optimizer = torch.optim.Adam(opt_params,lr=lr, betas=(0.5, 0.9))
        
        if True:
        #     scheduler = LambdaWarmUpCosineScheduler(warm_up_steps=300, lr_min=4.5e-7, 
        #         lr_max=10 * lr, lr_start=4.5e-6, max_decay_steps=2300, verbosity_interval=0)

        #     scheduler = [
        #         {
        #             'scheduler': LambdaLR(optimizer, lr_lambda=scheduler.schedule),
        #             'interval': 'step',
        #             'frequency': 1
        #         }]

        #     return [optimizer],  scheduler
        # else:

            return optimizer

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        _tokenizer = _Tokenizer()
        origin_texts = []
        images, texts, valid_lens = self.get_input(batch)
        captions = texts.cpu().numpy().tolist()
        for i in range(batch[2].size(0)):
            caption = _tokenizer.decode(captions[i][1:int(valid_lens[i]) - 1])
            origin_texts.append(caption.rstrip())
        i2i_rec, t2i_rec, t2t_rec, i2t_rec, image_quant_loss, text_quant_loss, i_coquant_loss, t_coquant_loss = self(images, texts)

        # 文本恢复成自然语言
        t2t_rec = torch.max(t2t_rec, 2)[1]          #返回最大值的索引
        i2t_rec = torch.max(i2t_rec, 2)[1]
        t2t_rec = t2t_rec.cpu().numpy().tolist()
        i2t_rec = i2t_rec.cpu().numpy().tolist()
        t2t = []
        i2t = []
        for i in range(len(t2t_rec)):
            t2t.append(_tokenizer.decode(t2t_rec[i][1:int(valid_lens[i]) - 1]).rstrip())
            i2t.append(_tokenizer.decode(i2t_rec[i][1:int(valid_lens[i]) - 1]).rstrip())

        log = dict()
        log["image_inputs"] = images                #[n,3,256,256]
        log["text_inputs"] = origin_texts
        log["i2i_reconstructions"] = i2i_rec        #[n,3,256,256]
        log['t2i_reconstructions'] = t2i_rec
        log['t2t_reconstructions'] = t2t
        log['i2t_reconstructions'] = i2t
       
        return log

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = 32
        h = w = 7
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs