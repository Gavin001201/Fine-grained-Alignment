import numpy as np
import matplotlib.pyplot as plt
import argparse, os, sys
sys.path.append('/mnt/workspace/Project/Fine-grained-Alignment')
sys.path.remove('/mnt/workspace/Project/taming-transformers2.0')
import torch
from omegaconf import OmegaConf
from PIL import Image
from main import instantiate_from_config
from torch.utils.data.dataloader import default_collate
from taming.modules.tokenizer.simple_tokenizer import SimpleTokenizer
import html
import ftfy
import regex as re

def save_img(xstart, fname):
    I = (xstart.clip(0,1)[0]*255).astype(np.uint8)
    Image.fromarray(I).save(fname)

rescale = lambda x: (x + 1.) / 2.

def bchw_to_st(x):
    return rescale(x.detach().cpu().numpy().transpose(0,2,3,1))

def split_image(image, num_rows, num_cols):
    height, width = image.shape[1], image.shape[2]
    row_size = height // num_rows
    col_size = width // num_cols

    regions = []
    for row in range(num_rows):
        for col in range(num_cols):
            region = image[0, row*row_size:(row+1)*row_size, col*col_size:(col+1)*col_size]
            regions.append(region)

    return regions

def visualize_image(image, num_rows, num_cols, characters, index, title):
    regions = split_image(image, num_rows, num_cols)

    fig, ax = plt.subplots()
    ax.imshow(image.squeeze().astype(np.uint8))
    ax.axis('off')

    for i, region in enumerate(regions):
        row = i // num_cols
        col = i % num_cols
        ax.text(col * (1/num_cols) + 0.5/num_cols, row * (1/num_rows) + 0.5/num_rows, characters[i],
                color='white', fontsize=8, ha='center', va='center', transform=ax.transAxes)
        
    visaul_output = os.getcwd() + '/Project/Fine-grained-Alignment/scripts/visual_output'
    if not os.path.exists(visaul_output):
        os.mkdir(visaul_output)

    plt.title(title)
    plt.savefig(visaul_output + '/{}.png'.format(index))  # 保存图片为PNG格式

class Tokenizer(SimpleTokenizer):
    def __init__(self):
        super().__init__()

    def whitespace_clean(self, text):
        text = re.sub(r'\s+', ' ', text)        #将'\s+'(任何空白字符)换成' '
        text = text.strip()                     #去除前后空格
        return text

    def basic_clean(self, text):
        text = ftfy.fix_text(text[0])                  #给定 Unicode 文本作为输入，修复其中的不一致和小故障
        
        #html.unescape，将字符串 s 中的所有命名和数字字符引用(例如 > ，& # 62; ，& x3e;)转换为相应的 unicode 字符
        text = html.unescape(html.unescape(text))
        return text.strip()

    def encode(self, text):     #编码
        bpe_tokens = []
        bpe_text = []
        text = self.whitespace_clean(self.basic_clean(text)).lower()  #修复不一致，去除空格

        for token in re.findall(self.pat, text):    # 按空格分词
            bb = token.encode('utf-8')
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_text += self.bpe(token).split(' ')
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))

        for i, item in enumerate(bpe_text):
            bpe_text[i] = re.sub(r'</w>', '', item)

        bpe_text.insert(0, 'bos')
        bpe_text.append('eos')

        return bpe_tokens, bpe_text
    

@torch.no_grad()
def run(model, dsets):
    dset = dsets.datasets['validation']
    indices = list(range(1000))
    for i in indices:
        example = default_collate([dset[i]])
        example['caption'] = [('A pan of homemade pizza with vegetable and cheese toppings.',)]
        input_text = example['caption'][0]

        tokenizer = Tokenizer()
        image, text, valid_lens = model.get_input(example, "image")
        bpe_tokens, bpe_text =  tokenizer.encode(list(example['caption'][0]))
        image = image.to(model.device)
        text = text.to(model.device)
        valid_lens = valid_lens.to(model.device)

        image_quant, image_quant_loss, info, image_hidden = model.encode(image)
        text_quant,  text_quant_loss, text_codebook_indices, text_hidden, mask = model.text_encode(text, valid_lens, image_hidden)

        text_indices = text_codebook_indices.tolist()[:int(valid_lens)]
        image_indices = info[2].tolist()
        candidate = []
        for j in range(256):
            if image_indices[j] in text_indices:
                index = text_indices.index(image_indices[j])
                candidate.append(bpe_text[index - 1])
            else:
                candidate.append('')

        i2i_rec, i2t_rec = model.decode(image_quant)              # [8, 3, 256, 256],   [8, 256, 49408]
        t2t_rec, t2i_rec  = model.text_decode(text_quant, valid_lens)        #不是整数, [8, 256, 49408],   [8, 3, 256, 256]

        t2t_rec = torch.max(t2t_rec, 2)[1]          #返回最大值的索引
        _tokenizer = SimpleTokenizer()
        t2t_rec = t2t_rec.cpu().numpy().tolist()
        t2t = _tokenizer.decode(t2t_rec[0][1:int(valid_lens[0])-1])

        new_image = bchw_to_st(t2i_rec).clip(0,1)[0]*255

        visualize_image(new_image, 16, 16, candidate, i, input_text[0])

        print(i)
    print('done')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
        default=" ",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=["/mnt/workspace/Project/taming-transformers2.0/configs/custom_vqgan.yaml"],
    )

    return parser


def get_data(config):
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    return data


def load_model_from_config(config, gpu=True, eval_mode=True):
    model = instantiate_from_config(config)
    sd = torch.load(opt.resume, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False) 
    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    
    return {"model": model}

def load_model_and_dset(config, eval_mode):
    dsets = get_data(config)   # calls data.config ...
    model = load_model_from_config(config.model, gpu=gpu, eval_mode=eval_mode)["model"]
    return dsets, model


if __name__ == "__main__":
    # sys.path.append(os.getcwd())
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    dsets, model = load_model_and_dset(config, eval_mode)
    run(model, dsets)