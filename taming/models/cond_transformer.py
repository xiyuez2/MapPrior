import os, math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from taming.util import vis_layer_6
from matplotlib import pyplot as plt 

from transformer import instantiate_from_config
from taming.modules.util import SOSProvider
import cv2

import mmcv
from torchpack.utils.config import configs
from torchpack import distributed as dist
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.utils import recursive_eval
from mmcv.parallel.data_container import DataContainer
import matplotlib.pyplot as plt

def take_threshold(data,threshold):
    # data: batch_size, 6, w, h 
    for i in range(6):
        data[:,i] = data[:,i] > threshold[i]
    return data

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class DenoiseTransformer_bev(pl.LightningModule):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 cond_stage_config,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 cond_stage_key="depth",
                 downsample_cond_size=-1,
                 pkeep=1.0,
                 sos_token=0,
                 unconditional=False,
                 out_loss_w = 100,
                 config_filename = "./bev_lib/configs/nuscenes/seg/lidar-centerpoint-bev128.yaml",
                 is_train = None
                 ):
        super().__init__()
        self.is_train = is_train
        self.out_loss_w = out_loss_w
        self.be_unconditional = unconditional
        self.sos_token = sos_token
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.init_first_stage_from_ckpt(first_stage_config)
        self.init_cond_stage_from_ckpt(cond_stage_config)
        self.save_dir = ""
        self.topk = 0
        self.sample_steps  = 0

        if permuter_config is None:
            permuter_config = {"target": "taming.modules.transformer.permuter.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)
        # self.feature_mdoel = feature_ex()

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.downsample_cond_size = downsample_cond_size
        self.pkeep = pkeep
        print("using pkeep:", self.pkeep)
        ## init bev backbone model
        # config_filename = "./bev_lib/configs/nuscenes/seg/lidar-centerpoint-bev128.yaml"
        print("in init model, is_train: ", is_train)
        self.nos_threshold = []
        if 'fusion' in config_filename:
            print("using fusion config nos_threshold in model")
            checkpoint_name = 'bev_lib/pretrained/bevfusion-seg.pth'
            self.nos_threshold = (0.45, 0.45, 0.45, 0.45, 0.45, 0.45)

        elif 'lidar' in config_filename:
            print("using lidar-only config nos_threshold in model")
            checkpoint_name = 'bev_lib/pretrained/lidar-only-seg.pth'
            self.nos_threshold = (0.50, 0.45, 0.45, 0.40, 0.35, 0.40) #(0.5,0.5,0.5,0.5,0.5,0.5) 
        else:
            print("using camera-only config nos_threshold in model")
            checkpoint_name = 'bev_lib/pretrained/camera-only-seg.pth'
            self.nos_threshold = (0.45, 0.45, 0.45, 0.45, 0.4, 0.45) #(0.45, 0.4, 0.35, 0.35, 0.45, 0.40) 

        print("using nos_threshold:", self.nos_threshold)
        print("using out loss weight:", self.out_loss_w)
        print("config: ", config_filename)
        
        configs.load(config_filename, recursive=True)
        cfg = Config(recursive_eval(configs), filename=config_filename)
        self.bev_cfg = cfg
        dataset = build_dataset(cfg.data.test)
        
        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        
        
        checkpoint = load_checkpoint(model,checkpoint_name,map_location="cpu")
        if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            model.CLASSES = dataset.CLASSES

        model = MMDataParallel(model, device_ids=[0])
        
        model = model.eval()
        model.train = disabled_train
        self.bev_backbone = model
        # self.bev_backbone_map = model
        
        ## define bev model for map


        if self.out_loss_w < 0:
            print('no out loss')

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    def init_cond_stage_from_ckpt(self, config):
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        elif config == "__is_unconditional__" or self.be_unconditional:
            print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
                  f"Prepending {self.sos_token} as a sos token.")
            self.be_unconditional = True
            self.cond_stage_key = self.first_stage_key
            self.cond_stage_model = SOSProvider(self.sos_token)
        else:
            model = instantiate_from_config(config)
            model = model.eval()
            model.train = disabled_train
            self.cond_stage_model = model

    def forward(self, x, c, f):
        # one step to produce the logits
        _, z_indices = self.encode_to_z(x)
        _, c_indices = self.encode_to_c(c)

        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices
        cz_indices = torch.cat((c_indices, a_indices), dim=1)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction
        logits, _ = self.transformer(cz_indices[:, :-1], f)
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, c_indices.shape[1]-1:]

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, x, c, f, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None):
        x = torch.cat((c,x),dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        if self.pkeep <= 0.0:
            # one pass suffices since input is pure noise anyway
            assert len(x.shape)==2
            noise_shape = (x.shape[0], steps-1)
            #noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
            noise = c.clone()[:,x.shape[1]-c.shape[1]:-1]
            x = torch.cat((x,noise),dim=1)
            logits, _ = self.transformer(x,f)
            # take all logits for now and scale by temp
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0]*shape[1],shape[2])
                ix = torch.multinomial(probs, num_samples=1)
                probs = probs.reshape(shape[0],shape[1],shape[2])
                ix = ix.reshape(shape[0],shape[1])
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # cut off conditioning
            x = ix[:, c.shape[1]-1:]
        else:
            for k in range(steps):
                callback(k)
                assert x.size(1) <= block_size # make sure model can see conditioning
                x_cond = x 
                logits, _ = self.transformer(x_cond,f)
                
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)
            # cut off conditioning
            x = x[:, c.shape[1]:]
        return x

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        indices = self.permuter(indices)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_c(self, c):
        if self.downsample_cond_size > -1:
            c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
        quant_c, _, [_,_,indices] = self.cond_stage_model.encode(c)
        if len(indices.shape) > 2 or len(indices.shape) == 1:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):
        log = dict()

        N = 4
        if lr_interface:
            x, c, f = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c, f = self.get_xc(batch, N)
        x = x.to(device=self.device)
        c = c.to(device=self.device)

        quant_z, z_indices = self.encode_to_z(x)
        quant_c, c_indices = self.encode_to_c(c)

        # sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,f,
                                   steps=z_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)

        # det sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,f,
                                   steps=z_indices.shape[1],
                                   sample=False,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_det = self.decode_to_img(index_sample, quant_z.shape)

        # reconstruction
        x_rec = self.decode_to_img(z_indices, quant_z.shape)

        log["inputs"] = x
        log["reconstructions"] = x_rec 

        if self.cond_stage_key in ["objects_bbox", "objects_center_points"]:
            figure_size = (x_rec.shape[2], x_rec.shape[3])
            dataset = kwargs["pl_module"].trainer.datamodule.datasets["validation"]
            label_for_category_no = dataset.get_textual_label_for_category_no
            plotter = dataset.conditional_builders[self.cond_stage_key].plot
            log["conditioning"] = torch.zeros_like(log["reconstructions"])
            for i in range(quant_c.shape[0]):
                log["conditioning"][i] = plotter(quant_c[i], label_for_category_no, figure_size)
            log["conditioning_rec"] = log["conditioning"]
        elif self.cond_stage_key != "image":
            cond_rec = self.cond_stage_model.decode(quant_c)
            if self.cond_stage_key == "segmentation":
                # get image from segmentation mask
                num_classes = cond_rec.shape[1]

                c = torch.argmax(c, dim=1, keepdim=True)
                c = F.one_hot(c, num_classes=num_classes)
                c = c.squeeze(1).permute(0, 3, 1, 2).float()
                c = self.cond_stage_model.to_rgb(c)

                cond_rec = torch.argmax(cond_rec, dim=1, keepdim=True)
                cond_rec = F.one_hot(cond_rec, num_classes=num_classes)
                cond_rec = cond_rec.squeeze(1).permute(0, 3, 1, 2).float()
                cond_rec = self.cond_stage_model.to_rgb(cond_rec)
            log["conditioning_rec"] = cond_rec 
            log["conditioning"] = c

        log["samples_nopix"] = x_sample_nopix
        log["samples_det"] = x_sample_det
        return log

    def get_input(self, key, batch):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double or x.dtype == torch.int:
            x = x.float()
        return x

    def get_xc(self, batch, N=None):

        nos, gt, features = self.bev_backbone(return_loss=False, rescale=True,**batch)
        nos = take_threshold(nos,self.nos_threshold)
        
        
        #return None
        if N is not None:
            gt = gt[:N]
            nos = nos[:N]
            features = features[:N]
        gt = gt.float()
        nos = nos.float()
        features = features.float()
        
        # print('after input',gt.size(),nos.size(),features.size())
        return gt, nos, features

    def shared_step(self, batch, batch_idx):
        x, c, f = self.get_xc(batch)

        out_loss_w = self.out_loss_w
        logits, target = self(x, c, f)

        logits_idx_one_hot = F.gumbel_softmax(logits, tau=1.0, dim=2, hard=True)
        b,wh,c = logits_idx_one_hot.shape
        logits_idx_one_hot = logits_idx_one_hot.view(b,int(wh**0.5),int(wh**0.5),c)
        code_book = self.first_stage_model.quantize.embedding.weight
        codes = torch.matmul(logits_idx_one_hot,  code_book)
        codes = codes.permute(0, 3, 1, 2).contiguous()
        reconstructions = self.first_stage_model.decode(codes)
        end_loss = torch.mean(torch.abs(x.contiguous() - reconstructions.contiguous()))

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1)) + end_loss * out_loss_w
        return loss

    @torch.no_grad()
    def log_det_res_bev(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):

        N = 4
        if lr_interface:
            x, c, f = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c, f = self.get_xc(batch, N)
        x = x.to(device=self.device)
        c = c.to(device=self.device)
        f = f.to(device=self.device)

        quant_z, z_indices = self.encode_to_z(x)
        quant_c, c_indices = self.encode_to_c(c)

        # det sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,f,
                                   steps=z_indices.shape[1],
                                   sample=True, ## warning: change this!!
                                   callback=callback if callback is not None else lambda k: None,top_k=top_k)
        x_sample_det = self.decode_to_img(index_sample, quant_z.shape)

        return x_sample_det,x,c


    def training_step(self, batch, batch_idx):
        # self.bev_backbone = self.bev_backbone.eval()
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx % 100 == 0:
            # to save time val only computes a subset of val set
            loss = self.shared_step(batch, batch_idx)
        else:
            loss = 0
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = 0

        top_ks = [self.topk] #[1]
        steps = self.sample_steps
        out_dir = self.save_dir #'out_loss_res'
        if not top_ks[0] == 1:
            print('saving sampling result using top ',top_ks[0],' code')
        for top_k in top_ks:
            for step in range(steps):
                res,x,c = self.log_det_res_bev(batch,top_k=top_k)
                res = res.detach().cpu().numpy()[0]
                c = c.detach().cpu().numpy()[0]
                
                # decide cond name
                if step == 0:
                    cond_name = out_dir + "/res_" + str(batch_idx)
                else:
                    cond_name = out_dir + "/res_" + str(batch_idx) + "step_" + str(step)
                # saving
                np.save(cond_name, res)
                if step == 0:
                    x = x.detach().cpu().numpy()[0]
                    np.save(out_dir + "/gt_"+str(batch_idx),x)
                    

                if batch_idx < 25:
                    # save visiualization
                    vis_res = vis_layer_6(res)
                    print('saving to:', cond_name)
                    cv2.imwrite(cond_name +'.png', np.uint8(vis_res[:,:,::-1]))
                    if step == 0:
                        vis_x = vis_layer_6(x)
                        vis_c = vis_layer_6(c)
                        cv2.imwrite(out_dir + "/nos_" + str(batch_idx) +'.png', np.uint8(vis_c[:,:,::-1]))
                        cv2.imwrite(out_dir + "/gt_" + str(batch_idx) +'.png', np.uint8(vis_x[:,:,::-1]))
        return loss

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.GroupNorm)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer
