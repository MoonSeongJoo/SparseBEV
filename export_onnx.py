import argparse
import subprocess
import time
from pathlib import Path
import importlib

import onnx
import torch
from torch import nn
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import wrap_fp16_model
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector,build_model
# from mmdet3d.models.necks.view_transformer import ASPP
from mmdet.models.backbones.resnet import BasicBlock
from onnxsim import simplify
# from tools.misc.fuse_conv_bn import fuse_module

from loaders.builder import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet benchmark a model")
    parser.add_argument("--config", help="test config file path")
    parser.add_argument("--checkpoint", default=None, help="checkpoint file")
    parser.add_argument("--samples", default=400, help="samples to benchmark")
    parser.add_argument("--log-interval", default=50, help="interval of logging")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--no-acceleration",
        action="store_true",
        help="Omit the pre-computation acceleration",
    )
    args = parser.parse_args()
    return args


def export_simplify_optimize_onnx(
    onnx_output_prefix,
    model,
    inputs,
    input_names=None,
    output_names=None,
    opset_version=14,
):
    # make dir
    onnx_output_prefix = Path(onnx_output_prefix)
    onnx_output_prefix.mkdir(parents=True, exist_ok=True)
    onnx_output_prefix = str(onnx_output_prefix / onnx_output_prefix.name)

    # Export the model to ONNX format
    torch.onnx.export(
        model,
        args=inputs,
        f=f"{onnx_output_prefix}.onnx",
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False,
    )

    # Simplify the model
    model_simp, check = simplify(f"{onnx_output_prefix}.onnx")
    assert check, "Simplified ONNX model could not be validated"
    onnx.checker.check_model(model_simp)
    onnx.save(model_simp, "/workspace/sparsebev/data/nuscenes/pretrained/sparsebev/AdaptiveMixing_simp.onnx")
    print(f">> Exported ONNX model to {onnx_output_prefix}_simp.onnx")

    # Optimize the simplified model
    process = subprocess.run(
        [
            "python3",
            "-m",
            "onnxoptimizer",
            f"{onnx_output_prefix}_simp.onnx",
            f"{onnx_output_prefix}_opt.onnx",
        ],
        check=False,
    )

    if process.returncode == 0:
        print(f">> Exported ONNX model to {onnx_output_prefix}_opt.onnx")
        return True
    else:
        print("onnxoptimizer execution has failed with exit code:", process.returncode)
        return False


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    cfg.data.test["data_root"] = "/data/nuscenes"
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    data = next(iter(data_loader))
    for key in data.keys():
        print(f"{key=}, {type(data[key][0])=} {len(data[key])}")

    cfg.model.train_cfg = None
    # register custom module
    importlib.import_module('models')
    importlib.import_module('loaders')
    # cfg.model.align_after_view_transfromation = True
    # if not args.no_acceleration:
    #     cfg.model.img_view_transformer.accelerate = True # 필요없음 
    # model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    model = build_model(cfg.model) # sparsebev 는 bulid_model로 register
    # fp16_cfg = cfg.get("fp16", None)
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)
    # if args.fuse_conv_bn:
    #     model = fuse_module(model)  # 필요없음

    # model = MMDataParallel(model, device_ids=[0])
    model.cuda()
    model.eval()

    # bev_encoder_input = {}

    # def save_outputs_hook(module, input):
    #     bev_encoder_input["bev_encoder"] = input[0].detach()

    # model.module.pts_bbox_head.register_forward_pre_hook(save_outputs_hook) # 얘도 필요없는 것 같음 지금은 

    # inputs = [d.cuda() for d in data["img"][0]]
    # inputs = [d.cuda() for d in data["img"][0].data] # 이것도 필요없는것 같은데 ...  
    # inputs_meta = data["img_metas"][0].data[0]
    
    # with torch.no_grad():
    #     feat_prev, inputs = model.module.extract_img_feat(
    #         inputs, pred_prev=True, img_metas=None
    #     )
    # data["img_inputs"][0] = inputs

    print("prepared data for inference...")

    model.forward = model.simple_test_online
    img = data["img"][0].data[0].to('cuda')
    # img =[d.cuda() for d in data["img"][0].data[0]]
    img_metas = data["img_metas"][0].data[0]

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        # model(
        #     return_loss=False,
        #     rescale=True,
        #     sequential=True,
        #     feat_prev=feat_prev,
        #     **data,
        # )   
        model(
            img_metas,
            img,
            rescale=False,
        )
    # torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    print(f"Elapsed time: {elapsed}s fps: {1/elapsed}")
    print(f"inference end")
    
    
    # x = torch.randn(1, 512, 16, 44)
    
    ### self-attn extract #####
    # model.forward = model.simple_test
    # model.forward = model.pts_bbox_head.transformer.decoder.decoder_layer.self_attn.inner_forward
    # input_img = data['img'][0].data[0]
    # input_img_metas = data['img_metas'][0].data[0]
    # x = torch.randn(1, 900,10)
    # y = torch.randn(1, 900,256)
    # z = None

    # torch.onnx.export(
    #     model,
    #     (x , y , z),
    #     'onnx_model.onnx',
    #     input_names=['query_bbox','query_feat','pre_attn_mask'],
    #     output_names=['output'],
    #     dynamic_axes=None,
    #     opset_version=11,
    #     do_constant_folding=True,  
    # )
    ### end of self-attn extract ####
    
    # sparsebevsampling_nn = SparseBEVSampling()
    # sparsebevsampling_nn.eval()
    # sparsebevsampling_nn.forward = sparsebevsampling_nn.inner_forward

#     AdaptiveMixing_nn = AdaptiveMixing(256,32)
#     AdaptiveMixing_nn.eval()
#     AdaptiveMixing_nn.forward = AdaptiveMixing_nn.inner_forward
    
#     input_mlvl_feats =[]
#     input_query_bbox = torch.randn(1, 900,10)
#     input_query_feat = torch.randn(1, 900,256)
#     mlvl_feats_temp = torch.randn(32,6,64,176,64)
#     for i in range(4):
#         input_mlvl_feats.append(mlvl_feats_temp)
#     input_time_diff = torch.randn(1,1,8,1)
#     input_lidar2img = torch.randn(1,48,4,4)
#     input_img_enc = torch .randn(900,4,32,64)

#     onnx_output_prefix = "AdaptiveMixing_nn"
#     export_simplify_optimize_onnx(
#         onnx_output_prefix,
#         AdaptiveMixing_nn,
#         (input_img_enc,input_query_feat),
#         input_names=['img_enc','query_feat'],
#         output_names=['out'],
#     )

#     print ("end of extracting onnx")
    
# from mmcv.runner import BaseModule    
# from models.sparsebev_sampling import sampling_4d, make_sample_points
# class SparseBEVSampling(BaseModule):
#     """Adaptive Spatio-temporal Sampling"""
#     def __init__(self, embed_dims=256, num_frames=4, num_groups=4, num_points=8, num_levels=4, pc_range=[], init_cfg=None):
#         super().__init__(init_cfg)

#         self.num_frames = 8
#         self.num_points = 4
#         self.num_groups = 4
#         self.num_levels = 4
#         self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

#         self.sampling_offset = nn.Linear(embed_dims, self.num_groups * self.num_points * 3)
#         self.scale_weights = nn.Linear(embed_dims, self.num_groups * self.num_points * self.num_levels)

#     def init_weights(self):
#         bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
#         nn.init.zeros_(self.sampling_offset.weight)
#         nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)

#     def inner_forward(self, query_bbox):
#         '''
#         query_bbox: [B, Q, 10]
#         query_feat: [B, Q, C]
#         '''
#         # B, Q = query_bbox.shape[:2]
#         # image_h, image_w, _ = (256,704,3)

#         # sampling offset of all frames
#         # sampling_offset = self.sampling_offset(query_feat)
#         # sampling_offset = sampling_offset.view(B, Q, self.num_groups * self.num_points, 3)
#         sampling_offset = torch.randn(1,900,16,3)
#         sampling_points = make_sample_points(query_bbox, sampling_offset, self.pc_range)  # [B, Q, GP(# of points), 3]
#         # sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups, self.num_points, 3)
#         # sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups*self.num_points, 3)
#         # sampling_points = sampling_points.expand(B, Q, self.num_frames, self.num_groups*self.num_points, 3) #[B, Q, F, GP, 3]

#         # warp sample points based on velocity
#         # time_diff = img_metas[0]['time_diff']  # [B, F]
#         # time_diff = time_diff[:, None, :, None]  # [B, 1, F, 1]
#         # time_diff = torch.randn(1,1,8,1)
#         # vel = query_bbox[..., 8:].detach()  # [B, Q, 2]
#         # vel = vel[:, :, None, :]  # [B, Q, 1, 2]
#         # dist = vel * time_diff  # [B, Q, F, 2]
#         # dist = dist[:, :, :, None, :]  # [B, Q, F, 1, 2]
#         # dist = dist[:, :, :, None, None, :]  # [B, Q, F, 1, 1, 2]
        
#         # sampling point에 dist 합치기
#         # sampling_points = torch.cat([
#         #     sampling_points[..., 0:2] - dist,
#         #     sampling_points[..., 2:3]
#         # ], dim=-1)

#         # # scale weights
#         # scale_weights = self.scale_weights(query_feat)
#         # # scale_weights = self.scale_weights(query_feat).view(B, Q, self.num_groups, 1, self.num_points, self.num_levels)
#         # scale_weights = torch.softmax(scale_weights, dim=-1)
#         # # scale_weights = scale_weights.expand(B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels)

#         return sampling_points
#         # # sampling
#         # sampled_feats = sampling_4d(
#         #     sampling_points,
#         #     mlvl_feats,
#         #     scale_weights,
#         #     # img_metas[0]['lidar2img'],
#         #     lidar2img,
#         #     image_h, image_w
#         # )  # [B, Q, G, FP, C]

#         # return sampled_feats
# import torch.nn.functional as F
# class AdaptiveMixing(nn.Module):
#     """Adaptive Mixing"""
#     def __init__(self, in_dim, in_points, n_groups=1, query_dim=None, out_dim=None, out_points=None):
#         super(AdaptiveMixing, self).__init__()

#         out_dim = out_dim if out_dim is not None else in_dim
#         out_points = out_points if out_points is not None else in_points
#         query_dim = query_dim if query_dim is not None else in_dim

#         self.query_dim = 256
#         self.in_dim = 256
#         self.in_points = 32
#         self.n_groups = 4
#         self.out_dim = 256
#         self.out_points = 128

#         self.eff_in_dim = in_dim // self.n_groups
#         self.eff_out_dim = out_dim // self.n_groups

#         self.m_parameters = self.eff_in_dim * self.eff_out_dim
#         self.s_parameters = self.in_points * self.out_points
#         self.total_parameters = self.m_parameters + self.s_parameters

#         self.parameter_generator = nn.Linear(self.query_dim, self.n_groups * self.total_parameters)
#         self.out_proj = nn.Linear(self.eff_out_dim * self.out_points * self.n_groups, self.query_dim)
#         self.act = nn.ReLU(inplace=True)

#     @torch.no_grad()
#     def init_weights(self):
#         nn.init.zeros_(self.parameter_generator.weight)

#     def inner_forward(self, x, query):
#         # B, Q, G, P, C = x.shape
#         # assert G == self.n_groups
#         # assert P == self.in_points
#         # assert C == self.eff_in_dim
#         B, Q, G, P, C = (1,900,4,32,64)

#         '''generate mixing parameters'''
#         params = self.parameter_generator(query)
#         params = params.reshape(B*Q, G, -1)
#         # out = x.reshape(B*Q, G, P, C)
#         out = x

#         M, S = params.split([self.m_parameters, self.s_parameters], 2)
#         M = M.reshape(B*Q, G, self.eff_in_dim, self.eff_out_dim)
#         S = S.reshape(B*Q, G, self.out_points, self.in_points)

#         '''adaptive channel mixing'''
#         out = torch.matmul(out, M)
#         out = F.layer_norm(out, [out.size(-2), out.size(-1)])
#         out = self.act(out)

#         '''adaptive point mixing'''
#         out = torch.matmul(S, out)  # implicitly transpose and matmul
#         out = F.layer_norm(out, [out.size(-2), out.size(-1)])
#         out = self.act(out)

#         '''linear transfomation to query dim'''
#         out = out.reshape(B, Q, -1)
#         out = self.out_proj(out)
#         out = query + out

#         return out


if __name__ == "__main__":
    main()