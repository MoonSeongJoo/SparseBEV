
import onnx
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import wrap_fp16_model,load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from mmdet3d.models import build_model
# from mmengine.exporter import ONNXExporter

from onnxsim import simplify
import importlib
import easydict
import numpy as np
from torch.onnx import OperatorExportTypes

### ref :from bevformer #####
bev_h_ = 200
bev_w_ = 200
_dim_ = 256
num_query=900
num_classes=10
code_size=10
#############################

# batch_size, num_classes, img_h, img_w
default_shapes = easydict.EasyDict(
    batch_size=1,
    img_h=928,
    img_w=1600,
    bev_h=200,
    bev_w=200,
    dim=256,
    num_query=900,
    num_classes=10,
    code_size=10,
    cameras=6,
)

input_meta = easydict.EasyDict(
    image=["batch_size", "cameras", 3, "img_h", "img_w"],
    prev_bev=["bev_h*bev_w", "batch_size", "dim"],
    use_prev_bev=[1],
    can_bus=[18],
    lidar2img=["batch_size", "cameras", 4, 4],
)

output_meta = easydict.EasyDict(
    bev_embed=["bev_h*bev_w", "batch_size", "dim"],
    outputs_classes=["cameras", "batch_size", "num_query", "num_classes"],
    outputs_coords=["cameras", "batch_size", "num_query", "code_size"],
)

def main(opset_version=13,
    verbose=False,
    cuda=True,
    inputs_data=None,
    dynamic_axes = None,):
    
    # register custom module
    importlib.import_module('models')
    importlib.import_module('loaders')

    # 1. 모델 구성
    config_file = './configs/r50_nuimg_704x256.py'
    checkpoint_file = './data/nuscenes/pretrained/sparsebev/r50_nuimg_704x256.pth'
    output_file = './data/nuscenes/pretrained/sparsebev/'

    cfg = Config.fromfile(config_file)

    model = build_model(cfg.model)
    
    # 2. 체크포인트 로드
    load_checkpoint(model, checkpoint_file, map_location='cpu')

    # if cuda:
    #     model.to("cuda")
    # else:
    #     model.to("cpu")

    # onnx_shapes = default_shapes
    # input_shapes = input_meta
    # output_shapes = output_meta
    # dynamic_axes = dynamic_axes
    
    # for key in onnx_shapes:
    #     if key in locals():
    #         raise RuntimeError(f"Variable {key} has been defined.")
    #     locals()[key] = onnx_shapes[key]

    # torch.random.manual_seed(0)
    # inputs = {}
    # for key in input_shapes.keys():
    #     if inputs_data is not None and key in inputs_data:
    #         inputs[key] = inputs_data[key]
    #         if isinstance(inputs[key], np.ndarray):
    #             inputs[key] = torch.from_numpy(inputs[key])
    #         assert isinstance(inputs[key], torch.Tensor)
    #     else:
    #         for i in range(len(input_shapes[key])):
    #             if isinstance(input_shapes[key][i], str):
    #                 input_shapes[key][i] = eval(input_shapes[key][i])
    #         inputs[key] = torch.randn(*input_shapes[key])
    #     if cuda:
    #         inputs[key] = inputs[key].cuda()
    
    # model.forward_test = model.forward_test
    # input_name = list(input_shapes.keys())
    # output_name = list(output_shapes.keys())

    # inputs = tuple(inputs.values())

    # torch.onnx.export(
    #     model,
    #     inputs,
    #     output_file,
    #     input_names=input_name,
    #     output_names=output_name,
    #     export_params=True,
    #     keep_initializers_as_inputs=True,
    #     do_constant_folding=False,
    #     verbose=verbose,
    #     opset_version=opset_version,
    #     dynamic_axes=dynamic_axes,
    #     operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,  
    # )

    # print(f"ONNX file has been saved in {output_file}")

    # 3. ONNX 모델 추출
    input_shape = (1, 48, 3, 256,704)  # 입력 데이터 크기
    img_metas = [{'img_shape': (224, 224), 'pad_shape': (224, 224), 'scale_factor': 1.0}]  # 이미지 메타데이터
    img = torch.randn(input_shape, device=device)
    input_data = [[img], [img_metas]]  # double-nested 형태

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    

    torch.onnx.export(
        model,
        input_data,
        'onnx_model.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None,
        opset_version=11,
        do_constant_folding=True,
        example_outputs=model(dummy_input)
    )

    print ("end of extracting onnx")
    
if __name__ == "__main__":
	main()