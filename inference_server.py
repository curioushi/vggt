#!/usr/bin/env python3
"""
VGGT HTTP推理服务器

用法:
    python inference_server.py --host 0.0.0.0 --port 22334 --device cpu

API端点:
    POST /inference - 图像推理
    GET /health - 健康检查
    GET /docs - API文档
"""

import os
import sys
import json
import argparse
import base64
import io
import time
import numpy as np
from scipy.ndimage import affine_transform
import torch
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Add vggt to path
sys.path.append("vggt/")

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


class InferenceRequest(BaseModel):
    """推理请求模型"""
    image: str  # Base64编码的图像数据
    image_format: str = "jpeg"  # 图像格式


class InferenceResponse(BaseModel):
    """推理响应模型"""
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


class VGGTInferenceServer:
    """VGGT推理服务器类"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.to_pil = transforms.ToPILImage()
        
    def load_model(self):
        """加载VGGT模型"""
        print("正在加载VGGT模型...")
        self.model = VGGT()
        
        # 下载并加载预训练权重
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        self.model.load_state_dict(torch.hub.load_state_dict_from_url(_URL, map_location='cpu'))
        
        self.model.eval()
        self.model.to(self.device)
        print("模型加载完成")
        
    def decode_base64_image(self, image_data: str, image_format: str) -> Image.Image:
        """解码Base64图像数据"""
        try:
            # 移除可能的data URL前缀
            if image_data.startswith('data:image/'):
                image_data = image_data.split(',')[1]
            
            # 解码Base64数据
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # 转换为RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            return image
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"图像解码失败: {str(e)}")
    
    def encode_image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """将PIL图像编码为Base64字符串"""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def encode_numpy_to_base64(self, array: np.ndarray) -> str:
        """将numpy数组编码为Base64字符串"""
        buffer = io.BytesIO()
        np.save(buffer, array)
        array_bytes = buffer.getvalue()
        return base64.b64encode(array_bytes).decode('utf-8')
    
    def preprocess_image_in_memory(self, image: Image.Image) -> torch.Tensor:
        """在内存中预处理图像"""
        # 保存图像到内存中的临时文件
        temp_buffer = io.BytesIO()
        image.save(temp_buffer, format='PNG')
        temp_buffer.seek(0)
        
        # 创建临时文件路径（实际上不会写入磁盘）
        temp_path = "temp_image.png"
        
        # 使用现有的预处理函数，但传入内存中的图像数据
        # 这里需要修改load_and_preprocess_images函数以支持内存中的图像
        # 或者直接在这里实现预处理逻辑
        
        # 临时解决方案：保存到临时文件，处理完后删除
        with open(temp_path, 'wb') as f:
            f.write(temp_buffer.getvalue())
        
        try:
            # 使用现有的预处理函数
            images, transforms = load_and_preprocess_images([temp_path], mode="pad", return_transforms=True)
            return images.to(self.device), transforms
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def run_inference(self, image: Image.Image) -> Dict[str, Any]:
        """运行模型推理"""
        print("开始预处理图像...")
        
        # 预处理图像
        images, transforms = self.preprocess_image_in_memory(image)
        print(f"预处理完成，图像形状: {images.shape}")
        
        # 运行推理
        print("开始推理...")
        start_time = time.time()
        
        with torch.no_grad():
            predictions = self.model(images)
        
        inference_time = time.time() - start_time
        print(f"推理完成，耗时: {inference_time:.2f}秒")
        
        # 后处理
        print("开始后处理...")
        
        # 转换pose编码为内外参矩阵
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], 
            images.shape[-2:]
        )
        
        # 生成世界坐标点
        depth_map = predictions["depth"]  # (S, H, W, 1)
        
        # 转换tensor为numpy
        depth_map_np = depth_map.cpu().numpy()
        # extrinsic_np = extrinsic.cpu().numpy()
        extrinsic_np = np.eye(4).reshape(1, 4, 4)
        intrinsic_np = intrinsic.cpu().numpy()
        
        # 形状调整
        if len(depth_map_np.shape) == 5:
            depth_map_np = depth_map_np.squeeze(1)
        elif len(depth_map_np.shape) == 4 and depth_map_np.shape[1] == 1:
            depth_map_np = depth_map_np.squeeze(1)[..., None]
        elif len(depth_map_np.shape) == 3:
            depth_map_np = depth_map_np[..., None]
        
        if len(extrinsic_np.shape) == 4:
            extrinsic_np = extrinsic_np.squeeze(1)
        if len(intrinsic_np.shape) == 4:
            intrinsic_np = intrinsic_np.squeeze(1)
        
        world_points = unproject_depth_map_to_point_map(
            depth_map_np, 
            extrinsic_np, 
            intrinsic_np
        )

        transform = transforms[0]
        transform_inv = np.linalg.inv(transform)
        depth_map_np = depth_map_np.squeeze(0).squeeze(-1)
        world_points = world_points.squeeze(0)
        world_points_conf = predictions.get("world_points_conf", torch.zeros_like(depth_map)).cpu().numpy().squeeze(0).squeeze(0)

        # Transform results back to original image size
        original_width, original_height = image.size
        depth_map_original = affine_transform(
            depth_map_np, 
            transform,
            output_shape=(original_height, original_width),
            mode='constant',
            cval=0.0
        )
        
        # Transform world points back to original size
        # world_points has shape (H, W, 3), need to transform each channel
        world_points_original = np.zeros((original_height, original_width, 3))
        for i in range(3):
            world_points_original[:, :, i] = affine_transform(
                world_points[:, :, i],
                transform,
                output_shape=(original_height, original_width),
                mode='constant',
                cval=0.0
            )
        
        # Transform world points confidence back to original size
        world_points_conf_original = affine_transform(
            world_points_conf,
            transform,
            output_shape=(original_height, original_width),
            mode='constant',
            cval=0.0
        )

        intrinsic_np_original = transform_inv @ intrinsic_np[0]
        intrinsic_np_original[0, 0], intrinsic_np_original[1, 1] = intrinsic_np_original[1, 1], intrinsic_np_original[0, 0]
        intrinsic_np_original[0, 2], intrinsic_np_original[1, 2] = intrinsic_np_original[1, 2], intrinsic_np_original[0, 2]
        
        # Update the variables with transformed results
        depth_map_np = depth_map_original
        world_points = world_points_original
        world_points_conf = world_points_conf_original
        intrinsic_np = intrinsic_np_original
        
        # 准备结果
        results = {
            'depth': depth_map_np,
            'world_points': world_points,
            'world_points_conf': world_points_conf,
            'intrinsic': intrinsic_np,
        }
        
        print("后处理完成")
        return results, inference_time
    
    def process_results_for_response(self, results: Dict[str, Any], original_image: Image.Image, inference_time: float) -> Dict[str, Any]:
        """处理结果以准备HTTP响应"""

        # 编码图像和world_points为Base64
        processed_image_b64 = self.encode_image_to_base64(original_image, "PNG")
        world_points_b64 = self.encode_numpy_to_base64(results['world_points'])
        world_points_conf_b64 = self.encode_numpy_to_base64(results['world_points_conf'])
        
        # 准备响应数据
        response_data = {
            "processed_image": processed_image_b64,
            "world_points": world_points_b64,
            "world_points_conf": world_points_conf_b64,
            "intrinsic": results['intrinsic'].tolist(),
            "metadata": {
                "processing_time": inference_time,
            }
        }
        
        return response_data


# 全局服务器实例
server = None


def create_app(device: str = "cpu"):
    """创建FastAPI应用"""
    from contextlib import asynccontextmanager
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期管理"""
        # 启动时加载模型
        global server
        server = VGGTInferenceServer(device=device)
        server.load_model()
        yield
        # 关闭时清理资源（如果需要）
    
    app = FastAPI(
        title="VGGT推理服务器",
        description="基于VGGT模型的图像推理HTTP服务",
        version="1.0.0",
        lifespan=lifespan
    )
    
    @app.get("/health")
    async def health_check():
        """健康检查端点"""
        return {"status": "healthy", "message": "VGGT推理服务器运行正常"}
    
    @app.post("/inference", response_model=InferenceResponse)
    async def inference_endpoint(request: InferenceRequest):
        """图像推理端点"""
        try:
            # 解码图像
            image = server.decode_base64_image(request.image, request.image_format)
            
            # 运行推理
            results, inference_time = server.run_inference(image)
            
            # 处理结果
            response_data = server.process_results_for_response(
                results, image, inference_time
            )
            
            return InferenceResponse(
                status="success",
                message=f"推理完成，耗时 {inference_time:.2f} 秒",
                data=response_data
            )
            
        except Exception as e:
            return InferenceResponse(
                status="error",
                message=f"推理失败: {str(e)}"
            )
    
    return app


app = create_app()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="VGGT推理服务器")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器主机地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=22334,
        help="服务器端口"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="使用设备 (cpu/cuda)"
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print(f"启动VGGT推理服务器...")
    print(f"主机: {args.host}")
    print(f"端口: {args.port}")
    print(f"设备: {args.device}")
    print(f"API文档: http://{args.host}:{args.port}/docs")
    
    app_instance = create_app(device=args.device)
    
    uvicorn.run(
        app_instance,
        host=args.host,
        port=args.port,
        reload=False
    )


if __name__ == "__main__":
    main()
