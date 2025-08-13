#!/usr/bin/env python3
"""
极简VGGT推理脚本

用法:
    python inference.py --input_image path/to/image.jpg --output_dir path/to/output

输入单个图像文件进行推理
输出目录将包含:
    - camera_params.json: 内外参矩阵
    - world_points.npz: 世界坐标点
    - point_cloud.ply: 带颜色的3D点云(ASCII格式，已过滤低置信度点)
    - preprocessed_000000.png: 预处理后的图像文件
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

# Add vggt to path
sys.path.append("vggt/")

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="VGGT推理脚本")
    parser.add_argument(
        "--input_image", 
        type=str, 
        required=True, 
        help="输入图像文件路径"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="输出结果目录路径"
    )
    return parser.parse_args()


def load_model():
    """加载VGGT模型"""
    print("正在加载VGGT模型...")
    model = VGGT()
    
    # 下载并加载预训练权重
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL, map_location='cpu'))
    
    model.eval()
    print("模型加载完成")
    return model


def run_inference(model, input_image):
    """运行模型推理"""
    print(f"处理图像文件: {input_image}")
    
    # 检查文件是否存在
    if not os.path.exists(input_image):
        raise ValueError(f"图像文件不存在: {input_image}")
    
    # 检查文件扩展名
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF']
    if not any(input_image.endswith(ext) for ext in valid_extensions):
        raise ValueError(f"不支持的图像格式。支持的格式: {valid_extensions}")
    
    # 加载和预处理单个图像
    print("预处理图像...")
    images = load_and_preprocess_images([input_image])  # 传入列表，包含单个图像路径
    images = images.to('cpu')
    print(f"预处理完成，图像形状: {images.shape}")
    
    # 运行推理 (使用CPU)
    print("开始推理...")
    with torch.no_grad():
        # CPU推理，不使用混合精度
        predictions = model(images)
    
    print("推理完成，正在后处理...")
    
    # 转换pose编码为内外参矩阵
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], 
        images.shape[-2:]
    )
    
    # 生成世界坐标点
    depth_map = predictions["depth"]  # (S, H, W, 1)
    
    # 转换tensor为numpy（因为geometry函数期望numpy输入）
    depth_map_np = depth_map.cpu().numpy()
    extrinsic_np = extrinsic.cpu().numpy()
    intrinsic_np = intrinsic.cpu().numpy()
    
    # 确保depth_map具有正确的形状 (S, H, W, 1)
    # unproject_depth_map_to_point_map 函数期望最后一维为1，可以被压缩
    # 原始形状可能是 (1, 1, 518, 518, 1)，需要调整为 (1, 518, 518, 1)
    if len(depth_map_np.shape) == 5:
        # 移除第二维：(1, 1, H, W, 1) -> (1, H, W, 1)
        depth_map_np = depth_map_np.squeeze(1)  
    elif len(depth_map_np.shape) == 4 and depth_map_np.shape[1] == 1:
        # 如果是 (B, 1, H, W)，需要添加最后一维
        depth_map_np = depth_map_np.squeeze(1)[..., None]
    elif len(depth_map_np.shape) == 3:
        # 如果是 (S, H, W)，添加最后一维
        depth_map_np = depth_map_np[..., None]
    
    # 同样处理extrinsic和intrinsic的形状
    if len(extrinsic_np.shape) == 4:  # (1, 1, 3, 4) -> (1, 3, 4)
        extrinsic_np = extrinsic_np.squeeze(1)
    if len(intrinsic_np.shape) == 4:  # (1, 1, 3, 3) -> (1, 3, 3)
        intrinsic_np = intrinsic_np.squeeze(1)
    
    world_points = unproject_depth_map_to_point_map(
        depth_map_np, 
        extrinsic_np, 
        intrinsic_np
    )
    
    # 转换为numpy数组
    results = {
        'extrinsic': extrinsic_np.squeeze(0),
        'intrinsic': intrinsic_np.squeeze(0),
        'world_points': world_points.squeeze(0),
        'depth': depth_map_np.squeeze(0).squeeze(-1),
        'world_points_conf': predictions.get("world_points_conf", torch.zeros_like(depth_map)).cpu().numpy().squeeze(0).squeeze(0),
        'preprocessed_images': images.cpu().numpy()
    }
    
    print("后处理完成")
    return results


def save_ply_ascii(world_points, colors, output_path):
    """
    手动保存PLY文件 (ASCII格式)
    
    Args:
        world_points (np.ndarray): 3D点坐标 (H*W, 3)
        colors (np.ndarray): RGB颜色 (H*W, 3), 值范围0-255
        output_path (str): 输出文件路径
    """
    num_points = world_points.shape[0]
    
    with open(output_path, 'w') as f:
        # 写入PLY头部
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # 写入顶点数据
        for i in range(num_points):
            x, y, z = world_points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def save_results(results, output_dir):
    """保存推理结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存相机参数到JSON
    camera_params = {
        'extrinsic': results['extrinsic'].tolist(),
        'intrinsic': results['intrinsic'].tolist()
    }
    
    json_path = os.path.join(output_dir, 'camera_params.json')
    with open(json_path, 'w') as f:
        json.dump(camera_params, f, indent=2)
    print(f"相机参数已保存到: {json_path}")
    
    # 保存世界坐标点到NPZ
    npz_path = os.path.join(output_dir, 'world_points.npz')
    np.savez(
        npz_path,
        world_points=results['world_points'],
        world_points_conf=results['world_points_conf'],
        depth=results['depth']
    )
    print(f"世界坐标点已保存到: {npz_path}")
    
    # 保存点云到PLY文件
    if 'world_points' in results and 'preprocessed_images' in results and 'world_points_conf' in results:
        world_points = results['world_points']  # (H, W, 3)
        world_points_conf = results['world_points_conf']  # (H, W)
        preprocessed_images = results['preprocessed_images']  # (1, 3, H, W)
        
        # 获取图像尺寸
        H, W = world_points.shape[:2]
        
        # 重塑数据为扁平格式
        points_flat = world_points.reshape(-1, 3)  # (H*W, 3)
        conf_flat = world_points_conf.reshape(-1)  # (H*W,)
        
        # 从预处理图像中获取颜色 (1, 3, H, W) -> (H, W, 3)
        if len(preprocessed_images.shape) == 4:
            colors_image = preprocessed_images[0].transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
        else:
            colors_image = preprocessed_images.transpose(1, 2, 0)
        
        # 转换颜色值到0-255范围并重塑为 (H*W, 3)
        colors_flat = (colors_image * 255).astype(np.uint8).reshape(-1, 3)
        
        valid_mask = conf_flat >= 3
        
        # 应用掩码过滤
        filtered_points = points_flat[valid_mask]
        filtered_colors = colors_flat[valid_mask]
        
        print(f"原始点数: {len(points_flat)}, 过滤后点数: {len(filtered_points)}")
        
        # 保存过滤后的PLY文件
        ply_path = os.path.join(output_dir, 'point_cloud.ply')
        save_ply_ascii(filtered_points, filtered_colors, ply_path)
        print(f"点云PLY文件已保存到: {ply_path}")
    
    # 保存预处理后的图像
    if 'preprocessed_images' in results:
        preprocessed_images = results['preprocessed_images']
        to_pil = transforms.ToPILImage()
        
        # 保存每张预处理后的图像
        for i in range(preprocessed_images.shape[0]):
            # 获取单张图像 (C, H, W)
            img_tensor = torch.from_numpy(preprocessed_images[i])
            
            # 转换为PIL图像
            pil_image = to_pil(img_tensor)
            
            # 保存图像到输出目录 (与JSON、NPZ同目录)
            save_path = os.path.join(output_dir, f"preprocessed_{i:06d}.png")
            pil_image.save(save_path)
        
        print(f"预处理图像已保存到: {output_dir} (共{preprocessed_images.shape[0]}张)")


def main():
    """主函数"""
    args = parse_args()
    
    # 检查输入图像文件是否存在
    if not os.path.exists(args.input_image):
        raise ValueError(f"输入图像文件不存在: {args.input_image}")
    
    # 强制使用CPU
    device = "cpu"
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_model()
    
    # 运行推理
    results = run_inference(model, args.input_image)
    
    # 保存结果
    save_results(results, args.output_dir)
    
    print("推理完成！")
        


if __name__ == "__main__":
    main()
