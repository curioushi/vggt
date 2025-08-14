#!/usr/bin/env python3
"""
VGGT推理服务器客户端示例

用法:
    python client_example.py --server_url http://localhost:22334 --image_path path/to/image.jpg
"""

import argparse
import base64
import json
import requests
from PIL import Image
import io
import numpy as np


def encode_image_to_base64(image_path: str) -> str:
    """将图像文件编码为Base64字符串"""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode('utf-8')


def decode_base64_image(image_b64: str) -> Image.Image:
    """解码Base64图像数据"""
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes))


def decode_base64_numpy(array_b64: str) -> np.ndarray:
    """解码Base64 numpy数组"""
    array_bytes = base64.b64decode(array_b64)
    buffer = io.BytesIO(array_bytes)
    return np.load(buffer)


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


def test_base64_endpoint(server_url: str, image_path: str):
    """测试Base64编码的推理端点"""
    print(f"测试Base64推理端点: {server_url}/inference")
    
    # 编码图像
    image_b64 = encode_image_to_base64(image_path)
    
    # 准备请求数据
    request_data = {
        "image": image_b64,
        "image_format": "jpeg",
        "confidence_threshold": 3.0
    }
    
    # 发送请求
    response = requests.post(f"{server_url}/inference", json=request_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"推理状态: {result['status']}")
        print(f"消息: {result['message']}")
        
        if result['status'] == 'success':
            data = result['data']
            
            # 解码并保存处理后的图像
            processed_image = decode_base64_image(data['processed_image'])
            processed_image.save('output_processed_image.png')
            print("处理后的图像已保存为: output_processed_image.png")
            
            # 解码并保存world_points
            world_points = decode_base64_numpy(data['world_points'])
            np.save('output_world_points.npy', world_points)
            print(f"World points已保存为: output_world_points.npy (形状: {world_points.shape})")

            # 解码并保存world_points_conf
            world_points_conf = decode_base64_numpy(data['world_points_conf'])
            np.save('output_world_points_conf.npy', world_points_conf)
            print(f"World points confidence已保存为: output_world_points_conf.npy (形状: {world_points_conf.shape})")

            # 保存PLY文件
            if 'world_points' in data and 'processed_image' in data and 'world_points_conf' in data:
                # 获取图像尺寸
                H, W = world_points.shape[:2]
                
                # 重塑数据为扁平格式
                points_flat = world_points.reshape(-1, 3)  # (H*W, 3)
                conf_flat = world_points_conf.reshape(-1)  # (H*W,)
                
                # 从处理后的图像中获取颜色
                processed_image = decode_base64_image(data['processed_image'])
                colors_image = np.array(processed_image)  # (H, W, 3)
                
                # 重塑颜色为 (H*W, 3)
                colors_flat = colors_image.reshape(-1, 3)
                
                # 应用置信度过滤
                valid_mask = conf_flat >= 3
                
                # 应用掩码过滤
                filtered_points = points_flat[valid_mask]
                filtered_colors = colors_flat[valid_mask]
                
                print(f"原始点数: {len(points_flat)}, 过滤后点数: {len(filtered_points)}")
                
                # 保存过滤后的PLY文件
                ply_path = 'output_point_cloud.ply'
                save_ply_ascii(filtered_points, filtered_colors, ply_path)
                print(f"点云PLY文件已保存到: {ply_path}")
            
            # 打印元数据
            metadata = data['metadata']
            print(f"处理时间: {metadata['processing_time']:.2f}秒")
            print(f"置信度阈值: {metadata['confidence_threshold']}")
            
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(f"错误信息: {response.text}")





def test_health_endpoint(server_url: str):
    """测试健康检查端点"""
    print(f"测试健康检查端点: {server_url}/health")
    
    response = requests.get(f"{server_url}/health")
    
    if response.status_code == 200:
        result = response.json()
        print(f"服务器状态: {result['status']}")
        print(f"消息: {result['message']}")
    else:
        print(f"健康检查失败，状态码: {response.status_code}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VGGT推理服务器客户端示例")
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://localhost:22334",
        help="服务器URL"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="输入图像文件路径"
    )
    
    args = parser.parse_args()
    
    # 测试健康检查
    test_health_endpoint(args.server_url)
    
    # 测试Base64推理端点
    test_base64_endpoint(args.server_url, args.image_path)
    

    
    print("\n测试完成！")


if __name__ == "__main__":
    main()
