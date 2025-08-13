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
            
            # 保存相机参数
            with open('output_camera_params.json', 'w') as f:
                json.dump(data['camera_params'], f, indent=2)
            print("相机参数已保存为: output_camera_params.json")
            
            # 打印元数据
            metadata = data['metadata']
            print(f"原始图像形状: {metadata['original_shape']}")
            print(f"处理后图像形状: {metadata['processed_shape']}")
            print(f"World points形状: {metadata['world_points_shape']}")
            print(f"处理时间: {metadata['processing_time']:.2f}秒")
            
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
