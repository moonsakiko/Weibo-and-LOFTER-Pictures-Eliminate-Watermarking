# 文件名: processor.py

import io
import zipfile
import re
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import streamlit as st
from PIL import Image # 引入我们新的“高质量编码”大师

# --- 缓存模型加载 (保持最终成功版) ---
@st.cache_resource
def load_yolo_model(model_name):
    # 根据我们最终的调试结果，这里可能需要绝对路径
    # 为了通用性，我们先尝试相对路径，如果部署失败再换成绝对路径
    if model_name == "LOFTER":
        model_path = "best.pt"
    elif model_name == "微博":
        model_path = "weibo.pt"
    else:
        raise ValueError("未知的模型选项！")

    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        # 兼容 Hugging Face 的绝对路径
        try:
            abs_model_path = f"/app/src/{model_path}"
            model = YOLO(abs_model_path)
            return model
        except Exception as e2:
             raise FileNotFoundError(f"在 {model_path} 和 {abs_model_path} 都无法加载YOLO模型！请确保模型文件已上传。错误: {e2}")

def repair_image_in_memory(wm_data, orig_data, model, config):
    """在内存中对单对图片进行修复 (最终保真版)"""
    
    # --- 1. 格式检测与初始解码 ---
    # 我们先用Pillow来嗅探一下原始格式
    original_format = Image.open(io.BytesIO(wm_data)).format

    wm_img_np = np.frombuffer(wm_data, np.uint8)
    orig_img_np = np.frombuffer(orig_data, np.uint8)
    # 使用 cv2.IMREAD_UNCHANGED 来保留PNG的透明通道(如果存在)
    high_res_img = cv2.imdecode(wm_img_np, cv2.IMREAD_UNCHANGED)
    low_res_img = cv2.imdecode(orig_img_np, cv2.IMREAD_UNCHANGED)
    
    if high_res_img is None or low_res_img is None: 
        return None, "图片解码失败"

    # --- 2. 所有修复逻辑 (使用我们最终调试好的、最健壮的版本) ---
    # 这部分代码我们已经验证过是正确的，完全保留
    h_high, w_high, *_ = high_res_img.shape
    search_region = high_res_img[h_high // 2:, :]
    results = model.predict(source=search_region, conf=config['YOLO_CONFIDENCE_THRESHOLD'], verbose=False)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None, "未在指定区域内定位到水印"

    all_xyxy = boxes.xyxy.cpu().numpy()
    x_min_rel = int(np.min(all_xyxy[:, 0]))
    y_min_rel = int(np.min(all_xyxy[:, 1]))
    x_max_rel = int(np.max(all_xyxy[:, 2]))
    y_max_rel = int(np.max(all_xyxy[:, 3]))

    y_min_abs = y_min_rel + h_high // 2
    y_max_abs = y_max_rel + h_high // 2

    original_height = y_max_abs - y_min_abs
    height_margin = int(original_height * config['HEIGHT_EXPANSION_RATIO'])
    base_margin = config.get('BASE_MARGIN', 5)

    y_start = max(0, y_min_abs - height_margin - base_margin)
    y_end = min(h_high, y_max_abs + height_margin + base_margin)
    
    model_choice = config['MODEL_CHOICE']
    if model_choice == "微博":
        x_start = max(0, x_min_rel - base_margin)
        x_end = w_high
    else: # LOFTER
        x_max_rel = int(np.max(all_xyxy[:, 2]))
        x_start = max(0, x_min_rel - base_margin)
        x_end = min(w_high, x_max_rel + base_margin)
        
    low_res_resized = cv2.resize(low_res_img, (w_high, h_high), interpolation=cv2.INTER_LANCZOS4)
    clean_patch = low_res_resized[y_start:y_end, x_start:x_end]

    if clean_patch.shape[0] == 0 or clean_patch.shape[1] == 0:
        return None, "修复补丁计算尺寸无效"
    
    high_res_img[y_start:y_end, x_start:x_end] = clean_patch
    
    # --- ⬇️⬇️⬇️ 3. 最终的“心脏移植”：根据原始格式，选择不同的编码器 ⬇️⬇️⬇️ ---
    
    if original_format == 'PNG':
        # 如果原始文件是PNG，我们用Pillow来保存，以求最大保真
        # a. 转换色彩通道：OpenCV是BGR(A)，Pillow是RGB(A)
        if high_res_img.shape[2] == 4:
            color_converted_img = cv2.cvtColor(high_res_img, cv2.COLOR_BGRA2RGBA)
        else:
            color_converted_img = cv2.cvtColor(high_res_img, cv2.COLOR_BGR2RGB)
        
        # b. 从NumPy数组创建Pillow图像
        pil_image = Image.fromarray(color_converted_img)
        
        # c. 保存到内存缓冲区
        buffer = io.BytesIO()
        # Pillow的PNG编码器参数：compress_level=1 (最少压缩，最大保真)
        pil_image.save(buffer, format='PNG', compress_level=1)
        buffer.seek(0)
        return buffer.getvalue(), f"修复成功 (PNG保真模式)"

    else: # 对于JPG和其他格式，我们依然信任高效的OpenCV
        _, buffer = cv2.imencode('.jpg', high_res_img, [cv2.IMWRITE_JPEG_QUALITY, 98])
        return buffer.tobytes(), f"修复成功 (JPG高效模式)"


def process_zip_with_selected_model(zip_file_obj, config, status_area):
    """根据前端选择的模型，在内存中处理ZIP包"""
    model = load_yolo_model(config['MODEL_CHOICE'])
    report_lines = [f"--- AI去水印处理报告 ({config['MODEL_CHOICE']}模型) ---"]
    
    input_zip = zipfile.ZipFile(zip_file_obj, 'r')
    files_map = {Path(f).stem: f for f in input_zip.namelist()}
    wm_pattern = re.compile(r"(.+)-wm$")
    tasks = []
    
    for base_name_stem, full_path in files_map.items():
        m = wm_pattern.match(base_name_stem)
        if m:
            base_id = m.group(1)
            orig_full_path = files_map.get(f"{base_id}-orig")
            if orig_full_path:
                tasks.append((full_path, orig_full_path))
    
    if not tasks:
        raise ValueError("ZIP包中未找到任何有效的图片对 (如 'id-wm.jpg' 和 'id-orig.jpg')")

    output_zip_buffer = io.BytesIO()
    with zipfile.ZipFile(output_zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as output_zip:
        for i, (wm_path, orig_path) in enumerate(tasks):
            status_area.text(f"正在处理第 {i+1}/{len(tasks)} 对图片: {Path(wm_path).name}")
            
            wm_data = input_zip.read(wm_path)
            orig_data = input_zip.read(orig_path)
            
            repaired_data, message = repair_image_in_memory(wm_data, orig_data, model, config)
            
            if repaired_data:
                # 保持原始文件名和格式
                output_zip.writestr(Path(wm_path).name, repaired_data)
                report_lines.append(f"  [成功] {Path(wm_path).name} - {message}")
            else:
                report_lines.append(f"  [失败] {Path(wm_path).name} - {message}")

    report = "\n".join(report_lines)
    output_zip_buffer.seek(0)
    return output_zip_buffer, report
