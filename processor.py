# 文件名: processor.py

import io
import zipfile
import re
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import streamlit as st

# --- 缓存模型加载 (升级版) ---
# 现在它可以根据传入的名字，加载并缓存不同的模型
@st.cache_resource
def load_yolo_model(model_name):
    # 根据模型名字，构建不同的文件路径
    # 你需要把 lofter.pt (LOFTER) 和 weibo.pt (微博) 都上传到仓库
    if model_name == "LOFTER":
        model_path = "lofter.pt"
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


# 文件名: processor.py
# ... (顶部的import和load_yolo_model函数保持不变) ...

def repair_image_in_memory(wm_data, orig_data, model, config):
    """在内存中对单对图片进行修复 (最终修正版)"""
    wm_img_np = np.frombuffer(wm_data, np.uint8)
    orig_img_np = np.frombuffer(orig_data, np.uint8)
    high_res_img = cv2.imdecode(wm_img_np, cv2.IMREAD_COLOR)
    low_res_img = cv2.imdecode(orig_img_np, cv2.IMREAD_COLOR)
    
    if high_res_img is None or low_res_img is None: 
        return None, "图片解码失败"

    h_high, w_high, _ = high_res_img.shape
    
    # --- 1. 划定搜索区域 (统一) ---
    search_x_start = int(w_high * config['SEARCH_REGION_RATIOS'][0])
    search_y_start = int(h_high * config['SEARCH_REGION_RATIOS'][1])
    search_x_end = int(w_high * config['SEARCH_REGION_RATIOS'][2])
    search_y_end = int(h_high * config['SEARCH_REGION_RATIOS'][3])
    search_region = high_res_img[search_y_start:search_y_end, search_x_start:search_x_end]
    
    # --- 2. YOLO 预测 ---
    results = model.predict(source=search_region, conf=config['YOLO_CONFIDENCE_THRESHOLD'], verbose=False)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None, "未在指定区域内定位到水印"

    # --- 3. 强制使用最可靠的、经过验证的修复区域计算逻辑 (源自你的本地微博脚本) ---
    all_xyxy = boxes.xyxy.cpu().numpy()
    
    # 获取相对于 "search_region" 的坐标
    x_min_rel = int(np.min(all_xyxy[:, 0]))
    y_min_rel = int(np.min(all_xyxy[:, 1]))
    x_max_rel = int(np.max(all_xyxy[:, 2])) # LOFTER需要x_max
    y_max_rel = int(np.max(all_xyxy[:, 3]))

    # 关键修正：将所有相对坐标，都转换回整张图的绝对坐标
    x_min_abs = x_min_rel + search_x_start
    y_min_abs = y_min_rel + search_y_start
    x_max_abs = x_max_rel + search_x_start
    y_max_abs = y_max_rel + search_y_start
        
    # 现在，我们用这套绝对坐标，来执行和你本地脚本完全一致的扩大逻辑
    original_height = y_max_abs - y_min_abs
    height_margin = int(original_height * config['HEIGHT_EXPANSION_RATIO'])
    base_margin = config.get('BASE_MARGIN', 5)
    
    y_start = max(0, y_min_abs - height_margin - base_margin)
    y_end = min(h_high, y_max_abs + height_margin + base_margin)
    
    model_choice = config['MODEL_CHOICE']
    if model_choice == "微博":
        # 微博模式下，x_start基于识别框左侧，x_end延伸
        x_start = max(0, x_min_abs - base_margin)
        x_end = w_high 
    else: # LOFTER模式，x基于识别框两侧扩大
        original_width = x_max_abs - x_min_abs
        width_margin = int((original_width * config['WIDTH_EXPANSION_RATIO']) / 2)
        x_start = max(0, x_min_abs - width_margin - base_margin)
        x_end = min(w_high, x_max_abs + width_margin + base_margin)
    
    # --- 4. 执行修复 (这部分逻辑早已被验证是正确的) ---
    low_res_resized = cv2.resize(low_res_img, (w_high, h_high), interpolation=cv2.INTER_LANCZOS4)
    clean_patch = low_res_resized[y_start:y_end, x_start:x_end]

    if clean_patch.shape[0] == 0 or clean_patch.shape[1] == 0:
        return None, "修复补丁计算尺寸无效"
    
    high_res_img[y_start:y_end, x_start:x_end] = clean_patch
    
    _, buffer = cv2.imencode('.jpg', high_res_img, [cv2.IMWRITE_JPEG_QUALITY, 98])
    return buffer.tobytes(), "修复成功"


# ... (process_zip_with_selected_model 函数保持不变) ...
# --- 新的、统一的入口函数 ---
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
                output_zip.writestr(Path(wm_path).name, repaired_data)
                report_lines.append(f"  [成功] {Path(wm_path).name} - {message}")
            else:
                report_lines.append(f"  [失败] {Path(wm_path).name} - {message}")

    report = "\n".join(report_lines)
    output_zip_buffer.seek(0)
    return output_zip_buffer, report
