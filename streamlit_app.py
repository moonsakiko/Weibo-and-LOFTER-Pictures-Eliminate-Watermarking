# 文件名: streamlit_app.py

import streamlit as st
import zipfile
import io
from pathlib import Path
from renamer import rename_files_in_memory
# ⬇️⬇️⬇️ 我们将从 processor 导入一个新的、更强大的函数 ⬇️⬇️⬇️
from processor import process_zip_with_selected_model

# --- 页面基础配置 ---
st.set_page_config(
    page_title="AI模型去水印工具",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 AI模型去水印工具")
st.caption("一个用于批量重命名，利用AI模型对LOFTER与微博图片智能去除水印的在线工具")

# --- 使用 Tab 来分隔两大功能 ---
tab1, tab2 = st.tabs(["1️⃣ 批量重命名工具", "2️⃣ 智能去水印工具"])


# ===================================================================
# ---                         功能一：批量重命名 (无需改动)           ---
# ===================================================================
with tab1:
    st.header("批量图片重命名")
    st.info("此工具用于将上传的图片对，按时间顺序两两为一组，将每组大/小文件自动命名为 `[ID]-wm.jpg` 和 `[ID]-orig.jpg` 格式。不能保证时间顺序正确请一次上传两张，一张有水印一张无水印，会自动按大小命名。")

    uploaded_files_rename = st.file_uploader(
        "上传包含图片对的文件夹 (或多选图片文件)",
        accept_multiple_files=True,
        key="renamer_uploader"
    )

    if uploaded_files_rename:
        st.success(f"已成功上传 {len(uploaded_files_rename)} 个文件。")
        if st.button("开始重命名", use_container_width=True):
            with st.spinner("正在分析并重命名图片..."):
                try:
                    renamed_zip_buffer, report = rename_files_in_memory(uploaded_files_rename)
                    
                    st.subheader("重命名报告:")
                    st.text(report)
                    
                    st.download_button(
                        label="📥 下载已重命名的图片 (ZIP包)",
                        data=renamed_zip_buffer,
                        file_name="renamed_images.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"处理时发生错误: {e}")


# ===================================================================
# ---                         功能二：智能去水印 (重大升级)           ---
# ===================================================================
with tab2:
    st.header("智能AI去水印")
    st.info("此工具利用YOLOv8模型，自动识别并修复已按 `[ID]-wm.jpg` 和 `[ID]-orig.jpg` 格式命名的图片对。")
    st.warning("注意：AI模型加载和图片处理需要时间，请耐心等待。")
    
    # --- 侧边栏配置 (重大升级) ---
    with st.sidebar:
        st.header("AI去水印配置")
        
        # ⬇️⬇️⬇️ 新增：模型选择器 ⬇️⬇️⬇️
        model_choice = st.selectbox(
            "选择要去水印的平台模型",
            ("LOFTER", "微博")
        )

        conf_threshold = st.slider("模型自信度门槛", 0.1, 1.0, 0.5, 0.05, key=f"conf_{model_choice}")
        
        # --- 根据模型选择，动态改变默认值 ---
        if model_choice == "微博":
            st.subheader("搜索区域 (微博模式，推荐默认)")
            col1, col2 = st.columns(2)
            # 微博水印固定在下半部分
            search_x_start = col1.slider("左", 0.0, 1.0, 0.0, 0.05, key="wb_sx_s")
            search_x_end = col2.slider("右", 0.0, 1.0, 1.0, 0.05, key="wb_sx_e")
            search_y_start = col1.slider("上", 0.0, 1.0, 0.5, 0.05, key="wb_sy_s")
            search_y_end = col2.slider("下", 0.0, 1.0, 1.0, 0.05, key="wb_sy_e")
            
            st.subheader("修复区域扩大比例")
            # 微博只需要扩大高度
            width_exp = 0.1 # 宽度默认扩大一点
            height_exp = st.slider("高度扩大", 0.0, 1.0, 0.1, 0.05, key="wb_h_exp")

        else: # LOFTER 模式
            st.subheader("搜索区域 (LOFTER模式，推荐默认)")
            col1, col2 = st.columns(2)
            # LOFTER水印位置固定下半，默认下半搜索
            search_x_start = col1.slider("左", 0.0, 1.0, 0.0, 0.05, key="lf_sx_s")
            search_x_end = col2.slider("右", 0.0, 1.0, 1.0, 0.05, key="lf_sx_e")
            search_y_start = col1.slider("上", 0.0, 1.0, 0.5, 0.05, key="lf_sy_s")
            search_y_end = col2.slider("下", 0.0, 1.0, 1.0, 0.05, key="lf_sy_e")
            
            st.subheader("修复区域扩大比例")
            width_exp = st.slider("宽度扩大", 0.0, 1.0, 0.2, 0.05, key="lf_w_exp")
            height_exp = st.slider("高度扩大", 0.0, 1.0, 0.1, 0.05, key="lf_h_exp")
        
        # 将所有配置打包成一个字典
        processor_config = {
            'MODEL_CHOICE': model_choice, # 告诉后端要用哪个模型
            'YOLO_CONFIDENCE_THRESHOLD': conf_threshold,
            'SEARCH_REGION_RATIOS': (search_x_start, search_y_start, search_x_end, search_y_end),
            'WIDTH_EXPANSION_RATIO': width_exp,
            'HEIGHT_EXPANSION_RATIO': height_exp,
            'BASE_MARGIN': 5 # 微博脚本里的一个固定边距参数
        }

    uploaded_zip_process = st.file_uploader(
        "上传包含已重命名图片对的ZIP包",
        type="zip",
        key="processor_uploader"
    )

    if uploaded_zip_process:
        st.success("ZIP文件上传成功！")
        if st.button(f"开始使用【{model_choice}】模型去水印", use_container_width=True):
            status_area = st.empty()
            with st.spinner(f"AI({model_choice}模型)正在全力工作中..."):
                try:
                    # 调用新的、更强大的处理函数
                    repaired_zip_buffer, report = process_zip_with_selected_model(
                        uploaded_zip_process, 
                        processor_config, 
                        status_area
                    )

                    status_area.empty()
                    st.subheader("处理报告:")
                    st.text(report)
                    
                    st.download_button(
                        label=f"📥 下载由【{model_choice}】模型修复的图片 (ZIP包)",
                        data=repaired_zip_buffer,
                        file_name=f"repaired_{model_choice}_images.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                except Exception as e:
                    status_area.empty()
                    st.error(f"处理时发生错误: {e}")
                    st.code(str(e))