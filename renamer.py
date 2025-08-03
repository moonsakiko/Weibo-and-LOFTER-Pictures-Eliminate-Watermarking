# 文件名: renamer.py

import io
import zipfile
from pathlib import Path

def sanitize_filename(name):
    """移除Windows文件名中的非法字符"""
    return "".join(c for c in name if c not in r'\/:*?"<>|')

def rename_files_in_memory(uploaded_files):
    """在内存中对上传的文件列表进行排序、配对和重命名"""
    report_lines = ["--- 重命名报告 ---"]
    
    # 1. 将上传的文件对象读入内存，并附带元数据
    files_with_meta = []
    for f in uploaded_files:
        files_with_meta.append({
            "name": f.name,
            "data": f.getvalue(),
            "size": f.size,
            "mtime": getattr(f, 'mtime', 0) # 尝试获取mtime，没有则为0
        })

    # 2. 按mtime排序，如果没有mtime则按原始顺序
    files_with_meta.sort(key=lambda x: x['mtime'])
    report_lines.append(f"[*] 发现 {len(files_with_meta)} 个文件，准备处理...")

    if len(files_with_meta) % 2 != 0:
        ignored_file = files_with_meta.pop()
        report_lines.append(f"[!] 警告: 文件总数为奇数。最后一个文件 '{ignored_file['name']}' 已被忽略。")

    # 3. 创建一个内存中的ZIP文件来存放结果
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        for i in range(0, len(files_with_meta), 2):
            file1 = files_with_meta[i]
            file2 = files_with_meta[i+1]

            wm_file, orig_file = (file1, file2) if file1['size'] > file2['size'] else (file2, file1)
            
            base_id = sanitize_filename(Path(wm_file['name']).stem)
            orig_ext = Path(orig_file['name']).suffix if Path(orig_file['name']).suffix else ".jpg"

            new_wm_name = f"{base_id}-wm{orig_ext}"
            new_orig_name = f"{base_id}-orig{orig_ext}"

            report_lines.append(f"  [配对成功] '{wm_file['name']}' + '{orig_file['name']}'")
            report_lines.append(f"    -> 重命名为 '{new_wm_name}' 和 '{new_orig_name}'")
            
            # 将重命名后的文件写入ZIP包
            zip_file.writestr(new_wm_name, wm_file['data'])
            zip_file.writestr(new_orig_name, orig_file['data'])

    report = "\n".join(report_lines)
    zip_buffer.seek(0)
    return zip_buffer, report