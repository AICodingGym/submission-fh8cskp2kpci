import os
# 在导入任何 huggingface 库之前，设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="microsoft/deberta-v3-base",
    local_dir="models/hf/microsoft_deberta-v3-base",
    resume_download=True,
    # 建议加上，避免使用符号链接，防止后续移动文件出错
    local_dir_use_symlinks=False 
)
