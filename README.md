# MOSS-Image DiT 适配器训练器

MOSS-Image 5B 单流 DiT 的适配器训练器，支持 LoRA / LoKr（走 LyCORIS）。TE、VAE、DiT 基座全部冻结，只训练挂在 DiT `_Block` 上的适配器；RTX 5080 16GB 单卡可跑。

详细配置清单见 [TUTORIAL.md](TUTORIAL.md)，现成配置见 [configs/](configs/)。

## 环境

- Python 3.11；CUDA 12.8+ 的 PyTorch。Blackwell / RTX 5080 不要装 cu121/cu124。

## 安装

```powershell
pip install -r requirements.txt
git clone https://github.com/KohakuBlueleaf/LyCORIS
pip install ./LyCORIS
```

## 模型

官方 BF16 SFT 权重：

- Hugging Face：<https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT>
- ModelScope：<https://modelscope.cn/models/openmoss/MOSS-Image-SFT>

下载后把配置里的 `model_dir` 指向本地目录，例如 `G:/ComfyUI/models/moss_image_sft_bf16`。

## 数据

`dataset.image_dir` 指向图片目录。caption 两种写法：

- **同名 `.txt`**（`caption_format: "plain"`）：图片同目录同 stem 的文本文件；没有该文件时用配置 `dataset.caption` 字段兜底。
- **官方打标 JSON**（`caption_format: "moss_official"`）：图片同 stem 的 `.json`，含 `caption.long/medium/short` 三级文本，训练时按 `caption_weights` 加权抽取一级。

## 训练流程

子命令按 cache → train 顺序执行：

```powershell
# 1. 编码 latent 与 caption embedding
python launch.py cache --config configs/moss_official.json --cache-dir temp/my-cache

# 2. 训练
python launch.py train --config configs/moss_official.json --cache-dir temp/my-cache --output-dir runs/my-run
```

`--output-dir` 必须是不存在的目录。换了图片、caption 文字或分桶设置，要换新的 `--cache-dir`。

## 独立采样 `samples.py`

`launch.py sample` 之外的独立入口：一次加载模型，可对**任意 prompt**（不需要预先写进 cache）和**多个 checkpoint** 出图：

```powershell
python samples.py --config runs/my-run/config.json `
  --adapter runs/my-run/step000629.safetensors --multiplier 0 1 `
  --prompts "..." --seed 42 --width 832 --height 1216 --steps 20 --cfg 3 `
  --output-dir temp/new-dir
```

`--adapter` 可传多个文件，`--multiplier 0` 只出基座图。`--output-dir` 同样必须是新目录。

## 配置预设

| 预设 | 用途 |
|---|---|
| `configs/moss_official.json` | 官方 JSON caption |
| `configs/plain_txt.json` | 同名 `.txt` caption |
| `configs/res768.json` | 768 面积档 |
| `configs/lokr_factor16.json` | LoKr，factor=16 |
