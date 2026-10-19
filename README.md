# MOSS-Image DiT 适配器训练器

MOSS-Image 5B 单流 DiT 的适配器训练器，支持 LoRA / LoKr（走 LyCORIS）。TE、VAE、DiT 基座全部冻结，只训练挂在 DiT `_Block` 上的适配器；RTX 5080 16GB 单卡可跑。详细配置清单见 [TUTORIAL.md](TUTORIAL.md)，现成配置见 [configs/](configs/)。

## 环境与安装

Python 3.11；CUDA 12.8+ 的 PyTorch（Blackwell / RTX 5080 不要装 cu121/cu124）。

```powershell
pip install -r requirements.txt
git clone https://github.com/KohakuBlueleaf/LyCORIS
pip install ./LyCORIS
```

Triton 由 `requirements.txt` 按平台安装（Linux `triton`，Windows `triton-windows`，实测 `3.8.0.post28`）。`compile_pointwise` 还要求 PyTorch 暴露 Inductor 的 `emulate_precision_casts`（实测 `2.14.0+cu130`）；缺任一项启动即显式报错，此时把配置里的 `compile_pointwise` 设为 `false` 即可照常跑，cuDNN 收益仍在。

## 模型

官方 BF16 SFT 权重：

- Hugging Face：<https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT>
- ModelScope：<https://modelscope.cn/models/openmoss/MOSS-Image-SFT>

下载后把配置里的 `model_dir` 指向本地目录，例如 `G:/ComfyUI/models/moss_image_sft_bf16`。

## 数据

`dataset.image_dir` 指向图片目录。caption 两种写法：

- **同名 `.txt`**（`caption_format: "plain"`）：图片同目录同 stem 的文本文件；没有该文件时用配置 `dataset.caption` 字段兜底。
- **官方打标 JSON**（`caption_format: "moss_official"`）：图片同 stem 的 `.json`，含 `caption.long/medium/short` 三级文本，训练时按 `caption_weights` 加权抽取一级。

改了图片、caption 文字或分桶设置，要重做 cache。

## 训练流程

子命令按 cache → train 顺序执行（都在 `moss_trainer/` 目录下运行）；`--cache-dir` 与 `--output-dir` 都必须是不存在的目录。

```powershell
# 1. 编码 latent 与 caption embedding
python launch.py cache --config configs/moss_official.json --cache-dir temp/my-cache

# 2. 训练
python launch.py train --config configs/moss_official.json --cache-dir temp/my-cache --output-dir runs/my-run
```

## 独立采样 samples.py

`launch.py sample` 之外的独立入口：一次加载模型，可对任意 prompt（不需要预先写进 cache）和多个 checkpoint 出图。prompt 支持字符串或字符串列表（`sample.prompt` / `--prompts`），训练预览、`launch.py sample` 和本脚本都会逐条出图。`--adapter` 可传多个文件；只采样基座时传 `--multiplier 0`、无需 adapter，非零 multiplier 必须提供 adapter。`--output-dir` 同样必须是新目录。

```powershell
python samples.py --config runs/my-run/config.json `
  --adapter runs/my-run/step000629.safetensors --multiplier 0 1 `
  --prompts "..." --seed 42 --width 832 --height 1216 --steps 20 --cfg 3 `
  --output-dir temp/new-dir
```

每张图的文件名包含 `prompt000` / `prompt001` 和 seed；`parameters.json` 保存完整 prompt、采样参数、来源与耗时。

## WebUI 工作台

WebUI 是独立的本地进程，不会被 `launch.py` 或训练代码导入；启动失败不影响 CLI，关闭浏览器或服务也不会停止已启动的任务。功能覆盖配置读写、cache/train 启动、训练期预览矩阵、独立 XY 评估与指标/图库观察；一次只运行一个 GPU 任务，避免争抢显存。

在 `moss_image_trainer/` 工作根使用项目 venv：

```powershell
.\venv\Scripts\python.exe -m pip install -r .\moss_trainer\requirements-webui.txt
.\venv\Scripts\python.exe -m moss_trainer.webui --host 127.0.0.1 --port 7860
```

浏览器访问 <http://127.0.0.1:7860>。仅本机使用，不提供账号认证，不应发布到公网；任务上的「停止」会终止其进程树，未保存的训练进度不会补存。工作台只管理自己启动的任务，不接管外部 CLI 进程；前端源码在 `webui/frontend/`，仅修改前端时需要 Node.js。

## 配置预设

| 预设 | 用途 |
|---|---|
| `configs/moss_official.json` | 官方 JSON caption |
| `configs/plain_txt.json` | 同名 `.txt` caption |
| `configs/res768.json` | 768 面积档 |
| `configs/lokr_factor16.json` | LoKr，factor=16 |

四个预设都显式写 `attention_backend="cudnn"` + `compile_pointwise=true`，并保持 `batch_size=1` + `gradient_accumulation=8`。

## 性能开关

- **`training.attention_backend`**（默认 `cudnn`）：cuDNN 优先的 attention，12 步 A/B 由 3.761 降到 2.815 s/sample。
- **`training.compile_pointwise`**（默认 `false`，四个预设开 `true`）：叠加点算子编译后，生产口径 2.215 s/sample（原基线 3.761）。
- **`training.compile_blocks`**（默认 `false`）：编译 42 个主 block 原始 forward，仅 LoRA，同口径再 +13.3%。
- **`training.checkpoint_group_size`**（默认 `1`）：设 `2` 把相邻 block 合成一组 checkpoint，省显存。

行为、约束与 caveat 见 [TUTORIAL.md](TUTORIAL.md) 第 5 节 `training`。以上加速都不 bitexact，且只有 12 步 A/B 与一个 epoch 的短程实测，20 epoch 的收敛与出图质量未验证；`compile_blocks` 仅支持 `adapter.algo="lora"`，LoKr 显式拒绝。
