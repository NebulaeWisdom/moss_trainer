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

`launch.py sample` 之外的独立入口：一次加载模型，可对任意 prompt（不需要预先写进 cache）和多个 checkpoint 出图。`sample.prompt` 支持字符串或字符串列表；训练预览、`launch.py sample` 和 `samples.py` 都会为列表中的每条 prompt 分别出图。`--adapter` 可传多个文件；只采样基座时传 `--multiplier 0`，无需 adapter。非零 multiplier 必须提供 adapter。`--output-dir` 同样必须是新目录。

```powershell
python samples.py --config runs/my-run/config.json `
  --adapter runs/my-run/step000629.safetensors --multiplier 0 1 `
  --prompts "..." --seed 42 --width 832 --height 1216 --steps 20 --cfg 3 `
  --output-dir temp/new-dir
```

只评估基座、多 prompt：

```powershell
python samples.py --config configs/moss_official.json --multiplier 0 `
  --prompts "an alpine lake at dawn" "a small observatory under the stars" `
  --output-dir temp/base-evaluation
```

每张图的文件名包含 `prompt000` / `prompt001` 和 seed；`parameters.json` 保存完整 prompt、采样参数、来源与耗时。

## WebUI 工作台

WebUI 是独立的本地进程，不会被 `launch.py` 或训练代码导入。训练仍可完全通过原有 CLI 使用；WebUI 启动失败不影响 CLI。关闭浏览器或 WebUI 服务**不会停止已启动的任务**；只有任务上的「停止」按钮会结束它管理的进程树。

在 `moss_image_trainer/` 工作根使用项目 venv：

```powershell
.\venv\Scripts\python.exe -m pip install -r .\moss_trainer\requirements-webui.txt
.\venv\Scripts\python.exe -m moss_trainer.webui --host 127.0.0.1 --port 7860
```

浏览器访问 <http://127.0.0.1:7860>。前端资源随仓库提供，运行时不依赖 CDN，也不需要 Node.js。默认仅供本机使用，不提供账号认证，不应直接发布到公网。

- **配置**：加载预设，通过分组表单编辑。字段只允许其合法类型；例如采样步数是整数，正向提示词允许文本或文本列表。提示词文本转列表会保留为首项；列表转文本显示首项，当前草稿中再次转列表仍保留其余项。保存时新建命名版本，不覆盖原文件；未保存的当前表单也可作为任务快照启动。
- **训练流程**：只有缓存 → 训练两个环节；输出自动进入唯一子目录。训练期采样由配置 `sample.preview` 控制，是训练进程的一部分，不是第三种平级任务。
- **训练预览**：训练任务详情按「固定探针 × 训练 Step」展示横向进化矩阵，并支持同一探针两个 Step 的 A/B 放大对比。探针按 prompt 编号及采样条件区分；不会把不同编号的图混作同一探针。缓存任务只展示日志和路径，不显示空图库。
- **XY 评估**：独立页面调用 `samples.py`，可多选多个 LoRA checkpoint 并勾选 base，一次加载模型运行全部组合。矩阵按 prompt / seed 分行、base / checkpoint 分列。评估使用独立配置草稿，不会改动训练期预览设置；多个 LoRA 必须匹配当前配置的模型、algo / rank / alpha 等结构参数。
- **训练后评估**：已完成训练可从检查点进入 XY 评估。事后补跑的图只进入新的评估输出目录，不冒充训练当时的预览，也不会写回原训练记录。
- **观察**：显示 CPU、RAM、磁盘和 NVIDIA GPU/显存/温度/功耗；不支持的硬件指标显示不可用，不伪造读数。任务日志、训练 loss / 学习率 / 耗时曲线与数值表、进化矩阵和 XY 图均来自真实任务文件。
- **进程与数据**：状态和日志保存在 `moss_trainer/.webui/`；服务重启后可继续观察原任务。一次只运行一个工作台 GPU 任务，避免缓存、训练和采样争抢显存。
- **停止**：当前训练器没有优雅停止协议，停止会终止任务进程树；保留已写入的 checkpoint、日志和图片，未保存的训练进度不会自动补存。
- **采样与恢复**：多 prompt 一项一条；加载 adapter 必须匹配 algo / rank / alpha 等设置。训练恢复只应选择可信的本地 `state.pt`，恢复参数以该状态文件中的配置为准。

WebUI 不会自动修改性能设置。如果当前环境不支持 Triton / Inductor，请在训练表单中关闭 `compile_pointwise`；具体限制见下方「性能开关」。

### 运行范围与前端源码

工作台只管理自己启动的任务，不会接管或停止另一个终端手工启动的 Python 进程。恢复已有 CLI 训练时，状态文件必须位于工作目录内，旁边保留该次训练的 `config.json`。指标展示最近 5,000 个 step（最多读取日志末尾 4 MiB），图库最多展示最近 500 张识别到的采样 PNG；原始文件不被裁剪或删除。

前端源码位于 `webui/frontend/`（Vue 3、Element Plus、ECharts），后端位于 `webui/`（FastAPI、psutil、NVML）。仅修改前端源码时需要 Node.js：

```powershell
# 在 moss_trainer/webui/frontend/ 中执行
npm install --cache ../../../temp/npm-cache
npm run build
```

每次构建生成新的 `webui/static/<build-id>/`，不会清空旧版本。重启 WebUI 后会选择最新完整版本；已运行训练不受影响。任务日志、配置版本、旧构建和验收产物都不会自动清理。

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
