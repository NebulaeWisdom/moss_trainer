# MOSS-Image：数据怎么走，配置改哪里

## 1. cache：把图片与caption存成训练用的tensor

```text
launch.py cache -> runtime.main
  -> data.cache_dataset
       扫描图片 -> 读取caption：plain用同名caption文件（缺则用dataset.caption），
                   moss_official用同名.json的long/medium/short三级
       -> 每张图 × 每个面积预算 分配一个宽×高的桶
       -> resize：按桶比例裁剪 -> 缩放到桶尺寸
          no_upscale：仅大图等比缩小 -> 两边向下量化 -> 裁掉余量
       -> 冻结FP32 VAE编码 -> 每条记录保存一个图片tensor（latent）
       -> stdout：`Cache images` 进度条（百分比/速率/已用时间/ETA）
  -> 保存VAE的BN统计 -> 释放VAE
  -> 收集数据集caption去重（plain每张一句；moss_official每张long/medium/short三级）
     + caption_dropout>0时补空字符串
  -> runtime._cache_text
       官方chat template -> 冻结Qwen3.5 9B TE
       -> 每个字符串保存一个文本tensor（embedding）-> 释放TE
       -> stdout：`Encode text` 进度条（百分比/速率/已用时间/ETA）
```

**cache只编码数据集caption（加`caption_dropout`需要的空字符串），不编码`sample.prompt`/`sample.negative_prompt`**：preview与独立sample都不读磁盘文本cache，所以改提示词不需要重做cache，cache里也没有可复用的提示词embedding。

| 阶段 | 代码位置 | 落盘文件，相对`--cache-dir` |
|---|---|---|
|扫图、分桶、裁剪/缩放、VAE编码|`data.cache_dataset`|`NNNNNN.safetensors`：每张处理后图片的latent，形状`[32,H/8,W/8]`；`manifest.json`：图片路径、完整caption、桶宽高、latent路径|
|保留图片tensor的归一化数据|`runtime.main`|`latent_stats.safetensors`：VAE BN的running_mean、running_var、eps|
|不同caption去重，逐句编码|`runtime._cache_text`|`text/<sha256>.safetensors`：每句的embedding，形状`[S,4096]`；`text_manifest.json`：完整字符串→文件路径|

**TE在cache与独立sample时加载；train只在启用preview时于启动阶段加载一次（见第2节），训练循环里始终不加载TE。** cache按`001.txt`的句子原样编码，不拆开、不重排；训练只读取落盘tensor，改了原图或文字，旧cache不会自动更新。独立sample现场编码提示词，既不读也不写这个文本cache（见第3节）。

cache阶段stdout只有两个进度条：扫图/VAE编码用`Cache images`，逐句编码用`Encode text`（都给百分比、已处理/总数、速率、已用时间/ETA）。旧的逐图、逐桶、逐句INFO行已移除；保留的是生命周期行——开始的分辨率/模式、TE加载秒数、结束的记录数/桶数/重复省略数/images/s/records/s与manifest路径。完整caption原文仍在`manifest.json`与`text_manifest.json`里，日志不含它。

## 2. train：读取缓存，梯度只更新DiT上的适配器

```text
launch.py train -> train.main
  -> CachedDataset读取manifest与文本索引（不一次加载全部latent）
  -> preview生效时（sample.preview.enabled且有非0周期）：保护训练RNG -> 冻结TE
       一次性编码全部unique sample.prompt与negative_prompt到CPU内存 -> 释放TE
       未生效则完全不加载TE
  -> load_dit：加载冻结BF16 DiT -> create_adapter：挂LyCORIS
  -> make_optimizers + make_schedulers
  -> BucketBatchSampler：按宽高分组 -> 组内打乱 -> 同尺寸成批 -> 批次打乱
  -> DataLoader：按需读latent + 文本embedding（每worker最多缓存32句）
       caption_dropout：选原句，或选已缓存的空字符串 -> 记录选中level与是否被换成空串
       （moss_official先按caption_weights抽long/medium/short一级）
  -> latent做2×2重排和BN归一化 -> 加噪声 -> DiT前向
  -> rectified-flow MSE -> backward -> 梯度累积/裁剪 -> optimizer.step
  -> scheduler.step -> 写train.jsonl + 刷新stdout进度条 -> 按周期保存 / 可选preview -> 下一批
```

| 阶段 | 代码位置 | 落盘文件，相对train的`--output-dir` |
|---|---|---|
|记录本轮配置、更新进度|`train.main`|`config.json`、`train.jsonl`：每step一条结构化记录（step、epoch、loss、LR、grad_norm、下一批位置、data_wait/step时间、显存）＋固定的`caption_stats`文本统计|
|保存可用于出图的适配器|`train.save_checkpoint`|周期保存`stepNNNNNN.safetensors`；结束保存`adapter.safetensors`|
|保存能继续训练的完整状态|`train.save_checkpoint`，`save.save_state=true`时|周期保存`stepNNNNNN.state.pt`；结束保存`state.pt`：raw参数/buffers、optimizer/scheduler、RNG、下一批位置、配置/缓存路径|
|训练过程中出一张图|`sample.generate_preview`|开启后保存`previewNNNNNN.png`；该事件的元数据只有输出路径、seed、宽高、采样步数、guidance、multiplier、显存/耗时，以及prompt/negative的字符数与token数（全为数值）|

普通训练update**不加载TE或VAE**；TE只在启动阶段（preview生效时）加载一次，训练循环里不再出现；preview也不加载TE、不搬动DiT，它复用启动时编码好的CPU embedding，只临时加载冻结VAE。加噪输入是`(1−σ)×latent+σ×noise`，DiT接收`1000σ`并预测`noise−latent`。训练数据、TE和基座DiT都不被optimizer改写。

**train的stdout与`caption_stats`**：训练不再每step打印一行JSON——stdout是一条进度条（百分比、已处理/总step、step/s、已用时间/ETA）加当前loss、LR、显存和token范围；其余只有start/save/preview/finish/error这类生命周期行。`train.jsonl`仍每step写一条结构化记录，但字段`captions`/`empty_captions`已由固定形状的`caption_stats`取代：`count`是本step样本数，`levels`给出`plain`/`long`/`medium`/`short`各自计数（按dropout前的选中level归属），`dropped`是被`caption_dropout`换成空串的条数，`chars_min`/`chars_max`是最终文本的字符数极值，`tokens_min`/`tokens_max`是最终选中embedding的token数极值。一个step里的多个microbatch汇总时：`count`/`levels`/`dropped`相加，min取最小、max取最大。这些字段只含整数，不含caption原文，所以每step的文本元数据大小有界、不随caption长度增长；这不等于整个`train.jsonl`每行字节数恒定（`data_indices`、`buckets`、`batch_sizes`仍随batch变化），也没有在千万级记录上做过吞吐实测。preview生效时的启动文本编码只在`start`事件里留下数值`preview_text_encoding_seconds`与`preview_text_encoding_count`（未启用或周期全为0时都是0），同样不含提示词原文。

## 3. sample：重载保存的适配器，与基座出图对照

```text
launch.py sample -> sample.main
  -> 加载DiT + 按run配置创建LyCORIS -> 严格加载adapter.safetensors
  -> 冻结TE按`text.gpu_memory_gib`/`text.cpu_memory_gib`预算加载，把全部unique `sample.prompt`与`sample.negative_prompt`现场编码到CPU内存 -> 卸载TE
     （preview用同一组`sample.*`参数，但在train启动阶段编码一次后全程复用）
  -> 加载VAE + 用上一步的CPU embedding（不查磁盘文本cache）
  -> 官方scheduler/sampler：同seed分别运行adapter multiplier=0与1
  -> 反BN + latent解包 -> VAE解码 -> 保存两张图和差异数值
```

| 阶段 | 代码位置 | 落盘文件，相对sample的`--output-dir` |
|---|---|---|
|现场编码提示词|`sample.main`复用`runtime.encode_text`|只在CPU内存，不落盘文本embedding|
|关闭/开启适配器各采样一次|`sample.main`、`sample.sample_latents`|不另存基座权重或中间latent|
|VAE解码、保存图片|`sample.decode_latents`、`sample.main`|prompt为字符串时写`base.png`、`lora.png`；为列表时每个prompt一个`promptNNN/`子目录，内含同名文件。`lora.png`名称固定，实际算法也可为LoKr|
|记录已加载权重和图像差异|`sample.main`|每个prompt的`comparison.json`：权重hash、tensor数、latent/像素差异，以及提示词的`prompt_chars`/`prompt_tokens`/`negative_prompt_chars`/`negative_prompt_tokens`（不含提示词原文）；prompt为列表时`--output-dir`根另写`comparisons.json`|

```text
 temp/<cache>/
   manifest.json | NNNNNN.safetensors | latent_stats.safetensors
   text_manifest.json | text/<sha256>.safetensors
 moss_trainer/runs/<run>/
   config.json | train.jsonl | adapter.safetensors | stepNNNNNN.safetensors
   state.pt | stepNNNNNN.state.pt                 # save_state=true
   previewNNNNNN.png                             # sample.preview.enabled=true，prompt为字符串
   previewNNNNNN_000.png | previewNNNNNN_001.png  # sample.preview.enabled=true，prompt为列表
   comparison/{base.png,lora.png,comparison.json} # 单独执行sample，prompt为字符串
   comparison/prompt000/{base.png,lora.png,comparison.json} # prompt为列表：prompt000/、prompt001/…
   comparison/comparisons.json                   # prompt为列表时根目录汇总
```

`comparison/`不是train自动生成的，它是下面sample命令指定的新目录；prompt为列表时按`prompt000/`、`prompt001/`分目录，根上另有`comparisons.json`。sample与preview的stdout、日志和`comparison*.json`都不含提示词原文：提示词只以`prompt_chars`/`prompt_tokens`/`negative_prompt_chars`/`negative_prompt_tokens`这类计数出现，prompt列表时另有`prompt_index`与`promptNNN/`目录名区分；完整提示词只保留在run的`config.json`，数据集caption原文只保留在cache的`manifest.json`/`text_manifest.json`里（cache不含preview/sample提示词）。

preview与独立sample读同一组`sample.*`参数，区别只在编码时机：preview在train启动时把这组提示词编码一次并在整个run内复用，独立sample每次运行都现场编码一次；两者都不读写磁盘文本cache。

## 4. 命令

在 `moss_trainer/` 目录下运行（与 README 一致）；这里的`new-cache`、`new-run`是要新建的目录名，不能覆盖已有产物。`sample`不接收`--cache-dir`：它按`--config`里的`sample.*`现场编码文本，不读文本cache。

```powershell
python launch.py cache --config configs/plain_txt.json --cache-dir temp/new-cache
python launch.py train --config configs/plain_txt.json --cache-dir temp/new-cache --output-dir runs/new-run
python launch.py sample --config runs/new-run/config.json --adapter runs/new-run/adapter.safetensors --output-dir runs/new-run/comparison
```

**resume**：`train --resume RUN/step000004.state.pt --output-dir NEW_RUN` 读取state中的配置、缓存路径、optimizer/scheduler、RNG和下一批位置，只跑原计划剩余步；不传`--config`，写新目录，恢复完整状态但不承诺CUDA逐bit一致。

## 5. 配置：默认值、具体行为、改动后果

“默认”指本节列出的示例取值（历史基准预设的取值；现预设见第6节）。每组标明它影响前面哪个阶段。需要新cache的改动，不能只编辑JSON然后继续读旧缓存。

### 模型：cache / train / sample

- **`model_dir`** — 默认`G:/ComfyUI/models/moss_image_sft_bf16`；填官方BF16本地目录。cache从其VAE/TE/tokenizer生成tensor，train/sample加载DiT；换模型包需重做cache，不能换成INT8单个文件。

### dataset：cache生成图片/文本文件，train按索引读取

- **`dataset.image_dir`〔cache〕** — 默认仓库`test_image`；填图片目录。程序递归扫描png/jpg/jpeg/webp/bmp，例如`test_image/001.webp`及子目录图片都会加入；换目录或增删图要新cache，不支持把这里填成单张图片路径。

- **`dataset.caption`〔cache〕** — 默认`"deepseek-tan"`。图片没有同名caption文件时，就用这句话；例如`001.webp`没有`001.txt`时用deepseek-tan，有`001.txt`则用文件内容。可填其他完整句子；设`null`就要求每张图都有同名文件，否则cache报错。改了要重做cache。

- **`dataset.caption_extension`〔cache〕** — 图片同名caption文件的**后缀名**，默认`.txt`。`001.webp`配`001.txt`就是`.txt`；文件存在就用它，不存在才用`caption`字段。按UTF-8读取，空文件表示空字符串；换后缀需新cache。图片同名文件这一路径未单独实测。仅`caption_format="plain"`使用它；`moss_official`固定读同名`.json`。

- **`dataset.caption_format`〔cache〕** — 默认`plain`。`plain`每张图一句caption（同名文件或`dataset.caption`）。`moss_official`改读同名`.json`里`caption.long`/`caption.medium`/`caption.short`三级文本，cache把三级**全部**编码，train每步按`caption_weights`抽一级；`caption_dropout`与空字符串处理不变。切换格式需新cache。

- **`dataset.caption_dir`〔cache〕** — 默认`null`，即caption文件与图片同目录。填目录时按图片相对`image_dir`的路径去该目录找同名caption；`moss_official.json`用它把caption集中在`test_image_caption/`。改值需新cache。

- **`dataset.caption_weights`〔train〕** — 默认`{"long":1,"medium":1,"short":1}`，仅`moss_official`使用。每步按权重归一化后抽`long`/`medium`/`short`一级；例如`{"long":1,"medium":0,"short":0}`只用长caption。三级都已进cache，改权重不用新cache；除默认等权外未实测。

- **`dataset.resolutions`〔cache〕** — 默认`[512,768]`，填正整数列表。**它是面积预算列表，不是输出尺寸**：512表示262144像素，768表示589824像素；resize在每个预算下生成一张宽高比桶表，每张图在每一级各选一个桶。加一级就是多一组不同面积的桶、每张图多一条记录；N图×K级最多N×K条，native同尺寸会去重。
  - step=s时，width从`s*ceil(sqrt(R²/max_aspect_ratio)/s)`至`s*floor(sqrt(R²*max_aspect_ratio)/s)`按s枚举，height为`s*round(R²/width/s)`，再按比例过滤。因此两边量化后面积只近似R²。
  - **已实测resize**：`[512,768]`、step64、ratio2，289图→578记录→14桶。512级：**384×704、448×576、512×512、576×448、640×384、704×384**；768级：**576×1024、640×896、704×832、768×768、832×704、896×640、960×640、1024×576**。
  - 384×704=270336、960×640=614400，都不精确等于R²；这组配置**不会生成512×768或768×512**，也不是全部变正方形。改列表要新cache。

- **`dataset.bucket_step`〔cache〕** — 默认`64`，填正的16倍数。resize按这个步长列桶宽高，native把两边向下取倍数；如500×700在native且不需缩小时，step64得到448×640。减小步长通常保留更多像素但桶更多；增大裁剪余量更大，短边不足一步会报错。改值要新cache。
  - **原生粒度是16**：官方推理管线要求宽高为正的16倍数（`vendor/pipeline_moss_image.py:358`、`MOSS-Image/USERGUIDE_ZH.md`），即VAE下采样8 × DiT的2×2 patch。本仓库默认64沿用kohya `bucket_reso_steps=64`（`other_trainer/sd-scripts/library/config_util.py:121`）的SD时代保守值；新版DiT脚本改为按模型算（`other_trainer/sd-scripts/anima_train.py:144` 用 `verify_bucket_reso_steps(16)`，Qwen-Image同为VAE8×patch2）。**MOSS-Image用16更贴合原图，且历史测试已验证可训练**（现预设 `moss_official.json`、`plain_txt.json`、`res768.json`、`lokr_factor16.json` 都用16）。
  - 已实测：`[1024]`+step16→26桶；step64→11桶（`[512,768]`+step64为14桶）。其他step未实测。

- **`dataset.max_aspect_ratio`〔cache〕** — 默认`2.0`，填≥1的数。resize用它确定桶width范围并过滤过宽组合；从2改1会让桶趋于方形，长图裁掉更多边缘。native不使用它；改resize值需新cache，其他值未实测。

- **`dataset.bucket_mode`〔cache〕** — 默认`resize`。`resize`按面积预算建宽高比桶表，选最接近原图比例的桶，先裁到桶比例再缩放；**小图会用BICUBIC放大**，缩小用LANCZOS。`no_upscale`只在W×H>R²时乘`R/sqrt(W*H)`等比缩小，小图保持原尺寸，然后两边向下取step倍数、裁掉余量，**不放大**。每级各处理一次，同图两个native结果尺寸相同时只保存一次。两模式均已实测；切换需新cache。

- **`dataset.random_crop`〔cache〕** — 默认`true`。true在可裁掉的余量中随机选位置，false取中心；如横向需裁100像素，前者随机决定左边裁多少，后者左右各裁50。**cache时只裁一次，之后每epoch读取同一个latent，不重裁**。改值需新cache；false未单独实测。

- **`dataset.repeats`〔train〕** — 默认`1`，填正整数。设2会让每条缓存记录的索引出现两次，再打乱/分批；578条记录变1156个取样位置，不复制文件。按epoch训练时更新数增加，固定max_steps时仍只更新规定次数；不用新cache，大于1未实测。

- **`dataset.caption_dropout`〔train，cache准备空字符串〕** — 默认`0.0`，填0～1。设0.25表示每次读一个样本时，有25%概率把其caption换成`""`，并读取空字符串对应的embedding；剩下75%读取原句。cache在该值>0时会额外编码`""`，所以不是运行时改embedding或清零tensor。已有空字符串缓存时可只改概率；没有就重新cache。0.25已实测。

- **`dataset.seed`〔cache与train取样〕** — 默认`42`，填整数。用它决定cache裁剪/VAE采样，并结合epoch和样本索引决定整句是否被换成`""`、以及`moss_official`时抽到哪一级；改43会换这些随机结果。想换图片crop必须新cache；训练读旧cache不会重新裁图。

### text：影响cache、train启动编码与独立sample的TE编码

- **`text.max_sequence_length`** — 默认`4096`，使用1～4096整数。超过这个token长度的文本会被截断，长度包含官方chat template；如改512，长caption后半段可能不再被编码。cache里的caption embedding要按新长度重做；独立sample与train启动编码的提示词是现场编码，不需要新cache。其他长度未实测。

- **`text.gpu_memory_gib`** — 默认`10`，填GiB预算。cache、train启动编码与独立sample加载TE时都按此预算：TE权重尽量放GPU，剩余放CPU；提高到12可多放GPU但给临时计算留下更少显存。它不是进程总显存上限，也不改变train的batch；其他值未实测。

- **`text.cpu_memory_gib`** — 默认`48`，填GiB预算。cache、train启动编码与独立sample加载TE时限制TE权重可映射到CPU的内存，例如改24不会把模型压缩一半，只会减少可用放置空间；两边预算不足时无法照原方式加载。其他值未实测。

### adapter：train创建并更新，sample按run配置重建

- **`adapter.algo`** — 默认`lora`；已实测`lora`/`lokr`。前者用两个低秩矩阵表示更新，后者用Kronecker因子；默认rank8下LoRA约1858万参数，LoKr自动factor+scalar约108万，不能只凭rank比较大小。`loha`等未实测；换算法新开训练，数据cache可复用。

- **`adapter.rank`** — 默认`8`，填正整数。LoRA中down/up的中间维就是8，改16通常把这两部分参数翻倍；LoKr中它控制需要低秩分解的因子，达到库阈值时可能改存完整因子。其他rank未实测。

- **`adapter.alpha`** — 默认`8`。普通LoRA增量乘alpha/rank，rank8时alpha8倍率1、alpha16倍率2；LoKr部分完整因子分支由库另定尺度，不能把该比值套给所有算法。其他alpha未实测。

- **`adapter.target_modules`** — 默认`["_Block"]`，填DiT模块类名列表。库在这些父模块里找Linear，当前命中42个block的252个线性层；不是层路径正则，也不会访问独立TE。换类名会增加/减少适配层；其他目标未实测。

- **`adapter.dropout`** — 默认`0.0`，取值范围0～1、具体支持依算法。它传给LyCORIS的适配分支dropout，非零时训练随机屏蔽部分激活，eval采样关闭；例如0.1不是把10%的图片caption换空。非零未实测。

- **`adapter.kwargs`** — 默认`{}`。放算法额外参数，例如`{"use_scalar":true,"factor":16}`，原样传给LyCORIS；不要再放已经由专用字段传入的algo/rank/preset。参数是否有效取决于库前向实现；当前bypass路径不能作为有效DoRA训练接口。

- **`adapter.kwargs.use_scalar`** — 默认省略即`false`；填布尔。LoKr设true会增加初始0的可训练scalar，并使用对应因子初始化；它会学习，不是固定的出图强度。它为0维，Muon配置下走辅助AdamW；已实测，state保存独立scalar，出图权重导出时将它折入因子。

- **`adapter.kwargs.factor`** — 默认省略即`-1`，仅LoKr。`-1`对输入/输出维分别找接近平衡的因数对；正数整除D时用`factor × D/factor`并按小/大排序，不整除则搜索受factor上限约束的因数。它决定Kronecker因子shape，**不是rank或倍率，参数量不保证单调**。rank8+scalar实测`-1→16`：q层w1从48×48变16×16，w2_a从64×8变192×8，总参数1,075,452→1,225,980；数据cache不用重做，sample必须使用相同factor。其他factor/decompose_both/unbalanced_factorization未实测。

### optimizer：train中的参数更新

Muon只接收`ndim==2`的参数，其余交给辅助AdamW；没有2D参数时只创建辅助组。默认AdamW则处理全部可训练参数。

- **`optimizer.type`** — 默认`adamw`，可选`adamw`/`muon`。AdamW用一阶/二阶矩调整逐元素更新；Muon对矩阵动量做正交化更新。两者都真实step过，但把adamw改muon不表示同一个1e-4会产生相同更新。

- **`optimizer.learning_rate`** — 默认`0.0001`，填正数。scheduler按此数预热/衰减；其他设置相同时1e-4改2e-4会放大更新。Muon另外乘每个矩阵的shape倍率，日志只显示这里的组LR。

- **`optimizer.weight_decay`** — 默认`0.01`，填非负数。对可训练adapter做decoupled衰减；例如设0就没有这项向零收缩，冻结基座始终不被衰减。主组与辅助组共用；其他值未实测。

- **`optimizer.aux_learning_rate`** — 默认省略时等于主LR，填正数。仅Muon模式使用；主LR=1e-4、此项=5e-5时，scalar等非2D参数的AdamW基准步长为矩阵组的一半。两组分别跟随scheduler；独立覆盖未实测。

- **`optimizer.kwargs`** — 默认`{}`。AdamW的betas/eps控制动量记忆和除法稳定项；Muon可设momentum/nesterov/ns_steps等，例如ns_steps从5增大就多做正交化迭代。只传主组，不传辅助AdamW；不要重复放lr/weight_decay。除下项外的覆盖值未实测。

- **`optimizer.kwargs.adjust_lr_fn`** — 默认省略/`null`等价`original`，仅Muon。矩阵A×B的组LR分别乘：`original`→`sqrt(max(1,A/B))`，`match_rms_adamw`→`0.2*sqrt(max(A,B))`，`spectral_unclamped`→`sqrt(A/B)`；最后一种对宽矩阵可以乘小于1的数。不是scheduler，不保证所有LoRA上下因子同倍率；仅match_rms_adamw已实测。

### scheduler：train每次update后的LR推进

- **`scheduler.type`** — 默认`constant`。constant保持LR；constant_with_warmup升至基准后保持；linear预热后线性降至0；cosine预热后按半周期余弦降至0。例如8步cosine与8步constant的后几步LR不同，不改变update次数。constant/cosine已实测；其他未实测，piecewise_constant缺少step_rules不能配置。

- **`scheduler.warmup_steps`** — 默认`0`，填非负update数。设2时前两步使用0、0.5×base LR，第3步到base LR再继续曲线；constant配正值自动使用constant_with_warmup。cosine+2步已实测。

- **`scheduler.num_training_steps`** — 输入默认没有。trainer计算总update数后写入run配置/state，用来决定衰减终点；例如max_steps8就生成8。新训练会重算，resume读取state中的原值，手改输入这项不能延长训练。

### training：train迭代、前向与反传

- **`training.max_steps`** — 默认`600`，填正整数或`null`。数的是optimizer更新，不是图片数；改8就最多更新8次。与epochs同设时先到者停，resume从已有step继续计到总数，不额外再跑这么多步。

- **`training.epochs`** — 默认`null`，填正整数或`null`。一次epoch读完缓存records×repeats；例如578条、batch1、累积1，epochs2代表1156次更新，但同时max_steps600就先在600停。至少与max_steps之一非null；epoch-only停止未实测。

- **`training.batch_size`** — 默认`1`，填正整数。只把同宽高桶的图片stack进一批；设2会同时前向两张、占更多显存、减少每epoch批次数，桶尾剩1张也保留。大于1未实测。

- **`training.gradient_accumulation`** — 默认`1`，填正整数。设4时先对4批分别backward累积梯度，再clip/step一次；epoch尾只剩2批时除以2并更新，不跨epoch凑4。578批因此是145次更新，最后一次2批；大于1/尾批边界未实测。

- **`training.seed`** — 默认`42`，填整数。控制adapter初始化、batch打乱、噪声/时间步；改43开新训练会换这些随机数，但图片crop仍来自已有cache。resume恢复保存的RNG，不从42重新开始。

- **`training.gradient_checkpointing`** — 默认`true`，填布尔。true不保存全部block中间激活，反向时重算以省显存；false少重算但需要保留更多激活。不是保存`.state.pt`的开关，false未实测。

- **`training.timestep_sampling`** — 默认`uniform`，可选`uniform`/`logit_normal`。前者均匀采0～1的σ，后者正态采样后经sigmoid；σ=0.9的输入主要是噪声，σ=0.1主要保留图片latent。后者可调整两类阶段的出现频率，未实测。

- **`training.logit_mean`** — 默认`0.0`，填实数，只用于logit_normal。设正数会把σ分布往1推、更多高噪声样本；负数往0推，uniform时无效。未实测。

- **`training.logit_std`** — 默认`1.0`，填非负数，只用于logit_normal。增大会让σ更常靠近0/1，设0则每次固定sigmoid(mean)；uniform时无效。未实测。

- **`training.max_grad_norm`** — 默认`1.0`，填正数。update前若全体adapter梯度范数为2，就整体缩到1再更新；不超过1则不缩。日志grad_norm是裁剪前值，0不是关闭开关；其他阈值未实测。

### save：train落盘，resume读取完整state

- **`save.every_steps`** — 默认`200`；正整数表示每N次update保存，0关闭该周期。例如设4会写step000004、step000008的适配器文件；是否另有完整state由下项决定，最终adapter始终保存。

- **`save.every_epochs`** — 默认`0`；设1表示每完成一个epoch保存一次，max_steps在epoch中间停止不算完成epoch。与step周期同时命中只保存一次；epoch周期未实测。

- **`save.save_state`** — 默认`true`。true在周期保存时另写`stepNNNNNN.state.pt`，结束写`state.pt`，包含raw参数/buffers、optimizer/scheduler、RNG及下一批位置；false只留推理用safetensors，**不能精确resume**，即使每一步都保存适配器也不行。

### sample.preview：train启动时编码文本，训练中只复用embedding

- **`sample.preview.enabled`** — 默认`false`；现预设都显式写`true`，并配`every_epochs=1`（每epoch出图）。只有它为`true`且`every_steps`/`every_epochs`至少一个非0时preview才生效；生效时train会在**加载DiT之前**保护训练RNG、加载一次冻结TE，把全部unique `sample.prompt`与`sample.negative_prompt`按`text.gpu_memory_gib`/`text.cpu_memory_gib`预算编码到CPU内存后释放TE，随后照常训练。`false`（或两个周期都是0）时train完全不加载TE。

- **`sample.preview.every_steps`** — 默认`0`；设4就每4次update用当前adapter和`sample.*`参数生成一张PNG。复用启动时编码好的CPU embedding、当前DiT/adapter和临时加载的冻结VAE，**不加载TE、不搬动DiT**，不消费训练RNG；会增加出图时间/显存。现预设`every_steps=0`，即不按update出图。

- **`sample.preview.every_epochs`** — 默认`0`；设1就每完成一个epoch出图，与step触发重合只出一次；文件仍按global step叫`previewNNNNNN.png`，不是按epoch编号。epoch触发未实测。

preview的文本embedding只存在于进程内：不写进`state.pt`/`config.json`，也不跨进程复用；`--resume`是新进程，只要有效preview仍开启就会在启动时重新编码一次。改`sample.prompt`/`sample.negative_prompt`不需要新cache，但编码发生在启动，改完要重启train才生效；`train.jsonl`的`start`事件固定带数值`preview_text_encoding_seconds`/`preview_text_encoding_count`（未启用或两个周期全为0时都是0），preview事件本身不再编码、也不含提示词原文。prompt为列表时每个prompt一张图，文件名`previewNNNNNN_000.png`、`previewNNNNNN_001.png`；为字符串时仍是`previewNNNNNN.png`。独立sample不受`enabled`影响（见下节）。

### loader：train读取CPU文件、组批、传GPU

- **`loader.num_workers`** — 默认`2`，填非负整数。2表示两个CPU进程并行读latent/embedding并组批，0则主进程读取；不让两个worker各加载一套TE/VAE。加worker可能减少等盘时间，也增加启动/RAM副本，其他数量未实测。

- **`loader.pin_memory`** — 默认`true`。把已组好的CPU batch放页锁定内存，以便non_blocking传GPU；不是把全部图片放显存。false少占页锁定内存但可能增加传输等待，未实测。

- **`loader.prefetch_factor`** — 默认`2`，填正整数。每worker提前准备N批，workers2×prefetch2约4批在途；改4可缓冲更长I/O波动但约翻倍预取内存，不加快DiT计算。worker0时不用；其他值未实测。

- **`loader.persistent_workers`** — 默认`true`。到下一epoch继续用原worker和各自最多32句的embedding缓存；false每轮结束回收、下轮重启。worker0时不用；跨epoch边界未实测。

### sample：preview与独立sample共用这组参数

```json
"sample": {
  "preview": {"enabled": false, "every_steps": 0, "every_epochs": 0},
  "prompt": "deepseek-tan",
  "negative_prompt": "",
  "width": 768,
  "height": 512,
  "num_inference_steps": 20,
  "guidance_scale": 3.0,
  "seed": 42
}
```

这组`sample.*`是采样参数的唯一来源：`prompt`/`negative_prompt`给文本，`width`/`height`给尺寸，`num_inference_steps`给采样步数，`guidance_scale`给CFG，`seed`给初始噪声；preview与独立sample都读它，`sample.preview.enabled`只决定train是否在启动时编码并周期出图。

- **`sample.prompt`** — 默认`"deepseek-tan"`，填完整字符串或字符串列表。**独立sample现场编码**：加载冻结TE，把全部unique prompt与negative_prompt编码到CPU内存后卸载TE，不读也不写磁盘文本cache，所以改prompt永远不需要重做cache。**train的preview同样不读cache**：它在train启动时把这组prompt编码到CPU后全程复用，改prompt要重启train（不是重做cache）。例如改成`"A blue-haired girl by a window."`立即生效；填`["deepseek-tan","A blue-haired girl by a window."]`时输出见第3节。

- **`sample.negative_prompt`** — 默认`""`，填完整字符串；**始终是单个字符串，prompt列表里的每一项共用它**，不能按prompt分开。独立sample把它和prompt一起现场编码（空串也真实编码）作为CFG负分支，train的preview则在启动时一并编码它；改成`"blurry"`同样不需要重做cache，preview下需重启train。非空值未实测。

- **`sample.width`** — 默认`768`，填正的16倍数。输出像素宽768对应DiT latent网格宽48；加宽会增加token数/推理开销，不改变训练桶，不需重编码相同prompt。其他宽度未实测。

- **`sample.height`** — 默认`512`，填正的16倍数。输出像素高512对应latent高32；宽高都翻倍时token数约变4倍。其他高度未实测。

- **`sample.num_inference_steps`** — 默认`20`，填正整数。请求官方scheduler做20次积分更新，每步用DiT预测；改40会增加出图时间，不给适配器多训练20步，也不保证一定更好。其他值未实测。

- **`sample.guidance_scale`** — 默认`3.0`，填非负数。预测按`负预测 + scale×(正预测−负预测)`组合；设1直接用正分支、不跑负分支，提高值会放大两者差异。其他值未实测。

- **`sample.seed`** — 默认`42`，填整数。决定采样初始噪声；每个prompt、以及同一轮base/adapter对照，都从配置的这个seed重置，改43会得到另一组图，不改训练RNG、不需新cache。

## 6. 预设文件

- [`plain_txt.json`](configs/plain_txt.json)：纯文本caption基准——`caption_format="plain"`（同名`.txt`，缺失时用`dataset.caption`）、`resolutions=[1024]`、`bucket_step=16`、LoRA rank4/alpha4、Muon lr5e-4、`epochs=20`、`batch_size=1`+`gradient_accumulation=8`、每epoch保存state、preview `enabled=true`+`every_epochs=1`（832×1216）。
- [`moss_official.json`](configs/moss_official.json)：`plain_txt.json`的官方caption版本——`caption_format="moss_official"`、`caption_dir=test_image_caption`、同名`.json`的long/medium/short三级全部进cache、每步等权抽一级；其余与`plain_txt.json`相同。
- [`res768.json`](configs/res768.json)：`moss_official.json`的768档——只把`resolutions`改为`[768]`。
- [`lokr_factor16.json`](configs/lokr_factor16.json)：`moss_official.json`的LoKr版本——`adapter.algo="lokr"`、`adapter.kwargs={"use_scalar":true,"factor":16}`；其余与`moss_official.json`相同。
