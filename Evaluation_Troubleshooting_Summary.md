# VG-LLM 模型验证与排错流程记录

本文档汇总了针对 VG-LLM 模型在评估阶段（1000步 checkpoint）遇到的 `KeyError: 'val'` 错误及解决流程。

## 1. 问题现象
在执行模型评估脚本时，程序抛出以下错误：
```
KeyError: 'val'
...
File "/data/xpxu/VG-LLM/src/lmms_eval/api/task.py", line 1105, in test_docs
    return self.dataset[self.config.test_split]
```
这表明数据集对象无法找到名为 `'val'` 的数据划分 (split)。

## 2. 根本原因分析
1. **数据集加载机制**：原配置 (`scan2cap.yaml` 和 `scanrefer.yaml`) 中指定 `test_split: val`，且未指明数据文件格式。系统默认以 HuggingFace dataset 格式去寻找 `val` split。
2. **实际数据格式**：服务器上的本地数据集（位于 `/home/xpxu/VG-LLM/data/evaluation/`）实际上是原生的 JSON 文件（如 `scan2cap_val_32frames.json`），而非 HuggingFace 的 dataset 目录结构，因此不存在默认的 `val` key。
3. **环境路径差异**：虽然用户在 `/home/xpxu/VG-LLM/` 目录下操作，但程序实际加载并执行的 Python 包路径位于 `/data/xpxu/VG-LLM/src/lmms_eval/`。

## 3. 解决方案
需要修改位于真正执行路径 (`/data/xpxu/VG-LLM/`) 下的任务配置文件 (`yaml`)，改用 `json` 加载器，并硬编码绝对路径指向目标 JSON 文件。

### 修改 `scan2cap.yaml`
修改文件：`/data/xpxu/VG-LLM/src/lmms_eval/tasks/scan2cap/scan2cap.yaml`
主要变更项：
```yaml
dataset_path: json
dataset_kwargs:
  data_files:
    validation: /home/xpxu/VG-LLM/data/evaluation/scan2cap/scan2cap_val_32frames.json
task: "scan2cap"
test_split: validation
```

### 修改 `scanrefer.yaml`
修改文件：`/data/xpxu/VG-LLM/src/lmms_eval/tasks/scanrefer/scanrefer.yaml`
主要变更项：
```yaml
dataset_path: json
dataset_kwargs:
  data_files:
    validation: /home/xpxu/VG-LLM/data/evaluation/scanrefer/scanrefer_val_32frames.json
task: "scanrefer"
test_split: validation
```
*(同时确保 `metadata.media_dir` 指向正确的绝对路径 `/home/xpxu/VG-LLM/data/media`)*

## 4. 补充排错：缺少 preprocessor_config.json

在解决完数据集路径问题后，可能遇到如下报错：
```
OSError: /data/xpxu/VG-LLM/output/checkpoint-1000 does not appear to have a file named preprocessor_config.json.
```

### 根本原因
HuggingFace `Trainer` 默认在 `save_steps` 保存 checkpoint 时，仅保存模型权重（`safetensors`）与 `config.json` 等，并不会自动保存 Tokenizer 与 Processor 相关的配置文件。评估时 `AutoProcessor` 加载该 checkpoint 会因文件缺失而失败。

### 解决方案
需从该模型的基础预训练权重目录中，把 Processor 和 Tokenizer 相关配置文件复制到当前的 Checkpoint 目录内。

执行以下指令进行复制：
```bash
# 1. 设定基础模型目录与 Checkpoint 目录
BASE_MODEL="/home/xpxu/VG-LLM/Qwen2.5-VL-3B-Instruct"
CKPT="/data/xpxu/VG-LLM/output/checkpoint-1000"

# 2. 拷贝必要的配置文件
cp $BASE_MODEL/preprocessor_config.json $CKPT/
cp $BASE_MODEL/tokenizer.json $CKPT/ 2>/dev/null || echo "跳过 tokenizer.json"
# 也可以一并补充 tokenizer_config.json, vocab.json, merges.txt 等

# 3. 验证复制结果
ls $CKPT/ | grep -E "preprocessor|tokenizer"
```

完成上述配置补全后，即可正常运行评估指令。

## 5. 补充排错：缺少 chat_template

在补充完基础配置文件后，如果抛出如下错误：
```
ValueError: No chat template is set for this processor. Please either set the `chat_template` attribute...
```

### 根本原因
由于训练时 `Trainer.save_checkpoint()` 保存的 `tokenizer_config.json` 覆盖了基础模型的配置，但在保存过程中丢失了原本静态的 `chat_template` 属性。同时，新版 `transformers` 要求调用 `processor.apply_chat_template()` 时必须存在该字段。

### 解决方案
直接修改 `lmms_eval` 的模型加载逻辑，给 Tokenizer 硬编码注入 Qwen2.5-VL 的标准 Chat Template 进行兜底。

**需修改的文件**：`/data/xpxu/VG-LLM/src/lmms_eval/models/vgllm.py`

**修改步骤 1：在 `__init__` (约 104 行) 注入 `chat_template` 兜底**
找到 `self._tokenizer = AutoTokenizer.from_pretrained(...)` 后方，插入以下代码：
```python
        # Ensure chat_template is available — HF Trainer checkpoints often strip it
        _QWEN25VL_CHAT_TEMPLATE = (
            "{% set image_count = namespace(value=0) %}"
            "{% set video_count = namespace(value=0) %}"
            "{% for message in messages %}"
            "{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}"
            "<|im_start|>{{ message['role'] }}\n"
            "{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n"
            "{% else %}"
            "{% for content in message['content'] %}"
            "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
            "{% set image_count.value = image_count.value + 1 %}"
            "<|vision_start|><|image_pad|><|vision_end|>"
            "{% elif content['type'] == 'video' or 'video' in content %}"
            "{% set video_count.value = video_count.value + 1 %}"
            "<|vision_start|><|video_pad|><|vision_end|>"
            "{% elif 'text' in content %}{{ content['text'] }}{% endif %}"
            "{% endfor %}<|im_end|>\n{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )
        _tok = self.processor.tokenizer
        if not getattr(_tok, "chat_template", None):
            eval_logger.warning("chat_template missing; injecting default Qwen2.5-VL template.")
            _tok.chat_template = _QWEN25VL_CHAT_TEMPLATE
            self._tokenizer.chat_template = _QWEN25VL_CHAT_TEMPLATE
```

**修改步骤 2：更改 `apply_chat_template` 的调用方**
找到 `generate_until` 方法中约 275 行附近的：
```python
text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```
修改为：
```python
text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```
保存修改并在服务器上生效。

---

## 6. 最终验证指令
在完成上述配置修正，并确认 Checkpoint 实际路径后，可使用以下指令启动验证：

> **注：** 如果遇到 `CUDA out of memory` 报错，说明默认分配的 GPU 正被占用。可以通过 `export CUDA_VISIBLE_DEVICES=6,7` 结合 `--num_processes=2` 来指定空闲的 GPU 运行。

### 4.1 运行 Scan2Cap (描述生成)
```bash
cd /home/xpxu/VG-LLM

export LMMS_EVAL_LAUNCHER="accelerate"
export NCCL_NVLS_ENABLE=0

CKPT="/data/xpxu/VG-LLM/output/checkpoint-1000"
mkdir -p logs/eval_step1000_scan2cap

accelerate launch --num_processes=8 --main_process_port 29501 -m lmms_eval \
    --model vgllm \
    --model_args "pretrained=$CKPT,use_flash_attention_2=true,max_num_frames=32,max_length=12800" \
    --tasks scan2cap \
    --batch_size 1 \
    --log_samples --log_samples_suffix step1000 \
    --output_path logs/eval_step1000_scan2cap
```

### 4.2 运行 ScanRefer (3D 定位)
```bash
cd /home/xpxu/VG-LLM

export LMMS_EVAL_LAUNCHER="accelerate"
export NCCL_NVLS_ENABLE=0

CKPT="/data/xpxu/VG-LLM/output/checkpoint-1000"
mkdir -p logs/eval_step1000_scanrefer

accelerate launch --num_processes=8 --main_process_port 29502 -m lmms_eval \
    --model vgllm \
    --model_args "pretrained=$CKPT,use_flash_attention_2=true,max_num_frames=32,max_length=12800,add_frame_index=true" \
    --tasks scanrefer \
    --batch_size 1 \
    --log_samples --log_samples_suffix step1000 \
    --output_path logs/eval_step1000_scanrefer
```
