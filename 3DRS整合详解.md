# 3DRS 整合详解：改了什么、为什么改、为什么这么改

## 一、背景：两个项目的关系

本项目 QueryVgllm 的核心架构是 **Qwen2.5-VL + VGGT 几何编码器**。它的做法是：用 VGGT 从视频帧中提取 3D 几何特征，再和 Qwen2.5-VL 的 2D 视觉特征做融合（加法/拼接/交叉注意力），最终送入 LLM 做文本生成。

3DRS 项目的核心架构是 **LLaVA-Next-Video + Qwen2-7B**。它的做法不同：它不在输入端融合 3D 特征，而是在 LLM 内部做 **3D 表征蒸馏**——往 LLM 的 token 序列里插入可学习的 query token，然后要求这些 token 的输出去"模仿"预提取的 VGGT 特征。此外它还有一个 **Grounding Head**，能根据文本指令从场景中定位具体物体。

**整合目标**：把 3DRS 的蒸馏 + Grounding + Query Token 能力移植过来，作为 QueryVgllm 的"可选增强模块"。两种 3D 理解路径（输入端融合 vs 内部蒸馏）可以同时存在、独立启用。

---

## 二、整体设计原则

在动手之前，我确立了三条原则：

1. **不破坏现有功能**。所有 3DRS 功能通过配置标志位控制，默认全部关闭。不加任何新参数启动训练时，模型行为和改动前完全一致。

2. **模块化注入**。不重写现有代码，而是在现有类中新增方法和属性。3DRS 的逻辑封装在 `_init_3drs_components()`、`feature_3d_alignment()`、`predict_box()` 等独立方法中，和原有的 `_init_geometry_encoder()`、`_process_geometry_features()` 互不干扰。

3. **保留实验灵活性**。3DRS 论文本身有大量消融实验（不同 query 类型、不同 grounding head、不同物体特征聚合方式），所以每个维度都设计了独立的配置参数，方便后续跑实验。

---

## 三、逐文件详解

### 3.1 新增：`src/qwen_vl/model/position_encoding.py`

**改了什么**：新建文件，包含两个类 `PositionEmbeddingSine3D` 和 `PositionEmbeddingMLP`。

**为什么要有这个文件**：3DRS 的一个核心思想是用世界坐标 (x, y, z) 给视觉 token 加上"空间位置感知"。普通的 ViT 只有 2D 位置编码（知道 patch 在图片的第几行第几列），但不知道它在真实 3D 空间中的位置。有了世界坐标位置编码，模型就能区分"近处的桌子"和"远处的墙壁"。

**为什么这么实现**：

- `PositionEmbeddingSine3D`：把每个坐标轴的值用不同频率的 sin/cos 编码，然后拼接。这和 Transformer 原始论文的位置编码是同一个思路，只是从 1D 推广到了 3D。好处是不需要学习，确定性强。
- `PositionEmbeddingMLP`：用一个两层 MLP 把 (x,y,z) 映射到 hidden_size 维。好处是更灵活，能学习到非线性的空间关系。
- 两种都保留，通过字符串参数选择（`"avg-sin3d"` 或 `"avg-mlp"`），因为 3DRS 论文中两种都测试过，结论不同场景下各有优劣。

---

### 3.2 修改：`src/qwen_vl/train/argument.py`

**改了什么**：在 `ModelArguments` 中新增了 16 个训练参数。

**为什么改这里**：HuggingFace Trainer 的参数系统要求所有可配置项都在 dataclass 中声明。只有在这里声明了，才能在训练脚本的命令行中用 `--ground_head_type infonce` 这样的方式传入。

**参数分组设计的考量**：

| 参数组 | 控制什么 | 为什么需要独立控制 |
|--------|---------|-------------------|
| Grounding Head (`ground_head_type`, `temperature`) | 物体定位头的类型和损失函数 | 3DRS 测试了 3 种头，需要单独切换 |
| Query Token (`query_type`, `query_size`, `query_image`) | 蒸馏载体的类型和数量 | 这是 3DRS 最核心的消融维度 |
| Object Feature (`obj_feature`) | 多视角物体特征的聚合策略 | 影响 grounding 精度，需要消融 |
| World PE (`world_position_embedding_type`, `voxel_size`) | 3D 位置编码方式 | 可选增强，不是必须 |
| Distillation (`distillation_mode`, `*_dim`, `*_weight`) | 蒸馏目标来源和损失权重 | 离线/在线模式、单/双教师切换 |

**所有参数的默认值都设为"不启用"**（None、False、或合理默认值），确保不加参数时系统行为不变。

---

### 3.3 修改：`src/qwen_vl/model/modeling_qwen2_5_vl.py` — 核心

这是改动最大的文件。分三个部分讲。

#### 3.3.1 `__init__` 中的 `_init_3drs_components()`

**改了什么**：在模型构造函数中，在原有的 `_init_geometry_encoder()` 之后、`post_init()` 之前，新增了一个 `_init_3drs_components(config)` 调用。

**为什么放在这个位置**：必须在 `post_init()` 之前，因为 `post_init()` 会初始化权重。如果在它之后才创建新模块，这些模块就不会经过标准的权重初始化流程。

**初始化了什么、为什么这么设计**：

**蒸馏投影头**（`proj_3d` / `proj_geometry` / `proj_depth`）：

```
LLM hidden states (hidden_size=3584) → MLP → 教师特征空间 (2048 或 1024)
```

为什么需要投影头？因为学生（LLM）和教师（VGGT/DAv2）的特征维度不同。投影头负责把学生特征映射到教师空间，然后在那个空间里算距离。这是知识蒸馏的标准做法。

为什么用两层 MLP + GELU？这是 3DRS 原始实现的结构，实验证明比单层线性层效果好，比三层 MLP 性价比更高。

**Query Token**（`geometry_query` / `depth_query` / `blank_query` / `reason_query`）：

这是 3DRS 最关键的创新。核心思想：与其在已有的 image token 上做蒸馏（会干扰它们的原始语义），不如在序列末尾额外插入一些"专用 token"，专门用来承载 3D 信息。

- `geometry_query`：正态初始化 (std=0.02)，经过 LLM 后用 `proj_geometry` 投影，和 VGGT 特征算距离。
- `depth_query`：同理，但目标是 DAv2 深度特征。
- `blank_query`：全零初始化、不施加蒸馏损失。这是消融实验用的：如果 blank_query 也能提升性能，说明提升来自"额外的 token 容量"而非"蒸馏信号"。
- `reason_query`：用于推理辅助的额外 token。

**Grounding Head**（三种）：

Grounding 的任务是：给定一句话（如"红色的椅子"），从场景中所有物体里找到对应的那个。

- `mlp`：最简单。把 ground token 的 hidden state 过一个 MLP，和每个物体的特征做点积打分。
- `infonce`：对比学习风格。物体特征和 query 特征分别投影后做归一化点积，用 InfoNCE 损失训练。额外有一个 `zero_target` 参数处理"场景中没有目标物体"的情况。
- `score`：双投影 + 逐元素乘积 + 打分 MLP。参数最多，表达能力最强。

为什么三种都保留？因为 3DRS 论文的消融实验显示不同场景下最优的 head 不同。保留选择权。

#### 3.3.2 新增的方法

**`process_feature()`**：

这个方法做的事情看起来复杂，其实就一件：把教师特征和学生特征对齐到相同的尺寸，以便逐点算距离。

难点在于：教师（VGGT）输出的空间分辨率是固定的（比如 14×14=196 个 patch），而学生（query token）的数量是可配置的（默认 4 个/帧）。所以需要用 `adaptive_avg_pool2d` 把教师特征从 14×14 池化到 2×2（对应 4 个 query），然后展平成序列。

还有一个适配问题：3DRS 原始代码是给 LLaVA 写的，LLaVA 每行 14 个 patch 后面会跟一个 newline token（所以一帧是 14×15=210 个 token）。Qwen2.5-VL 不是这样——它的分辨率是动态的（`grid_thw`）。所以 `process_feature()` 里做了一个 fallback：如果 token 数不等于 S×210，就直接 flatten 并截断对齐。

**`calculate_feature_loss()`**：

标准的归一化 L2 距离：先把两个向量都 L2 归一化，再算欧几里得距离的平方。教师端 detach，不回传梯度。这比直接用 MSE 好，因为消除了特征尺度的影响。

**`feature_3d_alignment()`**：

蒸馏路由器。根据 `query_type` 分发到不同的蒸馏路径：

- `query_type=None`：退回到 image token 上蒸馏（3DRS 早期版本的做法）
- `query_type="blank"`：返回零损失
- `query_type="geometry"`：在 geometry_query 上蒸馏
- `query_type="geometry,depth"`：geometry + depth 双蒸馏
- `query_image=True`：在 query 和 image token 上同时蒸馏

**`extract_object_feature()`**：

给定场景中每个物体的 3D bounding box (center_xyz + size_xyz)，这个方法要做的是：从 LLM 的 hidden states 中，找到哪些视觉 token 对应这个物体，然后把它们的特征聚合成一个向量。

做法是用世界坐标做空间匹配：对每个 patch，检查它的世界坐标是否落在物体的 bbox 内。落在里面的 patch 的特征取平均，就得到这个物体的特征。

`obj_feature` 参数控制多视角聚合策略：一个物体可能在多帧中出现，每帧提取一个特征向量后，是直接平均（默认），还是用余弦相似度加权（`sim`/`center_sim`），还是过滤掉异常视角（`filter`）。

**`predict_box()`**：

Grounding 分支的完整 forward。流程：

1. 运行 LLM 得到 hidden states
2. 找到 ground token（特殊 token，标记"哪个词在做 grounding"）的位置
3. 调用 `extract_object_feature()` 得到所有候选物体的特征
4. 用 Grounding Head 计算 ground token 和每个物体的匹配分数
5. 和 ground truth 标签算交叉熵或 InfoNCE 损失
6. 再加上蒸馏损失（`feature_3d_alignment`）

#### 3.3.3 `forward()` 方法的修改

**新增 6 个参数**：`video_dict`、`use_object_proposals`、`box_labels`、`img_pos_list`、`img_length_list`、`object_features`。全部默认 None/False。

**新增两段逻辑**（在 `outputs = self.model(...)` 之后）：

1. **Grounding 分支**：如果 `use_object_proposals=True` 且配置了 grounding head，调用 `predict_box()`，返回 grounding loss 替代 CLM loss。
2. **蒸馏增强**：如果不是 grounding 模式，但提供了 `video_dict`，在正常 CLM loss 之外，追加一个蒸馏损失。

**为什么蒸馏损失加在 CLM loss 后面而不是替代它**：因为蒸馏是辅助任务——主任务仍然是文本生成。蒸馏只是"顺便"让模型学到 3D 感知能力，不能牺牲文本生成的质量。所以两个 loss 相加。

---

### 3.4 新增：`src/qwen_vl/data/data_3d.py`

**改了什么**：新建文件，包含 `VideoProcessor3D` 类和 `merge_video_dict()` 函数。

**为什么不直接复用 3DRS 的 video_utils.py**：因为 3DRS 的代码和它自己的 LLaVA 框架深度耦合（比如它依赖 LLaVA 的 image_processor 接口），直接 import 会引入大量不需要的依赖。所以我重写了一份，保持数据处理逻辑一致，但接口适配 QueryVgllm 的数据管线。

**`VideoProcessor3D` 做了什么**：

1. **初始化时**：加载 EmbodiedScan 标注（场景元数据 .pkl）+ 物体 proposals（.json）。这些是 ScanNet 数据集的标准组件。
2. **帧采样**：支持 uniform（均匀采样）和 MC（Monte Carlo，按体素覆盖率贪心采样，优先选能看到更多新区域的帧）。
3. **世界坐标计算**（`calculate_world_coords`）：对每一帧，用深度图 + 相机内参 + 相机位姿（加上 axis-align 矩阵对齐坐标系），调用 `unproject()` 把每个像素反投影到世界坐标。
4. **特征加载**（`get_3d_features` / `get_depth_target`）：从磁盘读取预提取的 .npz 文件，带内存缓存避免重复 IO。
5. **`preprocess()`**：串联以上所有步骤，返回一个 `video_dict`，包含 images、world_coords、objects、feature_3d 等。

**`merge_video_dict()`**：DataLoader 的 collate 辅助。把一个 batch 中多个样本的 video_dict 按 key 合并（tensor 用 `torch.stack`，列表用拼接）。

---

### 3.5 修改：`src/qwen_vl/data/data_qwen.py`

**改了什么**：在 `DataCollatorForSupervisedDataset.__call__()` 末尾新增了两段代码。

**第一段**：检测 batch 中的 instances 是否包含 `video_dict`，如果有，调用 `merge_video_dict()` 合并后放入 batch 字典。

**第二段**：透传 grounding 相关的字段（`box_labels`、`img_pos_list`、`img_length_list` 等），让它们能传到 model.forward()。

**为什么改 collator 而不改 dataset**：因为 `video_dict` 的合并逻辑和 DataLoader 的 batch 拼接有关（不同样本的 tensor 要 stack），这是 collator 的职责。dataset 只负责单样本的预处理。

---

### 3.6 修改：`src/qwen_vl/train/train_qwen.py`

**改了什么**：两处。

**第一处：`set_model()` 函数**

新增一个循环，确保所有 3DRS 新增模块的参数始终 `requires_grad=True`。

为什么需要这个？因为 `set_model()` 在设置冻结策略时，会根据 `tune_mm_llm`、`tune_mm_vision` 等开关批量冻结参数。比如 `tune_mm_llm=True` 会解冻整个 LLM，但如果用户同时设了 `tune_mm_vision=False`，那 visual encoder 会被冻结——而 3DRS 的投影头、query token 等虽然不属于 visual encoder，但如果名字恰好被某个冻结规则匹配到就会出问题。所以最后加一个"白名单"循环，无论前面怎么冻结，3DRS 模块都强制解冻。

**第二处：模型初始化**

在把 `model_args` 的参数写入 `config` 的循环中，新增了 3DRS 的 14 个参数 key。这样这些参数就能从命令行传到 config，再传到 `_init_3drs_components(config)` 里控制模块初始化。

---

### 3.7 新增：`scripts/train_3drs/` 目录

**改了什么**：创建独立的训练脚本目录，包含 4 个脚本 + README。

**为什么不修改原有的 `scripts/train/train_3d.sh`**：用户明确要求"单独放在一个文件夹下，别放原来那个了，容易混"。而且从实验管理角度，3DRS 的实验和原有的 geometry encoder 实验是两套独立的消融矩阵，混在一起容易搞混。

**四个脚本的设计**：

- `train_geometry_query.sh`：主实验。Geometry Query + InfoNCE。这是 3DRS 论文的推荐配置。
- `train_geo_depth.sh`：双教师蒸馏。在 geometry 基础上加 depth 蒸馏。
- `train_blank_query.sh`：消融控制组。插入同样数量的 query token 但不施加蒸馏损失，用来验证性能提升到底来自蒸馏信号还是额外参数容量。
- `train_view_sim.sh`：多视角加权消融。用 `obj_feature="sim"` 测试余弦相似度加权聚合对 grounding 精度的影响。

---

## 四、数据流全景

```
训练脚本 (--query_type "geometry" --ground_head_type "infonce")
    │
    ▼
train_qwen.py
    ├── 解析参数 → config.query_type = "geometry"
    ├── 创建模型 → _init_3drs_components(config)
    │       ├── 创建 geometry_query (128×hidden_size)
    │       ├── 创建 proj_geometry (hidden→2048)
    │       ├── 创建 ground_head_obj + ground_head_query (InfoNCE)
    │       └── 创建 ground_head_zero_target
    ├── set_model() → 强制解冻上述模块
    └── 开始训练
        │
        ▼
    DataLoader → collator 合并 video_dict
        │
        ▼
    model.forward(input_ids, ..., video_dict, use_object_proposals, ...)
        │
        ├── [Grounding 模式] → predict_box()
        │       ├── LLM forward → hidden_states
        │       ├── extract_object_feature() → 物体特征
        │       ├── Grounding Head → 匹配分数
        │       ├── InfoNCE loss
        │       └── + feature_3d_alignment() → 蒸馏 loss
        │
        └── [标准模式] → CLM loss + feature_3d_alignment()
                                        │
                                        ├── process_feature() 对齐尺寸
                                        ├── proj_geometry() 投影
                                        └── calculate_feature_loss() 归一化L2
```

---

## 五、没改的部分（以及为什么不需要改）

| 文件 | 为什么不需要改 |
|------|--------------|
| `modeling_qwen2_5_vl.py` 中的原始 `Qwen2_5_VLForConditionalGeneration` | 那是不带 geometry encoder 的标准模型，3DRS 功能只加在 WithVGGT 子类中 |
| `feature_fusion.py` | 这是输入端融合的模块，和 3DRS 的内部蒸馏是并行的两条路径，互不干扰 |
| `loss.py` | 3DRS 的损失函数直接写在模型方法里（`calculate_feature_loss`、`predict_box`），不需要走外部 loss 管理器 |
| `data/__init__.py` / `data_list.py` | 数据集注册不需要改，scanrefer/scan2cap 等数据集已经注册过了 |
| `processing_qwen2_5_vl.py` | 图像预处理逻辑不需要改，3D 数据的预处理在 `data_3d.py` 中独立处理 |
