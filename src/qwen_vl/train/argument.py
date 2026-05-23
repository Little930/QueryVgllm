import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

    # Geometry encoder configuration
    use_geometry_encoder: bool = field(default=False)  # Whether to use 3D geometry encoder
    geometry_encoder_type: str = field(default="vggt")  # Type of geometry encoder ("vggt", "pi3")
    geometry_encoder_path: str = field(default="facebook/VGGT-1B/")  # Path to pre-trained geometry encoder model
    reference_frame: str = field(default="first")  # Reference frame for geometry encoding ("first", "last"), only available for vggt
    feature_fusion_method: str = field(default="add")  # Method to fuse geometry and visual features ("add", "concat", "cross_attention", "gate")
    fusion_num_layers: int = field(default=1)  # Number of layers in the cross-attention module when feature_fusion_method is "cross_attention"
    geometry_merger_type: str = field(default="mlp")  # Type of geometry feature merger ("mlp", "avg")
    # 3DRS related hyperparameters (default disabled)
    # ── 标志位 ──────────────────────────────────────────────────────────────────
    # use_geometry_encoder: 在线 VGGT encode → FeatureFusion 注入 image embedding（原有行为）
    # use_distillation    : 离线 feature_3d/feature_dav2 npz → query token 蒸馏损失（新增行为）
    # 两者可独立开关，也可同时启用（并存模式）
    # ─────────────────────────────────────────────────────────────────────────
    use_distillation: bool = field(default=False, metadata={"help": "Enable offline 3D feature distillation via query tokens"})
    ground_head_type: Optional[str] = field(default=None, metadata={"help": "Grounding head type (mlp, infonce, score)"})
    temperature: Optional[float] = field(default=None, metadata={"help": "Temperature for InfoNCE grounding loss"})
    # query_type: None=退化(image token蒸馏), blank=控制组(无蒸馏), geometry=VGGT蒸馏, depth=DAv2蒸馏, 可逗号组合
    query_type: Optional[str] = field(default=None, metadata={"help": "Query token distillation type: None/blank/geometry/depth or comma-separated combinations"})
    # query_size: 每个图像/视频追加的 query token 总数（不依赖帧数，与3DRS的32*query_size不同）
    query_size: int = field(default=16, metadata={"help": "Number of query tokens appended per image/video sequence"})
    query_image: bool = field(default=False, metadata={"help": "Whether to also distill on image tokens (in addition to query tokens)"})
    obj_feature: Optional[str] = field(default=None, metadata={"help": "Object feature aggregation strategy: None(avg)/center_sim/sim/filter"})
    world_position_embedding_type: Optional[str] = field(default=None, metadata={"help": "World position embedding type (sin3d, mlp)"})
    voxel_size: Optional[float] = field(default=None, metadata={"help": "Voxel size for world coordinate discretization"})
    geometry_dim: int = field(default=2048, metadata={"help": "Projection output dim for VGGT geometry features"})
    depth_dim: int = field(default=1024, metadata={"help": "Projection output dim for DAv2 depth features"})
    geometry_weight: float = field(default=1.0, metadata={"help": "Loss weight for geometry distillation"})
    depth_weight: float = field(default=0.5, metadata={"help": "Loss weight for depth distillation (3DRS default=0.5)"})
    ground_loss_weight: Optional[float] = field(default=None, metadata={"help": "Loss weight for grounding loss"})
    use_object_proposals: bool = field(default=False, metadata={"help": "Enable grounding using object proposals"})
@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    max_samples: int = field(default=-1)
    shuffle: bool = field(default=True)
    # ── 3DRS distillation data args ────────────────────────────────────────────
    # feature_dir: root dir containing {scene_id}/vggt.npz (and depth.npz)
    # Loaded by data_3d.py: get_3d_features() reads {feature_3d_path}/{scene_id}/vggt.npz
    feature_dir: Optional[str] = field(default=None, metadata={"help": "Root dir for offline 3D feature npz files ({feature_dir}/{scene_id}/vggt.npz)"})
    use_distillation: bool = field(default=False)
    query_type: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
