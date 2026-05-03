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

    # =====================================================
    # 3DRS Integration: Distillation, Grounding, Query Token
    # =====================================================
    # Grounding head configuration
    ground_head_type: Optional[str] = field(default=None, metadata={"help": "Grounding head type: None, 'infonce', 'mlp', 'score'"})
    ground_head_temperature: float = field(default=0.07, metadata={"help": "Temperature for InfoNCE grounding loss"})

    # Query token configuration  
    query_type: Optional[str] = field(default=None, metadata={"help": "Query token type: None, 'geometry', 'geometry,depth', 'blank', 'blank,reason'"})
    query_size: int = field(default=4, metadata={"help": "Number of query tokens per frame"})
    query_image: bool = field(default=False, metadata={"help": "Whether to also apply distillation on image tokens"})

    # Object feature aggregation
    obj_feature: Optional[str] = field(default=None, metadata={"help": "Object feature aggregation: None, 'sim', 'center_sim', 'filter'"})

    # 3D position encoding (applied to visual tokens using world coordinates)
    world_position_embedding_type: Optional[str] = field(default=None, metadata={"help": "World PE type: None, 'avg-sin3d', 'avg-discrete-sin3d', 'avg-mlp', etc."})
    voxel_size: float = field(default=0.1, metadata={"help": "Voxel size for discretization when using discrete world PE"})

    # Distillation configuration
    distillation_mode: str = field(default="offline", metadata={"help": "Distillation target source: 'offline' (precomputed features) or 'online' (from geometry_encoder)"})
    distillation_feature_dim: int = field(default=2048, metadata={"help": "Dimension of VGGT distillation target features"})
    distillation_depth_feature_dim: int = field(default=1024, metadata={"help": "Dimension of depth distillation target features (e.g. DAv2)"})
    distillation_loss_weight: float = field(default=1.0, metadata={"help": "Weight for 3D feature distillation loss"})
    grounding_loss_weight: float = field(default=1.0, metadata={"help": "Weight for grounding loss"})

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
