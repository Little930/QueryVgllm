# 3DRS 训练脚本说明
# ==================
#
# 目录结构：
# scripts/train_3drs/
# ├── README.md                   ← 本文件
# ├── train_geometry_query.sh     ← 🔴 主实验：Geometry Query 蒸馏 + InfoNCE
# ├── train_geo_depth.sh          ← 双教师蒸馏 (VGGT + DAv2)
# ├── train_blank_query.sh        ← 消融：Blank Query 控制组
# └── train_view_sim.sh           ← 消融：多视角加权物体特征
#
# 使用方法：
#   cd /path/to/QueryVgllm
#   bash scripts/train_3drs/train_geometry_query.sh
#
# 前提条件：
#   1. 服务器上有 ScanNet 数据 (posed images + depth + poses)
#   2. 已运行 VGGT 特征提取 (生成 .npz 文件)
#   3. 有 EmbodiedScan 标注 (.pkl) 和物体 proposals (.json)
#
# 数据目录结构 (相对于项目根目录)：
#   data/
#   ├── scannet/
#   │   ├── posed_images/                        # RGB + depth + pose
#   │   ├── posed_images_3d_feature_vggt/        # VGGT 离线特征
#   │   └── depth_features_v2/                   # DAv2 特征 (仅双教师需要)
#   ├── embodiedscan/
#   │   ├── embodiedscan_infos_train.pkl
#   │   ├── embodiedscan_infos_val.pkl
#   │   └── embodiedscan_infos_test.pkl
#   └── metadata/
#       ├── scannet_train_gt_box.json
#       └── scannet_val_pred_box.json
#
# 实验矩阵：
#   ┌──────────────────────┬───────────┬──────────┬──────────────┐
#   │ 脚本                 │ query_type│ 蒸馏教师 │ 目的         │
#   ├──────────────────────┼───────────┼──────────┼──────────────┤
#   │ train_geometry_query │ geometry  │ VGGT     │ 主实验       │
#   │ train_geo_depth      │ geo,depth │ VGGT+DAv2│ 双教师       │
#   │ train_blank_query    │ blank     │ 无       │ 容量消融     │
#   │ train_view_sim       │ geometry  │ VGGT     │ 聚合消融     │
#   └──────────────────────┴───────────┴──────────┴──────────────┘
