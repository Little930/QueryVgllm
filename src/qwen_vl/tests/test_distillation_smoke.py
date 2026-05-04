"""
3DRS Distillation Smoke Test
==============================
验证 query token 注入、蒸馏损失计算、梯度回传是否正常。
无需训练数据，纯合成张量测试。

使用方法：
  cd /path/to/QueryVgllm
  python -m src.qwen_vl.tests.test_distillation_smoke

  # 或直接：
  python src/qwen_vl/tests/test_distillation_smoke.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

import torch
import torch.nn.functional as F


def test_distillation_modules():
    """Test 1: 验证 _init_distillation_modules 正确初始化参数"""
    print("=" * 60)
    print("Test 1: _init_distillation_modules")
    print("=" * 60)

    from transformers import AutoConfig
    from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT

    # 用最小配置创建 config（不需要实际权重）
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-2B-Instruct", trust_remote_code=True)

    # 设置蒸馏配置
    config.use_geometry_encoder = False  # 不需要在线 encoder
    config.use_distillation = True
    config.query_type = "geometry,depth"
    config.query_size = 16
    config.query_image = False
    config.geometry_dim = 2048
    config.depth_dim = 1024
    config.hidden_size = getattr(config, 'hidden_size', 1536)  # 2B model

    # 只初始化蒸馏模块（不加载完整模型权重）
    model = Qwen2_5_VLForConditionalGenerationWithVGGT.__new__(
        Qwen2_5_VLForConditionalGenerationWithVGGT
    )
    model.config = config
    # 手动调用蒸馏初始化
    model._init_distillation_modules(config)

    # 检查参数是否存在
    checks = {
        "geometry_query": (model.geometry_query is not None, f"shape={model.geometry_query.shape if model.geometry_query is not None else 'N/A'}"),
        "depth_query": (model.depth_query is not None, f"shape={model.depth_query.shape if model.depth_query is not None else 'N/A'}"),
        "blank_query": (model.blank_query is None, "should be None (not in query_type)"),
        "proj_geometry": (hasattr(model, 'proj_geometry'), ""),
        "proj_depth": (hasattr(model, 'proj_depth'), ""),
        "proj_3d": (model.proj_3d is None, "should be None (query_image=False, query_type not None)"),
    }
    all_pass = True
    for name, (passed, info) in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {name}: {info}")
        if not passed:
            all_pass = False

    # 检查形状
    H = config.hidden_size
    assert model.geometry_query.shape == (16, H), f"Expected (16, {H}), got {model.geometry_query.shape}"
    assert model.depth_query.shape == (16, H), f"Expected (16, {H}), got {model.depth_query.shape}"
    print(f"\n  ✅ All shapes correct: query=(16, {H})")

    # 检查投影头输出维度
    dummy = torch.randn(1, H)
    geo_out = model.proj_geometry(dummy)
    depth_out = model.proj_depth(dummy)
    assert geo_out.shape == (1, 2048), f"proj_geometry output: {geo_out.shape}"
    assert depth_out.shape == (1, 1024), f"proj_depth output: {depth_out.shape}"
    print(f"  ✅ Projection heads: geometry→2048, depth→1024")

    print(f"\n{'PASSED' if all_pass else 'FAILED'}")
    return all_pass


def test_calculate_distillation_loss():
    """Test 2: 验证损失函数数值正确性"""
    print("\n" + "=" * 60)
    print("Test 2: calculate_distillation_loss")
    print("=" * 60)

    # 创建一个 mock object
    class MockModel:
        pass
    model = MockModel()
    from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT
    model.calculate_distillation_loss = Qwen2_5_VLForConditionalGenerationWithVGGT.calculate_distillation_loss.__get__(model)

    # Case 1: 相同特征 → 损失应为 0
    feat = torch.randn(10, 256)
    loss_same = model.calculate_distillation_loss(feat, feat.clone())
    print(f"  Same features → loss = {loss_same.item():.6f} (expect ~0)")
    assert loss_same.item() < 1e-5, f"Expected ~0, got {loss_same.item()}"
    print(f"  ✅ Zero loss for identical features")

    # Case 2: 随机特征 → 损失应大于 0
    feat_a = torch.randn(10, 256)
    feat_b = torch.randn(10, 256)
    loss_diff = model.calculate_distillation_loss(feat_a, feat_b)
    print(f"  Random features → loss = {loss_diff.item():.6f} (expect >0)")
    assert loss_diff.item() > 0.01, f"Expected >0, got {loss_diff.item()}"
    print(f"  ✅ Positive loss for different features")

    # Case 3: 梯度只流向 student
    teacher = torch.randn(10, 256, requires_grad=True)
    student = torch.randn(10, 256, requires_grad=True)
    loss = model.calculate_distillation_loss(teacher, student)
    loss.backward()
    assert student.grad is not None, "Student should have gradients"
    assert teacher.grad is None or (teacher.grad.abs().sum() == 0), "Teacher should NOT have gradients"
    print(f"  ✅ Gradient flows to student only")

    print(f"\nPASSED")
    return True


def test_inject_query_tokens():
    """Test 3: 验证 _inject_query_tokens 序列扩展正确性"""
    print("\n" + "=" * 60)
    print("Test 3: _inject_query_tokens")
    print("=" * 60)

    import torch.nn as nn
    from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT

    H = 64  # 小维度用于测试
    B, T = 1, 20

    # 创建 mock model
    class MockConfig:
        query_type = "geometry,depth"
        query_size = 4
        hidden_size = H
    
    model = Qwen2_5_VLForConditionalGenerationWithVGGT.__new__(
        Qwen2_5_VLForConditionalGenerationWithVGGT
    )
    model.config = MockConfig()
    model.geometry_query = nn.Parameter(torch.randn(4, H))
    model.depth_query = nn.Parameter(torch.randn(4, H))
    model.blank_query = None

    # 构造输入：token 5-14 是 visual tokens
    inputs_embeds = torch.randn(B, T, H)
    attention_mask = torch.ones(B, T, dtype=torch.long)
    position_ids = torch.arange(T).unsqueeze(0).unsqueeze(0).expand(3, B, T)
    labels = torch.randint(0, 100, (B, T))
    visual_mask = torch.zeros(B, T, dtype=torch.bool)
    visual_mask[0, 5:15] = True  # 10 visual tokens

    print(f"  Input: B={B}, T={T}, visual_tokens=10 (pos 5-14)")
    print(f"  Query: geometry=4, depth=4")

    new_e, new_a, new_p, new_l, gm, dm = model._inject_query_tokens(
        inputs_embeds, attention_mask, position_ids, labels, visual_mask
    )

    expected_T = T + 4 + 4  # +4 geometry +4 depth
    print(f"  Output: T_new={new_e.shape[1]} (expected {expected_T})")

    assert new_e.shape == (B, expected_T, H), f"embeds shape: {new_e.shape}"
    assert new_a.shape == (B, expected_T), f"attn_mask shape: {new_a.shape}"
    assert new_p.shape == (3, B, expected_T), f"position_ids shape: {new_p.shape}"
    assert new_l.shape == (B, expected_T), f"labels shape: {new_l.shape}"
    print(f"  ✅ All shapes correct")

    # 检查 geo_query_mask: 应有恰好 4 个 True
    assert gm.sum().item() == 4, f"geo_query_mask sum: {gm.sum().item()}"
    print(f"  ✅ geo_query_mask: {gm.sum().item()} tokens marked")

    # 检查 depth_query_mask: 应有恰好 4 个 True
    assert dm.sum().item() == 4, f"depth_query_mask sum: {dm.sum().item()}"
    print(f"  ✅ depth_query_mask: {dm.sum().item()} tokens marked")

    # 检查 labels 在 query token 位置是 IGNORE_INDEX
    geo_labels = new_l[gm]
    depth_labels = new_l[dm]
    assert (geo_labels == -100).all(), f"geo query labels should be -100"
    assert (depth_labels == -100).all(), f"depth query labels should be -100"
    print(f"  ✅ Query token labels = IGNORE_INDEX (-100)")

    # 检查 attention_mask 在 query 位置是 1
    assert (new_a[gm] == 1).all(), "geo query attention should be 1"
    assert (new_a[dm] == 1).all(), "depth query attention should be 1"
    print(f"  ✅ Query token attention_mask = 1")

    # 检查 pre-segment text tokens 保持不变
    assert torch.equal(new_e[0, :5], inputs_embeds[0, :5]), "Pre-visual text should be unchanged"
    print(f"  ✅ Pre-visual text tokens preserved")

    print(f"\nPASSED")
    return True


def test_compute_3d_distillation_loss():
    """Test 4: 验证路由器在各种 query_type 下的行为"""
    print("\n" + "=" * 60)
    print("Test 4: compute_3d_distillation_loss (routing)")
    print("=" * 60)

    import torch.nn as nn
    from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT

    H = 64
    B, T = 1, 20

    def make_model(query_type, query_image=False):
        class Cfg:
            hidden_size = H
            geometry_dim = 32
            depth_dim = 16
            geometry_weight = 1.0
            depth_weight = 0.5
        Cfg.query_type = query_type
        Cfg.query_size = 4
        Cfg.query_image = query_image
        
        m = Qwen2_5_VLForConditionalGenerationWithVGGT.__new__(
            Qwen2_5_VLForConditionalGenerationWithVGGT
        )
        m.config = Cfg()
        m._init_distillation_modules(Cfg())
        return m

    hidden = torch.randn(B, T, H)
    img_mask = torch.zeros(B, T, dtype=torch.bool)
    img_mask[0, 5:15] = True
    geo_mask = torch.zeros(B, T, dtype=torch.bool)
    geo_mask[0, 15:19] = True  # 4 geo query tokens
    depth_mask = torch.zeros(B, T, dtype=torch.bool)
    depth_mask[0, 19:23] = True  # would be out of range, but test with T=24
    hidden_ext = torch.randn(B, 24, H)
    geo_mask_ext = torch.zeros(B, 24, dtype=torch.bool)
    geo_mask_ext[0, 15:19] = True
    depth_mask_ext = torch.zeros(B, 24, dtype=torch.bool)
    depth_mask_ext[0, 19:23] = True
    img_mask_ext = torch.zeros(B, 24, dtype=torch.bool)
    img_mask_ext[0, 5:15] = True

    # Case 1: blank → loss = 0
    m = make_model("blank")
    vd = {"feature_3d": torch.randn(B, 8, 196, 32)}
    loss = m.compute_3d_distillation_loss(vd, hidden, img_mask, None, None)
    print(f"  blank → loss={loss.item():.4f} (expect 0)")
    assert loss.item() == 0.0
    print(f"  ✅ blank query returns zero loss")

    # Case 2: geometry → loss > 0
    m = make_model("geometry")
    vd = {"feature_3d": torch.randn(B, 1, 16, 32)}  # 1 image, 4x4=16 patches, dim=32
    loss = m.compute_3d_distillation_loss(vd, hidden_ext, img_mask_ext, geo_mask_ext, None)
    print(f"  geometry → loss={loss.item():.4f} (expect >0)")
    assert loss.item() > 0
    print(f"  ✅ geometry query produces positive loss")

    # Case 3: query_type=None → image-token distillation
    m = make_model(None)
    vd = {"feature_3d": torch.randn(B, 1, 100, 32)}
    loss = m.compute_3d_distillation_loss(vd, hidden, img_mask, None, None)
    print(f"  None (degenerate) → loss={loss.item():.4f} (expect >0)")
    assert loss.item() > 0
    print(f"  ✅ Degenerate image-token distillation works")

    print(f"\nPASSED")
    return True


def test_blank_query_type():
    """Test 5: 验证 blank query 确实只添加容量不产生损失"""
    print("\n" + "=" * 60)
    print("Test 5: blank_query capacity-only ablation")
    print("=" * 60)

    import torch.nn as nn
    from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT

    H = 64

    class Cfg:
        hidden_size = H
        query_type = "blank"
        query_size = 8
        query_image = False
        geometry_dim = 32
        depth_dim = 16
    
    m = Qwen2_5_VLForConditionalGenerationWithVGGT.__new__(
        Qwen2_5_VLForConditionalGenerationWithVGGT
    )
    m.config = Cfg()
    m._init_distillation_modules(Cfg())

    assert m.blank_query is not None, "blank_query should exist"
    assert m.blank_query.shape == (8, H)
    assert m.geometry_query is None, "geometry_query should be None for blank-only"
    assert m.depth_query is None, "depth_query should be None for blank-only"
    print(f"  ✅ blank_query shape=(8, {H}), no other queries")

    # Test injection
    B, T = 1, 15
    inputs = torch.randn(B, T, H)
    attn = torch.ones(B, T, dtype=torch.long)
    pos = torch.arange(T).unsqueeze(0).unsqueeze(0).expand(3, B, T)
    labels = torch.randint(0, 100, (B, T))
    vis_mask = torch.zeros(B, T, dtype=torch.bool)
    vis_mask[0, 3:8] = True

    new_e, new_a, new_p, new_l, gm, dm = m._inject_query_tokens(
        inputs, attn, pos, labels, vis_mask
    )
    expected_T = T + 8  # +8 blank tokens
    assert new_e.shape[1] == expected_T, f"Expected T={expected_T}, got {new_e.shape[1]}"
    assert gm.sum() == 0, "geo mask should be all zeros for blank"
    assert dm.sum() == 0, "depth mask should be all zeros for blank"
    print(f"  ✅ Sequence extended by 8, no geo/depth masks")

    print(f"\nPASSED")
    return True


if __name__ == "__main__":
    print("\n🔬 3DRS Distillation Smoke Test Suite\n")
    
    results = {}
    tests = [
        ("init_distillation_modules", test_distillation_modules),
        ("calculate_distillation_loss", test_calculate_distillation_loss),
        ("inject_query_tokens", test_inject_query_tokens),
        ("compute_3d_distillation_loss", test_compute_3d_distillation_loss),
        ("blank_query_ablation", test_blank_query_type),
    ]

    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"\n  ❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\n  {passed}/{total} tests passed")
    
    if passed < total:
        sys.exit(1)
    print("\n🎉 All tests passed!")
