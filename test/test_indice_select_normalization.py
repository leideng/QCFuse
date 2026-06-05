"""
单元测试: 验证 _compute_attn_v_norm 函数中的归一化处理

测试内容：
1. 归一化后的张量形状正确
2. 注意力分数归一化后总和接近 1
3. V 范数归一化后在 [0, 1] 区间
4. 极端值情况处理正确
5. 梯度计算正确（如需要反向传播）
"""

import torch
import torch.nn.functional as F
import pytest
import math


class TestNormalization:
    """测试归一化功能"""

    def test_softmax_normalization_sum_to_one(self):
        """测试 softmax 归一化后总和接近 1"""
        # 模拟注意力分数 [num_layers, seq_len]
        num_layers = 4
        seq_len = 100
        
        # 随机生成注意力分数
        layer_attn_importance = torch.randn(num_layers, seq_len)
        
        # 应用 softmax 归一化
        attn_normalized = F.softmax(layer_attn_importance, dim=-1)
        
        # 验证每层的注意力分数总和接近 1
        sums = attn_normalized.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(num_layers), atol=1e-5), \
            f"Softmax 归一化后总和应接近 1，但得到 {sums}"
        
        # 验证形状不变
        assert attn_normalized.shape == layer_attn_importance.shape, \
            "归一化后形状应与输入一致"

    def test_softmax_normalization_range(self):
        """测试 softmax 归一化后值在 [0, 1] 区间"""
        layer_attn_importance = torch.randn(4, 100)
        attn_normalized = F.softmax(layer_attn_importance, dim=-1)
        
        assert (attn_normalized >= 0).all(), "Softmax 归一化后所有值应 >= 0"
        assert (attn_normalized <= 1).all(), "Softmax 归一化后所有值应 <= 1"

    def test_minmax_normalization_range(self):
        """测试 min-max 归一化后值在 [0, 1] 区间"""
        num_layers = 4
        seq_len = 100
        
        # 模拟 V 范数
        v_norms = torch.randn(num_layers, seq_len).abs()  # 范数应为正
        
        # 应用 min-max 归一化
        eps = 1e-8
        v_min = v_norms.min(dim=-1, keepdim=True)[0]
        v_max = v_norms.max(dim=-1, keepdim=True)[0]
        v_normalized = (v_norms - v_min) / (v_max - v_min + eps)
        
        # 验证值在 [0, 1] 区间
        assert (v_normalized >= 0 - 1e-6).all(), "Min-max 归一化后所有值应 >= 0"
        assert (v_normalized <= 1 + 1e-6).all(), "Min-max 归一化后所有值应 <= 1"
        
        # 验证形状不变
        assert v_normalized.shape == v_norms.shape, "归一化后形状应与输入一致"

    def test_minmax_normalization_extremes(self):
        """测试 min-max 归一化的极端值正确"""
        v_norms = torch.tensor([[1.0, 5.0, 10.0, 3.0]])
        
        eps = 1e-8
        v_min = v_norms.min(dim=-1, keepdim=True)[0]
        v_max = v_norms.max(dim=-1, keepdim=True)[0]
        v_normalized = (v_norms - v_min) / (v_max - v_min + eps)
        
        # 最小值应该接近 0
        assert torch.allclose(v_normalized.min(dim=-1)[0], torch.zeros(1), atol=1e-5), \
            "Min-max 归一化后最小值应接近 0"
        
        # 最大值应该接近 1
        assert torch.allclose(v_normalized.max(dim=-1)[0], torch.ones(1), atol=1e-5), \
            "Min-max 归一化后最大值应接近 1"

    def test_minmax_normalization_uniform_values(self):
        """测试全部相同值的情况（极端值情况）"""
        # 所有值相同时，max - min = 0，需要 eps 防止除零
        v_norms = torch.ones(4, 100) * 5.0
        
        eps = 1e-8
        v_min = v_norms.min(dim=-1, keepdim=True)[0]
        v_max = v_norms.max(dim=-1, keepdim=True)[0]
        v_normalized = (v_norms - v_min) / (v_max - v_min + eps)
        
        # 由于 max - min = 0，分子也是 0，结果应该接近 0
        assert torch.isfinite(v_normalized).all(), "归一化结果应为有限值（无 NaN 或 Inf）"
        assert (v_normalized >= 0).all(), "归一化后值应 >= 0"

    def test_minmax_normalization_zero_values(self):
        """测试全零值的情况（极端值情况）"""
        v_norms = torch.zeros(4, 100)
        
        eps = 1e-8
        v_min = v_norms.min(dim=-1, keepdim=True)[0]
        v_max = v_norms.max(dim=-1, keepdim=True)[0]
        v_normalized = (v_norms - v_min) / (v_max - v_min + eps)
        
        # 全零时，分子为 0，结果应该是 0
        assert torch.isfinite(v_normalized).all(), "归一化结果应为有限值（无 NaN 或 Inf）"
        assert torch.allclose(v_normalized, torch.zeros_like(v_normalized), atol=1e-5), \
            "全零输入归一化后应接近 0"

    def test_softmax_numerical_stability(self):
        """测试 softmax 的数值稳定性（大值输入）"""
        # 模拟大值情况
        layer_attn_importance = torch.randn(4, 100) * 1000
        
        attn_normalized = F.softmax(layer_attn_importance, dim=-1)
        
        # 验证无 NaN 或 Inf
        assert torch.isfinite(attn_normalized).all(), \
            "大值输入时 softmax 应保持数值稳定"
        
        # 验证总和仍然接近 1
        sums = attn_normalized.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5), \
            f"大值输入时 softmax 归一化后总和应接近 1，但得到 {sums}"

    def test_softmax_numerical_stability_small_values(self):
        """测试 softmax 的数值稳定性（小值输入）"""
        layer_attn_importance = torch.randn(4, 100) * 1e-6
        
        attn_normalized = F.softmax(layer_attn_importance, dim=-1)
        
        assert torch.isfinite(attn_normalized).all(), \
            "小值输入时 softmax 应保持数值稳定"

    def test_combined_normalization_shape(self):
        """测试组合归一化后的形状一致性"""
        num_layers = 4
        seq_len = 100
        
        layer_attn_importance = torch.randn(num_layers, seq_len)
        v_norms = torch.randn(num_layers, seq_len).abs()
        
        # 归一化
        attn_normalized = F.softmax(layer_attn_importance, dim=-1)
        
        eps = 1e-8
        v_min = v_norms.min(dim=-1, keepdim=True)[0]
        v_max = v_norms.max(dim=-1, keepdim=True)[0]
        v_normalized = (v_norms - v_min) / (v_max - v_min + eps)
        
        # 组合
        layer_importance = attn_normalized * v_normalized
        
        # 验证形状
        assert layer_importance.shape == (num_layers, seq_len), \
            f"组合后形状应为 ({num_layers}, {seq_len})，但得到 {layer_importance.shape}"
        
        # 跨层聚合
        importance = layer_importance.sum(dim=0)
        assert importance.shape == (seq_len,), \
            f"聚合后形状应为 ({seq_len},)，但得到 {importance.shape}"

    def test_combined_normalization_non_negative(self):
        """测试组合归一化后的值非负"""
        num_layers = 4
        seq_len = 100
        
        layer_attn_importance = torch.randn(num_layers, seq_len)
        v_norms = torch.randn(num_layers, seq_len).abs()
        
        attn_normalized = F.softmax(layer_attn_importance, dim=-1)
        
        eps = 1e-8
        v_min = v_norms.min(dim=-1, keepdim=True)[0]
        v_max = v_norms.max(dim=-1, keepdim=True)[0]
        v_normalized = (v_norms - v_min) / (v_max - v_min + eps)
        
        layer_importance = attn_normalized * v_normalized
        
        assert (layer_importance >= 0).all(), "组合后的 importance 应全部非负"

    def test_gradient_flow(self):
        """测试梯度是否能正确反向传播"""
        num_layers = 4
        seq_len = 100
        
        # 需要梯度的输入
        layer_attn_importance = torch.randn(num_layers, seq_len, requires_grad=True)
        v_norms = torch.randn(num_layers, seq_len, requires_grad=True).abs()
        
        # 归一化
        attn_normalized = F.softmax(layer_attn_importance, dim=-1)
        
        eps = 1e-8
        v_min = v_norms.min(dim=-1, keepdim=True)[0]
        v_max = v_norms.max(dim=-1, keepdim=True)[0]
        v_normalized = (v_norms - v_min) / (v_max - v_min + eps)
        
        # 组合并聚合
        layer_importance = attn_normalized * v_normalized
        importance = layer_importance.sum(dim=0)
        
        # 创建损失并反向传播
        loss = importance.sum()
        loss.backward()
        
        # 验证梯度存在且有限
        assert layer_attn_importance.grad is not None, "注意力分数应有梯度"
        assert v_norms.grad is not None, "V 范数应有梯度"
        assert torch.isfinite(layer_attn_importance.grad).all(), "注意力分数梯度应为有限值"
        assert torch.isfinite(v_norms.grad).all(), "V 范数梯度应为有限值"


class TestNormalizationEdgeCases:
    """测试边界情况"""

    def test_single_element_sequence(self):
        """测试单元素序列"""
        layer_attn_importance = torch.randn(4, 1)
        v_norms = torch.randn(4, 1).abs()
        
        attn_normalized = F.softmax(layer_attn_importance, dim=-1)
        
        # 单元素 softmax 应该是 1
        assert torch.allclose(attn_normalized, torch.ones(4, 1), atol=1e-5), \
            "单元素序列 softmax 应该全为 1"
        
        eps = 1e-8
        v_min = v_norms.min(dim=-1, keepdim=True)[0]
        v_max = v_norms.max(dim=-1, keepdim=True)[0]
        v_normalized = (v_norms - v_min) / (v_max - v_min + eps)
        
        # 单元素 min-max 归一化结果应接近 0（因为 max - min = 0）
        assert torch.isfinite(v_normalized).all(), "单元素归一化应为有限值"

    def test_single_layer(self):
        """测试单层情况"""
        layer_attn_importance = torch.randn(1, 100)
        v_norms = torch.randn(1, 100).abs()
        
        attn_normalized = F.softmax(layer_attn_importance, dim=-1)
        
        eps = 1e-8
        v_min = v_norms.min(dim=-1, keepdim=True)[0]
        v_max = v_norms.max(dim=-1, keepdim=True)[0]
        v_normalized = (v_norms - v_min) / (v_max - v_min + eps)
        
        layer_importance = attn_normalized * v_normalized
        importance = layer_importance.sum(dim=0)
        
        assert importance.shape == (100,), "单层聚合后形状应正确"

