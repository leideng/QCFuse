"""
Triton kernels for efficient attention score computation.

Used by cache blending to compute importance scores for K tokens.
"""

import torch
import triton
import triton.language as tl
import math

@triton.jit
def _compute_lse_splitk_kernel(
    Q, K, M_OUT, L_OUT,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_mb, stride_mh, stride_msplit, stride_mm,
    n_heads_q: tl.constexpr,
    n_heads_k: tl.constexpr,
    n_ctx_q, n_ctx_k,
    sm_scale,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_split = tl.program_id(2)
    
    batch_idx = pid_bh // n_heads_q
    head_q = pid_bh % n_heads_q
    
    group_size = n_heads_q // n_heads_k
    h_k = head_q // group_size
    
    start_k_base = pid_split * SPLIT_K
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    
    q_mask = offs_m[:, None] < n_ctx_q
    q_ptrs = Q + (batch_idx * stride_qb + head_q * stride_qh) + \
             (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    for start_n in range(0, SPLIT_K, BLOCK_N):
        k_idx = start_k_base + start_n
            
        offs_n = k_idx + tl.arange(0, BLOCK_N)
        k_mask = offs_n[None, :] < n_ctx_k
        
        k_ptrs = K + (batch_idx * stride_kb + h_k * stride_kh) + \
                 (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd)
        k = tl.load(k_ptrs, mask=tl.broadcast_to(k_mask, (HEAD_DIM, BLOCK_N)), other=0.0)
        
        qk = tl.dot(q, k) * sm_scale
        qk = tl.where(
            tl.broadcast_to(offs_n[None, :] < n_ctx_k, (BLOCK_M, BLOCK_N)),
            qk,
            float("-inf")
        )
        
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        valid_m = m_new != -float("inf")
        alpha = tl.where(valid_m, tl.exp(m_i - m_new), 0.0)
        p_ij = tl.where(valid_m[:, None], tl.exp(qk - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p_ij, 1)
        m_i = m_new
        
    m_out_ptrs = M_OUT + batch_idx * stride_mb + head_q * stride_mh + pid_split * stride_msplit + offs_m * stride_mm
    l_out_ptrs = L_OUT + batch_idx * stride_mb + head_q * stride_mh + pid_split * stride_msplit + offs_m * stride_mm
    
    tl.store(m_out_ptrs, m_i, mask=offs_m < n_ctx_q)
    tl.store(l_out_ptrs, l_i, mask=offs_m < n_ctx_q)


@triton.jit
def _compute_lse_splitk_causal_kernel(
    Q, K, M_OUT, L_OUT,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_mb, stride_mh, stride_msplit, stride_mm,
    n_heads_q: tl.constexpr,
    n_heads_k: tl.constexpr,
    n_ctx_q, n_ctx_k,
    q_start,
    sm_scale,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_split = tl.program_id(2)

    batch_idx = pid_bh // n_heads_q
    head_q = pid_bh % n_heads_q

    group_size = n_heads_q // n_heads_k
    h_k = head_q // group_size

    start_k_base = pid_split * SPLIT_K

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_mask = offs_m[:, None] < n_ctx_q
    q_ptrs = Q + (batch_idx * stride_qb + head_q * stride_qh) + \
             (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    q_abs = q_start + offs_m

    for start_n in range(0, SPLIT_K, BLOCK_N):
        k_idx = start_k_base + start_n

        offs_n = k_idx + tl.arange(0, BLOCK_N)
        k_mask = offs_n[None, :] < n_ctx_k

        k_ptrs = K + (batch_idx * stride_kb + h_k * stride_kh) + \
                 (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd)
        k = tl.load(k_ptrs, mask=tl.broadcast_to(k_mask, (HEAD_DIM, BLOCK_N)), other=0.0)

        qk = tl.dot(q, k) * sm_scale
        valid = (
            tl.broadcast_to(offs_m[:, None] < n_ctx_q, (BLOCK_M, BLOCK_N))
            & tl.broadcast_to(offs_n[None, :] < n_ctx_k, (BLOCK_M, BLOCK_N))
        )
        if CAUSAL:
            valid = valid & (offs_n[None, :] <= q_abs[:, None])
        qk = tl.where(valid, qk, float("-inf"))

        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        valid_m = m_new != -float("inf")
        alpha = tl.where(valid_m, tl.exp(m_i - m_new), 0.0)
        p_ij = tl.where(valid_m[:, None], tl.exp(qk - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p_ij, 1)
        m_i = m_new

    m_out_ptrs = M_OUT + batch_idx * stride_mb + head_q * stride_mh + pid_split * stride_msplit + offs_m * stride_mm
    l_out_ptrs = L_OUT + batch_idx * stride_mb + head_q * stride_mh + pid_split * stride_msplit + offs_m * stride_mm

    tl.store(m_out_ptrs, m_i, mask=offs_m < n_ctx_q)
    tl.store(l_out_ptrs, l_i, mask=offs_m < n_ctx_q)


@triton.jit
def _reduce_lse_kernel(
    M_IN, L_IN, LSE_OUT,
    stride_mb, stride_mh, stride_msplit, stride_mm,
    stride_lse_b, stride_lse_h, stride_lse_m,
    n_splits,
    n_ctx_q,
    n_heads_q: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // n_heads_q
    pid_h = pid_bh % n_heads_q
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    m_global = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_global = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    for split_idx in range(n_splits):
        m_ptrs = M_IN + pid_b * stride_mb + pid_h * stride_mh + split_idx * stride_msplit + offs_m * stride_mm
        l_ptrs = L_IN + pid_b * stride_mb + pid_h * stride_mh + split_idx * stride_msplit + offs_m * stride_mm
        
        m_i = tl.load(m_ptrs, mask=offs_m < n_ctx_q, other=-float("inf"))
        l_i = tl.load(l_ptrs, mask=offs_m < n_ctx_q, other=0.0)
        
        m_new = tl.maximum(m_global, m_i)
        
        # Avoid inf-inf
        valid_m = m_new != -float("inf")
        alpha_global = tl.where(valid_m, tl.exp(m_global - m_new), 0.0)
        alpha_i = tl.where(valid_m, tl.exp(m_i - m_new), 0.0)
        
        l_global = l_global * alpha_global + l_i * alpha_i
        m_global = m_new
        
    lse = tl.where(l_global > 0.0, m_global + tl.log(l_global), -float("inf"))
    lse_ptrs = LSE_OUT + pid_b * stride_lse_b + pid_h * stride_lse_h + offs_m * stride_lse_m
    tl.store(lse_ptrs, lse, mask=offs_m < n_ctx_q)


@triton.jit 
def _flash_importance_kernel(
    Q, K, LSE, IMPORTANCE,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_lse_b, stride_lse_h, stride_lse_m,
    stride_imp_b, stride_imp_k,
    n_heads_q: tl.constexpr,
    n_heads_k: tl.constexpr,
    n_ctx_q, n_ctx_k,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    
    importance = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    group_size = n_heads_q // n_heads_k
    
    for h_k in range(n_heads_k):
        k_mask = offs_n[None, :] < n_ctx_k
        k_ptrs = K + (pid_b * stride_kb + h_k * stride_kh) + \
                 (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd)
        k = tl.load(k_ptrs, mask=tl.broadcast_to(k_mask, (HEAD_DIM, BLOCK_N)), other=0.0)
        
        for g in range(group_size):
            h_q = h_k * group_size + g
            
            for start_m in range(0, n_ctx_q, BLOCK_M):
                offs_m = start_m + tl.arange(0, BLOCK_M)
                q_mask = offs_m[:, None] < n_ctx_q
                
                q_ptrs = Q + (pid_b * stride_qb + h_q * stride_qh) + \
                         (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
                q = tl.load(q_ptrs, mask=q_mask, other=0.0)
                
                lse_ptrs = LSE + pid_b * stride_lse_b + h_q * stride_lse_h + offs_m * stride_lse_m
                lse = tl.load(lse_ptrs, mask=offs_m < n_ctx_q, other=0.0)
                
                qk = tl.dot(q, k) * sm_scale
                attn_weight = tl.exp(qk - lse[:, None])
                
                attn_weight = tl.where(
                    tl.broadcast_to(offs_m[:, None] < n_ctx_q, (BLOCK_M, BLOCK_N)) & 
                    tl.broadcast_to(offs_n[None, :] < n_ctx_k, (BLOCK_M, BLOCK_N)),
                    attn_weight,
                    0.0
                )
                
                max_p_over_M = tl.max(attn_weight, 0)
                importance = tl.maximum(importance, max_p_over_M)
    
    out_ptrs = IMPORTANCE + pid_b * stride_imp_b + offs_n * stride_imp_k
    tl.store(out_ptrs, importance, mask=offs_n < n_ctx_k)


@triton.jit
def _flash_importance_target_kernel(
    Q, K, LSE, IMPORTANCE,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_lse_b, stride_lse_h, stride_lse_m,
    stride_imp_b, stride_imp_k,
    n_heads_q: tl.constexpr,
    n_heads_k: tl.constexpr,
    n_ctx_q, n_ctx_k,
    target_start, target_len,
    q_start,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_local = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = target_start + offs_local
    offs_d = tl.arange(0, HEAD_DIM)

    importance = tl.zeros([BLOCK_N], dtype=tl.float32)

    group_size = n_heads_q // n_heads_k

    for h_k in range(n_heads_k):
        k_mask = (offs_local[None, :] < target_len) & (offs_n[None, :] < n_ctx_k)
        k_ptrs = K + (pid_b * stride_kb + h_k * stride_kh) + \
                 (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd)
        k = tl.load(k_ptrs, mask=tl.broadcast_to(k_mask, (HEAD_DIM, BLOCK_N)), other=0.0)

        for g in range(group_size):
            h_q = h_k * group_size + g

            for start_m in range(0, n_ctx_q, BLOCK_M):
                offs_m = start_m + tl.arange(0, BLOCK_M)
                q_mask = offs_m[:, None] < n_ctx_q

                q_ptrs = Q + (pid_b * stride_qb + h_q * stride_qh) + \
                         (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
                q = tl.load(q_ptrs, mask=q_mask, other=0.0)

                lse_ptrs = LSE + pid_b * stride_lse_b + h_q * stride_lse_h + offs_m * stride_lse_m
                lse = tl.load(lse_ptrs, mask=offs_m < n_ctx_q, other=0.0)

                qk = tl.dot(q, k) * sm_scale
                attn_weight = tl.exp(qk - lse[:, None])

                valid = (
                    tl.broadcast_to(offs_m[:, None] < n_ctx_q, (BLOCK_M, BLOCK_N))
                    & tl.broadcast_to(offs_local[None, :] < target_len, (BLOCK_M, BLOCK_N))
                    & tl.broadcast_to(offs_n[None, :] < n_ctx_k, (BLOCK_M, BLOCK_N))
                )
                if CAUSAL:
                    q_abs = q_start + offs_m
                    valid = valid & (offs_n[None, :] <= q_abs[:, None])
                attn_weight = tl.where(valid, attn_weight, 0.0)

                max_p_over_M = tl.max(attn_weight, 0)
                importance = tl.maximum(importance, max_p_over_M)

    out_ptrs = IMPORTANCE + pid_b * stride_imp_b + offs_local * stride_imp_k
    tl.store(out_ptrs, importance, mask=offs_local < target_len)


@triton.jit
def _flash_mean_target_kernel(
    Q, K, LSE, IMPORTANCE,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_lse_b, stride_lse_h, stride_lse_m,
    stride_imp_b, stride_imp_k,
    n_heads_q: tl.constexpr,
    n_heads_k: tl.constexpr,
    n_ctx_q, n_ctx_k,
    target_start, target_len,
    q_start,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_local = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = target_start + offs_local
    offs_d = tl.arange(0, HEAD_DIM)

    importance = tl.zeros([BLOCK_N], dtype=tl.float32)

    group_size = n_heads_q // n_heads_k

    for h_k in range(n_heads_k):
        k_mask = (offs_local[None, :] < target_len) & (offs_n[None, :] < n_ctx_k)
        k_ptrs = K + (pid_b * stride_kb + h_k * stride_kh) + \
                 (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd)
        k = tl.load(k_ptrs, mask=tl.broadcast_to(k_mask, (HEAD_DIM, BLOCK_N)), other=0.0)

        for g in range(group_size):
            h_q = h_k * group_size + g

            for start_m in range(0, n_ctx_q, BLOCK_M):
                offs_m = start_m + tl.arange(0, BLOCK_M)
                q_mask = offs_m[:, None] < n_ctx_q

                q_ptrs = Q + (pid_b * stride_qb + h_q * stride_qh) + \
                         (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
                q = tl.load(q_ptrs, mask=q_mask, other=0.0)

                lse_ptrs = LSE + pid_b * stride_lse_b + h_q * stride_lse_h + offs_m * stride_lse_m
                lse = tl.load(lse_ptrs, mask=offs_m < n_ctx_q, other=0.0)

                qk = tl.dot(q, k) * sm_scale
                attn_weight = tl.exp(qk - lse[:, None])

                valid = (
                    tl.broadcast_to(offs_m[:, None] < n_ctx_q, (BLOCK_M, BLOCK_N))
                    & tl.broadcast_to(offs_local[None, :] < target_len, (BLOCK_M, BLOCK_N))
                    & tl.broadcast_to(offs_n[None, :] < n_ctx_k, (BLOCK_M, BLOCK_N))
                )
                if CAUSAL:
                    q_abs = q_start + offs_m
                    valid = valid & (offs_n[None, :] <= q_abs[:, None])
                attn_weight = tl.where(valid, attn_weight, 0.0)

                importance += tl.sum(attn_weight, 0)

    importance = importance / (n_ctx_q * n_heads_q)
    out_ptrs = IMPORTANCE + pid_b * stride_imp_b + offs_local * stride_imp_k
    tl.store(out_ptrs, importance, mask=offs_local < target_len)


@triton.jit
def _sum_importance_kernel(
    IMPORTANCE_IN, IMPORTANCE_OUT,
    stride_in_b, stride_in_k,
    stride_out_k,
    n_batch, n_ctx_k,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    final_importance = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    for b in range(n_batch):
        in_ptrs = IMPORTANCE_IN + b * stride_in_b + offs_n * stride_in_k
        val = tl.load(in_ptrs, mask=offs_n < n_ctx_k, other=0.0)
        final_importance += val
        
    out_ptrs = IMPORTANCE_OUT + offs_n * stride_out_k
    tl.store(out_ptrs, final_importance, mask=offs_n < n_ctx_k)


@torch.compiler.disable
def compute_att_softmax_importance(
    q: torch.Tensor, 
    k: torch.Tensor
) -> torch.Tensor:
    """
    Compute softmax-based importance scores for each K position.

    The cross-layer sum is folded into the output so each token receives one
    aggregate importance score.
    
    Args:
        q: [Batch, Seq_Q, Heads_Q, Head_Dim]
        k: [Batch, Seq_K, Heads_K, Head_Dim]
    
    Returns:
        importance: [Seq_K]
    """
    batch, seq_q, heads_q, dim = q.shape
    _, seq_k, heads_k, _ = k.shape
    
    sm_scale = 1.0 / math.sqrt(dim)
    
    BLOCK_M = triton.next_power_of_2(seq_q) if seq_q > 0 else 64
    if BLOCK_M < 16:
        BLOCK_M = 16
    if BLOCK_M > 64:
        BLOCK_M = 64
    
    BLOCK_N = 128
    SPLIT_K = 1024
    num_splits = triton.cdiv(seq_k, SPLIT_K)
    
    # Allocations for Split-K
    m_out = torch.empty((batch, heads_q, num_splits, seq_q), device=q.device, dtype=torch.float32)
    l_out = torch.empty((batch, heads_q, num_splits, seq_q), device=q.device, dtype=torch.float32)
    lse = torch.empty((batch, heads_q, seq_q), device=q.device, dtype=torch.float32)
    
    # Pass 1.1: Local LSE computation
    grid_splitk = (triton.cdiv(seq_q, BLOCK_M), batch * heads_q, num_splits)
    _compute_lse_splitk_kernel[grid_splitk](
        q, k, m_out, l_out,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        m_out.stride(0), m_out.stride(1), m_out.stride(2), m_out.stride(3),
        heads_q, heads_k,
        seq_q, seq_k, sm_scale,
        SPLIT_K=SPLIT_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=dim,
        num_warps=4,
        num_stages=3,
    )
    
    # Pass 1.2: Global LSE reduction
    grid_reduce = (triton.cdiv(seq_q, BLOCK_M), batch * heads_q)
    _reduce_lse_kernel[grid_reduce](
        m_out, l_out, lse,
        m_out.stride(0), m_out.stride(1), m_out.stride(2), m_out.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        num_splits, seq_q, heads_q,
        BLOCK_M=BLOCK_M,
        num_warps=4,
    )
    
    # Pass 2: Importance computation
    BLOCK_M_IMP = triton.next_power_of_2(seq_q) if seq_q > 0 else 64
    if BLOCK_M_IMP < 32:
        BLOCK_M_IMP = 32
    if BLOCK_M_IMP > 128:
        BLOCK_M_IMP = 128
    
    BLOCK_N_IMP = 64
    importance = torch.empty((batch, seq_k), device=q.device, dtype=torch.float32)
    grid_imp = (triton.cdiv(seq_k, BLOCK_N_IMP), batch)
    
    _flash_importance_kernel[grid_imp](
        q, k, lse, importance,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        importance.stride(0), importance.stride(1),
        heads_q, heads_k,
        seq_q, seq_k, sm_scale,
        BLOCK_M=BLOCK_M_IMP,
        BLOCK_N=BLOCK_N_IMP,
        HEAD_DIM=dim,
        num_warps=4,
        num_stages=2,
    )
    
    # Pass 3: Importance sum over batch
    out_importance = torch.empty((seq_k,), device=q.device, dtype=torch.float32)
    grid_sum = (triton.cdiv(seq_k, BLOCK_N_IMP),)
    _sum_importance_kernel[grid_sum](
        importance, out_importance,
        importance.stride(0), importance.stride(1),
        out_importance.stride(0),
        batch, seq_k,
        BLOCK_N=BLOCK_N_IMP,
    )
    
    return out_importance


@torch.compiler.disable
def compute_att_full_softmax_importance_by_batch(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    target_start: int,
    target_len: int,
    q_start: int,
    causal: bool = True,
) -> torch.Tensor:
    """
    Compute per-batch target-K importance with the full visible KV denominator.

    Args:
        q: [Batch, Seq_Q, Heads_Q, Head_Dim]
        k: [Batch, Seq_K, Heads_K, Head_Dim], full denominator KV
        target_start: start offset of the target chunk inside k
        target_len: number of target chunk tokens to score
        q_start: start offset of q inside k for causal masking
        causal: if true, query i can only attend to K positions <= q_start + i

    Returns:
        importance: [Batch, target_len]
    """
    batch, seq_q, heads_q, dim = q.shape
    _, seq_k, heads_k, _ = k.shape

    target_start = int(target_start)
    target_len = int(target_len)
    q_start = int(q_start)
    target_start = max(0, min(target_start, seq_k))
    target_len = max(0, min(target_len, seq_k - target_start))
    if target_len == 0:
        return torch.empty((batch, 0), device=q.device, dtype=torch.float32)

    sm_scale = 1.0 / math.sqrt(dim)

    BLOCK_M = triton.next_power_of_2(seq_q) if seq_q > 0 else 64
    if BLOCK_M < 16:
        BLOCK_M = 16
    if BLOCK_M > 64:
        BLOCK_M = 64

    BLOCK_N = 128
    SPLIT_K = 1024
    num_splits = triton.cdiv(seq_k, SPLIT_K)

    m_out = torch.empty((batch, heads_q, num_splits, seq_q), device=q.device, dtype=torch.float32)
    l_out = torch.empty((batch, heads_q, num_splits, seq_q), device=q.device, dtype=torch.float32)
    lse = torch.empty((batch, heads_q, seq_q), device=q.device, dtype=torch.float32)

    grid_splitk = (triton.cdiv(seq_q, BLOCK_M), batch * heads_q, num_splits)
    _compute_lse_splitk_causal_kernel[grid_splitk](
        q, k, m_out, l_out,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        m_out.stride(0), m_out.stride(1), m_out.stride(2), m_out.stride(3),
        heads_q, heads_k,
        seq_q, seq_k, q_start, sm_scale,
        SPLIT_K=SPLIT_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=dim,
        CAUSAL=causal,
        num_warps=4,
        num_stages=3,
    )

    grid_reduce = (triton.cdiv(seq_q, BLOCK_M), batch * heads_q)
    _reduce_lse_kernel[grid_reduce](
        m_out, l_out, lse,
        m_out.stride(0), m_out.stride(1), m_out.stride(2), m_out.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        num_splits, seq_q, heads_q,
        BLOCK_M=BLOCK_M,
        num_warps=4,
    )

    BLOCK_M_IMP = triton.next_power_of_2(seq_q) if seq_q > 0 else 64
    if BLOCK_M_IMP < 32:
        BLOCK_M_IMP = 32
    if BLOCK_M_IMP > 128:
        BLOCK_M_IMP = 128

    BLOCK_N_IMP = 64
    importance = torch.empty((batch, target_len), device=q.device, dtype=torch.float32)
    grid_imp = (triton.cdiv(target_len, BLOCK_N_IMP), batch)
    _flash_importance_target_kernel[grid_imp](
        q, k, lse, importance,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        importance.stride(0), importance.stride(1),
        heads_q, heads_k,
        seq_q, seq_k,
        target_start, target_len,
        q_start, sm_scale,
        BLOCK_M=BLOCK_M_IMP,
        BLOCK_N=BLOCK_N_IMP,
        HEAD_DIM=dim,
        CAUSAL=causal,
        num_warps=4,
        num_stages=2,
    )

    return importance


@torch.compiler.disable
def compute_att_full_softmax_mean_by_batch(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    target_start: int,
    target_len: int,
    q_start: int,
    causal: bool = True,
) -> torch.Tensor:
    """
    Compute per-batch target-K mean attention mass with the full visible KV denominator.

    This differs from compute_att_full_softmax_importance_by_batch, which keeps
    the maximum softmax probability over query/head positions. Here each output
    value is the mean softmax probability over all query heads and query tokens.

    Args:
        q: [Batch, Seq_Q, Heads_Q, Head_Dim]
        k: [Batch, Seq_K, Heads_K, Head_Dim], full denominator KV
        target_start: start offset of the target chunk inside k
        target_len: number of target chunk tokens to score
        q_start: start offset of q inside k for causal masking
        causal: if true, query i can only attend to K positions <= q_start + i

    Returns:
        attention_mass: [Batch, target_len]
    """
    batch, seq_q, heads_q, dim = q.shape
    _, seq_k, heads_k, _ = k.shape

    target_start = int(target_start)
    target_len = int(target_len)
    q_start = int(q_start)
    target_start = max(0, min(target_start, seq_k))
    target_len = max(0, min(target_len, seq_k - target_start))
    if target_len == 0:
        return torch.empty((batch, 0), device=q.device, dtype=torch.float32)

    sm_scale = 1.0 / math.sqrt(dim)

    BLOCK_M = triton.next_power_of_2(seq_q) if seq_q > 0 else 64
    if BLOCK_M < 16:
        BLOCK_M = 16
    if BLOCK_M > 64:
        BLOCK_M = 64

    BLOCK_N = 128
    SPLIT_K = 1024
    num_splits = triton.cdiv(seq_k, SPLIT_K)

    m_out = torch.empty((batch, heads_q, num_splits, seq_q), device=q.device, dtype=torch.float32)
    l_out = torch.empty((batch, heads_q, num_splits, seq_q), device=q.device, dtype=torch.float32)
    lse = torch.empty((batch, heads_q, seq_q), device=q.device, dtype=torch.float32)

    grid_splitk = (triton.cdiv(seq_q, BLOCK_M), batch * heads_q, num_splits)
    _compute_lse_splitk_causal_kernel[grid_splitk](
        q, k, m_out, l_out,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        m_out.stride(0), m_out.stride(1), m_out.stride(2), m_out.stride(3),
        heads_q, heads_k,
        seq_q, seq_k, q_start, sm_scale,
        SPLIT_K=SPLIT_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=dim,
        CAUSAL=causal,
        num_warps=4,
        num_stages=3,
    )

    grid_reduce = (triton.cdiv(seq_q, BLOCK_M), batch * heads_q)
    _reduce_lse_kernel[grid_reduce](
        m_out, l_out, lse,
        m_out.stride(0), m_out.stride(1), m_out.stride(2), m_out.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        num_splits, seq_q, heads_q,
        BLOCK_M=BLOCK_M,
        num_warps=4,
    )

    BLOCK_M_IMP = triton.next_power_of_2(seq_q) if seq_q > 0 else 64
    if BLOCK_M_IMP < 32:
        BLOCK_M_IMP = 32
    if BLOCK_M_IMP > 128:
        BLOCK_M_IMP = 128

    BLOCK_N_IMP = 64
    attention_mass = torch.empty((batch, target_len), device=q.device, dtype=torch.float32)
    grid_imp = (triton.cdiv(target_len, BLOCK_N_IMP), batch)
    _flash_mean_target_kernel[grid_imp](
        q, k, lse, attention_mass,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        attention_mass.stride(0), attention_mass.stride(1),
        heads_q, heads_k,
        seq_q, seq_k,
        target_start, target_len,
        q_start, sm_scale,
        BLOCK_M=BLOCK_M_IMP,
        BLOCK_N=BLOCK_N_IMP,
        HEAD_DIM=dim,
        CAUSAL=causal,
        num_warps=4,
        num_stages=2,
    )

    return attention_mass


@torch.compiler.disable
def compute_att_full_softmax_mean(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    target_start: int,
    target_len: int,
    q_start: int,
    causal: bool = True,
) -> torch.Tensor:
    """
    Compute target-K mean attention mass averaged over layers/batch.

    Returns:
        attention_mass: [target_len]
    """
    attention_mass = compute_att_full_softmax_mean_by_batch(
        q,
        k,
        target_start=target_start,
        target_len=target_len,
        q_start=q_start,
        causal=causal,
    )

    batch = attention_mass.shape[0]
    target_len = attention_mass.shape[1]
    if target_len == 0:
        return torch.empty((0,), device=q.device, dtype=torch.float32)

    out_importance = torch.empty((target_len,), device=q.device, dtype=torch.float32)
    BLOCK_N_IMP = 64
    grid_sum = (triton.cdiv(target_len, BLOCK_N_IMP),)
    _sum_importance_kernel[grid_sum](
        attention_mass, out_importance,
        attention_mass.stride(0), attention_mass.stride(1),
        out_importance.stride(0),
        batch, target_len,
        BLOCK_N=BLOCK_N_IMP,
    )

    return out_importance / batch


@torch.compiler.disable
def compute_att_full_softmax_importance(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    target_start: int,
    target_len: int,
    q_start: int,
    causal: bool = True,
) -> torch.Tensor:
    """
    Compute target-K importance summed over batch/layers.

    Args:
        q: [Batch, Seq_Q, Heads_Q, Head_Dim]
        k: [Batch, Seq_K, Heads_K, Head_Dim], full denominator KV
        target_start: start offset of the target chunk inside k
        target_len: number of target chunk tokens to score
        q_start: start offset of q inside k for causal masking
        causal: if true, query i can only attend to K positions <= q_start + i

    Returns:
        importance: [target_len]
    """
    importance = compute_att_full_softmax_importance_by_batch(
        q,
        k,
        target_start=target_start,
        target_len=target_len,
        q_start=q_start,
        causal=causal,
    )

    batch = importance.shape[0]
    target_len = importance.shape[1]
    if target_len == 0:
        return torch.empty((0,), device=q.device, dtype=torch.float32)

    out_importance = torch.empty((target_len,), device=q.device, dtype=torch.float32)
    BLOCK_N_IMP = 64
    grid_sum = (triton.cdiv(target_len, BLOCK_N_IMP),)
    _sum_importance_kernel[grid_sum](
        importance, out_importance,
        importance.stride(0), importance.stride(1),
        out_importance.stride(0),
        batch, target_len,
        BLOCK_N=BLOCK_N_IMP,
    )

    return out_importance


def warmup_triton_kernels(
    device: str = "cuda",
    head_dims: list = None,
    num_warmup_iters: int = 1,
    num_layers: int = 32,
    num_heads: int = 32,
    num_kv_heads: int = 8,
):
    """
    Warm up Triton kernels with production-like tensor shapes and layout.
    
    Args:
        device: CUDA device.
        head_dims: Head-dimension list, typically [128].
        num_warmup_iters: Number of warmup iterations.
        num_layers: Number of model layers, used as the batch dimension.
        num_heads: Number of Q heads.
        num_kv_heads: Number of K/V heads.
    """
    if head_dims is None:
        head_dims = [128]

    # Adjust warmup shapes for short and medium Q lengths
    warmup_shapes = [
        (32, 512),
        (32, 2048),
        (32, 4096),
        (64, 512),
        (128, 4096),
        (512, 4096),
    ]
    
    for head_dim in head_dims:
        for seq_q, seq_k in warmup_shapes:
            q = torch.randn(
                num_layers, seq_q, num_heads, head_dim,
                device=device, dtype=torch.float16
            )
            k = torch.randn(
                num_layers, seq_k, num_kv_heads, head_dim,
                device=device, dtype=torch.float16
            )
            
            for _ in range(num_warmup_iters):
                _ = compute_att_softmax_importance(q, k)

            torch.cuda.synchronize()
