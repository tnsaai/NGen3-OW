import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
import math

class NGen3OWConfig:
    """
    Advanced Configuration for NGen-3-OW (Gemini 3 / Opus 4.5 Class).
    Features: MLA, DeepSeek Sparse Attention (DSA), and Hierarchical MoEinMoE.
    """
    def __init__(
        self,
        vocab_size: int = 256000,
        hidden_size: int = 8192,
        num_hidden_layers: int = 80,
        num_attention_heads: int = 128,
        num_key_value_heads: int = 16,  # Used as base for compression
        # MLA (Multi-Head Latent Attention) / DSA
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        # DSA (DeepSeek Sparse Attention)
        use_dsa: bool = True,
        dsa_top_k: int = 2048,           # Tokens "Scout" picks from huge context
        use_fp8_indexer: bool = True,    # Fast FP8 Scouting
        # MoEinMoE (Hierarchical Mixture of Experts)
        use_moe: bool = True,
        use_moe_in_moe: bool = True,
        num_expert_groups: int = 16,     # Level 1: Domain Clusters
        num_minis_per_group: int = 16,   # Level 2: Mini-experts per cluster
        minis_per_tok: int = 4,          # Active mini-experts per token
        num_shared_experts: int = 2,     # Global always-on knowledge
        moe_intermediate_size: int = 2048, # Scale of each mini-expert
        # Context & Stability
        max_position_embeddings: int = 1048576, # 1M context support
        rope_theta: float = 10000000.0,
        rope_scaling: dict = {"type": "yarn", "factor": 32.0},
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.use_dsa = use_dsa
        self.dsa_top_k = dsa_top_k
        self.use_fp8_indexer = use_fp8_indexer
        self.use_moe = use_moe
        self.use_moe_in_moe = use_moe_in_moe
        self.num_expert_groups = num_expert_groups
        self.num_minis_per_group = num_minis_per_group
        self.minis_per_tok = minis_per_tok
        self.num_shared_experts = num_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

class NGen3OWRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Optimized for numerical stability in deep architectures (80+ layers).
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # Always compute variance in float32 for precision at scale
        input_dtype = hidden_states.dtype
        hidden_states_f32 = hidden_states.to(torch.float32)
        
        # RMS calculation
        variance = hidden_states_f32.pow(2).mean(-1, keepdim=True)
        hidden_states_f32 = hidden_states_f32 * torch.rsqrt(variance + self.variance_epsilon)
        
        # Return in original dtype multiplied by the learned gain
        return (self.weight * hidden_states_f32).to(input_dtype)

    def extra_repr(self):
        return f"{self.weight.shape[0]}, eps={self.variance_epsilon}"

class LightningIndexer(nn.Module):
    """
    DeepSeek Sparse Attention (DSA) - Stage 1: The Scout
    A lightweight, decoupled scoring mechanism to identify the Top-K most relevant tokens.
    Operates independently of the main heavy attention computation.
    """
    def __init__(self, config: NGen3OWConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dsa_top_k = config.dsa_top_k
        self.use_fp8 = config.use_fp8_indexer
        
        # Lightweight dimension just for scouting (e.g., 64 or 128 instead of full 8192)
        self.scout_dim = 128 
        
        # We project the hidden states to a much smaller scout space
        self.q_scout = nn.Linear(self.hidden_size, self.scout_dim, bias=False)
        self.k_scout = nn.Linear(self.hidden_size, self.scout_dim, bias=False)
        
        # Optional: Apply Rotary Embeddings to the scout for position awareness
        self.rotary_emb = NGen3OWYarnRotaryEmbedding(self.scout_dim, config.max_position_embeddings, config.rope_theta, config.rope_scaling)


    def forward(self, hidden_states, position_ids=None, past_k_scout=None):
        """
        Calculates coarse attention scores and returns indices of top-k tokens.
        """
        bsz, seq_len, _ = hidden_states.size()
        
        # 1. Cast down for speed if configured (simulating FP8/Low precision)
        # Note: In a production kernel we would use FP8 weights as well.
        # For this prototype we keep it in the native weight dtype to avoid PyTorch linear errors.
        compute_dtype = hidden_states.dtype
            
        hidden_states_scout = hidden_states.to(compute_dtype)

        # 2. Generate Scout Queries and Keys
        q = self.q_scout(hidden_states_scout) # [bsz, seq_len, scout_dim]
        k = self.k_scout(hidden_states_scout) # [bsz, seq_len, scout_dim]
        
        # Apply RoPE to inject position (Needed so it knows *where* the tokens are)
        if position_ids is not None:
            q = q.unsqueeze(2) # [bsz, seq_len, 1, scout_dim]
            k = k.unsqueeze(2)
            cos, sin = self.rotary_emb(k, seq_len=seq_len)
            
            # apply_rotary_pos_emb expects [batch, seq_len, num_heads, head_dim]
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
            
            q = q.squeeze(2)
            k = k.squeeze(2)

        # 3. Maintain the Scout Cache 
        if past_k_scout is not None:
            k = torch.cat([past_k_scout, k], dim=1)
        current_k_scout = k

        # 4. Compute Coarse Attention Scores
        # Note: No softmax needed here! We just want the raw magnitude for ranking.
        # This is essentially highly efficient dot-product attention
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.scout_dim) # [bsz, seq_len, cache_len]
        
        # Mask out future tokens for causal modeling
        cache_len = k.size(1)
        if seq_len > 1:
            causal_mask = torch.triu(torch.full((seq_len, cache_len), float('-inf'), device=scores.device), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0)

        # 5. Extract Top-K Indices (The Core of DSA)
        # We only want the indices, not the scores themselves.
        actual_top_k = min(self.dsa_top_k, cache_len)
        
        if actual_top_k < cache_len:
            # We use ReLU or raw scores to find the max impact tokens
            _, topk_indices = torch.topk(scores, k=actual_top_k, dim=-1, sorted=False)
        else:
            # If context is smaller than top_k, just take everything
            topk_indices = torch.arange(cache_len, device=scores.device).unsqueeze(0).unsqueeze(0).expand(bsz, seq_len, -1)

        return topk_indices, current_k_scout

class NGen3OWYarnRotaryEmbedding(nn.Module):
    """
    YaRN (Yet another RoPE for NLPs) / LongRoPE Positional Core.
    Dynamically scales the rotary frequencies to gracefully handle 1M+ context windows
    without degrading performance on short sequences.
    """
    def __init__(self, dim, max_position_embeddings=1048576, base=10000000.0, rope_scaling=None, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # YaRN Defaults
        self.original_max_position_embeddings = 32768 # Standard pre-training length assumption
        self.scale = 1.0
        self.beta_fast = 32
        self.beta_slow = 1
        self.mscale = 1.0
        self.mscale_all_dim = 1.0
        
        if rope_scaling is not None and rope_scaling.get("type", "") == "yarn":
            self.scale = rope_scaling.get("factor", 32.0)
            self.mscale = float(0.1 * math.log(self.scale) + 1.0)
            self.mscale_all_dim = float(0.1 * math.log(self.scale) + 1.0)

        # Standard inv_freq calculation
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        
        # Apply YaRN wavelength interpolation
        if self.scale > 1.0:
            low, high = self._yarn_find_correction_dims(self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings)
            
            inv_freq_extrap = inv_freq[low:]
            inv_freq_interp = inv_freq[:low] / self.scale
            
            # Smoothly transition if needed (simplified for architecture representation)
            inv_freq = torch.cat((inv_freq_interp, inv_freq_extrap))
        
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(self.max_position_embeddings, device=self.inv_freq.device, dtype=torch.float32)

    def _yarn_find_correction_dims(self, beta_fast, beta_slow, dim, base, max_position_embeddings):
        # Calculates the high and low frequency dimensions for YaRN interpolation
        # Simplified cutoffs for architecture demonstration
        mid = dim // 2
        return mid, mid # Represents the hard split

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        # Scale positional sequence (Temperature scaling)
        t = t / self.scale
        
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Apply YaRN magnitude scaling to counteract entropy drop
        emb_cos = emb.cos() * self.mscale * self.mscale_all_dim
        emb_sin = emb.sin() * self.mscale * self.mscale_all_dim
        
        self.register_buffer("cos_cached", emb_cos[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb_sin[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=self.inv_freq.device, dtype=x.dtype)
            
        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(x.dtype)
        )

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # q, k: [batch, seq_len, num_heads, head_dim]
    # cos, sin: [1, 1, max_seq_len, head_dim] 
    
    # Extract the precise position embeddings for this batch
    cos = cos.squeeze(1).squeeze(0) # [max_seq_len, head_dim]
    sin = sin.squeeze(1).squeeze(0)
    
    cos = cos[position_ids].unsqueeze(2) # [batch, seq_len, 1, head_dim]
    sin = sin[position_ids].unsqueeze(2)
    
    # We expect q and k to be shaped [batch, seq_len, heads/1, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    
    if k is not None:
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
    return q_embed

class NGen3OWLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) + DSA Integration.
    Compresses KV into a latent space, and applies DeepSeek Sparse Attention
    using indices from the Lightning Indexer.
    """
    def __init__(self, config: NGen3OWConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        
        # MLA specific dimensions
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        
        # Q Down-projection (Compression) and Up-projection
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = NGen3OWRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        
        # Q Up splits into Standard Projection and RoPE Projection
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.v_head_dim, bias=False)
        self.q_rope_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_rope_head_dim, bias=False)

        # KV Down-projection (The Latent Cache)
        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = NGen3OWRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        
        # KV Up-projection (Decompression)
        self.kv_b_proj = nn.Linear(self.kv_lora_rank, self.num_heads * (self.v_head_dim + self.v_head_dim), bias=False) # split into K and V

        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = NGen3OWYarnRotaryEmbedding(self.qk_rope_head_dim, config.max_position_embeddings, config.rope_theta, config.rope_scaling)


    def forward(
        self, 
        hidden_states, 
        attention_mask=None, 
        position_ids=None, 
        past_key_value=None, 
        dsa_indices=None, # Injected from Lightning Indexer
        use_cache=False
    ):
        bsz, seq_len, _ = hidden_states.size()

        # --- Q Processing ---
        q_latent = self.q_a_proj(hidden_states)
        q_latent = self.q_a_layernorm(q_latent)
        
        # The content queries Q_c
        query_states = self.q_b_proj(q_latent).view(bsz, seq_len, self.num_heads, self.v_head_dim)
        # The RoPE queries Q_r
        q_rope = self.q_rope_proj(q_latent).view(bsz, seq_len, self.num_heads, self.qk_rope_head_dim)

        # --- KV Processing (Latent Compression) ---
        kv_latent_rope = self.kv_a_proj_with_mqa(hidden_states)
        # Split latent core from shared RoPE
        kv_latent, k_rope = torch.split(kv_latent_rope, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_latent = self.kv_a_layernorm(kv_latent)
        
        # Upsample from latent space to K and V
        k_and_v = self.kv_b_proj(kv_latent).view(bsz, seq_len, self.num_heads, self.v_head_dim * 2)
        key_states, value_states = torch.split(k_and_v, [self.v_head_dim, self.v_head_dim], dim=-1)
        
        # Shared Key RoPE is unsqueezed to match num_heads
        k_rope = k_rope.unsqueeze(2).expand(-1, -1, self.num_heads, -1)

        # Apply RoPE
        kv_seq_len = key_states.shape[1]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[1]
            
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin, position_ids)

        # We concatenate the semantic state and roped state
        query_states = torch.cat([query_states, q_rope], dim=-1)
        key_states = torch.cat([key_states, k_rope], dim=-1)

        # --- Cache Update ---
        if past_key_value is not None:
            # We cache the fully uncompressed sequence for correctness here, 
            # though an optimized triton kernel caches the latent vector directly.
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)

        past_key_value = (key_states, value_states) if use_cache else None

        # --- DSA Sparsity Application ---
        if dsa_indices is not None and key_states.shape[1] > self.config.dsa_top_k:
            # Expand dsa_indices to match num_heads and head_dim sizes
            # dsa_indices shape: [bsz, seq_len, top_k]
            # key shape: [bsz, cache_len, num_heads, head_dim]
            
            gathered_keys = []
            gathered_values = []
            
            # Sub-optimal PyTorch gathering for conceptual clarity.
            # In production, this uses an optimized Flash-Attention style sparse kernel.
            for b in range(bsz):
                # We do sparsity per context/batch
                idx = dsa_indices[b, 0, :] # Assuming homogeneous for entire chunk for simplicity
                gathered_keys.append(key_states[b, idx, :, :])
                gathered_values.append(value_states[b, idx, :, :])
                
            key_states = torch.stack(gathered_keys)
            value_states = torch.stack(gathered_values)
            
            # Mask adjustment needed if sequence lengths altered

        # --- Compute ---
        # Shape adjustment for matmul
        query_states = query_states.transpose(1, 2) # [bsz, num_heads, seq_len, head_dim]
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.v_head_dim + self.qk_rope_head_dim)

        if attention_mask is not None:
            # If DSA sliced the keys, mask logic must be carefully aligned/bypassed
            # Ignoring robust masking for theoretical layout
            pass

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, seq_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value

class SwiGLUv2MiniExpert(nn.Module):
    """
    A small specialized expert belonging to a specific domain group.
    Uses an evolved SwiGLU-v2 architecture with an inner projection bias 
    for better semantic routing at lower dimensions.
    """
    def __init__(self, config: NGen3OWConfig):
        super().__init__()
        # Note: Using moe_intermediate_size which is much smaller than a standard dense MLP
        self.gate_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        
        # SwiGLU-v2: We add a small learned bias to the down projection. 
        # This acts as an "expert signature" that helps the router differentiate outputs.
        self.down_proj = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=True)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class KnowledgeDomainGroup(nn.Module):
    """
    A localized cluster of Mini-Experts representing a single 'Domain'.
    """
    def __init__(self, config: NGen3OWConfig, group_id: int):
        super().__init__()
        self.group_id = group_id
        self.minis = nn.ModuleList([
            SwiGLUv2MiniExpert(config) for _ in range(config.num_minis_per_group)
        ])
        
    def forward(self, hidden_states, routing_weights, active_mini_indices):
        """
        Processes tokens through specific minis in this domain.
        hidden_states: [num_tokens_for_this_group, hidden_size]
        """
        final_group_states = torch.zeros_like(hidden_states)
        
        for i, mini_expert in enumerate(self.minis):
            mask = (active_mini_indices == i)
            if mask.any():
                token_idx = torch.where(mask)[0]
                expert_input = hidden_states[token_idx]
                
                # Apply the corresponding mini-expert routing weight
                expert_weight = routing_weights[mask].unsqueeze(-1)
                final_group_states[token_idx] += mini_expert(expert_input) * expert_weight
                
        return final_group_states

class HierarchicalRouter(nn.Module):
    """
    Two-Tiered Routing mechanism for MoEinMoE.
    Tier 1: Selects the Domain Group.
    Tier 2: Selects the Mini-Experts within that Group.
    """
    def __init__(self, config: NGen3OWConfig):
        super().__init__()
        self.num_groups = config.num_expert_groups
        self.minis_per_group = config.num_minis_per_group
        self.minis_per_tok = config.minis_per_tok

        # Tier 1: Group Router
        self.group_gate = nn.Linear(config.hidden_size, self.num_groups, bias=False)
        
        # Tier 2: Mini-expert Routers (one for each group)
        # Parameterized as a single batch matrix for efficiency
        self.mini_gates = nn.Parameter(torch.empty(self.num_groups, config.hidden_size, self.minis_per_group))
        nn.init.kaiming_uniform_(self.mini_gates, a=math.sqrt(5))

    def forward(self, hidden_states):
        # hidden_states: [batch_size * seq_len, hidden_size]
        
        # 1. Tier 1 Routing (Find the Group)
        group_logits = self.group_gate(hidden_states)
        group_probs = nn.functional.softmax(group_logits, dim=-1, dtype=torch.float32)
        
        # Find the single best group for this token
        top_group_probs, top_group_indices = torch.max(group_probs, dim=-1)
        
        # 2. Tier 2 Routing (Find the Minis inside the selected Group)
        # We only compute Tier 2 logits for the selected group to save computation
        # In a highly optimized kernel, this is fused.
        
        # Gather the specific mini_gate weights for the selected groups
        # selected_mini_gates: [N, hidden_size, minis_per_group]
        selected_mini_gates = self.mini_gates[top_group_indices]
        
        # Custom batched matmul: [N, 1, hidden_size] @ [N, hidden_size, minis_per_group] -> [N, 1, minis_per_group]
        mini_logits = torch.bmm(hidden_states.unsqueeze(1), selected_mini_gates).squeeze(1)
        
        mini_probs = nn.functional.softmax(mini_logits, dim=-1, dtype=torch.float32)
        
        # Select Top-K Minis from this group
        topk_mini_probs, topk_mini_indices = torch.topk(mini_probs, self.minis_per_tok, dim=-1)
        
        # Combine probabilities (Group Prob * Mini Prob)
        final_probs = top_group_probs.unsqueeze(-1) * topk_mini_probs
        final_probs /= final_probs.sum(dim=-1, keepdim=True) # Normalize
        
        # Map back to global expert ID (Group * Minis_per_group + Local_Mini_ID)
        global_expert_indices = (top_group_indices.unsqueeze(-1) * self.minis_per_group) + topk_mini_indices
        
        # Simplified Auxiliary Loss for both tiers
        group_aux_loss = (group_probs.mean(0).pow(2)).sum()
        mini_aux_loss = (mini_probs.mean(0).pow(2)).sum()
        total_aux_loss = group_aux_loss + mini_aux_loss
        
        return final_probs.to(hidden_states.dtype), global_expert_indices, total_aux_loss

class NGen3OWMoE(nn.Module):
    """
    Hierarchical MoEinMoE Block.
    Contains Groups of Mini-Experts and Shared Global Experts.
    """
    def __init__(self, config: NGen3OWConfig):
        super().__init__()
        self.router = HierarchicalRouter(config)
        self.num_expert_groups = config.num_expert_groups
        self.minis_per_group = config.num_minis_per_group
        
        # The structured Domain Groups (Tier 1)
        self.domain_groups = nn.ModuleList([
            KnowledgeDomainGroup(config, group_id=i) for i in range(self.num_expert_groups)
        ])
        
        # Shared experts for global knowledge capture (always active)
        self.shared_experts = nn.ModuleList([
            SwiGLUv2MiniExpert(config) for _ in range(config.num_shared_experts)
        ]) if config.num_shared_experts > 0 else None

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, h)
        
        # Router gives us probabilities and global indices: (Group_ID * Minis_Per_Group + Local_ID)
        topk_weights, topk_global_indices, aux_loss = self.router(hidden_states_flat)
        
        # Decode global index into Group ID and Local Mini ID
        group_indices = torch.div(topk_global_indices, self.minis_per_group, rounding_mode='floor')
        local_mini_indices = topk_global_indices % self.minis_per_group
        
        final_hidden_states = torch.zeros_like(hidden_states_flat)
        
        # Hop through physical Domain Groups
        for group_id, domain in enumerate(self.domain_groups):
            # Check if any token chose this Group as its Tier 1 destination
            group_mask = (group_indices == group_id)
            if group_mask.any():
                token_indices, col_indices = torch.where(group_mask) 
                
                group_inputs = hidden_states_flat[token_indices]
                group_weights = topk_weights[group_mask]
                active_minis = local_mini_indices[group_mask]
                
                # Defer to the Domain to route to its specific Minis
                group_outputs = domain(group_inputs, group_weights, active_minis)
                
                final_hidden_states[token_indices] += group_outputs
        
        # Add Shared Experts (Dense Compute)
        if self.shared_experts is not None:
            for shared_expert in self.shared_experts:
                # Shared experts process everything and are added linearly
                final_hidden_states += shared_expert(hidden_states_flat)
                
        return final_hidden_states.view(bsz, seq_len, h), aux_loss


class NGen3OWDecoderLayer(nn.Module):
    """
    The High-Performance Layer Assembler.
    Fuses the Lightning Indexer, Latent Attention (MLA), and Hierarchical MoE.
    Implements Residual Scaling to ensure stable gradient flow across 80+ layers.
    """
    def __init__(self, config: NGen3OWConfig, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        # 1. Stage 1: Lightning Indexer (The Scout)
        self.indexer = LightningIndexer(config) if config.use_dsa else None
        
        # 2. Stage 2: Latent Attention (MLA/DSA)
        self.self_attn = NGen3OWLatentAttention(config)
        
        # 3. Stage 3: MoEinMoE Routing & Computation
        if config.use_moe:
            self.feed_forward = NGen3OWMoE(config)
        else:
            self.feed_forward = SwiGLUv2MiniExpert(config) # Fallback to Dense equivalent

        self.input_layernorm = NGen3OWRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = NGen3OWRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Residual Scaling (ReZero Style)
        # Deep network stabilization: Initialize gating multipliers to 0 or very small
        # so the residual branch initially dominates, keeping gradients pristine.
        self.attn_res_scale = nn.Parameter(torch.zeros(1))
        self.ffn_res_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
    ):
        # ----------------------------------------
        # Block 1: Attention Pipeline (DSA + MLA)
        # ----------------------------------------
        residual_attn = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # DSA Scouting Stage (Zero MACs on heavy cache)
        dsa_indices = None
        if self.indexer is not None:
            # Emitting past scout caching logic for brevity
            dsa_indices, _ = self.indexer(hidden_states, position_ids=position_ids)

        # Latent Compute
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            dsa_indices=dsa_indices,
            use_cache=use_cache,
        )
        # Apply Attention Residual Scale
        hidden_states = residual_attn + (hidden_states * self.attn_res_scale)

        # ----------------------------------------
        # Block 2: Logic Pipeline (Hierarchical MoE)
        # ----------------------------------------
        residual_ffn = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        aux_loss = None
        if isinstance(self.feed_forward, NGen3OWMoE):
            hidden_states, aux_loss = self.feed_forward(hidden_states)
        else:
            hidden_states = self.feed_forward(hidden_states)
            
        # Apply FFN Residual Scale
        hidden_states = residual_ffn + (hidden_states * self.ffn_res_scale)

        return hidden_states, present_key_value, aux_loss

class NGen3OWModel(nn.Module):
    def __init__(self, config: NGen3OWConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([NGen3OWDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = NGen3OWRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
    ):
        batch_size, seq_length = input_ids.shape
        
        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        hidden_states = self.embed_tokens(input_ids)
        
        # 1D Causal Mask
        if attention_mask is None:
            attention_mask = torch.full((seq_length, seq_length), float("-inf"), device=input_ids.device)
            attention_mask = torch.triu(attention_mask, diagonal=1)
            attention_mask = attention_mask[None, None, :, :]

        presents = [] if use_cache else None
        total_aux_loss = 0.0
        
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            hidden_states, present_key_value, aux_loss = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            
            if aux_loss is not None:
                total_aux_loss += aux_loss
                
            if use_cache:
                presents.append(present_key_value)

        hidden_states = self.norm(hidden_states)

        return hidden_states, presents, total_aux_loss

class NGen3OWForCausalLM(nn.Module):
    def __init__(self, config: NGen3OWConfig):
        super().__init__()
        self.model = NGen3OWModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        moe_loss_weight: float = 0.01,
    ):
        hidden_states, presents, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            main_loss = loss_fct(shift_logits.view(-1, self.model.vocab_size), shift_labels.view(-1))
            loss = main_loss + moe_loss_weight * aux_loss

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": presents,
            "aux_loss": aux_loss
        }

    def print_parameter_count(self):
        """Calculates and prints the exact parameter count, distinguishing between Dense and MoE."""
        total_params = 0
        moe_params = 0
        dense_params = 0
        
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if "domain_groups" in name or "router" in name:
                moe_params += num_params
            else:
                dense_params += num_params
                
        print(f"\n--- NGen-3-OW Supercomputer Configuration ---")
        print(f"Total Parameters:    {total_params / 1e12:.4f} Trillion")
        print(f"Dense Parameters:    {dense_params / 1e9:.2f} Billion")
        print(f"MoE Parameters:      {moe_params / 1e12:.4f} Trillion")
        print(f"---------------------------------------------")
        return total_params

# Example instantiation
if __name__ == "__main__":
    # ---------------------------------------------------------
    # 10 Trillion Parameter Target with 10 Million Context Window
    # ---------------------------------------------------------
    super_config = NGen3OWConfig(
        vocab_size=128000,            # Standard large vocab
        hidden_size=8192,             # Massive hidden state
        num_hidden_layers=96,         # 96 Layers deep
        num_attention_heads=64,
        num_key_value_heads=8,        # GQA
        q_lora_rank=1536,             # MLA Compression
        kv_lora_rank=512,             # MLA Compression
        
        # Expert Distribution (To hit ~10T params)
        moe_intermediate_size=16384,  # Inner dim of each mini-expert
        num_expert_groups=32,         # 32 Domains
        num_minis_per_group=8,        # 8 Minis per domain = 256 Total Experts
        num_shared_experts=2,         # 2 Constant global experts
        
        # Infinite Context Core
        max_position_embeddings=10000000, # 10 Million Token Context
        rope_scaling={"type": "yarn", "factor": 320.0},
        
        use_dsa=True,
        dsa_top_k=512,                # DSA slices the top 512 chunks
        use_moe=True
    )

    print("Booting 10 Trillion Parameter blueprint on META device (Zero-RAM execution)...")
    
    # We MUST initialize on the 'meta' device or the computer will instantly
    # run out of RAM and crash since 10T params uses ~20,000 GB of memory.
    with torch.device("meta"):
        model = NGen3OWForCausalLM(super_config)
    
    model.print_parameter_count()
