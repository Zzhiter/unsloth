# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .llama_seq import FastLlamaModelSequenceClassification, PeftModelForSequenceClassification_fast_forward
from .llama import *
from ._utils import __version__
from .gemma import (
    GemmaFixedRotaryEmbedding,
    GemmaFixedLinearScalingRotaryEmbedding,
    fast_geglu_inference,
)

from torch.nn import CrossEntropyLoss

try:
    from transformers.models.gemma2.modeling_gemma2 import (
        Gemma2Attention,
        Gemma2DecoderLayer,
        Gemma2Model,
        Gemma2ForCausalLM,
        Gemma2RotaryEmbedding,
        apply_rotary_pos_emb,
        repeat_kv,
        Gemma2ForSequenceClassification
    )
except:
    from packaging.version import Version
    transformers_version = Version(transformers_version)
    if not transformers_version >= Version("4.42"):
        raise ImportError(
            f"Unsloth: Your transformers version of {transformers_version} does not support Gemma2.\n"\
            f"The minimum required version is 4.42.3.\n"\
            f'Try `pip install --upgrade "transformers>=4.42.3"`\n'\
            f"to obtain the latest transformers build, then restart this session."\
        )
    pass
pass

from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.models.llama.modeling_llama import SequenceClassifierOutputWithPast

# For Pytorch 2.1.1
try:
    from transformers.models.gemma2.modeling_gemma2 import (
        Gemma2SdpaAttention,
        Gemma2FlashAttention2,
    )
except:
    Gemma2SdpaAttention   = Gemma2Attention
    Gemma2FlashAttention2 = Gemma2Attention
pass

if HAS_FLASH_ATTENTION_SOFTCAPPING:
    from flash_attn import flash_attn_func

# [TODO] We must randomnly use torch.compile?
# I checked the gradients and formulas and I'm sure it's correct.
# I'm stumped :(
@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def fast_rms_layernorm_gemma2_compiled(layernorm, X, gemma = True):
    old_dtype = X.dtype
    X = X.float()
    X = X * torch.rsqrt(X.square().mean(-1, keepdim = True) + layernorm.eps) * \
        (1.0 + layernorm.weight.float())
    return X.to(old_dtype)
pass


# Logit softcapping
def Gemma2Attention_fast_forward(
    self,
    hidden_states:        torch.Tensor,
    causal_mask:          Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
    attention_mask:       Optional[torch.Tensor] = None,
    position_ids:         Optional[torch.LongTensor] = None,
    past_key_value:       Optional[Tuple[torch.Tensor]] = None,
    output_attentions:    bool = False,
    use_cache:            bool = False,
    padding_mask:         Optional[torch.LongTensor] = None,
    *args, **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    # Clear inference
    if hasattr(self, "paged_attention"):
        del self.paged_attention_K
        del self.paged_attention_V
        del self.paged_attention
        del self.temp_QA
        del self.temp_KV
        del self.RH_Q
        del self.attention
    pass

    bsz, q_len, _ = hidden_states.size()

    n_heads    = self.num_heads
    n_groups   = self.num_key_value_groups
    n_kv_heads = self.num_key_value_heads
    head_dim   = self.head_dim
    assert(n_kv_heads * n_groups == n_heads)

    Q, K, V = self.apply_qkv(self, hidden_states)
    Q = Q.view(bsz, q_len, n_heads,    head_dim).transpose(1, 2)
    K = K.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    V = V.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    kv_seq_len = K.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if position_ids is None:
        cos = self.rotary_emb.cos_cached
        sin = self.rotary_emb.sin_cached
        Q, K = fast_rope_embedding(Q, K, cos, sin)
    else:
        cos, sin = self.rotary_emb(V, seq_len = kv_seq_len)
        Q, K = inplace_rope_embedding(Q, K, cos, sin, position_ids)
    pass

    if past_key_value is not None:
        K = torch.cat([past_key_value[0], K], dim = 2)
        V = torch.cat([past_key_value[1], V], dim = 2)
    pass
    past_key_value = (K, V) if use_cache else None

    # Only enable if the attention_mask is True
    has_sliding_window = type(causal_mask) is bool and causal_mask is True
    if HAS_FLASH_ATTENTION_SOFTCAPPING and attention_mask is None:
        window = (-1, -1)
        if has_sliding_window:
            sw = getattr(self.config, "sliding_window", None)
            sw = kv_seq_len if (sw is None or sw == "null") else sw
            window = (-1, -1) if (kv_seq_len <= sw) else (sw, sw)
        pass

        # FA uses 1 / sqrt for softmax_scale!
        if not hasattr(self, "_flash_attention_softmax_scale"):
            self._flash_attention_softmax_scale = 1.0 / (self.config.query_pre_attn_scalar**0.5)
        pass

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        A = flash_attn_func(
            Q, K, V,
            causal = True,
            softcap = self.config.attn_logit_softcapping,
            softmax_scale = self._flash_attention_softmax_scale,
            window_size = window,
        )
        A = A.reshape(bsz, q_len, n_heads*head_dim)
    else:
        fx = slow_inference_attention_softcapping \
            if "_flag_for_generation" in kwargs else \
            slow_attention_softcapping
        A = fx(Q, K, V, causal_mask, self, bsz, kv_seq_len)
    pass
    A = self.apply_o(self, A)
    return A, None, past_key_value
pass


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L590
def Gemma2DecoderLayer_fast_forward(
    self,
    hidden_states:        torch.Tensor,
    causal_mask:          Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
    attention_mask:       Optional[torch.Tensor] = None,
    position_ids:         Optional[torch.LongTensor] = None,
    past_key_value:       Optional[Tuple[torch.Tensor]] = None,
    output_attentions:    Optional[bool] = False,
    use_cache:            Optional[bool] = False,
    padding_mask:         Optional[torch.LongTensor] = None,
    *args, **kwargs,
):
    if use_cache and hasattr(self, "_flag_for_generation"): #past_key_value is not None:
        out_weight = torch.empty(self.input_layernorm.weight.shape, dtype = torch.float32, device = "cuda:0")

        # Self Attention
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference_gemma(self.input_layernorm, hidden_states, out_weight)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
            _flag_for_generation=True,
        )
        hidden_states = fast_rms_layernorm_inference_gemma(self.post_attention_layernorm, hidden_states, out_weight)
        hidden_states += residual

        # Fully Connected
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference_gemma(self. pre_feedforward_layernorm, hidden_states, out_weight)
        hidden_states = fast_geglu_inference(self.mlp, hidden_states)
        hidden_states = fast_rms_layernorm_inference_gemma(self.post_feedforward_layernorm, hidden_states, out_weight)
        hidden_states += residual
    else:
        residual = hidden_states
        hidden_states = fast_rms_layernorm_gemma2_compiled(self.input_layernorm, hidden_states, gemma = True)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        hidden_states = fast_rms_layernorm_gemma2_compiled(self.post_attention_layernorm, hidden_states, gemma = True)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = fast_rms_layernorm_gemma2_compiled(self. pre_feedforward_layernorm, hidden_states, gemma = True)
        hidden_states = self.mlp(hidden_states)
        hidden_states = fast_rms_layernorm_gemma2_compiled(self.post_feedforward_layernorm, hidden_states, gemma = True)
        hidden_states = residual + hidden_states
    pass

    outputs = (hidden_states,)
    if output_attentions: outputs += (self_attn_weights,)
    if use_cache: outputs += (present_key_value,)
    return outputs
pass


from math import sqrt as math_sqrt
KV_CACHE_INCREMENT = 256 # KV Cache update size
torch_nn_functional_softmax = torch.nn.functional.softmax
torch_matmul = torch.matmul
torch_tanh   = torch.tanh

def Gemma2Attention_fast_forward_inference(
    self,
    hidden_states:  torch.Tensor,
    past_key_value: Optional[Tuple[torch.Tensor]],
    position_ids,
    do_prefill = False,
    attention_mask = None,
    use_sliding_window = False,
):
    Xn = hidden_states
    bsz, _, hd = hidden_states.size()
    K1, V1 = past_key_value
    dtype = Xn.dtype

    n_heads    = self.num_heads
    n_groups   = self.num_key_value_groups
    n_kv_heads = self.num_key_value_heads
    head_dim   = self.head_dim
    attention_size = n_heads*head_dim
    # assert(n_kv_heads * n_groups == n_heads)
    seq_len = K1.shape[-2]
    kv_seq_len = seq_len + 1

    # Prefill phase
    # if not hasattr(self, "paged_attention"):
    if do_prefill:
        self.paged_attention = torch.empty((KV_CACHE_INCREMENT+seq_len+1, 2, bsz, n_kv_heads, head_dim), dtype = dtype, device = "cuda:0")
        self.paged_attention_K = self.paged_attention[:,0]
        self.paged_attention_V = self.paged_attention[:,1]
        self.paged_attention_K[:seq_len] = K1.permute(2, 0, 1, 3)
        self.paged_attention_V[:seq_len] = V1.permute(2, 0, 1, 3)
        self.temp_QA = torch.empty((2, bsz, 1, attention_size), dtype = dtype, device = "cuda:0")
        self.temp_KV = torch.empty((2, bsz, 1, n_kv_heads*head_dim), dtype = dtype, device = "cuda:0")
        self.RH_Q = torch.empty((bsz, n_heads, 1, head_dim), dtype = dtype, device = "cuda:0")
        # Only for Gemma2
        self.temp_O  = torch.empty((1, bsz, self.hidden_size), dtype = dtype, device = "cuda:0")
        self.attention = torch.empty((bsz, n_heads, 1, KV_CACHE_INCREMENT+seq_len), dtype = dtype, device = "cuda:0")
        
        # See https://github.com/google/gemma_pytorch/commit/03e657582d17cb5a8617ebf333c1c16f3694670e
        # Gemma 9b should use 256 and not 224 (hs / nah). 27b uses the below
        # We default to using the config file itself
        # s = self.config.hidden_size // self.config.num_attention_heads
        self.scalar = 1.0 / math_sqrt(self.config.query_pre_attn_scalar)
        # self.scalar = 1.0 / math_sqrt(self.config.hidden_size // self.config.num_attention_heads)
        self.half_head_dim = head_dim // 2
        self.           t =       self.config.attn_logit_softcapping
        self.reciprocal_t = 1.0 / self.config.attn_logit_softcapping
    elif kv_seq_len >= self.paged_attention.shape[0]:
        self.paged_attention.resize_((self.paged_attention.shape[0]+KV_CACHE_INCREMENT, 2, bsz, n_kv_heads, head_dim))
        self.paged_attention_K = self.paged_attention[:,0]
        self.paged_attention_V = self.paged_attention[:,1]
        self.attention.resize_((bsz, n_heads, 1, self.attention.shape[-1]+KV_CACHE_INCREMENT))
    pass

    Qn = fast_linear_forward(self.q_proj, Xn, out = self.temp_QA[0])
    Kn = fast_linear_forward(self.k_proj, Xn, out = self.temp_KV[0])
    Vn = fast_linear_forward(self.v_proj, Xn, out = self.temp_KV[1])
    Qn = Qn.view(bsz, 1, n_heads,    head_dim).transpose(1, 2)
    Kn = Kn.view(bsz, 1, n_kv_heads, head_dim).transpose(1, 2)
    Vn = Vn.view(bsz, 1, n_kv_heads, head_dim).transpose(1, 2)

    # cos, sin = self.rotary_emb(Vn, seq_len = kv_seq_len)
    # Qn, Kn = inplace_rope_embedding(Qn, Kn, cos, sin, position_ids)
    cos = self.rotary_emb.cos_cached[position_ids].unsqueeze(1)
    sin = self.rotary_emb.sin_cached[position_ids].unsqueeze(1)
    h = self.half_head_dim

    RH_Q = self.RH_Q
    RH_Q[:,:,:,:h] = Qn[:,:,:,h:]
    RH_Q[:,:,:,h:] = Qn[:,:,:,:h]
    torch.neg(RH_Q[:,:,:,:h], out = RH_Q[:,:,:,:h])
    Qn *= cos
    Qn.addcmul_(RH_Q, sin)

    RH_K = RH_Q[:,:n_kv_heads,:,:] # torch.empty((n_kv_heads, 1, head_dim), dtype = dtype, device = "cuda:0")
    RH_K[:,:,:,:h] = Kn[:,:,:,h:]
    RH_K[:,:,:,h:] = Kn[:,:,:,:h]
    torch.neg(RH_K[:,:,:,:h], out = RH_K[:,:,:,:h])
    Kn *= cos
    Kn.addcmul_(RH_K, sin)
    
    # New KV cache
    # Kn = torch.cat([K1, Kn], dim = 2)
    # Vn = torch.cat([V1, Vn], dim = 2)
    self.paged_attention_K[seq_len] = Kn.permute(2, 0, 1, 3)
    self.paged_attention_V[seq_len] = Vn.permute(2, 0, 1, 3)
    Kn = self.paged_attention_K[:kv_seq_len].permute(1, 2, 0, 3)
    Vn = self.paged_attention_V[:kv_seq_len].permute(1, 2, 0, 3)

    # Handle sliding windows
    sliding_window = self.config.sliding_window
    if use_sliding_window and kv_seq_len > sliding_window:
        # From https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L193
        slicing_tokens = 1 - sliding_window
        Knn = Kn[:, :, slicing_tokens:, :]#.contiguous()
        Vnn = Vn[:, :, slicing_tokens:, :]#.contiguous()
    else:
        Knn, Vnn = Kn, Vn
    pass

    # Grouped query attention
    _, _, cached_len, _ = Knn.shape
    if n_groups != 1:
        Knn = Knn[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, cached_len, head_dim)
        Vnn = Vnn[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, cached_len, head_dim)
        Knn = Knn.reshape(bsz, n_heads, cached_len, head_dim)
        Vnn = Vnn.reshape(bsz, n_heads, cached_len, head_dim)
    pass
    # else:
    #     Knn, Vnn = Knn, Vnn
    # pass

    # Attention
    # if bsz == 1:
    Qn *= self.scalar # See https://github.com/ggerganov/llama.cpp/issues/7805#issuecomment-2153349963
    # It seems like doing (Q * scalar) @ K is better than (Q @ K) * scalar to stop overflows
    A = torch_matmul(Qn, Knn.transpose(2, 3), out = self.attention[:,:,:,:cached_len])
    # if attention_mask is not None: A += attention_mask # Must add attention_mask for batched

    A *= self.reciprocal_t; torch_tanh(A, out = A); A *= self.t;  # Logit softcapping

    A[:] = torch_nn_functional_softmax(A, dim = -1, dtype = torch.float32)#.to(A.dtype)
    A = torch_matmul(A, Vnn, out = Qn)
    # else:
    #     A = scaled_dot_product_attention(Qn, Knn, Vnn, attn_mask = attention_mask, is_causal = False)
    # pass
    A = A.transpose(1, 2)
    A = A.reshape(bsz, 1, attention_size)
    A = fast_linear_forward(self.o_proj, A, out = self.temp_O)
    return A, (Kn, Vn)
pass


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L825
# @torch.inference_mode
def Gemma2Model_fast_forward_inference(
    self,
    input_ids,
    past_key_values,
    position_ids,
    attention_mask = None,
):
    out_weight = torch.empty_like(self.model.layers[0].input_layernorm.weight, dtype = torch.float32, device = "cuda:0")
    input_ids = input_ids[:,:self.max_seq_length]
    hidden_states = self.model.embed_tokens(input_ids)
    hidden_states = hidden_states.to(self.config.torch_dtype)
    # 3072**0.5 = 55.5000 in bfloat16, whilst 55.4256 in float32
    # 2048**0.5 = 45.2500 in bfloat16, whilst 45.2548 in float32
    hidden_states *= torch.tensor(math_sqrt(self.config.hidden_size), dtype = hidden_states.dtype)

    bsz, q_len, hd = hidden_states.shape
    seq_len = past_key_values[0][0].shape[-2]
    if bsz != 1:
        if HAS_FLASH_ATTENTION_SOFTCAPPING:
            SWA = True
            GA  = False
        else:
            SWA = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (bsz, q_len),
                hidden_states,
                seq_len,
                sliding_window = self.config.sliding_window,
            )
            GA = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (bsz, q_len),
                hidden_states,
                seq_len,
            )
        pass
    else:
        SWA = attention_mask
        GA  = attention_mask
    pass
    next_decoder_cache = []
    for idx, decoder_layer in enumerate(self.model.layers):

        use_sliding_window = idx % 2 == 0

        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference_gemma(decoder_layer.input_layernorm, hidden_states, out_weight)
        hidden_states, present_key_value = Gemma2Attention_fast_forward_inference(
            decoder_layer.self_attn,
            hidden_states = hidden_states,
            past_key_value = past_key_values[idx],
            position_ids = position_ids,
            attention_mask = SWA if use_sliding_window else GA,
            do_prefill = not hasattr(decoder_layer.self_attn, "paged_attention"),
            use_sliding_window = use_sliding_window,
        )
        hidden_states = fast_rms_layernorm_inference_gemma(decoder_layer.post_attention_layernorm, hidden_states, out_weight)
        hidden_states += residual

        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference_gemma(decoder_layer. pre_feedforward_layernorm, hidden_states, out_weight)
        hidden_states = fast_geglu_inference(decoder_layer.mlp, hidden_states)
        hidden_states = fast_rms_layernorm_inference_gemma(decoder_layer.post_feedforward_layernorm, hidden_states, out_weight)
        hidden_states += residual

        next_decoder_cache.append(present_key_value)
    pass
    hidden_states = fast_rms_layernorm_inference_gemma(self.model.norm, hidden_states, out_weight)

    return BaseModelOutputWithPast(
        last_hidden_state = hidden_states,
        past_key_values = next_decoder_cache,
        hidden_states = [],
        attentions = [],
    )
pass

def Sequence_fast_forward(fast_forward_inference):
    def _Sequence_fast_forward(
        self,
        input_ids: torch.LongTensor = None,
        causal_mask: Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        *args, **kwargs,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:

        if past_key_values is not None:
            outputs = fast_forward_inference(
                self,
                input_ids,
                past_key_values,
                position_ids = position_ids,
                attention_mask = attention_mask,
            )
        else:
            causal_mask = xformers.attn_bias.LowerTriangularMask()
    
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            self.model._has_no_labels = labels is None

            outputs = self.model(
                input_ids=input_ids,
                causal_mask=causal_mask,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        pass

        hidden_states = outputs[0]
        bsz, q_len, hd = hidden_states.shape
        score = self.score.weight
        if bsz == 1 and q_len == 1:
            logits = torch.mv(score, hidden_states.ravel().to(score.dtype))
            logits = logits.unsqueeze(0).unsqueeze(0)
        else:
            logits = self.score(hidden_states.to(score.dtype))
        pass
        logits = logits.to(self.config.torch_dtype)
        
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]
            
        if self.model.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.model.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.model.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            #only work on the single_label_classification as for now
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            
        pass

        if not return_dict:
            output = (pooled_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    pass
    return _Sequence_fast_forward
pass


class FastGemma2ModelSequenceClassification(FastLlamaModelSequenceClassification):

    @staticmethod
    def pre_patch():
        init_name, function = patch_linear_scaling(
            model_name         = "gemma2",
            rope_module        = GemmaFixedRotaryEmbedding,
            scaled_rope_module = GemmaFixedLinearScalingRotaryEmbedding,
            attention_module   = Gemma2Attention,
        )
        if init_name is not None:
            exec(function, globals())
            Gemma2Attention.__init__  = eval(init_name)
        pass
        Gemma2Attention      .forward = Gemma2Attention_fast_forward
        Gemma2SdpaAttention  .forward = Gemma2Attention_fast_forward
        Gemma2FlashAttention2.forward = Gemma2Attention_fast_forward
        Gemma2DecoderLayer   .forward = Gemma2DecoderLayer_fast_forward
        Gemma2Model          .forward = LlamaModel_fast_forward
        Gemma2ForSequenceClassification.forward = Sequence_fast_forward(Gemma2Model_fast_forward_inference)
        PeftModelForSequenceClassification.forward = PeftModelForSequenceClassification_fast_forward
        fix_prepare_inputs_for_generation(Gemma2ForCausalLM)
        
        # Solves https://github.com/unslothai/unsloth/issues/168
        # Static KV Cache was introduced in 4.38.0, causing training to be much slower.
        # Inferene can now be CUDAGraphed, but we shall retain the old rotary embeddings.
        # https://github.com/huggingface/transformers/pull/27931
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
        import transformers.models.gemma2.modeling_gemma2
        transformers.models.gemma2.modeling_gemma2.Gemma2RotaryEmbedding = GemmaFixedRotaryEmbedding
        return
    pass


    @staticmethod
    def post_patch(model):
        # Patch model for Gemma
        layers = model.model.layers

        # Torch.compile fails on embedding matrix??
        # Workaround randomnly fixes it for torch versions < 2.2
        model.model.embed_tokens = torch.nn.Embedding.from_pretrained(model.model.embed_tokens.weight)
        model.config.update({"unsloth_version" : __version__})

        # # We also do this for the score
        # score = torch.nn.Linear(1, 1, bias = None)
        # del score.weight
        # score.weight = model.score.weight
        # score.in_features  = score.weight.shape[1]
        # score.out_features = score.weight.shape[0]
        # model.score = score

        # # Gemma has tied weights! This means score == embed_tokens
        # if model.model.embed_tokens.weight.data_ptr() != model.score.weight.data_ptr():
        #     score = torch.nn.Linear(1, 1, bias = None)
        #     del score.weight
        #     score.weight = model.model.embed_tokens.weight
        #     score.in_features  = score.weight.shape[1]
        #     score.out_features = score.weight.shape[0]
        #     model.score = score
        # pass

        # Also patch all dtypes - BnB seems to not allocate the correct type?
        # BnB default dtype seems to be float16!
        # correct_dtype = score.weight.dtype

        # for name, module in model.named_modules():
        #     if isinstance(module, (Bnb_Linear4bit, Peft_Linear4bit)):
        #         weight = module.weight
        #         quant_state = weight.quant_state

        #         if type(quant_state) is list:
        #             # BnB seems to have float16 as default!
        #             module.weight.quant_state[2] = correct_dtype # Cast to correct dtype
        #         else:
        #             # https://github.com/TimDettmers/bitsandbytes/pull/763/files
        #             quant_state.dtype = correct_dtype
        #         pass
        #     pass
        #     # Downcast RoPE embedding to correct data type
        #     # RoPE must be done in float32 for Gemma
        #     # if (name.endswith("rotary_emb") or hasattr(module, "cos_cached")) \
        #     #     and (module.cos_cached.dtype != correct_dtype):

        #     #     module.cos_cached = module.cos_cached.to(correct_dtype)
        #     #     module.sin_cached = module.sin_cached.to(correct_dtype)
        #     #     pass
        #     # pass
        # pass

        # Add 1 to weight
        # return output * (1 + self.weight)
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma/modeling_gemma.py#L89
        from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm

        # Freeze all parameters except LoRA
        # We do this first since += 1 seems to not be liked by requires_grad = True
        for name, param in model.named_parameters():
            if ".lora_A." in name or ".lora_B." in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        pass

        # Patch RMS Layernorm
        for name, module in model.named_modules():
            if isinstance(module, Gemma2RMSNorm):
                # Must be in float32
                # https://github.com/keras-team/keras-nlp/blob/v0.8.2/keras_nlp/models/gemma/rms_normalization.py#L36
                # module = module.to(torch.float32)
                # Leave + 1 to Triton kernel itself
                # module.weight += 1.0 # return output * (1 + self.weight)
                if not hasattr(module, "variance_epsilon"):
                    module.variance_epsilon = module.eps # Gemma doesn't use variance_epsilon
        pass

        # Clear deleted GPU items
        import gc
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        return model
    pass
pass
