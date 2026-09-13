# Relevant vLLM entry points for the Humming MoE route

Source: `dev_probe/humming-ldmatrix-s4-moe-plan.md`, §2.

Traced route: vLLM's WNA16 MoE oracle selects a Humming experts
implementation → its weight post-processing calls
`convert_to_humming_moe_kernel_format` → Humming prepares/transforms w13 and
w2 weights and layer metadata → the experts implementation calls
`may_quant_input` and `forward_layer` for w13 → applies the model's
intermediate activation → separately calls `may_quant_input` and
`forward_layer` for w2 → vLLM combines routed expert outputs with top-k
weights.

Files (vLLM):
- `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py`
- `vllm/model_executor/layers/quantization/utils/humming_utils.py`
- `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py`

Files (Humming):
- `humming/layer.py`

Most feature code belongs in Humming's common packing/loading/WGMMA
machinery. vLLM-side changes should be limited to demonstrated
configuration, compatibility, or test gaps in this specific route.
