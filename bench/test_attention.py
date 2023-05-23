from xformers.components import build_attention
import xformers.ops as xops
from model.modeling_chatglm import attention_fn
import torch

def check_given():
  B = 1
  SEQ = 2
  MODEL = 4
  HEADS = 2
  attention_name = "scaled_dot_product"
  my_config = {
      "name": attention_name, 
      "seq_len": SEQ,
      "model_dim": MODEL,
      "num_heads": HEADS,
      "dropout": 0.0,
      "causal": True # Create the causal mask in Attentionmask.make_causal (False means masked out)
  }

  additive_mask = torch.triu(
            torch.ones(SEQ, SEQ).float().cuda() * float("-inf"),
            diagonal=1,
        )
  print(additive_mask)
  attention = build_attention(my_config)
  q = torch.arange(0, B * SEQ * MODEL).view(B, SEQ, HEADS, MODEL // HEADS).float().cuda() / 10
  q = q.transpose(1,2)
  k = q
  v = q
  # Input of xformers is (B * H, seq_len, model_dim // H) or (B, H, seq_len, model_dim // H)
  # Output: (B * H, seq_len, model_dim // H) or (B, H, seq_len, model_dim // H)
  output = attention(q, k, v).permute(2, 0, 1, 3).contiguous().view(SEQ, B, MODEL)
  print(output.shape)
  print("xformes output: ", output)


  attention_mask = torch.ones((B, SEQ, SEQ)).cuda()
  attention_mask.tril_()
  attention_mask.unsqueeze_(1)
  attention_mask = (attention_mask < 0.5).bool()
  print(attention_mask)
  class Temp:
    def __init__(self, **kwargs):
      self.scale_mask_softmax = False

  temp = Temp()
  q = torch.arange(0, B * SEQ * MODEL).view(B, SEQ, HEADS, MODEL // HEADS).float().cuda() / 10

  q = q.transpose(0, 1).view(B * SEQ, HEADS, MODEL // HEADS)
  k = q
  v = q
  print(q)
  # Input of chatglm attention is (seq_len, B , H, model_dim // H)
  # Output: (seq_len, B , model_dim)
  chatglm_output = attention_fn(temp, q, k, v, attention_mask, MODEL, 0, scaling_attention_score=True)[0]
  print(chatglm_output.shape)
  print(chatglm_output)
  print("chatglm output: ", chatglm_output)
  print(torch.allclose(output, chatglm_output))

def check_chatglm_attention(B, SEQ, MODEL, q, k, v, output):
  attention_mask = torch.ones((B, SEQ, SEQ)).cuda()
  attention_mask.tril_()
  attention_mask.unsqueeze_(1)
  attention_mask = (attention_mask < 0.5).bool()
  class Temp:
    def __init__(self, **kwargs):
      self.scale_mask_softmax = False

  temp = Temp()
  layer_id = 1
  chatglm_output = attention_fn(temp, q, k, v, attention_mask, MODEL, layer_id, scaling_attention_score=True)[0]
  print(torch.allclose(output, chatglm_output))

def check_scaled_dot_product():
  B = 2
  SEQ = 1024
  MODEL = 4096
  HEADS = 32

  attention_name = "scaled_dot_product"
  my_config = {
      "name": attention_name,
      "seq_len": SEQ,
      "model_dim": MODEL,
      "num_heads": HEADS,
      "dropout": 0.0,
      "causal": True
  }

  attention = build_attention(my_config)
  q = torch.rand((B, HEADS, SEQ, MODEL // HEADS)).float().cuda()
  k = torch.rand((B, HEADS, SEQ, MODEL // HEADS)).float().cuda()
  v = torch.rand((B, HEADS, SEQ, MODEL // HEADS)).float().cuda()
  q_in = q.view(B * HEADS, SEQ, MODEL // HEADS)
  k_in = k.view(B * HEADS, SEQ, MODEL // HEADS)
  v_in = v.view(B * HEADS, SEQ, MODEL // HEADS)
  torch.cuda.cudart().cudaProfilerStart()
  output = attention(q_in, k_in, v_in).view(B, HEADS, SEQ, MODEL // HEADS).permute(2, 0, 1, 3).contiguous().view(SEQ, B, MODEL)
  torch.cuda.cudart().cudaProfilerStop()
  check_chatglm_attention(B, SEQ, MODEL, q.permute(2, 0, 1, 3), k.permute(2, 0, 1, 3), v.permute(2, 0, 1, 3), output)

def check_scaled_dot_product_with_mask():
  B = 2
  SEQ = 1024
  MODEL = 4096
  HEADS = 32

  q = torch.rand((SEQ, B, HEADS, MODEL // HEADS)).float().cuda()
  k = torch.rand((SEQ, B, HEADS, MODEL // HEADS)).float().cuda()
  v = torch.rand((SEQ, B, HEADS, MODEL // HEADS)).float().cuda()
  attention_mask = torch.ones((B, SEQ, SEQ)).cuda()
  attention_mask.tril_()
  attention_mask.unsqueeze_(1)
  attention_mask = (attention_mask < 0.5).bool()
  xformer_attention_mask = torch.zeros_like(attention_mask.squeeze(1)).masked_fill_(attention_mask.squeeze(1), float('-inf'))


  xformer_config = {
      "name": "scaled_dot_product",
      "model_dim": MODEL,
      "num_heads": HEADS,
      "dropout": 0.0,
  }
  xformer_attention_module = build_attention(xformer_config)
  class Temp:
    def __init__(self, **kwargs):
      self.scale_mask_softmax = False

  temp = Temp()
  layer_id = 1
  chatglm_output = attention_fn(temp, q, k, v, attention_mask, MODEL, layer_id, scaling_attention_score=True)[0]
  key_layer = k.permute(1, 2, 0, 3).contiguous().view(B * HEADS, SEQ, MODEL // HEADS)
  query_layer = q.permute(1, 2, 0, 3).contiguous().view(B * HEADS, SEQ, MODEL // HEADS)
  value_layer = v.permute(1, 2, 0, 3).contiguous().view(B * HEADS, SEQ, MODEL // HEADS)
  output = xformer_attention_module(query_layer, key_layer, value_layer, xformer_attention_mask).view(B, HEADS, SEQ, MODEL // HEADS).permute(2, 0, 1, 3).contiguous().view(SEQ, B, MODEL)
  print(torch.allclose(output, chatglm_output))

def check_given_mask():
  B = 1
  SEQ = 2
  MODEL = 4
  HEADS = 2
  q = torch.arange(0, B * SEQ * MODEL).view(SEQ, B, HEADS, MODEL // HEADS).half().cuda() / 10
  k = q
  v = q
  attention_mask = torch.ones((B, SEQ, SEQ)).cuda()
  attention_mask.tril_()
  attention_mask.unsqueeze_(1)
  attention_mask = (attention_mask < 0.5).bool()
  xformer_attention_mask_shape = (attention_mask.shape[0], attention_mask.shape[2], attention_mask.shape[3])
  xformer_attention_mask = torch.zeros(xformer_attention_mask_shape, dtype=k.dtype, device=k.device).masked_fill_(attention_mask.squeeze(1), float('-inf'))


  xformer_config = {
      "name": "scaled_dot_product",
      "model_dim": MODEL,
      "num_heads": HEADS,
      "dropout": 0.0,
  }
  xformer_attention_module = build_attention(xformer_config)
  class Temp:
    def __init__(self, **kwargs):
      self.scale_mask_softmax = False

  temp = Temp()
  layer_id = 1
  print(attention_mask)
  print(xformer_attention_mask)
  print(xformer_attention_mask.dtype)
  chatglm_output = attention_fn(temp, q, k, v, attention_mask, MODEL, layer_id, scaling_attention_score=True)[0]
  print(chatglm_output)
  key_layer = k.permute(1, 2, 0, 3).contiguous().view(B * HEADS, SEQ, MODEL // HEADS)
  query_layer = q.permute(1, 2, 0, 3).contiguous().view(B * HEADS, SEQ, MODEL // HEADS)
  value_layer = v.permute(1, 2, 0, 3).contiguous().view(B * HEADS, SEQ, MODEL // HEADS)
  output = xformer_attention_module(query_layer, key_layer, value_layer, xformer_attention_mask).view(B, HEADS, SEQ, MODEL // HEADS).permute(2, 0, 1, 3).contiguous().view(SEQ, B, MODEL)
  print(output)
  print(torch.allclose(output, chatglm_output))


def check_memeff_attention():
  B = 2
  SEQ = 1024
  MODEL = 4096
  HEADS = 32
  q = torch.rand((B, HEADS, SEQ, MODEL // HEADS)).float().cuda()
  k = torch.rand((B, HEADS, SEQ, MODEL // HEADS)).float().cuda()
  v = torch.rand((B, HEADS, SEQ, MODEL // HEADS)).float().cuda()
  q_in = q.view(B * HEADS, SEQ, MODEL // HEADS)
  k_in = k.view(B * HEADS, SEQ, MODEL // HEADS)
  v_in = v.view(B * HEADS, SEQ, MODEL // HEADS)
  addtitive_mask = xops.LowerTriangularMask()
  torch.cuda.cudart().cudaProfilerStart()
  output = xops.memory_efficient_attention(
    q_in, k_in, v_in,
    attn_bias=addtitive_mask
  ).view(B, HEADS, SEQ, MODEL // HEADS).permute(2, 0, 1, 3).contiguous().view(SEQ, B, MODEL)
  torch.cuda.cudart().cudaProfilerStop()
  check_chatglm_attention(B, SEQ, MODEL, q, k, v, output)

if __name__ == "__main__":
  SEED = 0 
  torch.manual_seed(SEED) 
  torch.cuda.manual_seed(SEED) 
  #check_given()
  #check_scaled_dot_product()
  #check_memeff_attention()
  #check_scaled_dot_product_with_mask()
  check_given_mask()