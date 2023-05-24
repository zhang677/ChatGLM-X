from xformers.components import build_attention
import xformers.ops as xops
import torch
import sys
sys.path.insert(1, '/home/nfs_data/zhanggh/ChatGLM-X/')
from model.origin_modeling_chatglm import attention_fn


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
            torch.ones(SEQ, SEQ).half().cuda() * float("-inf"),
            diagonal=1,
        )
  print(additive_mask)
  attention = build_attention(my_config)
  q = torch.arange(0, B * SEQ * MODEL).view(B, SEQ, HEADS, MODEL // HEADS).half().cuda() / 10
  q = q.transpose(1,2)
  k = q
  v = q
  # Input of xformers is (B * H, seq_len, model_dim // H) or (B, H, seq_len, model_dim // H)
  # Output: (B * H, seq_len, model_dim // H) or (B, H, seq_len, model_dim // H)
  output = attention(q, k, v).permute(2, 0, 1, 3).contiguous().view(SEQ, B, MODEL)
  print(output.shape)
  print("xformes output: ", output)


  attention_mask = torch.ones((B, SEQ, SEQ)).half().cuda()
  attention_mask.tril_()
  attention_mask.unsqueeze_(1)
  attention_mask = (attention_mask < 0.5).bool()
  print(attention_mask)
  class Temp:
    def __init__(self, **kwargs):
      self.scale_mask_softmax = False

  temp = Temp()
  q = torch.arange(0, B * SEQ * MODEL).view(B, SEQ, HEADS, MODEL // HEADS).half().cuda() / 10

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
  attention_mask = torch.ones((B, SEQ, SEQ)).half().cuda()
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

def check_memeff_attention_given():
  B = 2
  SEQ = 4
  MODEL = 16
  HEADS = 2
  q = torch.arange(0, B * SEQ * MODEL).view(SEQ, B, HEADS, MODEL // HEADS).half().cuda() / 10
  k = q
  v = q
  attention_mask = torch.ones((B, SEQ, SEQ)).half().cuda()
  attention_mask.tril_()
  attention_mask.unsqueeze_(1)
  attention_mask = (attention_mask < 0.5).bool()
  class Temp:
    def __init__(self, **kwargs):
      self.scale_mask_softmax = False

  temp = Temp()
  layer_id = 1
  chatglm_output = attention_fn(temp, q, k, v, attention_mask, MODEL, layer_id, scaling_attention_score=True)[0]

  key_layer = k.permute(1, 2, 0, 3).contiguous().view(B * HEADS, SEQ, MODEL // HEADS)
  query_layer = q.permute(1, 2, 0, 3).contiguous().view(B * HEADS, SEQ, MODEL // HEADS)
  value_layer = v.permute(1, 2, 0, 3).contiguous().view(B * HEADS, SEQ, MODEL // HEADS)
  # addtitive_mask = xops.LowerTriangularMask()
  addtitive_mask = torch.zeros_like(attention_mask.squeeze(1)).masked_fill_(attention_mask.squeeze(1), float('-inf'))
  # Repeat the mask for all heads
  addtitive_mask = addtitive_mask.repeat(HEADS, 1, 1).half()
  output = xops.memory_efficient_attention(
    query_layer, key_layer, value_layer,
    attn_bias=addtitive_mask
  ).view(B, HEADS, SEQ, MODEL // HEADS).permute(2, 0, 1, 3).contiguous().view(SEQ, B, MODEL)
  print(torch.allclose(output, chatglm_output))
  print(output)
  print(chatglm_output)


def check_memeff_attention():
  B = 1
  SEQ = 1024
  MODEL = 4096
  HEADS = 32
  # B = 2
  # SEQ = 4096
  # MODEL = 1024
  # HEADS = 8
  q = torch.rand((SEQ, B, HEADS, MODEL // HEADS)).half().cuda()
  k = torch.rand((SEQ, B, HEADS, MODEL // HEADS)).half().cuda()
  v = torch.rand((SEQ, B, HEADS, MODEL // HEADS)).half().cuda()
  attention_mask = torch.ones((B, SEQ, SEQ)).cuda()
  attention_mask.tril_()
  attention_mask.unsqueeze_(1)
  attention_mask = (attention_mask < 0.5).bool()
  # attention_mask = torch.zeros((B, 1, SEQ, SEQ)).bool().cuda()
  class Temp:
    def __init__(self, **kwargs):
      self.scale_mask_softmax = False

  temp = Temp()
  layer_id = 1
  q_in = q
  k_in = k
  v_in = v
  chatglm_output = attention_fn(temp, q_in, k_in, v_in, attention_mask, MODEL, layer_id, scaling_attention_score=True)[0]
  print(chatglm_output.shape)

  key_layer = k.transpose(0,1)
  query_layer = q.transpose(0,1)
  value_layer = v.transpose(0,1)
  addtitive_mask = torch.zeros(attention_mask.shape).half().cuda().masked_fill_(attention_mask, float('-inf'))
  
  addtitive_mask = addtitive_mask.expand(B, HEADS, SEQ, SEQ)#.view(B * HEADS, SEQ, SEQ)
  # addtitive_mask = torch.zeros((B, HEADS, SEQ, SEQ)).half().cuda()

  #torch.cuda.cudart().cudaProfilerStart()
  output = xops.memory_efficient_attention(
    query_layer, key_layer, value_layer, attn_bias=addtitive_mask, op = (xops.fmha.cutlass.FwOp, None)#
  )
  print(output.shape)
  output = output.transpose(0,1).view(SEQ, B, MODEL)
  # torch.cuda.cudart().cudaProfilerStop()
  # print all the positions that are not equal
  diff_ids = torch.nonzero(torch.logical_not(torch.isclose(output, chatglm_output)))
  # print the output indexed by diff_ids
  print(output[diff_ids[:,0], diff_ids[:,1], diff_ids[:,2]])
  print(chatglm_output[diff_ids[:,0], diff_ids[:,1], diff_ids[:,2]])  

if __name__ == "__main__":
  SEED = 0 
  torch.manual_seed(SEED) 
  torch.cuda.manual_seed(SEED) 
  #check_given()
  #check_scaled_dot_product()
  check_memeff_attention()
  #check_scaled_dot_product_with_mask()
  #check_given_mask()
  #check_memeff_attention_given()