import torch
import torch.nn.functional as F
import argparse
import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers import BertModel, BertTokenizer
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from datasets import load_dataset

def run_bert():
  def generate_word(tokenizer, model, prompt: str, prompt_tokens: int = 2048):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids[:, :prompt_tokens].to(model.device)
    with torch.no_grad():
      model(inputs)
    input_sentence = tokenizer.batch_decode(inputs, skip_special_tokens=True)[0]
    return input_sentence
  
  model_name = 'bert-base-uncased'
  tokenizer = BertTokenizer.from_pretrained(model_name)
  # 载入模型
  model = BertModel.from_pretrained(model_name)
  prompt_tokens = 1024
  dataset = load_dataset('lambada', split='train[:1]')
  test_sentence = dataset[0]['text']
  print(len(test_sentence))
  input_sentence = generate_word(tokenizer, model, test_sentence, prompt_tokens = prompt_tokens)
  print(input_sentence)

def run_llama():
# https://github.com/huggingface/transformers/issues/22222
  def generate_sentence(tokenizer, model, prompt: str, prompt_tokens: int = 64, max_tokens: int = 128, num_return_sequences: int = 3, num_beams: int = 4):
    # generate
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids[:, :prompt_tokens].to(model.device)
    max_new_tokens = max_tokens - inputs.shape[1] - 1
    print("Input length: {}".format(inputs.shape[1]))
    print("Max new tokens: {}".format(max_new_tokens))
    assert max_new_tokens > 0, "max_tokens should be larger than the length of the prompt"

    output_ids = model.generate(inputs, do_sample=True, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences, temperature=0.7)

    input_sentence = tokenizer.batch_decode(inputs, skip_special_tokens=True)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return input_sentence, outputs

  tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf", padding_side="right", model_max_length=2048)
  model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf", torch_dtype=torch.float16, device_map='auto')

  prompt_tokens = 1024
  dataset = load_dataset('lambada', split='train[:1]')
  test_sentence = dataset[0]['text']
  print(len(test_sentence))
  input_sentence, output = generate_sentence(tokenizer, model, test_sentence, prompt_tokens = prompt_tokens, max_tokens=2048, num_return_sequences=1)
  print(input_sentence)
  print("=====================================")
  for i, sentence in enumerate(output):
      print(i)
      print("=====================================")
      print(sentence)

def run_chatglm():
  def generate_sentence(tokenizer, model, prompt: str, prompt_tokens: int = 64, max_tokens: int = 128, num_return_sequences: int = 3, num_beams: int = 4):
    # generate
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids[:, :prompt_tokens].to(model.device)
    max_new_tokens = max_tokens - inputs.shape[1] - 1
    print("Input length: {}".format(inputs.shape[1]))
    print("Max new tokens: {}".format(max_new_tokens))
    assert max_new_tokens > 0, "max_tokens should be larger than the length of the prompt"
    input_sentence = tokenizer.batch_decode(inputs, skip_special_tokens=True)[0]
    outputs, _ = model.chat(tokenizer, input_sentence, history=[], do_sample=True, max_new_tokens=max_new_tokens)
    
    return input_sentence, outputs
  
  tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
  model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).quantize(4).half().cuda()
  prompt_tokens = 1024
  dataset = load_dataset('lambada', split='train[:1]')
  test_sentence = dataset[0]['text']
  print(len(test_sentence))
  input_sentence, output = generate_sentence(tokenizer, model, test_sentence, prompt_tokens = prompt_tokens, max_tokens=2048, num_return_sequences=1)
  print(input_sentence)
  print("=====================================")
  print(output)

def run_opt67():
  def generate_sentence(tokenizer, model, prompt: str, prompt_tokens: int = 64, max_tokens: int = 128, num_return_sequences: int = 3, num_beams: int = 4):
    # generate
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids[:, :prompt_tokens].to(model.device)
    max_new_tokens = max_tokens - inputs.shape[1] - 1
    print("Input length: {}".format(inputs.shape[1]))
    print("Max new tokens: {}".format(max_new_tokens))
    assert max_new_tokens > 0, "max_tokens should be larger than the length of the prompt"

    output_ids = model.generate(inputs, do_sample=True, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences, temperature=0.7)

    input_sentence = tokenizer.batch_decode(inputs, skip_special_tokens=True)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return input_sentence, outputs
  model_name = 'opt-6.7B'
  tokenizer = AutoTokenizer.from_pretrained('facebook/{}'.format(model_name), padding_side="right", model_max_length=2048)
  model = OPTForCausalLM.from_pretrained('facebook/{}'.format(model_name), torch_dtype=torch.float16, device_map='auto')

  prompt_tokens = 1024
  dataset = load_dataset('lambada', split='train[:1]')
  test_sentence = dataset[0]['text']
  print(len(test_sentence))
  input_sentence, output = generate_sentence(tokenizer, model, test_sentence, prompt_tokens = prompt_tokens, max_tokens=2048, num_return_sequences=1)
  print(input_sentence)
  print("=====================================")
  for i, sentence in enumerate(output):
      print(i)
      print("=====================================")
      print(sentence)
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, default="bert")
  args = parser.parse_args()
  if args.model == "bert":
    run_bert()
  elif args.model == "llama":
    run_llama()
  elif args.model == "opt-6.7b":
    run_opt67()
  elif args.model == "chatglm":
    run_chatglm()