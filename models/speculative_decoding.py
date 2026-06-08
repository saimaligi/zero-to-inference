import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

#Tokenizer is same for two models
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

#target model
target_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-7b-it",
    device_map="auto",
    dtype=torch.float16,
    revision="float16",
)

#draft_model
draft_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    device_map="auto",
    dtype=torch.float16,
    revision="float16",
)

prompt = "Write me a poem about Machine Learning."
input = tokenizer(prompt, return_tensors="pt").to(device)

#input_ids is a dict with keys: input_ids, attention_mask
input_ids = input['input_ids']
attention_mask = input['attention_mask']

generation_config = dict(
    attention_mask = attention_mask,
    do_sample = True, #Sampling not greedy 
    temperature = 1.0,
    top_k = 50,
    top_p = 1,
    max_new_tokens = 100,
    repetition_penalty=1.3,

)

print("target_model")
start = time.time()
with torch.no_grad():
    target_output = target_model.generate(input_ids,**generation_config)
end = time.time()
tokens_generated = target_output.shape[1] - input_ids.shape[1]
target_time = end - start
print(f"Tokens generated   : {tokens_generated}")
print(f"Time taken         : {target_time:.2f}s")
print(f"Speed              : {tokens_generated / target_time:.2f} tokens/sec")

print('--'*30)

print('draft_model')
start = time.time()
with torch.no_grad():
    draft_output = draft_model.generate(input_ids,**generation_config)
end = time.time()
tokens_generated = draft_output.shape[1] - input_ids.shape[1]
draft_time = end - start
print(f"Tokens generated   : {tokens_generated}")
print(f"Time taken         : {target_time:.2f}s")
print(f"Speed              : {tokens_generated / draft_time:.2f} tokens/sec")

