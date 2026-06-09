import torch
import time
import torch.nn.functional as F
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
    max_new_tokens = 500,
    repetition_penalty=1.3,

)

print("target_model")
start = time.time()
with torch.no_grad():
    target_output = target_model.generate(input_ids,attention_mask,**generation_config)
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
    draft_output = draft_model.generate(input_ids,attention_mask,**generation_config)
end = time.time()
tokens_generated = draft_output.shape[1] - input_ids.shape[1]
draft_time = end - start
print(f"Tokens generated   : {tokens_generated}")
print(f"Time taken         : {target_time:.2f}s")
print(f"Speed              : {tokens_generated / draft_time:.2f} tokens/sec")

#speculative decoding loop
prompt     = "Explain ML in simple terms"
inputs     = tokenizer(prompt, return_tensors='pt').to(device)
prompt_ids = inputs['input_ids']
prompt_len = prompt_ids.shape[1]

max_new_tokens = 100
K              = 4
generated_tokens = []
total_draft    = 0
total_accepted = 0

current_ids = prompt_ids.clone()

while len(generated_tokens) < max_new_tokens:

    # Step 1 — draft generates K tokens
    draft_token_ids   = []
    draft_token_probs = []

    for _ in range(K):
        with torch.no_grad():
            logits = draft_model(current_ids).logits[:, -1, :]
        probs      = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        draft_token_ids.append(next_token)
        draft_token_probs.append(probs[0, next_token.item()])

        current_ids = torch.cat([current_ids, next_token], dim=-1)
        total_draft += 1

    # Step 2 — target verifies all K tokens in ONE forward pass
    # current_ids now = prompt + K draft tokens
    with torch.no_grad():
        target_logits = target_model(current_ids).logits
    # shape: (1, prompt_len + K, vocab_size)

    # target logits at draft positions
    # position prompt_len-1 predicts token at prompt_len (first draft token)
    # position prompt_len   predicts token at prompt_len+1 (second draft token)
    # etc.

    accepted     = []
    all_accepted = True
    
    for i in range(K):
        pos      = prompt_len - 1 + i
        t_probs  = F.softmax(target_logits[:, pos, :], dim=-1)
        t_prob   = t_probs[0, draft_token_ids[i].item()]
        d_prob   = draft_token_probs[i]

        ratio = t_prob / d_prob
        u     = torch.rand(1).item()

        if u <= ratio:                        # ACCEPT
            accepted.append(draft_token_ids[i])
            total_accepted += 1
        else:                                 # REJECT — resample from corrected distribution
            corrected = torch.clamp(t_probs - d_prob, min=0)
            corrected = corrected / corrected.sum()
            resampled = torch.multinomial(corrected, num_samples=1)
            accepted.append(resampled)
            all_accepted = False
            break

    # Step 3 — bonus token if all K accepted
    if all_accepted:
        bonus_probs = F.softmax(target_logits[:, prompt_len - 1 + K, :], dim=-1)
        bonus_token = torch.multinomial(bonus_probs, num_samples=1)
        accepted.append(bonus_token)

    # Step 4 — reset current_ids to prompt + accepted tokens only
    current_ids = prompt_ids.clone()
    for token in accepted:
        current_ids = torch.cat([current_ids, token.reshape(1, 1)], dim=-1)
        generated_tokens.append(token.item())
        if token.item() == tokenizer.eos_token_id:
            break

    if generated_tokens and generated_tokens[-1] == tokenizer.eos_token_id:
        break

final_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
print(final_text)
print(f"Acceptance rate : {total_accepted / total_draft:.2%}")