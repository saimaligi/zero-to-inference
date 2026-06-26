import torch
import time
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
#print(device)

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

#print('models_loaded')

generation_config = dict(
    do_sample = True, #Sampling not greedy approach
    temperature = 1.0,
    top_k = 50,
    top_p = 1,
    max_new_tokens = 500,
    repetition_penalty=1.3,
)


#tokenizer to a prompt returns: dict with keys: input_ids, attention_mask

prompt = "Explain Machine Learning in simple terms"
input = tokenizer(prompt,return_tensors='pt').to(device) #return PyTorch tensors
prompt_ids = input['input_ids']
attention_mask = input['attention_mask']
current_ids = prompt_ids.clone()
prompt_len = prompt_ids.shape[1]

generated_tokens = []
max_new_tokens = 100
total_draft = 0
total_accepted = 0
k = 4 #Generating k token and verifying


#speculative decoding loop
while len(generated_tokens) < max_new_tokens:

    base_len = current_ids.shape[1]
    draft_token_probs = []
    draft_token_ids = []
    draft_probs = torch.zeros([k,draft_model.config.vocab_size],device=device)

    #1: Generate k tokens by draft_model
    for i in range(k): 
        with torch.no_grad():
            outputs = draft_model(current_ids).logits[:,-1,:]
            draft_probs[i] = F.softmax(outputs,dim=-1)
            probs = F.softmax(outputs,dim=-1)
            new_token = torch.multinomial(probs,num_samples=1)

            draft_token_ids.append(new_token.item())
            draft_token_probs.append(probs[0,new_token.item()].item())
            current_ids = torch.cat([current_ids,new_token],dim=-1)
            total_draft += 1
    
    #now get the logits from the target_model 
    #this is completely parallel not sequential
    #input : prompt+k_generated_tokens
    #output: prompt+k_generated_tokens+1

    #2: Get parallel logits from target model for the whole sequence
    with torch.no_grad():
        target_outputs = target_model(current_ids).logits
    
    curr_len = current_ids.shape[1]
    accepted = []
    all_accepted = True
    
    # 3. Verification loop
    for i in range(0,k):
        pos = base_len-1+i
        d_prob = draft_token_probs[i]
        t_probs = F.softmax(target_outputs[:,pos,:],dim=-1)
        t_prob = t_probs[0,draft_token_ids[i]].item()
        ratio = min(t_prob/d_prob,1)
        u = torch.rand(1).item() #picks from uniform dist [0,1)
        
        if u <= ratio:
            accepted.append(draft_token_ids[i])
            total_accepted += 1
        
        else:
            all_accepted=False
            resample_probs = torch.clamp(t_probs - draft_probs[i], min=0.0)
            if resample_probs.sum() == 0:
                # Fallback if distributions were perfectly disjoint (rare numerical edge case)
                resample_probs = t_probs
            else:
                resample_probs = resample_probs / resample_probs.sum()
            bonus_token = torch.multinomial(resample_probs, num_samples=1).item()
            accepted.append(bonus_token)
            break
    
    if all_accepted:
        bonus_probs = F.softmax(target_outputs[:,-1,:],dim=-1)
        bonus_token = torch.multinomial(bonus_probs, num_samples=1).item()
        accepted.append(bonus_token)
    

    # 4. Reconstruct history based on exactly what was accepted
    # Keep original tokens + accepted draft tokens
    current_ids = current_ids[:,:base_len]
    for token in accepted:
        token_tensor = torch.tensor([[token]]).to(device)
        current_ids = torch.cat([current_ids,token_tensor],dim=-1)
        generated_tokens.append(token)
        if token == tokenizer.eos_token_id:
            break

    if generated_tokens and generated_tokens[-1] == tokenizer.eos_token_id:
        break


final_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
print(final_text)
print(f"Acceptance rate : {total_accepted / total_draft:.2%}")
    







    








