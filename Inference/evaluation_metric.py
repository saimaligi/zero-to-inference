"""
    perplexity_score 
    speed before and after speculative decoding
"""

import torch
import time
device = "cuda" if torch.cuda.is_available() else "cpu"

def perplexity_score(model,input_ids):
    with torch.no_grad():
        outputs = model(input_ids=input_ids,labels=input_ids)
        perplexity_score = torch.exp(outputs.loss.item())
    print(f'{perplexity_score=}')



def generation_speed(model, tokenizer, prompt):
    input = tokenizer(prompt,return_tensors='pt').to(device)
    input_ids = input['input_ids']
    attention_mask = input['attention_mask']
    prompt_len = input_ids.shape[1]

    generate_config = dict(
        max_new_tokens = 200,
        do_sample = True,
        top_k = 50,
        top_p = 0.9,
        temperature = 1.0,
        repetition_penalty=1.3,
        pad_token_id       = tokenizer.eos_token_id 
    )
    
    start  = time.time()
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        **generate_config
    )
    end    = time.time()

    generated_ids  = output[0][prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tokens_generated = generated_ids.shape[0]
    time_taken       = end - start
    speed            = tokens_generated / time_taken

    print(f"Generated text       : {generated_text}")
    print(f"Tokens generated     : {tokens_generated}")
    print(f"Time taken           : {time_taken:.2f}s")
    print(f"Speed                : {speed:.2f} tokens/sec")
    
    return output[0]






