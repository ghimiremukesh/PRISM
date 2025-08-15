from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def build_chat(messages, tokenizer):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback: simple concatenation
    s = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        s += f"<|{role}|>\n{content}\n"
    s += "<|assistant|>\n"
    return s

def load_chat_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe, tokenizer

def generate_chat(pipe, tokenizer, messages, max_new_tokens=3072, temperature=0.0):
    prompt = build_chat(messages, tokenizer)
    output = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True)
    gen = output[0]['generated_text'][len(prompt):].strip()
    return gen

def self_critique_chat(pipe, tokenizer, user_prompt, system_prompt=None):
    # Compose messages: add system prompt if given
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    
    # Assistant responds
    response = generate_chat(pipe, tokenizer, messages)
    messages.append({"role": "assistant", "content": response})

    # Self-critique step
    critique_prompt = "Please critique your previous response. Is the solution correct? Are there any logical flaws in the steps? Are there any calculations that need to be checked? Please provide a detailed analysis of your response. On top of your analysis also state a verdict 'Revision needed: 1' or 'Revision needed: 0'."
    messages.append({"role": "user", "content": critique_prompt})
    critique = generate_chat(pipe, tokenizer, messages)
    messages.append({"role": "assistant", "content": critique})

    # Decide on revision
    if "Revision needed: 0" in critique.lower():
        print("Final response:\n", response)
        print("\nSelf-critique:\n", critique)
        return response, critique
    else:
        revision_prompt = "Based on your critique, please revise your previous answer. Make sure the answer is in a box: \\boxed{Your Answer}. Please stop generation immediately after outputting the box."
        messages.append({"role": "user", "content": revision_prompt})
        revised = generate_chat(pipe, tokenizer, messages)
        print("Initial response:\n", response)
        print("\nSelf-critique:\n", critique)
        print("\nRevised response:\n", revised)
        return revised, critique

if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-3B"  # Or any chat-tuned model
    pipe, tokenizer = load_chat_model(model_name)
    user_input = "Jason is trying to remember the five digit combination to his safe.  He knows that he only used digits 1 through 5 (possibly repeated), that every even digit was followed by an odd digit, and every odd digit was followed by an even digit.  How many possible combinations does Jason need to try?"
    system_prompt = "You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process step by step and then provide the user with the answer. Please enclose your final answer in the box: \\boxed{Your Answer}. Please stop generation immediately after outputing the box."
    self_critique_chat(pipe, tokenizer, user_input, system_prompt)

    import ipdb
    ipdb.set_trace()