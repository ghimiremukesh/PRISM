from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import math
import re


def build_prompt(messages, tokenizer):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if prompt.endswith(f"{tokenizer.eos_token}\n"):
        prompt = prompt[:-len(f"{tokenizer.eos_token}\n")]
    elif prompt.endswith(tokenizer.eos_token):
        prompt = prompt[:-len(tokenizer.eos_token)]
    return prompt


def get_reward_score(out, tokenizer):
    '''calculate the reward score'''
    generated_text = out.text
    logprobs = out.logprobs
    tokens = out.token_ids
    token_logprobs = logprobs

    # find the position of Yes/No token
    boxed_match = re.search(r'(Yes|No)\}', generated_text, re.IGNORECASE)
    yes_token = tokenizer.encode('Yes')[-1]
    no_token = tokenizer.encode('No')[-1]

    if boxed_match:
        decision = boxed_match.group(1).capitalize()
        if decision == "Yes":
            yes_index = len(tokens) - 1 - tokens[::-1].index(yes_token)
            yes_logprob = token_logprobs[yes_index][yes_token].logprob
            # convert logprob to probability
            yes_prob = math.exp(yes_logprob)  # e^log(prob) = prob

            # find the position of 'No' token
            try:
                no_logprob = token_logprobs[yes_index][no_token].logprob
                no_prob = math.exp(no_logprob)
            except KeyError:
                # set 'No' probability to the minimum logprob of the remaining 4 logprobs
                min_logprob = min(v.logprob for k, v in token_logprobs[yes_index].items())
                no_prob = math.exp(min_logprob)

            # calculate softmax value
            softmax_denominator = yes_prob + no_prob
            if softmax_denominator == 0:
                softmax_yes = 0.5  # in case of division by zero, assign neutral score
            else:
                softmax_yes = yes_prob / softmax_denominator

            return softmax_yes

        elif decision == "No":
            no_index = len(tokens) - 1 - tokens[::-1].index(no_token)
            no_logprob = token_logprobs[no_index][no_token].logprob
            # convert logprob to probability
            no_prob = math.exp(no_logprob)  # e^log(prob) = prob

            # find the position of 'Yes' token
            try:
                yes_logprob = token_logprobs[no_index][yes_token].logprob
                yes_prob = math.exp(yes_logprob)
            except KeyError:
                # set 'Yes' probability to the minimum logprob of the remaining 4 logprobs
                min_logprob = min(v.logprob for k, v in token_logprobs[no_index].items())
                yes_prob = math.exp(min_logprob)

            # calculate softmax value
            softmax_denominator = yes_prob + no_prob
            if softmax_denominator == 0:
                softmax_yes = 0.5  # in case of division by zero, assign neutral score
            else:
                softmax_yes = yes_prob / softmax_denominator

            return softmax_yes
    else:
        # return neutral score if no decision found
        print("No boxed{Yes/No} found in the output")
        return 0.5


# Load model and tokenizer
model = LLM(model="GenPRM/GenPRM-7B")
tokenizer = AutoTokenizer.from_pretrained("GenPRM/GenPRM-7B")

# Configure sampling parameters
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    max_tokens=8192,
    top_k=20,
    repetition_penalty=1.0
)

# Define the messages
messages = [
    {'role': 'system', 'content': 'You are a math teacher. Your task is to review and critique the paragraphs in solution step by step.'},
    {'role': 'user', 'content': 'Question: Let $f(x)=x^2-7x+18$ and let $g(f(x))=2x+3$. What is the sum of all possible values of $g(8)$?\n\nTo solve the problem, we need to first understand the given functions and how they interact with each other. We are given $f(x) = x^2 - 7x + 18$ and $g(f(x)) = 2x + 3$.'},
    # {'role': 'assistant', 'content': ''},
    # {'role': 'user', 'content': 'Lets check if 124 is divisible by 4.'},
    # {'role': 'assistant', 'content': ''},
]

# Generate prompt and get the model's output
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = model.generate(prompt, sampling_params)

# outputs = []
# reward_list = []
# for i in range(len(messages)):
#     if messages[i]['role'] != 'assistant':
#         continue
#     prompt = build_prompt(messages[:i], tokenizer)
#     output = model.generate(prompt, sampling_params)
#     messages[i]['content'] = output[0].outputs[0].text
#     outputs.append(output[0].outputs[0].text)
#     reward = get_reward_score(output[0].outputs[0], tokenizer)
#     reward_list.append(reward)

# print(outputs)

# print(reward_list)

# Print result
# print(f"Model output for the first solution step: {outputs[0].outputs[0].text}")
import ipdb
ipdb.set_trace()




