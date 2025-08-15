import re
import math
import io
import signal
from copy import *
from GenPRM.util import *
from contextlib import redirect_stdout
from transformers import AutoTokenizer, AutoModelForCausalLM
# from vllm import LLM, SamplingParams
from torch.nn.parallel import DataParallel, DistributedDataParallel

# import ipdb

TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
REPETITION_PENALTY = 1.0
version = 'v1.0'


def build_prompt(messages, tokenizer):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if prompt.endswith(f"{tokenizer.eos_token}\n"):
        prompt = prompt[:-len(f"{tokenizer.eos_token}\n")]
    elif prompt.endswith(tokenizer.eos_token):
        prompt = prompt[:-len(tokenizer.eos_token)]
    return prompt


class timeout:
    """timeout context manager"""

    def __init__(self, seconds=1):
        self.seconds = seconds

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)

    def handle_timeout(self, signum, frame):
        raise TimeoutError("Code execution timed out")


class CodeExecutor:
    """code executor"""

    def __init__(self):
        self.namespace = {}  # indicate the global namespace for exec
        self.code_pattern = re.compile(r'```python\s*(.*?)\s*```', re.DOTALL)

    def execute(self, text):
        # extract code block
        try:
            code_block = self.code_pattern.findall(text)[-1].strip()
        except Exception as e:
            actual = f"Code format error: No code found."
            return actual

        # execute code block
        try:
            f = io.StringIO()
            with redirect_stdout(f):
                with timeout(seconds=5):
                    exec(code_block, self.namespace)
            actual = f.getvalue().strip()
        except TimeoutError as te:
            actual = f"Code execute time out: {te}"
            print(actual)
        except Exception as e:
            actual = f"Code execute Error: {type(e).__name__}: {e}"
            print(actual)

        return actual


class GenPRM:
    def __init__(self, vllm_client, model_path, tensor_parallel_size=4):
        # Load the model and tokenizer
        timestamped_print(f"Loading model from {model_path}", level="INFO")
        # self.model = LLM(
        #     model=model_path,
        #     tensor_parallel_size=tensor_parallel_size,
        #     enable_chunked_prefill=True
        # )
        self.model = vllm_client
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        timestamped_print(f"GenPRM loaded successfully", level="INFO")

    def get_reward_score(self, out):
        '''calculate the reward score'''
        generated_text = out.text
        logprobs = out.logprobs
        tokens = out.token_ids
        token_logprobs = logprobs

        # find the position of Yes/No token
        boxed_match = re.search(r'(Yes|No)\}', generated_text, re.IGNORECASE)
        yes_token = self.tokenizer.encode('Yes')[-1]
        no_token = self.tokenizer.encode('No')[-1]

        if boxed_match:
            decision = boxed_match.group(1).capitalize()
            if decision == "Yes":
                try:
                    yes_index = len(tokens) - 1 - tokens[::-1].index(yes_token)
                    yes_logprob = token_logprobs[yes_index][str(yes_token)]["logprob"]
                    # convert logprob to probability
                    yes_prob = math.exp(yes_logprob)  # e^log(prob) = prob
                except ValueError:
                    yes_prob = 0.5
                    yes_index = -1

                # find the position of 'No' token
                try:
                    no_logprob = token_logprobs[yes_index][str(no_token)]["logprob"]
                    no_prob = math.exp(no_logprob)
                except KeyError:
                    # set 'No' probability to the minimum logprob of the remaining 4 logprobs
                    min_logprob = min(v["logprob"] for k, v in token_logprobs[yes_index].items())
                    no_prob = math.exp(min_logprob)

                # calculate softmax value
                softmax_denominator = yes_prob + no_prob
                if softmax_denominator == 0:
                    softmax_yes = 0.5  # in case of division by zero, assign neutral score
                else:
                    softmax_yes = yes_prob / softmax_denominator

                return softmax_yes

            elif decision == "No":
                try:
                    no_index = len(tokens) - 1 - tokens[::-1].index(no_token)
                    no_logprob = token_logprobs[no_index][str(no_token)]["logprob"]
                    # convert logprob to probability
                    no_prob = math.exp(no_logprob)  # e^log(prob) = prob
                except ValueError:
                        no_prob = 0.5
                        no_index = -1

                # find the position of 'Yes' token
                try:
                    yes_logprob = token_logprobs[no_index][str(yes_token)]["logprob"]
                    yes_prob = math.exp(yes_logprob)
                except KeyError:
                    # set 'Yes' probability to the minimum logprob of the remaining 4 logprobs
                    min_logprob = min(v["logprob"] for k, v in token_logprobs[no_index].items())
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
            timestamped_print("No boxed{Yes/No} found in the output", level="WARNING")
            return 0.5
        
    def get_reward_score_all(self, out):
        '''calculate the reward score'''
        generated_text = out.text
        logprobs = out.logprobs
        tokens = out.token_ids
        token_logprobs = logprobs

        actual_out_text = generated_text[-len(self.tokenizer.decode(tokens)):]  # get the actual output text from the generated text


        # find all positions of Yes/No tokens in boxed format
        boxed_matches = list(re.finditer(r'(Yes|No)\}', actual_out_text, re.IGNORECASE))
        yes_token = self.tokenizer.encode('Yes')[-1]
        no_token = self.tokenizer.encode('No')[-1]

        # ipdb.set_trace()
        if not boxed_matches:
            # return neutral score if no decision found
            timestamped_print("No boxed{Yes/No} found in the output", level="WARNING")
            return 0.5
        
        decisions = [match.group(1).capitalize() for match in boxed_matches]  # extract decisions
        token_idxs = [i for i in range(len(tokens)) if tokens[i] in (yes_token, no_token)]  # find indices of Yes/No tokens in the token list



        # collect scores for all decisions
        scores = []
        
        # for match in boxed_matches:
        for idx, decision in zip(token_idxs, decisions):
            # decision = match.group(1).capitalize()
            
            if decision == "Yes":
                yes_index = idx
                # yes_index = len(tokens) - 1 - tokens[::-1].index(yes_token)
                yes_logprob = token_logprobs[yes_index][str(yes_token)]["logprob"]
                # convert logprob to probability
                yes_prob = math.exp(yes_logprob)  # e^log(prob) = prob

                # find the position of 'No' token
                try:
                    no_logprob = token_logprobs[yes_index][str(no_token)]["logprob"]
                    no_prob = math.exp(no_logprob)
                except KeyError:
                    # set 'No' probability to the minimum logprob of the remaining 4 logprobs
                    min_logprob = min(v["logprob"] for k, v in token_logprobs[yes_index].items())
                    no_prob = math.exp(min_logprob)

                # calculate softmax value
                softmax_denominator = yes_prob + no_prob
                if softmax_denominator == 0:
                    softmax_yes = 0.5  # in case of division by zero, assign neutral score
                else:
                    softmax_yes = yes_prob / softmax_denominator

                scores.append(softmax_yes)

            elif decision == "No":
                no_index = idx
                # no_index = len(tokens) - 1 - tokens[::-1].index(no_token)
                no_logprob = token_logprobs[no_index][str(no_token)]["logprob"]
                # convert logprob to probability
                no_prob = math.exp(no_logprob)  # e^log(prob) = prob

                # find the position of 'Yes' token
                try:
                    yes_logprob = token_logprobs[no_index][str(yes_token)]["logprob"]
                    yes_prob = math.exp(yes_logprob)
                except KeyError:
                    # set 'Yes' probability to the minimum logprob of the remaining 4 logprobs
                    min_logprob = min(v["logprob"] for k, v in token_logprobs[no_index].items())
                    yes_prob = math.exp(min_logprob)

                # calculate softmax value
                softmax_denominator = yes_prob + no_prob
                if softmax_denominator == 0:
                    softmax_yes = 0.5  # in case of division by zero, assign neutral score
                else:
                    softmax_yes = yes_prob / softmax_denominator

                scores.append(softmax_yes)

        # calculate and return the average score
        if scores:
            average_score = sum(scores) / len(scores)
            # min_score = min(scores)
            return average_score
        else:
            return 0.5  # fallback to neutral score
        
    # def inference(
    #     self,
    #     messages,
    #     majority_num=1,
    #     cur_step=1,
    #     analyze=True,
    #     verify=True,
    #     execute=True,
    #     time_limit=3,
    #     max_tokens=2048,
    #     code_executor=None,
    #     analyze_template="<analyze>\nLet's analyze the Paragraph {cur_step} step by step: ",
    #     verify_template="<verify>\nLet's use python code to find any potential error:\n```python\n",
    #     output_template="<output>\n**Judgement**: $\\boxed",
    #     logging=True
    # ):
    #     '''
    #     messages: the input messages
    #     majority_num: the number of majority votes
    #     cur_step: the current step index (start from 1)
    #     analyze: whether to analyze the input
    #     verify: whether to verify the input
    #     execute: whether to execute the code
    #     time_limit: the time limit for code execution
    #     max_tokens: the maximum tokens for the output
    #     analyze_template: the template for analyze start
    #     verify_template: the template for verify start
    #     output_template: the template for output start
    #     logging: whether to log the process
    #     '''
    #     output_paths = []
    #     reward_scores = []
    #     for i in range(majority_num):
    #         # perform inference
    #         output_path, reward_score = self._single_inference(
    #             messages,
    #             cur_step=cur_step,
    #             analyze=analyze,
    #             verify=verify,
    #             execute=execute,
    #             time_limit=time_limit,
    #             max_tokens=max_tokens,
    #             code_executor=code_executor,
    #             analyze_template=analyze_template,
    #             verify_template=verify_template,
    #             output_template=output_template,
    #             logging=logging
    #         )

    #         output_paths.append(output_path)
    #         reward_scores.append(reward_score)

    #     return output_paths, sum(reward_scores) / len(reward_scores)

    def inference(
        self,
        messages_list,
        majority_num=1,
        cur_step=1,
        analyze=True,
        verify=True,
        execute=True,
        time_limit=3,
        max_tokens=2048,
        code_executor=None,
        analyze_template="<analyze>\nLet's analyze all the Paragraphs step by step:",
        verify_template="<verify>\nLet's use python code to find any potential error:\n```python\n",
        output_template="<output>Let's provide judgements for each of the paragraphs.\nParagraph 1 **Judgement**: $\\boxed",
        logging=True
    ):
        '''
        Batch inference for multiple different inputs with majority voting support.
        
        Args:
            messages_list: List of different message inputs to process
            majority_num: Number of majority votes per input
            Other parameters are the same as inference method
        
        Returns:
            List of tuples, each containing (output_paths, average_reward_score) for each input
        '''
        
        if majority_num == 1:
            # Direct batch processing without majority voting
            results = self._batch_inference(
                messages_list=messages_list,
                cur_step=cur_step,
                analyze=analyze,
                verify=verify,
                execute=execute,
                time_limit=time_limit,
                max_tokens=max_tokens,
                code_executor=code_executor,
                analyze_template=analyze_template,
                verify_template=verify_template,
                output_template=output_template,
                logging=logging
            )
            
            # Format results to match expected output format
            final_outputs = []
            final_rewards = []
            for output_text, reward_score in results:
                final_outputs.append([output_text])
                final_rewards.append(reward_score)
                # formatted_results.append(([output_text], reward_score))
            
            return final_outputs, final_rewards
        
        else:
            # Batch processing with majority voting
            # Create expanded list with each input repeated majority_num times
            expanded_messages_list = []
            input_indices = []  # Track which original input each item belongs to
            
            for i, messages in enumerate(messages_list):
                for j in range(majority_num):
                    expanded_messages_list.append(messages)
                    input_indices.append(i)
            
            # Run batch inference on all expanded inputs
            all_results = self._batch_inference(
                messages_list=expanded_messages_list,
                cur_step=cur_step,
                analyze=analyze,
                verify=verify,
                execute=execute,
                time_limit=time_limit,
                max_tokens=max_tokens,
                code_executor=code_executor,
                analyze_template=analyze_template,
                verify_template=verify_template,
                output_template=output_template,
                logging=logging
            )
            
            # Group results by original input
            grouped_results = [[] for _ in range(len(messages_list))]
            for idx, (output_text, reward_score) in enumerate(all_results):
                original_input_idx = input_indices[idx]
                grouped_results[original_input_idx].append((output_text, reward_score))
            
            # Format final results with average scores
            final_outputs = []
            final_rewards = []
            for group in grouped_results:
                output_paths = [result[0] for result in group]
                reward_scores = [result[1] for result in group]
                avg_score = sum(reward_scores) / len(reward_scores)
                # final_results.append((output_paths, avg_score))
                final_outputs.append(output_paths)
                final_rewards.append(avg_score)
            
            return final_outputs, final_rewards

    def _single_inference(
        self,
        messages,
        cur_step=1,
        analyze=True,
        verify=True,
        execute=True,
        time_limit=3,
        max_tokens=2048,
        code_executor=None,
        analyze_template="<analyze>\nLet's analyze the Paragraph {cur_step} step by step: ",
        verify_template="<verify>\nLet's use python code to find any potential error:\n```python\n",
        output_template="<output>\n**Judgement**: $\\boxed",
        logging=True
    ):
        context = {"cur_step": cur_step}
        analyze_start = analyze_template.format(**context)
        verify_start = verify_template.format(**context)
        output_start = output_template.format(**context)
        # Prepare the input

        # ipdb.set_trace()
        prompt = build_prompt(messages, self.tokenizer)

        # Generate the output
        # Stage 1
        if analyze:
            sampling_params = SamplingParams(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                stop=['</analyze>\n'],
                include_stop_str_in_output=True,
                max_tokens=max_tokens,
                logprobs=20,  # Number of log probabilities to return
                repetition_penalty=REPETITION_PENALTY
            )
            if logging:
                cprint(prompt + analyze_start, f'paragraph {cur_step} request 1')
            output1 = self.model.generate(prompt + analyze_start, sampling_params=sampling_params, use_tqdm=False)[0].outputs[0]
            if verify:
                cur_prompt = analyze_start + output1.text + verify_start  # generate <verify> if verify is True
            else:
                cur_prompt = analyze_start + output1.text + output_start  # directly generate <output> if verify is False

        elif verify:
            cur_prompt = verify_start
        else:
            cur_prompt = output_start

        # Stage 2
        cur_prompts = [cur_prompt]
        out_nodes = []
        cur_time = 0
        while len(cur_prompts) > 0:
            tokenized_prompt = self.tokenizer.tokenize(cur_prompts[0])
            left_tokens = max_tokens - len(tokenized_prompt)
            if left_tokens > 0 and cur_time < time_limit:
                if verify and execute:
                    sampling_params = SamplingParams(
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        top_k=TOP_K,
                        stop=['\n```\n', '</output>\n'],  # set the stop string
                        include_stop_str_in_output=True,  # include the stop string in the output
                        max_tokens=left_tokens,  # Maximum number of tokens to generate
                        logprobs=20,  # Number of log probabilities to return
                        repetition_penalty=REPETITION_PENALTY,
                    )
                else:
                    # not execute
                    sampling_params = SamplingParams(
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        top_k=TOP_K,
                        stop=['</output>\n'],  # set the stop string
                        include_stop_str_in_output=True,  # include the stop string in the output
                        max_tokens=left_tokens,  # Maximum number of tokens to generate
                        logprobs=20,  # Number of log probabilities to return
                        repetition_penalty=REPETITION_PENALTY,
                    )
                if logging:
                    cprint(prompt + cur_prompts[0], f'paragraph {cur_step} request {cur_time + 2}')
                output2 = self.model.generate(prompt + cur_prompts[0], sampling_params, use_tqdm=False)[0].outputs[0]
            else:
                # if the time limit is reached, or the left tokens are not enough
                if analyze:
                    # degrade into analyze mode
                    cur_prompts = [analyze_start + output1.text.split('</analyze>')[0] + '</analyze>\n' + output_start]
                else:
                    # enter the output mode
                    cur_prompts = [cur_prompts[0] + '</verify>\n' + output_start]
                tokenized_prompt = self.tokenizer.tokenize(cur_prompts[0])
                left_tokens = 20
                sampling_params = SamplingParams(
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    stop=['</output>\n'],  # set the stop string
                    include_stop_str_in_output=True,  # include the stop string in the output
                    max_tokens=left_tokens,  # Maximum number of tokens to generate
                    logprobs=20,  # Number of log probabilities to return
                    repetition_penalty=REPETITION_PENALTY,
                )
                if logging:
                    cprint(prompt + cur_prompts[0], f'paragraph {cur_step} request {cur_time + 2}')
                output2 = self.model.generate(prompt + cur_prompts[0], sampling_params, use_tqdm=False)[0].outputs[0]

            cur_time += 1
            new_prompts = []
            if output2.text.endswith('</output>\n'):
                output2.text = cur_prompts[0] + output2.text
                out_nodes.append(output2)
            else:
                if execute:
                    # execute the code
                    code_output = code_executor.execute(cur_prompts[0] + output2.text)
                    code_content = f"[Code Output]\n\n```\n{code_output}\n```\n"
                    new_prompts.append(cur_prompts[0] + output2.text + code_content)
                else:
                    new_prompts.append(cur_prompts[0] + output2.text + '[Code Output]\n\n```\n')

            cur_prompts = new_prompts

        output2 = out_nodes[0]

        # extract the Probability of Yes token as the reward score
        reward_score = self.get_reward_score(output2)

        return output2.text, reward_score
    

#     def _batch_inference(
#     self,
#     messages_list,
#     cur_step=1,
#     analyze=True,
#     verify=True,
#     execute=True,
#     time_limit=3,
#     max_tokens=2048,
#     code_executor=None,
#     analyze_template="<analyze>\nLet's analyze all the Paragraphs step by step: ",
#     verify_template="<verify>\nLet's use python code to find any potential error:\n```python\n",
#     output_template="<output>\n**Judgement**: $\\boxed",
#     logging=True
# ):
#         """
#         Batch inference for multiple message inputs.
        
#         Args:
#             messages_list: List of messages to process in batch
#             Other parameters are the same as _single_inference
        
#         Returns:
#             List of tuples containing (output_text, reward_score) for each input
#         """
#         from types import SimpleNamespace
        
#         context = {"cur_step": cur_step}
#         analyze_start = analyze_template.format(**context)
#         verify_start = verify_template.format(**context)
#         output_start = output_template.format(**context)
        
#         # Prepare prompts for all inputs
#         prompts = [build_prompt(messages, self.tokenizer) for messages in messages_list]
#         batch_size = len(prompts)
        
#         # Initialize results storage
#         results = []
        
#         # Stage 1 - Analyze phase
#         if analyze:
#             sampling_params_kwargs = {
#                 "n": 1, 
#                 "temperature": TEMPERATURE,
#                 "top_p": TOP_P,
#                 "top_k": TOP_K,
#                 "stop": ["</analyze>\n"],
#                 "include_stop_str_in_output": True,
#                 "max_tokens": max_tokens,
#                 "logprobs": 20,
#                 "repetition_penalty": REPETITION_PENALTY
#             }
            
#             # Prepare batch prompts for stage 1
#             stage1_prompts = [prompt + analyze_start for prompt in prompts]
            
#             if logging:
#                 for i, prompt in enumerate(stage1_prompts):
#                     cprint(prompt, f'batch item {i}, paragraph {cur_step} request 1')
            
#             # Generate outputs for all prompts in batch
#             outputs1_raw = self.model.generate(stage1_prompts, generation_kwargs=sampling_params_kwargs)
            
#             # Convert to SimpleNamespace objects
#             outputs1 = [SimpleNamespace(**output) for output in outputs1_raw]

#             # ipdb.set_trace()
            
#             # Prepare prompts for stage 2
#             cur_prompts = []
#             for i, output in enumerate(outputs1):
#                 if verify:
#                     cur_prompts.append(analyze_start + output.text + verify_start)
#                 else:
#                     cur_prompts.append(analyze_start + output.text + output_start)
        
#         elif verify:
#             cur_prompts = [verify_start] * batch_size
#             outputs1 = None
#         else:
#             cur_prompts = [output_start] * batch_size
#             outputs1 = None
        
#         # Stage 2 - Iterative verification and execution
#         # Track active prompts and their indices
#         active_indices = list(range(batch_size))
#         active_prompts = cur_prompts.copy()
#         out_nodes = [None] * batch_size
#         cur_time = 0
        
#         # If code execution is needed, ensure we have code executors for each input
#         if execute and code_executor is not None:
#             # Create separate code executors for each input to maintain isolated namespaces
#             code_executors = [CodeExecutor() for _ in range(batch_size)]
#         else:
#             code_executors = [None] * batch_size
        
#         while len(active_prompts) > 0 and cur_time < time_limit:
#             new_active_indices = []
#             new_active_prompts = []
#             batch_prompts_to_generate = []
#             prompt_to_index_map = []
            
#             # Prepare prompts for this iteration
#             for idx, (orig_idx, cur_prompt) in enumerate(zip(active_indices, active_prompts)):
#                 tokenized_prompt = self.tokenizer.tokenize(cur_prompt)
#                 left_tokens = max_tokens - len(tokenized_prompt)
                
#                 if left_tokens > 0:
#                     full_prompt = prompts[orig_idx] + cur_prompt
#                     batch_prompts_to_generate.append(full_prompt)
#                     prompt_to_index_map.append((idx, orig_idx))
            
#             if len(batch_prompts_to_generate) > 0:
#                 # Set sampling parameters
#                 if verify and execute:
#                     sampling_params_kwargs = {
#                         "n": 1, 
#                         "temperature": TEMPERATURE,
#                         "top_p": TOP_P,
#                         "top_k": TOP_K,
#                         "stop": ['\n```\n', '</output>\n'],
#                         "include_stop_str_in_output": True,
#                         "max_tokens": max_tokens,
#                         "logprobs": 20,
#                         "repetition_penalty": REPETITION_PENALTY
#                     }
#                 else:
#                     sampling_params_kwargs = {
#                         "n": 1, 
#                         "temperature": TEMPERATURE,
#                         "top_p": TOP_P,
#                         "top_k": TOP_K,
#                         "stop": ['</output>\n'],
#                         "include_stop_str_in_output": True,
#                         "max_tokens": max_tokens,
#                         "logprobs": 20,
#                         "repetition_penalty": REPETITION_PENALTY
#                     }

                
#                 if logging:
#                     for i, (_, orig_idx) in enumerate(prompt_to_index_map):
#                         cprint(batch_prompts_to_generate[i], f'batch item {orig_idx}, paragraph {cur_step} request {cur_time + 2}')
                
#                 # Generate outputs in batch
#                 outputs2_raw = self.model.generate(batch_prompts_to_generate, generation_kwargs=sampling_params_kwargs)
#                 outputs2 = [SimpleNamespace(**output) for output in outputs2_raw]
                
#                 # Process outputs
#                 for i, output in enumerate(outputs2):
#                     active_idx, orig_idx = prompt_to_index_map[i]
                    
#                     if output.text.endswith('</output>\n'):
#                         # Complete output, store it
#                         output.text = active_prompts[active_idx] + output.text
#                         out_nodes[orig_idx] = output
#                     else:
#                         # Need to continue processing
#                         if execute and code_executors[orig_idx] is not None:
#                             # Execute the code
#                             code_output = code_executors[orig_idx].execute(active_prompts[active_idx] + output.text)
#                             code_content = f"[Code Output]\n\n```\n{code_output}\n```\n"
#                             new_prompt = active_prompts[active_idx] + output.text + code_content
#                         else:
#                             new_prompt = active_prompts[active_idx] + output.text + '[Code Output]\n\n```\n'
                        
#                         new_active_indices.append(orig_idx)
#                         new_active_prompts.append(new_prompt)
            
#             # Handle prompts that couldn't be processed due to token limits
#             for idx, (orig_idx, cur_prompt) in enumerate(zip(active_indices, active_prompts)):
#                 if orig_idx not in [x[1] for x in prompt_to_index_map]:
#                     # This prompt wasn't processed, force completion
#                     if analyze and outputs1 is not None:
#                         # Degrade to analyze mode
#                         forced_prompt = analyze_start + outputs1[orig_idx].text.split('</analyze>')[0] + '</analyze>\n' + output_start
#                     else:
#                         # Enter output mode
#                         forced_prompt = cur_prompt + '</verify>\n' + output_start
                    
#                     # Generate final output with limited tokens
#                     sampling_params_kwargs = {
#                         "n": 1, 
#                         "temperature": TEMPERATURE,
#                         "top_p": TOP_P,
#                         "top_k": TOP_K,
#                         "stop": ['</output>\n'],
#                         "include_stop_str_in_output": True,
#                         "max_tokens": 20,
#                         "logprobs": 20,
#                         "repetition_penalty": REPETITION_PENALTY
#                     }
                    
#                     if logging:
#                         cprint(prompts[orig_idx] + forced_prompt, f'batch item {orig_idx}, paragraph {cur_step} request {cur_time + 2} (forced)')
                    
#                     forced_output_raw = self.model.generate([prompts[orig_idx] + forced_prompt], generation_kwargs=sampling_params_kwargs)[0]

#                     forced_output = SimpleNamespace(**forced_output_raw)
#                     forced_output.text = forced_prompt + forced_output.text
#                     out_nodes[orig_idx] = forced_output
            
#             active_indices = new_active_indices
#             active_prompts = new_active_prompts
#             cur_time += 1
        
#         # Force completion for any remaining active prompts
#         if len(active_indices) > 0:
#             remaining_prompts = []
#             remaining_indices = []
            
#             for orig_idx, cur_prompt in zip(active_indices, active_prompts):
#                 if analyze and outputs1 is not None:
#                     forced_prompt = analyze_start + outputs1[orig_idx].text.split('</analyze>')[0] + '</analyze>\n' + output_start
#                 else:
#                     forced_prompt = cur_prompt + '</verify>\n' + output_start
                
#                 remaining_prompts.append(prompts[orig_idx] + forced_prompt)
#                 remaining_indices.append(orig_idx)
            
#             sampling_params_kwargs = {
#                 "n": 1, 
#                 "temperature": TEMPERATURE,
#                 "top_p": TOP_P,
#                 "top_k": TOP_K,
#                 "stop": ['</output>\n'],
#                 "include_stop_str_in_output": True,
#                 "max_tokens": 20,
#                 "logprobs": 20,
#                 "repetition_penalty": REPETITION_PENALTY
#             }
            
#             if logging:
#                 for i, orig_idx in enumerate(remaining_indices):
#                     cprint(remaining_prompts[i], f'batch item {orig_idx}, paragraph {cur_step} final request (timeout)')
            
#             final_outputs_raw = self.model.generate(remaining_prompts, generation_kwargs=sampling_params_kwargs)

#             final_outputs = [SimpleNamespace(**output) for output in final_outputs_raw]
            
#             for i, output in enumerate(final_outputs):
#                 orig_idx = remaining_indices[i]
#                 forced_prompt = remaining_prompts[i].replace(prompts[orig_idx], '')
#                 output.text = forced_prompt + output.text
#                 out_nodes[orig_idx] = output
        

#         # ipdb.set_trace()
#         # Extract reward scores and prepare results
#         for i, output_node in enumerate(out_nodes):
#             if output_node is not None:
#                 reward_score = self.get_reward_score(output_node)
#                 # reward_score = self.get_reward_score_all(output_node)
#                 results.append((output_node.text, reward_score))
#             else:
#                 # Should not happen, but handle gracefully
#                 results.append(("Error: No output generated", 0.5))
        
#         # ipdb.set_trace()
#         return results

    def _batch_inference(
        self,
        messages_list,
        cur_step=1,
        analyze=True,
        verify=True,
        execute=True,
        time_limit=3,
        max_tokens=2048,
        code_executor=None,
        analyze_template="<analyze>\nLet's analyze all the Paragraphs step by step: ",
        verify_template="<verify>\nLet's use python code to find any potential error:\n```python\n",
        output_template="<output>\n**Judgement**: $\\boxed",
        logging=True
    ):
        """
        Batch inference for multiple message inputs.
        
        Args:
            messages_list: List of messages to process in batch
            Other parameters are the same as _single_inference
        
        Returns:
            List of tuples containing (output_text, reward_score) for each input
        """
        from types import SimpleNamespace
        
        context = {"cur_step": cur_step}
        analyze_start = analyze_template.format(**context)
        verify_start = verify_template.format(**context)
        output_start = output_template.format(**context)
        
        # Prepare prompts for all inputs
        prompts = [build_prompt(messages, self.tokenizer) for messages in messages_list]
        batch_size = len(prompts)
        
        # Initialize results storage
        results = []
        
        # Stage 1 - Analyze phase
        if analyze:
            sampling_params_kwargs = {
                "n": 1, 
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "stop": ["</analyze>\n"],
                "include_stop_str_in_output": True,
                "max_tokens": max_tokens,
                "logprobs": 20,
                "repetition_penalty": REPETITION_PENALTY
            }
            
            # Prepare batch prompts for stage 1
            stage1_prompts = [prompt + analyze_start for prompt in prompts]
            
            if logging:
                for i, prompt in enumerate(stage1_prompts):
                    cprint(prompt, f'batch item {i}, paragraph {cur_step} request 1')
            
            # Generate outputs for all prompts in batch
            outputs1_raw = self.model.generate(stage1_prompts, generation_kwargs=sampling_params_kwargs)
            
            # Convert to SimpleNamespace objects
            outputs1 = [SimpleNamespace(**output) for output in outputs1_raw]
            
            # Prepare prompts for stage 2
            cur_prompts = []
            for i, output in enumerate(outputs1):
                if verify:
                    cur_prompts.append(analyze_start + output.text + verify_start)
                else:
                    cur_prompts.append(analyze_start + output.text + output_start)
        
        # Stage 2 - Iterative verification and execution
        # Track active prompts and their indices
        active_prompts = cur_prompts.copy()
        out_nodes = [None] * batch_size
        
        sampling_params_kwargs = {
            "n": 1, 
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "stop": ['</output>\n'],
            "include_stop_str_in_output": True,
            "max_tokens": 100,
            "logprobs": 20,
            "repetition_penalty": REPETITION_PENALTY
        }
        
        # Generate outputs in batch
        outputs2_raw = self.model.generate(active_prompts, generation_kwargs=sampling_params_kwargs)
        outputs2 = [SimpleNamespace(**output) for output in outputs2_raw]
        
        # Process outputs
        for i, output in enumerate(outputs2):     
            output.text = active_prompts[i] + output.text
            out_nodes[i] = output
        
        # Extract reward scores and prepare results
        for i, output_node in enumerate(out_nodes):
            if output_node is not None:
                # reward_score = self.get_reward_score(output_node)
                reward_score = self.get_reward_score_all(output_node)
                results.append((output_node.text, reward_score))
            else:
                # Should not happen, but handle gracefully
                results.append(("Error: No output generated", 0.5))
        
        # ipdb.set_trace()
        return results