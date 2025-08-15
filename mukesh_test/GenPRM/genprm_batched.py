import re
import math
import io
import signal
from copy import *
from GenPRM.util import *
from contextlib import redirect_stdout
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from torch.nn.parallel import DataParallel, DistributedDataParallel

TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
REPETITION_PENALTY = 1.0
version = 'v1.0'


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
    def __init__(self, model_path, tensor_parallel_size):
        # Load the model and tokenizer
        timestamped_print(f"Loading model from {model_path}", level="INFO")
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            enable_chunked_prefill=True, 
            max_num_batched_tokens=8192,
            max_num_seqs=128,
            disable_log_stats=True,
        )

        # just use regular
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_path, 
        #     device_map="auto", 
        #     torch_dtype=torch.float16,
        #     low_cpu_mem_usage=True
        # )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        timestamped_print(f"GenPRM loaded successfully", level="INFO")

    # def get_reward_score(self, out):
    #     '''calculate the reward score'''
    #     generated_text = out.text
    #     logprobs = out.logprobs
    #     tokens = out.token_ids
    #     token_logprobs = logprobs

    #     # find the position of Yes/No token
    #     boxed_match = re.search(r'(Yes|No)\}', generated_text, re.IGNORECASE)
    #     yes_token = self.tokenizer.encode('Yes')[-1]
    #     no_token = self.tokenizer.encode('No')[-1]

    #     if boxed_match:
    #         decision = boxed_match.group(1).capitalize()
    #         if decision == "Yes":
    #             yes_index = len(tokens) - 1 - tokens[::-1].index(yes_token)
    #             yes_logprob = token_logprobs[yes_index][yes_token].logprob
    #             # convert logprob to probability
    #             yes_prob = math.exp(yes_logprob)  # e^log(prob) = prob

    #             # find the position of 'No' token
    #             try:
    #                 no_logprob = token_logprobs[yes_index][no_token].logprob
    #                 no_prob = math.exp(no_logprob)
    #             except KeyError:
    #                 # set 'No' probability to the minimum logprob of the remaining 4 logprobs
    #                 min_logprob = min(v.logprob for k, v in token_logprobs[yes_index].items())
    #                 no_prob = math.exp(min_logprob)

    #             # calculate softmax value
    #             softmax_denominator = yes_prob + no_prob
    #             if softmax_denominator == 0:
    #                 softmax_yes = 0.5  # in case of division by zero, assign neutral score
    #             else:
    #                 softmax_yes = yes_prob / softmax_denominator

    #             return softmax_yes

    #         elif decision == "No":
    #             no_index = len(tokens) - 1 - tokens[::-1].index(no_token)
    #             no_logprob = token_logprobs[no_index][no_token].logprob
    #             # convert logprob to probability
    #             no_prob = math.exp(no_logprob)  # e^log(prob) = prob

    #             # find the position of 'Yes' token
    #             try:
    #                 yes_logprob = token_logprobs[no_index][yes_token].logprob
    #                 yes_prob = math.exp(yes_logprob)
    #             except KeyError:
    #                 # set 'Yes' probability to the minimum logprob of the remaining 4 logprobs
    #                 min_logprob = min(v.logprob for k, v in token_logprobs[no_index].items())
    #                 yes_prob = math.exp(min_logprob)

    #             # calculate softmax value
    #             softmax_denominator = yes_prob + no_prob
    #             if softmax_denominator == 0:
    #                 softmax_yes = 0.5  # in case of division by zero, assign neutral score
    #             else:
    #                 softmax_yes = yes_prob / softmax_denominator

    #             return softmax_yes
    #     else:
    #         # return neutral score if no decision found
    #         timestamped_print("No boxed{Yes/No} found in the output", level="WARNING")
    #         return 0.5

    def get_reward_score(self, out):
        '''calculate the reward score'''
        generated_text = out.text
        logprobs = out.logprobs
        tokens = out.token_ids
        token_logprobs = logprobs

        # find the position of Yes/No token
        boxed_match = re.search(r'(Yes|No)\}', generated_text, re.IGNORECASE)
        
        # Get token IDs for Yes and No
        yes_token = self.tokenizer.encode('Yes', add_special_tokens=False)[-1]
        no_token = self.tokenizer.encode('No', add_special_tokens=False)[-1]
        
        # Also check for common variations that might be tokenized differently
        yes_variants = []
        no_variants = []
        try:
            # Check how "Yes}" and "No}" are tokenized
            yes_brace_tokens = self.tokenizer.encode('Yes}', add_special_tokens=False)
            no_brace_tokens = self.tokenizer.encode('No}', add_special_tokens=False)
            
            # If "Yes}" or "No}" is a single token, add it to variants
            if len(yes_brace_tokens) == 1:
                yes_variants.append(yes_brace_tokens[0])
            if len(no_brace_tokens) == 1:
                no_variants.append(no_brace_tokens[0])
                
            # Also check for "Yes" and "No" with different casings or contexts
            yes_variants.extend(self.tokenizer.encode('YES', add_special_tokens=False))
            no_variants.extend(self.tokenizer.encode('NO', add_special_tokens=False))
        except:
            pass

        if boxed_match:
            decision = boxed_match.group(1).capitalize()
            
            # Helper function to find token index with variants
            def find_token_index(target_token, variant_tokens, search_from_end=True):
                """Find the index of target token or any of its variants"""
                all_tokens_to_search = [target_token] + list(set(variant_tokens))
                
                if search_from_end:
                    # Search from the end (most recent tokens first)
                    for i in range(len(tokens) - 1, -1, -1):
                        if tokens[i] in all_tokens_to_search:
                            return i, tokens[i]
                else:
                    # Search from the beginning
                    for i in range(len(tokens)):
                        if tokens[i] in all_tokens_to_search:
                            return i, tokens[i]
                
                return None, None
            
            if decision == "Yes":
                yes_index, found_token = find_token_index(yes_token, yes_variants)
                
                if yes_index is None:
                    # Fallback: try to find any token that decodes to "Yes"
                    for i in range(len(tokens) - 1, max(0, len(tokens) - 20), -1):
                        decoded = self.tokenizer.decode([tokens[i]], skip_special_tokens=True).strip()
                        if decoded.lower() == 'yes':
                            yes_index = i
                            found_token = tokens[i]
                            break
                
                if yes_index is None:
                    timestamped_print(f"Yes token not found. Last 20 chars: '{generated_text[-20:]}'", level="WARNING")
                    return 0.5
                
                # Use the found token for logprob lookup
                yes_logprob = token_logprobs[yes_index][found_token].logprob
                yes_prob = math.exp(yes_logprob)

                # Get No probability at the same position
                no_prob = 1e-10  # default small probability
                
                # Try to find No token probability at the same position
                for no_tok in [no_token] + no_variants:
                    if no_tok in token_logprobs[yes_index]:
                        no_prob = math.exp(token_logprobs[yes_index][no_tok].logprob)
                        break
                else:
                    # If no No token found, use minimum probability
                    if len(token_logprobs[yes_index]) > 0:
                        min_logprob = min(v.logprob for k, v in token_logprobs[yes_index].items())
                        no_prob = math.exp(min_logprob)

                # calculate softmax value
                softmax_denominator = yes_prob + no_prob
                if softmax_denominator == 0:
                    softmax_yes = 0.5
                else:
                    softmax_yes = yes_prob / softmax_denominator

                return softmax_yes

            elif decision == "No":
                no_index, found_token = find_token_index(no_token, no_variants)
                
                if no_index is None:
                    # Fallback: try to find any token that decodes to "No"
                    for i in range(len(tokens) - 1, max(0, len(tokens) - 20), -1):
                        decoded = self.tokenizer.decode([tokens[i]], skip_special_tokens=True).strip()
                        if decoded.lower() == 'no':
                            no_index = i
                            found_token = tokens[i]
                            break
                
                if no_index is None:
                    timestamped_print(f"No token not found. Last 20 chars: '{generated_text[-20:]}'", level="WARNING")
                    return 0.5
                
                # Use the found token for logprob lookup
                no_logprob = token_logprobs[no_index][found_token].logprob
                no_prob = math.exp(no_logprob)

                # Get Yes probability at the same position
                yes_prob = 1e-10  # default small probability
                
                # Try to find Yes token probability at the same position
                for yes_tok in [yes_token] + yes_variants:
                    if yes_tok in token_logprobs[no_index]:
                        yes_prob = math.exp(token_logprobs[no_index][yes_tok].logprob)
                        break
                else:
                    # If no Yes token found, use minimum probability
                    if len(token_logprobs[no_index]) > 0:
                        min_logprob = min(v.logprob for k, v in token_logprobs[no_index].items())
                        yes_prob = math.exp(min_logprob)

                # calculate softmax value
                softmax_denominator = yes_prob + no_prob
                if softmax_denominator == 0:
                    softmax_yes = 0.5
                else:
                    softmax_yes = yes_prob / softmax_denominator

                return softmax_yes
        else:
            # return neutral score if no decision found
            timestamped_print("No boxed{Yes/No} found in the output", level="WARNING")
            return 0.5

    def batch_inference(
        self,
        batch_requests,
        batch_size=32,
        verify=False,
        execute=False,
        time_limit=3,
        max_tokens=2048,
        code_executor=None,
        analyze_template="<analyze>\nLet's analyze the Paragraph {cur_step} step by step: ",
        verify_template="<verify>\nLet's use python code to find any potential error:\n```python\n",
        output_template="<output>\n**Judgement**: $\\boxed",
        logging=False
    ):
        """
        Batch inference for multiple requests
        
        Args:
            batch_requests: List of dicts, each containing:
                - messages: the input messages
                - cur_step: the current step index
            batch_size: Number of prompts to process simultaneously
            Other args same as inference()
        
        Returns:
            List of tuples: [(output_text, reward_score), ...]
        """
        all_results = []
        
        # Process in batches
        for batch_start in range(0, len(batch_requests), batch_size):
            batch_end = min(batch_start + batch_size, len(batch_requests))
            current_batch = batch_requests[batch_start:batch_end]
            
            # Prepare all prompts for Stage 1
            batch_prompts = []
            batch_metadata = []
            
            for req in current_batch:
                messages = req['messages']
                cur_step = req['cur_step']
                
                context = {"cur_step": cur_step}
                analyze_start = analyze_template.format(**context)
                output_start = output_template.format(**context)
                
                prompt = build_prompt(messages, self.tokenizer)
                
                if not verify:
                    # Since verify=False in your use case
                    full_prompt = prompt + analyze_start
                    batch_prompts.append(full_prompt)
                    batch_metadata.append({
                        'cur_step': cur_step,
                        'analyze_start': analyze_start,
                        'output_start': output_start,
                        'base_prompt': prompt
                    })
            
            # Stage 1: Generate analyze section
            sampling_params = SamplingParams(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                stop=['</analyze>\n'],
                include_stop_str_in_output=True,
                max_tokens=max_tokens,
                logprobs=20,
                repetition_penalty=REPETITION_PENALTY
            )
            
            if logging:
                print(f"Processing batch {batch_start}-{batch_end}")
            
            stage1_outputs = self.model.generate(batch_prompts, sampling_params, use_tqdm=False)
            
            # Stage 2: Generate output section
            stage2_prompts = []
            for i, output in enumerate(stage1_outputs):
                output_text = output.outputs[0].text
                metadata = batch_metadata[i]
                
                # Construct prompt for output generation
                stage2_prompt = metadata['base_prompt'] + metadata['analyze_start'] + output_text + metadata['output_start']
                stage2_prompts.append(stage2_prompt)
            
            # Generate final outputs
            sampling_params2 = SamplingParams(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                stop=['</output>\n'],
                include_stop_str_in_output=True,
                max_tokens=20,  # Just need Yes/No
                logprobs=20,
                repetition_penalty=REPETITION_PENALTY
            )
            
            stage2_outputs = self.model.generate(stage2_prompts, sampling_params2, use_tqdm=False)
            
            # Extract results
            for i, (s1_out, s2_out) in enumerate(zip(stage1_outputs, stage2_outputs)):
                # Combine the full output
                full_text = batch_metadata[i]['analyze_start'] + s1_out.outputs[0].text + batch_metadata[i]['output_start'] + s2_out.outputs[0].text
                
                # Get reward score from the final output
                reward_score = self.get_reward_score(s2_out.outputs[0])
                
                all_results.append((full_text, reward_score))
        
        return all_results

    def inference(
        self,
        messages,
        majority_num=1,
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
        '''
        messages: the input messages
        majority_num: the number of majority votes
        cur_step: the current step index (start from 1)
        analyze: whether to analyze the input
        verify: whether to verify the input
        execute: whether to execute the code
        time_limit: the time limit for code execution
        max_tokens: the maximum tokens for the output
        analyze_template: the template for analyze start
        verify_template: the template for verify start
        output_template: the template for output start
        logging: whether to log the process
        '''
        output_paths = []
        reward_scores = []
        for i in range(majority_num):
            # perform inference
            output_path, reward_score = self._single_inference(
                messages,
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

            output_paths.append(output_path)
            reward_scores.append(reward_score)

        return output_paths, sum(reward_scores) / len(reward_scores)

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