# Code for PRISM: A Unified Framework for Post-Training LLMs Without Verifiable Rewards 

Paper: https://arxiv.org/pdf/2601.04700

### Citation

```
@misc{ghimire2026prismunifiedframeworkposttraining,
      title={PRISM: A Unified Framework for Post-Training LLMs Without Verifiable Rewards}, 
      author={Mukesh Ghimire and Aosong Feng and Liwen You and Youzhi Luo and Fang Liu and Xuan Zhu},
      year={2026},
      eprint={2601.04700},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.04700}, 
}
```

## INSTRUCTIONS FOR OPEN-R1 

Please follow the guidelines in the file [`OPENR1_README.md`](OPENR1_README.md) for installing the environment. Once that is completed, follow the guidelines below for setting up PRISM and other related information regarding PRISM. 

*Note: When you install open-r1 environment as outlined above, it automatically installs a modified [trl](https://github.com/ghimiremukesh/trl.git) to support PRISM training.*

---

## INSTRUCTIONS FOR PRISM

### OVERVIEW 

- Training scripts are in `open-r1-intuitor/`
	- RLIF: Intuitor: `run_intuitor.sh`
	- RLIF: Token-Entropy: `run_token_entropy.sh`
	- RLIF: Trajectory-Entropy: `run_traj_entropy.sh`
	- GRPO (with GT): `run_grpo.sh`
	- PRISM: `run_decay_intuitor_GenPRM.sh`

- Training configs are in `open-r1-intuitor/recipes`. Each model has config files for each of the training methods. 

- The implementation of RLIF methods and PRISM are in `open-r1-intuitor/src/open_r1`
	- The first point of entry for each of the methods is the file in the form `{method}.py` (e.g., `intuitor.py`, `grpo.py`, `decay_intuitor_prm.py`, ...)
		- These modules then call `{method}_trainer.py`, which contains the **advantage computation**, **prompts processing** and **logging**. Example files: `intuitor_trainier.py`, `decay_intuitor_prm_trainer.py`, `token_entropy_trainer.py`, ...

### CONFIG FILE DETAILS
- All configs are inside `open-r1-intuitor/recipes`, and contains parameters such as `model`, `dataset`, `learning_rate`, `kl_penalty`, `num_generations`, `gradient_accumulation_steps`, `per_device_train_batch_size`, and so on.
- For PRISM, the config file has additional parameters:
	- `include_intuitor`: True if we want to add self-certainty advantage 
	- `prm_model`: local model path or huggingface model name (e.g., `Qwen/Qwen2.5-3B`)

- `reward_funcs` parameter in the config file specifies which rewards to compute, such as `accuracy`, `format`, `noise_reward`, and several others. All reward functions are implemented in `open-r1-intuitor/src/open_r1/rewards.py`. Any additional reward functions for model's generation can be implemented in `rewards.py` and should be added to the registry inside 

	```
	634|  def get_reward_funcs(script_args) -> list[Callable]
				REWARD_FUNCS_REGISTRY = {
					"noise": noise_reward, 
					"accuracy": accuracy_reward,
					...
				}
	```
- After specifying `reward_funcs` in the config file, we need to specify the reward weights for each of the reward functions. To do so, add the respective reward weight for each of the `reward_funcs` as shown below:
	```
	reward_funcs:
	- accuracy
	- noise
	reward_weights:
	- 1.0  # this will include the accuracy reward when computing advantage that gets passed to the loss function
	- 0.0  # this will compute the noise reward, but not include it in the actual advantage computation. Can be used for monitoring purposes.
	```

- For GRPO (w/ ground-truth) training, set the `reward_funcs` to `accuracy` with `reward_weights` set to `1.0`.
- For INTUITOR and PRISM training, set the `reward_funcs` to `accuracy` with `reward_weights` set to `0.0` so that it is only used for monitoring and not learning. 

### DETAILS ON GenPRM

- After the base model (the model we want to train) generates its response, it is to be passed as input to the GenPRM model. 
- However, first we will need to parse only the QUESTION (PROMPT) and the model's response from the decoded generation, which consists of the entire chat including the system prompt for the base model. To do so, we need to first know the chat template of the model we are training, and extract everything after the system prompt. This is done using a helper function `extract_user_prompt` in line 86 of `open-r1-intuitor/src/open_r1/decay_intuitor_prm_trainer.py`. 
	- add the template to capture anything after system prompt. For example: 
		```
		pattern =  r'<\|im_start\|>user\n(.*?)<\|im_end\|>'  # for qwen models
		```
- The system prompt for the GenPRM model is then added to the chat in lines `1199-1211` in the same file. Replace the system prompt with any other system prompt you'd like to use for the GenPRM model. An example is below:
	```
	prm_prompts.append([
	{"role": "system", "content": "You are a math teacher. Your task is to review, critique the paragraphs in solution step by step, and check if the solution is complete. Pay attention that you should neither solve the problem nor give the final answer."},
	{"role": "user", "content": user_prompt}
	])
	```

- The outputs from GenPRM along with the reward scores can be obtained by calling:
	```
	self.prm_model.inference(prm_prompts, verify=False, execute=False, logging=False, 
							analyze_template="<analyze>\nLet's analyze the codes step by step:", 
							output_template="<output>Let's provide judgements for each of the major 
							steps.\nStep 1 **Judgement**: $\\boxed")
	```

- `self.prm_model.inference` then calls the `_batch_inference` function of `open-r1-intuitor/src/open_r1/GenPRM/genprm_batch_inference_w_completeness.py` script. The rewards are first computed for individual steps using the function `get_reward_score_all`. The complenetess reward is computed using `get_completion_score` function in the same script.

- To add any additional feedback from GenPRM, simply repeat the lines `1007-1043` and make any necessary changes with the reward computation of your choice. 

### ADDITIONAL REWARD FUNCTION

- The following additional reward that checks whether or not the model's generation has boxed answers is implemented in the `open-r1-intuitor/src/open_r1/rewards.py` in line 99:
	```
	# reward function for rewarding boxed answer 
	def boxed_reward(completions, **kwargs):
		"""
		Reward = 1.0 iff the final answer is enclosed in \boxed{...}
		and appears as the last non-whitespace content (optionally within
		$, $$, \( \), or \[ \] math delimiters). Else 0.0.
		"""
		pattern = re.compile(
			r"""
			\A
			[\s\S]*?                               # any prior content
			(?:                                    # optional opening math delimiters
				(?:\$\$|\$)\s* |
				\\[\(\[]\s*
			)?
			\\boxed\s*\{[^{}]*\}                   # the boxed final answer (no nested braces)
			(?:                                    # optional closing math delimiters
				\s*\\[\)\]] |
				\s*(?:\$\$|\$)
			)?
			\s*\Z                                  # only whitespace allowed after
			""",
			re.VERBOSE | re.DOTALL
		)

		def _content(item):
			# Mirror your structure: item[0]["content"]; fallback to str(item) if different
			try:
				return item[0]["content"]
			except Exception:
				return str(item)	

		contents = [_content(c) for c in completions]
		return [1.0 if pattern.match(text) else 0.0 for text in contents]
	```

- To use this reward during PRISM training, include the following in the config file:
	```
	reward_funcs:
	- accuracy
	- boxed
	reward_weights:
	- 0.0  # accuracy only for monitoring
	- 1.0  # boxed reward for computing advantage 
	```
- The reward functions specified this way will be automatically logged during training. This is done in `decay_intuitor_prm_trainer.py` in line 1378:
	```
	for i, reward_func_name in enumerate(self.reward_func_names):
		mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
		self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
		std_rewards = nanstd(rewards_per_func[:, i]).item()
		self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
	```


### ADVANTAGE COMPUTATION FOR PRISM

- The advantage is computed in the file `open-r1-intuitor/src/open_r1/decay_intuitor_prm_trainer.py` inside the function `_generate_and_score_completions` in lines `1306-1352`. It follows similar logic as in GRPO or INTUITOR's advantage computation. 

- The `advantages` in line `1348` computes the advantage for the reward functions specified in `reward_funcs` in the config file that have their respective `reward_weights > 0`. If all `reward_weights` are equal to zero, `advantages` is simply as matrix of zeros. 
	```
	1348| sce_advantage = sce_advantage + advantages
	```
- The final advantage that is used in downstream loss computation is then computed as the following depending on if `include_intuitor` is `True` or `False`
	```
	1325| if self.include_intuitor:
			... 
			# all previous computations 
			...

	1350|	final_advantage = self.sc_weight * sce_advantage + prm_advantages[process_slice]
	1351| else:
	1352|	final_advantages = prm_advantages[process_slice]
	```
- The weight $\gamma$ can be specified as a constant in the `__init__` as `self.sc_weight = 1.0` (line 407). To use a decaying $\gamma$ based on training iterations, uncomment the line `1346`:
	```
	1346|  # self.sc_weight = (1 - (self.state.global_step / self.state.max_steps)) ** 2  # decay the \gamma
	```
	
### DETAILS FOR EVALUATION

- For evaluation on math performance, use the following template as a reference. Lighteval package should already be installed with the `openr1` environment. If not, you can do so with `pip install lighteval` or via your preferred package manager. 
	```
	export VLLM_WORKER_MULTIPROC_METHOD=spawn
	export MODEL=Qwen/Qwen2.5-3B ## or your local model path
	export MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,gpu_memory_utilization=0.8,data_parallel_size=1,max_model_length=32768,generation_parameters={max_new_tokens:4096,temperature:0}"
	export OUTPUT_DIR=./results/
	export TASKA=math_500
	export TASKC=gsm8k
	export N=0
	lighteval vllm $MODEL_ARGS "lighteval|$TASKA|$N|0,lighteval|$TASKB|$N|0,lighteval|$TASKC|$N|0" \
		--use-chat-template \
		--output-dir $OUTPUT_DIR \
		--save-details
	```

- For evaluation on code generation, clone the repo: [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) and follow the instructions therein. 


## REFERENCES

This work is possible due to the following repos:

- [Intuitor](https://github.com/sunblaze-ucb/Intuitor)
- [GenPRM](https://github.com/RyanLiu112/GenPRM)


