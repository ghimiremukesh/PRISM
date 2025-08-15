import torch

def batch_generate(
    model,
    tokenizer,
    prompts_text,
    max_length,
    device,
    attention_mask=None,
    generation_config=None,
    use_vllm=False,
    vllm_client=None,
    accelerator=None,
    state=None,
    num_generations=1,
    repetition_penalty=1.0,
    temperature=1.0,
    top_p=1.0,
    top_k=None,
    min_p=None,
    guided_decoding_regex=None,
    processing_class=None,
):
    """
    Generates completions (batched) using the same logic as your original code.
    Handles both vLLM and regular model inference.
    Returns: batch of generated texts.
    """
    if use_vllm:
        # vLLM path
        all_prompts_text = prompts_text
        if accelerator.is_main_process:
            ordered_set_of_prompts = all_prompts_text[::num_generations]
            completion_ids = vllm_client.generate(
                prompts=ordered_set_of_prompts,
                n=num_generations,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=-1 if top_k is None else top_k,
                min_p=0.0 if min_p is None else min_p,
                max_tokens=max_length,
                guided_decoding_regex=guided_decoding_regex,
            )
        else:
            completion_ids = [None] * len(all_prompts_text)
        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        process_slice = slice(
            accelerator.process_index * len(prompts_text),
            (accelerator.process_index + 1) * len(prompts_text),
        )
        completion_ids = completion_ids[process_slice]
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=processing_class.pad_token_id)
        completions_text = processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        return completions_text
    else:
        # Regular HF model path
        inputs = tokenizer(
            prompts_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=max_length,
                **(generation_config or {})
            )
        # Remove prompt prefix from decoded output
        completions_text = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        no_prompt_completions = []
        for prompt, full in zip(prompts_text, completions_text):
            if full.startswith(prompt):
                no_prompt_completions.append(full[len(prompt):].strip())
            else:
                no_prompt_completions.append(full)
        return no_prompt_completions

def batch_self_critique_and_revision(
    prompts, completions, model, tokenizer, device,
    max_length, processing_class,
    use_vllm=False,
    vllm_client=None,
    accelerator=None,
    state=None,
    num_generations=1,
    repetition_penalty=1.0,
    temperature=1.0,
    top_p=1.0,
    top_k=None,
    min_p=None,
    guided_decoding_regex=None,
    generation_config=None,
):
    """
    Given prompts and completions, performs self-critique and revision where needed.
    Returns: revised_completions, critiques, revised_flags
    """
    critique_prompt_template = (
        "{}\n\n"
        "Please critique your previous response. Is the solution correct? Are there any logical flaws in the steps? "
        "Are there any calculations that need to be checked? Please provide a detailed analysis of your response. "
        "On top of your analysis also state a verdict 'Revision needed: 1' or 'Revision needed: 0'."
    )

    # 1. Generate critiques for all completions
    critique_prompts = [
        critique_prompt_template.format(completion)
        for completion in completions
    ]

    critiques = batch_generate(
        model, tokenizer, critique_prompts, max_length, device,
        generation_config=generation_config, use_vllm=use_vllm, vllm_client=vllm_client,
        accelerator=accelerator, state=state, num_generations=num_generations,
        repetition_penalty=repetition_penalty, temperature=temperature, top_p=top_p,
        top_k=top_k, min_p=min_p, guided_decoding_regex=guided_decoding_regex,
        processing_class=processing_class
    )

    # 2. Decide which need revision
    def parse_revision_needed(critique):
        verdict = "revision needed: 1"
        return verdict in critique.lower()

    needs_revision = [parse_revision_needed(c) for c in critiques]

    # 3. For those needing revision, generate revision prompts
    revision_prompt_template = (
        "Based on your critique, please revise your previous answer. "
        "Make sure the answer is in a box: \\boxed{Your Answer}. "
        "Please stop generation immediately after outputting the box."
    )

    revision_prompts = [
        f"{prompt}\n\nPrevious response:\n{completion}\n\nCritique:\n{critique}\n\n{revision_prompt_template}"
        for prompt, completion, critique, need_rev in zip(prompts, completions, critiques, needs_revision)
        if need_rev
    ]
    revised_completions = completions[:]  # start with original completions
    # If any need revision, generate in batch
    if revision_prompts:
        revised = batch_generate(
            model, tokenizer, revision_prompts, max_length, device,
            generation_config=generation_config, use_vllm=use_vllm, vllm_client=vllm_client,
            accelerator=accelerator, state=state, num_generations=1,
            repetition_penalty=repetition_penalty, temperature=temperature, top_p=top_p,
            top_k=top_k, min_p=min_p, guided_decoding_regex=guided_decoding_regex,
            processing_class=processing_class
        )
        # Replace only revised completions
        idx = 0
        for i, need_rev in enumerate(needs_revision):
            if need_rev:
                revised_completions[i] = revised[idx]
                idx += 1
    return revised_completions, critiques, needs_revision