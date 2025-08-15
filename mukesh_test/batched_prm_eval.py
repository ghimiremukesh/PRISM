import pandas as pd
import ipdb
from tqdm import tqdm
from GenPRM.genprm_batched import GenPRM

def process_chats_batch(genprm, chats, batch_size=32):
    """Process all chats using batch inference"""
    
    # Step 1: Collect all requests
    all_requests = []
    request_mapping = []  # (chat_idx, message_idx)
    
    print("Collecting all inference requests...")
    for chat_idx, sol in enumerate(chats):
        for i in range(len(sol)):
            if sol[i]['role'] != 'assistant':
                continue
            
            request = {
                'messages': sol[:i],
                'cur_step': int(i/2)
            }
            all_requests.append(request)
            request_mapping.append((chat_idx, i))
    
    print(f"Total requests to process: {len(all_requests)}")
    
    # Step 2: Batch process with progress bar
    print(f"Processing {len(all_requests)} requests in batches of {batch_size}")
    
    with tqdm(total=len(all_requests), desc='Generating Process Rewards (Batch)') as pbar:
        batch_results = []
        
        for batch_start in range(0, len(all_requests), batch_size):
            batch_end = min(batch_start + batch_size, len(all_requests))
            current_batch = all_requests[batch_start:batch_end]
            
            # Process current batch
            batch_outputs = genprm.batch_inference(
                current_batch,
                batch_size=len(current_batch),  # Process entire current batch at once
                verify=False,
                logging=False,
                max_tokens=4096
            )
            
            batch_results.extend(batch_outputs)
            pbar.update(len(current_batch))
    
    # Step 3: Reorganize results
    all_reward_list = []
    all_process_rewards = []
    prm_outputs = []
    
    # Initialize containers for each chat
    chat_rewards = [[] for _ in range(len(chats))]
    chat_outputs = [[] for _ in range(len(chats))]
    
    # Distribute results back to chats
    for (chat_idx, msg_idx), (output_text, reward) in zip(request_mapping, batch_results):
        chats[chat_idx][msg_idx]['content'] = output_text
        chat_rewards[chat_idx].append(reward)
        chat_outputs[chat_idx].append(output_text)
    
    # Calculate final metrics
    for rewards, outputs in zip(chat_rewards, chat_outputs):
        all_process_rewards.append(rewards)
        prm_outputs.append(outputs)
        all_reward_list.append(sum(rewards)/len(rewards) if rewards else 0)
    
    return all_reward_list, all_process_rewards, prm_outputs


def main():
    # Load data
    generation_data = pd.read_csv('outputs_math_500-Q2.5-3B-Ins.csv')[:10]
    
    problems = generation_data['problem']
    
    chats = []
    for problem in problems:
        chats.append([
            {"role": "system", "content": "You are a math teacher. Your task is to review and critique the paragraphs in solution step by step. Pay attention that you should neither solve the problem nor give the final answer."},
            {"role": "user", "content": problem}
        ])
    
    drafts = generation_data['initial_response']
    
    # Split draft by paragraph as steps
    drafts = [draft.split('\n\n') for draft in drafts]
    for chat, draft in zip(chats, drafts):
        for para in draft:
            chat.append({"role": "user", "content": para})
            chat.append({"role": "assistant", "content": ''})
    
    # Load model
    print("Loading GenPRM model...")
    genprm = GenPRM('GenPRM/GenPRM-7B', 4)
    
    # Choose processing method
    use_batch = True  # Set to False to use original sequential method
    
    if use_batch:
        print("\n=== Using BATCH processing (faster) ===")
        # Batch processing - much faster
        batch_size = 64  # Adjust based on your GPU memory
        all_reward_list, all_process_rewards, prm_outputs = process_chats_batch(
            genprm, chats, batch_size=batch_size
        )
    else:
        print("\n=== Using SEQUENTIAL processing (slower) ===")
        # Original sequential processing
        all_reward_list = []
        all_process_rewards = []
        prm_outputs = []
        
        with tqdm(total=len(chats), desc='Generating Process Rewards') as pbar:
            for sol in chats:
                curr_reward = []
                curr_out = []
                for i in range(len(sol)):
                    if sol[i]['role'] != 'assistant':
                        continue
                    
                    output, reward = genprm.inference(
                        sol[:i], 
                        cur_step=int(i/2), 
                        verify=False, 
                        logging=False,
                        execute=False,
                        max_tokens=4096
                    )
                    sol[i]['content'] = output[0]
                    curr_reward.append(reward)
                    curr_out.append(output[0])
                
                # Store all process rewards and compute total reward
                all_process_rewards.append(curr_reward)
                all_reward_list.append(sum(curr_reward)/len(curr_reward))
                prm_outputs.append(curr_out)
                
                pbar.update(1)
    
    # Save results
    generation_data['process_reward'] = all_reward_list
    generation_data['prm_output'] = prm_outputs
    
    generation_data.to_csv('genprm_test_2.csv')
    print(f"\nResults saved to genprm_test_2.csv")
    print(f"Average reward score: {sum(all_reward_list)/len(all_reward_list):.4f}")


if __name__ == "__main__":
    main()