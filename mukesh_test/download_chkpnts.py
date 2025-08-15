from huggingface_hub import hf_hub_download
import os

def download_huggingface_checkpoint(repo_id, filename, token, cache_dir=None):
    """
    Downloads a file from a Hugging Face model repository using a token.
    
    Parameters:
        repo_id (str): The model repo ID, e.g. "bert-base-uncased".
        filename (str): Name of the file to download (e.g. "pytorch_model.bin").
        token (str): Your Hugging Face access token.
        cache_dir (str, optional): Custom cache directory for downloads.
    
    Returns:
        str: Local path to the downloaded file.
    """
    os.environ["HF_TOKEN"] = token  # Set token for the current session

    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
        cache_dir=cache_dir
    )

    return file_path

# Example usage
if __name__ == "__main__":
    # Replace with your values
    token = "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" #aosong's read-only token
    filename = "pytorch_model.bin"

    # local_path = download_huggingface_checkpoint("afeng/Qwen2.5-SPO-7B-22", filename, token)
    # local_path = download_huggingface_checkpoint("afeng/Qwen2.5-GRPO-7B-22", filename, token)
    local_path = download_huggingface_checkpoint("afeng/Qwen2.5-14B-GRPO", filename, token)
    # local_path = download_huggingface_checkpoint("afeng/Qwen2.5-14B-SPO", filename, token)
    
    
    print(f"Downloaded file path: {local_path}")