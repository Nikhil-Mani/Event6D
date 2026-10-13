from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mickeykang/Event6D",
    repo_type="dataset",
    local_dir="./data/Event6D"
)