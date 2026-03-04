from huggingface_hub import list_repo_files

try:
    print("Listing files in ai4bharat/Kathbath...")
    files = list_repo_files(repo_id="ai4bharat/Kathbath", repo_type="dataset")
    
    print(f"Found {len(files)} files.")
    print("--- Top 20 Files ---")
    for f in files[:20]:
        print(f)
        
    print("\n--- Searching for 'tar' files ---")
    tar_files = [f for f in files if "tar" in f]
    for f in tar_files[:20]:
        print(f)
        
except Exception as e:
    print(f"Error: {e}")
