from huggingface_hub import dataset_info

try:
    print("Fetching dataset info for ai4bharat/Kathbath...")
    info = dataset_info("ai4bharat/Kathbath")
    
    print(f"Found {len(info.siblings)} files.")
    
    # Filter for tar files or 'data' folder
    candidates = [f.rfilename for f in info.siblings if "tar" in f.rfilename or "hindi" in f.rfilename]
    
    print("--- Candidate Files ---")
    for c in candidates[:50]:
        print(c)
        
except Exception as e:
    print(f"Error: {e}")
