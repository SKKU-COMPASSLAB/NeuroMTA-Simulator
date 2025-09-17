import os
import pandas as pd


PROFILE_DIR = os.path.join(os.path.dirname(__file__), ".profiles")


if __name__ == "__main__":
    results = []
    
    for filename in os.listdir(PROFILE_DIR):
        filepath = os.path.join(PROFILE_DIR, filename)
        
        df = pd.read_csv(filepath)
        
        df['command_id'] = df['command_id'].str.strip()
        mxu_row = df[df['command_id'] == 'mxu_tiled_gemm']
        
        result = mxu_row['duration'].iloc[0] / mxu_row['last_commit_time'].iloc[0]
        results.append(result)

    print(f"Average MXU Utilization: {sum(results)/len(results)*100:.2f}%")
