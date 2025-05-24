import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm  # For progress bars

def load_spotify_data(data_dir, max_files=None):
    """
    Load Spotify Million Playlist Dataset from JSON files.
    
    Parameters:
    - data_dir: Directory containing the mpd.slice.*.json files
    - max_files: Maximum number of files to process (for testing)
    
    Returns:
    - DataFrame with playlist information
    """
    print(f"Loading data from {data_dir}...")
    
    # Find all slice files
    slice_files = sorted(glob.glob(os.path.join(data_dir, "mpd.slice.*.json")))
    
    if max_files is not None:
        slice_files = slice_files[:max_files]
    
    print(f"Found {len(slice_files)} slice files")
    
    all_playlists = []
    
    # Load each slice file
    for file_path in tqdm(slice_files, desc="Loading files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Extract playlists from this slice
            for playlist in data['playlists']:
                # Extract basic playlist info
                playlist_info = {
                    'pid': playlist['pid'],
                    'name': playlist['name'],
                    'followers': playlist['num_followers'],
                    'num_tracks': playlist['num_tracks'],
                    'num_albums': playlist['num_albums'],
                    'collaborative': playlist['collaborative'] == 'true'
                }
                
                # Add to our collection
                all_playlists.append(playlist_info)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_playlists)
    print(f"Loaded {len(df)} playlists")
    
    return df

def analyze_follower_distribution(df, follower_col='followers'):
    """Analyze the distribution of followers in the dataset."""
    print(f"\nFollower Distribution Analysis:")
    print(f"Total playlists: {len(df)}")
    print(f"Min followers: {df[follower_col].min()}")
    print(f"Max followers: {df[follower_col].max()}")
    print(f"Mean followers: {df[follower_col].mean():.2f}")
    print(f"Median followers: {df[follower_col].median()}")
    
    # Follower distribution
    counts = Counter(df[follower_col])
    print(f"\nFollower counts:")
    for followers, count in sorted(counts.items())[:10]:
        print(f"{followers} followers: {count} playlists")
    print("...")
    for followers, count in sorted(counts.items())[-5:]:
        print(f"{followers} followers: {count} playlists")
    
    # Plot the distribution
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Histogram of follower counts (log scale)
    plt.subplot(1, 2, 1)
    plt.hist(df[follower_col], bins=50)
    plt.yscale('log')
    plt.title('Follower Distribution (Log Scale Y-Axis)')
    plt.xlabel('Follower Count')
    plt.ylabel('Number of Playlists (log)')
    
    # Plot 2: Histogram of log(follower counts)
    plt.subplot(1, 2, 2)
    plt.hist(np.log1p(df[follower_col]), bins=50)
    plt.title('Log(Follower) Distribution')
    plt.xlabel('Log(Follower Count + 1)')
    plt.ylabel('Number of Playlists')
    
    plt.tight_layout()
    plt.savefig('follower_distribution.png')
    plt.close()
    
    return counts


def create_min_size_follower_buckets(df, min_bucket_size=3000, follower_col='followers'):
    """
    Create follower buckets where EVERY bucket has at least min_bucket_size playlists.
    This ensures you can sample evenly from all buckets.
    """
    # Get the count of playlists for each follower count, sorted
    follower_distribution = df[follower_col].value_counts().sort_index()
    
    # Create buckets by accumulating follower counts until we reach minimum size
    bucket_ranges = []
    current_bucket_followers = []
    current_bucket_size = 0
    
    for follower_value, count in follower_distribution.items():
        current_bucket_followers.append(follower_value)
        current_bucket_size += count
        
        # If we've reached the minimum size, close this bucket
        if current_bucket_size >= min_bucket_size:
            bucket_ranges.append((current_bucket_followers[0], current_bucket_followers[-1]))
            current_bucket_followers = []
            current_bucket_size = 0
    
    # Handle remaining items - merge with the last bucket to ensure minimum size
    if current_bucket_followers:
        if bucket_ranges:
            # Extend the last bucket to include remaining followers
            last_bucket_min = bucket_ranges[-1][0]
            bucket_ranges[-1] = (last_bucket_min, current_bucket_followers[-1])
        else:
            # This shouldn't happen if total data > min_bucket_size, but handle it
            bucket_ranges.append((current_bucket_followers[0], current_bucket_followers[-1]))
    
    # Assign buckets to dataframe
    df['follower_bucket'] = None
    for i, (min_val, max_val) in enumerate(bucket_ranges):
        mask = (df[follower_col] >= min_val) & (df[follower_col] <= max_val)
        df.loc[mask, 'follower_bucket'] = i
    
    # Analyze and display buckets
    print("\nFollower Bucket Distribution (Minimum Size Guaranteed):")
    for i in range(len(bucket_ranges)):
        bucket_df = df[df['follower_bucket'] == i]
        count = len(bucket_df)
        min_followers = bucket_df[follower_col].min()
        max_followers = bucket_df[follower_col].max()
        print(f"Bucket {i}: {count} playlists, Followers range: {min_followers} - {max_followers}")
        
        # Warning if bucket is still too small
        if count < min_bucket_size:
            print(f"  WARNING: Bucket {i} has only {count} playlists (less than {min_bucket_size})")
    
    return df

def create_custom_follower_buckets(df, target_bucket_size=3000, follower_col='followers'):
    # Get the count of playlists for each follower count
    follower_counts = df[follower_col].value_counts().sort_index()
    
    # Initialize buckets
    buckets = []
    current_bucket = []
    current_bucket_size = 0
    current_bucket_min = follower_counts.index[0]
    bucket_ranges = []
    
    # Iterate through unique follower counts
    for follower_value, count in follower_counts.items():
        # If adding this follower count would exceed target size AND we already have some items,
        # complete the current bucket unless it would create a tiny bucket (less than 1/2 target size)
        if current_bucket_size > 0 and current_bucket_size + count > target_bucket_size and current_bucket_size >= target_bucket_size/2:
            # Complete the current bucket
            bucket_max = current_bucket[-1]
            bucket_ranges.append((current_bucket_min, bucket_max)) 
            current_bucket_min = follower_value
            current_bucket = [follower_value]
            current_bucket_size = count
        else:
            # Add this follower count to the current bucket
            current_bucket.append(follower_value)
            current_bucket_size += count
    
    # Add the last bucket if not empty
    if current_bucket:
        bucket_ranges.append((current_bucket_min, current_bucket[-1]))
    
    # Now assign buckets to the dataframe
    df['follower_bucket'] = -1  # Default value
    for i, (min_val, max_val) in enumerate(bucket_ranges):
        df.loc[(df[follower_col] >= min_val) & (df[follower_col] <= max_val), 'follower_bucket'] = i
    
    # Analyze the buckets
    bucket_counts = df['follower_bucket'].value_counts().sort_index()
    print("\nCustom Follower Bucket Distribution:")
    for bucket, count in bucket_counts.items():
        bucket_df = df[df['follower_bucket'] == bucket]
        print(f"Bucket {bucket}: {count} playlists, " +
              f"Followers range: {bucket_df[follower_col].min()} - {bucket_df[follower_col].max()}")
    
    return df

# Alternative version with more precise bucket size control
def create_follower_buckets_with_target_size(df, target_bucket_size=3000, follower_col='followers'):
    """
    Creates follower buckets with more precise control over target bucket size.
    This ensures unique follower counts are kept together while trying to hit the target size.
    """
    # Get distinct follower values and their counts, sorted
    follower_distribution = df[follower_col].value_counts().sort_index()
    
    # Define buckets by grouping follower values
    bucket_ranges = []
    current_bucket = []
    current_size = 0
    
    for follower_value, count in follower_distribution.items():
        # If this is a very large follower count (significantly exceeds target), it gets its own bucket
        if count > 2 * target_bucket_size:
            # If we have an existing bucket building, finalize it first
            if current_bucket:
                bucket_ranges.append((current_bucket[0], current_bucket[-1]))
                current_bucket = []
                current_size = 0
            
            # Create a standalone bucket for this large follower value
            bucket_ranges.append((follower_value, follower_value))
            continue
            
        # If adding this follower would make bucket too big and we already have items,
        # complete current bucket (unless it would create a tiny bucket)
        if current_bucket and current_size + count > target_bucket_size * 1.2 and current_size >= target_bucket_size * 0.5:
            bucket_ranges.append((current_bucket[0], current_bucket[-1]))
            current_bucket = [follower_value]
            current_size = count
        else:
            # Add to current bucket
            if not current_bucket:
                current_bucket = [follower_value]
            else:
                current_bucket.append(follower_value)
            current_size += count
    
    # Add the last bucket if not empty
    if current_bucket:
        bucket_ranges.append((current_bucket[0], current_bucket[-1]))
    
    # Assign buckets to dataframe
    df['follower_bucket'] = None
    for i, (min_val, max_val) in enumerate(bucket_ranges):
        mask = (df[follower_col] >= min_val) & (df[follower_col] <= max_val)
        df.loc[mask, 'follower_bucket'] = i
    
    # Analyze and display buckets
    bucket_stats = df.groupby('follower_bucket').agg({
        follower_col: ['min', 'max', 'count']
    })
    
    print("\nFollower Bucket Distribution:")
    for bucket, (min_val, max_val, count) in bucket_stats.iterrows():
        print(f"Bucket {bucket}: {count} playlists, Followers range: {min_val} - {max_val}")
    
    return df

def sample_evenly_from_buckets(df, sample_size_per_bucket=3000, bucket_col='follower_bucket', random_state=42):
    """
    Sample evenly from each bucket, ensuring each bucket contributes the same number of samples.
    """
    sampled_dfs = []
    
    print(f"\nSampling {sample_size_per_bucket} playlists from each bucket:")
    
    for bucket in sorted(df[bucket_col].unique()):
        bucket_df = df[df[bucket_col] == bucket]
        bucket_size = len(bucket_df)
        
        if bucket_size < sample_size_per_bucket:
            print(f"WARNING: Bucket {bucket} has only {bucket_size} playlists, can't sample {sample_size_per_bucket}")
            sampled_dfs.append(bucket_df)  # Take all available
        else:
            sampled = bucket_df.sample(n=sample_size_per_bucket, random_state=random_state)
            sampled_dfs.append(sampled)
            print(f"Bucket {bucket}: sampled {len(sampled)} from {bucket_size} playlists")
    
    result_df = pd.concat(sampled_dfs, ignore_index=True)
    
    print(f"\nFinal balanced dataset: {len(result_df)} playlists")
    print("Balanced Bucket Counts:")
    print(result_df[bucket_col].value_counts().sort_index())
    
    return result_df

def stratified_sampling(df, method='oversample', n_buckets=10, target_bucket_size=None, 
                       min_followers=None, max_per_bucket=None, follower_col='followers'):
    """
    Perform stratified sampling of the dataset to balance follower distributions.
    
    Parameters:
    - df: DataFrame with playlist features and follower counts
    - method: 'oversample', 'undersample', or 'hybrid'
    - n_buckets: Number of follower buckets to create
    - target_bucket_size: Target number of samples in each bucket (for 'hybrid')
    - min_followers: Minimum number of followers for a playlist to be included
    - max_per_bucket: Maximum number of samples to take from each bucket
    
    Returns:
    - Balanced DataFrame
    """
    # Filter by minimum followers if specified
    if min_followers is not None:
        print(f"\nFiltering playlists with at least {min_followers} followers...")
        original_count = len(df)
        df = df[df[follower_col] >= min_followers]
        print(f"Retained {len(df)} out of {original_count} playlists.")
    
    # Create follower buckets
    # df = create_custom_follower_buckets(df, follower_col=follower_col)
    df = create_min_size_follower_buckets(df, min_bucket_size=3000)
    balanced_df = sample_evenly_from_buckets(df, sample_size_per_bucket=3000)
    
    if method == 'oversample':
        # Oversample from buckets with fewer samples
        balanced_df = oversample_buckets(df, follower_col)
    elif method == 'undersample':
        # Undersample from buckets with more samples
        balanced_df = undersample_buckets(df, max_per_bucket, follower_col)
    elif method == 'hybrid':
        # Hybrid approach: oversample smaller buckets, undersample larger buckets
        balanced_df = hybrid_sampling(df, target_bucket_size, follower_col)
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    # Remove bucket columns used for sampling
    balanced_df = balanced_df.drop(columns=['follower_bucket', 'log_followers'], errors='ignore')
    
    return balanced_df

def oversample_buckets(df, follower_col='followers'):
    """Oversample from buckets with fewer samples to match the largest bucket."""
    bucket_counts = df['follower_bucket'].value_counts()
    max_bucket_size = bucket_counts.max()
    
    print(f"\nOversampling to {max_bucket_size} samples per bucket...")
    
    balanced_dfs = []
    for bucket in sorted(df['follower_bucket'].unique()):
        bucket_df = df[df['follower_bucket'] == bucket]
        n_samples = len(bucket_df)
        
        # If bucket has fewer samples than the largest bucket, oversample
        if n_samples < max_bucket_size:
            # Random sampling with replacement
            oversampled = bucket_df.sample(n=max_bucket_size, replace=True, random_state=42)
            balanced_dfs.append(oversampled)
        else:
            balanced_dfs.append(bucket_df)
    
    balanced_df = pd.concat(balanced_dfs)
    print(f"Oversampled dataset size: {len(balanced_df)}")
    
    # Verify the balance
    new_bucket_counts = balanced_df['follower_bucket'].value_counts().sort_index()
    print("\nBalanced Bucket Counts:")
    print(new_bucket_counts)
    
    return balanced_df

def undersample_buckets(df, max_per_bucket=None, follower_col='followers'):
    """Undersample from buckets with more samples to match the smallest bucket."""
    bucket_counts = df['follower_bucket'].value_counts()
    
    if max_per_bucket is None:
        max_per_bucket = bucket_counts.min()
    
    print(f"\nUndersampling to {max_per_bucket} samples per bucket...")
    
    balanced_dfs = []
    for bucket in sorted(df['follower_bucket'].unique()):
        bucket_df = df[df['follower_bucket'] == bucket]
        n_samples = len(bucket_df)
        
        # If bucket has more samples than target, undersample
        if n_samples > max_per_bucket:
            # Random sampling without replacement
            undersampled = bucket_df.sample(n=max_per_bucket, replace=False, random_state=42)
            balanced_dfs.append(undersampled)
        else:
            balanced_dfs.append(bucket_df)
    
    balanced_df = pd.concat(balanced_dfs)
    print(f"Undersampled dataset size: {len(balanced_df)}")
    
    # Verify the balance
    new_bucket_counts = balanced_df['follower_bucket'].value_counts().sort_index()
    print("\nBalanced Bucket Counts:")
    print(new_bucket_counts)
    
    return balanced_df

def hybrid_sampling(df, target_bucket_size=None, follower_col='followers'):
    """Hybrid approach: oversample smaller buckets, undersample larger buckets."""
    bucket_counts = df['follower_bucket'].value_counts()
    
    if target_bucket_size is None:
        target_bucket_size = int(bucket_counts.mean())
    
    print(f"\nHybrid sampling to {target_bucket_size} samples per bucket...")
    
    balanced_dfs = []
    for bucket in sorted(df['follower_bucket'].unique()):
        bucket_df = df[df['follower_bucket'] == bucket]
        n_samples = len(bucket_df)
        
        if n_samples > target_bucket_size:
            # Undersample
            sampled = bucket_df.sample(n=target_bucket_size, replace=False, random_state=42)
        elif n_samples < target_bucket_size:
            # Oversample
            sampled = bucket_df.sample(n=target_bucket_size, replace=True, random_state=42)
        else:
            sampled = bucket_df
            
        balanced_dfs.append(sampled)
    
    balanced_df = pd.concat(balanced_dfs)
    print(f"Hybrid sampled dataset size: {len(balanced_df)}")
    
    # Verify the balance
    new_bucket_counts = balanced_df['follower_bucket'].value_counts().sort_index()
    print("\nBalanced Bucket Counts:")
    print(new_bucket_counts)
    
    return balanced_df

def compare_distributions(original_df, balanced_df, follower_col='followers'):
    """Compare the follower distributions before and after balancing."""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Original histogram (log scale)
    plt.subplot(2, 2, 1)
    plt.hist(original_df[follower_col], bins=50, alpha=0.7)
    plt.yscale('log')
    plt.title('Original Follower Distribution')
    plt.xlabel('Follower Count')
    plt.ylabel('Number of Playlists (log)')
    
    # Plot 2: Balanced histogram (log scale)
    plt.subplot(2, 2, 2)
    plt.hist(balanced_df[follower_col], bins=50, alpha=0.7)
    plt.yscale('log')
    plt.title('Balanced Follower Distribution')
    plt.xlabel('Follower Count')
    plt.ylabel('Number of Playlists (log)')
    
    # Plot 3: Original log(followers) histogram
    plt.subplot(2, 2, 3)
    plt.hist(np.log1p(original_df[follower_col]), bins=50, alpha=0.7)
    plt.title('Original Log(Follower) Distribution')
    plt.xlabel('Log(Follower Count + 1)')
    plt.ylabel('Number of Playlists')
    
    # Plot 4: Balanced log(followers) histogram
    plt.subplot(2, 2, 4)
    plt.hist(np.log1p(balanced_df[follower_col]), bins=50, alpha=0.7)
    plt.title('Balanced Log(Follower) Distribution')
    plt.xlabel('Log(Follower Count + 1)')
    plt.ylabel('Number of Playlists')
    
    plt.tight_layout()
    plt.savefig('follower_distribution_comparison.png')
    plt.close()
    
    # Display statistics
    print("\nDistribution Statistics Comparison:")
    print(f"Original - Mean: {original_df[follower_col].mean():.2f}, Median: {original_df[follower_col].median()}")
    print(f"Balanced - Mean: {balanced_df[follower_col].mean():.2f}, Median: {balanced_df[follower_col].median()}")

def save_balanced_data(balanced_df, sample_json_path, output_file):
    """
    Save the balanced dataset back in JSON format.
    
    Parameters:
    - balanced_df: DataFrame with balanced playlist data
    - sample_json_path: Path to a sample JSON file to use as a template
    - output_file: Path to save the balanced dataset
    """
    # First, create a mapping of pid to all attributes in balanced_df
    # playlist_mapping = balanced_df.set_index('pid').to_dict('index')
    
    # Load a sample JSON to use as a template
    with open(sample_json_path, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)
    
    # Create a new JSON structure with just the balanced playlists
    balanced_data = {
        "info": {
            "generated_on": sample_data["info"]["generated_on"],
            "version": sample_data["info"]["version"],
            "description": "Balanced dataset with stratified sampling by follower counts"
        },
        "playlists": []
    }
    
    # Find the original playlists that match our balanced set
    pids_to_keep = set(balanced_df['pid'])
    
    # Iterate through all slice files to find matching playlists
    slice_files = glob.glob(os.path.join(os.path.dirname(sample_json_path), "mpd.slice.*.json"))
    
    print(f"\nFinding original playlist data for {len(pids_to_keep)} balanced playlists...")
    
    playlists_found = 0
    for file_path in tqdm(slice_files, desc="Processing files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Check each playlist
            for playlist in data['playlists']:
                if playlist['pid'] in pids_to_keep:
                    balanced_data['playlists'].append(playlist)
                    playlists_found += 1
                    
                    # Remove from our set to track progress
                    pids_to_keep.remove(playlist['pid'])
                    
                    # Early exit if we've found all playlists
                    if not pids_to_keep:
                        break
        
        # Early exit if we've found all playlists
        if not pids_to_keep:
            break
    
    print(f"Found {playlists_found} of {len(balanced_df)} playlists")
    
    # Save the balanced dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(balanced_data, f, indent=2)
    
    print(f"Balanced dataset saved to {output_file}")
    
    # Also save as CSV for easier analysis
    csv_output = output_file.replace('.json', '.csv')
    balanced_df.to_csv(csv_output, index=False)
    print(f"Balanced dataset summary saved to {csv_output}")

def main():
    # Directory containing the MPD slice files
    data_dir = "./spotify_dataset_files/data"
    
    # Load just the playlist metadata (not all tracks)
    df = load_spotify_data(data_dir, max_files=1000) 
    
    # Analyze original distribution
    analyze_follower_distribution(df, follower_col='followers')
    
    # 1. Filter out playlists with very few followers
    min_followers = 0  # Adjust based on your analysis
    filtered_df = df[df['followers'] >= min_followers]
    print(f"\nFiltered dataset size: {len(filtered_df)} (removed {len(df) - len(filtered_df)} playlists)")
    
    # 2. Hybrid approach
    balanced_df = stratified_sampling(
        filtered_df, 
        method='undersample',
        n_buckets=15,
        target_bucket_size=None,  # Adjust based on your dataset size
        follower_col='followers'
    )
    
    # Compare distributions
    compare_distributions(df, balanced_df)
    
    # Save the balanced dataset
    sample_json_path = os.path.join(data_dir, "mpd.slice.0-999.json")  # Use first slice as template
    save_balanced_data(balanced_df, sample_json_path, "balanced_spotify_playlists.json")

if __name__ == "__main__":
    main()