import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm  # For progress bars
import networkx as nx


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
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            # Extract playlists from this slice
            for playlist in data["playlists"]:
                # Extract basic playlist info
                playlist_info = {
                    "pid": playlist["pid"],
                    "name": playlist["name"],
                    "followers": playlist["num_followers"],
                    "num_tracks": playlist["num_tracks"],
                    "num_albums": playlist["num_albums"],
                    "collaborative": playlist["collaborative"] == "true",
                }

                # Add to our collection
                all_playlists.append(playlist_info)

    # Convert to DataFrame
    df = pd.DataFrame(all_playlists)
    print(f"Loaded {len(df)} playlists")

    return df


def analyze_follower_distribution(df, follower_col="followers"):
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
    plt.yscale("log")
    plt.title("Follower Distribution (Log Scale Y-Axis)")
    plt.xlabel("Follower Count")
    plt.ylabel("Number of Playlists (log)")

    # Plot 2: Histogram of log(follower counts)
    plt.subplot(1, 2, 2)
    plt.hist(np.log1p(df[follower_col]), bins=50)
    plt.title("Log(Follower) Distribution")
    plt.xlabel("Log(Follower Count + 1)")
    plt.ylabel("Number of Playlists")

    plt.tight_layout()
    plt.savefig("follower_distribution.png")
    plt.close()

    return counts


def create_balanced_binary_buckets(df, follower_col="followers", separator_followers=5):
    """
    Create two balanced follower buckets:
    1. All values >= separator_followers (complete data)
    2. Values < separator_followers (uniformly sampled to match bucket 1 size)

    This ensures balanced representation for neural network training.
    """
    import pandas as pd

    # Split data into two groups
    high_followers = df[df[follower_col] >= separator_followers]
    low_followers = df[df[follower_col] < separator_followers]

    print(f"Original distribution:")
    print(f"High followers (>= {separator_followers}): {len(high_followers)} samples")
    print(f"Low followers (< {separator_followers}): {len(low_followers)} samples")

    # (this will be our target size)
    target_size = len(high_followers)

    # If we don't have enough low follower samples, use all of them
    if len(low_followers) <= target_size:
        print(f"Warning: Not enough low follower samples to match high follower count.")
        print(f"Using all {len(low_followers)} low follower samples.")
        balanced_low_followers = low_followers
    else:
        # Uniformly sample from low followers to match high followers count
        balanced_low_followers = low_followers.sample(n=target_size, random_state=42)
        print(f"Sampled {target_size} low follower samples uniformly.")

    # Combine the balanced datasets
    balanced_df = pd.concat([high_followers, balanced_low_followers], ignore_index=True)

    balanced_df["follower_bucket"] = None
    for i, row in balanced_df.iterrows():
        if row[follower_col] >= separator_followers:
            balanced_df.loc[i, "follower_bucket"] = 0
        else:
            balanced_df.loc[i, "follower_bucket"] = 1

    # Analyze and display buckets
    bucket_counts = balanced_df["follower_bucket"].value_counts().sort_index()
    print("\nBalanced Bucket Distribution:")
    for bucket, count in bucket_counts.items():
        bucket_df = balanced_df[balanced_df["follower_bucket"] == bucket]
        print(
            f"Bucket {bucket}: {count} playlists, "
            + f"Followers range: {bucket_df[follower_col].min()} - {bucket_df[follower_col].max()}"
        )

    return balanced_df


def balanced_sampling(
    df,
    min_followers=None,
    follower_col="followers",
    separator_followers=5,
):
    # Filter by minimum followers if specified
    if min_followers is not None:
        print(f"\nFiltering playlists with at least {min_followers} followers...")
        original_count = len(df)
        df = df[df[follower_col] >= min_followers]
        print(f"Retained {len(df)} out of {original_count} playlists.")

    balanced_df = create_balanced_binary_buckets(
        df, separator_followers=separator_followers, follower_col=follower_col
    )

    # Remove bucket columns used for sampling
    balanced_df = balanced_df.drop(
        columns=["follower_bucket", "log_followers"], errors="ignore"
    )

    return balanced_df


def compare_distributions(original_df, balanced_df, follower_col="followers"):
    """Compare the follower distributions before and after balancing."""
    plt.figure(figsize=(15, 10))

    # Plot 1: Original histogram (log scale)
    plt.subplot(2, 2, 1)
    plt.hist(original_df[follower_col], bins=50, alpha=0.7)
    plt.yscale("log")
    plt.title("Original Follower Distribution")
    plt.xlabel("Follower Count")
    plt.ylabel("Number of Playlists (log)")

    # Plot 2: Balanced histogram (log scale)
    plt.subplot(2, 2, 2)
    plt.hist(balanced_df[follower_col], bins=50, alpha=0.7)
    plt.yscale("log")
    plt.title("Balanced Follower Distribution")
    plt.xlabel("Follower Count")
    plt.ylabel("Number of Playlists (log)")

    # Plot 3: Original log(followers) histogram
    plt.subplot(2, 2, 3)
    plt.hist(np.log1p(original_df[follower_col]), bins=50, alpha=0.7)
    plt.title("Original Log(Follower) Distribution")
    plt.xlabel("Log(Follower Count + 1)")
    plt.ylabel("Number of Playlists")

    # Plot 4: Balanced log(followers) histogram
    plt.subplot(2, 2, 4)
    plt.hist(np.log1p(balanced_df[follower_col]), bins=50, alpha=0.7)
    plt.title("Balanced Log(Follower) Distribution")
    plt.xlabel("Log(Follower Count + 1)")
    plt.ylabel("Number of Playlists")

    plt.tight_layout()
    plt.savefig("follower_distribution_comparison.png")
    plt.close()

    # Display statistics
    print("\nDistribution Statistics Comparison:")
    print(
        f"Original - Mean: {original_df[follower_col].mean():.2f}, Median: {original_df[follower_col].median()}"
    )
    print(
        f"Balanced - Mean: {balanced_df[follower_col].mean():.2f}, Median: {balanced_df[follower_col].median()}"
    )


def save_balanced_data(balanced_df, sample_json_path, output_file):
    # Load a sample JSON to use as a template
    with open(sample_json_path, "r", encoding="utf-8") as f:
        sample_data = json.load(f)

    # Create a new JSON structure with just the balanced playlists
    balanced_data = {
        "info": {
            "generated_on": sample_data["info"]["generated_on"],
            "version": sample_data["info"]["version"],
            "description": "Balanced dataset with stratified sampling by follower counts",
        },
        "playlists": [],
    }

    # Find the original playlists that match our balanced set
    pids_to_keep = set(balanced_df["pid"])

    # Iterate through all slice files to find matching playlists
    slice_files = glob.glob(
        os.path.join(os.path.dirname(sample_json_path), "mpd.slice.*.json")
    )

    print(
        f"\nFinding original playlist data for {len(pids_to_keep)} balanced playlists..."
    )

    playlists_found = 0
    for file_path in tqdm(slice_files, desc="Processing files"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            # Check each playlist
            for playlist in data["playlists"]:
                if playlist["pid"] in pids_to_keep:
                    balanced_data["playlists"].append(playlist)
                    playlists_found += 1

                    # Remove from our set to track progress
                    pids_to_keep.remove(playlist["pid"])

                    # Early exit if we've found all playlists
                    if not pids_to_keep:
                        break

        # Early exit if we've found all playlists
        if not pids_to_keep:
            break

    print(f"Found {playlists_found} of {len(balanced_df)} playlists")

    # Save the balanced dataset
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(balanced_data, f, indent=2)

    print(f"Balanced dataset saved to {output_file}")

    # Also save as CSV for easier analysis
    csv_output = output_file.replace(".json", ".csv")
    balanced_df.to_csv(csv_output, index=False)
    print(f"Balanced dataset summary saved to {csv_output}")


def main():
    # Directory containing the MPD slice files
    data_dir = "./spotify_dataset_files/data"

    # Load just the playlist metadata (not all tracks)
    df = load_spotify_data(data_dir, max_files=None)

    # Analyze original distribution
    analyze_follower_distribution(df, follower_col="followers")

    # create balanced dataset based on follower counts
    # separtor_followers = 10  # Define the threshold for separating binary buckets
    balanced_df = balanced_sampling(
        df,
        follower_col="followers",
        separator_followers=10,
    )

    compare_distributions(df, balanced_df)

    # Save the balanced dataset
    sample_json_path = os.path.join(
        data_dir, "mpd.slice.0-999.json"
    )  # Use first slice as template
    save_balanced_data(balanced_df, sample_json_path, "balanced_spotify_playlists.json")


if __name__ == "__main__":
    main()
