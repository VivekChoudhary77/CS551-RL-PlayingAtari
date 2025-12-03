
import os
import argparse
from moviepy import VideoFileClip, concatenate_videoclips
import re

def numerical_sort(value):
    """
    Extracts the number from the filename for sorting.
    e.g., '...step-0-to-1000...' -> 0
    """
    numbers = re.findall(r'\d+', value)
    if len(numbers) > 0:
        # Usually the first large number in the filename is the step count
        # Our format is ...-step-X-to-step-Y...
        # We want to sort by X.
        try:
            # Find the number after "step-"
            match = re.search(r'step-(\d+)-to-step', value)
            if match:
                return int(match.group(1))
            return int(numbers[-1]) # Fallback
        except:
            return 0
    return 0

def merge_videos(video_folder, game, algo, output_filename):
    """
    Merges all video clips for a given game and algorithm into one file.
    """
    # Regex to match files like: eval_dqn_pong_seed0_final-step-0-to-step-1000.mp4
    # Pattern: eval_{algo}_{game}_seed0_final-step-*.mp4
    # Note: game name in filename is usually lowercase.
    
    search_pattern = f"eval_{algo.lower()}_{game.lower()}_seed0"
    
    print(f"Searching for videos with pattern: {search_pattern} in {video_folder}")
    
    video_files = []
    for f in os.listdir(video_folder):
        if f.endswith(".mp4") and search_pattern in f and "merged" not in f:
            video_files.append(os.path.join(video_folder, f))
    
    if not video_files:
        print(f"No videos found for {algo} on {game}.")
        return

    # Sort files by step number so the video plays chronologically
    video_files.sort(key=numerical_sort)
    
    print(f"Found {len(video_files)} clips. Merging...")
    for v in video_files:
        print(f"  - {os.path.basename(v)}")

    try:
        clips = [VideoFileClip(f) for f in video_files]
        final_clip = concatenate_videoclips(clips, method="compose")
        
        output_path = os.path.join(video_folder, output_filename)
        final_clip.write_videofile(output_path, codec='libx264', audio=False, fps=30)
        
        print(f"Successfully saved merged video to: {output_path}")
        
        # Close clips to release resources
        for clip in clips:
            clip.close()
        final_clip.close()
        
    except Exception as e:
        print(f"Error merging videos: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge evaluation video clips into a single video.")
    parser.add_argument("--video_folder", type=str, default="videos", help="Path to video folder")
    parser.add_argument("--game", type=str, required=True, help="Game name (Pong or BeamRider)")
    parser.add_argument("--algo", type=str, required=True, help="Algorithm name (DQN, A2C, PPO)")
    parser.add_argument("--output", type=str, default=None, help="Output filename (optional)")
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f"Merged_{args.algo.upper()}_{args.game.upper()}.mp4"
        
    merge_videos(args.video_folder, args.game, args.algo, args.output)
