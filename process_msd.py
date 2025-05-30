import os
import json
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
from typing import Tuple, List, Dict
import h5py
from pathlib import Path

class AudioProcessor:
    def __init__(self, 
                 data_path: str, 
                 output_path: str, 
                 clip_duration: float = 5.0,
                 max_songs: int = 1000,
                 supported_formats: List[str] = None):
        """
        Initialize the audio processor
        
        Args:
            data_path: Path to the audio dataset
            output_path: Path to save processed clips
            clip_duration: Duration of each clip in seconds
            max_songs: Maximum number of songs to process
            supported_formats: List of supported audio file formats
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.clip_duration = clip_duration
        self.sample_rate = 22050  # Standard sample rate
        self.max_songs = max_songs
        self.supported_formats = supported_formats or ['.mp3', '.wav', '.ogg', '.flac']
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "clips").mkdir(exist_ok=True)
        (self.output_path / "pairs").mkdir(exist_ok=True)
        
    def process_song(self, song_path: Path) -> List[Dict]:
        """
        Process a single song into 5-second clips
        
        Args:
            song_path: Path to the song file
            
        Returns:
            List of dictionaries containing clip information
        """
        try:
            # Load audio file
            audio, sr = librosa.load(song_path, sr=self.sample_rate)
            
            # Calculate number of possible clips
            samples_per_clip = int(self.clip_duration * sr)
            num_clips = len(audio) // samples_per_clip
            
            clips = []
            for i in range(num_clips):
                start = i * samples_per_clip
                end = start + samples_per_clip
                clip = audio[start:end]
                
                # Save clip
                clip_path = self.output_path / "clips" / f"{song_path.stem}_{i}.npy"
                np.save(clip_path, clip)
                
                clips.append({
                    "path": str(clip_path),
                    "song_id": song_path.stem,
                    "clip_index": i,
                    "start_time": start / sr,
                    "end_time": end / sr
                })
                
            return clips
            
        except Exception as e:
            print(f"Error processing {song_path}: {e}")
            return []
    
    def create_pairs(self, all_clips: List[Dict], num_pairs: int = 1000) -> List[Dict]:
        """
        Create pairs of clips with different relationships
        
        Args:
            all_clips: List of all processed clips
            num_pairs: Number of pairs to create
            
        Returns:
            List of dictionaries containing pair information
        """
        pairs = []
        
        # Group clips by song
        song_clips = {}
        for clip in all_clips:
            if clip["song_id"] not in song_clips:
                song_clips[clip["song_id"]] = []
            song_clips[clip["song_id"]].append(clip)
        
        # Create pairs
        for _ in tqdm(range(num_pairs)):
            # Randomly choose relationship type
            rel_type = random.choice(["adjacent", "same_song", "different_song"])
            
            if rel_type == "adjacent":
                # Choose a random song with at least 2 clips
                song_id = random.choice([s for s in song_clips if len(song_clips[s]) >= 2])
                clips = song_clips[song_id]
                idx = random.randint(0, len(clips) - 2)
                clip1 = clips[idx]
                clip2 = clips[idx + 1]
                
            elif rel_type == "same_song":
                # Choose a random song with at least 2 clips
                song_id = random.choice([s for s in song_clips if len(song_clips[s]) >= 2])
                clips = song_clips[song_id]
                clip1, clip2 = random.sample(clips, 2)
                
            else:  # different_song
                # Choose two different songs
                song_ids = random.sample(list(song_clips.keys()), 2)
                clip1 = random.choice(song_clips[song_ids[0]])
                clip2 = random.choice(song_clips[song_ids[1]])
            
            pairs.append({
                "clip1": clip1,
                "clip2": clip2,
                "relationship": rel_type
            })
        
        return pairs
    
    def process_dataset(self):
        """
        Process the dataset up to max_songs
        """
        all_clips = []
        processed_songs = 0
        
        # Get all audio files
        audio_files = []
        for fmt in self.supported_formats:
            audio_files.extend(list(self.data_path.glob(f"**/*{fmt}")))
        
        # Process songs up to max_songs
        for song_path in tqdm(audio_files[:self.max_songs]):
            clips = self.process_song(song_path)
            all_clips.extend(clips)
            processed_songs += 1
            
            if processed_songs >= self.max_songs:
                break
        
        print(f"Processed {processed_songs} songs")
        
        # Create pairs
        pairs = self.create_pairs(all_clips)
        
        # Save pairs metadata
        with open(self.output_path / "pairs" / "metadata.json", "w") as f:
            json.dump(pairs, f, indent=2)
        
        return all_clips, pairs

class AudioDataset(Dataset):
    def __init__(self, pairs_path: str):
        """
        Initialize the dataset
        
        Args:
            pairs_path: Path to the pairs metadata file
        """
        with open(pairs_path, "r") as f:
            self.pairs = json.load(f)
        
        # Create relationship to index mapping
        self.relationship_map = {
            "adjacent": 0,
            "same_song": 1,
            "different_song": 2
        }
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load clips
        clip1 = np.load(pair["clip1"]["path"])
        clip2 = np.load(pair["clip2"]["path"])
        
        # Convert to torch tensors
        clip1 = torch.from_numpy(clip1).float()
        clip2 = torch.from_numpy(clip2).float()
        
        # Get relationship label
        label = self.relationship_map[pair["relationship"]]
        
        return clip1, clip2, label

if __name__ == "__main__":
    # Example usage with GTZAN dataset
    processor = AudioProcessor(
        data_path="/path/to/gtzan/genres",  # Path to your dataset
        output_path="/path/to/output",
        max_songs=100,  # Process only 100 songs
        supported_formats=['.wav']  # GTZAN uses .wav files
    )
    
    all_clips, pairs = processor.process_dataset()
    
    # Create dataset
    dataset = AudioDataset("/path/to/output/pairs/metadata.json")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    ) 