import os
from pydub import AudioSegment
import pandas as pd
import torch
from torch.utils.data import random_split
from utils import SoundDS
import numpy as np 

class AudioDataset:
    
    def __init__(self, args):
        
        self.create_dataset(args)
        
    def create_dataset(self, args):
        
        self.data_path = args.data_path
        self.repo_path = args.repo_path
        self.seed_only = args.seed_only
        self.num_epochs = args.num_epochs
        self.train_split = args.train_split
        self.break_count = args.break_count
        self.batch_size = args.batch_size
        self.clip_size = args.clip_size
        self.noise_level = args.noise_level
        self.overlap_class = args.overlap_class
        self.transition_class = args.transition_class
        
        print("Building datasets")
        
        if self.seed_only:
            adult_dirs = [self.data_path + 'seed_adults/']
            child_dirs = [self.data_path + 'seed_children/']
        else:
            adult_dirs = [self.data_path + 'seed_adults/', self.data_path + 'cv_adults/']
            child_dirs = [self.data_path + 'seed_children/', self.data_path + 'darcy_children/']
            
        clip_duration = 1000
        
        for directory_path in adult_dirs:
        
            for root, dirs, files in os.walk(directory_path):
                for file_name in files:
                  if file_name.endswith(".wav"):
          
                    audio = AudioSegment.from_wav(directory_path + file_name)
          
                    num_increments = len(audio) // clip_duration
                    increments = [audio[i * clip_duration : (i + 1) * clip_duration] for i in range(num_increments)]
          
                    for i, increment in enumerate(increments):
                      output_file = self.data_path + f'adults/{os.path.splitext(file_name)[0]}_{i + 1}.wav'
                      increment.export(output_file, format='wav')
                      
        for directory_path in child_dirs:
    
          for root, dirs, files in os.walk(directory_path):
              for file_name in files:
                if file_name.endswith(".wav"):
        
                  audio = AudioSegment.from_wav(directory_path + file_name)
        
                  num_increments = len(audio) // clip_duration
                  increments = [audio[i * clip_duration : (i + 1) * clip_duration] for i in range(num_increments)]
        
                  for i, increment in enumerate(increments):
                    output_file = self.data_path + f'children/{os.path.splitext(file_name)[0]}_{i + 1}.wav'
                    increment.export(output_file, format='wav')
        
        relative_paths = []
        subfolders = []
        
        directories = adult_dirs
        directories.append(child_dirs)
        
        for directory_path in directories:
        
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    
                    relative_path = os.path.relpath(os.path.join(root, file), start=directory_path)
                    subfolder = os.path.basename(root)
                    
                    relative_paths.append(relative_path)
                    subfolders.append(subfolder)
        
        df = pd.DataFrame({'relative_path': relative_paths, 'classID': subfolders})
        
        ds = SoundDS(df, self.data_path)
        
        num_items = len(ds)
        num_train = round(num_items * self.train_split)
        num_val = num_items - num_train
        train_ds, val_ds = random_split(ds, [num_train, num_val])
        
        self.train_dl = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_dl = torch.utils.data.DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        
        if args.class_weights:
            
            total_instances = df['classID'].count()
            adults_instances = df['classID'].value_counts()['adults']
            children_instances = total_instances - adults_instances
            
            weight_adults = adults_instances / (children_instances + adults_instances)
            weight_children = children_instances / (children_instances + adults_instances)
            
            self.class_weights = torch.tensor([weight_adults, weight_children], dtype=torch.float32)
    
    
      
    
    
        
    