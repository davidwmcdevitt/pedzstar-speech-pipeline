import os
from pydub import AudioSegment
import pandas as pd
import torch
from torch.utils.data import random_split
from utils import SoundDS
import numpy as np 
from pydub.silence import split_on_silence
import random

class AudioDataset:
    
    def __init__(self, args, num_classes):
        
        self.num_classes = num_classes
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
        
        child_ds_len = len(os.listdir(self.data_path + 'children/'))
        print(f'Child Obs: {child_ds_len}')
        
        for directory_path in adult_dirs:
            
            count = 0
        
            for root, dirs, files in os.walk(directory_path):
                for file_name in files:
                  if file_name.endswith(".wav"):
          
                    audio = AudioSegment.from_wav(directory_path + file_name)
          
                    num_increments = len(audio) // clip_duration
                    increments = [audio[i * clip_duration : (i + 1) * clip_duration] for i in range(num_increments)]
          
                    for i, increment in enumerate(increments):
                      output_file = self.data_path + f'adults/{os.path.splitext(file_name)[0]}_{i + 1}.wav'
                      increment.export(output_file, format='wav')
                      count +=1
                      
                    if directory_path == self.data_path + 'cv_adults/' and count >= child_ds_len:
                        break
         
        if args.overlap_class:
            
            child_file_list = os.listdir(self.data_path + 'children/')
            adult_file_list = os.listdir(self.data_path + 'adults/')
            
            children_mix = random.sample(child_file_list, int(np.ceil(len(child_file_list)*0.10)))
            
            adult_mix = random.sample(adult_file_list, int(np.ceil(len(adult_file_list)*0.10)))
            
            children_mix_audio = AudioSegment.silent(duration=0)
            adult_mix_audio = AudioSegment.silent(duration=0)
                          
            for i in children_mix:
                
                audio = AudioSegment.from_wav(self.data_path + 'children/' + i)
                segments = split_on_silence(audio, min_silence_len=50, silence_thresh=-30)
                if len(segments) <= 5 & len(segments)>0:
                  for segment in segments:
                    children_mix_audio += segment
                os.remove(self.data_path + 'children/' + i)
            
            
            for i in adult_mix:
                
              audio = AudioSegment.from_wav(self.data_path + 'adults/' + i)
              segments = split_on_silence(audio, min_silence_len=50, silence_thresh=-30)
              if len(segments) <= 5 & len(segments)>0:
                for segment in segments:
                  adult_mix_audio += segment
              os.remove(self.data_path + 'adults/' + i)
              
            for i in range(int(np.ceil(child_ds_len *0.85))):
            
              max_start_time = len(children_mix_audio) - clip_duration
            
              child_start_time = random.randint(0, max_start_time)
            
              max_start_time = len(adult_mix_audio) - clip_duration
            
              adult_start_time = random.randint(0, max_start_time)
            
              child_clip = children_mix_audio[child_start_time:child_start_time + clip_duration]
              adult_clip = adult_mix_audio[adult_start_time:adult_start_time + clip_duration]
              
              if adult_clip.sample_width != child_clip.sample_width:
                  adult_clip = adult_clip.set_sample_width(child_clip.sample_width)
            
              overlayed_samples = (np.array(child_clip.get_array_of_samples()) + np.array(adult_clip.get_array_of_samples())) // 2
            
              overlayed_audio = AudioSegment(
                  data=overlayed_samples.tobytes(),
                  frame_rate=child_clip.frame_rate,
                  sample_width=child_clip.sample_width,
                  channels=child_clip.channels
              )
              
              output_file = self.data_path + f'mixed/mixed_{i + 1}.wav'
              overlayed_audio.export(output_file, format='wav')
              
        if args.transition_class:
            
            child_file_list = os.listdir(self.data_path + 'children/')
            adult_file_list = os.listdir(self.data_path + 'adults/')
            
            children_trans = random.sample(child_file_list, int(np.ceil(len(child_file_list)*0.10)))
            
            adult_trans = random.sample(adult_file_list, int(np.ceil(len(adult_file_list)*0.10)))
            
            count = 0
              
            for i in range(int(np.ceil(child_ds_len * 0.85))):
                
                child_file = random.choice(children_trans)
                    
                adult_file = random.choice(adult_trans)
                
                child_clip = AudioSegment.from_wav(self.data_path + 'children/' + child_file)
                    
                adult_clip = AudioSegment.from_wav(self.data_path + 'adults/' + adult_file)
                
                if adult_clip.sample_width != child_clip.sample_width:
                    adult_clip = adult_clip.set_sample_width(child_clip.sample_width)
                    
                cushion = random.randint(10,100) * 2
                
                speaker_time = int((1000 - cushion) / 2)
                
                empty_space = AudioSegment.silent(duration=cushion)
                
                if random.randint(0,1) == 0:
                    
                    transition_audio = child_clip[0:speaker_time] + empty_space + adult_clip[0:speaker_time]
                    
                else:
                    
                    transition_audio = adult_clip[0:speaker_time] + empty_space + child_clip[0:speaker_time]
                    
                assert len(transition_audio) == 1000
                
                output_file = self.data_path + f'transitions/transition_{count + 1}.wav'
                
                count += 1
                
                transition_audio.export(output_file, format='wav')
                
            for child_file in children_trans:
                os.remove(self.data_path + 'children/' + child_file)
                
            for adult_file in adult_trans:
                os.remove(self.data_path + 'adults/' + adult_file)
            
        
        relative_paths = []
        classIDs = []
        
        directories = [self.data_path + 'adults/',self.data_path + 'children/']
        
        if args.transition_class:
            directories.append(self.data_path + 'transitions/')
        
        if args.overlap_class:
            directories.append(self.data_path + 'mixed/')
        
        for directory_path in directories:
        
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    
                    if 'adult' in file:
                        relative_path = 'adults/'+file
                        classID = 'adults'
                    if 'child' in file:
                        relative_path = 'children/'+file
                        classID = 'children'
                    if 'mixed' in file:
                        relative_path = 'mixed/'+file
                        classID = 'mixed'
                    if 'transition' in file:
                        relative_path = 'transitions/'+file
                        classID = 'transition'
                        
                    relative_paths.append(relative_path)
                    classIDs.append(classID)
        
        df = pd.DataFrame({'relative_path': relative_paths, 'classID': classIDs})
        
        ds = SoundDS(df, self.data_path, args)
        
        num_items = len(ds)
        num_train = round(num_items * self.train_split)
        num_val = num_items - num_train
        train_ds, val_ds = random_split(ds, [num_train, num_val])
        
        self.train_dl = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_dl = torch.utils.data.DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        
        if args.class_weights:
            total_instances = df['classID'].count()
            class_counts = df['classID'].value_counts()
            
            class_weights = torch.zeros(self.num_classes, dtype=torch.float32)
            
            for i, class_name in enumerate(class_counts.index):
                instances = class_counts[class_name]
                weight = total_instances / (instances * self.num_classes)
                class_weights[i] = weight
                
            total_sum = torch.sum(class_weights)
            
            normalized_tensor = class_weights / total_sum
            
            self.class_weights = normalized_tensor
            print(normalized_tensor)
    
    
      
    
    
        
    