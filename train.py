import torch
import torch.nn as nn
import time
from sklearn.metrics import confusion_matrix, f1_score
import argparse
from model import AudioClassifier
from data import AudioDataset
from IPython.display import clear_output
import os
from datetime import datetime
import numpy as np

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='PEDZSTAR Speech Pipeline Trainer')
    
    parser.add_argument('--repo_path', type=str, required=True, help='Path to GitHub repo')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--model_name', type=str, required=True, help='Model Name')
    parser.add_argument('--seed_only', action='store_true', help='Designates experiment input as SEED data only')
    parser.add_argument('--cnn_model', action='store_true', default=True, help='Designates CNN architecture')
    parser.add_argument('--w2v_model', action='store_true', default=False, help='Designates W2V architecture')
    parser.add_argument('--num_epochs', type=int, default=999, help='Number of training epochs')
    parser.add_argument('--train_split', type=float, default=0.9, help='Train split percentage')
    parser.add_argument('--break_count', type=int, default=20, help='Maximum number of epochs without learning')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--clip_size', type=float, default=1, help='Size of training and inference clip segment (in seconds)')
    parser.add_argument('--noise_level', type=float, default=0.2, help='Percent noise mixed in during training')
    parser.add_argument('--overlap_class', action='store_true', help='Add overlap class')
    parser.add_argument('--transition_class', action='store_true', help='Add transition class')
    parser.add_argument('--class_weights', action='store_true', help='Optimize with class weights')
    parser.add_argument('--log_training', action='store_true', help='Log training accuracy and loss')
    
    args = parser.parse_args()
    
    dataset = AudioDataset(args)
    
    data_path = args.data_path
    checkpoint_name = data_path + args.model_name + '/' + datetime.now().strftime('%Y-%m-%d') + '.pth'
    print(checkpoint_name)
    
    if not os.path.exists(args.repo_path + args.model_name):
        os.makedirs(args.repo_path + args.model_name)
    
    assert os.path.exists(args.repo_path + args.model_name)
    
    if args.log_training:
        os.makedirs(args.repo_path + args.model_name + 'log/')
    
    num_classes = 2
    
    if args.overlap_class:
        num_classes +=1
        
    if args.transition_class:
        num_classes +=1
        
        
    if args.cnn_model == True and args.w2v_model == False:
        model = AudioClassifier(num_classes)
        
    if args.cnn_model == False and args.w2v_model == True:
        pass
        
    if args.cnn_model == True and args.w2v_model == True:
        pass
        
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    next(model.parameters()).device
    
    if args.class_weights:
        criterion = nn.CrossEntropyLoss(weight=dataset.class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                  steps_per_epoch=int(len(dataset.train_dl)),
                                                  epochs=args.num_epochs,
                                                  anneal_strategy='linear')
    high_score = 0
    break_count = 0
    
    
    if args.log_training:
        train_loss = []
        train_acc = []
        train_epoch_time =[]
        val_acc = []
        val_f1_score = []
    
    for epoch in range(args.num_epochs):
        
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0
    
        start_time = time.time()
        
        for i, data in enumerate(dataset.train_dl):
            
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            clear_output(wait=True)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            
            _, prediction = torch.max(outputs,1)
            
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
            
    
        end_time = time.time()
        
        epoch_time = end_time - start_time
    
        num_batches = len(dataset.train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
        
        if args.log_training:
            train_loss.append(avg_loss)
            train_acc.append(acc)
            train_epoch_time.append(epoch_time)
        
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}, Time: {epoch_time:.2f} seconds')
    
        print("-"*80)
    
        print("Validation")
    
        correct_prediction = 0
        total_prediction = 0
        
        with torch.no_grad():
          all_labels = []
          all_predictions = []
          for data in dataset.val_dl:
              
            inputs, labels = data[0].to(device), data[1].to(device)
            
            outputs = model(inputs)
    
            _, prediction = torch.max(outputs,1)
            
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
    
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(prediction.cpu().numpy())
    
        acc = correct_prediction / total_prediction
        print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')
        
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        print(f'Weighted F1 Score: {f1:.2f}')
        
        confusion = confusion_matrix(all_labels, all_predictions)
        print('Confusion Matrix:')
        print(confusion)
        
        if args.log_training:
            val_acc.append(avg_loss)
            val_f1_score.append(f1)
    
        if acc > high_score:
          high_score = acc
          break_count = 0
          print("Saving model")
          checkpoint_name = args.repo_path + args.model_name + '/' + datetime.now().strftime('%Y-%m-%d') + '.pth'
          torch.save(model.state_dict(), checkpoint_name)
        else:
          break_count += 1
          if break_count > args.break_count:
            break
    
    if args.log_training:
        print(f'Saving Logs at {args.repo_path + args.model_name}log/')
        np.save(args.repo_path + args.model_name + 'log/train_loss.npy', train_loss)
        np.save(args.repo_path + args.model_name + 'log/train_acc.npy', train_acc)
        np.save(args.repo_path + args.model_name + 'log/train_epoch_time.npy', train_epoch_time)
        np.save(args.repo_path + args.model_name + 'log/val_acc.npy', val_acc)
        np.save(args.repo_path + args.model_name + 'log/val_f1_score.npy', val_f1_score)
        
    print('Finished Training')
    