import torch
import torch.nn as nn
import time
from sklearn.metrics import confusion_matrix, f1_score
import argparse
from model import AudioClassifier
from data import AudioDataset
from IPython.display import clear_output

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='PEDZSTAR Speech Pipeline Trainer')
    
    parser.add_argument('--data_path', type=str, required=True, help='Path to the training data')
    parser.add_argument('--seed_only', action='store_true', help='Designates experiment input as SEED data only ')
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
    
    model = AudioClassifier()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    next(model.parameters()).device
    print("here")
    '''
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
            #log training
            pass
    
        if acc > high_score:
          high_score = acc
          break_count = 0
          print("Saving model")
          model_name = data_path + ''
          torch.save(model.state_dict(), '/content/drive/MyDrive/PEDZSTAR/exp_full_2class_08292023.pth')
        else:
          break_count += 1
          if break_count > args.break_count:
            break
    
    print('Finished Training')
    '''