from __future__ import absolute_import, print_function, division
import time
import torch
import numpy as np
from torchmetrics import F1Score
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F


def current_learning_rate(optimizer):
    return optimizer.state_dict()['param_groups'][0]['lr']


def current_weight_decay(optimizer):
    return optimizer.state_dict()['param_groups'][0]['weight_decay']


def save_checkpoint(checkpoint_file, epoch, model_state_dict, optimizer_state_dict):
    states = {'epoch': epoch+1, 'state_dict': model_state_dict, 'optimizer': optimizer_state_dict}
    torch.save(states, checkpoint_file)


def load_checkpoint(checkpoint_file, map_location='cpu'):
    checkpoint = torch.load(checkpoint_file, map_location=map_location)
    epoch, weights, optimizer = checkpoint['epoch'], checkpoint['state_dict'], checkpoint['optimizer']
    return epoch, weights, optimizer


def count_samples_per_class(batch, num_classes):
    class_counts = np.zeros(num_classes, dtype=int)
    for index in range(len(batch)):
        label = batch[index]
        class_counts[label] += 1
    return class_counts


########################################################################################################################
#####################################################   Training   #####################################################
########################################################################################################################


def train_epoch(model, criterion, optimizer, device, data_loader,
                epoch: int):   

    model.train()

    # Epoch statistics
    steps_in_epoch = len(data_loader)
    losses = np.zeros(steps_in_epoch, dtype=np.float32)
    accuracies = np.zeros(steps_in_epoch, dtype=np.float32)
    f1_scores_weighted = np.zeros(steps_in_epoch, dtype=np.float32)

    for params in model.parameters():   
        pass
    
    if params.size()[0] == 2:
        f1_weighted = F1Score(task="binary", num_classes=params.size()[0], average="macro").to(device)  
    else:   
        f1_weighted = F1Score(task="multiclass", num_classes=params.size()[0], average="macro").to(device)
    

    epoch_start_time = time.time()
    pbar = tqdm(total = len(data_loader))
    for step, (clips, targets) in enumerate(data_loader):
        # Prepare for next iteration
        optimizer.zero_grad()
        
        # Move inputs to GPU memory
        clips = clips.to(device)
        targets = targets.to(device)

        # Feed-forward through the network
        logits = model(clips)

        try:
            (batch_size, num_classes, timestamps) = logits.size()

            assert len(clips) == targets.size(0) == batch_size and targets.size(1) == timestamps
        except ValueError:
            timestamps = 1
            (batch_size, num_classes) = logits.size()
            assert len(clips) == targets.size(0) == batch_size
        

        # Calculate loss
        loss = criterion(logits, targets.long())

        # Calculate accuracy
        _, preds = torch.max(logits.data, 1)        
        correct = torch.sum(preds == targets.data)
        
        accuracy = correct.double() / batch_size #/ timestamps 
        f1_score_weighted = f1_weighted(preds, targets.data)
  
        # Back-propagation and optimization step
        loss.backward()
        optimizer.step()
        
        # Save statistics
        accuracies[step] = accuracy.item()
        losses[step] = loss.item()
        f1_scores_weighted[step] = f1_score_weighted.item()

        pbar.update(1)

    pbar.close()
    # Epoch statistics
    epoch_duration = float(time.time() - epoch_start_time)
    epoch_avg_loss = float(np.mean(losses))
    epoch_avg_acc = float(np.mean(accuracies))
    epoch_avg_f1_weighted = float(np.mean(f1_scores_weighted))

    return epoch_avg_loss, epoch_avg_acc, epoch_avg_f1_weighted, epoch_duration

########################################################################################################################
####################################################   Validation   ####################################################
########################################################################################################################


def validation_epoch(model, criterion, device, data_loader,
                     epoch: int): 

    model.eval()

    # Epoch statistics
    steps_in_epoch = len(data_loader)
    losses = np.zeros(steps_in_epoch, dtype=np.float32)
    accuracies = np.zeros(steps_in_epoch, dtype=np.float32)
    f1_scores_weighted = np.zeros(steps_in_epoch, dtype=np.float32)
    for params in model.parameters():
        pass 
    if params.size()[0] == 2:
        f1_weighted = F1Score(task="binary", num_classes=params.size()[0], average="macro").to(device)  
    else:   
        f1_weighted = F1Score(task="multiclass", num_classes=params.size()[0], average="macro").to(device)

    epoch_start_time = time.time()
    with torch.no_grad():
        pbar = tqdm(total = len(data_loader))
        for step, (clips, targets) in enumerate(data_loader):

            # Move inputs to GPU memory
            clips = clips.to(device)
            targets = targets.to(device)

            # Feed-forward through the network
            logits = model(clips)

            try:
                (batch_size, num_classes, timestamps) = logits.size()
                assert len(clips) == targets.size() == batch_size and targets.size() == timestamps
            except ValueError:
                timestamps = 1
                (batch_size, num_classes) = logits.size()

            # Calculate loss

            loss = criterion(logits, targets.long())

            # Calculate accuracy
            _, preds = torch.max(logits.data, 1)
            
            correct = torch.sum(preds == targets.data)
            accuracy = correct.double() / batch_size #/ timestamps
            f1_score_weighted = f1_weighted(preds, targets.data)

            # Save statistics
            accuracies[step] = accuracy.item()
            losses[step] = loss.item()
            f1_scores_weighted[step] = f1_score_weighted.item()

            pbar.update(1)
    pbar.close()
    # Epoch statistics
    epoch_duration = float(time.time() - epoch_start_time)
    epoch_avg_loss = float(np.mean(losses))
    epoch_avg_acc = float(np.mean(accuracies))
    epoch_avg_f1_weighted = float(np.mean(f1_scores_weighted))

    return epoch_avg_loss, epoch_avg_acc, epoch_avg_f1_weighted, epoch_duration


########################################################################################################################
####################################################   Test   ##########################################################
########################################################################################################################

def testing_epoch(model, device, data_loader, verbose):
    if verbose:
        print('#' * 50 + '   TESTING   ' + '#' * 50)
        print('Starting with final testing phase.')

    model.eval()

    # Epoch statistics
    steps_in_epoch = len(data_loader)
    accuracies = np.zeros(steps_in_epoch, dtype=np.float32)
    f1_scores = np.zeros(steps_in_epoch, dtype=np.float32)

    for params in model.parameters():
        pass
    if params.size()[0] == 2:
        f1 = F1Score(task="binary", num_classes=params.size()[0], average="macro").to(device)  
    else:   
        f1 = F1Score(task="multiclass", num_classes=params.size()[0], average="macro").to(device)
    

    predictions, corrects = list(), list()

    epoch_start_time = time.time()
    with torch.no_grad():
        if verbose:
            pbar = tqdm(total = len(data_loader))
        for step, (clips, targets) in enumerate(data_loader):

            # Move inputs to GPU memory
            clips = clips.to(device)
            targets = targets.to(device)

            # Feed-forward through the network
            logits = model(clips)

            try:
                (batch_size, num_classes, timestamps) = logits.size()
                assert len(clips) == targets.size(0) == batch_size and targets.size(1) == timestamps
            except ValueError:
                timestamps = 1
                (batch_size, num_classes) = logits.size()
                assert len(clips) == targets.size(0) == batch_size
                

            # Calculate accuracy
            _, preds = torch.max(logits.data, 1)
            correct = torch.sum(preds == targets.data)
            accuracy = correct.double() / batch_size / timestamps
            f1_score = f1(preds, targets.data)
            
            # Store predictions and targets
            predictions.extend(preds.detach().cpu().tolist())
            corrects.extend(targets.detach().cpu().tolist())

            # Save statistics
            accuracies[step] = accuracy.item()
            f1_scores[step] = f1_score.item()
            if verbose:
                pbar.update(1)
    if verbose:
        pbar.close()
    # Epoch statistics
    epoch_duration = float(time.time() - epoch_start_time)
    epoch_avg_acc = float(np.mean(accuracies))
    epoch_avg_f1 = float(np.mean(f1_scores))


    return epoch_avg_acc, epoch_avg_f1, epoch_duration, predictions, corrects


#################################################################################################
##################################### 3D ########################################################
#################################################################################################
def train_epoch_3d(model, criterion, optimizer, device, data_loader,
                epoch: int):  
    model.train()

    # Epoch statistics
    steps_in_epoch = len(data_loader)
    losses = np.zeros(steps_in_epoch, dtype=np.float32)
    accuracies = np.zeros(steps_in_epoch, dtype=np.float32)
    f1_scores_weighted = np.zeros(steps_in_epoch, dtype=np.float32)

    for params in model.parameters():   
        pass
    
    if params.size()[0] == 2:
        f1_weighted = F1Score(task="binary", num_classes=params.size()[0], average="macro").to(device)  
    else:   
        f1_weighted = F1Score(task="multiclass", num_classes=params.size()[0], average="macro").to(device)
    
    epoch_start_time = time.time()
    pbar = tqdm(total = len(data_loader))
    for step, (clips, targets) in enumerate(data_loader):

        # Prepare for next iteration
        optimizer.zero_grad()
        
        # Move inputs to GPU memory      
        clips = clips.permute(0, 2, 1, 3, 4)
        clips = clips.to(device)
        targets = targets.to(device)

        # Feed-forward through the network
        logits = model(clips)
        
        try:
            (batch_size, num_classes, timestamps) = logits.size()
            assert len(clips) == targets.size(0) == batch_size and targets.size(1) == timestamps
        except ValueError:
            timestamps = 1
            (batch_size, num_classes) = logits.size()
            assert len(clips) == targets.size(0) == batch_size

        # Calculate loss
        loss = criterion(logits, targets.long())
        
        # Calculate accuracy
        _, preds = torch.max(F.softmax(logits.data, dim=1), 1)        

        correct = torch.sum(preds == targets.data)
        accuracy = correct.double() / batch_size 
        f1_score_weighted = f1_weighted(preds, targets.data)

        # Back-propagation and optimization step
        loss.backward()
        optimizer.step()

        # Save statistics
        accuracies[step] = accuracy.item()
        losses[step] = loss.item()
        f1_scores_weighted[step] = f1_score_weighted.item()

        pbar.update(1)
        
    pbar.close()
    # Epoch statistics
    epoch_duration = float(time.time() - epoch_start_time)
    epoch_avg_loss = float(np.mean(losses))
    epoch_avg_acc = float(np.mean(accuracies))
    epoch_avg_f1_weighted = float(np.mean(f1_scores_weighted))

    return epoch_avg_loss, epoch_avg_acc, epoch_avg_f1_weighted, epoch_duration


def validation_epoch_3d(model, criterion, device, data_loader,
                     epoch: int): 

    model.eval()

    # Epoch statistics
    steps_in_epoch = len(data_loader)
    losses = np.zeros(steps_in_epoch, dtype=np.float32)
    accuracies = np.zeros(steps_in_epoch, dtype=np.float32)
    f1_scores_weighted = np.zeros(steps_in_epoch, dtype=np.float32)

    for params in model.parameters():
        pass  
    if params.size()[0] == 2:
        f1_weighted = F1Score(task="binary", num_classes=params.size()[0], average="macro").to(device)  
    else:   
        f1_weighted = F1Score(task="multiclass", num_classes=params.size()[0], average="macro").to(device)
    

    epoch_start_time = time.time()
    with torch.no_grad():
        pbar = tqdm(total = len(data_loader))
        for step, (clips, targets) in enumerate(data_loader):
            #start_time = time.time()
            
            clips = clips.permute(0, 2, 1, 3, 4)
            # Move inputs to GPU memory
            clips = clips.to(device)
            targets = targets.to(device)
           
            # Feed-forward through the network
            logits = model(clips)

            try:
                (batch_size, num_classes, timestamps) = logits.size()
                assert len(clips) == targets.size(0) == batch_size and targets.size(1) == timestamps
            except ValueError:
                timestamps = 1
                (batch_size, num_classes) = logits.size()
                assert len(clips) == targets.size(0) == batch_size

            # Calculate loss
            loss = criterion(logits, targets.long())

            # Calculate accuracy

            _, preds = torch.max(F.softmax(logits.data, dim=1), 1)
            correct = torch.sum(preds == targets.data)
            accuracy = correct.double() / batch_size 
            f1_score_weighted = f1_weighted(preds, targets.data)

            # Save statistics
            accuracies[step] = accuracy.item()
            losses[step] = loss.item()
            f1_scores_weighted[step] = f1_score_weighted.item()

            pbar.update(1)
    pbar.close()
    # Epoch statistics
    epoch_duration = float(time.time() - epoch_start_time)
    epoch_avg_loss = float(np.mean(losses))
    epoch_avg_acc = float(np.mean(accuracies))
    epoch_avg_f1_weighted = float(np.mean(f1_scores_weighted))


    return epoch_avg_loss, epoch_avg_acc, epoch_avg_f1_weighted, epoch_duration



def testing_epoch_3d(model, device, data_loader, verbose):
    if verbose:
        print('#' * 50 + '   TESTING   ' + '#' * 50)
        print('Starting with final testing phase.')

    model.eval()

    # Epoch statistics
    steps_in_epoch = len(data_loader)
    accuracies = np.zeros(steps_in_epoch, dtype=np.float32)
    f1_scores = np.zeros(steps_in_epoch, dtype=np.float32)

    for params in model.parameters():
        pass
    if params.size()[0] == 2:
        f1 = F1Score(task="binary", num_classes=params.size()[0], average="macro").to(device)  
    else:   
        f1 = F1Score(task="multiclass", num_classes=params.size()[0], average="macro").to(device)
    

    predictions, corrects = list(), list()

    epoch_start_time = time.time()
    with torch.no_grad():
        if verbose:
            pbar = tqdm(total = len(data_loader))
        for step, (clips, targets) in enumerate(data_loader):
            
            clips = clips.permute(0, 2, 1, 3, 4)
            # Move inputs to GPU memory
            clips = clips.to(device)
            targets = targets.to(device)

            # Feed-forward through the network
            logits = model(clips)

            try:
                (batch_size, num_classes, timestamps) = logits.size()
                assert len(clips) == targets.size(0) == batch_size and targets.size(1) == timestamps
            except ValueError:
                timestamps = 1
                (batch_size, num_classes) = logits.size()
                assert len(clips) == targets.size(0) == batch_size
                

            # Calculate accuracy
            _, preds = torch.max(logits.data, 1)
            correct = torch.sum(preds == targets.data)
            accuracy = correct.double() / batch_size / timestamps
            f1_score = f1(preds, targets.data)
            
            # Store predictions and targets
            predictions.extend(preds.detach().cpu().tolist())
            corrects.extend(targets.detach().cpu().tolist())

            # Save statistics
            accuracies[step] = accuracy.item()
            f1_scores[step] = f1_score.item()
            if verbose:
                pbar.update(1)
    if verbose:
        pbar.close()
    # Epoch statistics
    epoch_duration = float(time.time() - epoch_start_time)
    epoch_avg_acc = float(np.mean(accuracies))
    epoch_avg_f1 = float(np.mean(f1_scores))

    return epoch_avg_acc, epoch_avg_f1, epoch_duration, predictions, corrects
