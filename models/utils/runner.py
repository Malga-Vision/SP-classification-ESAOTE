import os
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from .iterators import train_epoch, validation_epoch, testing_epoch, save_checkpoint, load_checkpoint, train_epoch_3d, validation_epoch_3d, testing_epoch_3d
from tqdm import tqdm
from typing import Union

class NoGPUError(Exception):
    pass



def adapt_sononet_head_weights(check_point_new, num_labels):
   
    head = ["_adaptation.3.weight",
            "_adaptation.4.weight", "_adaptation.4.bias", "_adaptation.4.running_mean", "_adaptation.4.running_var"]

    if check_point_new[head[0]].size()[0] == num_labels:
        return check_point_new
    for l in head:
        l_num, l_param = l.split('.')[-2:]

        if int(l_num) == 3:
            
            check_point_new[l] = torch.zeros([num_labels, *check_point_new[l].size()[1:]])
            torch.nn.init.kaiming_uniform_(check_point_new[l], mode='fan_in', nonlinearity='relu')
        else:  
            check_point_new[l] = torch.zeros(num_labels)
    return check_point_new


############## Compute accuracy and losses at the end of epoch ############################
def train(model, train_loader, val_loader,
          checkpoint_dir: str = None, pretraining_dir: str = None,
          class_weights: Union[torch.Tensor, None] = None, class_smoothing: float = 0.,
          max_num_epochs: int = 50, patience: int = 5,
          lr: float = 0.005, momentum: float = 0.9, weight_decay: float = 0.,
          lr_scheduler: Union[str, None] = 'plateau', lr_sched_patience: int = 2, optimizer: str = 'sgd',
          device=torch.device('cpu'), seed: int = None):
    if seed is not None:
        torch.manual_seed(seed)

    # configuration options
    early_stopping = True if 0 < patience < max_num_epochs else False
    assert optimizer in ['sgd', 'adam'], 'Optimizer can either be "sgd" or "adam".'

    # configure optimizer and criterion
    if class_weights == None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=class_smoothing)
    if optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)  
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # load pre-training checkpoint (weights), if present
    if pretraining_dir:
        print("Pretraining directory: ", pretraining_dir)
        checkpoint_path_old = os.path.join(pretraining_dir, 'ckpt_best_loss.pth')
        assert os.path.isfile(checkpoint_path_old), \
            f'No checkpoint weights available in the given pre-training directory {pretraining_dir}'
        check_point = torch.load(checkpoint_path_old, weights_only=True)
        for params in model.parameters():
            pass
        num_labels = params.size()[0]

        check_point_adapted = adapt_sononet_head_weights(check_point, num_labels=num_labels)
        
        model.load_state_dict(check_point_adapted)
        
    # configure learning-rate scheduler
    if lr_scheduler == 'plateau':
        assert lr_sched_patience < patience, 'Patience for lr-scheduler should be smaller than early-stopping patience.'
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.7, patience=lr_sched_patience, threshold=0.0001, threshold_mode='abs')
    elif lr_scheduler == 'multistep':
        assert lr_sched_patience < patience, 'Patience for lr-scheduler should be smaller than early-stopping patience.'
        lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lr_sched_patience, 3*lr_sched_patience])
    else:  
        lr_sched = None

    # run the training loop for defined number of epochs
    history = {'loss': [], 'accuracy': [], 'f1score': [],
               'val_loss': [], 'val_accuracy': [], 'val_f1score': []}
    best_epoch, best_val_f1, best_val_acc, best_val_loss = 0, 0., 0., np.inf

    for epoch in range(max_num_epochs):
        print('#' * 50 + '  Epoch {}  '.format(epoch+1) + '#' * 50)
        print('Training phase: ')       
        
        train_loss, train_acc, train_f1, train_duration = train_epoch(
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    data_loader=train_loader,
                    epoch=epoch
        )
        
        
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['f1score'].append(train_f1)
        
        # perform one validation epoch
        print('Validation phase:')
        val_loss, val_acc, val_f1, val_duration = validation_epoch(
                    model=model,
                    criterion=criterion,
                    device=device,
                    data_loader=val_loader,
                    epoch=epoch
        )
        
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_f1score'].append(val_f1)
        
        # update learning rate
        if lr_scheduler == 'plateau':
            print(f"Learning rate before scheduler step: {optimizer.param_groups[0]['lr']}")
            lr_sched.step(val_loss)
            print(f"Learning rate after scheduler step: {optimizer.param_groups[0]['lr']}")
        elif lr_scheduler == 'multistep':
            lr_sched.step(epoch)

        
        print('\n' + '-'*50 + f'  SUMMARY  ' + '-'*50)
        print(f'Training Phase.')
        print(f'  Total Duration:         {int(np.ceil(train_duration / 60)) :d} minutes')
        print(f'  Train Loss:     {train_loss :.3f}')
        print(f'  Train Accuracy: {train_acc :.3f}')
        print(f'  Train F1-score: {train_f1 :.3f}')
        
        print('Validation Phase.')
        print(f'  Total Duration:              {int(np.ceil(val_duration / 60)) :d} minutes')
        print(f'  Validation Loss:     {val_loss :.3f}')
        print(f'  Validation Accuracy: {val_acc :.3f}')
        print(f'  Validation F1-score: {val_f1 :.3f}')

        if isinstance(model, nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()


        # saving best model based on min loss
        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            print(f'Found new best validation loss: {best_val_loss :.3f}')
            if checkpoint_dir:
                checkpoint_path = os.path.join(checkpoint_dir, 'ckpt_best_loss.pth')
                
                torch.save(model.state_dict(), checkpoint_path)
                print(f'Model checkpoint (best loss) written to:         {checkpoint_path}')

        # early stopping
        if early_stopping and epoch >= best_epoch + patience:
            if all([loss > best_val_loss for loss in history['val_loss'][-patience:]]):
                # all last validation losses are bigger than the best one
                print(f'Early stopping because loss on validation set did not decrease in the last {patience} epochs.')
                break

        print('-'*130 + '\n')

    # store training history
    history_path = os.path.join(checkpoint_dir, 'history')
    with open(history_path, 'wb') as outfile:
        pickle.dump(history, outfile)  # Read as: history = pickle.load(open(history_file, "rb"))

    return [float(best_val_f1), float(best_val_acc), float(best_val_loss), best_epoch if early_stopping else max_num_epochs-1], history


def test(model, test_loader,
        checkpoint_dir: str = None,
        device=torch.device('cpu'),
        verbose = True):

    # load training checkpoint
    if checkpoint_dir is None or not os.path.isdir(checkpoint_dir):
        raise ValueError('Must specify a valid checkpoint directory from where to load pre-trained model weights.')
    checkpoint_path = os.path.join(checkpoint_dir, 'ckpt_best_loss.pth')

    model.load_state_dict(torch.load(checkpoint_path, weights_only=False, map_location=device))
    
    # Perform one validation epoch
    test_acc, test_f1, test_duration, predictions, targets = testing_epoch(
        model=model,
        device=device,
        data_loader=test_loader,
        verbose=verbose
    )
    if verbose:
        print(f'\nTest Phase.')
        print(f'  Total Duration:        {int(np.ceil(test_duration / 60)) :d} minutes')
        print(f'  Average Test Accuracy: {test_acc :.3f}\n')
        print(f'  Average Test F1-score: {test_f1 :.3f}\n')

    return [float(test_f1), float(test_acc)], [predictions, targets]


#################################################################################################
##################################### 3D ########################################################
#################################################################################################
def train_3d(model, train_loader, val_loader,
          checkpoint_dir: str = None, pretraining_dir: str = None,
          class_weights: Union[torch.Tensor, None] = None, class_smoothing: float = 0.,
          max_num_epochs: int = 50, patience: int = 5,
          lr: float = 0.005, momentum: float = 0.9, weight_decay: float = 0.,
          lr_scheduler: Union[str, None] = 'plateau', lr_sched_patience: int = 2, optimizer: str = 'sgd',
          device=torch.device('cpu'), seed: int = None):
    if seed is not None:
        torch.manual_seed(seed)

    # configuration options
    early_stopping = True if 0 < patience < max_num_epochs else False
    assert optimizer in ['sgd', 'adam'], 'Optimizer can either be "sgd" or "adam".'

    # configure optimizer and criterion
    if class_weights == None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=class_smoothing)
    if optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)  
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # load pre-training checkpoint (weights), if present
    if pretraining_dir:
        checkpoint_path_old = os.path.join(pretraining_dir, 'ckpt_best_loss.pth')
        assert os.path.isfile(checkpoint_path_old), \
            f'No checkpoint weights available in the given pre-training directory {pretraining_dir}'
        check_point = torch.load(checkpoint_path_old)
        for params in model.parameters():
            pass
        num_labels = params.size()[0]
        
        check_point_adapted = adapt_sononet_head_weights(check_point, num_labels=num_labels)
        model.load_state_dict(check_point_adapted)
        
    # configure learning-rate scheduler
    if lr_scheduler == 'plateau':
        assert lr_sched_patience < patience, 'Patience for lr-scheduler should be smaller than early-stopping patience.'
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.1, patience=lr_sched_patience, verbose=True)
       
    elif lr_scheduler == 'multistep':
        assert lr_sched_patience < patience, 'Patience for lr-scheduler should be smaller than early-stopping patience.'
        lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lr_sched_patience, 3*lr_sched_patience])
    else:  
        lr_sched = None

    # run the training loop for defined number of epochs
    history = {'loss': [], 'accuracy': [], 'f1score': [],
               'val_loss': [], 'val_accuracy': [], 'val_f1score': []}
    best_epoch, best_val_f1, best_val_acc, best_val_loss = 0, 0., 0., np.inf


    #print("aaa Model: ",next(model.parameters()).size())
    for epoch in range(max_num_epochs):
        print('#' * 50 + '  Epoch {}  '.format(epoch+1) + '#' * 50)
        print('Training phase: ')
        # perform one training epoch        
        train_loss, train_acc, train_f1, train_duration = train_epoch_3d(
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    data_loader=train_loader,
                    epoch=epoch
        )
        
        
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['f1score'].append(train_f1)
        
        # perform one validation epoch
        print('Validation phase:')
        val_loss, val_acc, val_f1, val_duration = validation_epoch_3d(
                    model=model,
                    criterion=criterion,
                    device=device,
                    data_loader=val_loader,
                    epoch=epoch
        )
        
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_f1score'].append(val_f1)
        
        # update learning rate
        if lr_scheduler == 'plateau':
            print(f"Learning rate before scheduler step: {optimizer.param_groups[0]['lr']}")
            lr_sched.step(val_loss)
            print(f"Learning rate after scheduler step: {optimizer.param_groups[0]['lr']}")
        elif lr_scheduler == 'multistep':
            lr_sched.step(epoch)

        
        print('\n' + '-'*50 + f'  SUMMARY  ' + '-'*50)
        print(f'Training Phase.')
        print(f'  Total Duration:         {int(np.ceil(train_duration / 60)) :d} minutes')
        print(f'  Train Loss:     {train_loss :.3f}')
        print(f'  Train Accuracy: {train_acc :.3f}')
        print(f'  Train F1-score: {train_f1 :.3f}')
        
        print('Validation Phase.')
        print(f'  Total Duration:              {int(np.ceil(val_duration / 60)) :d} minutes')
        print(f'  Validation Loss:     {val_loss :.3f}')
        print(f'  Validation Accuracy: {val_acc :.3f}')
        print(f'  Validation F1-score: {val_f1 :.3f}')

        # saving best model based on min loss
        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            print(f'Found new best validation loss: {best_val_loss :.3f}')
            if checkpoint_dir:
                checkpoint_path = os.path.join(checkpoint_dir, 'ckpt_best_loss.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print(f'Model checkpoint (best loss) written to:         {checkpoint_path}')

        # early stopping
        if early_stopping and epoch >= best_epoch + patience:
            if all([loss > best_val_loss for loss in history['val_loss'][-patience:]]):
                # all last validation losses are bigger than the best one
                print(f'Early stopping because loss on validation set did not decrease in the last {patience} epochs.')
                break

        print('-'*130 + '\n')

    # store training history
    history_path = os.path.join(checkpoint_dir, 'history')
    with open(history_path, 'wb') as outfile:
        pickle.dump(history, outfile)  # Read as: history = pickle.load(open(history_file, "rb"))

    return [float(best_val_f1), float(best_val_acc), float(best_val_loss), best_epoch if early_stopping else max_num_epochs-1], history




def test_3d(model, test_loader,
        checkpoint_dir: str = None,
        device=torch.device('cpu'), verbose=True):

    # load training checkpoint

    if checkpoint_dir is None or not os.path.isdir(checkpoint_dir):
        raise ValueError('Must specify a valid checkpoint directory from where to load pre-trained model weights.')
    checkpoint_path = os.path.join(checkpoint_dir, 'ckpt_best_loss.pth')
   
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    
    # Perform one validation epoch
    test_acc, test_f1, test_duration, predictions, targets = testing_epoch_3d(
        model=model,
        device=device,
        data_loader=test_loader,
        verbose=verbose
    )

    if verbose:
        print(f'\nTest Phase.')
        print(f'  Total Duration:        {int(np.ceil(test_duration / 60)) :d} minutes')
        print(f'  Average Test Accuracy: {test_acc :.3f}\n')
        print(f'  Average Test F1-score: {test_f1 :.3f}\n')

   
    return [float(test_f1), float(test_acc)], [predictions, targets]

