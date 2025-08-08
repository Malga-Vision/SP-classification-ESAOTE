import os
import json
import torch
import numpy as np
import pandas as pd
import glob
from torch.utils.data import random_split



def splitter(all_infos: pd.DataFrame, exams_per_patient: list, val_split: float, verbose: bool = False, random_seed: int = None):
    """
    We put in the validation set the last val_split % of patients (having all 5 scans
    expected in the protocol) from all 4 clinical centers. All the remaining data
    goes in the training set.
    """

    # Retrieve location of US acquisitions (examination centers) from CVAT name of the exams and sort data by location
    all_infos['location'] = all_infos['name-cvat'].apply(lambda x: x.split('-')[0])
    all_infos = all_infos.sort_values(by=['location'])

    # NOTE: all exams where patient info is missing will be assigned to patient=0 (and will have complete-scans=False),
    #       actual patients instead start from 1 and can either have complete-scans=True or False.
    all_infos['patient'] = 0
    all_infos['complete-scans'] = False

    # Loop through patients
    missing_video = []
    for k_patient, patient_exams in enumerate(exams_per_patient):
        all_idx = []
        # Loop through the exams (US video acquisitions) of each patient
        for exam in patient_exams:
            exam = int(exam)  # it is the US video id of the exam
            try:
                # Store in all_idx the index of the exam for which we have the corresponding video in the dataset
                idx = all_infos.loc[all_infos['id-video'] == exam].index.values[0]
                all_idx.append(idx)
            except IndexError:
                # Store in missing_video the id of the exam for which we have no (or not-annotated) video in the dataset
                missing_video.append(exam)
        # And set the patient id in "infos" for all exams for which we have the corresponding video in the dataset
        all_infos.loc[all_idx, 'patient'] = k_patient + 1
        # If we have 5 exams for the given patient set complete scans to True
        if len(all_idx) == 5:
            all_infos.loc[all_idx, 'complete-scans'] = True
    all_infos['patient'].astype(int)

    # Find the list of unique examination centers (locations)
    locations = np.unique(list(all_infos['location'])).astype('str')
    # patients = np.arange(0, len(exams_per_patient)+1)

    # Find incomplete patient exams (i.e. those exams where patients had more or less than 5 exams)
    incomplete_df = all_infos[~all_infos['complete-scans']].drop(['complete-scans'], axis='columns')
    # Find complete patient exams (i.e. 5 exams per patient)
    complete_df = all_infos.drop(incomplete_df.index, axis='rows').drop(['complete-scans'], axis='columns')
    # Sort them by location, patient and acquisition date
    complete_df = complete_df.sort_values(['location', 'patient', 'date'])

    # Store infos of the exams in training or validation set, respectively
    train_df, test_df = pd.DataFrame(), pd.DataFrame()
    # Loop through examination centers
    for location in locations:
        # Find those exams that are within a complete scan sequence and being acquired in the given acquisition center
        loc_df = complete_df.loc[complete_df['location'] == location]
        assert len(loc_df) % 5 == 0

        # Shuffle the location-specific DataFrame
        if random_seed is not None:
            loc_df = loc_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # Compute the number of exams from the given acquisition center that should go in the test set
        n_vids4test = int(round(len(loc_df) / 5 * val_split)) * 5
        # Store the last exams in the validation set and all the first ones in the training set
        test_df = pd.concat([test_df, loc_df.tail(n_vids4test)])
        train_df = pd.concat([train_df, loc_df.head(len(loc_df) - n_vids4test)])
    # Add all incomplete exams to the training set
    train_df = pd.concat([train_df, incomplete_df])

    if verbose:
        # Print some information on the train-val splitting
        print(f'Number of patients with all 5 exams are {len(complete_df) // 5}, i.e. {len(complete_df)} videos.')
        print(f'There are also {len(set(incomplete_df["patient"]))} patients with in-complete (or over-complete) exams, i.e. {len(incomplete_df)} videos.')
        n_exams_per_patient = list(map(lambda x: len(x), exams_per_patient))
        n_exams_more5 = list(filter(lambda x: x > 5, n_exams_per_patient))
        n_exams_less5 = list(filter(lambda x: x < 5, n_exams_per_patient))
        print(f'Specifically, there are {len(n_exams_more5)} patients having more than 5 exams {set(n_exams_more5)}, and {len(n_exams_less5)} with less {set(n_exams_less5)}.')
        n_patients_in_test = int(round(len(complete_df) / 5 * val_split))
        n_patients_in_train = int((len(complete_df) - n_patients_in_test * 5) // 5)
        print(f'Number of patients with all 5 exams in training set are {n_patients_in_train} (plus all incomplete/overcomplete ones), i.e. a total of {n_patients_in_train * 5 + len(incomplete_df)} videos.')
        print(f'Instead, in the validation set there are {n_patients_in_test} patients with all 5 exams (no incomplete ones), i.e. {n_patients_in_test * 5} videos.')

    return train_df, test_df


def split2d_train_validation(info_path, valid_split: float = 0.2, verbose = False, random_seed: int = None):
    with open(os.path.join(info_path, 'exams.json'), 'r') as f:
        exams_per_patient = json.load(f)
    all_train_infos = pd.read_csv(os.path.join(info_path, 'train-infos.csv'))
    train_path = os.path.join(info_path, 'train')
    img_paths = sorted(glob.glob(os.path.join(train_path, '*', '*.png')))

    train_df, valid_df = splitter(all_train_infos, exams_per_patient, valid_split, verbose = verbose, random_seed=random_seed)
    valid_files = valid_df['name-video'].apply(lambda f: f[:-4]).values.tolist()  

    train_indices, valid_indices = [], []
    valid_paths, train_paths = [], []
    
    for idx, file in enumerate(img_paths):
        filename = os.path.split(file)[-1][:-8]
        if filename in valid_files:
            valid_indices.append(idx)
            valid_paths.append(file)
        else:
            train_indices.append(idx)
            train_paths.append(file)

    if verbose == True:
        print(f'After splitting the whole {len(img_paths)} image dataset, the training set consists of '
            f'{len(train_indices)} samples while the validation set of {len(valid_indices)}. '
            f'Hence actual validation split is {int(len(valid_indices)/len(img_paths)*100)} %.')

    return train_paths, valid_paths






def split3d_train_validation(info_path, valid_split: float = 0.2, clip_size: int = 5, verbose = False, random_seed: int =None):

    with open(os.path.join(info_path, 'exams.json'), 'r') as f:
        exams_per_patient = json.load(f)
    all_train_infos = pd.read_csv(os.path.join(info_path, 'train-infos.csv'))

    video_dirs = sorted(glob.glob(os.path.join(info_path, 'train', 'videos', '*')))    

    video_paths = list(map(lambda p: sorted(glob.glob(os.path.join(p, '*.png'))), video_dirs))

    train_df, valid_df = splitter(all_train_infos, exams_per_patient, valid_split, verbose = True, random_seed=random_seed)
    valid_files = valid_df['name-video'].apply(lambda f: f[:-4]).values.tolist()  


    valid_paths, train_paths = [], []


    for idx_video, video in enumerate(video_paths):
        filename = os.path.split(video_paths[idx_video][0])[-1][:-8]
        if filename in valid_files:
            valid_paths.append(video)
        else:
            train_paths.append(video)

    train_clips = split_videos_in_clips(train_paths, info_path, clip_size)
    valid_clips = split_videos_in_clips(valid_paths, info_path, clip_size)

    if verbose == True:
        print(f'After splitting videos in clips we have:\n'
            f'- Training set composed of {len(train_clips)} clips\n'
            f'- Validation set composed of {len(valid_clips)} clips\n'
            f'Hence actual validation split is {int(len(valid_clips)/(len(valid_clips) + len(train_clips))*100)} %.')

    return train_clips, valid_clips



def same_labels(clip, info_path):     
    #check that all the labels within the clip are the same. We want "clean" labels
    with open(os.path.join(info_path, 'classes.json'), 'r') as f:
        classes = json.load(f)
    lbls = []
    
    for frame_path in clip:
        with open(frame_path.replace('videos', 'labels')[:-4] + '.txt', 'r') as f:   
            lbls.append(int(classes[f.read()]))
        
    #print(lbls)
    if lbls.count(lbls[0]) == len(lbls):
        return True
    return False    
    

def split_videos_in_clips(split_paths, info_path, clip_size: int = 5, overlap: int = 0):
    # Return a list of clips, where each clip contains "clip_size" frames with a given overlap and all frames having same label
    clips = []
    stride = clip_size - overlap  # Define the step size
    
    for video in split_paths:
        video_clips = [video[i:i+clip_size] for i in range(0, len(video) - clip_size + 1, stride) 
                       if same_labels(video[i:i+clip_size], info_path)]
        clips.extend(video_clips)
    
    return clips




def split_train_validation_random(data, valid_split: float = 0.2, seed: int = None):
    n_valid = int(len(data) * valid_split)
    if seed:
        torch.manual_seed(seed)
    dataset_train, dataset_valid = random_split(data, [len(data)-n_valid, n_valid])
    print(f'After splitting the whole {len(data)} image dataset, the training set consists of '
          f'{len(data)-n_valid} samples while the validation set of {n_valid}. '
          f'Hence actual validation split is {int(n_valid/len(data)*100)} %.')
    return dataset_train, dataset_valid
