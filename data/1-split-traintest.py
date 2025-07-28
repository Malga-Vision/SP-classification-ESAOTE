import os
import json
import numpy as np
import pandas as pd
import argparse
import shutil

TEST_SPLIT = 0.15


def splitter(all_infos: pd.DataFrame, exams_per_patient: list, verbose: bool = True):
    """
    We put in the test set the last 15% of patients (in chronological order and having all 5 scans expected in the protocol)
    from all 4 clinical centers (Aquila, Firenze, Palermo, Sassari). All the remaining data goes in the training set.
    """

    # Retrieve location of US acquisitions (examination centers) from CVAT name of the exams and sort data by location
    all_infos['location'] = all_infos['name-cvat'].apply(lambda x: x.split('-')[0])
    all_infos = all_infos.sort_values(by=['location'])

    # NOTE: all exams where patient info is missing will be assigned to patient=0 (and will have complete-scans=False),
    #       actual patients instead start from 1 and can either have complete-scans=True or False.
    all_infos['patient'] = 0
    all_infos['complete-scans'] = False

    missing_video = []
    for k_patient, patient_exams in enumerate(exams_per_patient):
        all_idx = []
       
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

  
    locations = np.unique(list(all_infos['location'])).astype('str')


    # Find incomplete patient exams (i.e. those exams where patients had more or less than 5 exams)
    incomplete_df = all_infos[~all_infos['complete-scans']].drop(['complete-scans', 'width', 'height'],
                                                                 axis='columns')
    # Find complete patient exams (i.e. 5 exams per patient)
    complete_df = all_infos.drop(incomplete_df.index, axis='rows').drop(['complete-scans', 'width', 'height'],
                                                                        axis='columns')
    # Sort them by location, patient and acquisition date
    complete_df = complete_df.sort_values(['location', 'patient', 'date'])

    # Store infos of the exams in training or test set, respectively
    train_df, test_df = pd.DataFrame(), pd.DataFrame()

    for location in locations:

        loc_df = complete_df.loc[complete_df['location'] == location]
        assert len(loc_df) % 5 == 0
        # Compute the number of exams from the given acquisition center that should go in the test set
        n_vids4test = int(round(len(loc_df) / 5 * TEST_SPLIT)) * 5
        # Store the last exams in the test set and all the first ones in the training set
        test_df = pd.concat([test_df, loc_df.tail(n_vids4test)])
        train_df = pd.concat([train_df, loc_df.head(len(loc_df) - n_vids4test)])
    # Add all incomplete exams to the training set
    train_df = pd.concat([train_df, incomplete_df])

    if verbose:
       
        print(f'Number of patients with all 5 exams are {len(complete_df) // 5}, i.e. {len(complete_df)} videos.')
        print(f'There are also {len(set(incomplete_df["patient"]))} patients with in-complete (or over-complete) exams, i.e. {len(incomplete_df)} videos.')
        n_exams_per_patient = list(map(lambda x: len(x), exams_per_patient))
        n_exams_more5 = list(filter(lambda x: x > 5, n_exams_per_patient))
        n_exams_less5 = list(filter(lambda x: x < 5, n_exams_per_patient))
        print(f'Specifically, there are {len(n_exams_more5)} patients having more than 5 exams {set(n_exams_more5)}, and {len(n_exams_less5)} with less {set(n_exams_less5)}.')
        n_patients_in_test = int(round(len(complete_df) / 5 * TEST_SPLIT))
        n_patients_in_train = int((len(complete_df) - n_patients_in_test * 5) // 5)
        print(f'Number of patients with all 5 exams in training set are {n_patients_in_train} (plus all incomplete/overcomplete ones), i.e. a total of {n_patients_in_train * 5 + len(incomplete_df)} videos.')
        print(f'Instead, in the test set there are {n_patients_in_test} patients with all 5 exams (no incomplete ones), i.e. {n_patients_in_test * 5} videos.')

    return train_df, test_df



#-data_dir '/data01/simone/data/us/liver/esaote'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, help='Directory of the full dataset.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Define directory where "videos" and "annotations" folders are stored and retrieve them
    main_path = args.data_dir

    # Create a "split" folder inside the main directory: here we will create "train" and "test" folders
    split_path = os.path.join(main_path, 'split')
    
    if os.path.exists(split_path):
        try:
            shutil.rmtree(split_path)
            print(f"Directory '{split_path}' and its contents successfully deleted.")
        except OSError as e:
            print(f"Error: {e}")
            
    os.makedirs(split_path, exist_ok=True)
    
    # Read information of the US exams (videos) divided by patient
    with open(os.path.join(main_path, 'exams.json'), 'r') as f:
        exams_per_patient = json.load(f)

    # Read the "infos" dataframe previously generated
    all_infos = pd.read_csv(os.path.join(main_path, 'summary', 'infos.csv'))
    all_infos = all_infos.set_index('idx')

    # Split data in train and test dataframes by calling the splitter() function
    train_df, test_df = splitter(all_infos, exams_per_patient)

    # Log the resulting dataframes to disk
    train_df.to_csv(os.path.join(split_path, 'train-infos.csv'))
    test_df.to_csv(os.path.join(split_path, 'test-infos.csv'))
