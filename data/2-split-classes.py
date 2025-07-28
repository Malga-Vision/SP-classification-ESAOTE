import os
import glob
import json
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import argparse


def find_class(frame_labels: list):
    """
    Usage:
    - input (list): lista di etichette per le bounding boxes presenti in un dato frame
    - output (str): piano per la classificazione (uno dei 6 definiti sotto)

    Description:
    1 - diramazione-porta: Presenza contemporanea di almeno due dei landmark Ramo dx Porta,
        Ramo sx Porta, e Dir. Porta. Rappresenta il punto di partenza delle prime tre sequenze.
        Non saranno così distinguibili, ma le etichette sono troppo rumorose per andare nel dettaglio.

    2 - confluenza-sovraepatiche: presenza simultanea di "Vena cava" e almeno un landmark "Dir. Sovraepatiche".

    3 - rapporto-epatorenale: Presnza del landmark "Rene".

    4 - cuore: Presnza del landmark "Cuore". Se in compresenza con la "diramazione sovraepatiche", 
        vince quest'ultima se classificata e non fate multilabel.

    5 - colecisti: Colecisti. Solo se da sola. Come per il cuore, se c'è un altro piano valido vince quest'ultimo.

    6 - ilo: Ilo.

    7 - other: tutte le immagini che non soddisfano nessuna delle precedenti condizioni
    """

    non_used_labels = ['Stomaco', 'Fine fegato', 'Altro']
    k_label = 0
    for frame_label in frame_labels:
        if frame_label in non_used_labels:
            frame_labels.pop(k_label)
            k_label -= 1
        k_label += 1

    cls = 'other'

    if ('Ramo sx Porta' in frame_labels and 'Ramo dx Porta' in frame_labels and 'Dir. porta' in frame_labels) \
            or ('Ramo sx Porta' in frame_labels and 'Ramo dx Porta' in frame_labels) \
            or ('Ramo sx Porta' in frame_labels and 'Dir. porta' in frame_labels) \
            or ('Ramo dx Porta' in frame_labels and 'Dir. porta' in frame_labels):
        cls = 'diramazione-porta'

    elif 'Vena cava inferiore' in frame_labels and 'Dir sovraepatiche' in frame_labels:
        cls = 'confluenza-sovraepatiche'

    elif 'Colecisti' in frame_labels and len(frame_labels) == 1:
        cls = 'colecisti'

    elif 'Cuore' in frame_labels:
        cls = 'cuore'

    elif 'Rene' in frame_labels:
        cls = 'rapporto-epatorenale'

    elif 'Ilo' in frame_labels:
        cls = 'ilo'

    return cls


def populate_data_split(df: pd.DataFrame, out_path: str,
                        video_path: str, annotation_path: str, all_tracks: pd.DataFrame):
    """
    It creates 3 folders (videos, annotations, labels) in the given out_path directory and populates them with
    video folders and their PNG images, full XML annotation files, and a class label file per image as derived from
    the annotations according to the given rules (see find_class function above).
    """

    os.makedirs(os.path.join(out_path, 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'videos'), exist_ok=True)

    for idx in tqdm(sorted(list(df.index)), desc='Progress: '):
        id_video = df['id-video'].loc[idx]
        id_annotation = df['id-annotation'].loc[idx]

        img_files = sorted(glob.glob(os.path.join(video_path, f'*-{id_video}_NOVERLAY-000', '*.png')))
        annotation_file = os.path.join(annotation_path, f'{id_annotation}.xml')

        annotations = all_tracks.loc[idx]

        shutil.copyfile(annotation_file, os.path.join(out_path, 'annotations', str(id_video) + '.xml'))
        os.makedirs(os.path.join(out_path, 'videos', str(id_video)), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'labels', str(id_video)), exist_ok=True)
        for k_img, imfile in enumerate(img_files):
            img_labels = annotations['label'].loc[annotations['frame'] == k_img].values.tolist()
            img_class = find_class(img_labels)

            imfilename = os.path.split(imfile)[-1]
            shutil.copyfile(imfile, os.path.join(out_path, 'videos', str(id_video), imfilename))
            with open(os.path.join(out_path, 'labels', str(id_video), imfilename[:-4] + '.txt'), 'w') as f:
                f.write(img_class)


class EsaoteUSLabels(Dataset):
    """
    Class for loading all labels in the train or test set.
    """
    def __init__(self, root: str):
        label_path = os.path.join(root, 'labels')
        assert os.path.exists(label_path)
        self.lbl_paths = sorted(glob.glob(os.path.join(label_path, '*', '*.txt')))
        with open(os.path.join(os.path.split(root)[0], 'classes.json'), 'r') as f:
            self.classes = json.load(f)
        self.num_classes = len(self.classes)

    def __getitem__(self, index):
        lbl_path = self.lbl_paths[index]
        with open(lbl_path, 'r') as f:
            lbl = self.classes[f.read()]  # retrieve the class id of the given file
        return lbl

    def __len__(self):
        return len(self.lbl_paths)


def plot_labels_distribution(root: str):
    """
    Plot distribution of the data in the classes
    """

    dataset = EsaoteUSLabels(root=root)
    classes = dataset.classes


    labels = np.zeros(len(dataset), dtype=int)


    k = 0
    for y in tqdm(dataset, desc='Progress:'):
        labels[k] = y
        k += 1
    assert labels.max() <= len(classes)-1


    vals, bins = np.histogram(labels, bins=len(classes), range=(0, len(classes)))

    dir, split = os.path.split(root)
    np.save(os.path.join(dir, split+'-class-number.npy'), vals)


    plt.figure()
    plt.bar(x=bins[:-1], height=vals/np.sum(vals)*100)
    plt.ylim(0, 60)
    plt.savefig(os.path.join(dir, split+'-class-distribution.svg'))
    plt.show()

#-data_dir '/data01/simone/data/us/liver/esaote'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, help='Directory of the full dataset.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Define directory where "videos" and "annotations" folders are stored and retrieve them
    main_path = args.data_dir
    video_path = os.path.join(main_path, 'videos')
    annotation_path = os.path.join(main_path, 'annotations')

    split_path = os.path.join(main_path, 'split')
    train_path, test_path = os.path.join(split_path, 'train'), os.path.join(split_path, 'test')


    train_df = pd.read_csv(os.path.join(split_path, 'train-infos.csv'))
    train_df = train_df.set_index('idx')
    test_df = pd.read_csv(os.path.join(split_path, 'test-infos.csv'))
    test_df = test_df.set_index('idx')


    all_tracks = pd.read_csv(os.path.join(main_path, 'summary', 'tracks.csv'))
    all_tracks = all_tracks.set_index('idx')


    classes = {'diramazione-porta': 0, 'confluenza-sovraepatiche': 1, 'colecisti': 2, 'cuore': 3,
               'rapporto-epatorenale': 4, 'ilo': 5, 'other': 6}
    with open(os.path.join(split_path, 'classes.json'), 'w') as f:
        json.dump(classes, f)

    print('\n' + '----'*5 + ' Training set ' + '----'*5)

    populate_data_split(train_df, train_path,
                        video_path, annotation_path, all_tracks)

    plot_labels_distribution(train_path)
    print('Done!')

    print('\n' + '----'*5 + ' Test set ' + '----'*5)

    populate_data_split(test_df, test_path,
                        video_path, annotation_path, all_tracks)

    plot_labels_distribution(test_path)
    print('Done!')
