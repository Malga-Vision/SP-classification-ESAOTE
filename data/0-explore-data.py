import os
import cv2
import tqdm
import glob
import json
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import shutil
import argparse



def parse_data(video_path: str, annotation_path: str, output_path: str):
    """
    Read all XML files of annotations from the annotation_path and create 3 CSV files in output_path:
        1) full-infos.csv - 1 row per each video and 9 columns:
                - 'idx': ordered number corresponding to the video (same for all 3 dataframes)
                - 'id-video': the identifier of the video
                - 'id-annotation': the identifier of the corresponding annotations file (name of the file)
                - 'name-video': full name of the video folder
                - 'name-cvat': full name of the video on CVAT
                - 'date': date of video acquisition
                - 'size': number of frames in the video
                - 'width/height': width/height of each frame
        2) infos.csv - 1 row per each video (only clean data, i.e. videos having annotations and correct number
           of frames) and 9 columns (same ones as full-infos.csv).
        3) tracks.csv - 1 row per each annotation (only for clean data, but note that there are several
           annotations in one video and also there may be non-annotated frames or frames with more than one annotation,
           hence the number of rows in tracks.csv is different from infos.csv, though correspondences are preserved
           by the 'idx' column) and 10 columns:
                - 'idx': ordered number corresponding to the video (same for all 3 dataframes)
                - 'frame': the frame where the annotation was put
                - 'outside':
                - 'occluded':
                - 'keyframe': whether it has been manually annotated by a physician or obtained by interpolation
                - 'xtl/ytl': x/y top-left coordinate of the annotation box
                - 'xbr/ybr': x/y bottom-right coordinate of the annotation box
                - 'label': label attributed to the annotation box
    """

    annotations = sorted(glob.glob(os.path.join(annotation_path, '*.xml')))

    # Create directory for errors detected from annotations
    errors_path = os.path.join(output_path, 'errors')
    
    os.makedirs(os.path.join(errors_path, 'bad-annotations'), exist_ok=True)
    
    print('\nAnalyzing the content of all files and storing main info in CSV pandas tables...')
    print(f'Check out the following path: {output_path}')

    # Initialize infos and tracks datframes, as well as dataframes containing bad annotations or missing videos
    all_infos, all_tracks = pd.DataFrame(), pd.DataFrame()
    noann_infos, novid_infos = pd.DataFrame(), pd.DataFrame()
    full_infos = pd.DataFrame()

    # Loop through all annotations (one annotation file per video) and read them one by one
    k = 0
    for file in tqdm.tqdm(annotations, desc='Progress: '):
        # Store the index of the video in "infos"
        info_dict = {'idx': k}

        # Passing the path of the xml document to enable the parsing process
        tree = ET.parse(file)
        root = tree.getroot()

        infos, tracks = pd.DataFrame(), pd.DataFrame()
        frames2check = pd.DataFrame()
        for child in root:
            # Read meta-data first (check structure of xml file to understand better)
            if child.tag == 'meta':
                for subchild in child:
                    if subchild.tag == 'task':
                        for subsubchild in subchild:
                            # Save annotation id
                            if subsubchild.tag == 'id':
                                info_dict['id-annotation'] = int(subsubchild.text)
                            # Save full name of the video on CVAT as well as the video id, date of acquisition and
                            # full name of the video folder
                            elif subsubchild.tag == 'name':
                                name = subsubchild.text
                                info_dict['name-cvat'] = name
                                last_part = name.find('_NOVERLAY')
                                info_dict['id-video'] = int(name[last_part-1:last_part-20:-1][::-1])
                                info_dict['date'] = int(name.split("-")[1])
                                info_dict['name-video'] = f'{info_dict["date"]}-{info_dict["id-video"]}_NOVERLAY-000'
                            # Save the resolution (height and width) of each image (they are all same size)
                            elif subsubchild.tag == 'original_size':
                                for s in subsubchild:
                                    info_dict[s.tag] = int(s.text)
                            # Save the size of the video (number of frames it must contain)
                            elif subsubchild.tag == 'size':
                                info_dict['size'] = int(subsubchild.text)
                        # Store all such data in "infos" dataframe
                        infos = pd.concat([infos, pd.DataFrame.from_dict(info_dict, orient='index').T])
            # Read tracks (boxes coordinates)
            elif child.tag == 'track':
                for subchild in child:
                    # Store x,y top-left and bottom-right coordinates of a box as well as its label and the index of
                    # the video in "tracks"
                    if subchild.tag == 'box':
                        track_dict = {key: int(subchild.attrib[key]) for key in ['frame', 'outside', 'occluded', 'keyframe']}
                        track_dict['xtl'] = float(subchild.attrib['xtl'])
                        track_dict['ytl'] = float(subchild.attrib['ytl'])
                        track_dict['xbr'] = float(subchild.attrib['xbr'])
                        track_dict['ybr'] = float(subchild.attrib['ybr'])
                        track_dict['idx'] = k
                        track_dict['label'] = child.attrib['label']
                        tracks = pd.concat([tracks, pd.DataFrame.from_dict(track_dict, orient='index').T])
                    # For polygonal annotations, build a box enclosing the polygon and store its x,y top-left and
                    # bottom-right coordinates as well as its label and the index of the video in "tracks"
                    elif subchild.tag == 'polygon':
                        track_dict = {key: int(subchild.attrib[key]) for key in ['frame', 'outside', 'occluded', 'keyframe']}
                        points = list(map(lambda xy: (int(round(float(xy.split(',')[0]))), int(round(float(xy.split(',')[1])))),
                                          subchild.attrib['points'].split(';')))
                        x, y, w, h = cv2.boundingRect(np.array([points, ]))
                        track_dict['xtl'] = float(x)
                        track_dict['ytl'] = float(y)
                        track_dict['xbr'] = float(x+w)
                        track_dict['ybr'] = float(y+h)
                        track_dict['idx'] = k
                        track_dict['label'] = child.attrib['label']
                        tracks = pd.concat([tracks, pd.DataFrame.from_dict(track_dict, orient='index').T])

                        # Store info of the video having polygonal annotations in a "frames2check" dataframe
                        check_dict = subchild.attrib
                        check_dict['id-annotation'] = info_dict['id-annotation']
                        check_dict['id-video'] = info_dict['id-video']
                        frames2check = pd.concat([frames2check, pd.DataFrame.from_dict(check_dict, orient='index').T])
        infos = infos.set_index('idx')
        full_infos = pd.concat([full_infos, infos])

        # Check whether the video folder contains incorrect number of frames and store its actual size (number of frames
        # in its folder) in the missing-videos dataframe
        images = sorted(glob.glob(os.path.join(video_path, infos['name-video'].values[0], '*.png')))
        if int(infos['size'].iloc[0]) != len(images):
            infos['actual-size'] = len(images)
            novid_infos = pd.concat([novid_infos, infos])
        # Check if there is no annotation available for the given video and store its info in bad-annotations dataframe
        elif tracks.empty:
            noann_infos = pd.concat([noann_infos, infos])
        # Store all results in the "infos" and "tracks" dataframes if the previous two conditions are not verified
        else:
            tracks = tracks.sort_values('label').set_index('idx')
            all_infos = pd.concat([all_infos, infos])
            all_tracks = pd.concat([all_tracks, tracks])

        # Store info on frames with polygonal annotations (if any)
        if not frames2check.empty:
            frames2check.to_csv(os.path.join(errors_path, 'bad-annotations', os.path.split(file)[-1][:-4]+'.csv'))

        k += 1

        
    full_infos.to_csv(os.path.join(output_path, 'full-infos.csv'))
    all_infos.to_csv(os.path.join(output_path, 'infos.csv'))
    all_tracks.to_csv(os.path.join(output_path, 'tracks.csv'))


    if not noann_infos.empty:
        noann_infos.to_csv(os.path.join(errors_path, 'no-annotations-infos.csv'))
    if not novid_infos.empty:
        novid_infos.to_csv(os.path.join(errors_path, 'no-videos-infos.csv'))

    return full_infos, [all_infos, all_tracks], [noann_infos, novid_infos]


def plot_statistics(output_path: str):
    """
    Generate and save some plots of data statistics in a 'figs' folder within output_path
    """

    figs_path = os.path.join(output_path, 'figs')
    os.makedirs(figs_path, exist_ok=True)

    all_infos = pd.read_csv(os.path.join(output_path, 'infos.csv'))
    all_tracks = pd.read_csv(os.path.join(output_path, 'tracks.csv'))
    # full_infos = pd.read_csv(os.path.join(output_path, 'full-infos.csv'))
    # noann_infos = pd.read_csv(os.path.join(errors_path, 'no-annotations-infos.csv'))
    # novid_infos = os.path.join(errors_path, 'no-videos-infos.csv')

    indices = np.array(list(set(all_tracks['idx'])), dtype=int)
    labels = list(set(all_tracks['label']))


    with open(os.path.join(main_path, 'exams.json'), 'r') as f:
        exams_per_patient = json.load(f)
    patients = list(range(len(exams_per_patient)))
    num_exams = np.zeros(len(patients), dtype=int)
    for pat in patients:
        num_exams[pat] = len(exams_per_patient[pat])


    hist, bins = np.histogram(num_exams, bins=num_exams.max(), range=(1, num_exams.max() + 1))
    plt.figure()
    plt.xlabel('# scans (exams)')
    plt.ylabel('# patients')
    plt.plot(bins[:-1], hist)
    plt.savefig(os.path.join(figs_path, 'exams-per-patient.png'))
    plt.close()

    plt.figure()
    all_infos['size'].plot(kind='kde')
    plt.savefig(os.path.join(figs_path, 'annot-size-distr.png'))
    plt.close()
    plt.figure()
    all_infos.hist(column='size', bins=30, grid=False, figsize=(12, 8), color='#86bf91', zorder=2, rwidth=0.9)
    plt.savefig(os.path.join(figs_path, 'annot-size-hist.png'))
    plt.close()

    with open(os.path.join(figs_path, 'labels.txt'), 'w') as fp:
        fp.write('\n'.join(labels))
    all_tracks['label'].value_counts(normalize=True).to_csv(os.path.join(figs_path, 'labels-represented.csv'))

    res = []
    for idx in indices:
        temp1 = all_tracks.loc[all_tracks['idx'] == idx][['label', 'frame']]
        repeating_frames = temp1['frame'].value_counts()
        res.extend(repeating_frames)
    res = np.array(res)
    a, b = np.histogram(res, bins=res.max(), range=(1, res.max() + 1))
    plt.figure()
    plt.plot(b[:-1], a)
    plt.savefig(os.path.join(figs_path, 'labels-per-frame-hist.png'))
    plt.close()

    infos = all_infos.set_index('idx')
    df = pd.DataFrame(index=indices, columns=list(labels), dtype=int)
    df_norepeats = pd.DataFrame(index=indices, columns=list(labels) + ['repeated ' + label for label in labels],
                                dtype=int)
    for label in labels:
        temp = all_tracks[['label', 'idx', 'frame']].loc[all_tracks['label'] == label][['idx', 'frame']]
        frames_per_idx = temp['idx'].value_counts(sort=False)
        df[label].loc[frames_per_idx.index] = frames_per_idx
        repeats = pd.Series(index=indices, dtype=int)
        for idx in frames_per_idx.index:
            repeats.loc[idx] = temp.loc[temp['idx'] == idx].duplicated(subset=['frame']).sum()

        df_norepeats[label].loc[frames_per_idx.index] = frames_per_idx
        df_norepeats['repeated ' + label].loc[frames_per_idx.index] = repeats.loc[frames_per_idx.index]
    df = df.fillna(0).astype(int)
    df['labels present'] = df[labels].astype(bool).T.sum()
    df['% annotated frames'] = df[labels].T.sum() / all_infos['size'] * 100
    df['% annotated frames'] = df['% annotated frames'].fillna(0)
    df[['id-annotation', 'id-video']] = infos[['id-annotation', 'id-video']]
    df_norepeats = df_norepeats.fillna(0).astype(int)
    all_repeats = pd.Series(index=indices, dtype=int)
    for idx in indices:
        all_repeats.loc[idx] = all_tracks[['idx', 'frame']].loc[all_tracks['idx'] == idx].duplicated(
            subset=['frame']).sum()

    df_norepeats['repetitions'] = all_repeats
    df_norepeats['labels present'] = df_norepeats[labels].astype(bool).T.sum()
    df_norepeats['% annotated frames'] = (df_norepeats[labels].T.sum() - all_repeats) / all_infos['size'] * 100
    df_norepeats['% annotated frames'] = df_norepeats['% annotated frames'].fillna(0)
    df_norepeats[['id-annotation', 'id-video']] = infos[['id-annotation', 'id-video']]

    df.to_csv(os.path.join(figs_path, 'frames-per-label-per-video.csv'))
    df[labels].sum().to_csv(os.path.join(figs_path, 'frames-per-label.csv'))
    df[labels].astype(bool).sum().to_csv(os.path.join(figs_path, 'videos-per-label.csv'))

    df_norepeats.to_csv(os.path.join(figs_path, 'frames-per-label-per-video_norepeats.csv'))
    df_norepeats[labels].sum().to_csv(os.path.join(figs_path, 'frames-per-label_norepeats.csv'))
    df_norepeats[labels].astype(bool).sum().to_csv(os.path.join(figs_path, 'videos-per-label_norepeats.csv'))

    plt.figure()
    df.hist(column='labels present', bins=len(labels), grid=False, figsize=(12, 8), color='#86bf91', zorder=2,
            rwidth=0.9)
    plt.savefig(os.path.join(figs_path, 'labels-per-video.png'))
    plt.close()
    plt.figure()
    df.hist(column='% annotated frames', bins=len(labels), grid=False, figsize=(12, 8), color='#86bf91', zorder=2,
            rwidth=0.9)
    plt.savefig(os.path.join(figs_path, 'annotframes-per-video.png'))
    plt.close()

    plt.figure()
    df_norepeats.hist(column='labels present', bins=len(labels), grid=False, figsize=(12, 8), color='#86bf91', zorder=2,
                      rwidth=0.9)
    plt.savefig(os.path.join(figs_path, 'labels-per-video_norepeats.png'))
    plt.close()
    plt.figure()
    df_norepeats.hist(column='% annotated frames', bins=len(labels), grid=False, figsize=(12, 8), color='#86bf91',
                      zorder=2, rwidth=0.9)
    plt.savefig(os.path.join(figs_path, 'annotframes-per-video_norepeats.png'))
    plt.close()


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

    output_path = os.path.join(main_path, 'summary')
    if os.path.exists(output_path):
        try:
            shutil.rmtree(output_path)
            print(f"Directory '{output_path}' and its contents successfully deleted.")
        except OSError as e:
            print(f"Error: {e}")


    parse_data(video_path, annotation_path, output_path)
    plot_statistics(output_path)
