# Standard-Plane Classification in US Liver Image Sequences

A novel screening protocol of the liver with ultrasound (US) imaging devices is currently being proposed by SIRM
(the italian society of medical radiology), in collaboration with Esaote (company producing medical devices).
The protocol is based on a set of 5 standardized probe movements, hence providing a same number of US videos which 
represent scans of single liver parts. The operator - commonly a nurse - should detect the most informative images 
within each US video, in order to later provide them to physicians for diagnostic purposes. Such frames are known as 
standard planes and are identified by the presence of specific anatomical structures within the image. 
Given the nature of this imaging technique (being highly noisy and subject to device settings and manual skills of 
the operator) and the resulting challenge of recognizing anatomical structures (often not clearly visible even by expert 
physicians), the standard plane detection task is non-trivial and strongly operator dependent. Nonetheless, 
one aspect that seems to aid expert users is the temporal evolution of the data within the performed motion scan 
(combined with some prior background knowledge of human anatomy). Our aim is hence to develop a deep learning pipeline 
for the automatic classification of single frames (standard planes) within US image sequences.  

We start by following a 2D approach with a 2D CNN architecture named SonoNet [[1]](#1), which proved to achieve 
state-of-the-art results on US fetal standard plane detection task. As a first later approach concerning the usage of time information, 
instead, we propose to employ a 3D CNN model in order to exploit both spatial and temporal information on a short timescale. 
Specifically, we implemented a 3D extension of the mentioned SonoNet architecture. Extending convolutions 
to the third (temporal) domain should aid the network in solving ambiguous situations where some parts of anatomical 
structures are not clearly visible (or partly occluded) within a single frame, though could appear in nearby frames.
Based on [[2]](#2) we also implemented SonoNet(2+1)D model. It is a 3D version of SonoNet2D, but each 2D convolution layers
is replaced with a SpatioTemporal block, which consists of 2D convolution layer followed by a 1D convolution layer, with a ReLU activation function between these two layers.
In this way we have a model which is comparable to the SonoNet2D, in terms of trainable parameters, but with a number of non-linear operations which is the double respect with the 3D model, 
potentially leading to best results. 


------

------

## Dataset Information

The dataset we adopt is made of US videos acquired during liver screening sessions performed by various (undefined) 
number of operators from 4 different acquisition centres. Videos have been annotated with bounding boxes surrounding 
11 interesting anatomical structures (plus an additional generic "Altro" box):

    Cuore
    Vena Cava
    Rene
    Fine fegato
    Stomaco
    Ilo
    Colecisti
    Diramazione sovraepatiche
    Diramazione Porta
    Ramo dx. Porta
    Ramo sx. Porta

The current version of the dataset comprises 2093 videos (>350k images) from 413 patients. 
Videos have a different number of frames, and all frames have a resolution of 1200x760.
Of all such exams, 1845 are coherently annotated with bounding boxes, 241 have some polygonal-shaped annotations, and 
7 are not annotated. The last 7 exams are later excluded, while for the 241 videos, the box coordinates are obtained 
from each polygon (by building the smallest box enclosing such a polygon).

As mentioned before, the protocol requires 5 scans (hence 5 US videos) per patient, although this is not always the case
for all of them. Specifically:
- 116 videos correspond to incomplete or over-complete exams (i.e. 33 patients): 10 patients (65 videos) have more 
than 5 exams (6, 7, or 10), and 23 (51 videos) have fewer (from 1 to 4).
- 20 additional incomplete exams (5 patients) due to missing correspondences of such videos in the actual dataset.
- 77 videos have no information on the patient (7 of which have no annotations and are therefore completely excluded).
- 1880 videos remain, for which the corresponding 375 patients have all 5 exams.

Therefore, the full dataset is made of 2086 videos (116+20+77-7+1880) from 413 patients (some have no patient info).
Therefore, by using a 15% test split, we end up with 56 patients with complete exams (280 videos) in the test set,
and 320 complete (1600 videos) plus 37 incomplete/over-complete/missing-patient (206 videos) in the training set.
Note that no information is available on the specific type of scan each video represents.

------

------

## Classes Description

The current (rough) version of the protocol defines 6 main image classes based on the presence of specific 
anatomical structures within a frame. The rules for defining these 6 standard planes are the following:

    0 - "diramazione-porta": Simultaneous presence of at least two of the landmarks "Ramo dx Porta", "Ramo sx Porta", 
        and "Dir. Porta". It represents the starting point of the first three scans. They will not be so distinguishable
        but labels are too noisy to go into detail.

    1 - "confluenza-sovraepatiche": Simultaneous presence of "Vena cava" and at least one "Dir. Sovraepatiche" landmark.

    2 - "rapporto-epatorenale": Presence of the landmark "Rene".

    3 - "cuore": Presence of the landmark "Cuore". If in co-presence with the class "confluenza-sovraepatiche", the 
        latter wins.

    4 - "colecisti": Lonely presence of the landmark "Colecisti". As for the previous label, if there's another valid 
        standard plane, the latter wins.

    5 - "ilo": Presence of the landmark "Ilo".

    6 - "other": All images that don't satisfy any of the previous conditions.

------

------

## Code Organization

The project is divided in 2 folders:

- **<u>data</u>**: prepare data for 2D and 3D model training starting from the raw US liver dataset.
- **<u>models</u>**: define and train 2D-SonoNet architectures, as well as 3D-SonoNet and (2+1)D-SonoNet extensions.

Let's see them in more detail:

### **<u>data</u>**
This folder contains all scripts for preparing the dataset and viewing some statistics. 
The raw dataset must be contained in 2 folders **<em>videos</em>** and **<em>annotations</em>** within the directory 
**<em>data_directory</em>**. The first (videos) folder should contain N folders (one per video) 
denominated as {date of acquisition}-{video id}_NOVERLAY-000 (e.g. 20220323-6378366256915869660_NOVERLAY-000) and such 
folders should contain PNG images having the same name of the folder but with the last 3 digits representing the frame 
number within the sequence (e.g. 20220323-6378366256915869660_NOVERLAY-023 for the 24th frame). The second (annotations) 
folder, instead, should contain N files XML (one per video), denominated as {annotation id}.xml (e.g. 371.xml); the 
corresponding video (i.e. the video id corresponding to such annotation id) should be specified within the XML file.
There must also be an **<em>exams.json</em>** file, containing a list of lists dividing videos (their id) based on the patient 
from which they were obtained.

Scripts must be executed in the following order:

> - **_0-explore-data.py_**: this script reads XML annotation files of all videos and creates two CSV files 
> "infos.csv" and "tracks.csv" in **<em>data_directory/summary</em>**. Such files summarize all 
> annotations and exams information (for valid data only, e.g. videos without annotations are discarded and not included 
> in the CSV summary). The script also generates some plots of data statistics and stores information on some issues 
> within the data.
> - **_1-split-traintest.py_**: split the data in train and test sets within **<em>data_directory/split</em>**. 
> We put in the test set the last 15% of patients (in chronological order and having all 5 scans expected in the protocol) 
> from all 4 clinical centers (Aquila, Firenze, Palermo, Sassari). All the remaining data goes in the training set. 
> Read the "Dataset Information" section above for more details. Note that this script only creates 2 CSV files named
> **<em>train-infos.csv</em>** and **<em>test-infos.csv</em>** (storing all information required to retrieve videos and 
> annotations from the two sets) within the directory **<em>data_directory/split</em>**, actual 
> data is then replicated from the next script divided in **<em>train</em>** and **<em>test</em>** folders.
> - **_2-split-classes.py_**: populates **<em>train</em>** and **<em>test</em>** folders within **<em>data_directorysplit</em>**.
> Each of them contains 3 folders: **<em>videos</em>** (where each exam is a video folder with PNG images, same as for 
> raw data), **<em>annotations</em>** (where each exam is an XML file), and **<em>labels</em>** (where each exam is a 
> video folder with a TXT file per image... yeah, that was not memory-efficient).
> - **_3-prepare-data2d.py_**: create **<em>data_directory/2d-split</em>** and populate its
> **<em>train</em>** and **<em>test</em>** directories with 7 folders (named from **<em>0</em>** to **<em>6</em>**). 
> Each folder contains only the PNG images passing the time sub-sampling procedure: we take both frames within a video
> sequence for which the SSIM value is lower than the average SSIM throughout the whole video.

### **<u>models</u>**
This folder contains main scripts for running experiments with different models. See the "usage" note at the beginning 
of each of them.
> - **_sononet2d-traintest.py_**: train and test the 2D SonoNet-16/32/64 model.
> - **_sononet2d-traintest_3d_comparable.py_**: trains and evaluates the 2D SonoNet-16/32/64 model using the same dataset as the 3D models for direct comparison.
> - **_sononet3d-traintest.py_**: train and test the 3D SonoNet-16/32/64 model.
> - **_temporal_test.py_**: loads a test video and visualizes the predictions of different models for temporal comparison.
> - **_2d_vs_3d_**: computes per-video accuracy on the test set for both 2D and 3D models, and calculates the average accuracy..

Such scripts use code from the following Python packages:

> **<u>utils</u>**:
> This folder contains python files with many general-purpose utility functions.
>> - **_augments.py_**: defines data augmentation methods for US images.
>> - **_datareader.py_**: defines a class for loading either the 2D or the 3D version of our dataset.
>> - **_datasplit.py_**: defines functions for splitting the dataset into training and validation sets.
>> - **_iterators.py_**: define basic training and testing loops for a single epoch.
>> - **_runner.py_**: defines train and test functions.
>> - **_visualize.py_**: defines a useful function for plotting a confusion matrix and saving it as a PNG image.
>
> **<u>sononet2d</u>**:
> This folder contains the 2D implementation of the SonoNet-16/32/64 model.
>> - **_models.py_**: defines the SonoNet2D class. The number of features in the hidden layers of the network can be 
>> set by choosing between 3 configurations (16, 32, and 64). The network may be used in "classification mode"
>> (the output is given by the adaptation layer) or for "feature extraction" (no adaptation layer is defined and the 
>> output is the set of features in the last convolutional layer): this last functionality is achieved by setting the 
>> _features_only_ parameter to True (useful to check on which image parts the network is focusing its attention). 
>> Finally, by setting the _train_classifier_only_ parameter to True it is possible to freeze learning in all 
>> convolutional layers (only the adaptation layer will be trained).
>> - **_remap-weights.py_**: convert SonoNet weights (downloaded from the [reference repository](https://github.com/rdroste/SonoNet_PyTorch))
>> to be compatible with our implementation of the model.
>
> **<u>sononet3d</u>**:
> This folder contains the 3D and (2+1)D extensions of the standard SonoNet-16/32/64 model implementation.
>> - **_models.py_**: defines the SonoNet3D and SonoNet(2+1)D classes. For the SonoNet3D all 2D convolutional and pooling layers are changed to 
>> their 3D extension. Instead, in the (2+1)D model, the 3D convolutional layers are replaced by a SpatioTemporal block, where the standard convolution
>> is decomposed into a 2D convolution followed by a 1D convovolution. As for the 2D case, the number of features in the hidden layers of the network can be 
>> set by choosing between 3 configurations (16, 32, and 64).
>> 

Results of each experiment are stored in the following folder:

> **<u>logs</u>**:
> This folder contains weights of all trained models, as well as test evaluation results and confusion matrices plots.
> Note: for all experiments on our dataset, the following parameters have been tweaked from their default values 
> (unless differently specified): batch_size=128, lr=0.001, max_num_epochs=100, (early-stopping) patience=10, lr_sched_patience=4. 
> Also note that this folder has not been uploaded on GitHub due to its large size, but it is available on the server.
>> 
>> SonoNet pre-trained weights:
>> - **<u>weights4sononet2d</u>** / **<u>FetalDB</u>**: pretrained weights of all SonoNet configurations (16, 32, and 64 initial 
>> features) from the FetalDB dataset. Each configuration has its own folder (SonoNet-16, SonoNet-32, and SonoNet-64) 
>> where weights are stored in "ckpt_best_loss.pth" file. Such files were obtained from those denoted as "old", which 
>> are the ones provided in [this repository](https://github.com/rdroste/SonoNet_PyTorch) (same weights but not directly 
>> compatible with our model definition).
>> SonoNet checkpoints:
>> - **<u>sononet2d-scratch-noreg</u>**: Weights of both SonoNet-16 (num_features=16) and SonoNet-64 (num_features=64) 
>> configurations trained from scratch on our dataset.
>> - **<u>sononet2d-pretrain-noreg</u>**: Weights of SonoNet-64 pretrained on FetalDB and fully fine-tuned on our dataset 
>> (num_features=64, pretrain_dir='./logs/weights4sononet/FetalDB')
>> - **<u>sononet2d-pretrain</u>**: Weights of SonoNet-64 pretrained on FetalDB and fully fine-tuned on our dataset 
>> using some regularization (num_features=64, pretrain_dir='./logs/weights4sononet/FetalDB', weight_decay=0.0001)
>> - **<u>sononet2d-pretrain-lastlayer</u>**: Weights of SonoNet-64 pretrained on FetalDB and with the classifier head 
>> (adaptation layer) fine-tuned on our dataset using some regularization (num_features=64, 
>> pretrain_dir='./logs/weights4sononet/FetalDB', weight_decay=0.0001, train_classifier_only=True)

------

------

## References

<a id="1">[1]</a>
Baumgartner C.F., Kamnitsas K., Matthew J., Fletcher T.P., Smith S., Koch L.M., Kainz B., and Rueckert D. (2017). 
SonoNet: real-time detection and localisation of fetal standard scan planes in freehand ultrasound. 
IEEE transactions on medical imaging, 36(11), pp.2204-2215.
[[link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7974824)]

<a id="2">[2]</a>
Tran, D., Wang, H., Torresani, L., Ray, J., LeCun, Y., & Paluri, M. (2018). 
A closer look at spatiotemporal convolutions for action recognition. 
In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition (pp. 6450-6459).
[[link](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tran_A_Closer_Look_CVPR_2018_paper.pdf)]
