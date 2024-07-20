## Introduction
Currently, there are some AI models for generating video from prompts. However, we are particularly impressed with MoCoGAN, which can be trained from text to class to generate video. Our text-to-video generation source code is referenced from [MoCoGAN](https://github.com/CarloP95/mocogan/tree/a71449c0b617265b8c5193449b8121267941bf4c) [[1]](#1).

The setup begins with the installation of essential tools like ffmpeg and necessary Python dependencies. Users are instructed to download and prepare the Actions as Space-Time Shapes Data Set. The guide then details the steps to resize the videos, train the video generator, and train the text-to-class model. Finally, we explain how to combine these models to achieve text-to-video generation. Unit tests and evaluation procedures are included to ensure the models' accuracy and effectiveness. 
## Implementation
### Installation
Install ffmpeg: 
```
sudo apt update
sudo apt-get install ffmpeg
```
Replace np.float and np.int by float and int in ffmpeg.py:
```
vim ~/.local/lib/python3.8/site-packages/skvideo/io/ffmpeg.py
```

Install dependencies: 
```
pip install -r requirement.txt
```
### Data preparation
Download Actions as Space-Time Shapes Data Set [here](https://www.wisdom.weizmann.ac.il/%7Evision/SpaceTimeActions.html), and put it into a folder named: raw_data.
### Resize videos

```
python resize.py
```
### Train video generator
If using GPUs, there is no need to set --cuda:
```
python train.py  --cuda 1 \
--ngpu 7 \
--batch_size 16 \
--n_epochs 10000 \
--lr 0.0002 \
--pre_train -1 \
--i_epochs_saveV 500 \
--i_epochs_checkpoint 100 \
--i_epochs_display 10
```
### Train text to class
In the `text_to_class` folder, run:
```
python train.py --cuda 1 --ngpu 7 \
--n_epochs 10000 \
--batch_size 64 \
--embed_size 512 \
--lr 0.0001 \
--save_interval 100 \
--numClasses 10 \
--path data/action_classes.txt
```
### Text to video
To implement text-to-video generation, combine the text-to-class and video generator models:
```
python main.py \
--cuda 1 \
--ngpu 7 \
--video_path trained_models/VideoGenerator_epoch-120000 \
--text_path text_to_class/LSTM-checkpoint-3700
```
## Run unit tests
### Text to class
To run the unit tests for the text-to-class model:
```
python .\unit_test\text_to_class.test.py
```
### Generate videos
To run the unit tests for generating videos:
```
python .\unit_test\generate_videos.test.py
```

## Evaluate Models
### Text to class
To evaluate the text-to-class model:
```
python .\evaluate_models\eval_text_to_class.py
```
### Generate videos
To evaluate the video generation model:
```
python .\evaluate_models\eval_generate_videos.py
```
## References
<a id="1">[1]</a> 
Tulyakov, S., Liu, M. Y., Yang, X., & Kautz, J. (2018). Mocogan: Decomposing motion and content for video generation. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 1526-1535).