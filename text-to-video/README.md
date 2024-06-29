## Introduction
Text to video AI models.
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
If using GPUs then don't need to set --cuda:
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
In the folder text_to_video, run:
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
To implement text to video, we combine two models: text to class and video generator.
```
python main.py \
--cuda 1 \
--ngpu 7 \
--video_path trained_models/VideoGenerator_epoch-120000 \
--text_path text_to_class/LSTM-checkpoint-3700
```