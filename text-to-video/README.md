## Introduction
Text to video AI models.
## Implementation
Install ffmpeg: 
```
sudo apt update
sudo apt-get install ffmpeg
```
Install dependencies: 
```
pip install -r requirement.txt
```
### Data preparation
Download UCF101 - Action Recognition Data Set [here](https://www.crcv.ucf.edu/data/UCF101.php), and put it into raw_data.
### Resize videos

```
python resize.py
```
### Train models
If using GPUs then don't need to set --cuda, --niter = number of epoches:
```
python train.py --cuda -1 --ngpu 1 --niter 1 --pre-train -1 
```
