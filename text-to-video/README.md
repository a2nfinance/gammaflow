## Introduction
Text to video AI models.
## Implementation
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
### Trai models
```
python train.py --cuda -1 --ngpu 1 --niter 1 --pre-train -1 
```
