# Description 
This model demonstrated the performance of ShelfNets on the people segmantation tasks

# How to run
## Setup
Run this command
```
pythons setup.py install
```

## Download the pre-trained weights 
You can download the pre-trained weights trained on Pascal VOC from [here](https://drive.google.com/drive/folders/1k23TpBDsP9_gnb3LZlEcYyF4yoVzW99Z)

# How to run
```
python3 experiments/segmentation/test_single_scale.py --backbone resnet50 --dataset pascal_voc --resume trained/shelfnet50_weights.tar
```

The output would be saved in outdir foler
