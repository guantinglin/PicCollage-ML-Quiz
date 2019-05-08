# Correlation-estimation-with-CNNs

## How to train?

First, prepare the data that needed. You can download it from [here](https://drive.google.com/file/d/1kNgjfb3FF4pnGO__wy0hCgekKgfv5VMa/view) and ][here](https://drive.google.com/file/d/1iUuhI78_8SW9MC6QB9wQeo0kjuAbk6AD/view)

After downloading, run these commands 
```
unzip train_imgs.zip
unzip train_responses.csv.zip
mkdir data
python3 create_datasets.py
```

Then start training
```
python3 train.py
```

This model should reach the final output on the testing split with L1 regression loss = 0.0075 and L2 regression loss = 0.0001
