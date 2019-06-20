# Segmentation

Pixel-wise segmentation on the VOC2012 dataset using pytorch, which refers to [piwise](https://github.com/bodokaiser/piwise).

Three networks has been implemented:

- [x] fcn-vgg16-interpolate
- [x] fcn-vgg19-interpolate
- [x] fcn-vgg19-deconv

## Setup

### Download

Download [image archive](http://host.robots.ox.ac.uk/pascal/VOC/) and extract and do:

```
mv VOCdevkit/VOC2012/JPEGImages data/images
mv VOCdevkit/VOC2012/SegmentationClass data/classes
```

### Split train/val set

You can find the original `train.txt`, `val.txt` and `trainval.txt` from

```
VOCdevkit/VOC2012/trainval/VOC2012/ImageSets/Segmentation
```

The original 
$$
train:val \approx 1:1
$$
I re-split train/val set as
$$
train:val \approx 4:1
$$
You can find `my_train.txt`, `my_val.txt`, `trainval.txt` from

```
./data
```

### Install

```
conda create -n ml-py37 python=3.7
conda activate ml-py37
conda install pytorch=1.0
conda install pillow
conda install torchvision
conda install -c conda-forge visdom
```

## Usage

For latest documentation use:

```
python main.py --help
python main.py  train --help
python main.py  test --help
```

### Training

If you want to have visualization open an extra tab with:

```
python -m visdom.server -port 5000
```

Train the fcn-vgg16-interpolate model 100 epochs with cuda support, visualization, checkpoints every 5 epochs and save the model to `./log`, using `my_train.txt` and `my_val.txt`:

```
python main.py --cuda --model=fcn-vgg16-interpolate train --num-epochs=100 --batch-size=1 --epochs-save=5 --log_dir=log --datadir=data --train_list=./data/my_train.txt --val_list=./data/my_val.txt
```

You can also simplify the args because some default parameters are set:

```
python main.py --cuda --model=fcn-vgg16-interpolate train --log_dir=log 
```

If you have two gpus, `--double-cuda` can be opened:

```
python main.py --cuda --double-cudas --model=fcn-vgg19-deconv train --log_dir=log
```

You can also begin with the model trained by last time:

```
python main.py --state=log/fcn-vgg16-interpolate.pth --cuda --model=fcn-vgg16-interpolate train --log_dir=log
```

### Test

```
python main.py --state=log/fcn-vgg19-deconv.pth --cuda --model=fcn-vgg19-deconv test test.jpg label.png
```

The input image will be resized to 256x256 firstly,  and the resized image can be saved by `--resized-image` :

```
python main.py --state=log/fcn-vgg19-deconv.pth --cuda --model=fcn-vgg19-deconv test test.jpg label.png --resized_image=resized_test.jpg
```

If the model is trained by two gpus,  `--double-cudas` is necessary:

```
python main.py --state=log/fcn-vgg19-deconv.pth --cuda --double-cudas --model=fcn-vgg19-deconv test test.jpg label.png --resized_image=resized_test.jpg
```

![result](https://github.com/MondayYuan/FCN/blob/master/fig/total.png)
