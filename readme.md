# Feature Similarity Loss and Transductive Learning for MAVOC(Track1)-PBVS2022
This is the Pytorch implementation for testing and training the method called in 'Feature Similarity Loss and Transductive Learning for Aerial View Object Classification' for PBVS 2022 Multi-modal Aerial View Object Classification Challenge - Track 1 (SAR).

## Setup
We ran our experiments with Ubuntu 18.04, CUDA 11.0, Python 3.7.9, and Pytorch 1.7.0. For testing or training the method, we recommend creating a virtual environment by [Anaconda](https://www.anaconda.com/products/individual). Please open a terminal in the root of the repository for running the following commands and scripts.
```
conda env create -f environment.yml
conda activate pytorch170cu11
```

## Testing
In order to test the method, you should firstly download the data from the competition [web page](https://codalab.lisn.upsaclay.fr/competitions/1388#participate) and unzip it into `Datasets/test_images/-1`, where `-1` means that they do not have labels. Or you could use the subset of the test data that we have put in the repository for just running the code. Then, please download our pertained models (which achieves the 31.23% top-1 accuracy on the test phase) from [Here](https://drive.google.com/drive/folders/1i-mOleUGNbw66WtraV3GYl5qEwORddsQ?usp=sharing) and put them into `trained_model`.

Finally, you could simply run the script. `CUDA_VISIBLE_DEVICES` in the script is used to specify which GPU to use in a multi-GPU device.

```
bash test.sh
```
On an NVIDIA RTX 2080Ti GPU, the run time for testing one image is about 5.5ms when the batch size is set to 128. You could set a smaller batch size in the script if there is no enough GPU memory, which has no effect on the results but on the run time. After running the above script, the results `results.csv` will be saved in `results/`.
It is noted that if you have trained the model by yourself (described in the next part), please copy the last trained model (`last_backbone.pth` and `last_cls_head.pth`) saved in `train_log/SwinB-Stage2` to the folder `trained_model` before running the script.

## Training
### Dataset prepare
If you want to train the model by yourself,  please firstly [download](https://codalab.lisn.upsaclay.fr/competitions/1388#participate) and unzip the training data into the `Datasets` , and the `Datasets` folder is organized like:
```
Datasets
|---train_images
|   |---0
|   |   |---SAR_xxx.png
|   |   |---...
|   |---1
|   |   |---SAR_xxx.png
|   |   |---...
|   '''
|---test_images
|   |--- -1
|   |    |---SAR_xxx.png
|   |    |---SAR_xxx.png
|   |    |---...
```
It is noted that all the test images are putted into a folder named `-1`, which means that they do not have labels. This operation is used for unifying the folder formats. 

### Pretrained model prepare

Please download the pretraiend Swin backbone from [Here](https://drive.google.com/drive/folders/11ObxezC0S6hcKg2DsmYsXCVgCNWIEW8d?usp=sharing) or the `Swin-B/ImageNet-22k/224x224/22K model` in Table `ImageNet-1K and ImageNet-22K Pretrained Models` provided in the Swin [official implementation](https://github.com/microsoft/Swin-Transformer).
The pretrained backbone `swin_base_patch4_window7_224_22k.pth`  should be putted into the folder `pretrained_backbone`,

### Start training

Then, our method could be trained by sequentially running the scripts. It is noted that a single GPU with at least 11GB memory is required for training our method. 

```
bash train_stage1.sh
bash train_stage2.sh
```
On an NVIDIA RTX 2080Ti GPU, training the method with both two training stages takes approximate 3.5 hours. The models (\*.pth) and the log file (\*.txt) will be saved to `train_log/SwinB-Stage<1/2>`. Moreover, we give our training log files in `train_log_history`.
