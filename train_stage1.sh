CUDA_VISIBLE_DEVICES=0 python\
 trainer_stage1.py\
 --save_path train_log/SwinB-Stage1\
 --save_freq 100\
 --train_folder Datasets/train_images\
 --test_folder Datasets/test_images\
 --batch_size 32\
 --epochs 100\
 --model_name SwinB