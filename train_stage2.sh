CUDA_VISIBLE_DEVICES=0 python\
 trainer_stage2.py\
 --save_path train_log/SwinB-Stage2\
 --save_freq 19\
 --train_folder Datasets/train_images\
 --test_folder Datasets/test_images\
 --batch_size 32\
 --epochs 19\
 --model_name SwinB\
 --load_path train_log/SwinB-Stage1\
 --sample_num 62\
 --a_sim 0.5
