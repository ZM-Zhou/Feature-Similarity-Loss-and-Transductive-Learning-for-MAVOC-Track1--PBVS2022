CUDA_VISIBLE_DEVICES=0 python\
 trainer_stage2.py\
 --test_only True\
 --test_folder Datasets/test_images\
 --batch_size 128\
 --save_path trained_model