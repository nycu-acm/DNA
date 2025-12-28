#UWSC
CUDA_VISIBLE_DEVICES=0 python test.py -c character_dict_t_only --sample_size all --store_type both --save_dir Generated/result --model Saved/best.pth

#UWUC
CUDA_VISIBLE_DEVICES=0 python test.py -c character_dict_t_only_unseen --dataset 1 --sample_size all --store_type both --save_dir Generated/result_unseen --model Saved/best.pth



