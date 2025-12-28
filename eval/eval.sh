# DTW
python evaluate.py --data_path ../Generated/result --metric DTW

# CS
python evaluate.py --data_path ../Generated/result --metric Content_score --pretrained_model evaluators/mainuscript/trad_CS_UWSC_best.pth

# GS
python gs.py --gen_dir ../Generated/result/test2 --real_dir ../Generated/result/test2_gt

# FID
python -m pytorch_fid ../Generated/result/test2_gt ../Generated/result/test2

