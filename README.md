# Online Handwriting Font Generation

## Updates
[24/10/30]: Inference codes.

[24/11/13]: Training codes.

[24/11/27]: Testing codes.

## To be updated
1. 

## Downloads
Download the model and content images from https://drive.google.com/drive/folders/1jvRByh6SeR6kwn2hUKP_RHyPp4iYg_8S?usp=drive_link
Download the well-trained SDT model from https://github.com/dailenson/SDT?tab=readme-ov-file (if needed)

## Command

### Train
    bash train.sh

### Test
    python test.py --save_dir [存放路徑] --model [模型路徑]
    
`--cfg`: config file

`--model`: model path

`--save_dir`: save path

`--store_type`: output type (`img`, `online`, `both`)

`--sample_size`: number of characters for each writer

### Evaluate
    python evaluate.py --data_path [Test存放路徑]
    
### Inference
    python inference.py --model Saved/best.pth --save_dir results/inference_samples --style data/inference_style_samples --store_type img -c [中文字串]

`--cfg`: config file

`--model`: model path

`--save_dir`: save path

`--style`: dir of style images

`--store_type`: output type (`img`, `online`, `both`)

`-c`, `--characters`: characters you want to generate

## Style samples
    Project
    |--- data
           |--- inference_style_samples
                 |--- style1
                 |--- style2
                       |--- img1.png
                       |--- img2.png
                       |--- ...
                 |--- ...
