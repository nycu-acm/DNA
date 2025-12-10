# DNA: Dual-branch Network with Adaptation for Open-Set Online Handwriting Generation

Accepted in WACV 2026.

## Overview
We introduces a Dual-branch Network with Adaptation (DNA) to address the demanding challenge of unseen Online Handwriting Generation (OHG). Existing OHG approaches often struggle to synthesize characters that were unseen during training (UWUC: Unseen-Writer-Unseen-Characters), particularly in complex glyph-based languages such as Traditional Chinese.
<img width="1793" height="862" alt="image" src="https://github.com/user-attachments/assets/33503e64-5799-4c94-9fe8-3a7655477048" />

## Downloads
Download the pretrained model from https://drive.google.com/drive/folders/1jvRByh6SeR6kwn2hUKP_RHyPp4iYg_8S?usp=drive_link
Traditional Chinese dataset comes from [SCUT-COUCH 2009数据库](http://www.hcii-lab.net/data/SCUTCOUCH/CN/download.html)
Japanese dataset comes from [SDT](https://github.com/dailenson/SDT?tab=readme-ov-file)

## Environment
* Python: 3.8+
* Torch: 1.12.1+

## Command

### Train
    bash train.sh

### Test
    bash test.sh
    
`--cfg`: config file

`--model`: model path

`--save_dir`: save path

`--store_type`: output type (`img`, `online`, `both`)

`--sample_size`: number of characters for each writer

### Evaluate
    bash eval/eval.sh
    
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

## Acknowledgement
[SDT](https://github.com/dailenson/SDT?tab=readme-ov-file)
[SCUT-COUCH 2009数据库](http://www.hcii-lab.net/data/SCUTCOUCH/CN/download.html)
TUAT HANDS dataset
[HTR-best-practices](https://github.com/georgeretsi/HTR-best-practices)

## Citation
    

