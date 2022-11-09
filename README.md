
# STEVE: Slot-Transformer for Videos 

#### [[arXiv](https://arxiv.org/abs/2205.14065)] [[project](https://sites.google.com/view/slot-transformer-for-videos)]

This is the **official PyTorch implementation** of the STEVE model. 

In this branch, we provide the scripts to evaluate and compute the video-level FG-ARI.

### Dataset
Any of the MOVi-A/B/C/D/E datasets with masks can be downloaded using the script `download_movi_with_masks.py`. The remaining datasets used in the paper i.e., CATER-Tex, MOVi-Tex, MOVi-Solid, Youtube-Traffic and Youtube-Aquarium, shall be released soon.

### Evaluation
Following is an example command to run the evaluation script:
```bash
python eval_fgari_video.py --data_path "eval_data/*" --trained_model_paths "saved_model_seed_0.pt" "saved_model_seed_1.pt" "saved_model_seed_2.pt" 
```

### Outputs
The script produces the standard output on the terminal screen showing the FG-ARI mean and standard deviation.

### Citation
```
@article{Singh2022SimpleUO,
  title={Simple Unsupervised Object-Centric Learning for Complex and Naturalistic Videos},
  author={Gautam Singh and Yi-Fu Wu and Sungjin Ahn},
  journal={ArXiv},
  year={2022},
  volume={abs/2205.14065}
}
```