
# STEVE: Slot-Transformer for Videos 

#### [[arXiv](https://arxiv.org/abs/2205.14065)] [[project](https://sites.google.com/view/slot-transformer-for-videos)]

This is the **official PyTorch implementation** of the STEVE model and its training script. We also provide a downloader script for the MOVi datasets.

<img src="https://i.imgur.com/P6seoRd.gif">

While the code is designed to handle videos, it can be run on static images by providing single-frame "videos" as input. This can be seen as SLATE and this was also used as the SLATE baseline in the paper.

### Dataset
Any of the MOVi-A/B/C/D/E datasets can be downloaded using the script `download_movi.py`.

The remaining datasets used in the paper i.e., CATER-Tex, MOVi-Tex, MOVi-Solid, Youtube-Traffic and Youtube-Aquarium, shall be released soon.

### Training
To train the model, simply execute:
```bash
python train.py
```
Check `train.py` to see the full list of training arguments. You can use the `--data_path` argument to point to the path of your dataset.

### Outputs
The training code produces Tensorboard logs. To see these logs, run Tensorboard on the logging directory that was provided in the training argument `--log_path`. These logs contain the training loss curves and visualizations of reconstructions and object attention maps.

### Running as SLATE
To run on images or single frame "videos", set the episode length to 1 by setting the argument `--ep_len 1`. When doing this, you may want to increase the number of slot attention iterations using the argument `--num_iterations`. This is because STEVE typically uses a small value like 1 or 2 because it can use several frames of a video to do more refinement. However, because SLATE has to do refinement using a single frame, one may need to set this to 3 or 7 as prescribed in the original SLATE implementation.

### Code Files
This repository provides the following files.
- `train.py` contains the training script.
- `steve.py` provides the model class for STEVE.
- `data.py` contains the dataset class.
- `download_movi.py` contains code to download the MOVi-A/B/C/D/E datasets.
- `dvae.py` provides the model class for Discrete-VAE.
- `transformer.py` provides the model classes for Transformer.
- `utils.py` provides helper classes and functions.

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