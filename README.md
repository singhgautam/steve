
# STEVE: Slot-Transformer for Videos 
*NeurIPS 2022*

#### [[arXiv](https://arxiv.org/abs/2205.14065)] [[project](https://sites.google.com/view/slot-transformer-for-videos)]

This is the **official PyTorch implementation** of the STEVE model and its training script. We also provide a downloader script for the MOVi datasets.

<img src="https://i.imgur.com/P6seoRd.gif">

### Authors
Gautam Singh and Yi-Fu Wu and Sungjin Ahn

### Dataset
Any of the MOVi-A/B/C/D/E datasets can be downloaded using the script `download_movi.py`. The remaining datasets used in the paper i.e., CATER-Tex, MOVi-Tex, MOVi-Solid, Youtube-Traffic and Youtube-Aquarium, shall be released soon.

### Training
To train the model, simply execute:
```bash
python train.py
```
Check `train.py` to see the full list of training arguments. You can use the `--data_path` argument to point to the path of your dataset.

### Outputs
The training code produces Tensorboard logs. To see these logs, run Tensorboard on the logging directory that was provided in the training argument `--log_path`. These logs contain the training loss curves and visualizations of reconstructions and object attention maps.

### Packages Required
The following packages may need to be installed first.
- [PyTorch](https://pytorch.org/)
- [TensorBoard](https://pypi.org/project/tensorboard/) for logging.
- [MoviePy](https://pypi.org/project/moviepy/) to produce video visualizations in the tensorboard logs.

### Code Files
This repository provides the following files.
- `train.py` contains the training script.
- `steve.py` provides the model class for STEVE.
- `data.py` contains the dataset class.
- `download_movi.py` contains code to download the MOVi-A/B/C/D/E datasets.
- `dvae.py` provides the model class for Discrete-VAE.
- `transformer.py` provides the model classes for Transformer.
- `utils.py` provides helper classes and functions.

### Evaluation
See the branch named `evaluate` for the evaluation scripts.

### Citation
```
@inproceedings{
  singh2022simple,
  title={Simple Unsupervised Object-Centric Learning for Complex and Naturalistic Videos},
  author={Gautam Singh and Yi-Fu Wu and Sungjin Ahn},
  booktitle={Advances in Neural Information Processing Systems},
  editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year={2022},
  url={https://openreview.net/forum?id=eYfIM88MTUE}
}
```
