<div align="center">

# Real3D: Scaling Up Large Reconstruction Models with Real-World Images


<p align="center">
    <a href="https://hwjiang1510.github.io/">Hanwen Jiang</a>,
    <a href="https://www.cs.utexas.edu/~huangqx/">Qixing Huang</a>,
    <a href="https://geopavlakos.github.io/">Georgios Pavlakos</a>
</p>


</div>

--------------------------------------------------------------------------------

<div align="center">
    <a href="https://hwjiang1510.github.io/Real3D/"><strong>Project Page</strong></a> |
    <a href="https://arxiv.org/abs/2406.08479"><strong>Paper</strong></a> | 
    <a href="https://huggingface.co/spaces/hwjiang/Real3D"><strong>HuggingFace Demo</strong></a>
</div>

<br>

![overview.png](assets/overview.png "overview.png")

**Abstract**: As single-view 3D reconstruction is ill-posed due to the ambiguity from 2D to 3D, the reconstruction models have to learn generic shape and texture priors from large data. The default strategy for training single-view Large Reconstruction Models (LRMs) follows the fully supervised route, using synthetic 3D assets or multi-view captures. Although these resources simplify the training procedure, they are hard to scale up beyond the existing datasets and they are not necessarily representative of the real distribution of object shapes. To address these limitations, in this paper, we introduce Real3D, the first LRM system that can be trained using single-view real-world images. Real3D introduces a novel self-training framework that can benefit from both the existing 3D/multi-view synthetic data and diverse single-view real images. We propose two unsupervised losses that allow us to supervise LRMs at the pixel- and semantic-level, even for training examples without ground-truth 3D or novel views. To further improve performance and scale up the image data, we develop an automatic data curation approach to collect high-quality examples from in-the-wild images. Our experiments show that Real3D consistently outperforms prior work in four diverse evaluation settings that include real and synthetic data, as well as both in-domain and out-of-domain shapes.


## Installation
First, install the environment.
```
conda create --name real3d python=3.8
conda activate real3d

# Install pytorch, we use:
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt 
```

Then, download the [model weight](https://huggingface.co/hwjiang/Real3D/resolve/main/model_both_trained_v1.ckpt?download=true) and put it to `./checkpoints/model_both_trained_v1.ckpt`.


## Demo
Use `./run.sh` and modify your image path and foreground segmentation config accordingly. Tune the chunk size to fit your GPU.


## Training
### Data preparation
This repo uses MVImgNet, CO3D, OmniObject3D and our collected real images. Please see [this file](./assets/data_preparation.md).

### Step 0: (Optional) Fine-tune TripoSR
As TripoSR predicts 3D shapes with randomized scales, we first need to fine-tune it on Objaverse. We provide the [fine-tuned model weight](https://huggingface.co/hwjiang/Real3D/resolve/main/model_both.ckpt?download=true), so you can put it to `./checkpoint/model_both.ckpt` and skip this stage.

### Step 1: Self-training on real images
Use `./train_sv.sh`.


## Evaluation
Use `./eval.sh` and modify the script and config accordingly.
For example, to evaluate on CO3D with ground-truth multiviews, use `eval_mv.py` and `./config/eval/eval_mv_co3d.yaml`. To evaluate on single-view images, use `eval_sv.py` and `./config/eval/eval_sv.yaml`.


## TODO
- [ ] Release real-world data.


## Acknowledgement
This repo is developed based on [TripoSR](https://github.com/VAST-AI-Research/TripoSR/).


## BibTex
```
@article{jiang2024real3d,
   title={Real3D: Scaling Up Large Reconstruction Models with Real-World Images},
   author={Jiang, Hanwen and Huang, Qixing and Pavlakos, Georgios},
   booktitle={arXiv preprint arXiv:2406.08479},
   year={2024},
}
```
