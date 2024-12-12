
# DiffGlue Implementation

Pytorch implementation of DiffGlue for ACM MM'24 paper "[DiffGlue: Diffusion-Aided Image Feature Matching](https://dlnext.acm.org/doi/10.1145/3664647.3681069)", by [Shihua Zhang](https://scholar.google.com/citations?user=7f_tYK4AAAAJ) and [Jiayi Ma](https://scholar.google.com/citations?user=73trMQkAAAAJ).

In this paper, we propose a novel method called DiffGlue that introduces the Diffusion Model into the sparse image feature matching framework. Concretely, based on the incrementally iterative diffusion and denoising processes, DiffGlue can be guided by the prior from the Diffusion Model and trained step by step on the optimization path, approaching the optimal solution progressively. Besides, it contains a special Assignment-Guided Attention as a bridge to merge the Diffusion Model and sparse image feature matching, which injects the inherent prior into GNN thereby ameliorating the message delivery.

This repository contains the training code on both [SuperPoint](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w9/html/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.html) and [ALIKED](https://ieeexplore.ieee.org/abstract/document/10111017) descriptors, and the evaluation code for homography estimation on [HPatches](https://openaccess.thecvf.com/content_cvpr_2017/html/Balntas_HPatches_A_Benchmark_CVPR_2017_paper.html) and relative pose estimation on [MegaDepth1500](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_MegaDepth_Learning_Single-View_CVPR_2018_paper.html) dataset, all of which are described in our paper.

If you find this project useful, please cite:

```
@inproceedings{zhang2024diffglue,
  title={DiffGlue: Diffusion-Aided Image Feature Matching},
  author={Zhang, Shihua and Ma, Jiayi},
  booktitle={Proceedings of the ACM International Conference on Multimedia},
  pages={8451--8460},
  year={2024}
}
```

## Requirements

Environment with Python 3.9, PyTorch 2.1.0, and CUDA 12.1 is recommended. Other versions might work, but lower versions might bring a slight performance drop (*e.g.*, when using PyTorch 2.0.1 and CUDA11.8, the AUC@5&deg; on MegaDepth1500 with SuperPoint and RANSAC for relative pose estimation drops ~0.5, from ~50.2 to ~49.7).

All training and testing processes can be performed on at most 2x NVIDIA RTX 3090 GPUs with 24GB of VRAM each.

We recommend a RAM size of 192GB or more to ensure that all training processes can run without unexpected interruption.

## Quick Start

After creating and activating a practicable environment, our repository and other dependencies can be easily downloaded and installed through git and pip.

```
git clone https://github.com/SuhZhang/DiffGlue
cd DiffGlue
pip install -r requirements.txt
```

Then download the pre-trained models from [here](https://drive.google.com/drive/folders/1YHd7MJaKki7e5wHqepXJLVboGYxmyRf2?usp=sharing). All weight files should be saved in `./demo/models/weights/`.

You can run the feature matching process with `demo.py`.

```
cd ./demo && python demo.py
```

## Datasets

All datasets in this repository have auto-downloaders. You can also find the URL links of datasets in the corresponding files under `./DiffGlue/scripts/datasets/`.

## Preparations

Before the training and testing process, some preparations are needed.

First, navigate to `./DiffGlue/scripts/`. Then, run `settings.py` to set the paths for datasets and outputs.

```
cd ./DiffGlue/scripts && python settings.py
```

## Training

As mentioned in the paper, we adopt a two-stage training:

1. Pre-train on a large dataset of synthetic homographies applied to internet images. We use the 1M-image distractor set of the [Oxford-Paris](https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Oxford_and_CVPR_2018_paper.html) retrieval dataset. It requires ~450 GB of disk space.
2. Fine-tune on the [MegaDepth](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_MegaDepth_Learning_Single-View_CVPR_2018_paper.html) dataset, which is based on PhotoTourism pictures of popular landmarks around the world. It exhibits more complex and realistic appearance and viewpoint changes. It requires ~420 GB of disk space.

### First Training Stage

Run the script to start training for the first training stage on SuperPoint.

```
python -m scripts.train SP+DiffGlue_homography --conf scripts/configs/superpoint+diffglue_homography.yaml --run_benchmarks
```

Training on ALIKED is conducted similarly.

```
python -m scripts.train ALIKED+DiffGlue_homography --conf scripts/configs/aliked+diffglue_homography.yaml --distributed True --run_benchmarks
```

### Second Training Stage

To speed up the training on MegaDepth, we suggest to cache the local descriptions before training (requires ~150 GB of disk space).

```
python -m scripts.cache.export_megadepth --method sp --num_workers 8
```

For ALIKED, change the `--method` option.

```
python -m scripts.cache.export_megadepth --method aliked --num_workers 8
```

Then run the training process on SuperPoint,

```
python -m scripts.train SP+DiffGlue_megadepth --conf scripts/configs/superpoint+diffglue_megadepth.yaml train.load_experiment=SP+DiffGlue_homography data.load_features.do=True --run_benchmarks --distributed
```

or ALIKED.

```
python -m scripts.train ALIKED+DiffGlue_megadepth --conf scripts/configs/aliked+diffglue_megadepth.yaml train.load_experiment=ALIKED+DiffGlue_homography data.load_features.do=True --run_benchmarks --distributed
```

Running this fine-tuning process directly is also available.

```
python -m scripts.train SP+DiffGlue_megadepth --conf scripts/configs/superpoint+diffglue_megadepth.yaml train.load_experiment=SP+DiffGlue_homography --run_benchmarks --distributed
```

Or

```
python -m scripts.train ALIKED+DiffGlue_megadepth --conf scripts/configs/aliked+diffglue_megadepth.yaml train.load_experiment=ALIKED+DiffGlue_homography --run_benchmarks --distributed
```

## Testing

We provide evaluation scripts for homography estimation on [HPatches](https://openaccess.thecvf.com/content_cvpr_2017/html/Balntas_HPatches_A_Benchmark_CVPR_2017_paper.html) (~1.8 GB) and relative pose estimation on [MegaDepth1500](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_MegaDepth_Learning_Single-View_CVPR_2018_paper.html) (~1.5GB).

### HPatches

You can easily evaluate the pre-trained SuperPoint+DiffGlue model on HPatches,

```
python -m scripts.eval.hpatches --conf superpoint+diffglue-official --checkpoint ../demo/models/weights/SP_DiffGlue.tar
```

and the pre-trained ALIKED+DiffGlue model.

```
python -m scripts.eval.hpatches --conf aliked+diffglue-official --checkpoint ../demo/models/weights/ALIKED_DiffGlue.tar
```

### MegaDepth1500

You can also easily evaluate the pre-trained SuperPoint+DiffGlue model on MegaDepth1500,

```
python -m scripts.eval.megadepth1500 --conf superpoint+diffglue-official --checkpoint ../demo/models/weights/SP_DiffGlue.tar
```

and the pre-trained ALIKED+DiffGlue model.

```
python -m scripts.eval.megadepth1500 --conf aliked+diffglue-official --checkpoint ../demo/models/weights/ALIKED_DiffGlue.tar
```

## Acknowledgement

This code is partly borrowed from [LightGlue](https://github.com/cvg/LightGlue), [glue-factory](https://github.com/cvg/glue-factory), [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork), [DDPM](https://github.com/hojonathanho/diffusion), and [DDIM](https://github.com/ermongroup/ddim). If using the code related to data generation, training, and evaluation, please cite these papers.

```
@inproceedings{lindenberger2023lightglue,
  title={Lightglue: Local feature matching at light speed},
  author={Lindenberger, Philipp and Sarlin, Paul-Edouard and Pollefeys, Marc},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={17627--17638},
  year={2023}
}
@inproceedings{detone2018superpoint,
  title={Superpoint: Self-supervised interest point detection and description},
  author={DeTone, Daniel and Malisiewicz, Tomasz and Rabinovich, Andrew},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={224--236},
  year={2018}
}
@article{ho2020denoising,
  title={Denoising diffusion probabilistic models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={6840--6851},
  year={2020}
}
@inproceedings{
  song2021denoising,
  title={Denoising Diffusion Implicit Models},
  author={Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
  booktitle={Proceedings of the International Conference on Learning Representations},
  pages={1--20},
  year={2021}
}
```
