# DGE: Direct Gaussian 3D Editing by Consistent Multi-view Editing

[Minghao Chen](https://silent-chen.github.io), [Iro Laina](), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/)

[Paper](https://arxiv.org/abs/2404.18929) | [Webpage](https://silent-chen.github.io/DGE/) 


<br/><br/>

<div align="center">
    <img width="100%" alt="teaser" src="https://github.com/silent-chen/DGE/blob/gh-page/resources/more_examples.png?raw=true"/>
</div>

In this work, we introduce Direct Gaussian Editor (DGE), a novel method for fast 3D editing. We consider the task of 3D editing as a two-stage process, where the first stage focuses on achieving multi-view consistent 2D editing, followed by a secondary stage dedicated to precise 3D fitting.


## Abstract
We consider the problem of editing 3D objects and scenes based on open-ended language instructions. The established paradigm to solve this problem is to use a 2D image generator or editor to guide the 3D editing process. However, this is often slow as it requires do update a computationally expensive 3D representations such as a neural radiance field, and to do so by using contradictory guidance from a 2D model which is inherently not multi-view consistent. We thus introduce the Direct Gaussian Editor (DGE), a method that addresses these issues in two ways. First, we modify a given high-quality image editor like InstructPix2Pix to be multi-view consistent. We do so by utilizing a training-free approach which integrates cues from the underlying 3D geometry of the scene. Second, given a multi-view consistent edited sequence of images of the object, we directly and efficiently optimize the 3D object representation, which is based on 3D Gaussian Splatting. Because it does not require to apply edits incrementally and iteratively, DGE is significantly more efficient than existing approaches, and comes with other perks such as allowing selective editing of parts.

## Environment Setup

To set up the enviroment you can easily run the following command:
```buildoutcfg
conda create -n DGE python=3.8 -y 

# Install torch
# CUDA 11.7
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117


# CUDA 11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda activate DGE
pip install -r requirements.txt
```
For other CUDA version please following the instruction [here](https://pytorch.org/get-started/previous-versions/) to install torch >= 2.x.
## Editing 

If you don't have a trained 3DGS, you can follow the instructions in the original [3DGS repo](https://github.com/graphdeco-inria/gaussian-splatting) to perform reconstruction first.

Once you get a trained 3DGS and its corresponding trainig datasewt, you can follow instructions below to perform editing.

A simplest example call is given here.  Detailed configuration can be found in the `./configs/dge.yaml`.

```buildoutcfg
python launch.py --config configs/dge.yaml --train \
        data.source="PATH_TO_DATA" \
        system.gs_source="PATH_TO_PRETRAINED_GS_MODEL"
        system.prompt_processor.prompt="YOUR PROMPT"
```

We also provide some example scripts in `./script`. 

### Local Editing
For local editing, you should provide the prompt for segmentation:
```buildoutcfg
python launch.py --config configs/dge.yaml --train \
        data.source="PATH_TO_DATA" \
        system.gs_source="PATH_TO_PRETRAINED_GS_MODEL" \
        system.prompt_processor.prompt="YOUR PROMPT"
        system.seg_prompt="PROMPT_FOR_SEG"
```

## Random Notes on Improving Results
- If you find the editing result is not obvious, consider change `system.guidance.guidance_scale` through config file or command line.
- We follow [GaussianEditor](https://github.com/buaacyw/GaussianEditor), which center crop and resize the image to 512 x 512 as InstructPix2Pix gives best results in this resolution. If you find the edited 3DGS has obvious artifacts, you may try to use the original resolution by set `data.use_original_resolution` to True.

- [Local Editing] If you find the mask is not correct for gaussians, try to adjust the segmentation threshold in `configs\dge.yaml` or specifiy it with `system.mask_thres`.

## Citation

If this repo is helpful for you, please consider to cite it. Thank you! :)

```bibtex
@article{chen2024dge,
      title={DGE: Direct Gaussian 3D Editing by Consistent Multi-view Editing}, 
      author={Minghao Chen and Iro Laina and Andrea Vedaldi},
      journal={arXiv preprint arXiv:2404.18929},
      year={2024}
}
```

## Acknowledgements

This research is supported by ERC-CoG UNION 101001212. Iro Laina is also partially supported by the VisualAI EPSRC grant (EP/T028572/1).

The code is largely based on [GaussianEditor](https://github.com/buaacyw/GaussianEditor) and [TokenFlow](https://github.com/omerbt/TokenFlow) and also inspired by worderful projects:

- [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Threestudio](https://github.com/threestudio-project/threestudio)
- [Instruct-nerf2nerf](https://github.com/ayaanzhaque/instruct-nerf2nerf)
- [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix)


