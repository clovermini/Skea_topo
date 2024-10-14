# Skea_topo
The PyTorch implementation of the skea-topo aware loss proposed in paper: [Enhancing Boundary Segmentation for Topological Accuracy with Skeleton-based Methods](https://arxiv.org/pdf/2404.18539). <br>
This paper have been accepted by IJCAI2024. <br>

## Abstract
Topological consistency plays a crucial role in the task of boundary segmentation for reticular images, such as cell membrane segmentation in neuron electron microscopic images, grain boundary segmentation in material microscopic images and road segmentation in aerial images. In these fields, topological changes in segmentation results have a serious impact on the downstream tasks, which can even exceed the misalignment of the boundary itself. To enhance the topology accuracy in segmentation results, we propose the Skea-Topo Aware loss, which is a novel loss function that takes into account the shape of each object and topological significance of the pixels. It consists of two components. First, the skeleton-aware weighted loss improves the segmentation accuracy by better modeling the object geometry with skeletons. Second, a boundary rectified term effectively identifies and emphasizes topological critical pixels in the prediction errors using both foreground and background skeletons in the ground truth and predictions. Experiments prove that our method improves topological consistency by up to 7 points in VI compared to 13 state-of-art methods, based on objective and subjective assessments across three different boundary segmentation datasets.

<p align = "center">
<img src="https://github.com/clovermini/Skea_topo/blob/main/images/main.png">
</p>

## Environment

    python 3.8
    pytorch 1.13.0+cu116
    gala (for evaluation)

gala is installed according to https://github.com/janelia-flyem/gala.

## Prepare datasets and running
- SNEMI3D can be download from https://zenodo.org/record/7142003, and it should be placed in ./data/snemi3d/.  <br>
- IRON can be download from https://github.com/Keep-Passion/pure_iron_grain_data_sets, and it should be placed in ./data/iron/.  <br>
- MASS ROAD can be download from https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset, and it should be placed in ./data/mass_road/.  <br>

Usage Demo:

    # prepare datasets and calculate mean and std for each datasets
    python ./data/data_generator.py
    
    # generate skeleton aware weighted map:
    python ./data/skeleton_aware_loss_gen.py

    # generate foreground and background skeletons for labels
    python ./data/skeleton_gen.py

    # train
    run train.ipynb

    # eval 
    run eval.ipynb

## Results

The example results are shown as follows: 
<div align = "center">
<img src="https://github.com/clovermini/Skea_topo/blob/main/images/results.png" width="800">
</div>
<div align = "center">
<img src="https://github.com/clovermini/Skea_topo/blob/main/images/snemi3d.png" width="800">
</div>
<div align = "center">
<img src="https://github.com/clovermini/Skea_topo/blob/main/images/iron.png" width="800">
</div>
<div align = "center">
<img src="https://github.com/clovermini/Skea_topo/blob/main/images/mass_road.png" width="800">
</div>

## Citation
If you find our work is useful in your research or applications, please consider giving us a star 🌟 and citing it.

    @inproceedings{liu2024enhancing,
      title     = {Enhancing Boundary Segmentation for Topological Accuracy with Skeleton-based Methods},
      author    = {Liu, Chuni and Ma, Boyuan and Ban, Xiaojuan and Xie, Yujie and Wang, Hao and Xue, Weihua and Ma, Jingchao and Xu, Ke},
      booktitle = {Proceedings of the Thirty-Third International Joint Conference on
                   Artificial Intelligence, {IJCAI-24}},
      publisher = {International Joint Conferences on Artificial Intelligence Organization},
      editor    = {Kate Larson},
      pages     = {1092--1100},
      year      = {2024},
      month     = {8},
      note      = {Main Track},
      doi       = {10.24963/ijcai.2024/121},
      url       = {https://doi.org/10.24963/ijcai.2024/121},
    }



