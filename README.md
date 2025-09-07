# Skea_topo
The PyTorch implementation of the skea-topo aware loss proposed in paper: [Enhancing Boundary Segmentation for Topological Accuracy with Skeleton-based Methods](https://arxiv.org/pdf/2404.18539). <br>
- ✅ First published at **IJCAI 2024**  
- 📌 Extended and **accepted in *Pattern Recognition***  
- 📖 [Journal Version (Pattern Recognition)](https://www.sciencedirect.com/science/article/pii/S0031320325009264)  

## 🔑 Highlights of the Journal Version

1. **Extended to new datasets and scenarios**  
   - Adaptation to the **DRIVE** dataset, where the background typically forms a single connected object.  
   - Application to a newly introduced **Al-La MicroData** dataset for aluminum–lanthanum dendrite microstructure segmentation (50 slices, 1024×1024 px each).  
   - Dataset annotated by materials scientists and publicly released:  
     👉 [Al-La MicroData Download](https://pan.baidu.com/s/1rdM9Xj2mx9MinIj83ecnUw?pwd=3s57)  

2. **Detailed ablation studies**  
   - Extensive experiments across five diverse datasets to assess robustness.  

3. **Additional baseline comparisons**  
   - Inclusion of methods such as **ARS** for fairer evaluation.  

4. **Refined figures and tables**  
   - Improved clarity and readability in visual presentation.  


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
- SNEMI3D can be downloaded from https://zenodo.org/record/7142003, and it should be placed in ./data/snemi3d/.  <br>
- IRON can be downloaded from https://github.com/Keep-Passion/pure_iron_grain_data_sets, and it should be placed in ./data/iron/.  <br>
- MASS ROAD can be downloaded from https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset, and it should be placed in ./data/mass_road/.  <br>

Usage Demo:

    # prepare datasets and calculate mean and std for each datasets
    python ./data/data_generator.py
    
    # generate skeleton aware weighted map:
    python ./data/skeleton_aware_loss_gen.py

    # skeleton aware weighted map for DRIVE:
    python ./data/skeleton_aware_loss_gen_single_object.py

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
      pages     = {1092--1100},
      year      = {2024},
      month     = {8},
      doi       = {10.24963/ijcai.2024/121},
      url       = {https://doi.org/10.24963/ijcai.2024/121},
    }
or
    @article{LIU2026112265,
        title = {Skea-Topo: A skeleton-aware loss function for topologically accurate boundary segmentation},
        journal = {Pattern Recognition},
        volume = {171},
        pages = {112265},
        year = {2026},
        issn = {0031-3203},
        doi = {https://doi.org/10.1016/j.patcog.2025.112265},
        url = {https://www.sciencedirect.com/science/article/pii/S0031320325009264},
        author = {Chuni Liu and Boyuan Ma and Yujie Xie and Xiaojuan Ban and Haiyou Huang and Hao Wang and Weihua Xue and Ke Xu},
}



