# LD²
This is the code for *"LD²: Scalable Heterophilous Graph Neural Network with Decoupled Embeddings"*, NeurIPS 2023.

[Conference Page - Poster/Video/Slides](https://neurips.cc/virtual/2023/poster/72677) | [OpenReview](https://openreview.net/forum?id=7zkFc9TGKz) | [GitHub](https://github.com/gdmnl/LD2)

### Citation

If you find this work useful, please cite our paper:
>  Ningyi Liao, Siqiang Luo, Xiang Li, and Jieming Shi.  
>  LD2: Scalable Heterophilous Graph Neural Network with Decoupled Embeddings.  
>  Advances in Neural Information Processing Systems 36, 2023.
```
@inproceedings{liao2023ld2,
  title={{LD2}: Scalable Heterophilous Graph Neural Network with Decoupled Embedding},
  author={Liao, Ningyi and Luo, Siqiang and Li, Xiang and Shi, Jieming},
  booktitle={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023},
  month={Dec},
  url = {https://neurips.cc/virtual/2023/poster/72677},
}
```

## Experiment

### Data Preparation
1. Use [LINKX source](https://github.com/CUAI/Non-Homophily-Large-Scale) to download raw data.
2. Run command `python data_convert.py` to generate processed files under path `data/[dataset_name]` similar to the example folder `data/actor`:
  * `feats.npy`: features in .npy array
  * `labels.npz`: node label information
    * 'label': labels (number or one-hot)
    * 'idx_train/idx_val/idx_test': indices of training/validation/test nodes
  * `adj_el.bin`, `adj_pl.bin`, `attribute.txt`, `deg.npz`: graph files for precomputation

### Precompute
1. Environment: CMake 3.16, C++ 14. Dependencies: [eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page), [Spectra](https://spectralib.org)
2. Configure and run: `python setup.py build_ext --inplace`

### Train & Test
1. Install dependencies: `conda create --name [envname] --file requirements.txt`
2. Run minibatch experiment: `python run_mini.py -f [seed] -c [config_file] -v [device]`

## Reference & Links
**Baselines**: [LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale) | [GloGNN](https://github.com/RecklessRonan/GloGNN) | [ACM](https://github.com/SitaoLuan/ACM-GNN) | 
[PPRGo](https://github.com/TUM-DAML/pprgo_pytorch) | [AGP](https://github.com/wanghzccls/AGP-Approximate_Graph_Propagation)

**Datasets**: [Yelp - GraphSAINT](https://github.com/GraphSAINT/GraphSAINT) | [Reddit - PPRGo](https://github.com/TUM-DAML/pprgo_pytorch) | [Amazon - Cluster-GCN](http://manikvarma.org/downloads/XC/XMLRepository.html) | 
[tolokers](https://github.com/yandex-research/heterophilous-graphs) | [Others - LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale)

## TODOs
NOTE (2023-11-20): the current version has different propagation computation due to some new optimization strategies, which may cause accuracy and speed differences. We will update the code soon. 

* [ ] Adj propagation parallel and clocking
* [ ] Unify feature and one-time propagation
* [ ] Upload dataset configs 
