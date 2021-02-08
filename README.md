# IRGPR

This is a pytorch re-implementation of the paper [Personalized Re-ranking with Item Relationships for E-commerce](https://dl.acm.org/doi/abs/10.1145/3340531.3412332), built upon [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) library.

### Citation
```
@inproceedings{liu2020personalized,
  title={Personalized Re-ranking with Item Relationships for E-commerce},
  author={Liu, Weiwen and Liu, Qing and Tang, Ruiming and Chen, Junyang and He, Xiuqiang and Heng, Pheng Ann},
  booktitle={Proceedings of the 29th ACM International Conference on Information \& Knowledge Management},
  pages={925--934},
  year={2020}
}
```

### Dependecies
* Python3.7
* PyTorch
* PyG
* networkx
* pandas
* gensim

### Experiment Data
* [Amazon Review Data](https://jmcauley.ucsd.edu/data/amazon/)

### Experiment
* Before running, please modify the corresponding Amazon data category in ```amazon_rerank_loader.py```.
```
python run_irgpr.py --lr [lr] --node_emb [node embedding dim]
```
* We provided a sample processed data ```Amazon_Video_Games.pt```.
