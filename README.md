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
* To process a dataset from raw files, please get the following files from [Amazon Review Data](https://jmcauley.ucsd.edu/data/amazon/) and put them in the ```raw``` directory.
```
meta_Video_Games.json.gz
ratings_Video_Games.csv
reviews_Video_Games.json.gz
```
* Obtain node features from reviews by ```gensim.models.doc2vec```.
* As well as the initial ranked lists.
* We provided a sample processed heterogenous graph ```Amazon_Video_Games.pt``` from Amazon Video Games raw data, so that you can directly load the processed the data and train the model.

### Experiment
* Before running, please modify the corresponding Amazon data category in ```amazon_rerank_loader.py```.
```
python run_irgpr.py --lr [lr] --node_emb [node embedding dim]
```

