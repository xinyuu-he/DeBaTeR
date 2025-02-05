# DeBaTeR

This is the official code repository for paper [DeBaTeR: Denoising Bipartite Temporal Graph for Recommendation](https://arxiv.org/abs/2411.09181).

Model training, data processing codes and processed data are available in this directory.

Available dataset options are: ml-100k, ml-1m, yelp, amazon (and their perturbed/noisy version, e.g., yelp-p)

Note: DeBaTeR-L is implemented based on BOD code repository ([BOD](https://github.com/CoderWZW/BOD)), and this BOD repository is implemented based on open-source SELFRec repository ([SELFRec](https://github.com/Coder-Yu/SELFRec)), which we also use for running a couple of baselines (e.g., SimGCL, NCL, Bert4Rec, etc)

To run DeBaTeR-A on ml-100k, for example:

```python
cd DeBaTeR-A
python main.py --d=ml-100k
```

Same for DeBaTeR-L:

```python
cd DeBaTeR-L
python main.py --d=ml-100k
```

Processed datasets are available at ./DeBaTeR-A/dataset and ./DeBaTeR-L/dataset. Data preprocessing codes are available at ./DeBaTeR-A/dataset/[dataset name]/[dataset name].py and ./DeBaTeR-A/dataset/graph_perturb.py.

Raw data can be downloaded at:

- [ML-100K](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset)
- [ML-1M](https://grouplens.org/datasets/movielens/1m/)
- [Yelp](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset?select=yelp_academic_dataset_review.json)
- [Amazon](https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Movies_and_TV_5.json.gz)
