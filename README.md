# DeBaTeR

## Codes and data

Codes and data are available in /code directory.

Available dataset options are: ml-100k, ml-1m, yelp, amazon

Note: DeBaTeR-L is implemented based on BOD code repository ([BOD\]](https://github.com/CoderWZW/BOD)), and this BOD repository is implemented based on open-source SELFRec repository ([SELFRec](https://github.com/Coder-Yu/SELFRec)), which we also use for running a couple of baselines (e.g., SimGCL, NCL, Bert4Rec, etc)

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

Processed datasets are available at ./DeBaTeR-A/dataset and ./DeBaTeR-L/dataset. Preprocessing codes are available at ./DeBaTeR-A/dataset/[dataset name]/[dataset name].py.

Raw data can be downloaded at:

- [ML-100K](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset)
- [ML-1M](https://grouplens.org/datasets/movielens/1m/)
- [Yelp](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset?select=yelp_academic_dataset_review.json)
- [Amazon](https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Movies_and_TV_5.json.gz)



## Updated experimental results

We add **three new baselines** to the tables: LightGCN (a model without CL), Bert4Rec (state-of-the-art sequential model for recommendation), and CL4SRec (a follow-up work of SaSRec). We also include **standard deviations** (in parentheses) in the updated tables.

![](exp_vanilla.png)

![](exp_noisy.png)

Although Bert4Rec outperforms our model on Amazon dataset, it performs much worse on others. The relative improvements in percentage against other 7 baselines per metric are:

- Vanilla dataset:
  - Prec, Recall, NDCG@10: 5.08%, 4.08%, 4.09%
  - Prec, Recall, NDCG@20: 4.02%, 3.21%, 3.49%
- Noisy dataset:
  - Prec, Recall, NDCG@10: 5.22%, 3.36%, 1.49%
  - Prec, Recall, NDCG@20: 4.31%, 1.95%, 1.43%



## Additional Ablation Study on ML-1M

<img src="abla-100k.png" style="zoom:66%" />

<img src="abla_vanilla.png" style="zoom:50%" />

<img src="abla_noisy.png" style="zoom:50%" />

The pattern generally aligns with the ablation study on ML-100K. It can be seen that removing time information from $R(\cdot)$ or $W(\cdot)$ will lead to less robustness against noise, while removing time information from $\mathcal{L}, p$ lead to a more robust algorithm while slightly affecting model performance. However, a new observation on ML-1M shows that time information in $W(\cdot)$ and $\mathcal{L}, p$ combined will tend to have higher NDCG but not precision and recall, which aligns with the observation that DeBaTeR-L​ is more suitable for ranking tasks.