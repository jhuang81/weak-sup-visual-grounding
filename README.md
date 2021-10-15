# Improving Weakly Supervised Visual Grounding by Contrastive Knowledge Distillation

This repository is the official implementation of CVPR 2021 paper: [Improving Weakly Supervised Visual Grounding by Contrastive Knowledge Distillation](https://arxiv.org/pdf/2007.01951.pdf). 


## Requirements

* Tensorflow-1-15

## Training

To train the NCE model(s) in the paper, run this command:

```train
python train_nce_distill_model.py \
  --region_feat_path=region_features.hdf5 \
  --phrase_feat_path=phrase_features.hdf5 \
  --glove_path=glove.hdf5
```

To train the NCE+Distill model(s) in the paper, run this command:

```train
python train_nce_distill_model.py \
  --region_feat_path=region_features.hdf5 \
  --phrase_feat_path=phrase_features.hdf5 \
  --glove_path=glove.hdf5 \
  --phrase_to_label_json=phrase_to_label.json
```

## Evaluation

To evaluate the model on Flickr30K, run:

```eval
python eval_model.py \
  --region_feat_path=region_features_test.hdf5 \
  --phrase_feat_path=phrase_features_test.hdf5 \
  --glove_path=glove.hdf5 \
  --restore_path=checkpoint.meta
```


## Pre-trained Models

You can download pretrained models using `Res101 VG` features here:

- [NCE+Distill](https://drive.google.com/drive/folders/1q8MCAdNOXaEHAIQBqw4dcd402xdZUqsU)
- [NCE](https://drive.google.com/drive/folders/1VOuhMGeCGhfSpbKixCcnzX06MztEeMGA)

You can also find the features on Flickr30K test split [here](https://drive.google.com/drive/folders/1pIF6K4Rs_0HJeAeN4q281SOBbqwnMuVv).

The pretrained models achieve the following performance on Flickr30K test split:

| Model Name |  R@1 |  R@5 | R@10 |
|----------- | ---- | ---- | ---- |
| NCE+Distill | 0.5310 | 0.7394 | 0.7875 |
| NCE | 0.5135 | 0.7338 | 0.7833 |


## Citation

If you use our implementation in your research or wish to refer to the results published in our paper, please use the following BibTeX entry.

```
@InProceedings{Wang_2021_CVPR,
    author    = {Wang, Liwei and Huang, Jing and Li, Yin and Xu, Kun and Yang, Zhengyuan and Yu, Dong},
    title     = {Improving Weakly Supervised Visual Grounding by Contrastive Knowledge Distillation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {14090-14100}
}

```
