# FedAlign
Navigating Alignment for Non-identical Client Class Sets: A Label Name-Anchored Federated Learning Framework. To appear in KDD 2023. [[arXiv](https://arxiv.org/pdf/2301.00489.pdf)]


## Run Experiments
### Datasets
Download and put the datasets in `data` folder. The datasets we used are listed below.

**ExtraSensory:** http://extrasensory.ucsd.edu/

**MIMIC-III:** https://physionet.org/content/mimiciii/1.4/. Please follow [this GitHub repo](https://github.com/SmokeShine/Convolutional-Attention-forMultiLabel-classification-CAML) to preprocess the data.

**PAMAP2:** https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring

**Reuters-21578 R8:** https://ana.cachopo.org/datasets-for-single-label-text-categorization

### Environment
Below are the packages we used for our experiments.
```
python==3.9.16
networkx==2.8.8
node2vec==0.4.6
numpy==1.21.4
pandas==1.5.3
pytorch_nlp==0.5.0
scikit_learn==1.2.2
torch==1.11.0
tqdm==4.65.0
```
### Run
Specify the task by `-t` and the gpu device by `-g`. For example, to run FedAlign on PAMAP2 with gpu 6, run:
```
python main.py -t pamap2 --fedalign -g 6
```

## Citation
Please cite the following paper if you found our framework useful. Thanks!
```
@article{zhang2023federated,
  title={Federated Learning with Client-Exclusive Classes},
  author={Zhang, Jiayun and Zhang, Xiyuan and Zhang, Xinyang and Hong, Dezhi and Gupta, Rajesh K and Shang, Jingbo},
  journal={arXiv preprint arXiv:2301.00489},
  year={2023}
}
```