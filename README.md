## IAM: A Comprehensive and Large-Scale Dataset for Integrated Argument Mining Tasks

[![PWC](https://img.shields.io/badge/PapersWithCode-Benchmark-%232cafb1)](https://paperswithcode.com/paper/iam-a-comprehensive-and-large-scale-dataset)

This repository implements our ACL 2022 research paper [IAM: A Comprehensive and Large-Scale Dataset for Integrated Argument Mining Tasks](https://arxiv.org/pdf/2203.12257.pdf). In this paper, we introduce a comprehensive and large dataset named IAM, which can be applied to a series of argument mining tasks. We show the data and code for 5 tasks mentioned in our paper as below.

### Task 1: [Claim Extraction](https://github.com/LiyingCheng95/IAM/tree/main/claims)

All claims data are shown in `all_claims.txt`. We separate them randomly into `train/dev/test.txt`. Each file has 5 columns:

- **claim_label**: `C` represents the current claim candidate sentence is a claim for the current topic, `O` represents non-claim.
- **topic_sentence**
- **claim_candidate_sentence**
- **article_id**
- **stance_label**: `1` represents support, `-1` represents contest, `0` represents no relation/non-claim.

### Task 2.1: [Stance Classification](https://github.com/LiyingCheng95/IAM/tree/main/stance)

We filter all claim sentences from the data shown in Task 1, and also have `train/dev/test.txt`. The data format is the same as shown previously.

### Task 2.2: [Stance Classification - Chinese version](https://github.com/LiyingCheng95/IAM/tree/main/stance_Chinese)

Here, we also provide a Chinese stance classification dataset, which was used in [NLPCC 2021 shared task Track 1](https://github.com/AIDebater/Argumentative-Text-Understanding-for-AI-Debater-NLPCC2021). There are only `train.txt` and `test.txt`, which has 3 columns:

- **topic_sentence**
- **candidate_sentence**
- **stance_label**: `Support`, `Against`, `Neutral`.

### Task 3: [Evidence Extraction](https://github.com/LiyingCheng95/IAM/tree/main/evidence)

Evidence data are shown in `evidences1.txt`, and we separate them randomly into `train/dev/test.txt`. For each topic, we choose around 15 sentences before and after as evidence candidates to form a short paragraph for each instance. Each file has 5 columns:
- **evidence_label**: `E` represents the current evidence candidate sentence is a piece of evidence for the given claim sentence. `O` represents non-evidence.
- **claim_sentence**
- **evidence_candidate_sentence**
- **article_id**
- **full_label**: `C-index` represents the index-th claim sentence in the article. `E-B/I-index` represents the evidence span for the claim labeled as `C-index`. `O` represents for non-evidence/non-claim. Multiple labels are separated by `|`.

### Task 4: [CESC: Claim Extraction with Stance Classification](https://github.com/LiyingCheng95/IAM/tree/main/CESC)

Refer to the data used for Task 1.

### Task 5: [CEPE: Claim-Evidence Pair Extraction](https://github.com/LiyingCheng95/IAM/tree/main/CEPE)

We add the topic information on top of the data used in Task 3. Each `train/dev/test.txt` file has 7 columns:
- **claim_label**: `C` represents the current claim candidate sentence is a claim for the current topic, `O` represents non-claim.
- **topic_sentence**
- **evidence_label**: `E` represents the current evidence candidate sentence is a piece of evidence for the given claim sentence. `O` represents non-evidence.
- **claim_sentence**
- **evidence_candidate_sentence**
- **article_id**
- **full_label**: `C-index` represents the index-th claim sentence in the article. `E-B/I-index` represents the evidence span for the claim labeled as `C-index`. `O` represents for non-evidence/non-claim. Multiple labels are separated by `|`.

If you are using the multi-task model ([MLMC](https://aclanthology.org/2021.acl-long.496.pdf)), you have to use `process_json.py` to process the files into json format, then follow the instructions shown in [this repository](https://github.com/TianyuTerry/MLMC), and don't forget to cite us!


### Code Usage
For sentence pair classification models, simply train the model using ```python train.py``` and test the model using ```python main.py```.

### Citation
```
@inproceedings{cheng2022iam,
  title={IAM: A Comprehensive and Large-Scale Dataset for Integrated Argument Mining Tasks},
  author={Cheng, Liying and Bing, Lidong and He, Ruidan and Yu, Qian and Zhang, Yan and Si, Luo},
  booktitle={Proceedings of ACL},
  year={2022}
}

@@inproceedings{yuan2021overview,
  title={Overview of Argumentative Text Understanding for AI Debater Challenge},
  author={Yuan, Jian and Cheng, Liying and He, Ruidan and Li, Yinzi and Bing, Lidong and Wei, Zhongyu and Liu, Qin and Shen, Chenhui and Zhang, Shuonan and Sun, Changlong and others},
  booktitle={Proceedings of NLPCC},
  year={2021}
}
```
