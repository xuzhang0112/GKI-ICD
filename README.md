# GKI-ICD: A General Knowledge Injection Framework for ICD Coding
Offical Code for Paper [A General Knowledge Injection Framework for ICD Coding](https://arxiv.org/abs/2505.18708) 

*Accepted by ACL 2025 Findings*


## Preparation

Download [RoBERTa-base-PM-M3-Voc-distill-align](https://github.com/facebookresearch/bio-lm/blob/main/README.md).

Download [MIMIC-III Dataset V1.4](https://physionet.org/content/mimiciii/1.4/), and run the jupyter notebook in `preprocess/prepare_mimic3.ipynb` to get `train.pkl`, `dev.pkl` and `test.pkl`. 

Files containing code knowledge for GKI-ICD have been included in this repo, which can be found in `data/mimic3` and `data/mimic3_50`. These files are `code_description.csv` for description knowledge, `code_synonym.json` for synonym knowledge,  `code_hierarchy.csv`, `extra_code_description.csv`, `group_description.csv` for hierarchy knowledge.

Checkpoints of GKI-ICD can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1sawHOx_TMi_kTuM-xeaWX7_nbRihqYuu?usp=share_link).

## Run

Train from scratch and test the peformance:
```
bash scripts/train_mimic3_full.sh
```

Test the peformance of GKI-ICD with given checkpoint:
```
bash scripts/test_mimic3_full.sh
```

### Script Arguments


If you are intrested in our work, here are some additonal explanations for the arguments in the shell scripts.


#### Basic Path
| Arg                | Func                                                                  | Example                                                             |
| ------------------ | --------------------------------------------------------------------- | ------------------------------------------------------------------- |
| dataset            | specify dataset                                                       | --dataset mimic3_50                                                 |
| name               | specify experiment name (related to output directory and swanlab log) | --name plm-ca/debug/...                                             |
| model_name_or_path | specify path of downloaded BERT checkpoint                            | --model_name_or_path models/RoBERTa-base-PM-M3-Voc-distill-align-hf |

#### Enable PLM-CA
| Arg                 | Func                  | Example               |
| ------------------- | --------------------- | --------------------- |
| max_length          |                       | --max_length 8192     |
| chunk_size          |                       | --chunk_size 128      |
| use_cross_attention | choose plm-icd/plm-ca | --use_cross_attention |

#### Enable GKI-ICD
| Arg              | Func                         | Example               |
| ---------------- | ---------------------------- | --------------------- |
| embed_code_query | w/ description knowledge     | --embed_code_query    |
| use_guidance     | w/ description knowledge     | --use_guidance        |
| use_shuffle      |                              | --use_shuffle         |
| use_synonyms     | w/ synonym knowledge         | --use_synonyms        |
| use_hierarchy    | w/ hierarchy knowledge       | --use_hierarchy       |
| use_sim_loss     |                              | --use_sim_loss        |
| lambda_sim_loss  | control knowledge gap        | --lambda_sim_loss 1.0 |
| use_rdrop        | enable R-drop regularization | --use_rdrop           |
| rdrop_alpha      |                              | --rdrop_alpha 10.0    |


#### Advanced
| Arg                 | Func                                   | Example               |
| ------------------- | -------------------------------------- | --------------------- |
| use_swanlab         | [swanlab](https://docs.swanlab.cn/en/) | --use_swanlab         |
| find_best_threshold | search for max micro f1_score          | --find_best_threshold |
| save_group_result   | long tail analysis                     | --save_group_results  |
| save_pred_result    | case analysis                          | --save_pred_results   |