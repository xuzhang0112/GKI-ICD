# GKI-ICD: A General Knowledge Injection Framework for ICD Coding


## Preparation

Download [RoBERTa-base-PM-M3-Voc-distill-align](https://github.com/facebookresearch/bio-lm/blob/main/README.md).

Download [MIMIC-III Dataset V1.4](https://physionet.org/content/mimiciii/1.4/), and run the jupyter notebook in `preprocess/prepare_mimic3.ipynb` to get `train.pkl`, `dev.pkl` and `test.pkl`.



## Run

To train and test:
```
bash src/train.sh
```

To evaluate:
```
bash src/test.sh
```

## Script Arguments
### Basic Path
| Arg                | Func                                             | Example                                        |
| ------------------ | ------------------------------------------------ | ---------------------------------------------- |
| dataset            | specify dataset                                  | mimic3/mimic3_50                               |
| name               | specify experiment name                          | plm-ca/debug/...                               |
| output_dir         | specify where the checkpint and results is saved |
| model_name_or_path | specify path of downloaded BERT checkpoint       | models/RoBERTa-base-PM-M3-Voc-distill-align-hf |

### Enable PLM-CA
| Arg                 | Func                  | Example               |
| ------------------- | --------------------- | --------------------- |
| max_length          |                       | --max_length 8192     |
| chunk_size          |                       | --chunk_size 128      |
| use_cross_attention | choose plm-icd/plm-ca | --use_cross_attention |

### Enable GKI-ICD
| Arg              | Func                         | Example               |
| ---------------- | ---------------------------- | --------------------- |
| embed_code_query | w/ description knowledge     | --embed_code_query    |
| use_guidance     | w/ description knowledge     | --use_guidance        |
| use_shuffle      |                              | --use_shuffle         |
| use_synonyms     | w/ synonym knowledge         | --use_synonyms        |
| use_hierarchy    | w/ hierarchy knowledge       | --use_hierarchy       |
| use_sim_loss     |                              | --use_sim_loss        |
| lambda_sim_loss  | control knowledge gap        | --lambda_sim_loss 0.0 |
| use_rdrop        | enable R-drop regularization | --use_rdrop           |
| rdrop_alpha      |                              | --rdrop_alpha 5.0     |


### Advanced
| Arg                 | Func                                   | Example               |
| ------------------- | -------------------------------------- | --------------------- |
| use_swanlab         | [swanlab](https://docs.swanlab.cn/en/) | --use_swanlab         |
| find_best_threshold | search for max micro f1_score          | --find_best_threshold |
| save_group_result   | long tail analysis                     | --save_group_results  |
| save_pred_result    | case analysis                          | --save_pred_results   |