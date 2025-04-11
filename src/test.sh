dataset=mimic3_50
name=plm_icd_ca
output_dir=../models/$dataset/$name
cat $0 > $output_dir/test.sh
export CUDA_VISIBLE_DEVICES=4
accelerate launch \
    main.py \
    --dataset $dataset \
    --name $name \
    --train_file ../data/$dataset/train.pkl \
    --validation_file ../data/$dataset/dev.pkl \
    --test_file ../data/$dataset/new/test.pkl \
    --code_description_file ../data/$dataset/code/codes.csv \
    --embed_code_query \
    --max_length 6144 \
    --chunk_size 128 \
    --model_name_or_path ../models/RoBERTa-base-PM-M3-Voc-distill-align-hf \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 0 \
    --use_cross_attention \
    --use_guidance \
    --use_shuffle \
    --use_synonyms \
    --code_synonyms_file ../data/icd9/reference/icd9_synonyms.json \
    --use_hierarchy \
    --extra_code_description_file ../data/$dataset/code/extra_codes.csv \
    --code_group_file ../data/$dataset/code/groups.csv \
    --code_relation_file ../data/$dataset/code/hierarchy.csv \
    --code_distribution_file ../data/$dataset/code/distribution.csv \
    --use_sim_loss \
    --lambda_sim_loss 1.0 \
    --find_best_threshold \
    --output_dir $output_dir
    
