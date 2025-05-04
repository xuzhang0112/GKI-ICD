dataset=mimic3_50
name=plm-ca
output_dir=models/$dataset/$name
mkdir -p $output_dir
cat $0 > $output_dir/test.sh
export CUDA_VISIBLE_DEVICES=0
accelerate launch \
    src/main.py \
    --dataset $dataset \
    --name $name \
    --train_file data/$dataset/train.pkl \
    --validation_file data/$dataset/dev.pkl \
    --test_file data/$dataset/test.pkl \
    --code_description_file data/$dataset/code_description.csv \
    --code_distribution_file data/$dataset/code_distribution.csv \
    --code_synonyms_file data/$dataset/code_synonym.json \
    --extra_code_description_file data/$dataset/extra_code_description.csv \
    --code_group_file data/$dataset/group_description.csv \
    --code_relation_file data/$dataset/code_hierarchy.csv \
    --model_name_or_path models/RoBERTa-base-PM-M3-Voc-distill-align-hf \
    --max_length 6144 \
    --chunk_size 128 \
    --per_device_eval_batch_size 1 \
    --use_cross_attention \
    --find_best_threshold \
    --output_dir $output_dir
    
