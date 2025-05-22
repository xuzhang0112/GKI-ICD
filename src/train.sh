dataset=mimic3
name=all_no_rdrop
output_dir=models/$dataset/$name
mkdir -p $output_dir
cat $0 > $output_dir/train.sh
export CUDA_VISIBLE_DEVICES=7
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
    --max_length 8192 \
    --chunk_size 128 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 12 \
    --num_warmup_steps 2000 \
    --use_cross_attention \
    --embed_code_query \
    --use_guidance \
    --use_shuffle \
    --use_synonyms \
    --use_hierarchy \
    --use_sim_loss \
    --lambda_sim_loss 0.0 \
    --seed 42 \
    --use_swanlab \
    --find_best_threshold \
    --output_dir $output_dir
    
