dataset=mimic3
name=new_text_6144_sim_1.0_rdrop_10.0_biaffine
output_dir=../models/$dataset/$name
mkdir -p $output_dir
cat $0 > $output_dir/train.sh
export CUDA_VISIBLE_DEVICES=5
accelerate launch \
    main.py \
    --dataset $dataset \
    --name $name \
    --train_file ../data/$dataset/train.pkl \
    --validation_file ../data/$dataset/dev.pkl \
    --test_file ../data/$dataset/test.pkl \
    --code_description_file ../data/$dataset/code_description.csv \
    --code_distribution_file ../data/$dataset/code_distribution.csv \
    --code_synonyms_file ../data/$dataset/code_synonym.json \
    --extra_code_description_file ../data/$dataset/extra_code_description.csv \
    --code_group_file ../data/$dataset/group_description.csv \
    --code_relation_file ../data/$dataset/code_hierarchy.csv \
    --embed_code_query \
    --max_length 6144 \
    --chunk_size 128 \
    --model_name_or_path ../models/RoBERTa-base-PM-M3-Voc-distill-align-hf \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 12 \
    --num_warmup_steps 2000 \
    --use_cross_attention \
    --use_guidance \
    --use_shuffle \
    --use_synonyms \
    --use_hierarchy \
    --use_sim_loss \
    --lambda_sim_loss 1.0 \
    --use_rdrop \
    --rdrop_alpha 10.0 \
    --use_biaffine \
    --find_best_threshold \
    --use_wandb \
    --output_dir $output_dir
    
