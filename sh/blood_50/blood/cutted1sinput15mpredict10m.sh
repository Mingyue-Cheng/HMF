model_name=HMF
data=BLOOD
data_path=all_patient_1.csv
split_name=cutted1sinput15mpredict10m

python -u runa.py \
  --is_training 1 \
  --root_path ./datasets/blood \
  --save_path ./results/our_blood \
  --iter_path 0 \
  --split_name $split_name\
  --model $model_name \
  --data $data \
  --data_path $data_path \
  --features M \
  --seq_len 900 \
  --label_len 450 \
  --pred_len 600 \
  --learning_rate 1e-2 \
  --factor 3 \
  --enc_in 2 \
  --dec_in 2 \
  --c_out 2 \
  --train_epochs 1 \
  --d_model 16\
  --des 'Exp' \
  --itr 1 \
  --d_ff 64\
  --dropout 0.1\
  --target OT\
  --conv_stride 150\
  --conv_kernel 150\
  --conv_padding 0\
  --patch_num 50\
  --batch_size 320\
  --patience 3