model_name=DLinear
num=50
  
python -u runa.py \
  --is_training 1 \
  --root_path ./datasets/blood_outdata_part_$num \
  --save_path ./results/out_blood_$num \
  --iter_path 0 \
  --split_name cutted3sinput15mpredict5m\
  --model $model_name \
  --data BLOOD \
  --data_path all_patient_3.csv \
  --features MS \
  --seq_len 300 \
  --label_len 150 \
  --pred_len 100 \
  --learning_rate 1e-4 \
  --factor 3 \
  --enc_in 2 \
  --dec_in 2 \
  --c_out 1 \
  --train_epochs 100 \
  --dropout 0.1\
  --target map\
  --batch_size 32\
  --patience 3
  
python -u runa.py \
  --is_training 1 \
  --root_path ./datasets/blood_outdata_part_$num \
  --save_path ./results/out_blood_$num \
  --iter_path 0 \
  --split_name cutted3sinput15mpredict10m\
  --model $model_name \
  --data BLOOD \
  --data_path all_patient_3.csv \
  --features MS \
  --seq_len 300 \
  --label_len 150 \
  --pred_len 200 \
  --learning_rate 1e-4 \
  --factor 3 \
  --enc_in 2 \
  --dec_in 2 \
  --c_out 1 \
  --train_epochs 100 \
  --dropout 0.1\
  --target map\
  --batch_size 32\
  --patience 3
  
python -u runa.py \
  --is_training 1 \
  --root_path ./datasets/blood_outdata_part_$num \
  --save_path ./results/out_blood_$num \
  --iter_path 0 \
  --split_name cutted3sinput15mpredict15m\
  --model $model_name \
  --data BLOOD \
  --data_path all_patient_3.csv \
  --features MS \
  --seq_len 300 \
  --label_len 150 \
  --pred_len 300 \
  --learning_rate 1e-4 \
  --factor 3 \
  --enc_in 2 \
  --dec_in 2 \
  --c_out 1 \
  --train_epochs 100 \
  --dropout 0.1\
  --target map\
  --batch_size 32\
  --patience 3