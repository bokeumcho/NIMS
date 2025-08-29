# NIMS

## Non_recursive model
```
nohup python /home/work/team3/bokeum/main_steps.py \
  --gpu 0 \
  --model_name SimVP_kT \
  --T 6 --steps 6 --T_interval 10 \
  --batch_size 32 --lr 5e-2 --epochs 1 \
  --C 4 --C_enc 16 --C_hid 64 \
  --Ns 4 --Nt 8 --groups 4 \
  --workdir runs/exp_kt_6hr --save_config \
> nohup_logs/nohup_steps_kt.out &
```

## Recursive model
```
nohup python /home/work/team3/bokeum/main_steps.py \
  --gpu 3 \
  --model_name SimVP_AR_Decoder \
  --T 6 --steps 6 --T_interval 10 \
  --batch_size 16 --lr 5e-2 --epochs 1 \
  --C 4 --C_enc 16 --C_hid 64 \
  --Ns 4 --Nt 8 --groups 4 \
  --workdir runs/exp_ar_6hr \
> nohup_logs/nohup_steps_ar.out &
```
