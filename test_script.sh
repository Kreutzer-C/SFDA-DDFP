cd /opt/data/private/SFDA-DDFP

python3 test.py     --model_path '/opt/data/private/SFDA-DDFP/results/UNet_Abdomen_CT2MR_pmt/T2026-03-25 17:01:03_ftLSET1_Pmt_UNet_Data_w1110_/saved_models/best_model_step_14_dice_0.8706.pth'     --data_root datasets/chaos     --target_site mr     --gpu_id 0     --arch Pmt_UNet     --save_vis
