python3 -W ignore::UserWarning  train.py \
--dataroot /data1/juliew/mini_dataset/online_mario2sonic_lite  \
--checkpoints_dir  /data1/juliew/checkpoints \
--name mario \
--config_json examples/example_ddpm_mario.json \
--gpu_ids 1  \
--output_display_env  test_mario_unet  \
--output_display_freq 1  \
--output_print_freq 1 \
--G_diff_n_timestep_test 5  \
--G_diff_n_timestep_train 2000 \
--G_unet_mha_channel_mults 1 2 4 8  \
--G_unet_mha_res_blocks 2 2 2 2 \
--train_batch_size 1 \
--G_unet_mha_attn_res 1 2 4 8  \
--data_num_threads 1  \
