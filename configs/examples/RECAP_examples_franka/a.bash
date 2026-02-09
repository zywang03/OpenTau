accelerate launch \
    --config_file /data/OpenTau/configs/examples/RECAP_examples_franka/accelerate_ddp_config.yaml \
    /data/OpenTau/src/opentau/scripts/train.py \
    --config_path=/data/OpenTau/configs/examples/RECAP_examples_franka/pi0_train_config.json \
    --output_dir=/data/output_ckpt/train/pi0 \
    --steps 40

accelerate launch \
    --config_file /data/OpenTau/configs/examples/RECAP_examples_franka/accelerate_ddp_config.yaml \
    /data/OpenTau/src/opentau/scripts/train.py \
    --config_path=/data/OpenTau/configs/examples/RECAP_examples_franka/value_config.json \
    --output_dir=/data/output_ckpt/train/value \
    --steps 40