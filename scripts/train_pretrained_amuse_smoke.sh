#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

smoke_output_dir="${RYNNWORLD_SMOKE_OUTPUT_DIR:-outputs/pretrained-stage3-amuse-smoke-1step}"
smoke_batch_size="${RYNNWORLD_SMOKE_BATCH_SIZE:-1}"
smoke_manifest="${RYNNWORLD_SMOKE_MANIFEST:-data/sample.json}"

exec accelerate launch \
  --use_deepspeed \
  --num_processes 1 \
  --num_machines 1 \
  --gpu_ids 0 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --deepspeed_config_file configs_zero/zero3_offload_amuse.yaml \
  --zero3_init_flag true \
  --zero3_save_16bit_model false \
  finetune_rynnworld4d.py \
  --model_path pretrained/Wan2.2-TI2V-5B-Diffusers \
  --model_name rynnworld4d \
  --model_type rynnworld4d \
  --training_type sft \
  --output_dir "${smoke_output_dir}" \
  --report_to tensorboard \
  --train_resolution 25x480x640 \
  --train_epochs 1 \
  --train_steps 1 \
  --seed 42 \
  --batch_size "${smoke_batch_size}" \
  --gradient_accumulation_steps 1 \
  --mixed_precision bf16 \
  --num_workers 0 \
  --gradient_checkpointing true \
  --learning_rate 1e-6 \
  --optimizer amuse \
  --beta2 0.999 \
  --epsilon 1e-10 \
  --weight_decay 0.01 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --checkpointing_steps 999999 \
  --checkpointing_limit 1 \
  --save_final_checkpoint false \
  --do_validation false \
  --validation_dir "${smoke_manifest}" \
  --cache_dir data/sample_latents \
  --prompt "" \
  --is_concat true \
  --fusion_mode joint \
  --share_ffn false \
  --joint_start_layer 0 \
  --joint_end_layer 30 \
  --joint_every_n_layers 3 \
  --joint_frame_wise true \
  --joint_use_rope true \
  --joint_unidirectional false \
  --joint_video_decay true \
  --joint_video_decay_steps 700 \
  --joint_out_lr 1e-6 \
  --joint_other_lr_multiplier 1.0 \
  --loss_weight_flow 1.0 \
  --use_ema false \
  --freeze_non_joint true \
  --branch_dropout_prob 0.0 \
  --periodic_inference_steps 0 \
  --load_stage2_model_weights pretrained/RynnWorld-4D \
  --amuse_muon_lr 1e-5 \
  --amuse_aux_lr 1e-6 \
  --amuse_beta1 0.4 \
  --amuse_momentum 0.95 \
  --amuse_rho 0.3 \
  --amuse_r 0.0 \
  --amuse_weight_lr_power 2.0 \
  --amuse_warmup_ratio 0.05 \
  --amuse_min_warmup_steps 2 \
  --amuse_weight_decay_at_y 0.0
