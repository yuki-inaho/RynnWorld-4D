# Modified from: https://github.com/huggingface/finetrainers
# RynnWorld4D trainer extends the finetrainers Wan I2V trainer with 3-branch
# (video + depth + optical flow) joint training and cosine decay mechanisms.

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import copy
import torch
from diffusers import (
    AutoencoderKLWan,              
    UniPCMultistepScheduler,
    WanImageToVideoPipeline,    
    WanTransformer3DModel,
)

from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
import numpy as np
from numpy import dtype
from transformers import T5TokenizerFast, UMT5EncoderModel
from typing_extensions import override
from transformers import AutoConfig

from core.finetune.schemas import Wan_Components as Components
from core.finetune.trainer import Trainer
from core.finetune.models.wan_i2v.wan_trainer import WanI2VTrainer
from core.finetune.utils import unwrap_model
from core.finetune.models.wan_i2v.module import Wan_Components, patched_wan_time_text_image_embedding_forward, wan_forward, RynnWorld4DTransformer3DModel
from core.finetune.models.wan_i2v.module_joint import JointRynnWorld4DTransformer3DModel

from ..utils import register
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.optimization import get_scheduler
from diffusers.configuration_utils import register_to_config
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.training_utils import EMAModel
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.modeling_utils import ModelMixin
import json
from core.finetune.schemas import Args, Components, State
from accelerate.accelerator import Accelerator, DistributedType
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
import types
from termcolor import cprint
from core.finetune.utils import (
    cast_training_params,
    free_memory,
    get_intermediate_ckpt_path,
    get_latest_ckpt_path_to_resume_from,
    get_memory_statistics,
    get_optimizer,
    string_to_filename,
    unload_model,
    unwrap_model,
)
from tqdm import tqdm
from itertools import chain
from accelerate import init_empty_weights
from accelerate import load_checkpoint_and_dispatch
import deepspeed
from einops import rearrange
import functools
import hashlib
import random

from core.finetune.optim import (
    AmuseInternalScheduler,
    attach_zero3_amuse,
    build_amuse_optimizer,
    find_amuse_optimizer,
    set_amuse_mode,
)
from core.finetune.compact_checkpoint import (
    load_compact_amuse_checkpoint,
    save_compact_amuse_checkpoint,
)
from core.finetune.tensorboard import ScalarTensorBoardLogger
from core.finetune.topk_checkpointing import TopKCheckpointManager

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

try:
    from diffusers.models.transformers.transformer_wan import WanTimeTextImageEmbedding
except ImportError as e:
    cprint("❌ Critical Error: Could not import `WanTimeTextImageEmbedding` for monkey-patching.", 'red')
    cprint("   The structure of the `diffusers` library may have changed.", 'red')
    raise e

WanTimeTextImageEmbedding.forward = patched_wan_time_text_image_embedding_forward
cprint("✅ [Monkey Patch Applied] `WanTimeTextImageEmbedding.forward` has been replaced to ensure float32 stability during mixed-precision training.", "green")

WanTransformer3DModel.forward = wan_forward
cprint("✅ [Monkey Patch Applied] `WanTransformer3DModel.forward` has been added control video latent.", "green")


class RynnWorld4DTrainer(WanI2VTrainer):
    UNLOAD_LIST = ["text_encoder","vae"]
    @override
    def __init__(self, args: Args) -> None:
        self.args = args

        self.state = State(
            weight_dtype=self.get_training_dtype(),
            train_frames=self.args.train_resolution[0],
            train_height=self.args.train_resolution[1],
            train_width=self.args.train_resolution[2],
        )

        self._init_distributed()
        self.state.weight_dtype = self.get_training_dtype()
        self.components: Components = self.load_components()
        self.dataset: Dataset = None
        self.data_loader: DataLoader = None

        self.optimizer = None
        self.lr_scheduler = None

        if self.accelerator.is_main_process:
            print("\n" + "="*60)
            print("   ACCELERATE DISTRIBUTED SANITY CHECK (from Python code)")
            print(f"   - Total number of processes found: {self.accelerator.num_processes}")
            print(f"   - Current process is the main process: {self.accelerator.is_main_process}")
            print(f"   - Device for the main process: {self.accelerator.device}")
            print("="*60 + "\n")
        self._init_logging()
        self._init_directories()
        self.state.using_deepspeed = self.accelerator.state.deepspeed_plugin is not None

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Wan_Components()
        model_path = str(self.args.model_path)

        cprint(f"Loading components from: {model_path}",'green')
        components.pipeline_cls = WanImageToVideoPipeline
        ds_plugin = self.accelerator.state.deepspeed_plugin
        is_zero3 = ds_plugin is not None and ds_plugin.zero_stage == 3

        if is_zero3:
            # Accelerator registers a Transformers-global ZeRO-3 hook. T5 and
            # the VAE are preprocessing-only components and are never wrapped
            # by the DeepSpeed engine, so partitioning them leaves unusable
            # zero-sized parameters. The transformer is partitioned explicitly
            # below and does not depend on this global hook.
            from transformers.integrations.deepspeed import unset_hf_deepspeed_config

            unset_hf_deepspeed_config()

        components.tokenizer = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer")
        components.text_encoder = UMT5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
        components.vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae")
        components.scheduler = UniPCMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")

        # Select model class based on fusion_mode
        if self.args.fusion_mode == "joint":
            model_cls = JointRynnWorld4DTransformer3DModel
            model_kwargs = dict(
                subfolder="transformer", eps=1e-5, share_ffn=self.args.share_ffn,
                joint_start_layer=self.args.joint_start_layer,
                joint_end_layer=getattr(self.args, 'joint_end_layer', -1),
                joint_every_n_layers=getattr(self.args, 'joint_every_n_layers', 1),
                joint_frame_wise=getattr(self.args, 'joint_frame_wise', False),
                joint_use_rope=getattr(self.args, 'joint_use_rope', False),
                joint_unidirectional=getattr(self.args, 'joint_unidirectional', False),
            )
        else:
            model_cls = RynnWorld4DTransformer3DModel
            model_kwargs = dict(subfolder="transformer", eps=1e-5, fusion_mode=self.args.fusion_mode, share_ffn=self.args.share_ffn)

        if is_zero3:
            import deepspeed
            full_checkpoint = None
            requested_checkpoint = getattr(self.args, "load_stage2_model_weights", None)
            if requested_checkpoint:
                candidate = os.path.join(
                    requested_checkpoint,
                    "pytorch_model",
                    "mp_rank_00_model_states.pt",
                )
                if os.path.isfile(candidate):
                    full_checkpoint = candidate

            # Accelerate resolves DeepSpeed's automatic batch fields only in
            # accelerator.prepare(), which happens after this model is built.
            # zero.Init needs numeric values immediately, so resolve a private
            # copy without mutating the plugin config that Accelerate owns.
            zero_init_config = copy.deepcopy(ds_plugin.deepspeed_config)
            zero_init_config["train_micro_batch_size_per_gpu"] = self.args.batch_size
            zero_init_config["gradient_accumulation_steps"] = self.args.gradient_accumulation_steps
            zero_init_config["train_batch_size"] = (
                self.args.batch_size
                * self.args.gradient_accumulation_steps
                * self.accelerator.num_processes
            )

            with deepspeed.zero.Init(config_dict_or_path=zero_init_config):
                # Diffusers builds VAE/T5 weights through meta tensors when loading.
                # DeepSpeed zero.Init cannot partition those meta tensors, and neither
                # component participates in the training forward pass. Keep them out
                # of zero.Init and use the context only for the 15.9B transformer.
                if full_checkpoint is not None:
                    # A complete RynnWorld checkpoint will replace every transformer
                    # parameter later. Construct the exact tri-branch architecture
                    # directly from config, avoiding Diffusers' nested meta-tensor
                    # loader and an unnecessary 5B base-weight materialization.
                    base_config = dict(
                        WanTransformer3DModel.load_config(
                            model_path,
                            subfolder="transformer",
                        )
                    )
                    base_config.update({k: v for k, v in model_kwargs.items() if k != "subfolder"})
                    components.high_noise_model = model_cls.from_config(base_config)
                else:
                    components.high_noise_model = model_cls.from_pretrained(model_path, **model_kwargs)
        else:
            components.high_noise_model = model_cls.from_pretrained(model_path, **model_kwargs)

        boundary_ratio = 0
        num_train_timesteps = components.scheduler.config.num_train_timesteps
        self.state.moe_boundary = int(num_train_timesteps * boundary_ratio)
        cprint(f'MoE boundary set to timestep {self.state.moe_boundary}', 'green')

        try:
            components.high_noise_model.enable_xformers_memory_efficient_attention()
            cprint("✅ Successfully enabled xformers memory efficient attention.", "green")
        except Exception as e:
            cprint(f"Could not enable xformers. Fallback might be used automatically by PyTorch 2.0+. Error: {e}", "yellow")

        return components

    @override
    def prepare_for_training(self) -> None:
        if self.args.optimizer.lower() == "amuse":
            high_noise_model, self.optimizer, self.data_loader = self.accelerator.prepare(
                self.components.high_noise_model,
                self.optimizer,
                self.data_loader,
            )
            raw_optimizer = attach_zero3_amuse(self.optimizer)
            self.lr_scheduler = AmuseInternalScheduler(raw_optimizer)
        else:
            high_noise_model, self.optimizer, self.data_loader, self.lr_scheduler = self.accelerator.prepare(
                self.components.high_noise_model,
                self.optimizer,
                self.data_loader,
                self.lr_scheduler,
            )

        if getattr(self, "validation_loader", None) is not None:
            self.validation_loader = self.accelerator.prepare_data_loader(self.validation_loader)

        self.components.high_noise_model = high_noise_model

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.state.overwrote_max_train_steps:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.train_epochs = math.ceil(self.args.train_steps / num_update_steps_per_epoch)
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch

        # Initialize EMA (only for trainable parameters to minimize overhead)
        self.ema_model = None
        if self.args.use_ema:
            unwrapped = unwrap_model(self.accelerator, self.components.high_noise_model)
            trainable_params = [p for p in unwrapped.parameters() if p.requires_grad]
            self.ema_model = EMAModel(
                trainable_params,
                decay=self.args.ema_decay,
            )
            # Move EMA shadow params to CPU to save GPU memory
            for i in range(len(self.ema_model.shadow_params)):
                self.ema_model.shadow_params[i] = self.ema_model.shadow_params[i].detach().cpu()
            # Store param names for later weight extraction
            self._ema_param_names = [n for n, p in unwrapped.named_parameters() if p.requires_grad]
            num_ema_params = sum(p.numel() for p in trainable_params)
            cprint(f"✅ EMA initialized on CPU (decay={self.args.ema_decay}, tracking {len(trainable_params)} params, {num_ema_params/1e6:.1f}M parameters).", "green")

    @override
    def prepare_trackers(self) -> None:
        if getattr(self.args, "managed_training", False):
            logger.info("Using managed scalar-only TensorBoard logging")
            return
        super().prepare_trackers()

    @override
    def __prepare_saving_loading_hooks(self, transformer_lora_config):
        def save_model_hook(models: list, weights: list, output_dir: str):
            if self.accelerator.is_main_process:
                unwrapped_high_noise_model = unwrap_model(self.accelerator, self.components.high_noise_model)
                
                # save lora weight
                high_noise_lora_layers_to_save = get_peft_model_state_dict(
                    unwrapped_high_noise_model, adapter_name="high_noise"
                )

                self.components.pipeline_cls.save_lora_weights(
                    save_directory=os.path.join(output_dir, "high_noise_lora"),
                    transformer_lora_layers=high_noise_lora_layers_to_save,
                )

                # Save depth/flow branch trainable layers (non-LoRA)
                fw_keywords = ("depth", "flow", "video_to_depth", "video_to_flow", "depth_to_video", "flow_to_video", "joint_")
                rynnworld4d_state_dict = {}
                for name, param in unwrapped_high_noise_model.named_parameters():
                    if any(kw in name for kw in fw_keywords) and 'lora' not in name:
                        clean_name = name.replace(".base_layer.", ".")
                        rynnworld4d_state_dict[clean_name] = param.cpu()

                if rynnworld4d_state_dict:
                    rynnworld4d_save_path = os.path.join(output_dir, "rynnworld4d_layers.bin")
                    torch.save(rynnworld4d_state_dict, rynnworld4d_save_path)
                    logger.info(f"Successfully saved {len(rynnworld4d_state_dict)} RynnWorld4D layers.")
                # rynnworld4d_state_dict = {
                #     name: param.cpu()
                #     for name, param in unwrapped_high_noise_model.named_parameters()
                #     if param.requires_grad and 'lora' not in name
                # }
                # if rynnworld4d_state_dict:
                #     rynnworld4d_save_path = os.path.join(output_dir, "rynnworld4d_layers.bin")
                #     torch.save(rynnworld4d_state_dict, rynnworld4d_save_path)
                
                logger.info(f"Successfully saved high-noise LoRA and RynnWorld4D branch weights to {output_dir}")

            indices_to_pop = []
            for i, model in enumerate(models):
                if model is self.components.high_noise_model:
                    indices_to_pop.append(i)

            for i in sorted(indices_to_pop, reverse=True):
                models.pop(i)
                if weights:
                    weights.pop(i)
                
        def load_model_hook(models: list, input_dir: str):
            high_noise_model_ = unwrap_model(self.accelerator, self.components.high_noise_model)
            
            high_noise_lora_path = os.path.join(input_dir, "high_noise_lora")
            if os.path.exists(high_noise_lora_path):
                high_noise_state_dict = self.components.pipeline_cls.lora_state_dict(high_noise_lora_path)
                set_peft_model_state_dict(high_noise_model_, high_noise_state_dict, adapter_name="high_noise")
                logger.info(f"Successfully loaded LoRA weights from {high_noise_lora_path}")

            rynnworld4d_layers_path = os.path.join(input_dir, "rynnworld4d_layers.bin")
            if os.path.exists(rynnworld4d_layers_path):
                try:
                    saved_weights = torch.load(rynnworld4d_layers_path, map_location="cpu")
                    model_sd = high_noise_model_.state_dict()
                    
                    new_sd = {}
                    matched_count = 0
                    for k, v in saved_weights.items():
                        if k in model_sd:
                            new_sd[k] = v
                            matched_count += 1
                        else:
                            peft_key = k.replace(".weight", ".base_layer.weight").replace(".bias", ".base_layer.bias")
                            if peft_key in model_sd:
                                new_sd[peft_key] = v
                                matched_count += 1
                            else:
                                pass

                    high_noise_model_.load_state_dict(new_sd, strict=False)
                    logger.info(f"Successfully loaded {matched_count} RynnWorld4D branch weights from {rynnworld4d_layers_path}")
                except Exception as e:
                    logger.error(f"Failed to load RynnWorld4D branch weights: {e}")

            indices_to_pop = []
            for i, model in enumerate(models):
                if model is self.components.high_noise_model:
                    indices_to_pop.append(i)
            
            for i in sorted(indices_to_pop, reverse=True):
                models.pop(i)

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    @override
    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")
        weight_dtype = self.state.weight_dtype

        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError("Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead.")

        # For LoRA, we freeze all the parameters
        # For SFT, we train all the parameters in transformer model
        for attr_name, component in vars(self.components).items():
            if hasattr(component, "requires_grad_"):
                if self.args.training_type == "sft" and ("high_noise_model" in attr_name or "low_noise_model" in attr_name):
                    component.requires_grad_(True)
                else:
                    component.requires_grad_(False)

        if self.args.training_type == "lora":
            transformer_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights=True,
                target_modules=self.args.target_modules,
            )
            self.components.high_noise_model.add_adapter(transformer_lora_config, adapter_name="high_noise")
            self.components.high_noise_model.requires_grad_(False)

            # Unfreeze LoRA parameters
            for name, param in self.components.high_noise_model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True

            # Unfreeze depth/flow specific layers
            rynnworld4d_trainable_prefixes = (
                "patch_embedding_depth.",
                "patch_embedding_flow.",
                "norm_out_depth.",
                "norm_out_flow.",
                "proj_out_depth.",
                "proj_out_flow.",
            )
            # Fusion layer keywords based on fusion_mode
            if self.args.fusion_mode == "none":
                fusion_keywords = ()  # No fusion layers to train
            elif self.args.fusion_mode == "unidirectional":
                fusion_keywords = ("depth_to_video_zero", "flow_to_video_zero")
            elif self.args.fusion_mode == "joint":
                # Joint attention layers: all joint_* parameters
                fusion_keywords = ("joint_",)
            else:  # bidirectional
                fusion_keywords = (
                    "video_to_depth_zero", "video_to_flow_zero",
                    "depth_to_video_zero", "flow_to_video_zero",
                )
            
            rynnworld4d_trainable_keywords = fusion_keywords + (
                "attn1_depth.",
                "attn1_flow.",
                "attn2_depth.",
                "attn2_flow.",
                "norm1_depth.",
                "norm1_flow.",
                "norm2_depth.",
                "norm2_flow.",
                "norm3_depth.",
                "norm3_flow.",
            )
            # If not sharing FFN, also unfreeze independent depth/flow FFNs
            if not self.args.share_ffn:
                rynnworld4d_trainable_keywords = rynnworld4d_trainable_keywords + (
                    "ffn_depth.",
                    "ffn_flow.",
                )
            for name, param in self.components.high_noise_model.named_parameters():
                if name.startswith(rynnworld4d_trainable_prefixes) or any(kw in name for kw in rynnworld4d_trainable_keywords):
                    param.requires_grad = True

            # Unidirectional joint attention: video-side q/out/gate/modality_embed are unused
            # in the forward pass (video receives no joint injection), so freeze them to
            # keep optimizer state lean. joint_kv_video and joint_align_video stay trainable
            # because depth/flow attend to video's K/V.
            if (self.args.fusion_mode == "joint"
                    and getattr(self.args, "joint_unidirectional", False)):
                unused_video_joint_keys = (
                    "joint_q_video.",
                    "joint_out_video.",
                    "joint_gate_video",
                    "modality_embed_video",
                )
                frozen_count = 0
                for name, param in self.components.high_noise_model.named_parameters():
                    if any(k in name for k in unused_video_joint_keys):
                        param.requires_grad = False
                        frozen_count += 1
                cprint(f"  joint_unidirectional=True: froze {frozen_count} unused video-side joint params (q_video/out_video/gate_video/modality_embed_video).", "yellow")

            # joint_video_decay: freeze joint_gate_video and joint_out_video so the
            # trainer can decay the buffer instead of learning it.
            if (self.args.fusion_mode == "joint"
                    and getattr(self.args, "joint_video_decay", False)
                    and not getattr(self.args, "joint_unidirectional", False)):
                decay_freeze_keys = ("joint_gate_video", "joint_out_video.")
                frozen_count = 0
                for name, param in self.components.high_noise_model.named_parameters():
                    if any(k in name for k in decay_freeze_keys):
                        param.requires_grad = False
                        frozen_count += 1
                cprint(f"  joint_video_decay=True: froze {frozen_count} video-side joint params (gate_video/out_video) — decay buffer controls contribution.", "yellow")

            cprint(f"✅ Unfroze depth/flow branch layers. Fusion mode: {self.args.fusion_mode}.", "green")

            self.__prepare_saving_loading_hooks(transformer_lora_config)

        # Load stage1 weights if specified (for cross-stage training) — works for both LoRA and SFT
        if self.args.resume_from_stage1 and os.path.isdir(self.args.resume_from_stage1):
            stage1_dir = self.args.resume_from_stage1
            cprint(f"Loading stage1 weights from {stage1_dir}...", "cyan")
            
            unwrapped_model = self.accelerator.unwrap_model(self.components.high_noise_model)
            
            if self.args.training_type == "lora":
                # Load LoRA weights
                stage1_lora_path = os.path.join(stage1_dir, "high_noise_lora")
                if os.path.exists(stage1_lora_path):
                    from peft import set_peft_model_state_dict
                    from safetensors.torch import load_file
                    lora_sd = load_file(os.path.join(stage1_lora_path, "pytorch_lora_weights.safetensors"))
                    set_peft_model_state_dict(unwrapped_model, lora_sd, adapter_name="high_noise")
                    cprint(f"  Loaded stage1 LoRA weights.", "green")
                
                # Load RynnWorld4D layers (depth/flow branches, no fusion in stage1)
                stage1_fw_path = os.path.join(stage1_dir, "rynnworld4d_layers.bin")
                if os.path.exists(stage1_fw_path):
                    stage1_sd = torch.load(stage1_fw_path, map_location="cpu")
                    model_sd = unwrapped_model.state_dict()
                    # Only load keys that exist in current model (stage1 has no fusion layers)
                    matched = 0
                    for k, v in stage1_sd.items():
                        clean_k = k.replace(".base_layer.", ".")
                        if clean_k in model_sd:
                            model_sd[clean_k] = v.to(model_sd[clean_k].device)
                            matched += 1
                    unwrapped_model.load_state_dict(model_sd, strict=False)
                    cprint(f"  Loaded {matched} stage1 RynnWorld4D layer weights (depth/flow branches).", "green")
            
            elif self.args.training_type == "sft":
                # Load full model checkpoint from SFT stage1
                # DeepSpeed ZeRO-3 saves consolidated weights in pytorch_model/mp_rank_00_model_states.pt
                sft_model_path = os.path.join(stage1_dir, "pytorch_model", "mp_rank_00_model_states.pt")
                if os.path.exists(sft_model_path):
                    cprint(f"  Loading SFT stage1 model from {sft_model_path}...", "cyan")
                    stage1_sd = torch.load(sft_model_path, map_location="cpu")
                    
                    # The saved checkpoint may contain keys under "module" (DeepSpeed format)
                    if "module" in stage1_sd:
                        stage1_sd = stage1_sd["module"]
                    
                    model_sd = unwrapped_model.state_dict()
                    matched = 0
                    skipped = 0
                    for k, v in stage1_sd.items():
                        # Handle potential key prefix differences
                        clean_k = k.replace("module.", "")
                        if clean_k in model_sd:
                            if model_sd[clean_k].shape == v.shape:
                                model_sd[clean_k] = v.to(model_sd[clean_k].device)
                                matched += 1
                            else:
                                skipped += 1
                                cprint(f"    Skipping {clean_k}: shape mismatch (saved: {v.shape}, current: {model_sd[clean_k].shape})", "yellow")
                        elif k in model_sd:
                            if model_sd[k].shape == v.shape:
                                model_sd[k] = v.to(model_sd[k].device)
                                matched += 1
                            else:
                                skipped += 1
                    
                    unwrapped_model.load_state_dict(model_sd, strict=False)
                    cprint(f"  Loaded {matched} SFT stage1 model weights.", "green")
                    if skipped > 0:
                        cprint(f"  Skipped {skipped} keys (likely fusion layers added in stage2).", "yellow")
                else:
                    cprint(f"  Warning: SFT checkpoint not found at {sft_model_path}. Skipping stage1 weight loading.", "red")
            
            cprint("Stage1 weights loaded successfully.", "green")

        # Load stage2 model weights from a checkpoint saved with different GPU count
        # This loads ONLY model weights (not optimizer states), avoiding ZeRO world size mismatch
        self._stage2_ema_path = None
        if getattr(self.args, 'load_stage2_model_weights', None) and os.path.isdir(self.args.load_stage2_model_weights):
            stage2_dir = self.args.load_stage2_model_weights
            cprint(f"Loading stage2 model weights (model only, no optimizer) from {stage2_dir}...", "cyan")

            unwrapped_model = self.accelerator.unwrap_model(self.components.high_noise_model)
            sft_model_path = os.path.join(stage2_dir, "pytorch_model", "mp_rank_00_model_states.pt")
            if os.path.exists(sft_model_path):
                stage2_sd = torch.load(
                    sft_model_path,
                    map_location="cpu",
                    mmap=True,
                    weights_only=False,
                )
                if "module" in stage2_sd:
                    stage2_sd = stage2_sd["module"]

                parameter_shapes = {
                    name: tuple(getattr(param, "ds_shape", param.shape))
                    for name, param in unwrapped_model.named_parameters()
                }
                buffer_shapes = {
                    name: tuple(buffer.shape)
                    for name, buffer in unwrapped_model.named_buffers()
                }
                expected_shapes = {**parameter_shapes, **buffer_shapes}
                cleaned_state_dict = {}
                skipped = []
                for k, v in stage2_sd.items():
                    clean_k = k.replace("module.", "")
                    target_key = clean_k if clean_k in expected_shapes else k
                    if target_key in expected_shapes and tuple(v.shape) == expected_shapes[target_key]:
                        cleaned_state_dict[target_key] = v
                    else:
                        skipped.append(k)

                missing_parameters = sorted(set(parameter_shapes) - set(cleaned_state_dict))
                if missing_parameters:
                    raise RuntimeError(
                        "Full RynnWorld checkpoint is missing model parameters; "
                        f"first missing keys: {missing_parameters[:5]}"
                    )

                is_zero3 = (
                    self.accelerator.state.deepspeed_plugin is not None
                    and self.accelerator.state.deepspeed_plugin.zero_stage == 3
                )
                if is_zero3:
                    load_errors = []

                    def load_zero3_module(module, prefix=""):
                        direct_parameters = list(module.parameters(recurse=False))
                        with deepspeed.zero.GatheredParameters(direct_parameters, modifier_rank=0):
                            if self.accelerator.is_main_process:
                                module._load_from_state_dict(
                                    cleaned_state_dict,
                                    prefix,
                                    {},
                                    False,
                                    [],
                                    [],
                                    load_errors,
                                )
                        for child_name, child in module._modules.items():
                            if child is not None:
                                load_zero3_module(child, prefix + child_name + ".")

                    load_zero3_module(unwrapped_model)
                    if load_errors:
                        raise RuntimeError(
                            "Failed to load the full RynnWorld checkpoint under ZeRO-3: "
                            + "; ".join(load_errors[:5])
                        )
                else:
                    unwrapped_model.load_state_dict(cleaned_state_dict, strict=False)

                matched = len(cleaned_state_dict)
                cprint(f"  Loaded {matched} stage2 model weights (model only, fresh optimizer).", "green")
                if skipped:
                    cprint(f"  Skipped {len(skipped)} non-model or shape-mismatched keys.", "yellow")
                del cleaned_state_dict, stage2_sd
                import gc
                gc.collect()
            else:
                cprint(f"  Warning: model states not found at {sft_model_path}", "red")

            ema_path = os.path.join(stage2_dir, "ema_weights.pt")
            if os.path.exists(ema_path):
                self._stage2_ema_path = ema_path
                cprint(f"  Found EMA weights at {ema_path}, will load after EMA init.", "green")
            else:
                cprint(f"  No EMA weights found at {ema_path}, starting fresh EMA.", "yellow")

        # Load components needed for training to GPU (except transformer), and cast them to the specified data type
        # ignore_list = ["high_noise_xmodel", "low_noise_model"] + self.UNLOAD_LIST
        ignore_list = ["high_noise_model"] + self.UNLOAD_LIST
        self.move_components_to_device(dtype=weight_dtype, ignore_list=ignore_list)

        if self.args.gradient_checkpointing:
            is_zero3 = (
                self.accelerator.state.deepspeed_plugin is not None
                and self.accelerator.state.deepspeed_plugin.zero_stage == 3
            )
            if is_zero3:
                def zero3_checkpoint(module, *inputs):
                    # DeepSpeed's reentrant checkpoint re-enters module.__call__,
                    # so ZeRO-3 pre/post-forward hooks gather and repartition the
                    # parameters during backward recomputation. Ensure at least
                    # one activation requires gradients, as required by the
                    # reentrant checkpoint implementation when upstream layers
                    # are frozen by freeze_non_joint.
                    inputs = list(inputs)
                    if not any(torch.is_tensor(value) and value.requires_grad for value in inputs):
                        inputs[0] = inputs[0].detach().requires_grad_(True)
                    return deepspeed.checkpointing.checkpoint(module.__call__, *inputs)

                self.components.high_noise_model.enable_gradient_checkpointing(
                    gradient_checkpointing_func=zero3_checkpoint
                )
                cprint("✅ DeepSpeed-compatible reentrant gradient checkpointing enabled.", "green")
            else:
                self.components.high_noise_model.enable_gradient_checkpointing()
                cprint("✅ Gradient checkpointing enabled.", "green")

    @override
    def prepare_optimizer(self) -> None:
        logger.info("Initializing optimizer and lr scheduler")

        # ZeRO-3 replaces the local tensor storage with an empty partition before
        # optimizer construction.  ds_numel/ds_shape retain the logical values
        # and keep parameter accounting useful in both ZeRO and ordinary runs.
        def logical_numel(param):
            return int(getattr(param, "ds_numel", param.numel()))

        def logical_shape(param):
            return getattr(param, "ds_shape", param.shape)

        # Apply freeze_non_joint BEFORE casting so we don't waste fp32 memory on
        # parameters that are about to be frozen.
        if getattr(self.args, 'fusion_mode', '') == 'joint' and getattr(self.args, 'freeze_non_joint', False):
            for name, param in self.components.high_noise_model.named_parameters():
                if 'joint_' not in name and 'modality_embed' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        # When joint_unidirectional=True, video-side q/out/gate/modality_embed are unused
        # (no joint injection into video). Freeze them so optimizer state stays small.
        if (getattr(self.args, 'fusion_mode', '') == 'joint'
                and getattr(self.args, 'joint_unidirectional', False)):
            unused_video_joint_keys = (
                "joint_q_video.",
                "joint_out_video.",
                "joint_gate_video",
                "modality_embed_video",
            )
            for name, param in self.components.high_noise_model.named_parameters():
                if any(k in name for k in unused_video_joint_keys):
                    param.requires_grad = False

        # joint_video_decay: freeze joint_gate_video and joint_out_video
        if (getattr(self.args, 'fusion_mode', '') == 'joint'
                and getattr(self.args, 'joint_video_decay', False)
                and not getattr(self.args, 'joint_unidirectional', False)):
            decay_freeze_keys = ("joint_gate_video", "joint_out_video.")
            for name, param in self.components.high_noise_model.named_parameters():
                if any(k in name for k in decay_freeze_keys):
                    param.requires_grad = False

        if self.accelerator.is_main_process:
            trainable_params_count = 0
            trainable_tensor_count = 0
            print("\n" + "="*60)
            print("   CHECKING TRAINABLE PARAMETERS")
            for name, param in self.components.high_noise_model.named_parameters():
                if param.requires_grad:
                    if not getattr(self.args, "managed_training", False):
                        print(f"   - Trainable: {name}, shape: {logical_shape(param)}")
                    trainable_params_count += logical_numel(param)
                    trainable_tensor_count += 1
            print(f"   >>> Trainable Parameter Tensors: {trainable_tensor_count}")
            print(f"   >>> Total Trainable Parameters: {trainable_params_count / 1_000_000:.2f} M")
            print("="*60 + "\n")

        # Make sure the trainable params are in float32
        cast_training_params([self.components.high_noise_model], dtype=torch.float32)

        # For LoRA, we only want to train the LoRA weights
        # For SFT, we want to train all the
        trainable_parameters_high = list(filter(lambda p: p.requires_grad, self.components.high_noise_model.parameters()))
        trainable_parameters = trainable_parameters_high
        transformer_parameters_with_lr = {
            "params": trainable_parameters,
            "lr": self.args.learning_rate,
        }
        params_to_optimize = [transformer_parameters_with_lr]

        # Joint attention parameters: joint_kv/q/norm get multiplier, joint_out uses dedicated lr
        if getattr(self.args, 'fusion_mode', '') == 'joint':
            joint_out_lr = getattr(self.args, 'joint_out_lr', 5e-4)
            joint_other_lr_multiplier = getattr(self.args, 'joint_other_lr_multiplier', 10.0)

            joint_out_params = []
            joint_other_params = []
            other_params = []
            for name, param in self.components.high_noise_model.named_parameters():
                if not param.requires_grad:
                    continue
                if 'joint_out' in name:
                    joint_out_params.append(param)
                elif 'joint_' in name or 'modality_embed' in name:
                    joint_other_params.append(param)
                else:
                    other_params.append(param)

            if getattr(self.args, 'freeze_non_joint', False):
                # freeze_non_joint already applied above; other_params is empty.
                # Still respect joint_out_lr / joint_other_lr_multiplier.
                groups = []
                if joint_other_params:
                    groups.append({"params": joint_other_params, "lr": self.args.learning_rate * joint_other_lr_multiplier})
                if joint_out_params:
                    groups.append({"params": joint_out_params, "lr": joint_out_lr})
                if groups:
                    params_to_optimize = groups
                trainable_count = sum(logical_numel(p) for g in params_to_optimize for p in g["params"])
                cprint(
                    f"🔒 freeze_non_joint=True: training only joint_*/modality_embed "
                    f"({trainable_count/1e6:.1f}M params, joint_out lr={joint_out_lr:.1e}, "
                    f"joint_other lr={self.args.learning_rate * joint_other_lr_multiplier:.1e})",
                    "yellow",
                )
            elif joint_out_params or joint_other_params:
                params_to_optimize = [
                    {"params": other_params, "lr": self.args.learning_rate},
                    {"params": joint_other_params, "lr": self.args.learning_rate * joint_other_lr_multiplier},
                    {"params": joint_out_params, "lr": joint_out_lr},
                ]
                out_count = sum(logical_numel(p) for p in joint_out_params)
                other_joint_count = sum(logical_numel(p) for p in joint_other_params)
                cprint(f"✅ joint_out (zero-init): {out_count/1e6:.1f}M params, lr={joint_out_lr:.1e}", "green")
                cprint(f"✅ joint_kv/q/norm/align: {other_joint_count/1e6:.1f}M params, lr={self.args.learning_rate * joint_other_lr_multiplier:.1e}", "green")

        self.state.num_trainable_parameters = sum(logical_numel(p) for p in trainable_parameters)

        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.args.train_steps is None:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        if self.args.optimizer.lower() == "amuse":
            plugin = self.accelerator.state.deepspeed_plugin
            if plugin is not None:
                deepspeed_config = plugin.deepspeed_config
                if "optimizer" in deepspeed_config or "scheduler" in deepspeed_config:
                    raise ValueError("AMUSE requires a DeepSpeed profile without optimizer or scheduler")
            result = build_amuse_optimizer(
                self.components.high_noise_model,
                total_optimizer_steps=self.args.train_steps,
                muon_lr=self.args.amuse_muon_lr,
                aux_lr=self.args.amuse_aux_lr,
                joint_out_multiplier=(self.args.joint_out_lr / self.args.amuse_aux_lr),
                joint_other_multiplier=self.args.joint_other_lr_multiplier,
                beta1=self.args.amuse_beta1,
                beta2=self.args.beta2,
                eps=self.args.epsilon,
                momentum=self.args.amuse_momentum,
                rho=self.args.amuse_rho,
                r=self.args.amuse_r,
                weight_lr_power=self.args.amuse_weight_lr_power,
                warmup_ratio=self.args.amuse_warmup_ratio,
                min_warmup_steps=self.args.amuse_min_warmup_steps,
                weight_decay=self.args.weight_decay,
                weight_decay_at_y=self.args.amuse_weight_decay_at_y,
            )
            self.state.amuse_group_fingerprint = result.grouping.fingerprint
            self.state.amuse_warmup_steps = result.warmup_steps
            self.optimizer = result.optimizer
            self.lr_scheduler = AmuseInternalScheduler(result.optimizer)
            cprint(
                f"✅ AMUSE initialized: Muon={result.grouping.muon_numel/1e6:.1f}M, "
                f"aux={result.grouping.fallback_numel/1e6:.1f}M, warmup={result.warmup_steps}",
                "green",
            )
            return

        use_deepspeed_opt = (
            self.accelerator.state.deepspeed_plugin is not None
            and "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.learning_rate,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_deepspeed=use_deepspeed_opt,
        )

        use_deepspeed_lr_scheduler = (
            self.accelerator.state.deepspeed_plugin is not None
            and "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        # AcceleratedScheduler invokes the underlying scheduler num_processes times
        # per sync_gradient event (when split_batches=False, the default). The cosine
        # schedule must be sized to match, otherwise warmup/decay race ahead num_processes×.
        total_training_steps = self.args.train_steps * self.accelerator.num_processes
        num_warmup_steps = self.args.lr_warmup_steps * self.accelerator.num_processes

        if use_deepspeed_lr_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=total_training_steps,
                num_warmup_steps=num_warmup_steps,
            )
        else:
            def cosine_warmup_lambda(step):
                if step < num_warmup_steps:
                    return float(step) / float(max(1, num_warmup_steps))
                progress = float(step - num_warmup_steps) / float(max(1, total_training_steps - num_warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * progress)))
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=cosine_warmup_lambda,
            )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {"encoded_videos": [], "encoded_depth": [], "encoded_flow": [], "img_latent": [], "depth_latent": [], "flow_latent": [],
                "null_embedding": [], "text_embedding": [], "sample_ids": []}
        for sample in samples:
            encoded_video = sample["encoded_video"]
            encoded_depth = sample["encoded_depth"]
            encoded_flow = sample["encoded_flow"]
            img_latent = sample["img_latent"]
            depth_latent = sample["depth_latent"]
            flow_latent = sample["flow_latent"]
            null_embedding = sample["null_embedding"]
            text_embedding = sample["text_embedding"]


            ret["encoded_videos"].append(encoded_video)
            ret["encoded_depth"].append(encoded_depth)
            ret["encoded_flow"].append(encoded_flow)
            ret["img_latent"].append(img_latent)
            ret["depth_latent"].append(depth_latent)
            ret["flow_latent"].append(flow_latent)
            ret["null_embedding"].append(null_embedding)
            ret["text_embedding"].append(text_embedding)
            if "sample_id" in sample:
                ret["sample_ids"].append(sample["sample_id"])

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["encoded_depth"] = torch.stack(ret["encoded_depth"])
        ret["encoded_flow"] = torch.stack(ret["encoded_flow"])
        ret["img_latent"] = torch.stack(ret["img_latent"])
        ret["depth_latent"] = torch.stack(ret["depth_latent"])
        ret["flow_latent"] = torch.stack(ret["flow_latent"])
        ret["null_embedding"] = torch.stack(ret["null_embedding"])
        ret["text_embedding"] = torch.stack(ret["text_embedding"])
        return ret

    @override
    def prepare_dataset(self) -> None:
        if getattr(self.args, "dataset_format", "legacy") == "colmap_rgbdf":
            if not self.args.managed_training:
                raise ValueError("COLMAP RGB-DF requires managed_training=true")
            if self.args.optimizer.lower() != "amuse":
                raise ValueError("managed COLMAP RGB-DF checkpoints currently require AMUSE")
            if self.args.train_manifest is None or self.args.val_manifest is None:
                raise ValueError("COLMAP RGB-DF requires train_manifest and val_manifest")
            from core.finetune.datasets.colmap_rgbdf_latents import ColmapRGBDFLatentDataset

            self.dataset = ColmapRGBDFLatentDataset(
                self.args.train_manifest,
                expected_split="train",
            )
            self.validation_dataset = ColmapRGBDFLatentDataset(
                self.args.val_manifest,
                expected_split="val",
            )
            unload_model(self.components.vae)
            unload_model(self.components.text_encoder)
            free_memory()
            self.data_loader = torch.utils.data.DataLoader(
                self.dataset,
                collate_fn=self.collate_fn,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_memory,
                shuffle=True,
            )
            self.validation_loader = torch.utils.data.DataLoader(
                self.validation_dataset,
                collate_fn=self.collate_fn,
                batch_size=1,
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_memory,
                shuffle=False,
            )
            self.managed_logger = ScalarTensorBoardLogger(
                Path(self.args.output_dir) / "tensorboard",
                is_main_process=(
                    self.accelerator.is_main_process
                    and getattr(self.args, "managed_tensorboard", True)
                ),
            )
            if self.accelerator.num_processes != 1:
                raise ValueError("managed AMUSE top-K training currently requires one process")
            self.topk_manager = TopKCheckpointManager(
                Path(self.args.output_dir) / "checkpoints",
                k=self.args.topk,
                monitor=self.args.checkpoint_monitor,
                mode=self.args.checkpoint_mode,
            )
            return

        self.components.vae = self.components.vae.to(self.accelerator.device, dtype=self.state.weight_dtype)
        self.components.text_encoder = self.components.text_encoder.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )
        logger.info("Initializing dataset and dataloader")
        from core.finetune.datasets import RynnWorld4DDataset

        self.dataset = RynnWorld4DDataset(
            data_root=self.args.validation_dir, 
            device=self.accelerator.device,           
            trainer=self,
            cache_dir=self.args.cache_dir,
            prompt=self.args.prompt,
        )

        # Prepare VAE and text encoder for encoding
        # self.components.vae.requires_grad_(False)
        # self.components.text_encoder.requires_grad_(False)


        # Precompute latent for video and prompt embedding
        logger.info("Precomputing latent for video and prompt embedding ... Done")

        unload_model(self.components.vae)
        unload_model(self.components.text_encoder)
        free_memory()

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True,
        )
        if hasattr(self.components, "text_encoder"):
            self.components.text_encoder.to("cpu")
        
        if hasattr(self.components, "vae"):
            self.components.vae.to("cpu")

        import gc
        gc.collect()
        torch.cuda.empty_cache()

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        target_module = getattr(self.components.high_noise_model, "module", self.components.high_noise_model)
        model_dtype = target_module.patch_embedding.weight.dtype
        device = self.components.high_noise_model.device

        # latent torch.Size([1, 48, 7, 30, 52])
        # img_latent torch.Size([1, 48, 1, 30, 52])
        # control_video torch.Size([1, 48, 7, 30, 52])
        video_latent = batch["encoded_videos"].to(model_dtype) # [B, 16, 21, 30, 52]
        depth_video_latent = batch["encoded_depth"].to(model_dtype)
        flow_video_latent = batch["encoded_flow"].to(model_dtype)

        img_latent = batch["img_latent"].to(model_dtype)
        depth_latent = batch["depth_latent"].to(model_dtype)
        flow_latent = batch["flow_latent"].to(model_dtype)

        null_embedding = batch["null_embedding"].to(model_dtype)
        text_embedding = batch["text_embedding"].to(model_dtype)

        batch_size, num_channels, num_frames, height, width = video_latent.shape
        
        # Shared noise across the three branches: RGB / depth / flow start from the
        # SAME noise so their denoising trajectories are aligned, which is the most
        # direct way to improve cross-modal consistency. Shapes are guaranteed equal
        # by the dataset's shape-consistency check.
        noise_video = torch.randn_like(video_latent)
        noise_depth = noise_video.clone()
        noise_flow = noise_video.clone()

        num_train_timesteps = self.components.scheduler.config.num_train_timesteps
        # timesteps_idx = torch.randint(0, num_train_timesteps, (batch_size,), device=device).long()
        # sigmas = self.components.scheduler.sigmas.to(device=device, dtype=model_dtype)
        # sigma_t = sigmas[timesteps_idx] / num_train_timesteps
        # sigma_view = sigma_t.view(batch_size, 1, 1, 1, 1)

        timesteps_idx = torch.randint(0, num_train_timesteps, (batch_size,), device=device).long()
        s = timesteps_idx.float() / num_train_timesteps
        flow_shift = self.components.scheduler.config.flow_shift  # 5.0
        sigma_t = flow_shift * s / (1 + (flow_shift - 1) * s)
        sigma_view = sigma_t.view(batch_size, 1, 1, 1, 1).to(model_dtype)
        shifted_timesteps = sigma_t * num_train_timesteps                                                                                                                                                                                         
                                                                                                                                                                                                                    
        noisy_latents = (1.0 - sigma_view) * video_latent + sigma_view * noise_video
        noisy_latents_depth = (1.0 - sigma_view) * depth_video_latent + sigma_view * noise_depth
        noisy_latents_flow = (1.0 - sigma_view) * flow_video_latent + sigma_view * noise_flow

        target_video = noise_video - video_latent
        target_depth = noise_depth - depth_video_latent
        target_flow = noise_flow - flow_video_latent

        noisy_latents[:, :, 0:1, :, :] = img_latent
        noisy_latents_depth[:, :, 0:1, :, :] = depth_latent
        noisy_latents_flow[:, :, 0:1, :, :] = flow_latent

        # Branch dropout: randomly corrupt one branch's noisy latents (except the first frame).
        # Each DP rank samples independently — different ranks see different data shards
        # anyway, so per-rank dropout just adds more diversity to the global batch.
        dropout_hit = False
        if getattr(self.args, 'branch_dropout_prob', 0.0) > 0:
            allowed_modes = [m for m in self.args.branch_dropout_modes if m != 'video']
            if len(allowed_modes) != len(self.args.branch_dropout_modes) and self.accelerator.is_main_process:
                cprint("⚠️ branch_dropout_modes included 'video'; RGB anchor is protected and will never be dropped.", "yellow")
            if allowed_modes and random.random() < self.args.branch_dropout_prob:
                chosen = random.choice(allowed_modes)
                if chosen == 'depth':
                    noisy_latents_depth[:, :, 1:, :, :] = torch.randn_like(noisy_latents_depth[:, :, 1:, :, :])
                else:
                    noisy_latents_flow[:, :, 1:, :, :] = torch.randn_like(noisy_latents_flow[:, :, 1:, :, :])
                dropout_hit = True

        # Classifier-free guidance: randomly drop text conditioning (per-rank, same rationale).
        if random.random() < 0.15:
            text_embedding = null_embedding

        first_frame_mask = torch.ones(1, 1, num_frames, height, width, device=device)
        first_frame_mask[:, :, 0] = 0

        # temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * timesteps).flatten()
        # timestep = temp_ts.unsqueeze(0).expand(video_latent.shape[0], -1)
        # temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * timesteps_idx.view(-1, 1, 1, 1).float()).flatten(1)
        temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * shifted_timesteps.view(-1, 1, 1, 1).float()).flatten(1)
        timestep_input = temp_ts.to(model_dtype)

        video_pred, depth_pred, flow_pred = self.components.high_noise_model(
            hidden_states=noisy_latents,
            hidden_states_depth=noisy_latents_depth,
            hidden_states_flow=noisy_latents_flow,
            timestep=timestep_input,
            encoder_hidden_states=text_embedding,
            encoder_hidden_states_image=None,
            attention_kwargs=None,
            return_dict=False,
        )

        loss_video = F.mse_loss(
            video_pred[:, :, 1:].float(), 
            target_video[:, :, 1:].float(), 
            reduction="mean"
        )

        loss_depth = F.mse_loss(
            depth_pred[:, :, 1:].float(), 
            target_depth[:, :, 1:].float(), 
            reduction="mean"
        )

        loss_flow = F.mse_loss(
            flow_pred[:, :, 1:].float(), 
            target_flow[:, :, 1:].float(), 
            reduction="mean"
        )

        loss = loss_video + loss_depth + self.args.loss_weight_flow * loss_flow
        # loss = loss_video
        return loss, {
            "loss_video": loss_video.detach(),
            "loss_depth": loss_depth.detach(),
            "loss_flow": loss_flow.detach(),
            "dropout_hit": dropout_hit,
        }

    def _validate_managed(self, global_step: int) -> dict[str, float]:
        if not getattr(self.args, "metric_validation", False):
            raise ValueError("managed validation is disabled")
        raw_optimizer = find_amuse_optimizer(self.optimizer) if self.args.optimizer.lower() == "amuse" else None
        python_state = random.getstate()
        cpu_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all()
        self.components.high_noise_model.eval()
        if raw_optimizer is not None:
            set_amuse_mode(self.optimizer, training=False)
        totals = {"total": 0.0, "rgb": 0.0, "depth": 0.0, "flow": 0.0}
        count = 0
        try:
            with torch.no_grad():
                for batch in self.validation_loader:
                    sample_ids = batch.get("sample_ids", [])
                    seed_payload = ":".join(sample_ids).encode()
                    sample_seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "little") % (2**31)
                    random.seed(sample_seed)
                    torch.manual_seed(sample_seed)
                    torch.cuda.manual_seed_all(sample_seed)
                    loss, info = self.compute_loss(batch)
                    totals["total"] += float(loss)
                    totals["rgb"] += float(info["loss_video"])
                    totals["depth"] += float(info["loss_depth"])
                    totals["flow"] += float(info["loss_flow"])
                    count += int(batch["encoded_videos"].shape[0])
        finally:
            torch.random.set_rng_state(cpu_state)
            torch.cuda.set_rng_state_all(cuda_states)
            random.setstate(python_state)
            if raw_optimizer is not None:
                set_amuse_mode(self.optimizer, training=True)
            self.components.high_noise_model.train()
        if count == 0:
            raise ValueError("validation loader is empty")
        metrics = {f"val/loss_{name}": value / count for name, value in totals.items()}
        if self.accelerator.is_main_process:
            self.managed_logger.log_many(metrics, global_step)
            self.managed_logger.flush()
        return metrics

    def _managed_validation_checkpoint(self, epoch: int, global_step: int) -> None:
        metrics = self._validate_managed(global_step)
        if not self.accelerator.is_main_process:
            return
        metric = metrics[self.args.checkpoint_monitor]
        raw_optimizer = find_amuse_optimizer(self.optimizer) if self.args.optimizer.lower() == "amuse" else None
        if raw_optimizer is not None:
            set_amuse_mode(self.optimizer, training=False)
        self.components.high_noise_model.eval()
        try:
            self.topk_manager.consider(
                metric,
                epoch=epoch,
                step=global_step,
                save=lambda path: save_compact_amuse_checkpoint(
                    path,
                    model=self.components.high_noise_model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    metric=metric,
                    config_fingerprint=self._managed_config_fingerprint(),
                    grouping_fingerprint=self.state.amuse_group_fingerprint,
                    pretrained_fingerprint=self._managed_pretrained_fingerprint(),
                    data_fingerprints={
                        "train": self.dataset.fingerprint,
                        "val": self.validation_dataset.fingerprint,
                    },
                    generator=self.state.generator,
                ),
            )
        finally:
            if raw_optimizer is not None:
                set_amuse_mode(self.optimizer, training=True)
            self.components.high_noise_model.train()

    def _managed_config_fingerprint(self) -> str:
        fields = (
            "amuse_aux_lr",
            "amuse_beta1",
            "amuse_min_warmup_steps",
            "amuse_momentum",
            "amuse_muon_lr",
            "amuse_r",
            "amuse_rho",
            "amuse_warmup_ratio",
            "amuse_weight_decay_at_y",
            "amuse_weight_lr_power",
            "batch_size",
            "beta2",
            "branch_dropout_modes",
            "branch_dropout_prob",
            "epsilon",
            "freeze_non_joint",
            "fusion_mode",
            "gradient_accumulation_steps",
            "joint_end_layer",
            "joint_every_n_layers",
            "joint_frame_wise",
            "joint_other_lr_multiplier",
            "joint_out_lr",
            "joint_start_layer",
            "joint_unidirectional",
            "joint_use_rope",
            "joint_video_decay",
            "joint_video_decay_steps",
            "loss_weight_flow",
            "max_grad_norm",
            "mixed_precision",
            "model_name",
            "model_type",
            "optimizer",
            "seed",
            "share_ffn",
            "train_resolution",
            "training_type",
            "weight_decay",
        )
        payload = {name: getattr(self.args, name) for name in fields}
        payload["amuse_warmup_steps"] = self.state.amuse_warmup_steps
        payload["scheduler_config"] = dict(self.components.scheduler.config)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _managed_pretrained_fingerprint(self) -> str:
        cached = getattr(self, "_pretrained_fingerprint", None)
        if cached is not None:
            return cached
        root = Path(self.args.load_stage2_model_weights or "")
        checkpoint = root / "pytorch_model" / "mp_rank_00_model_states.pt"
        if not checkpoint.is_file():
            raise RuntimeError("managed checkpointing requires the released pretrained model artifact")
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            while chunk := handle.read(16 * 1024 * 1024):
                digest.update(chunk)
        self._pretrained_fingerprint = digest.hexdigest()
        return self._pretrained_fingerprint

    @override
    def train(self) -> None:
        logger.info("Starting training")

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.total_batch_size_count = (self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps)
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset),
            "train epochs": self.args.train_epochs,
            "train steps": self.args.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.data_loader),
            "train batch size total count": self.state.total_batch_size_count,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        # Potentially load in the weights and states from a previous save
        if getattr(self.args, "managed_training", False):
            resume_from_checkpoint_path = self.args.resume_from_checkpoint
            if resume_from_checkpoint_path is not None:
                if str(resume_from_checkpoint_path) == "latest" or not Path(resume_from_checkpoint_path).is_dir():
                    raise ValueError("managed training requires an existing explicit compact checkpoint directory")
                resume = load_compact_amuse_checkpoint(
                    resume_from_checkpoint_path,
                    model=self.components.high_noise_model,
                    optimizer=self.optimizer,
                    config_fingerprint=self._managed_config_fingerprint(),
                    grouping_fingerprint=self.state.amuse_group_fingerprint,
                    pretrained_fingerprint=self._managed_pretrained_fingerprint(),
                    data_fingerprints={
                        "train": self.dataset.fingerprint,
                        "val": self.validation_dataset.fingerprint,
                    },
                    generator=self.state.generator,
                )
                initial_global_step = resume.global_step
                global_step = resume.global_step
                first_epoch = resume.first_epoch
                if global_step >= self.args.train_steps or first_epoch >= self.args.train_epochs:
                    raise ValueError("compact checkpoint has no remaining configured training steps")
        else:
            (
                resume_from_checkpoint_path,
                initial_global_step,
                global_step,
                first_epoch,
            ) = get_latest_ckpt_path_to_resume_from(
                resume_from_checkpoint=self.args.resume_from_checkpoint,
                num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
                output_dir=self.args.output_dir,
            )
        if resume_from_checkpoint_path is not None and not getattr(self.args, "managed_training", False):
            self.accelerator.load_state(resume_from_checkpoint_path)
            # Resume EMA weights if available
            if self.ema_model is not None:
                ema_path = os.path.join(resume_from_checkpoint_path, "ema_weights.pt")
                if os.path.exists(ema_path):
                    try:
                        ema_state_dict = torch.load(ema_path, map_location="cpu", weights_only=False)
                        # Load shadow params by matching names
                        for i, name in enumerate(self._ema_param_names):
                            if name in ema_state_dict:
                                self.ema_model.shadow_params[i].copy_(ema_state_dict[name])
                        cprint(f"✅ EMA weights resumed from {ema_path} ({len(ema_state_dict)} params)", "green")
                    except Exception as e:
                        cprint(f"⚠️ Failed to load EMA from {ema_path}: {e}. Starting fresh EMA.", "yellow")
                else:
                    cprint(f"⚠️ EMA enabled but no ema_weights.pt found in {resume_from_checkpoint_path}. Starting fresh EMA.", "yellow")

        # Load EMA weights from stage2 checkpoint if using load_stage2_model_weights (different GPU count)
        if self._stage2_ema_path is not None and self.ema_model is not None:
            try:
                ema_state_dict = torch.load(self._stage2_ema_path, map_location="cpu", weights_only=False)
                loaded = 0
                for i, name in enumerate(self._ema_param_names):
                    if name in ema_state_dict:
                        self.ema_model.shadow_params[i].copy_(ema_state_dict[name])
                        loaded += 1
                cprint(f"✅ EMA weights loaded from {self._stage2_ema_path} ({loaded}/{len(self._ema_param_names)} params matched)", "green")
            except Exception as e:
                cprint(f"⚠️ Failed to load stage2 EMA from {self._stage2_ema_path}: {e}. Starting fresh EMA.", "yellow")

        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.accelerator.is_local_main_process,
        )

        import time

        step_start_time = time.time()
        
        total_samples_processed = 0
        training_start_time = time.time()
        
        if self.accelerator.is_main_process:
            print("\n" + "="*80)
            print(f"{'Global Step':<15} | {'Samples This Step':<20} | {'Instant Throughput':<25} | {'Average Throughput':<25}")
            print("="*80)

        if self.args.optimizer.lower() == "amuse":
            set_amuse_mode(self.optimizer, training=True)

        free_memory()
        for epoch in range(first_epoch, self.args.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.args.train_epochs})")

            self.components.high_noise_model.train()
            # models_to_accumulate = [self.components.transformer]
            models_to_accumulate = [self.components.high_noise_model]

            for step, batch in enumerate(self.data_loader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}

                # Update joint_gate_video_decay (cosine 1.0→0.0 over decay_steps)
                # before the forward pass so this step's decay is applied correctly.
                if getattr(self.args, 'joint_video_decay', False):
                    decay_steps = getattr(self.args, 'joint_video_decay_steps', 700)
                    progress = min(1.0, global_step / decay_steps)
                    decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                    unwrapped_for_decay = unwrap_model(self.accelerator, self.components.high_noise_model)
                    for blk in unwrapped_for_decay.blocks:
                        if hasattr(blk, 'joint_gate_video_decay'):
                            blk.joint_gate_video_decay.fill_(decay)

                with accelerator.accumulate(models_to_accumulate):
                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    loss, loss_info = self.compute_loss(batch)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            # grad_norm = self.components.transformer.get_global_grad_norm()
                            grad_norm_high = self.components.high_noise_model.get_global_grad_norm()
                            # grad_norm_low = self.components.low_noise_model.get_global_grad_norm()
                            grad_norm = (grad_norm_high**2)**0.5
                            # In some cases the grad norm may not return a float
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        else:
                            unwrapped_model = self.accelerator.unwrap_model(self.components.high_noise_model)
                            grad_norm = accelerator.clip_grad_norm_(
                                unwrapped_model.parameters(), 
                                self.args.max_grad_norm
                            )
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()

                        logs["grad_norm"] = grad_norm

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    # Update EMA weights only on optimizer-step boundaries to avoid
                    # 4× redundant CPU copies under gradient accumulation.
                    if self.ema_model is not None and accelerator.sync_gradients:
                        unwrapped = unwrap_model(self.accelerator, self.components.high_noise_model)
                        trainable_params_cpu = [
                            p.detach().to("cpu", non_blocking=True)
                            for p in unwrapped.parameters() if p.requires_grad
                        ]
                        self.ema_model.step(trainable_params_cpu)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self._maybe_save_checkpoint(global_step)

                    # Periodic inference is currently disabled — the run_periodic_inference
                    # helper imports (RynnWorld4DPipeline, export_to_video, load_file) are not
                    # wired up in this module. Until those imports land, log and skip rather
                    # than silently call into a stub.
                    if getattr(self.args, 'periodic_inference_steps', 0) > 0:
                        if global_step % self.args.periodic_inference_steps == 0:
                            if self.accelerator.is_main_process:
                                logger.info(
                                    f"[step {global_step}] periodic inference is disabled "
                                    "(missing RynnWorld4DPipeline/export_to_video imports)."
                                )

                    samples_in_this_step = (
                        batch['encoded_videos'].shape[0]
                        * self.accelerator.num_processes
                        * self.args.gradient_accumulation_steps
                    )
                    
                    step_end_time = time.time()
                    step_duration = step_end_time - step_start_time
                    instant_throughput = samples_in_this_step / step_duration if step_duration > 0 else 0
                    step_start_time = step_end_time

                    total_samples_processed += samples_in_this_step
                    total_training_time = time.time() - training_start_time
                    average_throughput = total_samples_processed / total_training_time if total_training_time > 0 else 0

                    if self.accelerator.is_main_process and (global_step % 10 == 0 or global_step == 1):
                        print(f"{global_step:<15} | {samples_in_this_step:<20} | {instant_throughput:<25.2f} samples/sec | {average_throughput:<25.2f} samples/sec")


                logs["loss"] = loss.detach().item()
                logs["lr"] = self.lr_scheduler.get_last_lr()[0]

                # Per-branch loss & dropout-clean loss for diagnosing modality consistency.
                logs["loss_video"] = loss_info["loss_video"].item()
                logs["loss_depth"] = loss_info["loss_depth"].item()
                logs["loss_flow"]  = loss_info["loss_flow"].item()
                if not loss_info["dropout_hit"]:
                    logs["loss_clean"] = logs["loss"]

                if getattr(self.args, "managed_training", False) and accelerator.sync_gradients:
                    learning_rates = self.lr_scheduler.get_last_lr()
                    managed_logs = {
                        "train/loss_total": logs["loss"],
                        "train/loss_rgb": logs["loss_video"],
                        "train/loss_depth": logs["loss_depth"],
                        "train/loss_flow": logs["loss_flow"],
                        "train/grad_norm": logs["grad_norm"],
                        "optimizer/lr": learning_rates[0],
                        "system/gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                        "system/gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                        "system/samples_per_second": instant_throughput,
                    }
                    if self.args.optimizer.lower() == "amuse":
                        managed_logs["optimizer/lr_muon"] = learning_rates[0]
                        managed_logs["optimizer/lr_aux"] = learning_rates[-1]
                    self.managed_logger.log_many(managed_logs, global_step)

                # joint_gate means every 50 optimizer steps — confirms cross-modal pathways
                # are active (gate=0 means the joint branch has been shut off by the model).
                if accelerator.sync_gradients and global_step % 50 == 0:
                    unwrapped_for_gate = unwrap_model(self.accelerator, self.components.high_noise_model)
                    g_v, g_d, g_f, n_blk = 0.0, 0.0, 0.0, 0
                    r_v, r_d, r_f, n_ratio = 0.0, 0.0, 0.0, 0
                    for blk in unwrapped_for_gate.blocks:
                        if hasattr(blk, "joint_gate_video"):
                            g_v += float(blk.joint_gate_video.detach().tanh().abs().item())
                            g_d += float(blk.joint_gate_depth.detach().tanh().abs().item())
                            g_f += float(blk.joint_gate_flow.detach().tanh().abs().item())
                            n_blk += 1
                            if hasattr(blk, "_joint_ratio_video"):
                                r_v += float(blk._joint_ratio_video.item())
                                r_d += float(blk._joint_ratio_depth.item())
                                r_f += float(blk._joint_ratio_flow.item())
                                n_ratio += 1
                    if n_blk > 0:
                        logs["joint_gate_video"] = g_v / n_blk
                        logs["joint_gate_depth"] = g_d / n_blk
                        logs["joint_gate_flow"]  = g_f / n_blk
                    if n_ratio > 0:
                        logs["joint_ratio_video"] = r_v / n_ratio
                        logs["joint_ratio_depth"] = r_d / n_ratio
                        logs["joint_ratio_flow"]  = r_f / n_ratio

                # Log the current decay factor so we can verify the schedule in tensorboard.
                if getattr(self.args, 'joint_video_decay', False) and accelerator.sync_gradients and global_step % 50 == 0:
                    decay_steps = getattr(self.args, 'joint_video_decay_steps', 700)
                    logs["joint_gate_video_decay"] = 0.5 * (1.0 + math.cos(math.pi * min(1.0, global_step / decay_steps)))

                progress_bar.set_postfix(logs)

                if self.accelerator.is_main_process and (global_step % 10 == 0 or global_step == 1):
                    log_str = f"Epoch: {epoch+1}, Step: {global_step}/{self.args.train_steps}, "
                    log_str += f"Loss: {logs['loss']:.4f}, LR: {logs['lr']:.2e}"
                    if "grad_norm" in logs:
                        log_str += f", Grad Norm: {logs['grad_norm']:.4f}"
                    
                    logger.info(log_str)

                accelerator.log(logs, step=global_step)

                if global_step >= self.args.train_steps:
                    break

            if (
                getattr(self.args, "managed_training", False)
                and (epoch + 1) % self.args.validation_every_epochs == 0
            ):
                self._managed_validation_checkpoint(epoch, global_step)

            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

        accelerator.wait_for_everyone()
        if getattr(self.args, "save_final_checkpoint", True) and not getattr(self.args, "managed_training", False):
            self._maybe_save_checkpoint(global_step, must_save=True)

        # Final periodic inference — disabled until run_periodic_inference imports are wired up
        if getattr(self.args, 'periodic_inference_steps', 0) > 0:
            free_memory()
            if self.accelerator.is_main_process:
                logger.info("Final periodic inference skipped (imports not wired up).")

        if self.args.do_validation:
            free_memory()
            self.validate(global_step)

        if getattr(self.args, "managed_training", False):
            self.managed_logger.flush()
            self.managed_logger.close()

        del self.components
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()

    @override
    def _maybe_save_checkpoint(self, global_step: int, must_save: bool = False):
        if getattr(self.args, "managed_training", False):
            return
        if not (must_save or global_step % self.args.checkpointing_steps == 0):
            return
        save_path = get_intermediate_ckpt_path(
            checkpointing_limit=self.args.checkpointing_limit,
            step=global_step,
            output_dir=self.args.output_dir,
        )
        if self.accelerator.is_main_process:
            logger.info(f"Checkpointing at step {global_step}")
            logger.info(f"Saving state to {save_path}")
            os.makedirs(save_path, exist_ok=True)
        # Free up fragmented GPU memory before ZeRO state gather
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(save_path, safe_serialization=True)

        # Save EMA weights
        if self.ema_model is not None and self.accelerator.is_main_process:
            ema_save_path = os.path.join(save_path, "ema_weights.pt")
            # Save as named state dict for easy inference loading
            ema_state_dict = {}
            for name, shadow_param in zip(self._ema_param_names, self.ema_model.shadow_params):
                ema_state_dict[name] = shadow_param.detach().cpu().clone()
            torch.save(ema_state_dict, ema_save_path)
            logger.info(f"EMA weights saved to {ema_save_path} ({len(ema_state_dict)} params)")

    def _get_latest_checkpoint_for_lora(self) -> Optional[str]:
        """Find the latest checkpoint directory that might contain LoRA weights."""
        output_dir = str(self.args.output_dir)
        if not os.path.exists(output_dir):
            return None
        
        ckpt_dirs = []
        for d in os.listdir(output_dir):
            full_path = os.path.join(output_dir, d)
            if os.path.isdir(full_path) and d.startswith("checkpoint-"):
                try:
                    step = int(d.split("-")[1])
                    ckpt_dirs.append((step, full_path))
                except ValueError:
                    continue
        
        if not ckpt_dirs:
            return None
        
        ckpt_dirs.sort(key=lambda x: x[0])
        return ckpt_dirs[-1][1]

    def run_periodic_inference(self, global_step: int) -> None:
        """Periodic inference is currently disabled.

        The original implementation depended on RynnWorld4DPipeline / export_to_video /
        load_file imports that aren't wired up in this module. Until those land we
        log and skip rather than silently doing nothing.
        """
        if not self.accelerator.is_main_process:
            return
        logger.info(
            f"[step {global_step}] run_periodic_inference is a no-op until "
            "RynnWorld4DPipeline/export_to_video/load_file imports are wired up."
        )


register("rynnworld4d", "sft", RynnWorld4DTrainer)
