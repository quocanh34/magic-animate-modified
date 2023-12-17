# Copyright 2023 ByteDance and/or its affiliates.
#
# Copyright (2023) MagicAnimate Authors
#
# ByteDance, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from ByteDance or
# its affiliates is strictly prohibited.
import argparse
import argparse
import datetime
import inspect
import os
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from collections import OrderedDict

import torch

from diffusers import AutoencoderKL, DDIMScheduler, UniPCMultistepScheduler

from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from magicanimate.models.unet_controlnet import UNet3DConditionModel
from magicanimate.models.controlnet import ControlNetModel
from magicanimate.models.appearance_encoder import AppearanceEncoderModel
from magicanimate.models.mutual_self_attention import ReferenceAttentionControl
from magicanimate.pipelines.pipeline_animation import AnimationPipeline
from magicanimate.models.multicontrolnet import ControlNetProcessor #fix
from magicanimate.utils.util import save_videos_grid
from accelerate.utils import set_seed

from magicanimate.utils.videoreader import VideoReader

from einops import rearrange, repeat

import csv, pdb, glob
from safetensors import safe_open
import math
from pathlib import Path

class MagicAnimate():
    def __init__(self, config="configs/prompts/animation.yaml") -> None:
        print("Initializing MagicAnimate Pipeline...")
        *_, func_args = inspect.getargvalues(inspect.currentframe())
        func_args = dict(func_args)
        
        config  = OmegaConf.load(config)
        
        inference_config = OmegaConf.load(config.inference_config)
            
        motion_module = config.motion_module
       
        ### >>> create animation pipeline >>> ###
        tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_path, subfolder="text_encoder")
        if config.pretrained_unet_path:
            unet = UNet3DConditionModel.from_pretrained_2d(config.pretrained_unet_path, unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))
        else:
            unet = UNet3DConditionModel.from_pretrained_2d(config.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))
        self.appearance_encoder = AppearanceEncoderModel.from_pretrained(config.pretrained_appearance_encoder_path, subfolder="appearance_encoder").cuda()
        self.reference_control_writer = ReferenceAttentionControl(self.appearance_encoder, do_classifier_free_guidance=True, mode='write', fusion_blocks=config.fusion_blocks)
        self.reference_control_reader = ReferenceAttentionControl(unet, do_classifier_free_guidance=True, mode='read', fusion_blocks=config.fusion_blocks)
        if config.pretrained_vae_path is not None:
            vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path)
        else:
            vae = AutoencoderKL.from_pretrained(config.pretrained_model_path, subfolder="vae")

        ### Load controlnet
        controlnet1  = ControlNetModel.from_pretrained(config.pretrained_controlnet_path1) #fix
        controlnet2  = ControlNetModel.from_pretrained(config.pretrained_controlnet_path2) #fix

        vae.to(torch.float16)
        unet.to(torch.float16)
        text_encoder.to(torch.float16)
        # controlnet.to(torch.float16)
        controlnet1 = controlnet1.to(torch.float16).to("cuda") #fix
        controlnet2 = controlnet2.to(torch.float16).to("cuda") #fix

        self.appearance_encoder.to(torch.float16)
        
        unet.enable_xformers_memory_efficient_attention()
        self.appearance_encoder.enable_xformers_memory_efficient_attention()
        # controlnet.enable_xformers_memory_efficient_attention()
        controlnet1.enable_xformers_memory_efficient_attention() #fix
        controlnet2.enable_xformers_memory_efficient_attention() #fix

        self.processors = [ControlNetProcessor(controlnet1), ControlNetProcessor(controlnet2)]

        self.pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            # NOTE: UniPCMultistepScheduler
        ).to("cuda")

        # 1. unet ckpt
        # 1.1 motion module
        motion_module_state_dict = torch.load(motion_module, map_location="cpu")
        if "global_step" in motion_module_state_dict: func_args.update({"global_step": motion_module_state_dict["global_step"]})
        motion_module_state_dict = motion_module_state_dict['state_dict'] if 'state_dict' in motion_module_state_dict else motion_module_state_dict
        try:
            # extra steps for self-trained models
            state_dict = OrderedDict()
            for key in motion_module_state_dict.keys():
                if key.startswith("module."):
                    _key = key.split("module.")[-1]
                    state_dict[_key] = motion_module_state_dict[key]
                else:
                    state_dict[key] = motion_module_state_dict[key]
            motion_module_state_dict = state_dict
            del state_dict
            missing, unexpected = self.pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
            assert len(unexpected) == 0
        except:
            _tmp_ = OrderedDict()
            for key in motion_module_state_dict.keys():
                if "motion_modules" in key:
                    if key.startswith("unet."):
                        _key = key.split('unet.')[-1]
                        _tmp_[_key] = motion_module_state_dict[key]
                    else:
                        _tmp_[key] = motion_module_state_dict[key]
            missing, unexpected = unet.load_state_dict(_tmp_, strict=False)
            assert len(unexpected) == 0
            del _tmp_
        del motion_module_state_dict

        self.pipeline.to("cuda")
        self.L = config.L
        
        print("Initialization Done!")
        
    def __call__(self, source_image, motion_sequence_list, random_seed, step, guidance_scale, size=512):
            prompt = n_prompt = ""
            random_seed = int(random_seed)
            step = int(step)
            guidance_scale = float(guidance_scale)
            samples_per_video = []
            # manually set random seed for reproduction
            if random_seed != -1: 
                torch.manual_seed(random_seed)
                set_seed(random_seed)
            else:
                torch.seed()
            motion_sequence1 = motion_sequence_list[0]
            motion_sequence2 = motion_sequence_list[1]

            if motion_sequence1.endswith('.mp4'):
                control1 = VideoReader(motion_sequence1).read()
                if control1[0].shape[0] != size:
                    control1 = [np.array(Image.fromarray(c).resize((size, size))) for c in control1]
                control1 = np.array(control1)

            if motion_sequence2.endswith('.mp4'):
                control2 = VideoReader(motion_sequence2).read()
                if control2[0].shape[0] != size:
                    control2 = [np.array(Image.fromarray(c).resize((size, size))) for c in control2]
                control2 = np.array(control2)
            
            if source_image.shape[0] != size:
                source_image = np.array(Image.fromarray(source_image).resize((size, size)))
            H, W, C = source_image.shape
            
            init_latents = None
            original_length = control1.shape[0]

            if control1.shape[0] % self.L > 0:
                control1 = np.pad(control1, ((0, self.L-control1.shape[0] % self.L), (0, 0), (0, 0), (0, 0)), mode='edge')

            if control2.shape[0] % self.L > 0:
                control2 = np.pad(control2, ((0, self.L-control2.shape[0] % self.L), (0, 0), (0, 0), (0, 0)), mode='edge')

            generator = torch.Generator(device=torch.device("cuda:0"))

            generator.manual_seed(torch.initial_seed())

            sample = self.pipeline(
                prompt,
                processors              = self.processors,
                negative_prompt         = n_prompt,
                num_inference_steps     = step,
                guidance_scale          = guidance_scale,
                width                   = W,
                height                  = H,
                video_length            = len(control1),
                controlnet_condition1    = control1,
                controlnet_condition2    = control2,
                init_latents            = init_latents,
                generator               = generator,
                appearance_encoder       = self.appearance_encoder, 
                reference_control_writer = self.reference_control_writer,
                reference_control_reader = self.reference_control_reader,
                source_image             = source_image,
            ).videos

            source_images = np.array([source_image] * original_length)
            source_images = rearrange(torch.from_numpy(source_images), "t h w c -> 1 c t h w") / 255.0
            samples_per_video.append(source_images)
            
            control = (control1*8+control2*2)/10
            control = control / 255.0
            control = rearrange(control, "t h w c -> 1 c t h w")
            control = torch.from_numpy(control)
            samples_per_video.append(control[:, :, :original_length])

            samples_per_video.append(sample[:, :, :original_length])

            samples_per_video = torch.cat(samples_per_video)

            time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            savedir = f"demo/outputs"
            animation_path = f"{savedir}/{time_str}.mp4"

            os.makedirs(savedir, exist_ok=True)
            save_videos_grid(samples_per_video, animation_path)
            
            return animation_path