import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import PIL.Image
import torch
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from controlnet import ControlNetModel #fix
# from diffusers import AutoencoderKL, ControlNetModel, UNet2DConditionModel 
# from diffusers.schedulers import KarrasDiffusionSchedulers 
from diffusers.utils import (
    PIL_INTERPOLATION,
    # is_accelerate_available, 
    # is_accelerate_version, 
    logging,
    # randn_tensor, 
    # replace_example_docstring, 
)
# from diffusers.pipeline_utils import DiffusionPipeline 
# from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker 
# from diffusers.models.controlnet import ControlNetOutput 

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ControlNetProcessor(object):
    def __init__(
        self,
        controlnet: ControlNetModel,
        # image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]],
        controlnet_cond = torch.FloatTensor, #fix
        conditioning_scale: float = 1.0,
    ):
        self.controlnet = controlnet
        # self.image = image
        self.controlnet_cond = controlnet_cond #fix
        self.conditioning_scale = conditioning_scale

    # def _default_height_width(self, height, width, image):
    #     if isinstance(image, list):
    #         image = image[0]

    #     if height is None:
    #         if isinstance(image, PIL.Image.Image):
    #             height = image.height
    #         elif isinstance(image, torch.Tensor):
    #             height = image.shape[3]

    #         height = (height // 8) * 8  # round down to nearest multiple of 8

    #     if width is None:
    #         if isinstance(image, PIL.Image.Image):
    #             width = image.width
    #         elif isinstance(image, torch.Tensor):
    #             width = image.shape[2]

    #         width = (width // 8) * 8  # round down to nearest multiple of 8

    #     return height, width

    # def default_height_width(self, height, width):
    #     return self._default_height_width(height, width, self.image)

    # def _prepare_image(self, image, width, height, batch_size, num_images_per_prompt, device, dtype):
    #     if not isinstance(image, torch.Tensor):
    #         if isinstance(image, PIL.Image.Image):
    #             image = [image]

    #         if isinstance(image[0], PIL.Image.Image):
    #             image = [
    #                 np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image
    #             ]
    #             image = np.concatenate(image, axis=0)
    #             image = np.array(image).astype(np.float32) / 255.0
    #             image = image.transpose(0, 3, 1, 2)
    #             image = torch.from_numpy(image)
    #         elif isinstance(image[0], torch.Tensor):
    #             image = torch.cat(image, dim=0)

    #     image_batch_size = image.shape[0]

    #     if image_batch_size == 1:
    #         repeat_by = batch_size
    #     else:
    #         # image batch size is the same as prompt batch size
    #         repeat_by = num_images_per_prompt

    #     image = image.repeat_interleave(repeat_by, dim=0)

    #     image = image.to(device=device, dtype=dtype)

    #     return image

    # def _check_inputs(self, image, prompt, prompt_embeds):
    #     image_is_pil = isinstance(image, PIL.Image.Image)
    #     image_is_tensor = isinstance(image, torch.Tensor)
    #     image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
    #     image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)

    #     if not image_is_pil and not image_is_tensor and not image_is_pil_list and not image_is_tensor_list:
    #         raise TypeError(
    #             "image must be passed and be one of PIL image, torch tensor, list of PIL images, or list of torch tensors"
    #         )

    #     if image_is_pil:
    #         image_batch_size = 1
    #     elif image_is_tensor:
    #         image_batch_size = image.shape[0]
    #     elif image_is_pil_list:
    #         image_batch_size = len(image)
    #     elif image_is_tensor_list:
    #         image_batch_size = len(image)

    #     if prompt is not None and isinstance(prompt, str):
    #         prompt_batch_size = 1
    #     elif prompt is not None and isinstance(prompt, list):
    #         prompt_batch_size = len(prompt)
    #     elif prompt_embeds is not None:
    #         prompt_batch_size = prompt_embeds.shape[0]

    #     if image_batch_size != 1 and image_batch_size != prompt_batch_size:
    #         raise ValueError(
    #             f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
    #         )

    # def check_inputs(self, prompt, prompt_embeds):
    #     self._check_inputs(self.image, prompt, prompt_embeds)

    # def prepare_image(self, width, height, batch_size, num_images_per_prompt, device, do_classifier_free_guidance):
    #     self.image = self._prepare_image(
    #         self.image, width, height, batch_size, num_images_per_prompt, device, self.controlnet.dtype
    #     )
    #     if do_classifier_free_guidance:
    #         self.image = torch.cat([self.image] * 2)

    def __call__(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Tuple:
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            sample=sample,
            controlnet_cond=self.controlnet_cond, #fix
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )
        down_block_res_samples = [
            down_block_res_sample * self.conditioning_scale for down_block_res_sample in down_block_res_samples
        ]
        mid_block_res_sample *= self.conditioning_scale
        return (down_block_res_samples, mid_block_res_sample)
