import os
import time
from typing import Dict, List

import numpy as np
import torch
import yaml
from PIL import Image
from omegaconf import OmegaConf
import argparse

from libs.utils.utils import merge_sweep_config
from libs.model import make_pipeline
from libs.model.module.scheduler import CustomDDIMScheduler
from libs.utils.controlnet_processor import make_processor


def freecontrol_generate(args):
    control_type = args.condition
    used_approach = args.approach

    if not control_type == "None":
        processor = make_processor(control_type.lower())
    else:
        processor = lambda x: Image.open(x).convert("RGB") if type(x) == str else x

    model_path = model_dict[args.sd_version][args.model_ckpt]['path']

    gradio_update_parameter = {
        'sd_config--approach': args.approach,
        'sd_config--guidance_scale': args.scale,
        'sd_config--steps': args.ddim_steps,
        'sd_config--seed': args.seed,
        'sd_config--dreambooth': False,
        'sd_config--prompt': args.prompt,
        'sd_config--negative_prompt': args.negative_prompt,
        'sd_config--obj_pairs': str(args.paired_objs),
        'sd_config--pca_paths': [pca_basis_dict[args.sd_version][args.model_ckpt][args.pca_basis]],

        'data--inversion--prompt': args.inversion_prompt,
        'data--inversion--fixed_size': None,

        'guidance--pca_guidance--end_step': int(args.pca_guidance_steps * args.ddim_steps),
        'guidance--pca_guidance--weight': args.pca_guidance_weight,
        'guidance--pca_guidance--structure_guidance--n_components': args.pca_guidance_components,
        'guidance--pca_guidance--structure_guidance--normalize': bool(args.pca_guidance_normalized),
        'guidance--pca_guidance--structure_guidance--mask_tr': args.pca_masked_tr,
        'guidance--pca_guidance--structure_guidance--penalty_factor': args.pca_guidance_penalty_factor,

        'guidance--pca_guidance--warm_up--apply': True if args.pca_warm_up_step > 0 else False,
        'guidance--pca_guidance--warm_up--end_step': int(args.pca_warm_up_step * args.ddim_steps),
        'guidance--pca_guidance--appearance_guidance--apply': True if args.pca_texture_reg_tr > 0 else False,
        'guidance--pca_guidance--appearance_guidance--tr': args.pca_texture_reg_tr,
        'guidance--pca_guidance--appearance_guidance--reg_factor': args.pca_texture_reg_factor,

        'guidance--cross_attn--end_step': int(args.pca_guidance_steps * args.ddim_steps),
        'guidance--cross_attn--weight': 0,
    }

    input_config = gradio_update_parameter

    base_config = yaml.load(open("config/base.yaml", "r"), Loader=yaml.FullLoader)
    config = merge_sweep_config(base_config=base_config, update=input_config)
    config = OmegaConf.create(config)

    pipeline_name = "SDPipeline"

    pipeline = make_pipeline(pipeline_name,
                             model_path,
                             torch_dtype=torch.float16).to('cuda')
    pipeline.scheduler = CustomDDIMScheduler.from_pretrained(model_path, subfolder="scheduler")

    inversion_config = config.data.inversion

    # if used_approach == "360PanT_F":
    #     img = processor(args.condition_image)
    # elif used_approach == "FreeControl":
    #     img = processor(args.condition_image)
    # else:
    #     raise ValueError(f"Invalid method: {used_approach}")
    img = processor(args.condition_image)
    if control_type == "scribble" or control_type == "canny":
        img = Image.fromarray(255 - np.array(img))
      
    if used_approach == "360PanT_F":
        width_pano, height_pano = img.size  # Note: PIL uses (width, height) order

        left_width = 3 * width_pano // 4
        right_width = width_pano - left_width

        left_half = img.crop((0, 0, left_width, height_pano))
        right_half = img.crop((left_width, 0, width_pano, height_pano))

        extended_image = Image.new('RGB', (width_pano + left_width + right_width, height_pano))
        extended_image.paste(right_half, (0, 0))
        extended_image.paste(img, (right_width, 0))
        extended_image.paste(left_half, (right_width + width_pano, 0))
        img = extended_image
        print("360PanT (F) is selected.")
    elif used_approach == "FreeControl":
        print("FreeControl is selected")
    else:
        raise ValueError(f"Invalid method: {used_approach}")

    condition_image_latents = pipeline.invert(img=img, inversion_config=inversion_config)

    inverted_data = {"condition_input": [condition_image_latents], }

    g = torch.Generator()
    g.manual_seed(config.sd_config.seed)

    img_list = pipeline(prompt=config.sd_config.prompt,
                        negative_prompt=config.sd_config.negative_prompt,
                        num_inference_steps=config.sd_config.steps,
                        generator=g,
                        config=config,
                        inverted_data=inverted_data)[0]

    if control_type != "None":
        img_list.insert(0, img)

    # timestamp = time.strftime("%Y%m%d-%H%M%S")
    # output_folder = os.path.join("output_images", config.sd_config.prompt)
    img_name = os.path.splitext(os.path.basename(args.condition_image))[0]
    img_save_folder_name = used_approach + "-" + img_name + "-" + config.sd_config.prompt
    output_folder = os.path.join("output_images", img_save_folder_name)
    os.makedirs(output_folder, exist_ok=True)

    for idx, image in enumerate(img_list):
        image_path = os.path.join(output_folder, f"output_image_{idx}.png")

        if used_approach == "360PanT_F":
            if control_type != "None" and idx == 0:
                # Save the first image without cropping for this specific case
                image.save(image_path)
            else:
                # Crop all other images in this approach 
                left = 256
                top = 0
                width = 1024
                height = 512
                cropped_img = image.crop((left, top, left + width, top + height))
                cropped_img.save(image_path)
        else:
            # Save the image directly for other approaches
            image.save(image_path)
    print("Images saved as output_image_0.png, output_image_1.png, etc.")


def load_ckpt_pca_list(config_path='config/gradio_info.yaml'):
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist")

    with open(config_path, 'r') as f:
        gradio_config = yaml.safe_load(f)

    models: Dict = gradio_config['checkpoints']
    pca_basis_dict: Dict = dict()

    for model_version in list(models.keys()):
        for model_name in list(models[model_version].keys()):
            if "naive" not in model_name and not os.path.isfile(models[model_version][model_name]["path"]):
                models[model_version].pop(model_name)
            else:
                basis_dict = models[model_version][model_name]["pca_basis"]
                for key in list(basis_dict.keys()):
                    if not os.path.isfile(basis_dict[key]):
                        basis_dict.pop(key)
                if model_version not in pca_basis_dict.keys():
                    pca_basis_dict[model_version]: Dict = dict()
                if model_name not in pca_basis_dict[model_version].keys():
                    pca_basis_dict[model_version][model_name]: Dict = dict()
                pca_basis_dict[model_version][model_name].update(basis_dict)

    return models, pca_basis_dict


def main():
    global model_dict, pca_basis_dict
    model_dict, pca_basis_dict = load_ckpt_pca_list()

    parser = argparse.ArgumentParser(description='FreeControl Image Generation')

    parser.add_argument('--approach', type=str, default='360PanT_F')
    parser.add_argument('--condition_image', type=str, required=True, help='Path to the condition image')
    parser.add_argument('--prompt', type=str, required=True, help='Generation prompt')
    parser.add_argument('--scale', type=float, default=7.5, help='Guidance scale')
    parser.add_argument('--ddim_steps', type=int, default=200, help='DDIM steps')
    parser.add_argument('--sd_version', type=str, required=True, help='Stable Diffusion version')
    parser.add_argument('--model_ckpt', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--pca_guidance_steps', type=float, default=0.6, help='PCA guidance end steps')
    parser.add_argument('--pca_guidance_components', type=int, default=64, help='Number of PCA components')
    parser.add_argument('--pca_guidance_weight', type=float, default=600, help='PCA guidance weight')
    parser.add_argument('--pca_guidance_normalized', type=bool, default=True, help='PCA guidance normalized')
    parser.add_argument('--pca_masked_tr', type=float, default=0.3, help='PCA masked threshold')
    parser.add_argument('--pca_guidance_penalty_factor', type=float, default=10, help='PCA guidance penalty factor')
    parser.add_argument('--pca_warm_up_step', type=float, default=0.05, help='PCA warm up step')
    parser.add_argument('--pca_texture_reg_tr', type=float, default=0.5, help='PCA texture reg threshold')
    parser.add_argument('--pca_texture_reg_factor', type=float, default=0.1, help='PCA texture reg factor')
    parser.add_argument('--negative_prompt', type=str, default="", help='Negative prompt')
    parser.add_argument('--seed', type=int, default=2028, help='Seed')
    parser.add_argument('--paired_objs', type=str, default="(dog; lion)", help='Paired subjects')
    parser.add_argument('--pca_basis', type=str, required=True, help='PCA basis')
    parser.add_argument('--inversion_prompt', type=str, required=True, help='Inversion prompt')
    parser.add_argument('--condition', type=str, default="None", help='Condition type')
    # parser.add_argument('--img_size', type=int, default=512, help='Image size')

    args = parser.parse_args()

    freecontrol_generate(args)


if __name__ == '__main__':
    main()