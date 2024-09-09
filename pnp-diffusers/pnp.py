import glob
import os
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
from PIL import Image
import yaml
from tqdm import tqdm
from transformers import logging
from diffusers import DDIMScheduler, StableDiffusionPipeline

from pnp_utils import *

# suppress partial model loading warning
logging.set_verbosity_error()

class PNP(nn.Module):
    def __init__(self, config, method):
        super().__init__()
        self.config = config
        self.device = config["device"]
        sd_version = config["sd_version"]
        self.method = method

        if sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {sd_version} not supported.')

        # Create SD models
        print('Loading SD model')

        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float16).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(config["n_timesteps"], device=self.device)
        print('SD model loaded')

        # load image
        # self.image, self.eps = self.get_data()
        self.eps = self.get_data()

        self.text_embeds = self.get_text_embeds(config["prompt"], config["negative_prompt"])
        self.pnp_guidance_embeds = self.get_text_embeds("", "").chunk(2)[0]


    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def decode_latent(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
            if self.method == "360PanT":
                img = img[:, :, :, 256:1280]
        return img

    # @torch.autocast(device_type='cuda', dtype=torch.float32)
    # def get_data(self):
    #     # load image
    #     image = Image.open(self.config["image_path"]).convert('RGB') 
    #     # image = image.resize((512, 512), resample=Image.Resampling.LANCZOS)
    #     image = T.ToTensor()(image).to(self.device)
    #     # get noise
    #     latents_path = os.path.join(self.config["latents_path"], os.path.splitext(os.path.basename(self.config["image_path"]))[0], f'noisy_latents_{self.scheduler.timesteps[0]}.pt')
    #     noisy_latent = torch.load(latents_path).to(self.device)
    #     return image, noisy_latent

    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def get_data(self):
        # get noise
        latents_path = os.path.join(self.config["latents_path"], os.path.splitext(os.path.basename(self.config["image_path"]))[0], f'noisy_latents_{self.scheduler.timesteps[0]}.pt')
        noisy_latent = torch.load(latents_path).to(self.device)
        return noisy_latent

    def get_views(self, panorama_height, panorama_width, window_size=[64,128], stride=16):
        panorama_height /= 8
        panorama_width /= 8
        num_blocks_height = (panorama_height - window_size[0]) // stride + 1
        num_blocks_width = (panorama_width - window_size[1]) // stride + 1
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        views = []
        for i in range(total_num_blocks):
            h_start = int((i // num_blocks_width) * stride)
            h_end = h_start + window_size[0]
            w_start = int((i % num_blocks_width) * stride)
            w_end = w_start + window_size[1]
            views.append((h_start, h_end, w_start, w_end))
        return views

    @torch.no_grad()
    def denoise_step(self, x, t):
        # register the time step and features in pnp injection modules
        source_latents = load_source_latents_t(t, os.path.join(self.config["latents_path"], os.path.splitext(os.path.basename(self.config["image_path"]))[0]))

        if self.method == "PnP":
            latent_model_input = torch.cat([source_latents] + ([x] * 2))

            register_time(self, t.item())

            # compute text embeddings
            text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds], dim=0)

            # apply the denoising network
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

            # perform guidance
            _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
            noise_pred = noise_pred_uncond + self.config["guidance_scale"] * (noise_pred_cond - noise_pred_uncond)

            # compute the denoising step with the reference model
            denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']

        elif self.method == "360PanT":
        
            count_t = torch.zeros_like(x)
            value_t = torch.zeros_like(x)
            views_t = self.get_views(512, 2048)

            # initialize the values of latent_view_t and source_latents_view_t
            latent_view_t = x[:, :, :, 64:192]
            source_latents_view_t = source_latents[:, :, :, 64:192]

            #### pre-denoising operations twice on the stitch block ####
            for ii_md in range(2):
                latent_view_t[:, :, :, 0:64] = x[:, :, :, 192:256] #left part of the stitch block
                latent_view_t[:, :, :, 64:128] = x[:, :, :, 0:64] #right part of the stitch block

                source_latents_view_t[:, :, :, 0:64] = source_latents[:, :, :, 192:256] #left part of the stitch block
                source_latents_view_t[:, :, :, 64:128] = source_latents[:, :, :, 0:64] #right part of the stitch block


                latent_model_input = torch.cat([source_latents_view_t] + ([latent_view_t] * 2))


                register_time(self, t.item())

                # compute text embeddings
                text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds], dim=0)
                text_embed_input_inv = torch.zeros_like(text_embed_input)

                # apply the denoising network
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

                # perform guidance
                _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
                noise_pred = noise_pred_uncond + self.config["guidance_scale"] * (noise_pred_cond - noise_pred_uncond)

                # compute the denoising step with the reference model
                denoised_latent_view = self.scheduler.step(noise_pred, t, latent_view_t)['prev_sample']

                value_t[:, :, :, 192:256] += denoised_latent_view[:, :, :, 0:64]
                count_t[:, :, :, 192:256] += 1

                value_t[:, :, :, 0:64] += denoised_latent_view[:, :, :, 64:128]
                count_t[:, :, :, 0:64] += 1


            for h_start, h_end, w_start, w_end in views_t:
                source_latents_view = source_latents[:, :, h_start:h_end, w_start:w_end]
                x_view = x[:, :, h_start:h_end, w_start:w_end]

                latent_model_input = torch.cat([source_latents_view] + ([x_view] * 2))


                register_time(self, t.item())

                # compute text embeddings
                text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds], dim=0)
                text_embed_input_inv = torch.zeros_like(text_embed_input)

                # apply the denoising network
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

                # perform guidance
                _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
                noise_pred = noise_pred_uncond + self.config["guidance_scale"] * (noise_pred_cond - noise_pred_uncond)

                # compute the denoising step with the reference model
                denoised_latent_view = self.scheduler.step(noise_pred, t, x_view)['prev_sample']

                value_t[:, :, h_start:h_end, w_start:w_end] += denoised_latent_view
                count_t[:, :, h_start:h_end, w_start:w_end] += 1

            denoised_latent = torch.where(count_t > 0, value_t / count_t, value_t)

        else:
            raise ValueError(f"Invalid method: {self.method}")

        return denoised_latent

    def init_pnp(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control_efficient(self, self.qk_injection_timesteps)
        register_conv_control_efficient(self, self.conv_injection_timesteps)

    def run_pnp(self):
        pnp_f_t = int(self.config["n_timesteps"] * self.config["pnp_f_t"])
        pnp_attn_t = int(self.config["n_timesteps"] * self.config["pnp_attn_t"])
        self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        edited_img = self.sample_loop(self.eps)


    def sample_loop(self, x):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
                x = self.denoise_step(x, t)

            decoded_latent = self.decode_latent(x)
            T.ToPILImage()(decoded_latent[0]).save(f'{self.config["output_path"]}/output-{self.config["prompt"]}.png')
                
        return decoded_latent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='360PanT')
    parser.add_argument('--config_path', type=str, default='/content/360PanT/pnp-diffusers/config_pnp.yaml')
    opt = parser.parse_args()
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)
    os.makedirs(config["output_path"], exist_ok=True)
    with open(os.path.join(config["output_path"], "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    seed_everything(config["seed"])
    print(config)
    pnp = PNP(config, opt.method)
    pnp.run_pnp()