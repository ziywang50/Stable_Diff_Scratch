import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH//8
LATENTS_HEIGHT = HEIGHT//8

def generate(prompt,uncond_prompt=None,input_image=None,strength=0.8, do_cfg=True, cfg_scale=7.5, sampler_name="ddpm", n_inference_steps=50,
    models={}, seed=None, device=None,idle_device=None, tokenizer=None):
    with torch.no_grad():
        if not 0<strength<=1:
            raise ValueError("strength must be between 0 and 1")
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        clip = models["clip"].to(device)
        if do_cfg:
            #Convert prompts into tokens using tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            #(Batch_siz, Seq_len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            #(Batch_size, Seq_len) -> (Batch_size, Seq_len, dim)
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            #(Batch_size, seq_len) -> (Batch_size, seq_len, dim)
            uncond_context = clip(uncond_tokens)
            context=torch.cat([cond_context, uncond_context])
        else:
            #Convert it to a list of tokens
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            #(1,77, 768)
            context = clip(tokens)
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler name")
        
        latents_shape = (1,4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"].to(device)
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            # (Height, Width, Channel)
            input_image_tensor = np.array(input_image_tensor)
            #height, width, channel
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            input_image_tensor = rescale(input_image_tensor, (0,255), (-1,1))
            #(Height, Width, Channel) -> (Batch_size, height, width, channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0,3,1,2)
            encoder_noise = torch.randn(latents_shape, generator = generator, device=device)
            #run the image through the encoder of VAE
            latents = encoder(input_image_tensor, encoder_noise)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            to_idle(encoder)
        else:
            #If text to image, start with some random noise
            latents = torch.randn(latents_shape, generator = generator, device=device)
        diffusion = models["diffusion"].to(device)
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            #(1,320)
            time_embedding = get_time_embedding(timestep).to(device)
            model_input = latents
            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2,1,1,1)
            model_output = diffusion(model_input, context, time_embedding)
            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            latents = sampler.step(timestep, latents, model_output)
        to_idle(diffusion)
        decoder = models["decoder"]
        decoder.to(device)
        images = decoder(latents)
        to_idle(decoder)
        images = rescale(images, (-1,1), (0,255), clamp=True)
        #(Batch_Size, Channel, Height, Width) -> (Batch_size, height, width, channel)
        images = images.permute(0,2,3,1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
def rescale(x: torch.Tensor, from_range: tuple, to_range: tuple, clamp=False) -> torch.Tensor:
    from_min, from_max = from_range
    to_min, to_max = to_range
    x -= from_min
    x *= (to_max - to_min) / (from_max - from_min)
    x += to_min
    if clamp:
        x = x.clamp(to_min, to_max)
    return x

def get_time_embedding(timestep):
    #(160, )
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32)/160)
    #(1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    #Shape: (1, 160*2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
