"""Independent native tensor pipeline: one conditional prediction and Euler update."""

import logging

import torch
from diffusers import LTX2ConditionPipeline
from diffusers.video_processor import VideoProcessor
from comfy import utils as comfy_utils

from .utils import (canvas_geometry, compatible_frames, decoded_to_images,
                    images_to_video, seed_generator)

logger = logging.getLogger(__name__)


def one_step(transformer, latent, context, mask, sigma, fps, seed, progress):
    noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype,
                        generator=seed_generator(seed, latent.device))
    noisy = ((1 - sigma) * latent.float() + sigma * noise.float()).to(torch.bfloat16)
    del noise
    velocity = transformer(noisy, context, mask, sigma, fps, progress)
    # FlowMatch Euler: x_next = x + (sigma_next - sigma) * velocity.
    # The validated unshifted one-step schedule ends at sigma_next=0.
    return noisy.float() - sigma * velocity.float()


def decode_video(decoder, latent, mode, tiles, seed, recover):
    if mode == "tiled":
        decoder.enable_tiling(**tiles)
    else:
        decoder.disable_tiling()
    logger.info("[mAI SoL H3 Refiner] Decoding %s.", "tiled" if mode == "tiled" else "untiled")
    try:
        return decoder.decode(latent, generator=seed_generator(seed, latent.device), num_inference_steps=1).sample
    except torch.cuda.OutOfMemoryError:
        if mode != "auto":
            raise
    # Leave the except block first: its traceback owns failed decode tensors.
    logger.info("[mAI SoL H3 Refiner] Untiled decoder ran out of VRAM. Retrying with tiled decoding.")
    recover()
    decoder.enable_tiling(**tiles)
    # Restart decoder noise; the failed attempt must not advance this seed.
    return decoder.decode(latent, generator=seed_generator(seed, latent.device), num_inference_steps=1).sample


def run_pipeline(runtime, images, prompt, fps, width, height, sigma, seed, decoder_seed, mode, tiles):
    device = runtime.device
    source_frames = images.shape[0]
    cw, ch = canvas_geometry(width, height)
    frames = compatible_frames(source_frames, runtime.minimum_frames)
    if frames != source_frames:
        logger.info("[mAI SoL H3 Refiner] Input: %d frames. Padding internally to %d frames for LTX compatibility.", source_frames, frames)
    logger.info("[mAI SoL H3 Refiner] Target: %dx%d. Internal canvas: %dx%d. Sigma: %.10f", width, height, cw, ch, sigma)
    progress = comfy_utils.ProgressBar(100)

    text = runtime.activate("text_encoder", 3 * 1024**3)
    tokens = runtime.tokenizer([prompt.strip()], padding="max_length", max_length=1024,
                               truncation=True, add_special_tokens=True, return_tensors="pt").to(device)
    encoded = text(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask,
                   output_hidden_states=True, use_cache=False)
    # This is Diffusers' LTX2ConditionPipeline hidden-state packing, including
    # the embedding layer, as shipped with the Gemma4Unified H3 text component.
    embeddings = runtime.park(torch.stack(encoded.hidden_states, dim=-1).flatten(2, 3))
    mask = runtime.park(tokens.attention_mask)
    del encoded, tokens, text
    progress.update_absolute(10)

    connectors = runtime.activate("connectors", 3 * 1024**3)
    context, _, mask = connectors(embeddings.to(device), mask.to(device), padding_side="left")
    context, mask = runtime.park(context), runtime.park(mask)
    del embeddings, connectors
    progress.update_absolute(20)

    vae = runtime.activate("vae", 8 * 1024**3)
    processor = VideoProcessor(vae_scale_factor=32)
    pixels = processor.preprocess_video(images_to_video(images, frames), height=ch // 2, width=cw // 2)
    vae.enable_tiling()
    latent = runtime.park(vae.encode(pixels.to(device=device, dtype=torch.bfloat16)).latent_dist.mode())
    # Statistics must survive VAE offload without holding its CUDA storage.
    mean, std, scale = vae.latents_mean.detach().cpu().clone(), vae.latents_std.detach().cpu().clone(), vae.config.scaling_factor
    del pixels, vae
    progress.update_absolute(35)

    upsampler = runtime.activate("latent_upsampler", 4 * 1024**3)
    latent = upsampler(latent.to(device))
    latent = runtime.park(LTX2ConditionPipeline._normalize_latents(latent, mean, std, scale))
    del upsampler
    progress.update_absolute(45)

    transformer = runtime.activate("transformer", 8 * 1024**3)
    logger.info("[mAI SoL H3 Refiner] Running one-step refinement.")
    latent = runtime.park(one_step(transformer, latent.to(device), context.to(device), mask.to(device),
                                   sigma, fps, seed, lambda done, total: progress.update_absolute(45 + 35 * done // total)))
    del context, mask, transformer
    progress.update_absolute(80)

    decoder = runtime.activate("diffusion_decoder", 16 * 1024**3)
    latent = LTX2ConditionPipeline._denormalize_latents(latent, mean, std, scale).to(device=device, dtype=torch.bfloat16)
    decoded = decode_video(decoder, latent, mode, tiles, decoder_seed, runtime.recover_decode_oom)
    result = decoded_to_images(decoded, source_frames, width, height)
    progress.update_absolute(100)
    logger.info("[mAI SoL H3 Refiner] Refinement complete: %d frames.", result.shape[0])
    return result
