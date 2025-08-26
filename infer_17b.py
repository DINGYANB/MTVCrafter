import os
import cv2
import math
import torch
import pickle
import imageio
import torchvision
import numpy as np

from PIL import Image
from einops import rearrange
from decord import VideoReader
from omegaconf import OmegaConf
from torchvision import transforms
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from diffusers import FlowMatchEulerDiscreteScheduler

from models import (MTVCrafterPipeline17B, Transformer17B,
                    AutoencoderKLWan, CLIPModel, WanT5EncoderModel,
                    get_1d_rotary_pos_embed, get_3d_motion_spatial_embed)
from models import SMPL_VQVAE, VectorQuantizer, Encoder, Decoder
from draw_pose import get_control_conditions
from wan_lora import WanLoraWrapper


# inference with dynamic resolutions
ASPECT_RATIO_512 = {
    '0.25': [256.0, 1024.0], '0.26': [256.0, 992.0], '0.27': [256.0, 960.0], '0.28': [256.0, 928.0],
    '0.32': [288.0, 896.0], '0.33': [288.0, 864.0], '0.35': [288.0, 832.0], '0.4': [320.0, 800.0],
    '0.42': [320.0, 768.0], '0.48': [352.0, 736.0], '0.5': [352.0, 704.0], '0.52': [352.0, 672.0],
    '0.57': [384.0, 672.0], '0.6': [384.0, 640.0], '0.68': [416.0, 608.0], '0.72': [416.0, 576.0],
    '0.78': [448.0, 576.0], '0.82': [448.0, 544.0], '0.88': [480.0, 544.0], '0.94': [480.0, 512.0],
    '1.0': [512.0, 512.0], '1.07': [512.0, 480.0], '1.13': [544.0, 480.0], '1.21': [544.0, 448.0],
    '1.29': [576.0, 448.0], '1.38': [576.0, 416.0], '1.46': [608.0, 416.0], '1.67': [640.0, 384.0],
    '1.75': [672.0, 384.0], '2.0': [704.0, 352.0], '2.09': [736.0, 352.0], '2.4': [768.0, 320.0],
    '2.5': [800.0, 320.0], '2.89': [832.0, 288.0], '3.0': [864.0, 288.0], '3.11': [896.0, 288.0],
    '3.62': [928.0, 256.0], '3.75': [960.0, 256.0], '3.88': [992.0, 256.0], '4.0': [1024.0, 256.0]
}


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


def concat_and_save(video_list, save_path, rescale_params, method=1, fps=20):
    processed_videos = []
    for video, rescale in zip(video_list, rescale_params):
        if video is None:
            continue
        video = video.squeeze(0)  # (T, C, H, W)
        if rescale:
            video = (video + 1.0) / 2.0  # [-1,1] -> [0,1]
        video = video.clamp(0, 1)
        processed_videos.append(video)

    if method == 1:
        cat_video = torch.cat(processed_videos, dim=2)
    elif method == 2:
        cat_video = torch.cat(processed_videos, dim=3)
    cat_video = (cat_video * 255).byte().permute(0, 2, 3, 1).cpu().numpy()

    writer = imageio.get_writer(save_path, fps=fps)
    for frame in cat_video:
        writer.append_data(frame)
    writer.close()


def prepare_motion_embeddings(num_frames, num_joints, joints_mean, joints_std, theta=10000, device='cuda'):
    time_embed = get_1d_rotary_pos_embed(44, num_frames, theta, use_real=True)
    time_embed_cos = time_embed[0][:, None, :].expand(-1, num_joints, -1).reshape(num_frames*num_joints, -1)
    time_embed_sin = time_embed[1][:, None, :].expand(-1, num_joints, -1).reshape(num_frames*num_joints, -1)
    spatial_motion_embed = get_3d_motion_spatial_embed(84, num_joints, joints_mean, joints_std, theta)
    spatial_embed_cos = spatial_motion_embed[0][None, :, :].expand(num_frames, -1, -1).reshape(num_frames*num_joints, -1)
    spatial_embed_sin = spatial_motion_embed[1][None, :, :].expand(num_frames, -1, -1).reshape(num_frames*num_joints, -1)
    motion_embed_cos = torch.cat([time_embed_cos, spatial_embed_cos], dim=-1).to(device=device)
    motion_embed_sin = torch.cat([time_embed_sin, spatial_embed_sin], dim=-1).to(device=device)
    return motion_embed_cos, motion_embed_sin


def get_nearest_resize_shape(h, w):
    aspect_ratio = h / w
    ratios = np.array([float(k) for k in ASPECT_RATIO_512.keys()])
    closest_key = str(ratios[np.argmin(np.abs(ratios - aspect_ratio))])
    return ASPECT_RATIO_512[closest_key]


def inference(device, motion_data_path, ref_image_path='', output_dir='inference_output', prompt="The character is dancing."):
    video_length = 49       # Number of frames per clip during training
    overlap = 9             # Number of overlapping frames between adjacent clips during long video inference
    infer_num = 2           # Number of clips to infer for long video generation               
    guidance_scale = 6.0    # Classifier-free guidance scale for the text prompt
    fps = 20                # Frame rate of the saved video
    stride = 1              # Temporal stride for frame sampling
    use_first_clip = False  # Whether to use the first frame of the first clip to compute CLIP embeddings during long video inference
    min_num_frames = (video_length - overlap) * (infer_num - 1) + video_length
    weight_dtype = torch.bfloat16
    seed = 42
    generator = torch.Generator(device=device).manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    video_transforms = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)

    # Load data
    with open(motion_data_path, 'rb') as f:
        data_list = pickle.load(f)
    if not isinstance(data_list, list):
        data_list = [data_list]

    # Load dataset statistics
    global_mean = np.load("data/mean.npy")
    global_std = np.load("data/std.npy")
    selected_indexes = list(range(0, len(data_list)))

    # download the full repo (cached after the first download)
    base_dir = snapshot_download("yanboding/MTVCrafter")
    
    
    lora_path = "wan2.1/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"

    # Load config
    config = OmegaConf.load("wan2.1/wan_civitai.yaml")

    # Get transformer
    transformer3d = Transformer17B.from_pretrained(os.path.join(base_dir, "MV-DiT/Wan-2-1")).to(device, weight_dtype)

    # Get text encoder
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join("wan2.1", config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        torch_dtype=weight_dtype,
    ).to(device, weight_dtype)

    # Get clip image encoder
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join("wan2.1", config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    )

    # Get vae
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join("wan2.1", config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(device, weight_dtype)

    # Get scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join("wan2.1", config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    # Get motion tokenizer
    motion_encoder = Encoder(
        in_channels=3,
        mid_channels=[128, 512],
        out_channels=3072,
        downsample_time=[2, 2],
        downsample_joint=[1, 1]
    )
    motion_quant = VectorQuantizer(nb_code=8192, code_dim=3072, is_train=False)
    motion_decoder = Decoder(
        in_channels=3072,
        mid_channels=[512, 128],
        out_channels=3,
        upsample_rate=2.0,
        frame_upsample_rate=[2.0, 2.0],
        joint_upsample_rate=[1.0, 1.0]
    )
    ckpt_path = os.path.join(base_dir, "4DMoT/mp_rank_00_model_states.pt")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    vqvae = SMPL_VQVAE(motion_encoder, motion_decoder, motion_quant).to(device)
    print(vqvae.load_state_dict(state_dict['module'], strict=True))

    # Load pipeline
    pipeline = MTVCrafterPipeline17B(
        vae=vae, 
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer3d,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
    )
    pipeline = pipeline.to(device)

    # Add lora to enhance details (recommend)
    if lora_path != "":
        lora_wrapper = WanLoraWrapper(pipeline.transformer)
        for lora_path, lora_scale in zip([lora_path], [1]):
            lora_name = lora_wrapper.load_lora(lora_path)
            lora_wrapper.apply_lora(lora_name, lora_scale)

    # Start inference
    for index in selected_indexes:
        if index == 0 and ref_image_path != "":
            output_dir = f"{output_dir}/{ref_image_path.split('/')[-1].replace('.png', '')}"
            os.makedirs(output_dir, exist_ok=True)

        # read video
        data_dict = data_list[index]    
        if data_dict['video_length'] < min_num_frames:
            continue
        input_video_path = data_dict["video_path"]
        vr = VideoReader(input_video_path)

        # save_path
        if ref_image_path == "":
            save_path = os.path.join(output_dir, f"{index}_{guidance_scale}_{os.path.basename(input_video_path)}")
        else:
            ref_name = os.path.splitext(os.path.basename(ref_image_path))[0]
            save_path = os.path.join(output_dir, f"{index}_{guidance_scale}_{ref_name}_{os.path.basename(input_video_path)}")

        with torch.no_grad():
            with torch.autocast("cuda", dtype=weight_dtype):
                start_idx = 0
                previous_sample = None
                total_sample = None
                total_pixel_values = None
                total_recon_control_pixel_values = None
                total_ref_images = None
                for clip_index in range(infer_num):
                    indices = np.linspace(start_idx, start_idx + stride * (video_length - 1), video_length, dtype=int)
                    frames = vr.get_batch(indices).asnumpy()
                    if ref_image_path == "":
                        h, w = frames[0].shape[:2]
                        new_h, new_w = get_nearest_resize_shape(h, w)
                        new_h, new_w = int(new_h), int(new_w)
                    else:
                        ref_image = Image.open(ref_image_path).convert("RGB")
                        ref_w, ref_h = ref_image.size  # PIL uses (width, height)
                        new_h, new_w = get_nearest_resize_shape(ref_h, ref_w)
                        new_h, new_w = int(new_h), int(new_w)
                    frames_resized = np.stack([
                        cv2.resize(frame, (new_w, new_h)) for frame in frames
                    ])
                    pixel_values = torch.from_numpy(frames_resized).permute(0, 3, 1, 2).contiguous().unsqueeze(0) / 255.
                    pixel_values = video_transforms(pixel_values)

                    # ref pixel value and mask pixel value
                    mask = torch.zeros((1, video_length, 1, new_h, new_w), dtype=torch.uint8)
                    mask[:, 1:, :, :, :] = 1
                    if ref_image_path == "":
                        ref_image = pixel_values[:, 0:1, :, :, :]
                        mask_pixel_values = pixel_values * (1 - mask)
                    else:
                        ref_image = Image.open(ref_image_path).convert("RGB")
                        ref_image = ref_image.resize((new_w, new_h), Image.BICUBIC)
                        ref_image = torch.from_numpy(np.array(ref_image)).permute(2, 0, 1).unsqueeze(0).float() / 255.
                        ref_image = ref_image.unsqueeze(1)
                        ref_image = video_transforms(ref_image)
                        mask_pixel_values = ref_image.repeat(1, video_length, 1, 1, 1) * (1 - mask)
                    clip_pixel_values = (ref_image[0][0].permute(1, 2, 0).contiguous() * 0.5 + 0.5) * 255
                    clip_image = Image.fromarray(np.uint8(clip_pixel_values))

                    # smpl poses
                    try:
                        smpl_poses = np.array([pose[0][0].cpu().numpy() for pose in data_dict['pose']['joints3d_nonparam'][:min_num_frames]])
                        poses = smpl_poses[indices]
                    except:
                        poses = data['pose'][sample_indexes]

                    norm_poses = torch.tensor((poses - global_mean) / global_std).unsqueeze(0)
                    motion_tokens, vq_loss = vqvae(norm_poses.to(device), return_vq=True)
                    print(f"vq loss: {vq_loss}")
                    motion_rotary_emb = prepare_motion_embeddings((video_length - 1) // 4 + 1, 24, global_mean, global_std, device=device)
                    recon_motion = vqvae(norm_poses.to(device))[0][0].to(dtype=torch.float32).cpu().detach() * global_std + global_mean
                    recon_control_pixel_values = get_control_conditions(recon_motion, new_h, new_w)
                
                    print(f"{index} --- {clip_index} --- {indices[0]} --- {prompt}")
                    sample = pipeline(
                        prompt = prompt,
                        num_frames = video_length,
                        num_inference_steps = 10 if lora_path != "" else 50,
                        negative_prompt = "bad hands, extra limbs, fused fingers, blurry, low quality",
                        height = new_h,
                        width = new_w,
                        guidance_scale = guidance_scale,
                        generator = generator,
                        video = pixel_values,
                        mask = mask,
                        mask_video = mask_pixel_values,
                        clip_image = first_clip_image if use_first_clip and index != 0 else clip_image,
                        ref_image = ref_image,
                        control_video = None,
                        motion_tokens = motion_tokens,
                        motion_rotary_emb = motion_rotary_emb,
                        previous_sample = previous_sample,
                        lora_path = lora_path,
                        use_first_clip_image = use_first_clip,
                        overlap = overlap
                    ).videos
                    
                    if clip_index == 0:
                        if use_first_clip:
                            first_clip_image = clip_image
                        total_sample = sample[:,:,:-overlap]
                        total_pixel_values = pixel_values[:,:-overlap]
                        total_recon_control_pixel_values = recon_control_pixel_values[:,:-overlap]
                        if infer_num == 1:
                            total_sample = sample
                            total_pixel_values = pixel_values
                            total_recon_control_pixel_values = recon_control_pixel_values
                            total_ref_images = ref_image.repeat(1, video_length, 1, 1, 1)
                        else:
                            total_ref_images = ref_image.repeat(1, video_length-overlap, 1, 1, 1)
                    elif clip_index != infer_num - 1:
                        total_sample = torch.cat([total_sample, sample[:,:,:-overlap]], dim=2)
                        total_pixel_values = torch.cat([total_pixel_values, pixel_values[:,:-overlap]], dim=1)
                        total_recon_control_pixel_values = torch.cat([total_recon_control_pixel_values, recon_control_pixel_values[:,:-overlap]], dim=1)
                        total_ref_images = torch.cat([total_ref_images, previous_sample[:,:,-1:].transpose(1, 2).repeat(1, video_length-overlap, 1, 1, 1)*2-1], dim=1)
                    else:
                        total_sample = torch.cat([total_sample, sample], dim=2)
                        total_pixel_values = torch.cat([total_pixel_values, pixel_values], dim=1)
                        total_recon_control_pixel_values = torch.cat([total_recon_control_pixel_values, recon_control_pixel_values], dim=1)
                        total_ref_images = torch.cat([total_ref_images, previous_sample[:,:,-1:].transpose(1, 2).repeat(1, video_length, 1, 1, 1)*2-1], dim=1)
                        
                    previous_sample = sample[:,:,-overlap:]
                    start_idx = indices[-overlap]

                print(f"save to {save_path}")
                concat_and_save([total_pixel_values, total_ref_images, total_recon_control_pixel_values, total_sample.transpose(1, 2)], save_path, rescale_params=[True, True, True, False], method=2, fps=fps)



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_image_path', type=str, default="", required=False, help="path to the reference character image")
    parser.add_argument('--motion_data_path', type=str, default="data/sampled_data.pkl", required=False, help='path to the motion sequence')
    parser.add_argument('--output_dir', type=str, default="inference_output", required=False, help="where to save the generated video")
    parser.add_argument('--prompt', type=str, default="The character is dancing.", required=False, help="text prompt of the generated video")
    
    args = parser.parse_args()
    inference(device='cuda:0', motion_data_path=args.motion_data_path, ref_image_path=args.ref_image_path, output_dir=args.output_dir, prompt=args.prompt)


if __name__ == '__main__':
    main()
