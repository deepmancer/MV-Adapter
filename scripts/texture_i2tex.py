import argparse
import faulthandler
import gc
import logging
import os
import random
import signal
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import bpy

from mvadapter.pipelines.pipeline_texture import ModProcessConfig, TexturePipeline
from mvadapter.utils import make_image_grid

# Enable faulthandler to get better stack traces on segfaults
faulthandler.enable()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger("texture_i2tex")

# Set up signal handler for debugging
def signal_handler(signum, frame):
    logger.error(f"Received signal {signum}")
    logger.error("Stack trace:")
    traceback.print_stack(frame)
    sys.exit(1)

signal.signal(signal.SIGTERM, signal_handler)


def _clear_blender_scene():
    """Clear all objects and orphaned data from Blender scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


def _export_blender_mesh(output_path: str):
    """Export current Blender scene to GLB format."""
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',
        export_texcoords=True,
        export_normals=True,
        export_materials='EXPORT',
        export_colors=True,
        export_cameras=False,
        export_lights=False,
        export_apply=True,
        export_yup=False,
        export_draco_mesh_compression_enable=False
    )


def align_input_mesh_transform(input_mesh_path: str, output_mesh_path: str):
    """Apply axis transformation to input mesh."""
    _clear_blender_scene()
    
    bpy.ops.import_scene.gltf(filepath=input_mesh_path)
    imported_objects = bpy.context.selected_objects

    for obj in imported_objects:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    _export_blender_mesh(output_mesh_path)
    _clear_blender_scene()


def align_output_mesh_transform(input_mesh_path: str, output_mesh_path: str):
    """Apply axis revert transformation to output mesh (negates Y and Z axes)."""
    _clear_blender_scene()
    
    bpy.ops.import_scene.gltf(filepath=input_mesh_path)
    imported_objects = bpy.context.selected_objects

    for obj in imported_objects:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            obj.scale.y *= -1
            obj.scale.z *= -1
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    _export_blender_mesh(output_mesh_path)
    _clear_blender_scene()


def _setup_pipeline_variant(variant: str, sdxl_model_id: str):
    """Setup pipeline configuration based on variant."""
    if variant == "sdxl":
        from .inference_ig2mv_sdxl import prepare_pipeline, remove_bg as remove_bg_fn_impl, run_pipeline
        return {
            'prepare_pipeline': prepare_pipeline,
            'remove_bg_fn_impl': remove_bg_fn_impl,
            'run_pipeline': run_pipeline,
            'base_model': sdxl_model_id,
            'vae_model': "madebyollin/sdxl-vae-fp16-fix",
            'height': 768,
            'width': 768,
            'uv_size': 4096
        }
    elif variant == "sd21":
        from .inference_ig2mv_sd import prepare_pipeline, remove_bg as remove_bg_fn_impl, run_pipeline
        return {
            'prepare_pipeline': prepare_pipeline,
            'remove_bg_fn_impl': remove_bg_fn_impl,
            'run_pipeline': run_pipeline,
            'base_model': "stabilityai/stable-diffusion-2-1-base",
            'vae_model': None,
            'height': 512,
            'width': 512,
            'uv_size': 2048
        }
    else:
        raise ValueError(f"Invalid variant: {variant}")


def _setup_background_remover(device: str):
    """Setup BiRefNet model for background removal."""
    birefnet = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    )
    birefnet.to(device)
    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return birefnet, transform_image


def _get_mesh_path(data_dir: Path, shape_provider: str, sample_id: str) -> Optional[Path]:
    """Find mesh file for given sample."""
    mesh_names = ["textured_mesh.glb", "shape_mesh.glb"]
    for name in mesh_names:
        mesh_path = data_dir / shape_provider / sample_id / name
        if mesh_path.exists():
            return mesh_path
    return None


def _load_text_prompt(sample_id: str, prompt_dir: Optional[str], default_text: str) -> str:
    """Load custom prompt or return default."""
    if prompt_dir is None:
        return default_text
    
    prompt_path = Path(prompt_dir) / f"{sample_id}.txt"
    if not prompt_path.exists():
        return default_text
    
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_content = f.read().strip()
        return prompt_content if prompt_content else default_text
    except Exception as e:
        print(f"Warning: Could not read prompt file {prompt_path}: {e}")
        return default_text


def _handle_mesh_alignment(mesh_path: Path, sample_id: str, align_input: bool) -> tuple[str, Optional[str]]:
    """Handle input mesh alignment if requested."""
    if not align_input:
        return str(mesh_path), None
    
    print("Applying input mesh alignment transformation...")
    temp_aligned_path = os.path.join(tempfile.gettempdir(), f"{sample_id}_aligned_input.glb")
    
    try:
        align_input_mesh_transform(str(mesh_path), temp_aligned_path)
        print("Input mesh aligned and saved to temporary location")
        return temp_aligned_path, temp_aligned_path
    except Exception as e:
        print(f"Warning: Failed to align input mesh: {e}")
        print("Proceeding with original mesh...")
        return str(mesh_path), None


def _cleanup_temp_file(file_path: Optional[str]):
    """Clean up temporary file if it exists."""
    if file_path is not None and os.path.exists(file_path):
        try:
            os.remove(file_path)
            print("Cleaned up temporary file")
        except Exception as e:
            print(f"Warning: Failed to clean up temporary file: {e}")


def _apply_output_alignment(output_path: str) -> bool:
    """Apply alignment transformation to output mesh."""
    print("Applying output mesh alignment transformation...")
    temp_output_path = output_path.replace(".glb", "_temp.glb")
    
    try:
        os.rename(output_path, temp_output_path)
        align_output_mesh_transform(temp_output_path, output_path)
        os.remove(temp_output_path)
        print(f"Output mesh aligned and saved to {output_path}")
        return True
    except Exception as e:
        print(f"Warning: Failed to align output mesh: {e}")
        if os.path.exists(temp_output_path):
            os.rename(temp_output_path, output_path)
        print("Keeping original unaligned output mesh...")
        return False


def inference_i2gtex(
    data_dir: str,
    output_dir: str,
    prompt_dir: Optional[str] = None,
    default_text: str = "high quality photo, photograph of a person",
    shape_provider: str = "hunyuan",
    variant: str = "sdxl",
    sdxl_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    device: str = "cuda",
    seed: int = -1,
    reference_conditioning_scale: float = 1.0,
    preprocess_mesh: bool = False,
    remove_bg: bool = False,
    align_input_mesh: bool = False,
    align_output_mesh: bool = False,
):
    num_views = 6
    config = _setup_pipeline_variant(variant, sdxl_model_id)

    print("Preparing pipelines...")
    pipe = config['prepare_pipeline'](
        base_model=config['base_model'],
        vae_model=config['vae_model'],
        unet_model=None,
        lora_model=None,
        adapter_path="huanngzh/mv-adapter",
        scheduler=None,
        num_views=num_views,
        device=device,
        dtype=torch.float16,
    )
    
    remove_bg_fn = None
    if remove_bg:
        birefnet, transform_image = _setup_background_remover(device)
        remove_bg_fn = lambda x: config['remove_bg_fn_impl'](x, birefnet, transform_image, device)

    texture_pipe = TexturePipeline(
        upscaler_ckpt_path="./checkpoints/RealESRGAN_x2plus.pth",
        inpaint_ckpt_path="./checkpoints/big-lama.pt",
        device=device,
        context_type="gl",  # Use GL context for better stability with pymeshlab/open3d
    )
    print("Pipelines ready.")

    image_dir_path = Path(data_dir) / "matted_image_centered"
    sample_ids = [f.stem for f in image_dir_path.glob("*.png")]
    
    if not sample_ids:
        print(f"No images found in {image_dir_path}, exiting...")
        return
    
    print(f"Found {len(sample_ids)} samples to process")
    random.shuffle(sample_ids)

    successful_samples = []
    failed_samples = []
    
    for idx, sample_id in enumerate(sample_ids, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing sample {idx}/{len(sample_ids)}: {sample_id}")
        logger.info(f"{'='*60}")
        
        image_path = image_dir_path / f"{sample_id}.png"
        mesh_path = _get_mesh_path(Path(data_dir), shape_provider, sample_id)

        if not image_path.exists():
            logger.warning(f"Image not found at {image_path}, skipping...")
            continue
        if mesh_path is None:
            logger.warning(f"Mesh not found for sample {sample_id}, skipping...")
            continue
        
        mesh_path_to_use, temp_aligned_mesh_path = _handle_mesh_alignment(
            mesh_path, sample_id, align_input_mesh
        )
        
        text_prompt = _load_text_prompt(sample_id, prompt_dir, default_text)
        if prompt_dir is not None and Path(prompt_dir) / f"{sample_id}.txt":
            logger.info(f"Using prompt: {text_prompt[:50]}..." if len(text_prompt) > 50 else f"Using prompt: {text_prompt}")
        
        sample_output_dir = Path(output_dir) / shape_provider / sample_id
        os.makedirs(sample_output_dir, exist_ok=True)

        try:
            logger.info("Step 1/3: Generating multi-view images...")
            images, _, _, _ = config['run_pipeline'](
                pipe,
                mesh_path=mesh_path_to_use,
                num_views=num_views,
                text=text_prompt,
                image=str(image_path),
                height=config['height'],
                width=config['width'],
                num_inference_steps=50,
                guidance_scale=3.0,
                seed=seed,
                reference_conditioning_scale=reference_conditioning_scale,
                negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
                device=device,
                remove_bg_fn=remove_bg_fn,
            )
            
            mv_path = sample_output_dir / "multiview.png"
            make_image_grid(images, rows=1).save(str(mv_path))
            logger.info(f"Multi-view image saved to {mv_path}")

            # Force cleanup before texture generation
            logger.info("Cleaning up GPU memory...")
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            logger.info("Step 2/3: Generating textured mesh...")
            logger.info(f"  Input mesh: {mesh_path_to_use}")
            logger.info(f"  UV size: {config['uv_size']}")
            logger.info(f"  Preprocess mesh: {preprocess_mesh}")
            
            out = texture_pipe(
                mesh_path=mesh_path_to_use,
                save_dir=str(sample_output_dir),
                save_name="textured_mesh",
                uv_unwarp=True,
                preprocess_mesh=preprocess_mesh,
                uv_size=config['uv_size'],
                rgb_path=str(mv_path),
                rgb_process_config=ModProcessConfig(view_upscale=True, inpaint_mode="view"),
                camera_azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
            )
            logger.info(f"Step 3/3: Textured mesh saved to {out.shaded_model_save_path}")
            
            if align_output_mesh and out.shaded_model_save_path is not None:
                _apply_output_alignment(out.shaded_model_save_path)
            
            _cleanup_temp_file(temp_aligned_mesh_path)
            successful_samples.append(sample_id)
            logger.info(f"Sample {sample_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing sample {sample_id}: {e}")
            logger.error("Full traceback:")
            traceback.print_exc()
            _cleanup_temp_file(temp_aligned_mesh_path)
            failed_samples.append(sample_id)
            
            # Cleanup after error
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples: {len(sample_ids)}")
    logger.info(f"Successful: {len(successful_samples)}")
    logger.info(f"Failed: {len(failed_samples)}")
    
    if failed_samples:
        logger.info(f"\nFailed samples:")
        for sample_id in failed_samples:
            logger.info(f"  - {sample_id}")
    
    logger.info(f"\nAll outputs saved to {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image-to-texture inference for multiple samples")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    parser.add_argument("--variant", type=str, default="sdxl", choices=["sdxl", "sd21"], 
                        help="Model variant to use")
    # I/O
    parser.add_argument("--data_dir", type=str, default="/workspace/outputs_new/")
    parser.add_argument("--prompt_dir", type=str, default="/workspace/outputs_new/prompt/general")
    parser.add_argument("--default_text", type=str, default="high quality photo, photograph of a person")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default="/workspace/outputs_new/mvadapter/")
    parser.add_argument("--shape_provider", type=str, default="hunyuan", choices=["hunyuan", "hi3dgen"])
    parser.add_argument(
        "--sdxl_model_id", type=str, default="SG161222/RealVisXL_V4.0",
        choices=["stabilityai/stable-diffusion-xl-base-1.0", "SG161222/RealVisXL_V4.0"],
    )
    # Extra
    parser.add_argument("--reference_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--preprocess_mesh", default=True, action="store_true",
                        help="Whether to preprocess the mesh (Warning: may cause segmentation faults with some meshes)")
    parser.add_argument("--remove_bg", action="store_true",
                        help="Whether to remove background from images")
    parser.add_argument("--align_input_mesh", action="store_true",
                        help="Whether to align input mesh to upright position, should be True when using hunyuan shapes")
    parser.add_argument("--align_output_mesh", default=True, action="store_true",
                        help="Whether to align output mesh by negating Y and Z axes")
    args = parser.parse_args()
    
    inference_i2gtex(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        prompt_dir=args.prompt_dir,
        default_text=args.default_text,
        shape_provider=args.shape_provider,
        sdxl_model_id=args.sdxl_model_id,
        variant=args.variant,
        device=args.device,
        seed=args.seed,
        reference_conditioning_scale=args.reference_conditioning_scale,
        preprocess_mesh=args.preprocess_mesh,
        remove_bg=args.remove_bg,
        align_input_mesh=args.align_input_mesh,
        align_output_mesh=args.align_output_mesh,
    )