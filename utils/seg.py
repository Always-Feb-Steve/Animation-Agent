import os
import subprocess
import torch
import cv2
from sam2.build_sam import build_sam2_video_predictor
from contextlib import nullcontext
import shutil


def extract_frames(video_path, output_dir):
    output_pattern = os.path.join(output_dir, "%05d.jpg")

    # Remove the entire directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Create fresh directory
    os.makedirs(output_dir)

    # FFmpeg command
    cmd = ["ffmpeg", "-i", video_path, "-qscale:v", "2", output_pattern]

    # Run the command
    subprocess.run(cmd, check=True)
    print(f"Frames saved to: {output_dir}")


def propagate_mask_through_video(
    video_path,
    first_frame_mask,
    input_threshold=0.85,
    output_threshold=0.9,
    sam2_checkpoint="./checkpoints/sam2_hiera_large.pt",
    model_cfg="configs/sam2/sam2_hiera_l.yaml",
):
    """
    Propagate a mask from the first frame through all frames in a video.

    Args:
        video_path (str): Path to the input video file
        first_frame_mask (torch.Tensor): Tensor containing the first frame mask
        threshold (float, optional): Threshold for mask binarization. Defaults to 0.8.
        sam2_checkpoint (str, optional): Path to the SAM2 model checkpoint. Defaults to "../checkpoints/sam2_hiera_large.pt".
        model_cfg (str, optional): SAM2 model configuration. Defaults to "sam2_hiera_l.yaml".

    Returns:
        torch.Tensor: Tensor of binary masks for each frame
    """
    # Set up device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Configure precision for different devices
    if device.type == "cuda":
        # Turn on tfloat32 for Ampere GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Wrap the computation in autocast context manager if using CUDA
    if device.type == "cuda":
        autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast_context = (
            nullcontext()
        )  # Use a dummy context manager for non-CUDA devices

    with autocast_context:
        # Load the SAM2 predictor
        predictor = build_sam2_video_predictor(
            model_cfg, sam2_checkpoint, device=device
        )

        video_frame_dir = os.path.join(os.path.dirname(video_path), "frames")
        extract_frames(video_path, video_frame_dir)
        # Initialize the SAM2 inference state with the video
        inference_state = predictor.init_state(video_path=video_frame_dir)
        predictor.reset_state(inference_state)

        # Threshold the first frame mask
        mask_tensor = first_frame_mask.float() > input_threshold
        target_size = mask_tensor.shape

        # Add the mask to the inference state for the first frame
        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state, frame_idx=0, obj_id=1, mask=mask_tensor
        )

        # Run propagation throughout the video and collect results
        video_masks = {}  # Dictionary to store masks for all frames
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            for i, out_obj_id in enumerate(out_obj_ids):
                binary_mask = out_mask_logits[i] > output_threshold

                # Store the mask for this frame
                if out_frame_idx not in video_masks:
                    video_masks[out_frame_idx] = {}
                video_masks[out_frame_idx][out_obj_id] = binary_mask

        # Get total number of frames from video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Extract final masks list (assuming a single object)
        all_masks = []
        for frame_idx in range(total_frames):
            if frame_idx in video_masks and 1 in video_masks[frame_idx]:
                mask = video_masks[frame_idx][1]
                if mask.ndim == 3:
                    mask = mask.squeeze(0)
                # Resize mask if needed using torch operations
                if mask.shape != target_size:
                    mask = (
                        torch.nn.functional.interpolate(
                            mask.unsqueeze(0).unsqueeze(0).float(),
                            size=target_size,
                            mode="nearest",
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )
                all_masks.append(mask.float())
            else:
                # If a frame doesn't have a mask, use an empty mask
                all_masks.append(
                    torch.zeros(target_size, dtype=torch.float32, device=device)
                )

    # Stack all masks into a single tensor - now outside the autocast context so will be back to normal dtype
    return torch.stack(all_masks)


def propagate_multi_mask_through_video(
    video_path,
    gt_seg_imggroup_silhouettes,
    input_threshold=0.85,
    output_threshold=0.9,
    sam2_checkpoint="./checkpoints/sam2_hiera_large.pt",
    model_cfg="configs/sam2/sam2_hiera_l.yaml",
):
    """
    Propagate a mask from the first frame through all frames in a video.

    Args:
        video_path (str): Path to the input video file
        first_frame_mask (torch.Tensor): Tensor containing the first frame mask
        threshold (float, optional): Threshold for mask binarization. Defaults to 0.8.
        sam2_checkpoint (str, optional): Path to the SAM2 model checkpoint. Defaults to "../checkpoints/sam2_hiera_large.pt".
        model_cfg (str, optional): SAM2 model configuration. Defaults to "sam2_hiera_l.yaml".

    Returns:
        torch.Tensor: Tensor of binary masks for each frame
    """
    # Set up device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Configure precision for different devices
    if device.type == "cuda":
        # Turn on tfloat32 for Ampere GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Wrap the computation in autocast context manager if using CUDA
    if device.type == "cuda":
        autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast_context = (
            nullcontext()
        )  # Use a dummy context manager for non-CUDA devices

    with autocast_context:
        # Load the SAM2 predictor
        predictor = build_sam2_video_predictor(
            model_cfg, sam2_checkpoint, device=device
        )

        video_frame_dir = os.path.join(os.path.dirname(video_path), "frames")
        extract_frames(video_path, video_frame_dir)
        # Initialize the SAM2 inference state with the video
        inference_state = predictor.init_state(video_path=video_frame_dir)
        predictor.reset_state(inference_state)

        # Threshold the first frame mask
        target_size = gt_seg_imggroup_silhouettes.shape[1:3]
        mask_nums = gt_seg_imggroup_silhouettes.shape[3]

        # add first frame group masks to the inference state
        for i in range(mask_nums):
            mask_tensor = gt_seg_imggroup_silhouettes[:, :, :, i].squeeze(0)
            mask_tensor = mask_tensor.float() > input_threshold
            _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state=inference_state, frame_idx=0, obj_id=i, mask=mask_tensor
            )

        # Run propagation throughout the video and collect results
        video_masks = {}  # Dictionary to store masks for all frames
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            for i, out_obj_id in enumerate(out_obj_ids):
                binary_mask = out_mask_logits[i] > output_threshold

                # Store the mask for this frame
                if out_frame_idx not in video_masks:
                    video_masks[out_frame_idx] = {}
                video_masks[out_frame_idx][out_obj_id] = binary_mask

        # Get total number of frames from video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Extract final masks list (assuming a single object)
        all_masks = []
        for frame_idx in range(total_frames):
            for i in range(mask_nums):
                if frame_idx in video_masks and i in video_masks[frame_idx]:
                    mask = video_masks[frame_idx][i]
                    if mask.ndim == 3:
                        mask = mask.squeeze(0)
                    # Resize mask if needed using torch operations
                    if mask.shape != target_size:
                        mask = (
                            torch.nn.functional.interpolate(
                                mask.unsqueeze(0).unsqueeze(0).float(),
                                size=target_size,
                                mode="nearest",
                            )
                            .squeeze(0)
                            .squeeze(0)
                            .float()
                        )
                    all_masks.append(mask)
                else:
                    # If a frame doesn't have a mask, use an empty mask
                    all_masks.append(
                        torch.zeros(target_size, dtype=torch.float32, device=device)
                    )
    # Stack all masks into a single tensor - now outside the autocast context so will be back to normal dtype
    return torch.stack(all_masks).reshape(total_frames, mask_nums, target_size[0], target_size[1])