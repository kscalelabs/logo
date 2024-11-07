"""Renders a walk through latent space a large version of the logo.

To convert to a GIF:

ffmpeg -i logo.mp4 -vf palettegen logo.png
ffmpeg -framerate 30 -i logo.mp4 -i logo.png -lavfi paletteuse logo.gif
"""

import argparse

import cv2
import torch
import tqdm

from logo.train import Task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path")
    parser.add_argument("-p", "--pixels", type=int, default=1024)
    parser.add_argument("-o", "--output-path", type=str, default="logo.mp4")
    parser.add_argument("-n", "--num-chunks", type=int, default=128)
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--num-waypoints", type=int, default=12)
    args = parser.parse_args()

    task = Task.get_task_from_ckpt(args.ckpt_path)
    task.device_manager.module_to(task)
    device = task.device_manager.device

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(args.output_path, fourcc, args.fps, (args.pixels, args.pixels))

    # Generate waypoints for the walk
    waypoints = [torch.randn(1, task.config.kl_latent_dims, device=device)]
    for _ in range(args.num_waypoints - 1):
        next_point = torch.randn_like(waypoints[-1])
        waypoints.append(next_point)

    # Add the first point again to complete the loop
    waypoints.append(waypoints[0])

    with task.device_manager.autocast_context():
        for frame in tqdm.tqdm(range(args.num_frames)):
            # Calculate which segment we're in and local interpolation factor
            segment_length = args.num_frames / args.num_waypoints
            segment_idx = int(frame / segment_length)
            local_t = (frame / segment_length) - segment_idx

            # Interpolate between current segment's start and end points
            current_eps = waypoints[segment_idx] * (1 - local_t) + waypoints[segment_idx + 1] * local_t

            # Render frame
            arr = task.render(device, args.pixels, args.num_chunks, current_eps, use_tqdm=False)

            # Convert to video frame format (BGR)
            frame_rgb = (arr * 255).round().to(torch.uint8).cpu().numpy()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Write frame
            video.write(frame_bgr)

    video.release()


if __name__ == "__main__":
    main()
