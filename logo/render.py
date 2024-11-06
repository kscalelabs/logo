"""Renders a large version of the logo."""

import argparse

import PIL.Image as PILImage
import torch

from logo.train import Task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path")
    parser.add_argument("-p", "--pixels", type=int, default=4096)
    parser.add_argument("-o", "--output-path", type=str, default="logo.png")
    parser.add_argument("-n", "--num-chunks", type=int, default=1024)
    args = parser.parse_args()

    task = Task.get_task_from_ckpt(args.ckpt_path)
    task.device_manager.module_to(task)
    with task.device_manager.autocast_context():
        arr = task.render(task.device_manager.device, args.pixels, args.num_chunks)
    img = PILImage.fromarray((arr * 255).round().to(torch.uint8).cpu().numpy())
    img.save(args.output_path)


if __name__ == "__main__":
    main()
