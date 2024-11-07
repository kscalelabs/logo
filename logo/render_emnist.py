"""Renders a large version of the logo."""

import argparse
import functools
from pathlib import Path

import PIL.Image as PILImage
import torch
import tqdm

from logo.train_emnist import CLASSES, Task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path")
    parser.add_argument("-p", "--pixels", type=int, default=512)
    parser.add_argument("-o", "--output-path", type=str, default="logo.png")
    parser.add_argument("--sampling-timesteps", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=1)
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    task = Task.get_task_from_ckpt(args.ckpt_path)
    task.device_manager.module_to(task)
    with task.device_manager.autocast_context(), torch.inference_mode():
        for i in tqdm.tqdm(range(args.num_samples)):
            shape, device = (1, args.pixels, args.pixels), task.device_manager.device
            # label = torch.randint(0, 26, (1,), device=device)
            label_id = CLASSES.index("k")
            label = torch.tensor([label_id], device=device)
            images_hw = task.diff.sample(
                model=functools.partial(task.diffusion_step, labels=label),
                shape=shape,
                device=device,
                num_steps=args.sampling_timesteps,
            )[0, 0]

            images_hw = images_hw.clamp(0, 1)

            img = PILImage.fromarray((images_hw * 255).round().to(torch.uint8).cpu().numpy(), mode="L")
            output_path_i = output_path.with_name(f"{output_path.stem}_{i}{output_path.suffix}")
            img.save(output_path_i)


if __name__ == "__main__":
    main()
