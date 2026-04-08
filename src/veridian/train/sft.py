from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from veridian.config import load_data_config, load_model_config, load_training_config
from veridian.data import PackedSFTDataset
from veridian.modeling import VeridianForCausalLM
from veridian.train.checkpointing import load_checkpoint, save_checkpoint
from veridian.train.optim import build_optimizer, build_scheduler

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT VERIDIAN on Russian-heavy chat data.")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--tokenizer-path", default="artifacts/tokenizer/tokenizer.json")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    model_config = load_model_config(args.model_config)
    train_config = load_training_config(args.train_config)

    set_seed(train_config.seed)
    device = torch.device(train_config.device)
    model = VeridianForCausalLM(model_config).to(device)
    model.set_gradient_checkpointing(train_config.activation_checkpointing)

    optimizer = build_optimizer(model, train_config)
    scheduler = build_scheduler(optimizer, train_config)

    start_step = 0
    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location="cpu",
        )
        start_step = int(checkpoint.get("step", 0))

    dataset = PackedSFTDataset(
        data_config=load_data_config(args.data_config),
        tokenizer_path=args.tokenizer_path,
        seed=train_config.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.micro_batch_size_sequences,
        num_workers=0,
    )
    data_iter = iter(dataloader)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step, train_config.num_train_steps):
        running_loss = 0.0

        for _ in range(train_config.gradient_accumulation_steps):
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = output.loss / train_config.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach().cpu())

        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step % train_config.log_every == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"step={step} loss={running_loss:.4f} lr={current_lr:.6e}",
                flush=True,
            )

        if step > 0 and step % train_config.save_every == 0:
            ckpt_path = Path(train_config.checkpoint_dir) / f"step_{step:07d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, scheduler, step)


if __name__ == "__main__":
    main()
