import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from transformers import AutoModel, AutoImageProcessor, get_cosine_schedule_with_warmup
import numpy as np
import os
import argparse
import joblib
import warnings
# Add this line to ignore the specific scikit-learn warning
warnings.filterwarnings("ignore", message="X does not have valid feature names, but MinMaxScaler was fitted with feature names")
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random

# --- IMPROVED: Safer data loading function ---
def safe_collate(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

class CarlaDataset(Dataset):
    # --- BUG FIX: __init__ now accepts the scaler object ---
    # The dataset now works with RAW, un-scaled labels and will apply scaling internally.
    def __init__(self, metadata_df, scaler, image_processor=None, augment_flipping=False):
        self.metadata = metadata_df
        self.scaler = scaler  # The fitted scaler is passed in
        self.image_paths = self.metadata['image_path'].values
        self.labels = self.metadata[['speed', 'steering']].values  # These are now raw values
        self.image_processor = image_processor
        self.augment_flipping = augment_flipping

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            # 1. Get RAW speed and steering
            speed, steering = self.labels[idx][0], self.labels[idx][1]

            # 2. Apply augmentation (flipping) on the RAW steering value
            if self.augment_flipping and random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                steering = -steering

            # 3. Scale the labels *after* potential augmentation
            # The scaler expects a 2D array, so we reshape and then select the first (only) row.
            scaled_labels = self.scaler.transform(np.array([[speed, steering]]))[0]
            label = torch.tensor(scaled_labels, dtype=torch.float32)

        except (FileNotFoundError, OSError):
            return None, None

        if self.image_processor:
            processed = self.image_processor(images=image, return_tensors="pt")
            image = processed['pixel_values'][0]
        return image, label

class BehavioralCloningSegFormer(nn.Module):
    def __init__(self, model_name, num_outputs=2):
        super().__init__()
        self.segformer = AutoModel.from_pretrained(model_name)
        self.regression_head = nn.Linear(self.segformer.config.hidden_sizes[-1], num_outputs)

    def forward(self, pixel_values):
        outputs = self.segformer(pixel_values=pixel_values, output_hidden_states=False)
        pooled = torch.mean(outputs.last_hidden_state, dim=[2,3])
        return self.regression_head(pooled)

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"lr_{args.lr}_bs_{args.batch_size}_wspeed_{args.weight_speed}"
    if args.freeze_base: run_name += "_frozen"
    writer = SummaryWriter(f"runs/{run_name}")

    checkpoint_dir = os.path.join("checkpoints", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    print(f"--- Starting Run: {run_name} ---")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    image_processor = AutoImageProcessor.from_pretrained(args.model_name)
    df = pd.read_csv("metadata.csv").sample(frac=1, random_state=args.seed).reset_index(drop=True)

    train_size, val_size = int(0.8 * len(df)), int(0.1 * len(df))
    train_df, val_df = df.iloc[:train_size].copy(), df.iloc[train_size:train_size+val_size].copy()

    # --- NEW FEATURE: Use a subset of the training data ---
    if args.train_subset_size is not None:
        print(f"--- Limiting training data to first {args.train_subset_size} samples ---")
        train_df = train_df.iloc[:args.train_subset_size].copy()

    # Fit the scaler on the final training dataframe (raw values)
    scaler = MinMaxScaler()
    scaler.fit(train_df[['speed','steering']])
    joblib.dump(scaler, 'label_scaler.pkl')

    # --- BUG FIX: In-place scaling of dataframes is REMOVED ---
    # The scaling will now happen inside the Dataset class.

    sampler = None
    if args.use_weighted_sampling:
        # This is now cleaner as train_df contains raw steering values
        steering_values = train_df['steering'].values
        sample_weights = torch.DoubleTensor((abs(steering_values) + 0.1) ** 1.5)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights))

    # --- BUG FIX: Pass the fitted scaler object to the Dataset instances ---
    train_dataset = CarlaDataset(train_df, scaler, image_processor, augment_flipping=args.augment_flipping)
    val_dataset = CarlaDataset(val_df, scaler, image_processor)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None), num_workers=args.num_workers, collate_fn=safe_collate, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=safe_collate, pin_memory=True)

    model = BehavioralCloningSegFormer(model_name=args.model_name).to(device)

    if args.freeze_base:
        print("--- FREEZING BASE MODEL ---")
        for param in model.segformer.parameters(): param.requires_grad = False

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(trainable_params, lr=args.lr)
    criterion = nn.MSELoss(reduction='none')
    scaler_amp = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    scheduler = None
    if args.scheduler == 'cosine':
        num_training_steps = args.epochs * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
        print("Using Cosine Annealing scheduler with warmup.")

    start_epoch, best_val_loss, patience_counter = 0, float('inf'), 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_epoch, best_val_loss = checkpoint.get('epoch', 0) + 1, checkpoint.get('best_val_loss', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except ValueError:
            print("Could not load optimizer/scheduler state, re-initializing.")
        print(f"Resuming from checkpoint at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_progress = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")

        for i, (images, labels) in enumerate(epoch_progress):
            if not images.numel(): continue
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                outputs = model(images)
                loss_unreduced = criterion(outputs, labels)
                speed_loss, steering_loss = loss_unreduced[:,0].mean(), loss_unreduced[:,1].mean()
                total_loss = args.weight_speed*speed_loss + (1.0 - args.weight_speed)*steering_loss

            scaler_amp.scale(total_loss).backward()
            if args.max_grad_norm > 0:
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            if scheduler: scheduler.step()

            global_step = epoch*len(train_loader)+i
            writer.add_scalar('Loss/train_total', total_loss.item(), global_step)
            if (i+1)%50==0 and scheduler: writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], global_step)

        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                if not images.numel(): continue
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    outputs = model(images)
                    loss_unreduced = criterion(outputs, labels)
                    total_loss = args.weight_speed*loss_unreduced[:,0].mean() + (1.0 - args.weight_speed)*loss_unreduced[:,1].mean()
                val_total_loss += total_loss.item()

        avg_val_loss = val_total_loss/len(val_loader) if val_loader else 0
        writer.add_scalar('Loss/val_total', avg_val_loss, epoch)
        print(f"\nEpoch [{epoch+1}] Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss, patience_counter = avg_val_loss, 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Validation loss improved. Saving best model to {best_model_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.patience}")

        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_loss': best_val_loss, 'patience_counter': patience_counter
        }, checkpoint_path)

        if patience_counter >= args.patience:
            print("Early stopping triggered."); break

    print("Training finished.")
    writer.close()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a behavioral cloning model.")
    # --- NEW FEATURE: Argument to limit training data size ---
    parser.add_argument('--train_subset_size', type=int, default=None, help="Limit training to the first N samples. E.g., 500.")
    
    parser.add_argument('--model_name', type=str, default="nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--weight_speed', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--freeze_base', action='store_true')
    parser.add_argument('--use_weighted_sampling', action='store_true')
    parser.add_argument('--augment_flipping', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--scheduler', type=str, default=None, choices=['cosine'])
    parser.add_argument('--warmup_steps', type=int, default=500, help="Number of warmup steps for scheduler.")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Max gradient norm for clipping.")

    args = parser.parse_args()
    main(args)