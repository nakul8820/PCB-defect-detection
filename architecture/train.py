import os
import gc
import random
import torch
import wandb
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback

from dataset import PCBDataset, SubsetWithTransform, get_transforms
from model import PCBInspector
from inference import plot_pcb_random_results


class MemoryCleanupCallback(Callback):
    """Callback to clean up memory periodically."""
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()


def create_datasets_and_loaders(root_dir, batch_size=16, num_workers=2):
    """Create train/val/test datasets and dataloaders."""
    
    base_dataset = PCBDataset(root_dir, transform=None)
    print(f"Total valid image/annotation pairs found: {len(base_dataset)}")
    
    if len(base_dataset) == 0:
        raise RuntimeError("No images found. Check dataset path.")
    
    dataset_size = len(base_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.2 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    indices = list(range(dataset_size))
    random.seed(42)
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_transform, val_transform = get_transforms()
    
    train_ds = SubsetWithTransform(base_dataset, train_indices, transform=train_transform)
    val_ds = SubsetWithTransform(base_dataset, val_indices, transform=val_transform)
    test_ds = SubsetWithTransform(base_dataset, test_indices, transform=val_transform)
    
    print(f"Split complete: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    
    def collate_fn(batch): 
        return tuple(zip(*batch))
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=8, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def train_pcb_model(
    root_dir,
    wandb_api_key=None,
    project_name="PCB_Inspection",
    run_name="pcb_detection_run",
    max_epochs=50,
    lr=0.0005,
    batch_size=16,
    num_workers=2
):
    """Complete training pipeline."""
    
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    
    train_loader, val_loader, test_loader = create_datasets_and_loaders(
        root_dir, batch_size, num_workers
    )
    
    wandb.finish()
    
    wandb_logger = WandbLogger(
        project=project_name, 
        name=run_name,
        log_model=False,
        reinit=True
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="pcb_checkpoints/", 
        filename="best-pcb-{epoch:02d}-{val_mAP:.3f}",
        monitor="val/mAP", 
        mode="max", 
        save_top_k=2,
        save_last=True 
    )
    
    early_stop = EarlyStopping(
        monitor="val/mAP",
        patience=15,
        mode="max",
        verbose=True
    )
    
    model = PCBInspector(lr=lr)
    
    trainer = L.Trainer(
        max_epochs=max_epochs,
        num_sanity_val_steps=2,
        logger=wandb_logger,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=5,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        callbacks=[
            checkpoint_callback,
            early_stop,
            MemoryCleanupCallback()
        ],
        accumulate_grad_batches=2
    )
    
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print(f"\nTesting best model: {checkpoint_callback.best_model_path}")
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")
    
    if trainer.global_rank == 0:
        best_model_path = checkpoint_callback.best_model_path
        test_ds = test_loader.dataset
        num_to_sample = min(len(test_ds), 8)
        
        if best_model_path:
            print(f"Uploading best model to WandB: {best_model_path}")
            artifact = wandb.Artifact('best_pcb_model', type='model')
            artifact.add_file(best_model_path)
            wandb.log_artifact(artifact)
            
        test_results = {k: v.item() if hasattr(v, 'item') else v 
                       for k, v in trainer.callback_metrics.items()}
        
        columns = ["Defect Class", "mAP Score"]
        summary_table = wandb.Table(columns=columns)
        
        for name in model.class_names[1:]:
            metric_key = f"test/mAP_{name}"
            if metric_key not in test_results:
                metric_key = f"val/mAP_{name}"
                
            score = test_results.get(metric_key, 0.0)
            summary_table.add_data(name, float(score))
        
        wandb.log({"Final_Class_Performance": summary_table})
        
        final_path = "pcb_checkpoints/final_model_weights.ckpt"
        trainer.save_checkpoint(final_path)
        
        print(f"\n{'='*60}")
        print(f"Training & Testing Complete!")
        best_score = checkpoint_callback.best_model_score
        if best_score is not None:
            print(f"Best Validation mAP: {best_score:.4f}")
        print(f"Final weights saved to: {final_path}")
        print(f"{'='*60}")

        plot_pcb_random_results(best_model_path, test_ds, sample_count=num_to_sample)
        wandb.finish()


if __name__ == "__main__":
    root_dir = "/kaggle/input/pcb-defects/PCB_DATASET"
    
    train_pcb_model(
        root_dir=root_dir,
        project_name="PCB_Inspection",
        run_name="production_run_v1",
        max_epochs=50,
        lr=0.0005,
        batch_size=16,
        num_workers=2
    )
