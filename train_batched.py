#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MFN batched training script (MFNversion_2)
File path: /public_new/work_space/fengjiali/MKwithML/MFNversion_2/dual_branch/train_batched.py

Features:
1. Support batched training to handle large datasets with lower memory usage.
2. Pretraining mode: use simulated data to train the displacement head.
3. Fine-tuning mode: use real data to train both heads and the LSTM branch.
4. Optional model accumulation: continue each batch from the previous checkpoint.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import json
from datetime import datetime
import logging
import random
import warnings

warnings.filterwarnings('ignore')

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from model_definition import create_mfn_model, orthogonal_loss
from data_processor import SimulateDataProcessor, CleanedDataProcessor, MFNDataset, mfn_collate_fn


def setup_logging(save_dir, batch_num):
    """Configure logging for the current batch"""
    log_file = os.path.join(save_dir, f'batch_{batch_num}_training.log')
    
    logger = logging.getLogger(f'batch_{batch_num}')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_folder_batches(data_processor, num_batches, max_folders=None, random_seed=42):
    """Split data folders into evenly sized batches"""
    print(f"\n[Batching] Start scanning data folders...")
    print(f"[Batching] Target number of batches: {num_batches}")
    if max_folders:
        print(f"[Batching] Maximum folder cap: {max_folders}")
    
    # Collect every available folder
    all_folders = data_processor.get_all_data_folders(max_folders=max_folders)
    
    if len(all_folders) == 0:
        print("[Error] No valid data folders found")
        return []
    
    print(f"[Batching] Located {len(all_folders)} data folders")
    
    # Shuffle deterministically
    random.seed(random_seed)
    random.shuffle(all_folders)
    
    # Determine the number of folders per batch
    folders_per_batch = len(all_folders) // num_batches
    remainder = len(all_folders) % num_batches
    
    batches = []
    start_idx = 0
    
    for i in range(num_batches):
        # Distribute the remainder across the first batches
        current_batch_size = folders_per_batch + (1 if i < remainder else 0)
        end_idx = start_idx + current_batch_size
        
        batch_folders = all_folders[start_idx:end_idx]
        batches.append(batch_folders)
        
        print(f"[Batching] Batch {i+1}: {len(batch_folders)} folders")
        
        start_idx = end_idx
    
    return batches


def save_batch_info(save_dir, batch_num, batch_folders):
    """Persist metadata describing a processed batch"""
    batch_info = {
        'batch_number': batch_num,
        'total_folders': len(batch_folders),
        'folders': batch_folders,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    info_file = os.path.join(save_dir, f'batch_{batch_num}_info.json')
    with open(info_file, 'w') as f:
        json.dump(batch_info, f, indent=2)


class MFNBatchedTrainer:
    """Trainer that handles MFN batched training"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Prepare output directory
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Persist configuration
        self._save_config()
        
        # Initialize the appropriate data processor
        if args.mode == 'pretrain':
            self.data_processor = SimulateDataProcessor(args.data_root)
        else:
            self.data_processor = CleanedDataProcessor(args.data_root)
        
        # Model-related state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.resume_info = None  # Store information needed for resuming training
        
        # Training history
        self.batch_summaries = []
        self.all_train_losses = []
        self.all_val_losses = []
        
        # Load resume checkpoint if provided
        if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
            print(f"\n[Resume] Found checkpoint: {args.resume_checkpoint}")
            self.resume_info = self._load_resume_checkpoint(args.resume_checkpoint)
            print(f"[Resume] Loaded resume metadata successfully")
        
        print(f"\n{'='*60}")
        print("Batched trainer ready")
        print(f"Mode: {args.mode}")
        print(f"Device: {self.device}")
        print(f"Batches: {args.num_batches}")
        if args.resume_checkpoint:
            print("Resume mode: Enabled")
            print(f"Resume checkpoint: {args.resume_checkpoint}")
            print(f"Continue from batch {args.resume_from_batch}")
        print(f"{'='*60}\n")
    
    def _load_resume_checkpoint(self, checkpoint_path):
        """Load checkpoint metadata for resume training"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            resume_info = {
                'checkpoint': checkpoint,
                'model_state_dict': checkpoint.get('model_state_dict'),
                'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
                'scheduler_state_dict': checkpoint.get('scheduler_state_dict'),
                'batch_number': checkpoint.get('batch_number', 1),
                'epoch': checkpoint.get('epoch', 0),
                'args': checkpoint.get('args', {})
            }
            return resume_info
        except Exception as e:
            print(f"[Error] Failed to load resume checkpoint: {e}")
            return None
    
    def _save_config(self):
        """Persist the training configuration to disk"""
        config = vars(self.args)
        config_path = os.path.join(self.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"[Config] Training configuration saved to: {config_path}")
    
    def _create_model(self):
        """Instantiate the MFN model"""
        print("\n[Model] Creating model...")
        
        model = create_mfn_model(
            img_h=self.args.img_h,
            img_w=self.args.img_w,
            use_orthogonal_loss=(self.args.mode == 'finetune' and self.args.use_orth_loss)
        )
        
        model = model.to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"[Model] Total parameters: {total_params:,}")
        print(f"[Model] Trainable parameters: {trainable_params:,}")
        
        return model
    
    def _create_optimizer(self):
        """Construct the optimizer"""
        if self.args.optimizer == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        else:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=self.args.weight_decay
            )
        
        print(f"[Optimizer] Using {self.args.optimizer} with lr={self.args.lr}")
        return optimizer
    
    def _create_scheduler(self):
        """Set up the learning rate scheduler if requested"""
        if self.args.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.epochs_per_batch,
                eta_min=self.args.lr * 0.01
            )
        elif self.args.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.args.step_size,
                gamma=self.args.gamma_scheduler
            )
        elif self.args.scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.7,
                patience=5,
                verbose=True,
                min_lr=1e-6
            )
        else:
            scheduler = None
        
        if scheduler:
            print(f"[Scheduler] Using {self.args.scheduler} scheduler")
        return scheduler
    
    def load_previous_model(self, batch_num):
        """Load the checkpoint from the previous batch"""
        if batch_num > 1:
            prev_model_path = os.path.join(
                self.output_dir,
                f'batch_{batch_num-1}_final_model.pth'
            )
            
            if os.path.exists(prev_model_path):
                print(f"[Load] Restoring model from previous batch: {prev_model_path}")
                checkpoint = torch.load(prev_model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                if not self.args.reset_optimizer and 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("[Load] Optimizer state restored")
                
                return True
        return False
    
    def train_batch(self, batch_num, batch_folders):
        """Train a single batch of folders"""
        logger = setup_logging(self.output_dir, batch_num)
        logger.info(f"{'='*60}")
        logger.info(f"Starting batch {batch_num}/{self.args.num_batches}")
        logger.info(f"Batch contains {len(batch_folders)} folders")
        logger.info(f"{'='*60}")
        
        # Save batch-level metadata
        save_batch_info(self.output_dir, batch_num, batch_folders)
        
        # Process current batch folders
        logger.info("[Data] Begin processing current batch...")
        batch_data_list = self.data_processor.process_all_data_for_batch(batch_folders)
        
        if batch_data_list is None or not batch_data_list:
            logger.error("[Error] Failed to process data for this batch")
            return None
        
        logger.info(f"[Data] Processed {len(batch_data_list)} folders successfully")
        
        # Initialize the model during the first batch
        if self.model is None:
            self.model = self._create_model()
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()
            
            # If resume checkpoint is provided, load it first
            if self.args.resume_checkpoint and self.resume_info:
                logger.info(f"[Resume] Loading checkpoint: {self.args.resume_checkpoint}")
                self.model.load_state_dict(self.resume_info['model_state_dict'])
                
                if self.resume_info['optimizer_state_dict'] and not self.args.reset_optimizer:
                    self.optimizer.load_state_dict(self.resume_info['optimizer_state_dict'])
                    logger.info("[Resume] Optimizer state restored")
                
                if self.resume_info['scheduler_state_dict'] and self.scheduler:
                    try:
                        self.scheduler.load_state_dict(self.resume_info['scheduler_state_dict'])
                        logger.info("[Resume] Scheduler state restored")
                    except:
                        logger.warning("[Resume] Failed to load scheduler state; using fresh state")
                
                logger.info(f"[Resume] Training will continue from batch {self.args.resume_from_batch}")
            
            # Fine-tuning mode: load pretrained weights when not resuming
            elif self.args.mode == 'finetune' and self.args.pretrain_checkpoint:
                logger.info(f"[Model] Loading pretrained weights: {self.args.pretrain_checkpoint}")
                checkpoint = torch.load(self.args.pretrain_checkpoint, map_location='cpu')
                
                model_dict = self.model.state_dict()
                pretrained_dict = checkpoint['model_state_dict']
                pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                 if k in model_dict and v.size() == model_dict[k].size()}
                
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)
                logger.info(f"[Model] Loaded {len(pretrained_dict)} pretrained parameters")
                
                # Apply freezing strategy
                self.model.freeze_mobilevit()
                self.model.freeze_cnn_early_layers(num_layers=self.args.freeze_cnn_layers)
        
            # Load previous batch checkpoint when accumulation is enabled
        if self.args.use_model_accumulation:
            self.load_previous_model(batch_num)
        
        # Build dataset
        mode = 'pretrain' if self.args.mode == 'pretrain' else 'finetune'
        add_noise = self.args.add_noise if self.args.mode == 'pretrain' else False
        
        dataset = MFNDataset(
            batch_data_list,
            seq_len_min=self.args.seq_len_min,
            seq_len_max=self.args.seq_len_max,
            seq_step=self.args.seq_step,
            mode=mode,
            add_noise=add_noise,
            noise_std=self.args.noise_std
        )
        
        # Split into train/validation subsets
        total_size = len(dataset)
        train_ratio = self.args.train_ratio
        train_size = int(total_size * train_ratio)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Prepare dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=lambda batch: mfn_collate_fn(batch, mode=mode),
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: mfn_collate_fn(batch, mode=mode),
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        logger.info(f"[Data] Train size: {len(train_dataset)}")
        logger.info(f"[Data] Validation size: {len(val_dataset)}")
        
        # Track training progress for the current batch
        batch_train_losses = []
        batch_val_losses = []
        best_val_loss = float('inf')
        
        mse_loss = nn.MSELoss()
        ce_loss = nn.CrossEntropyLoss()
        
        for epoch in range(1, self.args.epochs_per_batch + 1):
            # Train
            train_metrics = self._train_epoch(
                train_loader, epoch, mse_loss, ce_loss, logger
            )
            batch_train_losses.append(train_metrics['train_loss'])
            
            # Validate
            val_metrics = self._validate_epoch(
                val_loader, mse_loss, ce_loss, logger
            )
            batch_val_losses.append(val_metrics['val_loss'])
            
            # Step scheduler
            if self.scheduler:
                if self.args.scheduler == 'plateau':
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Save best checkpoint
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self._save_checkpoint(batch_num, epoch, train_metrics, val_metrics, is_best=True)
            
            # Logging summary
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.args.mode == 'pretrain':
                logger.info(f"Epoch {epoch}/{self.args.epochs_per_batch} - "
                           f"Train Loss: {train_metrics['train_loss']:.6f}, "
                           f"Val Loss: {val_metrics['val_loss']:.6f}, "
                           f"LR: {current_lr:.8f}")
            else:  # finetune
                logger.info(f"Epoch {epoch}/{self.args.epochs_per_batch} - "
                           f"Train Loss: {train_metrics['train_loss']:.6f} "
                           f"(Disp: {train_metrics['train_disp_loss']:.6f}, "
                           f"Order: {train_metrics['train_order_loss']:.6f}, "
                           f"Orth: {train_metrics['train_orth_loss']:.6f}, "
                           f"Acc: {train_metrics['train_order_acc']:.4f}) | "
                           f"Val Loss: {val_metrics['val_loss']:.6f} "
                           f"(Disp: {val_metrics['val_disp_loss']:.6f}, "
                           f"Order: {val_metrics['val_order_loss']:.6f}, "
                           f"Acc: {val_metrics['val_order_acc']:.4f}) | "
                           f"LR: {current_lr:.8f}")
        
        # Save final checkpoint for this batch
        self._save_checkpoint(batch_num, self.args.epochs_per_batch, 
                            train_metrics, val_metrics, is_final=True)
        
        # Record batch summary statistics
        batch_summary = {
            'batch_number': batch_num,
            'folders_count': len(batch_folders),
            'samples_count': len(dataset),
            'train_losses': batch_train_losses,
            'val_losses': batch_val_losses,
            'best_val_loss': best_val_loss,
            'final_train_loss': batch_train_losses[-1],
            'final_val_loss': batch_val_losses[-1],
            'mode': self.args.mode
        }
        
        self.batch_summaries.append(batch_summary)
        self.all_train_losses.extend(batch_train_losses)
        self.all_val_losses.extend(batch_val_losses)
        
        logger.info(f"Batch {batch_num} training finished")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        
        return batch_summary
    
    def _train_epoch(self, train_loader, epoch, mse_loss, ce_loss, logger):
        """Run one training epoch"""
        self.model.train()
        
        total_loss = 0
        total_disp_loss = 0
        total_order_loss = 0
        total_orth_loss = 0
        total_order_correct = 0
        total_samples = 0
        num_batches = 0
        
        for batch_data in train_loader:
            try:
                if self.args.mode == 'pretrain':
                    # Collate returns: img_ch1, img_ch2, img_ch3, img_3ch, num_seq, labels, mask, lengths
                    img_ch1, img_ch2, img_ch3, img_3ch, num_seq, labels, mask, lengths = batch_data
                    
                    img_ch1 = img_ch1.to(self.device, non_blocking=True)
                    img_ch2 = img_ch2.to(self.device, non_blocking=True)
                    img_ch3 = img_ch3.to(self.device, non_blocking=True)
                    img_3ch = img_3ch.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    mask = mask.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(img_ch1, img_ch2, img_ch3, img_3ch, mode='pretrain')
                    
                    # Only compute loss where mask=True, ignore NaN targets
                    valid_mask = mask.bool()
                    valid_count = int(valid_mask.sum().item())
                    if valid_count == 0:
                        # Skip this minibatch if no valid labels are present
                        logger.info(f"Epoch {epoch} - skipped minibatch (no valid pretrain labels)")
                        continue
                    
                    preds_valid = outputs[valid_mask]
                    labels_valid = labels[valid_mask]
                    loss = mse_loss(preds_valid, labels_valid)
                    total_disp_loss += loss.item()
                else:  # finetune
                    # Collate returns: img_ch1, img_ch2, img_ch3, img_3ch, num_seq, disp_labels, order_labels, mask, lengths
                    img_ch1, img_ch2, img_ch3, img_3ch, num_seq, disp_labels, order_labels, mask, lengths = batch_data
                    
                    img_ch1 = img_ch1.to(self.device, non_blocking=True)
                    img_ch2 = img_ch2.to(self.device, non_blocking=True)
                    img_ch3 = img_ch3.to(self.device, non_blocking=True)
                    img_3ch = img_3ch.to(self.device, non_blocking=True)
                    num_seq = num_seq.to(self.device, non_blocking=True)
                    disp_labels = disp_labels.to(self.device, non_blocking=True)
                    order_labels = order_labels.to(self.device, non_blocking=True)
                    lengths = lengths.to(self.device, non_blocking=True)
                    mask = mask.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(img_ch1, img_ch2, img_ch3, img_3ch,
                                       num_seq, lengths, mode='finetune')
                    
                    # Compute displacement MSE on valid samples only
                    valid_mask_f = mask.bool()
                    valid_count_f = int(valid_mask_f.sum().item())
                    if valid_count_f > 0:
                        preds_disp = outputs['displacement'][valid_mask_f]
                        labels_disp = disp_labels[valid_mask_f]
                        disp_mse_loss = mse_loss(preds_disp, labels_disp)
                    else:
                        # Fall back to zero so that order loss still backpropagates
                        disp_mse_loss = torch.tensor(0.0, device=self.device)
                    
                    # Compute order cross-entropy
                    order_ce_loss = ce_loss(outputs['order_logits'], order_labels)
                    
                    # Track order accuracy for logging
                    _, predicted = torch.max(outputs['order_logits'], 1)
                    total_order_correct += (predicted == order_labels).sum().item()
                    total_samples += order_labels.size(0)
                    
                    # Compute orthogonality penalty if enabled
                    orth_loss_value = 0
                    if self.args.use_orth_loss and 'displacement_hidden' in outputs:
                        orth_loss_value = orthogonal_loss(outputs['displacement_hidden'], outputs['order_hidden'])
                    
                    # Scale the auxiliary loss for comparable magnitudes
                    orth_loss_scaled_for_disp = orth_loss_value * 5000 if isinstance(orth_loss_value, torch.Tensor) else 0
                    orth_loss_scaled_for_order = orth_loss_value * 5 if isinstance(orth_loss_value, torch.Tensor) else 0
                    
                    # Compose final losses for each head
                    disp_head_loss = disp_mse_loss + orth_loss_scaled_for_disp
                    
                    # Order head includes CE plus scaled orthogonality term
                    order_head_loss = order_ce_loss + orth_loss_scaled_for_order
                    
                    # Total loss aggregates both heads
                    loss = disp_head_loss + order_head_loss
                    
                    # Accumulate metrics for logging
                    total_disp_loss += disp_head_loss.item()
                    total_order_loss += order_head_loss.item()
                    total_orth_loss += orth_loss_value.item() if isinstance(orth_loss_value, torch.Tensor) else 0
                
                total_loss += loss.item()
                
                loss.backward()
                
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                
                self.optimizer.step()
                num_batches += 1
                
            except Exception as e:
                logger.error(f"Error encountered during training minibatch: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_disp_loss = total_disp_loss / num_batches if num_batches > 0 else 0.0
        
        metrics = {
            'train_loss': avg_loss,
            'train_disp_loss': avg_disp_loss
        }
        
        if self.args.mode == 'finetune':
            metrics['train_order_loss'] = total_order_loss / num_batches if num_batches > 0 else 0.0
            metrics['train_orth_loss'] = total_orth_loss / num_batches if num_batches > 0 else 0.0
            metrics['train_order_acc'] = total_order_correct / total_samples if total_samples > 0 else 0.0
        
        return metrics
    
    @torch.no_grad()
    def _validate_epoch(self, val_loader, mse_loss, ce_loss, logger):
        """Run validation for one epoch"""
        self.model.eval()
        
        total_loss = 0
        total_disp_loss = 0
        total_order_loss = 0
        total_orth_loss = 0
        total_order_correct = 0
        total_samples = 0
        num_batches = 0
        
        for batch_data in val_loader:
            try:
                if self.args.mode == 'pretrain':
                    # Collate returns: img_ch1, img_ch2, img_ch3, img_3ch, num_seq, labels, mask, lengths
                    img_ch1, img_ch2, img_ch3, img_3ch, num_seq, labels, mask, lengths = batch_data
                    
                    img_ch1 = img_ch1.to(self.device)
                    img_ch2 = img_ch2.to(self.device)
                    img_ch3 = img_ch3.to(self.device)
                    img_3ch = img_3ch.to(self.device)
                    labels = labels.to(self.device)
                    mask = mask.to(self.device)
                    
                    outputs = self.model(img_ch1, img_ch2, img_ch3, img_3ch, mode='pretrain')
                    
                    # Only evaluate masked (valid) samples
                    valid_mask = mask.bool()
                    valid_count = int(valid_mask.sum().item())
                    if valid_count == 0:
                        logger.info("Validation minibatch skipped (no valid labels)")
                        continue
                    
                    preds_valid = outputs[valid_mask]
                    labels_valid = labels[valid_mask]
                    loss = mse_loss(preds_valid, labels_valid)
                    total_disp_loss += loss.item()
                else:  # finetune
                    # Collate returns: img_ch1, img_ch2, img_ch3, img_3ch, num_seq, disp_labels, order_labels, mask, lengths
                    img_ch1, img_ch2, img_ch3, img_3ch, num_seq, disp_labels, order_labels, mask, lengths = batch_data
                    
                    img_ch1 = img_ch1.to(self.device)
                    img_ch2 = img_ch2.to(self.device)
                    img_ch3 = img_ch3.to(self.device)
                    img_3ch = img_3ch.to(self.device)
                    num_seq = num_seq.to(self.device)
                    disp_labels = disp_labels.to(self.device)
                    order_labels = order_labels.to(self.device)
                    lengths = lengths.to(self.device)
                    mask = mask.to(self.device)
                    
                    outputs = self.model(img_ch1, img_ch2, img_ch3, img_3ch,
                                       num_seq, lengths, mode='finetune')
                    
                    # Compute displacement MSE on valid samples only
                    valid_mask_f = mask.bool()
                    valid_count_f = int(valid_mask_f.sum().item())
                    if valid_count_f > 0:
                        preds_disp = outputs['displacement'][valid_mask_f]
                        labels_disp = disp_labels[valid_mask_f]
                        disp_mse_loss = mse_loss(preds_disp, labels_disp)
                    else:
                        # Use zero displacement loss if nothing is valid
                        disp_mse_loss = torch.tensor(0.0, device=self.device)
                    
                    # Compute order cross-entropy
                    order_ce_loss = ce_loss(outputs['order_logits'], order_labels)
                    
                    # Orthogonal loss is not applied during evaluation
                    orth_loss_value = 0
                    
                    # Displacement head loss reduces to pure MSE here
                    disp_head_loss = disp_mse_loss
                    
                    # Order head loss is cross-entropy
                    order_head_loss = order_ce_loss
                    
                    # Combine losses
                    loss = disp_head_loss + order_head_loss
                    
                    total_disp_loss += disp_head_loss.item()
                    total_order_loss += order_head_loss.item()
                    total_orth_loss += orth_loss_value
                    
                    # Track accuracy
                    _, predicted = torch.max(outputs['order_logits'], 1)
                    total_order_correct += (predicted == order_labels).sum().item()
                    total_samples += order_labels.size(0)
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                logger.error(f"Error encountered during validation minibatch: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_disp_loss = total_disp_loss / num_batches if num_batches > 0 else 0.0
        
        metrics = {
            'val_loss': avg_loss,
            'val_disp_loss': avg_disp_loss
        }
        
        if self.args.mode == 'finetune':
            metrics['val_order_loss'] = total_order_loss / num_batches if num_batches > 0 else 0.0
            metrics['val_orth_loss'] = total_orth_loss / num_batches if num_batches > 0 else 0.0
            metrics['val_order_acc'] = total_order_correct / total_samples if total_samples > 0 else 0
        
        return metrics
    
    def _save_checkpoint(self, batch_num, epoch, train_metrics, val_metrics, is_best=False, is_final=False):
        """Persist a training checkpoint"""
        checkpoint = {
            'batch_number': batch_num,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'args': vars(self.args)
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_final:
            filename = f'batch_{batch_num}_final_model.pth'
        elif is_best:
            filename = f'batch_{batch_num}_best_model.pth'
        else:
            filename = f'batch_{batch_num}_epoch_{epoch}.pth'
        
        filepath = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, filepath)
    
    def save_training_summary(self):
        """Write an aggregated training summary to disk"""
        summary = {
            'total_batches': len(self.batch_summaries),
            'total_epochs': len(self.all_train_losses),
            'batch_summaries': self.batch_summaries,
            'config': vars(self.args),
            'mode': self.args.mode,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        summary_path = os.path.join(self.output_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def run_batched_training(self):
        """Execute the full batched training workflow"""
        print(f"\n{'='*60}")
        print(f"Starting batched training - mode: {self.args.mode}")
        print(f"{'='*60}\n")
        
        # Retrieve batches of folders
        folder_batches = get_folder_batches(
            self.data_processor,
            self.args.num_batches,
            max_folders=self.args.max_folders,
            random_seed=self.args.seed
        )
        
        if not folder_batches:
            print("[Error] Unable to create folder batches")
            return
        
        # Save batch partition metadata
        batch_division_info = {
            'num_batches': self.args.num_batches,
            'batches': [{'batch_number': i+1, 'folders_count': len(batch)}
                       for i, batch in enumerate(folder_batches)],
            'mode': self.args.mode,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        division_path = os.path.join(self.output_dir, 'batch_division.json')
        with open(division_path, 'w') as f:
            json.dump(batch_division_info, f, indent=2)
        
        # Iterate through each batch sequentially
        for batch_num, batch_folders in enumerate(folder_batches, 1):
            if batch_num < self.args.resume_from_batch:
                print(f"Skip batch {batch_num}; resume begins at batch {self.args.resume_from_batch}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Start training batch {batch_num}/{self.args.num_batches}")
            print(f"{'='*60}")
            
            batch_summary = self.train_batch(batch_num, batch_folders)
            
            if batch_summary:
                print(f"\nBatch {batch_num} completed:")
                print(f"  - Folders: {batch_summary['folders_count']}")
                print(f"  - Samples: {batch_summary['samples_count']}")
                print(f"  - Final train loss: {batch_summary['final_train_loss']:.6f}")
                print(f"  - Final val loss: {batch_summary['final_val_loss']:.6f}")
                print(f"  - Best val loss: {batch_summary['best_val_loss']:.6f}")
        
        # Save training summary
        self.save_training_summary()
        
        print(f"\n{'='*60}")
        print("Batched training complete!")
        print(f"Artifacts saved to: {self.output_dir}")
        print(f"{'='*60}\n")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='MFN batched training script')
    
    # Core arguments
    parser.add_argument('--mode', type=str, required=True, choices=['pretrain', 'finetune'],
                       help='Training mode: pretrain or finetune')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory for the dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory where outputs will be written')
    
    # Batching parameters
    parser.add_argument('--num_batches', type=int, default=5,
                       help='Number of folder batches to create')
    parser.add_argument('--epochs_per_batch', type=int, default=20,
                       help='Training epochs per batch')
    parser.add_argument('--use_model_accumulation', action='store_true',
                       help='Continue each batch from the previous checkpoint')
    parser.add_argument('--reset_optimizer', action='store_true',
                       help='Reset optimizer state for every batch')
    
    # Data parameters
    parser.add_argument('--max_folders', type=int, default=None,
                       help='Optional cap on number of folders to use')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Proportion of samples for the training split')
    parser.add_argument('--seq_len_min', type=int, default=5,
                       help='Minimum sequence length to sample')
    parser.add_argument('--seq_len_max', type=int, default=10,
                       help='Maximum sequence length to sample')
    parser.add_argument('--seq_step', type=int, default=1,
                       help='Sliding window step size when sampling sequences')
    parser.add_argument('--add_noise', action='store_true',
                       help='Add synthetic noise during pretraining')
    parser.add_argument('--noise_std', type=float, default=0.01,
                       help='Standard deviation of injected noise')
    
    # Model parameters
    parser.add_argument('--img_h', type=int, default=1024,
                       help='Input image height')
    parser.add_argument('--img_w', type=int, default=1280,
                       help='Input image width')
    parser.add_argument('--pretrain_checkpoint', type=str, default=None,
                       help='Path to pretrained checkpoint (used in finetune mode)')
    parser.add_argument('--freeze_cnn_layers', type=int, default=3,
                       help='Number of initial CNN layers to freeze during finetuning')
    
    # Optimization parameters
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay value')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer choice')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='Learning rate scheduler strategy')
    parser.add_argument('--step_size', type=int, default=10,
                       help='Step size for StepLR')
    parser.add_argument('--gamma_scheduler', type=float, default=0.7,
                       help='Decay factor for StepLR')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping threshold')
    
    # Loss weights (used during finetuning)
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Weight for the displacement loss')
    parser.add_argument('--beta', type=float, default=0.5,
                       help='Weight for the order classification loss')
    parser.add_argument('--gamma_loss', type=float, default=0.1,
                       help='Weight for the orthogonality penalty')
    parser.add_argument('--use_orth_loss', action='store_true',
                       help='Enable orthogonality loss during finetuning')
    
    # Miscellaneous parameters
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Resume training parameters
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                       help='Resume training from a checkpoint (overrides pretrain/model accumulation)')
    parser.add_argument('--resume_from_batch', type=int, default=1,
                       help='Batch index to resume from (default restarts at batch 1)')

    return parser.parse_args()


def main():
    """Program entry point"""
    args = parse_args()
    
    # Seed everything for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create trainer
    trainer = MFNBatchedTrainer(args)
    
    # Launch training
    trainer.run_batched_training()


if __name__ == '__main__':
    main()
