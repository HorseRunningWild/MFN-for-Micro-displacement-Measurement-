#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MFN variable dataset-size training script (MFNversion_2)
Original path: /public_new/work_space/fengjiali/MKwithML/MFNversion_2/dual_branch/train_variable_data_size.py

Capabilities:
1. Supports training on dataset fractions (1/5 through 5/5)
2. Early stopping: stops when validation CE stays below a threshold for N epochs
3. Detailed logging to analyze data requirements for finetuning
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

# Add project path to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from model_definition import create_mfn_model, orthogonal_loss
from data_processor import FinetuneDataProcessor, MFNDataset, mfn_collate_fn


def setup_logging(save_dir):
    """Configure logging to file and console"""
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, 'training.log')
    
    logger = logging.getLogger('variable_data_size')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, mode='w')
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


class VariableDataSizeTrainer:
    """Trainer that sweeps over dataset-size fractions"""
    
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Persist configuration
        self._save_config()
        
        # Training history containers
        self.train_losses = []
        self.val_losses = []
        self.val_ce_losses = []  # Track validation CE loss separately
        self.val_accuracies = []
        self.test_accuracies = []  # Record test accuracy
        self.test_ce_losses = []  # Record test CE loss
        self.best_val_loss = float('inf')
        self.acc_stable_count = 0  # Number of consecutive epochs meeting the accuracy criterion
        
        self.logger.info(f"Initialization complete - device: {self.device}")
        self.logger.info(f"Data fraction: {args.data_fraction:.2%}")
        self.logger.info(f"Accuracy threshold: {args.acc_threshold}, patience: {args.patience}")
        if args.test_data_root:
            self.logger.info(f"Test dataset path: {args.test_data_root}")
            self.logger.info(f"Test dataset partitions used: {args.test_data_fractions}")
    
    def _save_config(self):
        """Persist training configuration to disk"""
        config = vars(self.args)
        config_path = os.path.join(self.args.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        self.logger.info(f"Configuration saved: {config_path}")
    
    def load_data(self):
        """Load training and validation data"""
        self.logger.info("Loading training data...")
        
        # Initialize data processor
        data_processor = FinetuneDataProcessor(self.args.data_root)
        
        # Fetch all folders
        all_folders = data_processor.get_all_data_folders()
        total_folders = len(all_folders)
        
        if total_folders == 0:
            self.logger.error("No data folders found!")
            return None, None
        
        # Determine how many folders to use
        num_folders_to_use = int(total_folders * self.args.data_fraction)
        if num_folders_to_use == 0:
            num_folders_to_use = 1
        
        # Randomly select folders
        random.seed(self.args.seed)
        random.shuffle(all_folders)
        selected_folders = all_folders[:num_folders_to_use]
        
        self.logger.info(f"Total folders: {total_folders}")
        self.logger.info(f"Using folders: {num_folders_to_use} ({self.args.data_fraction:.2%})")
        
        # Process data into sequences
        batch_data_list = data_processor.process_all_data_for_batch(selected_folders)
        
        if not batch_data_list:
            self.logger.error("Failed to process data!")
            return None, None
        
        # Build dataset
        dataset = MFNDataset(
            batch_data_list,
            seq_len_min=self.args.seq_len_min,
            seq_len_max=self.args.seq_len_max,
            seq_step=self.args.seq_step,
            mode='finetune',
            add_noise=False
        )
        
        # Split into train/validation
        total_size = len(dataset)
        train_size = int(total_size * self.args.train_ratio)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=lambda batch: mfn_collate_fn(batch, mode='finetune'),
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: mfn_collate_fn(batch, mode='finetune'),
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def load_test_data(self):
        """Load optional test dataset"""
        if not self.args.test_data_root:
            return None
        
        self.logger.info("Loading test dataset...")
        
        # Initialize processor for test set
        test_processor = FinetuneDataProcessor(self.args.test_data_root)
        
        # Retrieve all test folders
        all_test_folders = test_processor.get_all_data_folders()
        total_test_folders = len(all_test_folders)
        
        if total_test_folders == 0:
            self.logger.warning("No test folders found!")
            return None
        
        # Divide test folders into five parts
        folders_per_part = total_test_folders // 5
        remainder = total_test_folders % 5
        
        test_folder_parts = []
        start_idx = 0
        for i in range(5):
            end_idx = start_idx + folders_per_part + (1 if i < remainder else 0)
            test_folder_parts.append(all_test_folders[start_idx:end_idx])
            start_idx = end_idx
        
        self.logger.info(f"Total test folders: {total_test_folders}")
        self.logger.info(f"Split into 5 parts, roughly {folders_per_part} folders each")
        
        # Select partitions based on user input
        # test_data_fractions format: "1,2,3" means use partitions 1, 2, and 3
        if self.args.test_data_fractions and self.args.test_data_fractions.strip():
            selected_parts = [int(x.strip()) for x in self.args.test_data_fractions.split(',') if x.strip()]
            selected_parts = [p - 1 for p in selected_parts if 1 <= p <= 5]  # Convert to 0-based indices
        else:
            selected_parts = [0, 1, 2, 3, 4]  # Default: use all partitions
        
        # Aggregate selected test folders
        selected_test_folders = []
        for part_idx in selected_parts:
            selected_test_folders.extend(test_folder_parts[part_idx])
        
        self.logger.info(f"Using test partitions: {[p+1 for p in selected_parts]}")
        self.logger.info(f"Selected test folders: {len(selected_test_folders)}")
        
        # Process test data
        test_data_list = test_processor.process_all_data_for_batch(selected_test_folders)
        
        if not test_data_list:
            self.logger.warning("Failed to process test data!")
            return None
        
        # Build dataset for testing
        test_dataset = MFNDataset(
            test_data_list,
            seq_len_min=self.args.seq_len_min,
            seq_len_max=self.args.seq_len_max,
            seq_step=self.args.seq_step,
            mode='finetune',
            add_noise=False
        )
        
        # Create dataloader for testing
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: mfn_collate_fn(batch, mode='finetune'),
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        self.logger.info(f"Test samples: {len(test_dataset)}")
        
        return test_loader
    
    def create_model(self):
        """Instantiate MFN model and load checkpoints"""
        self.logger.info("Building model...")
        
        model = create_mfn_model(
            img_h=self.args.img_h,
            img_w=self.args.img_w,
            use_orthogonal_loss=self.args.use_orth_loss
        )
        
        # Load pretrained weights if provided
        if self.args.pretrain_checkpoint:
            self.logger.info(f"Loading pretrained weights from: {self.args.pretrain_checkpoint}")
            checkpoint = torch.load(self.args.pretrain_checkpoint, map_location='cpu')
            
            model_dict = model.state_dict()
            pretrained_dict = checkpoint['model_state_dict']
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and v.size() == model_dict[k].size()}
            
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            self.logger.info(f"Loaded {len(pretrained_dict)} pretrained tensors")
            
            # Freezing strategy
            model.freeze_mobilevit()
            model.freeze_cnn_early_layers(num_layers=self.args.freeze_cnn_layers)
        
        model = model.to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def create_optimizer(self, model):
        """Configure optimizer"""
        if self.args.optimizer == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        else:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=self.args.weight_decay
            )
        
        self.logger.info(f"Using {self.args.optimizer} optimizer, lr: {self.args.lr}")
        return optimizer
    
    def create_scheduler(self, optimizer):
        """Create learning rate scheduler"""
        if self.args.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.args.max_epochs,
                eta_min=self.args.lr * 0.01
            )
        elif self.args.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.7
            )
        elif self.args.scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.7,
                patience=5,
                verbose=True,
                min_lr=1e-6
            )
        else:
            scheduler = None
        
        if scheduler:
            self.logger.info(f"Using {self.args.scheduler} scheduler")
        return scheduler
    
    def train_epoch(self, model, train_loader, optimizer, mse_loss, ce_loss, epoch):
        """Train for a single epoch"""
        model.train()
        
        total_loss = 0
        total_disp_loss = 0
        total_order_loss = 0
        total_order_correct = 0
        total_samples = 0
        num_batches = 0
        
        for batch_data in train_loader:
            try:
                img_ch1, img_ch2, img_ch3, img_3ch, num_seq, disp_labels, order_labels, mask, lengths = batch_data
                
                img_ch1 = img_ch1.to(self.device, non_blocking=True)
                img_ch2 = img_ch2.to(self.device, non_blocking=True)
                img_ch3 = img_ch3.to(self.device, non_blocking=True)
                img_3ch = img_3ch.to(self.device, non_blocking=True)
                num_seq = num_seq.to(self.device, non_blocking=True)
                disp_labels = disp_labels.to(self.device, non_blocking=True)
                order_labels = order_labels.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)
                lengths = lengths.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(img_ch1, img_ch2, img_ch3, img_3ch,
                               num_seq, lengths, mode='finetune')
                
                # Compute losses
                disp_mse_loss = mse_loss(outputs['displacement'], disp_labels)
                order_ce_loss = ce_loss(outputs['order_logits'], order_labels)
                
                # Compute accuracy
                _, predicted = torch.max(outputs['order_logits'], 1)
                total_order_correct += (predicted == order_labels).sum().item()
                total_samples += order_labels.size(0)
                
                # Orthogonality penalty
                orth_loss_value = 0
                if self.args.use_orth_loss and 'displacement_hidden' in outputs:
                    orth_loss_value = orthogonal_loss(outputs['displacement_hidden'], outputs['order_hidden'])
                    orth_loss_scaled_for_disp = orth_loss_value * 5000
                    orth_loss_scaled_for_order = orth_loss_value * 5
                else:
                    orth_loss_scaled_for_disp = 0
                    orth_loss_scaled_for_order = 0
                
                # Aggregate loss
                disp_head_loss = disp_mse_loss + orth_loss_scaled_for_disp
                order_head_loss = order_ce_loss + orth_loss_scaled_for_order
                loss = disp_head_loss + order_head_loss
                
                total_loss += loss.item()
                total_disp_loss += disp_head_loss.item()
                total_order_loss += order_head_loss.item()
                
                loss.backward()
                
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)
                
                optimizer.step()
                num_batches += 1
                
            except Exception as e:
                self.logger.error(f"Training batch error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        metrics = {
            'train_loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'train_disp_loss': total_disp_loss / num_batches if num_batches > 0 else 0.0,
            'train_order_loss': total_order_loss / num_batches if num_batches > 0 else 0.0,
            'train_order_acc': total_order_correct / total_samples if total_samples > 0 else 0.0
        }
        
        return metrics
    
    @torch.no_grad()
    def validate_epoch(self, model, val_loader, mse_loss, ce_loss, epoch):
        """Validate model for one epoch"""
        model.eval()
        
        total_loss = 0
        total_disp_loss = 0
        total_order_loss = 0
        total_order_ce_loss = 0  # Track CE loss alone (excluding orthogonality)
        total_order_correct = 0
        total_samples = 0
        num_batches = 0
        
        for batch_data in val_loader:
            try:
                img_ch1, img_ch2, img_ch3, img_3ch, num_seq, disp_labels, order_labels, mask, lengths = batch_data
                
                img_ch1 = img_ch1.to(self.device)
                img_ch2 = img_ch2.to(self.device)
                img_ch3 = img_ch3.to(self.device)
                img_3ch = img_3ch.to(self.device)
                num_seq = num_seq.to(self.device)
                disp_labels = disp_labels.to(self.device)
                order_labels = order_labels.to(self.device)
                mask = mask.to(self.device)
                lengths = lengths.to(self.device)
                
                outputs = model(img_ch1, img_ch2, img_ch3, img_3ch,
                               num_seq, lengths, mode='finetune')
                
                # Compute losses
                disp_mse_loss = mse_loss(outputs['displacement'], disp_labels)
                order_ce_loss = ce_loss(outputs['order_logits'], order_labels)
                
                # Accumulate pure CE loss
                total_order_ce_loss += order_ce_loss.item()
                
                # Aggregate loss
                disp_head_loss = disp_mse_loss
                order_head_loss = order_ce_loss
                loss = disp_head_loss + order_head_loss
                
                total_loss += loss.item()
                total_disp_loss += disp_head_loss.item()
                total_order_loss += order_head_loss.item()
                
                # Compute accuracy
                _, predicted = torch.max(outputs['order_logits'], 1)
                total_order_correct += (predicted == order_labels).sum().item()
                total_samples += order_labels.size(0)
                
                num_batches += 1
                
            except Exception as e:
                self.logger.error(f"Validation batch error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        metrics = {
            'val_loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'val_disp_loss': total_disp_loss / num_batches if num_batches > 0 else 0.0,
            'val_order_loss': total_order_loss / num_batches if num_batches > 0 else 0.0,
            'val_order_ce_loss': total_order_ce_loss / num_batches if num_batches > 0 else 0.0,  # Pure CE loss
            'val_order_acc': total_order_correct / total_samples if total_samples > 0 else 0.0
        }
        
        return metrics
    
    @torch.no_grad()
    def test_epoch(self, model, test_loader, mse_loss, ce_loss, epoch):
        """Evaluate on the test set"""
        model.eval()
        
        total_loss = 0
        total_disp_loss = 0
        total_order_loss = 0
        total_order_ce_loss = 0
        total_order_correct = 0
        total_samples = 0
        num_batches = 0
        
        for batch_data in test_loader:
            try:
                img_ch1, img_ch2, img_ch3, img_3ch, num_seq, disp_labels, order_labels, mask, lengths = batch_data
                
                img_ch1 = img_ch1.to(self.device)
                img_ch2 = img_ch2.to(self.device)
                img_ch3 = img_ch3.to(self.device)
                img_3ch = img_3ch.to(self.device)
                num_seq = num_seq.to(self.device)
                disp_labels = disp_labels.to(self.device)
                order_labels = order_labels.to(self.device)
                mask = mask.to(self.device)
                lengths = lengths.to(self.device)
                
                outputs = model(img_ch1, img_ch2, img_ch3, img_3ch,
                               num_seq, lengths, mode='finetune')
                
                # Compute losses
                disp_mse_loss = mse_loss(outputs['displacement'], disp_labels)
                order_ce_loss = ce_loss(outputs['order_logits'], order_labels)
                
                total_order_ce_loss += order_ce_loss.item()
                
                disp_head_loss = disp_mse_loss
                order_head_loss = order_ce_loss
                loss = disp_head_loss + order_head_loss
                
                total_loss += loss.item()
                total_disp_loss += disp_head_loss.item()
                total_order_loss += order_head_loss.item()
                
                # Compute accuracy
                _, predicted = torch.max(outputs['order_logits'], 1)
                total_order_correct += (predicted == order_labels).sum().item()
                total_samples += order_labels.size(0)
                
                num_batches += 1
                
            except Exception as e:
                self.logger.error(f"Test batch error: {e}")
                continue
        
        metrics = {
            'test_loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'test_disp_loss': total_disp_loss / num_batches if num_batches > 0 else 0.0,
            'test_order_loss': total_order_loss / num_batches if num_batches > 0 else 0.0,
            'test_order_ce_loss': total_order_ce_loss / num_batches if num_batches > 0 else 0.0,
            'test_order_acc': total_order_correct / total_samples if total_samples > 0 else 0.0
        }
        
        return metrics
    
    def check_early_stopping(self, val_acc):
        """Check early-stopping condition based on accuracy"""
        if (val_acc > self.args.acc_threshold) and (val_acc < 1.0):
            self.acc_stable_count += 1
        else:
            self.acc_stable_count = 0
        
        return self.acc_stable_count >= self.args.patience
    
    def save_checkpoint(self, model, optimizer, epoch, metrics, is_best=False):
        """Persist a training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'args': vars(self.args)
        }
        
        if is_best:
            filename = 'best_model.pth'
        else:
            filename = f'checkpoint_epoch_{epoch}.pth'
        
        filepath = os.path.join(self.args.output_dir, filename)
        torch.save(checkpoint, filepath)
    
    def save_summary(self, stopped_epoch, reason):
        """Write a JSON summary of the run"""
        summary = {
            'data_fraction': self.args.data_fraction,
            'stopped_epoch': stopped_epoch,
            'stop_reason': reason,
            'acc_threshold': self.args.acc_threshold,
            'patience': self.args.patience,
            'best_val_loss': self.best_val_loss,
            'final_val_ce_loss': self.val_ce_losses[-1] if self.val_ce_losses else None,
            'final_val_acc': self.val_accuracies[-1] if self.val_accuracies else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_ce_losses': self.val_ce_losses,
            'val_accuracies': self.val_accuracies,
            'config': vars(self.args),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        summary_path = os.path.join(self.args.output_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Training summary saved: {summary_path}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("="*60)
        self.logger.info("Starting training")
        self.logger.info("="*60)
        
        # Load training/validation data
        train_loader, val_loader = self.load_data()
        if train_loader is None:
            return
        
        # Load test data if provided
        test_loader = self.load_test_data()
        
        # Create model and optimizer stack
        model = self.create_model()
        optimizer = self.create_optimizer(model)
        scheduler = self.create_scheduler(optimizer)
        
        # Loss functions
        mse_loss = nn.MSELoss()
        ce_loss = nn.CrossEntropyLoss()
        
        # Epoch loop
        for epoch in range(1, self.args.max_epochs + 1):
            # Train
            train_metrics = self.train_epoch(model, train_loader, optimizer, mse_loss, ce_loss, epoch)
            self.train_losses.append(train_metrics['train_loss'])
            
            # Validate
            val_metrics = self.validate_epoch(model, val_loader, mse_loss, ce_loss, epoch)
            self.val_losses.append(val_metrics['val_loss'])
            self.val_ce_losses.append(val_metrics['val_order_ce_loss'])
            self.val_accuracies.append(val_metrics['val_order_acc'])
            
            # Evaluate on test set every epoch
            test_metrics = None
            if test_loader:
                test_metrics = self.test_epoch(model, test_loader, mse_loss, ce_loss, epoch)
                self.test_accuracies.append(test_metrics['test_order_acc'])
                self.test_ce_losses.append(test_metrics['test_order_ce_loss'])
            
            # Scheduler step
            if scheduler:
                if self.args.scheduler == 'plateau':
                    scheduler.step(val_metrics['val_loss'])
                else:
                    scheduler.step()
            
            # Save best checkpoint
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(model, optimizer, epoch, val_metrics, is_best=True)
            
            # Compose log line
            current_lr = optimizer.param_groups[0]['lr']
            log_msg = (
                f"Epoch {epoch}/{self.args.max_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.6f} "
                f"(Disp: {train_metrics['train_disp_loss']:.6f}, "
                f"Order: {train_metrics['train_order_loss']:.6f}, "
                f"Acc: {train_metrics['train_order_acc']:.4f}) | "
                f"Val Loss: {val_metrics['val_loss']:.6f} "
                f"(Disp: {val_metrics['val_disp_loss']:.6f}, "
                f"Order: {val_metrics['val_order_loss']:.6f}, "
                f"CE: {val_metrics['val_order_ce_loss']:.6f}, "
                f"Acc: {val_metrics['val_order_acc']:.4f})"
            )
            
            # Append test metrics if available
            if test_metrics:
                log_msg += (
                    f" | Test Acc: {test_metrics['test_order_acc']:.4f}, "
                    f"CE: {test_metrics['test_order_ce_loss']:.6f}"
                )
            
            log_msg += f" | LR: {current_lr:.8f} | Accuracy stability count: {self.acc_stable_count}/{self.args.patience}"
            
            self.logger.info(log_msg)
            
            # Check early-stopping condition
            if self.check_early_stopping(val_metrics['val_order_acc']):
                self.logger.info("="*60)
                self.logger.info(f"Early stopping triggered! Validation accuracy exceeded {self.args.acc_threshold} for {self.args.patience} consecutive epochs")
                self.logger.info(f"Final validation accuracy: {val_metrics['val_order_acc']:.4f}")
                if test_metrics:
                    self.logger.info(f"Final test accuracy: {test_metrics['test_order_acc']:.4f}")
                self.logger.info("="*60)
                self.save_summary(epoch, 'early_stopping_acc_stable')
                break
        else:
            # Hit max epochs
            self.logger.info("="*60)
            self.logger.info(f"Reached max epochs: {self.args.max_epochs}")
            self.logger.info("="*60)
            self.save_summary(self.args.max_epochs, 'max_epochs_reached')
        
        self.logger.info("Training complete!")
        
        # Final evaluation on test set if available
        if test_loader:
            self.logger.info("="*60)
            self.logger.info("Evaluating on test set")
            self.logger.info("="*60)
            test_metrics = self.test_epoch(model, test_loader, mse_loss, ce_loss, epoch)
            self.test_accuracies.append(test_metrics['test_order_acc'])
            self.test_ce_losses.append(test_metrics['test_order_ce_loss'])
            self.logger.info(
                f"Test Loss: {test_metrics['test_loss']:.6f} "
                f"(Disp: {test_metrics['test_disp_loss']:.6f}, "
                f"Order: {test_metrics['test_order_loss']:.6f}, "
                f"CE: {test_metrics['test_order_ce_loss']:.6f}, "
                f"Acc: {test_metrics['test_order_acc']:.4f})"
            )


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='MFN variable dataset-size training script')
    
    # Core parameters
    parser.add_argument('--mode', type=str, default='finetune', choices=['finetune'],
                       help='Training mode (finetune only)')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Dataset root directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory path')
    
    # Dataset usage fraction
    parser.add_argument('--data_fraction', type=float, required=True,
                       help='Fraction of folders to use (0.0 ~ 1.0)')
    
    # Early stopping
    parser.add_argument('--max_epochs', type=int, default=200,
                       help='Maximum training epochs')
    parser.add_argument('--acc_threshold', type=float, default=0.85,
                       help='Accuracy threshold')
    parser.add_argument('--patience', type=int, default=3,
                       help='Patience (number of consecutive epochs above threshold)')
    
    # Data sampling
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training split ratio')
    parser.add_argument('--seq_len_min', type=int, default=5,
                       help='Minimum sequence length')
    parser.add_argument('--seq_len_max', type=int, default=10,
                       help='Maximum sequence length')
    parser.add_argument('--seq_step', type=int, default=1,
                       help='Sequence sampling stride')
    
    # Model configuration
    parser.add_argument('--img_h', type=int, default=1024,
                       help='Input image height')
    parser.add_argument('--img_w', type=int, default=1280,
                       help='Input image width')
    parser.add_argument('--pretrain_checkpoint', type=str, required=True,
                       help='Pretrained checkpoint path')
    parser.add_argument('--freeze_cnn_layers', type=int, default=3,
                       help='Freeze the first N CNN layers')
    
    # Optimization settings
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping value')
    parser.add_argument('--use_orth_loss', action='store_true',
                       help='Enable orthogonality loss')
    
    # Miscellaneous
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data-loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--test_data_root', type=str, default=None,
                       help='Test dataset root directory')
    parser.add_argument('--test_data_fractions', type=str, default=None,
                       help='Comma-separated test partitions (e.g. "1,2,3")')
    
    return parser.parse_args()


def main():
    """Program entry point"""
    args = parse_args()
    
    # Seed RNGs
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Configure logging
    logger = setup_logging(args.output_dir)
    
    # Instantiate trainer
    trainer = VariableDataSizeTrainer(args, logger)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
