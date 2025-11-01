#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upgraded MFN model (MFNversion_2) with a dual-head architecture
File path: /public_new/work_space/fengjiali/MKwithML/MFNversion_2/dual_branch/model_definition.py

Design overview:
1. Image pipeline: CNN branch (three single-channel inputs) + MobileViT branch (three-channel merge)
2. LSTM branch: processes numerical time-series features
3. Dual outputs:
    - Displacement head: regress displacement within [0, λ/2]
    - Order head: classify orders {-2, -1, 0, 1, 2} (five classes)
4. Training strategy:
    - Pretraining: train the displacement head using only image branches
    - Fine-tuning: freeze early MobileViT/CNN layers, train both heads + LSTM, add orthogonal loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import MobileViTV2Config, MobileViTV2Model


# ==================== CNN branch ====================
class ImageChannelCNN(nn.Module):
    """CNN branch operating on a single-channel image"""
    def __init__(self, input_channels=1, output_features=128):
        super(ImageChannelCNN, self).__init__()
        
        self.conv1_out_channels = 16
        self.conv2_out_channels = 32
        self.conv3_out_channels = 64
        self.conv4_out_channels = 128
        self.conv5_out_channels = 256
        
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(input_channels, self.conv1_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.conv1_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(self.conv1_out_channels, self.conv2_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.conv2_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(self.conv2_out_channels, self.conv3_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.conv3_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(self.conv3_out_channels, self.conv4_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.conv4_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(self.conv4_out_channels, self.conv5_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.conv5_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc_input_dim = self.conv5_out_channels * 4 * 4
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, output_features)
        )
    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# ==================== MobileViT branch ====================
class MobileViTImageProcessor(nn.Module):
    """MobileViTV2-based image processor"""
    def __init__(self, output_features=128, image_size=224):
        super(MobileViTImageProcessor, self).__init__()
        
        self.output_features = output_features
        self.image_size = image_size
        
        # Configure the MobileViTV2 model
        self.config = MobileViTV2Config(
            num_channels=3,
            image_size=image_size,
            patch_size=2,
            hidden_sizes=[64, 80, 96, 128, 160, 640],
            neck_hidden_sizes=[16, 16, 24, 24, 48, 64],
            num_attention_heads=4,
            mlp_ratio=2.0,
            expand_ratio=2.0,
            hidden_act="swish",
            conv_kernel_size=3,
            output_stride=32,
            classifier_dropout_prob=0.1,
            use_layer_norm=True,
            layer_norm_eps=1e-5
        )
        
        # Create the MobileViTV2 backbone
        self.mobilevit_backbone = MobileViTV2Model(self.config)
        
        # Feature adapter mapping backbone outputs to target dimension
        backbone_output_dim = self.config.aspp_out_channels
        
        self.feature_adapter = nn.Sequential(
            nn.Linear(backbone_output_dim, output_features * 2),
            nn.BatchNorm1d(output_features * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(output_features * 2, output_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        # Input preprocessing layer
        self.input_resizer = nn.AdaptiveAvgPool2d((image_size, image_size))
    
    def forward(self, x):
        # Resize the incoming image tensor
        x = self.input_resizer(x)
        
        # Forward through the MobileViTV2 backbone
        outputs = self.mobilevit_backbone(pixel_values=x)
        
        # Extract feature embeddings
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            cls_feats = outputs.pooler_output
        else:
            cls_feats = outputs.last_hidden_state[:, 0, :]
        
        # Handle the batch_size=1 case safely
        if cls_feats.size(0) == 1:
            bn_modules = [m for m in self.feature_adapter.modules() if isinstance(m, nn.BatchNorm1d)]
            bn_states = [m.training for m in bn_modules]
            for m in bn_modules:
                m.eval()
            try:
                features = self.feature_adapter(cls_feats)
            finally:
                for m, s in zip(bn_modules, bn_states):
                    m.train(s)
        else:
            features = self.feature_adapter(cls_feats)
        
        return features


# ==================== LSTM Branch ====================
class NumericalChannelLSTM(nn.Module):
    """LSTM branch for numerical features"""
    def __init__(self, input_features=7, lstm_hidden_size=128, output_features=128):
        super(NumericalChannelLSTM, self).__init__()
        
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = 1
        self.dropout_rate = 0.3
        
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            dropout=0 if self.lstm_num_layers == 1 else self.dropout_rate
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, self.lstm_hidden_size // 2),
            nn.BatchNorm1d(self.lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            
            nn.Linear(self.lstm_hidden_size // 2, output_features),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
    
    def forward(self, x, lengths=None):
        if lengths is not None:
            lengths_cpu = lengths.cpu() if lengths.is_cuda else lengths
            packed_x = torch.nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_lstm_out, (h_n, c_n) = self.lstm(packed_x)
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_lstm_out, batch_first=True)
            
            batch_size = x.size(0)
            last_time_step_outs = []
            for i in range(batch_size):
                seq_len = lengths_cpu[i].item()
                last_time_step_outs.append(lstm_out[i, seq_len - 1])
            last_time_step_out = torch.stack(last_time_step_outs, dim=0)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)
            last_time_step_out = lstm_out[:, -1, :]
        
        x = self.fc_layers(last_time_step_out)
        return x


# ==================== MFN Dual-Head Model ====================
class MFNDualHeadModel(nn.Module):
    """
    Upgraded MFN dual-head model.

    Architecture:
    - Image feature extraction: CNN (three single-channel inputs) + MobileViT (three-channel input)
    - LSTM branch: temporal numerical feature encoder
    - Displacement head: regress [0, λ/2] displacement using image features only
    - Order head: classify {-2, -1, 0, 1, 2} using both image and LSTM features
    """
    def __init__(self,
                 img_h=1024,
                 img_w=1280,
                 cnn_branch_features=128,
                 mobilevit_features=128,
                 numerical_input_features=7,
                 numerical_lstm_hidden_size=128,
                 numerical_output_features=128,
                 displacement_head_hidden=[256, 128, 64],
                 order_head_hidden=[256, 128, 64],
                 num_order_classes=5,  # {-2, -1, 0, 1, 2}
                 use_orthogonal_loss=True):
        """
        Args:
            img_h, img_w: Input image dimensions.
            cnn_branch_features: Output dimension for each CNN branch.
            mobilevit_features: Output dimension for the MobileViT branch.
            numerical_input_features: Dimensionality of numerical features.
            numerical_lstm_hidden_size: Hidden size for the numerical LSTM.
            numerical_output_features: Output dimension of the LSTM branch.
            displacement_head_hidden: Hidden layer dimensions for the displacement head.
            order_head_hidden: Hidden layer dimensions for the order head.
            num_order_classes: Number of order categories (5 for {-2, -1, 0, 1, 2}).
            use_orthogonal_loss: Whether to compute the orthogonality loss.
        """
        super(MFNDualHeadModel, self).__init__()
        
        self.use_orthogonal_loss = use_orthogonal_loss
        self.num_order_classes = num_order_classes
        
        # === Image feature extraction ===
        # CNN branches (three single-channel inputs)
        self.cnn_branch_ch1 = ImageChannelCNN(input_channels=1, output_features=cnn_branch_features)
        self.cnn_branch_ch2 = ImageChannelCNN(input_channels=1, output_features=cnn_branch_features)
        self.cnn_branch_ch3 = ImageChannelCNN(input_channels=1, output_features=cnn_branch_features)
        self.cnn_total_features = cnn_branch_features * 3
        
        # MobileViT branch (three-channel input)
        self.mobilevit_processor = MobileViTImageProcessor(
            output_features=mobilevit_features,
            image_size=224
        )
        self.mobilevit_features = mobilevit_features
        
        # Combined image feature dimensionality
        self.image_features_dim = self.cnn_total_features + self.mobilevit_features
        
        # === LSTM branch ===
        self.lstm_branch = NumericalChannelLSTM(
            input_features=numerical_input_features,
            lstm_hidden_size=numerical_lstm_hidden_size,
            output_features=numerical_output_features
        )
        self.numerical_features = numerical_output_features
        
        # === Displacement head ===
        # Uses image features only (no LSTM features)
        displacement_layers = []
        in_dim = self.image_features_dim
        
        for hidden_dim in displacement_head_hidden:
            displacement_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            in_dim = hidden_dim
        
        displacement_layers.append(nn.Linear(in_dim, 1))
        self.displacement_head = nn.Sequential(*displacement_layers)
        
        # Track displacement head hidden dimension for orthogonality loss
        self.displacement_hidden_dim = displacement_head_hidden[-1]
        
        # === Order head ===
        # Uses concatenated image and LSTM features
        order_input_dim = self.image_features_dim + self.numerical_features
        order_layers = []
        in_dim = order_input_dim
        
        for hidden_dim in order_head_hidden:
            order_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            in_dim = hidden_dim
        
        order_layers.append(nn.Linear(in_dim, num_order_classes))  # Five classes represent orders {-2, -1, 0, 1, 2}
        self.order_head = nn.Sequential(*order_layers)
        
        # Track order head hidden dimension for orthogonality loss
        self.order_hidden_dim = order_head_hidden[-1]
    
    def extract_image_features(self, img_ch1, img_ch2, img_ch3, img_3channel_mobilevit):
        """Extract fused image features"""
        # CNN features
        features_cnn_ch1 = self.cnn_branch_ch1(img_ch1)
        features_cnn_ch2 = self.cnn_branch_ch2(img_ch2)
        features_cnn_ch3 = self.cnn_branch_ch3(img_ch3)
        features_cnn = torch.cat([features_cnn_ch1, features_cnn_ch2, features_cnn_ch3], dim=1)
        
        # MobileViT features
        features_mobilevit = self.mobilevit_processor(img_3channel_mobilevit)
        
        # Concatenate all image features
        image_features = torch.cat([features_cnn, features_mobilevit], dim=1)
        
        return image_features
    
    def forward(self, img_ch1, img_ch2, img_ch3, img_3channel_mobilevit, 
                numerical_seq=None, lengths=None, mode='full'):
        """
        Forward pass for the dual-head model.

        Args:
            img_ch1, img_ch2, img_ch3: Single-channel images (B, 1, H, W).
            img_3channel_mobilevit: Three-channel image input (B, 3, 224, 224).
            numerical_seq: Numerical sequences (B, seq_len, D) or None.
            lengths: Sequence lengths (B,) or None.
            mode: 'pretrain' (displacement head only), 'finetune' (dual heads + LSTM), or 'full' (default inference).

        Returns:
            Output dictionary or tensor depending on the selected mode.
        """
        # Extract image features
        image_features = self.extract_image_features(img_ch1, img_ch2, img_ch3, img_3channel_mobilevit)
        
        # Predict displacement
        displacement_pred = self.displacement_head(image_features)
        
        if mode == 'pretrain':
            # Pretraining returns only the displacement estimate
            return displacement_pred.squeeze(-1)
        
        # Extract LSTM features when numerical data is provided
        if numerical_seq is not None:
            lstm_features = self.lstm_branch(numerical_seq, lengths)
        else:
            lstm_features = torch.zeros(image_features.size(0), self.numerical_features, device=image_features.device)
        
        # Concatenate image and LSTM features
        combined_features = torch.cat([image_features, lstm_features], dim=1)
        
        # Predict order logits
        order_logits = self.order_head(combined_features)
        
        if mode == 'finetune' or mode == 'full':
            # Fine-tuning / full mode returns both displacement and order outputs
            outputs = {
                'displacement': displacement_pred.squeeze(-1),
                'order_logits': order_logits
            }
            
            # Optionally capture hidden representations for orthogonality loss
            if self.use_orthogonal_loss and self.training:
                # Collect the penultimate displacement head features
                disp_hidden = image_features
                for layer in self.displacement_head[:-1]:
                    disp_hidden = layer(disp_hidden)
                
                # Collect the penultimate order head features
                order_hidden = combined_features
                for layer in self.order_head[:-1]:
                    order_hidden = layer(order_hidden)
                
                outputs['displacement_hidden'] = disp_hidden
                outputs['order_hidden'] = order_hidden
            
            return outputs
    
    def freeze_mobilevit(self):
        """Freeze MobileViT backbone parameters"""
        for param in self.mobilevit_processor.mobilevit_backbone.parameters():
            param.requires_grad = False
        print("[Freeze] MobileViT backbone frozen")
    
    def freeze_cnn_early_layers(self, num_layers=3):
        """Freeze the first N convolutional layers in each CNN branch"""
        for cnn_branch in [self.cnn_branch_ch1, self.cnn_branch_ch2, self.cnn_branch_ch3]:
            layer_count = 0
            for module in cnn_branch.cnn_layers:
                if isinstance(module, nn.Conv2d):
                    layer_count += 1
                    if layer_count <= num_layers:
                        for param in module.parameters():
                            param.requires_grad = False
        print(f"[Freeze] Frozen the first {num_layers} CNN layers")
    
    def unfreeze_all(self):
        """Unfreeze every parameter in the model"""
        for param in self.parameters():
            param.requires_grad = True
        print("[Unfreeze] All parameters are trainable again")


# ==================== Helper functions ====================
def create_mfn_model(img_h=1024, img_w=1280, **kwargs):
    """Factory helper that instantiates an MFN model"""
    model = MFNDualHeadModel(img_h=img_h, img_w=img_w, **kwargs)
    return model


def orthogonal_loss(h_disp, h_order):
    """
    Compute the orthogonal loss between two hidden representations.
    L_orth = ||h_disp^T @ h_order||^2

    Args:
        h_disp: Displacement head hidden representation (B, D1).
        h_order: Order head hidden representation (B, D2).

    Returns:
        Scalar orthogonality penalty.
    """
    # Normalize
    h_disp_norm = F.normalize(h_disp, p=2, dim=1)
    h_order_norm = F.normalize(h_order, p=2, dim=1)
    
    # Compute squared inner products
    inner_product = torch.sum(h_disp_norm * h_order_norm, dim=1)
    loss = torch.mean(inner_product ** 2)
    
    return loss


if __name__ == "__main__":
    """Quick smoke test for the MFN model"""
    print("=== Test MFN model ===")
    
    # Instantiate model
    model = create_mfn_model()
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Order class count: {model.num_order_classes}")
    
    # Create dummy inputs
    batch_size = 4
    img_ch1 = torch.randn(batch_size, 1, 1024, 1280)
    img_ch2 = torch.randn(batch_size, 1, 1024, 1280)
    img_ch3 = torch.randn(batch_size, 1, 1024, 1280)
    img_3channel = torch.randn(batch_size, 3, 224, 224)
    numerical_seq = torch.randn(batch_size, 10, 7)
    lengths = torch.tensor([10, 8, 10, 7])
    
    # Pretraining mode
    print("\n=== Test pretrain mode ===")
    model.train()
    displacement = model(img_ch1, img_ch2, img_ch3, img_3channel, mode='pretrain')
    print(f"Displacement output shape: {displacement.shape}")
    
    # Fine-tuning mode
    print("\n=== Test finetune mode ===")
    outputs = model(img_ch1, img_ch2, img_ch3, img_3channel, 
                   numerical_seq, lengths, mode='finetune')
    print(f"Displacement shape: {outputs['displacement'].shape}")
    print(f"Order logits shape: {outputs['order_logits'].shape}")
    print(f"Order classes: {model.num_order_classes} (represents -2, -1, 0, 1, 2)")
    
    if 'displacement_hidden' in outputs:
        print(f"Displacement hidden shape: {outputs['displacement_hidden'].shape}")
        print(f"Order hidden shape: {outputs['order_hidden'].shape}")
        
        # Evaluate orthogonal loss
        orth_loss = orthogonal_loss(outputs['displacement_hidden'], outputs['order_hidden'])
        print(f"Orthogonal loss: {orth_loss.item():.6f}")
    
    print("\n=== Test freezing helpers ===")
    model.freeze_mobilevit()
    model.freeze_cnn_early_layers(num_layers=3)
    print(f"Trainable parameters after freezing: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print("\n=== Model test complete ===")
