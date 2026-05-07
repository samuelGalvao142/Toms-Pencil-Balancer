# =============================================================================
# SNN Implementation for real time control of mechanical systems from image
# processing and event cameras
# =============================================================================
# Disclaimer: The architecture is based on code provided by Geronimo Marin 
# Hurtado, available at:
# https://github.com/Geronimo9177/snn-event-regression
# Most recent access: Feb 17, 2026
# =============================================================================
# This pipeline will have the purpose of processing and predicting the tilt
# angle of a pencil on top of a pencil balancing robot, from feedback provided
# by two Davis346 event cameras.
# Adaptations and real time implementation by Samuel Galvao, supervised by
# Professor Takashi Tanaka.
# =============================================================================

import torch
import dv_processing as dv
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, functional

class SNN_Net(nn.Module):
    """
    Complete Spiking Neural Network for event-based regression.
    
    This network processes event camera frames through a series of convolutional
    blocks followed by fully connected layers to predict a continuous angle value.
    
    Architecture:
        Input [B, 2, 346, 260] (ON/OFF event frames)
          ↓
        Convolutional Blocks (with normalization + LIF neurons)
          ↓
        Adaptive Pooling (progressive downsampling)
          ↓
        Flatten
          ↓
        FC Layer + LIF
          ↓
        Output FC + LIF (infinite threshold for regression)
          ↓
        Output: Membrane potential (continuous value)
    
    Note:
        For temporal sequences, call forward() for each timestep T.
        Remember to reset neuron states between sequences with functional.reset_net(model).
    """
    def __init__(self, tau=2.0, final_tau=20.0, layer_list=None, hidden=256, v_reset=0.0,
                 surrogate_function=surrogate.ATan(), connect_f="ADD", Plif=False, 
                 norm_type="BN", learnable_norm=True, init_scale=5.0):
        
        if layer_list is None:
            if block_type.lower() == "sew":
                layer_list = layer_list_sew
            elif block_type.lower() == "plain":
                layer_list = layer_list_plain
            elif block_type.lower() == "spiking":
                layer_list = layer_list_spiking
            else:
                raise ValueError("Invalid block type")

        super().__init__()

        self.norm_type = norm_type
        
        # ====================================================================
        # Build sequential convolution pipeline
        # ====================================================================
        conv_blocks = []
        in_channels = 2  # Start with 2 channels (ON/OFF events)

        for cfg in layer_list:
            channels = cfg["channels"]
            mid_channels = cfg["mid_channels"]

            # Channel adjustment layer (if needed)
            if in_channels != channels:
                if cfg["up_kernel_size"] == 3:
                    conv_blocks.append(conv3x3(in_channels, channels, norm_type=norm_type, learnable=learnable_norm, init_scale=init_scale))
                elif cfg["up_kernel_size"] == 1:
                    conv_blocks.append(conv1x1(in_channels, channels, norm_type=norm_type, learnable=learnable_norm, init_scale=init_scale))
                else:
                    raise NotImplementedError

                # Add LIF after channel adjustment
                conv_blocks.append(
                    neuron.ParametricLIFNode(init_tau=tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=True)
                    if Plif else neuron.LIFNode(tau=tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=True)
                )
                in_channels = channels

            # Add building blocks
            for _ in range(cfg["num_blocks"]):
                if cfg["block_type"] == "sew":
                    conv_blocks.append(
                        SEWBlock(in_channels, mid_channels, tau=tau, Plif=Plif, v_reset=v_reset,
                                stride1=cfg["stride_1"], stride2=cfg["stride_2"], connect_f=connect_f,  
                                norm_type=norm_type, learnable_norm=learnable_norm, init_scale=init_scale)
                    )
                elif cfg["block_type"] == "plain":
                    conv_blocks.append(
                        PlainBlock(in_channels, mid_channels, tau=tau, Plif=Plif, v_reset=v_reset,
                                  stride1=cfg["stride_1"], stride2=cfg["stride_2"],
                                  norm_type=norm_type, learnable_norm=learnable_norm, init_scale=init_scale)
                    )
                elif cfg["block_type"] == "spiking":
                    conv_blocks.append(
                        SpikingBlock(in_channels, mid_channels, tau=tau, Plif=Plif, v_reset=v_reset,
                                    stride1=cfg["stride_1"], stride2=cfg["stride_2"],
                                    norm_type=norm_type, learnable_norm=learnable_norm, init_scale=init_scale)
                    )
                else:
                    raise NotImplementedError

            # Add MaxPool (if specified)
            if "k_pool" in cfg and cfg["k_pool"] > 1:
                conv_blocks.append(nn.MaxPool2d(kernel_size=cfg["k_pool"]))

        self.conv = nn.ModuleList(conv_blocks)

        # ====================================================================
        # Automatic computation of flattened feature dimension
        # ====================================================================
        # Run a dummy forward pass to determine the output size
        with torch.no_grad():
            dummy = torch.zeros(1, 2, 346, 260)  # DAVIS346 input size
            for m in self.conv:
                dummy = m(dummy)
            flat_dim = dummy.numel()

        # ====================================================================
        # Fully Connected Layers
        # ====================================================================
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flat_dim, hidden, bias=False)
        
        # Hidden LIF neurons
        self.lif_hidden = (neuron.ParametricLIFNode(init_tau=tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=True) 
                          if Plif else neuron.LIFNode(tau=tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=True))
        
        # Output layer
        self.fc_out = nn.Linear(hidden, 1, bias=False)
        
        # Output LIF with INFINITE threshold (for regression)
        # This neuron never spikes - we read its membrane potential as the prediction
        self.lif_out = neuron.LIFNode(tau=final_tau, v_threshold=float('inf'), 
                                      surrogate_function=surrogate_function, detach_reset=True)

    def detach(self):
        """
        Detach membrane potentials of all LIF neurons.
        
        Used in TBPTT (Truncated Backpropagation Through Time) to truncate
        gradients and prevent backpropagation through the entire sequence.
        """
        for m in self.modules():
            if isinstance(m, neuron.BaseNode):
                m.v.detach_()

    # ========================================================================
    # SPIKE ACTIVITY MONITORING
    # ========================================================================
    def _register_spike_hooks(self):
        """Register forward hooks on all LIF layers to record spike activity."""
        self.spike_record = {}
        self.hooks = []

        for name, module in self.named_modules():
            if isinstance(module, neuron.BaseNode):   # LIF or PLIF neurons
                hook = module.register_forward_hook(self._make_spike_hook(name))
                self.hooks.append(hook)

    def _make_spike_hook(self, name):
        """Create a hook function that captures spike output."""
        def hook(module, inp, out):
            # Store spikes for this layer 
            self.spike_record[name] = out.detach().clone()
        return hook

    def enable_spike_recording(self):
        """Enable spike activity recording for all LIF neurons."""
        self._register_spike_hooks()

    def disable_spike_recording(self):
        """Disable spike recording and clean up hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.spike_record = {}

    # ========================================================================
    # NORMALIZATION STATISTICS MONITORING
    # ========================================================================
    def _register_norm_hooks(self):
        """Register hooks on all normalization layers to capture statistics."""
        self.norm_stats = {}
        self.norm_hooks = []

        # Determine which normalization type to monitor
        if self.norm_type == "BN":
            module_type = nn.BatchNorm2d
        elif self.norm_type == "RMS":
            module_type = RMSNorm2d
        else:
            print(f"No normalization monitoring available for norm_type={self.norm_type}")
            return

        norm_counter = 0
        for name, module in self.named_modules():
            if isinstance(module, module_type):
                hook = module.register_forward_hook(
                    self._make_norm_hook(f"{self.norm_type}_{norm_counter}_{name}")
                )
                self.norm_hooks.append(hook)
                norm_counter += 1

    def _make_norm_hook(self, layer_name):
        """Create a hook that captures normalization input/output statistics."""
        def hook(module, inp, out):
            input_tensor = inp[0].detach()
            output_tensor = out.detach()

            # Compute per-channel statistics
            # Input/output shape: [B, C, H, W]
            B, C, H, W = input_tensor.shape

            # Flatten spatial dimensions: [B, C, H*W]
            input_flat = input_tensor.view(B, C, -1)
            output_flat = output_tensor.view(B, C, -1)

            # Store comprehensive statistics
            self.norm_stats[layer_name] = {
                # INPUT statistics (before normalization)
                'input_mean_per_channel': input_flat.mean(dim=(0, 2)).cpu().numpy(),  # [C]
                'input_std_per_channel': input_flat.std(dim=(0, 2)).cpu().numpy(),
                'input_min_per_channel': input_flat.min(dim=2)[0].mean(dim=0).cpu().numpy(),
                'input_max_per_channel': input_flat.max(dim=2)[0].mean(dim=0).cpu().numpy(),

                # OUTPUT statistics (after normalization)
                'output_mean_per_channel': output_flat.mean(dim=(0, 2)).cpu().numpy(),
                'output_std_per_channel': output_flat.std(dim=(0, 2)).cpu().numpy(),
                'output_min_per_channel': output_flat.min(dim=2)[0].mean(dim=0).cpu().numpy(),
                'output_max_per_channel': output_flat.max(dim=2)[0].mean(dim=0).cpu().numpy(),
            }
        return hook

    def enable_norm_monitoring(self):
        """Enable normalization statistics monitoring."""
        self._register_norm_hooks()

    def disable_norm_monitoring(self):
        """Disable normalization monitoring and clean up hooks."""
        for hook in self.norm_hooks:
            hook.remove()
        self.norm_hooks = []
        self.norm_stats = {}

    def get_norm_stats(self):
        """Return captured normalization statistics."""
        return self.norm_stats

    # ========================================================================
    # FORWARD PASS
    # ========================================================================
    def forward(self, x):
        """
        Forward pass for a single timestep.
        Note:
            This processes ONE timestep. For temporal sequences, call this
            method T times (once per frame) without resetting neuron states.
        """
        out = x

        # Pass through convolutional blocks
        for m in self.conv:
            out = m(out)

        # Fully connected layers
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.lif_hidden(out)

        # Output layer (LIF with infinite threshold)
        out = self.fc_out(out)
        _ = self.lif_out(out)

        # Return membrane potential (continuous value for regression)
        return self.lif_out.v 
    
# ============================================================================
# SEW (Spiking Element-Wise) Architecture
# ============================================================================
layer_list_sew = [
    # STEM: Early aggressive downsampling to reduce computational cost
    {
        'channels': 16,          # Output channels
        'up_kernel_size': 3,     # 3x3 conv for initial feature extraction
        'mid_channels': 16,      # Intermediate channels in block
        'num_blocks': 1,         # Single block
        'block_type': 'sew',     # SEW block type
        'k_pool': 2,             # 2x2 MaxPool → /2 spatial reduction
        'stride_1': 2,           # First conv: stride 2 → /2 reduction
        'stride_2': 1            # Second conv: stride 1 → preserve size
    },

    # LAYER 1: Maintain resolution, basic feature extraction
    {
        'channels': 16, 
        'up_kernel_size': 1,     # 1x1 conv (efficient channel adjustment)
        'mid_channels': 16,
        'num_blocks': 1,
        'block_type': 'sew',
        'k_pool': 1,             # No pooling
        'stride_1': 1,
        'stride_2': 1
    },

    # LAYER 2: Increase capacity + downsample
    {
        'channels': 24, 
        'up_kernel_size': 1,
        'mid_channels': 24,
        'num_blocks': 1,
        'block_type': 'sew',
        'k_pool': 2,             # /2 spatial reduction
        'stride_1': 1,
        'stride_2': 1
    },

    # LAYER 3: Further increase capacity + downsample
    {
        'channels': 32, 
        'up_kernel_size': 1,
        'mid_channels': 32,
        'num_blocks': 1,
        'block_type': 'sew',
        'k_pool': 2,             # /2 spatial reduction
        'stride_1': 1,
        'stride_2': 1
    },
]

# ============================================================================
# Plain Architecture (No Residual Connections)
# ============================================================================
layer_list_plain = [
    # STEM: Early downsampling
    {
        'channels': 16, 
        'up_kernel_size': 3,
        'mid_channels': 16,
        'num_blocks': 1,
        'block_type': 'plain',
        'k_pool': 2,
        'stride_1': 2,
        'stride_2': 1
    },

    # LAYER 1: Minimal expansion
    {
        'channels': 16, 
        'up_kernel_size': 1,
        'mid_channels': 16,
        'num_blocks': 1,
        'block_type': 'plain',
        'k_pool': 1,
        'stride_1': 1,
        'stride_2': 1
    },

    # LAYER 2: Modest capacity increase
    {
        'channels': 24, 
        'up_kernel_size': 1,
        'mid_channels': 24,
        'num_blocks': 1,
        'block_type': 'plain',
        'k_pool': 2,
        'stride_1': 1,
        'stride_2': 1
    },

    # LAYER 3: Final expansion
    {
        'channels': 32, 
        'up_kernel_size': 1,
        'mid_channels': 32,
        'num_blocks': 1,
        'block_type': 'plain',
        'k_pool': 2,
        'stride_1': 1,
        'stride_2': 1
    },
]

# ============================================================================
# Spiking ResNet Architecture
# ============================================================================
layer_list_spiking = [
    # STEM: Strong early downsampling
    {
        'channels': 16, 
        'up_kernel_size': 3,
        'mid_channels': 16,
        'num_blocks': 1,
        'block_type': 'spiking',
        'k_pool': 2,
        'stride_1': 2,
        'stride_2': 1
    },

    # LAYER 1: Basic blocks
    {
        'channels': 16, 
        'up_kernel_size': 1,
        'mid_channels': 16,
        'num_blocks': 1,
        'block_type': 'spiking',
        'k_pool': 1,
        'stride_1': 1,
        'stride_2': 1
    },

    # LAYER 2: Gentle downsampling
    {
        'channels': 24, 
        'up_kernel_size': 1,
        'mid_channels': 24,
        'num_blocks': 1,
        'block_type': 'spiking',
        'k_pool': 2,
        'stride_1': 1,
        'stride_2': 1
    },

    # LAYER 3: Final expansion
    {
        'channels': 32, 
        'up_kernel_size': 1,
        'mid_channels': 32,
        'num_blocks': 1,
        'block_type': 'spiking',
        'k_pool': 2,
        'stride_1': 1,
        'stride_2': 1
    },
]

# ============================================================================
# Training Configuration Dictionary
# ============================================================================
# This dictionary contains all hyperparameters for complete reproducibility
CONFIG = {
# Model architecture
"block_type" : 'SEW',
"hidden" : 256,
"tau" : 2.0,
"final_tau" : 20.0,
"reset_type" : 'soft',
"norm_type" : 'BN',
"learnable_norm" : True,
"init_scale" : 5.0,
}

# ============================================================================
# Convolution Helpers
# ============================================================================
def conv3x3(in_ch, out_ch, stride=1, norm_type="BN", learnable=True, init_scale=5.0):
    """
    3x3 convolution with normalization.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        get_norm_layer(norm_type, out_ch, learnable=learnable, init_scale=init_scale)
    )

def conv1x1(in_ch, out_ch, stride=1, norm_type="BN", learnable=True, init_scale=5.0):
    """
    1x1 convolution with normalization.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
        get_norm_layer(norm_type, out_ch, learnable=learnable, init_scale=init_scale)
    )

# ============================================================================
# Normalization Factory Function
# ============================================================================
def get_norm_layer(norm_type: str, num_features: int, learnable: bool = True, eps: float = 1e-5, init_scale: float = 5.0
):
    """
    Factory function to create normalization layers.
    """
    if norm_type == "BN":
        # Standard BatchNorm2d
        return nn.BatchNorm2d(num_features, affine=learnable, eps=eps)

    elif norm_type == "RMS":
        # RMS Normalization
        return RMSNorm2d(num_features, eps=eps, affine=learnable)

    elif norm_type == "MUL":
        # Simple scaling
        return MultiplyBy(init_scale, learnable=learnable)

    elif norm_type is None:
        # No normalization
        return nn.Identity()

    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")

# ============================================================================
# Plain Block (No Residual Connection)
# ============================================================================
class PlainBlock(nn.Module):
    """
    Plain convolutional block without residual connections.
    
    Architecture:
        Conv3x3 → Norm → LIF → Conv3x3 → Norm → LIF
    """
    def __init__(self, in_channels, mid_channels, tau=2.0, Plif=False, v_reset=0.0,
                 surrogate_function=surrogate.ATan(), stride1=1, stride2=1,
                 norm_type="BN", learnable_norm=True, init_scale=5.0):
        super(PlainBlock, self).__init__()

        self.conv = nn.Sequential(
            # First conv-norm-LIF
            conv3x3(in_channels, mid_channels, stride=stride1, norm_type=norm_type, learnable=learnable_norm, init_scale=init_scale),
            neuron.ParametricLIFNode(init_tau=tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=True)
            if Plif else neuron.LIFNode(tau=tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=True),

            # Second conv-norm-LIF
            conv3x3(mid_channels, in_channels, stride=stride2, norm_type=norm_type, learnable=learnable_norm, init_scale=init_scale),
            neuron.ParametricLIFNode(init_tau=tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=True)
            if Plif else neuron.LIFNode(tau=tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=True)
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)

# ============================================================================
# Spiking ResNet Block
# ============================================================================
class SpikingBlock(nn.Module):
    """
    Spiking ResNet block with residual connection.
    
    Architecture:
        x → Conv3x3 → Norm → LIF → Conv3x3 → Norm → (+) → LIF → out
        └──────────────── downsample ────────────────┘
    """
    def __init__(self, in_channels, mid_channels, tau=2.0, Plif=False, v_reset=0.0,
                 surrogate_function=surrogate.ATan(), stride1=1, stride2=1,
                 norm_type="BN", learnable_norm=True, init_scale=5.0):
        super(SpikingBlock, self).__init__()

        # Main path
        self.conv = nn.Sequential(
            conv3x3(in_channels, mid_channels, stride=stride1, norm_type=norm_type, learnable=learnable_norm, init_scale=init_scale),
            neuron.ParametricLIFNode(init_tau=tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=True)
            if Plif else neuron.LIFNode(tau=tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=True),

            nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1, stride=stride2, bias=False),
            get_norm_layer(norm_type=norm_type, num_features=in_channels, learnable=learnable_norm, init_scale=init_scale)
        )

        # Output LIF (after addition)
        self.sn = neuron.ParametricLIFNode(init_tau=tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=True) \
                  if Plif else neuron.LIFNode(tau=tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=True)

        # Shortcut connection (downsample if needed)
        if stride1 == 2:
            self.downsample = conv1x1(in_channels, in_channels, stride=2, norm_type=norm_type, learnable=learnable_norm, init_scale=init_scale)
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor):
        # Residual connection: x + conv(x)
        return self.sn(self.downsample(x) + self.conv(x))

# ============================================================================
# SEW (Spiking Element-Wise) Block
# ============================================================================
class SEWBlock(nn.Module):
    """
    Spiking Element-Wise residual block with flexible connection functions.
    
    Architecture:
        x → Conv3x3 → Norm → LIF → Conv3x3 → Norm → LIF → (+) → out
        └──────────────── downsample → LIF ────────────────┘
    """
    def __init__(self, in_channels, mid_channels, tau=2.0, Plif=False, v_reset=0.0,
                 surrogate_function=surrogate.ATan(), stride1=1, stride2=1,
                 connect_f="ADD", norm_type="BN", learnable_norm=True, init_scale=5.0):
        super().__init__()

        self.connect_f = connect_f

        # First conv-norm-LIF
        self.conv1 = conv3x3(in_channels, mid_channels, stride=stride1, norm_type=norm_type, learnable=learnable_norm, init_scale=init_scale)
        self.lif1 = neuron.ParametricLIFNode(init_tau=tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=True) \
                    if Plif else neuron.LIFNode(tau=tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=True)

        # Second conv-norm-LIF
        self.conv2 = conv3x3(mid_channels, in_channels, stride=stride2, norm_type=norm_type, learnable=learnable_norm, init_scale=init_scale)
        self.lif2 = neuron.ParametricLIFNode(init_tau=tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=True) \
                    if Plif else neuron.LIFNode(tau=tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=True)

        # Shortcut (with LIF)
        if stride1 == 2:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, in_channels, stride=2, norm_type=norm_type, learnable=learnable_norm, init_scale=init_scale),
                neuron.ParametricLIFNode(init_tau=tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=True)
                if Plif else neuron.LIFNode(tau=tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=True)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        # Shortcut path (produces spikes)
        identity = self.downsample(x)

        # Main path
        spk1 = self.lif1(self.conv1(x))
        spk2 = self.lif2(self.conv2(spk1))

        # Apply connection function
        if self.connect_f == "ADD":
            return spk2 + identity
        elif self.connect_f == "AND":
            return spk2 * identity 
        elif self.connect_f == "IAND":
            return identity * (1 - spk2) 
        else:
            raise NotImplementedError

block_type = "SEW"

# ============================================================================
# Parameters for model definition
# ============================================================================
model = SNN_Net(
    tau=CONFIG["tau"],
    final_tau=CONFIG["final_tau"],
    hidden=CONFIG["hidden"],
    norm_type=CONFIG["norm_type"],
    learnable_norm=CONFIG["learnable_norm"],
    init_scale=CONFIG["init_scale"]
)