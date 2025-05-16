import torch.nn as nn
from .basic_module import BasicModule
# from .AttentionModule import AttentionModule # Not used in the active unet class
from .unet_parts import DoubleConv, Down, Up, OutConv  # UpNoSkip for the commented variant


class UNet(BasicModule):
    """
    U-Net architecture with modifications for potentially smaller feature maps
    or reduced computational complexity.

    This implementation allows for bilinear upsampling or transposed convolutions.
    It features a configurable number of input channels and output classes.
    The "shared_layers" are part of the encoder path.
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 1, bilinear: bool = False):
        """
        Initializes the U-Net model.

        Args:
            n_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).
            n_classes (int): Number of output classes (e.g., 1 for binary segmentation or regression).
            bilinear (bool): If True, use bilinear upsampling. Otherwise, use transposed convolutions.
        """
        super(UNet, self).__init__()
        self.model_name = 'unet'  # Or a more specific name like 'unet_SDBU'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder (Downsampling path)
        # The paper mentions SDBU enhances a U-Net.
        # The "shared_layers" here seems to be a portion of the encoder.
        # The original U-Net typically has more layers. This is a reduced version.
        # Initial convolution block
        self.inc = DoubleConv(n_channels, 32)
        # Downsampling blocks
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        # self.down3 = Down(128, 256) # Example if more depth is needed
        # self.down4 = Down(256, 512 // factor) # Example for deeper U-Net

        # Decoder (Upsampling path)
        # The 'factor' is used if bilinear=True, as Up typically halves channels before concat if not bilinear.
        # However, the provided unet_parts.Up might handle this internally.
        # The channel numbers in Up(in_ch, out_ch) depend on the concatenation with skip connections.
        # For Up(128, 64, ...): in_ch = 128 (from previous layer) + 64 (from skip connection x2)
        # This means the Up class needs to handle the halving of channels of the upsampled feature map
        # to match the skip connection size, or the output of Down needs to be adjusted.
        # Let's assume unet_parts.Up handles this correctly:
        # Up(channels_from_lower_layer, channels_from_skip_connection_output, bilinear)

        # self.up1 = Up(512, 256 // factor, bilinear) # Example for deeper U-Net
        # self.up2 = Up(256, 128 // factor, bilinear) # Example for deeper U-Net
        self.up1 = Up(128, 64, bilinear)  # Input to Up is 128 (from down2) + 64 (from x2)
        self.up2 = Up(64, 32, bilinear)  # Input to Up is 64 (from up1) + 32 (from x1)

        # Final output convolution
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        """
        Forward pass of the U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H, W).
        """
        # Encoder
        x1 = self.inc(x)  # Output channels: 32
        x2 = self.down1(x1)  # Output channels: 64
        x3 = self.down2(x2)  # Output channels: 128
        # x4 = self.down3(x3) # Example for deeper U-Net
        # x5 = self.down4(x4) # Example for deeper U-Net

        # Decoder with skip connections
        # x = self.up1(x5, x4) # Example for deeper U-Net
        # x = self.up2(x, x3)  # Example for deeper U-Net
        x = self.up1(x3, x2)  # x2 is the skip connection (64 channels)
        x = self.up2(x, x1)  # x1 is the skip connection (32 channels)

        logits = self.outc(x)

        # Optional: print shape for debugging during development
        # print(f"Input shape: {x.shape}")
        # print(f"x1 shape: {x1.shape}")
        # print(f"x2 shape: {x2.shape}")
        # print(f"x3 shape: {x3.shape}")
        # print(f"After up1 shape: {x.shape}") # After self.up1(x3, x2)
        # print(f"After up2 shape: {x.shape}") # After self.up2(x, x1)
        # print(f"Logits shape: {logits.shape}")

        return logits


# Version 1 (Using nn.Sequential for shared_layers - forward pass needs adjustment)
# class unet_v1(BasicModule):
#     def __init__(self,  bilinear=False):
#         super(unet_v1, self).__init__()
#         self.model_name = 'unet_v1'
#         n_channels=1
#         n_classes=1
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         # This shared_layers definition is problematic for standard U-Net skip connections
#         # if not carefully handled in forward.
#         self.shared_layers = nn.Sequential(
#             DoubleConv(n_channels, 32), # Output x1
#             Down(32, 64),              # Output x2
#             Down(64, 128)              # Output x_encoded (bottom of U)
#         )
#         # factor = 2 if bilinear else 1 # factor not explicitly used here

#         self.up1 = Up(128, 64, bilinear) # Expects 128 from below, 64 from skip
#         self.up2 = Up(64, 32, bilinear)  # Expects 64 from below, 32 from skip
#         self.outc = OutConv(32, n_classes)

#     def forward(self, x):
#         # Need to explicitly get intermediate layers for skip connections
#         x1 = self.shared_layers[0](x)
#         x2 = self.shared_layers[1](x1)
#         x_encoded = self.shared_layers[2](x2) # This is the output after all shared_layers

#         # The original forward pass for this version was:
#         # x1 = self.shared_layers[0](x)  <-- This is correct for first skip
#         # x2 = self.shared_layers[1](x1) <-- This is correct for second skip
#         # x_original_pass = self.shared_layers(x) <-- This recomputes everything, x_original_pass is x_encoded
#         # x_up1_out = self.up1(x_original_pass, x2) # Correct: upsample x_encoded, concat with x2
#         # x_up2_out = self.up2(x_up1_out, x1) # Correct: upsample x_up1_out, concat with x1
#         # logits = self.outc(x_up2_out)

#         # Corrected forward pass for this structure:
#         up_x1 = self.up1(x_encoded, x2) # Upsample from bottom, skip connect x2
#         up_x2 = self.up2(up_x1, x1)     # Upsample further, skip connect x1
#         logits = self.outc(up_x2)
#         # print(logits.shape)
#         return logits


# Version 2 (a debugging/testing setup)
# import numpy as np
# class unet_v2_debug(BasicModule):
#     def __init__(self,  bilinear=False):
#         super(unet_v2_debug, self).__init__()
#         self.model_name = 'unet_v2_debug'
#         n_channels=1
#         n_classes=1
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 32)
#         self.outc = OutConv(32, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         # The following lines (x2 to x7 and param1 to param6) seem to be for inspecting
#         # intermediate activations and parameters within the DoubleConv block,
#         # not part of a standard U-Net forward pass.
#         # x2=self.inc.double_conv[0](x)
#         # param1=self.inc.double_conv[0].state_dict()
#         # x3 = self.inc.double_conv[1](x2)
#         # param2 = self.inc.double_conv[1].state_dict()
#         # x4 = self.inc.double_conv[2](x3)
#         # param3 = self.inc.double_conv[2].state_dict()
#         # x5 = self.inc.double_conv[3](x4)
#         # param4 = self.inc.double_conv[3].state_dict()
#         # x6 = self.inc.double_conv[4](x5)
#         # param5 = self.inc.double_conv[4].state_dict()
#         # x7 = self.inc.double_conv[5](x6)
#         # param6 = self.inc.double_conv[5].state_dict()
#         logits = self.outc(x1)
#         # print(logits.shape)
#         return logits

# Version 3 (Similar to the active one, but `shared_layers` is only the first DoubleConv)
# class unet_v3(BasicModule):
#     def __init__(self,  bilinear=False):
#         super(unet_v3, self).__init__()
#         self.model_name = 'unet_v3'
#         n_channels=1
#         n_classes=1
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc_layer = DoubleConv(n_channels, 32) # Renamed from shared_layers for clarity
#         self.down1 = Down(32, 64)
#         self.down2 = Down(64, 128)
#         # factor = 2 if bilinear else 1 # Not explicitly used in this version's layer defs

#         self.up1 = Up(128, 64, bilinear)
#         self.up2 = Up(64, 32, bilinear)
#         self.outc = OutConv(32, n_classes)

#     def forward(self, x):
#         x1 = self.inc_layer(x)
#         # m2 = self.inc_layer.state_dict() # For debugging parameters, not part of forward pass
#         x2 = self.down1(x1)
#         x_encoded = self.down2(x2)

#         up_x1 = self.up1(x_encoded, x2)
#         up_x2 = self.up2(up_x1, x1)
#         logits = self.outc(up_x2)
#         # print(logits.shape)
#         return logits


# Version 4 (U-Net with UpNoSkip - assumes UpNoSkip is defined in unet_parts)
# from .unet_parts import UpNoSkip # Make sure this is imported if using this version
# class unet_v4_noskip(BasicModule):
#     def __init__(self, bilinear=False):
#         super(unet_v4_noskip, self).__init__()
#         self.model_name = 'unet_v4_noskip'
#         n_channels = 1
#         n_classes = 1
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.encoder = nn.Sequential(
#             DoubleConv(n_channels, 32),
#             Down(32, 64),
#             Down(64, 128)
#         )
#         # factor = 2 if bilinear else 1 # Not explicitly used here

#         # UpNoSkip implies the second argument (skip connection) is not used for concatenation
#         self.up1 = UpNoSkip(128, 64, bilinear) # Input is 128, output should be 64
#         self.up2 = UpNoSkip(64, 32, bilinear)  # Input is 64, output should be 32
#         self.outc = OutConv(32, n_classes)

#     def forward(self, x):
#         # The original forward pass for this version had a slight confusion:
#         # x1 = self.shared_layers[0](x) # This is DoubleConv(n_channels, 32) output
#         # x2 = self.shared_layers[1](x1) # This is Down(32, 64) output
#         # x_encoded_original = self.shared_layers(x) # This is Down(64, 128) output

#         # If UpNoSkip does not take a skip connection for concatenation:
#         x_encoded = self.encoder(x)
#         up_x1 = self.up1(x_encoded) # Pass only the upsampled feature
#         up_x2 = self.up2(up_x1)   # Pass only the upsampled feature
#         logits = self.outc(up_x2)
#         # print(logits.shape)
#         return logits