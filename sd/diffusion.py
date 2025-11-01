import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd:int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, n_embd*4)
        self.linear_2 = nn.Linear(n_embd*4, 4*n_embd)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #(x, 320)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        #(x, 1280)
        return x
class SwitchSequential(nn.Sequential):
    def forward(self, x:torch.Tensor, context:torch.Tensor, time:torch.Tensor) -> torch.Tensor:
        for layer in self:
            # 如果是注意力层，就给它 x 和 context（文字信息）
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            # 如果是残差块，就给它 x 和 time（时间信息）  
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            # 如果是普通层（如Conv2d），只给它 x
            else:
                x = layer(x)
        return x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            #(Batch_size, 320, height/8, width/8) -> (Batch_size, 640, height/16, height/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            #(Batch_size, 640, height/16, width/16) -> (Batch_size, 640, height/32, height/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            #(Batch_size, 1280, height/32, width/32) -> (Batch_size, 1280, height/64, height/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280))
            #(Batch_size, 1280, height/64, width/64) -> (Batch_size, 1280, height/64, height/64)
            ]
        )
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280)
        )

        self.decoders = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UpSample(1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), UpSample(1280)),
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), UpSample(640)),
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40))
        ])
    def forward(self, x, context, time):
        #x:(Batch_size, 4, Height/8, Width/8)
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)
        x = self.bottleneck(x, context, time)
        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)
        return x
                          

class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):
        #feature: (Batch_size, In_channels, Height, Width)
        #time(1,1280)
        residue = feature
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        feature = self.conv_feature(feature)
        # (1, 1280) -> (1, 1280)
        time = F.silu(time)
        time = self.linear_time(time)
        # (Batch_Size, Out_Channels, Height, Width) + (1, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Height, Width)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return merged + self.residual_layer(residue)
    
class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_heads:int, n_embd:int, d_context:int=768):
        super().__init__()
        channels = n_heads * n_embd
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_heads, channels, in_proj_bias = False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_heads, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4*channels*2)
        self.linear_geglu_2 = nn.Linear(4*channels, channels)
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    def forward(self, x, context):
        residue_long = x
        x = self.groupnorm(x)
        x = self.conv_input(x)
        n, c, h, w = x.shape
        #Normalization + self-attention
        x = x.view(n, c, h*w).transpose(-1,-2)
        residue_short = x
        x = self.attention_1(self.layernorm_1(x))
        x += residue_short
        residue_short = x
        #Normalization + cross-attention
                #Cross Attention
        x = self.attention_2(self.layernorm_2(x), context)
        x += residue_short
        residue_short = x
        #Normalization + Feedforward with geglu and skip connection
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short
        #Batch_size, height*width, features -> (batch_size, features, height*width)
        x = x.transpose(-1, -2).view(n, c, h, w)
        return self.conv_output(x) + residue_long

class UpSample(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # (Batch_size, features, height, width) -> (Batch_size, features, height*2, width*2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)
    
class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
    def forward(self, x):
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        #(Batch_size, 4, Height/8, Width/8)
        return x

class Diffusion(nn.Module):

    def __init__(self):
        super().__init__()  
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        #latent: (Batch_size,4,height/8, width/8)
        #context: (batch_size, seq_len, dim)
        #Time: (1,320
        #(1,320) -> (1,1200)
        time = self.time_embedding(time)
        #(Batch, 4, height/8, width/8)
        output = self.unet(latent, context, time)
        output = self.final(output)
        # (Batch, 4, Height / 8, Width / 8)
        return output