import torch
from torch import nn
from transformers import GPT2Config, GPT2Model


# class DiffusionModel(torch.nn.Module):
#     def __init__(self, gpt2_model_name='gpt2', image_embedding_dim=2048):
#         super(DiffusionModel, self).__init__()
#
#
#         # Load pre-trained GPT-2 model
#         self.gpt2_config = GPT2Config.from_pretrained(gpt2_model_name)
#         self.gpt2_model = GPT2Model(self.gpt2_config)
#
#         # Linear layer to project image embeddings to GPT-2's hidden size
#         self.image_embedding_projection = torch.nn.Linear(image_embedding_dim, self.gpt2_config.n_embd)
#
#         # Position embeddings
#         self.position_embeddings = torch.nn.Embedding(self.gpt2_config.max_position_embeddings, self.gpt2_config.n_embd)
#
#     def forward(self, noisy_label_vector, y_c, t_tensor):
#         # Concatenate inputs along the sequence dimension
#         inputs = torch.cat([noisy_label_vector.unsqueeze(1), y_c.unsqueeze(1), t_tensor.unsqueeze(1)], dim=1)
#
#         # Project inputs to match GPT-2's hidden size
#         projected_inputs = self.image_embedding_projection(inputs)
#
#         # Add position embeddings
#         seq_length = projected_inputs.size(1)
#         position_ids = torch.arange(seq_length, dtype=torch.long, device=projected_inputs.device)
#         position_embeddings = self.position_embeddings(position_ids)
#         projected_inputs = projected_inputs + position_embeddings.unsqueeze(0)
#
#         # Process with GPT-2
#         gpt2_outputs = self.gpt2_model(inputs_embeds=projected_inputs)
#
#         return gpt2_outputs


# class Bottleneck(nn.Module):
#     def __init__(self, in_channels, out_channels, reduction=4):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // reduction, 1),
#             nn.BatchNorm2d(in_channels // reduction),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // reduction, in_channels // reduction, 3, padding=1),
#             nn.BatchNorm2d(in_channels // reduction),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // reduction, out_channels, 1),
#             nn.BatchNorm2d(out_channels),
#         )
#
#     def forward(self, x):
#         out = self.block(x)
#         return out + x
# class DiffusionModel(nn.Module):
#         def __init__(self, channels_in, kernel_size=3):
#             super().__init__()
#             self.kernel_size = kernel_size
#             self.time_embedding = nn.Embedding(1280, channels_in)
#
#             if kernel_size == 3:
#                 self.pred = nn.Sequential(
#                     Bottleneck(channels_in, channels_in),
#                     Bottleneck(channels_in, channels_in),
#                     nn.Conv2d(channels_in, channels_in, 1),
#                     nn.BatchNorm2d(channels_in)
#                 )
#             else:
#                 self.pred = nn.Sequential(
#                     nn.Conv2d(channels_in, channels_in * 4, 1),
#                     nn.BatchNorm2d(channels_in * 4),
#                     nn.ReLU(inplace=True),
#                     nn.Conv2d(channels_in * 4, channels_in, 1),
#                     nn.BatchNorm2d(channels_in),
#                     nn.Conv2d(channels_in, channels_in * 4, 1),
#                     nn.BatchNorm2d(channels_in * 4),
#                     nn.ReLU(inplace=True),
#                     nn.Conv2d(channels_in * 4, channels_in, 1)
#                 )
#
#         def forward(self, noisy_image, t):
#             if t.dtype != torch.long:
#                 t = t.type(torch.long)
#             feat = noisy_image
#             feat = feat + self.time_embedding(t)[..., None, None]
#             ret = self.pred(feat)
#             return ret

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class DiffusionModel(nn.Module):
    def __init__(self, channels_in):
        super(DiffusionModel, self).__init__()
        self.time_embedding = nn.Embedding(1280, channels_in)

        self.pred = nn.Sequential(
            Bottleneck(channels_in, channels_in),
            Bottleneck(channels_in, channels_in),
            nn.Linear(channels_in, channels_in),
            nn.BatchNorm1d(channels_in),
            nn.Softmax(dim=1)  # Applying Softmax to ensure the output is a valid probability distribution
        )

    def forward(self, noisy_labels, t):
        if t.dtype != torch.long:
            t = t.type(torch.long)
        feat = noisy_labels + self.time_embedding(t)
        ret = self.pred(feat)
        return ret