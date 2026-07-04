import torch
import torch.nn as nn

from einops import rearrange, repeat, pack, unpack
from Attention_module.SE_block import se_block
from Attention_module.DANet import DAnet
from vit_pytorch import ViT, ViT_2, TransformerDecoder


class pyramid_Block(nn.Module):
    def __init__(self, inchannel, outchannel, stride=(1, 1)):
        super(pyramid_Block, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(inchannel * 2, outchannel, kernel_size=(3, 3),
                      stride=stride, padding=(1, 1), bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
        self.left1 = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, kernel_size=(3, 3),
                      stride=stride, padding=(1, 1), bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, kernel_size=(5, 5),
                      stride=stride, padding=(2, 2), bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
        self.left3 = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, kernel_size=(7, 7),
                      stride=stride, padding=(3, 3), bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, kernel_size=(3, 3), padding=(0, 0), stride=stride, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_out = self.conv0(x)
        out1 = self.left1(x_out)
        out2 = x_out + out1
        out2 = self.left2(out2)
        out3 = x_out + out2
        out3 = self.left3(out3)
        out = x_out + out1 + out2 + out3
        out = self.conv1_1(out)
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class HICDetector(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert (seq_len % patch_size) == 0

        self.dim = dim
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer1 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.transformer2 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.transformer3 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, T1, T2):
        x = torch.concat([T1, T2], dim=-1)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)
        x, ps = pack([cls_tokens, x], 'b * d')
        x = self.dropout(x)
        x1 = self.transformer1(x)
        x2 = self.transformer2(x1 + x)
        x3 = self.transformer3(x2 + x)
        cls_tokens, _ = unpack(x3, ps, 'b * d')
        cls_prob = self.mlp_head(cls_tokens)
        return cls_prob


class Temporal_Endmember_Network(nn.Module):
    def __init__(self, channel, n_classes, device="cuda"):
        super(Temporal_Endmember_Network, self).__init__()
        self.device = device
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=31)
        self.conv2 = nn.Conv1d(64, 64, 31)
        self.conv3 = nn.Conv1d(64, 64, 17)
        self.conv4 = nn.Conv1d(64, 64, 17)
        self.conv5 = nn.Conv1d(64, 64, 11)
        self.conv6 = nn.Conv1d(64, 64, 11)
        self.conv7 = nn.Conv1d(64, 64, 7)
        self.liner1 = nn.Linear(106, channel)
        self.liner2 = nn.Linear(channel, n_classes)
        self.channel = channel
        self.act = nn.GELU()

    def forward(self, T1, T2):
        x = torch.cat([T1, T2], dim=1)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))
        mid_features = self.act(self.conv7(x))
        mid_features = mid_features.mean(dim=1)
        x = self.liner1(mid_features)
        cls_prob = self.liner2(x)
        return cls_prob


class BCNN(nn.Module):
    def __init__(self, num_of_bands, num_of_class, patch_size):
        super(BCNN, self).__init__()
        self.num_of_bands = num_of_bands
        self.num_of_class = num_of_class
        self.left = nn.Sequential(
            nn.Conv2d(self.num_of_bands, self.num_of_bands, kernel_size=(3, 3), padding=0, stride=1),
            nn.BatchNorm2d(self.num_of_bands),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_of_bands, self.num_of_bands, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(self.num_of_bands),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_of_bands, self.num_of_bands, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(self.num_of_bands),
            nn.ReLU(inplace=True)
        )
        self.pooling = torch.nn.MaxPool2d(2, stride=1)
        self.dence = nn.Linear(self.num_of_bands, self.num_of_class)
        self.dence1 = nn.Linear(64, self.num_of_class)
        self.avg_pool = nn.AvgPool2d((patch_size - 2, patch_size - 2))

    def forward(self, x1, x2=None):
        x1 = self.left(x1)
        x2 = self.left(x2)
        x1_transpose = torch.einsum('bnij -> bnji', x1)
        x1_x2 = torch.einsum('bnik,bnkj -> bnij', x1_transpose, x2)
        x1_x2 = self.avg_pool(x1_x2)
        x1_x2 = torch.squeeze(x1_x2)
        res = self.dence(x1_x2)
        return res


class cnn_DANet(nn.Module):
    def __init__(self, num_of_bands, num_of_class, patch_size):
        super(cnn_DANet, self).__init__()
        self.num_of_bands = num_of_bands
        self.num_of_class = num_of_class
        self.cnn1 = pyramid_Block(self.num_of_bands, self.num_of_bands)
        self.avg_pool = nn.AvgPool2d((patch_size - 2, patch_size - 2))
        self.dense = nn.Linear(self.num_of_bands, self.num_of_class)
        self.DANet = DAnet(self.num_of_bands, self.num_of_class)

    def forward(self, x1, x2=None):
        x1 = torch.concat([x1, x2], dim=1)
        x1_cnn = self.cnn1(x1)
        x1_DANet = self.DANet(x1_cnn)
        output_res = self.avg_pool(x1_DANet)
        output_res = output_res.contiguous().view(output_res.size(0), output_res.size(1))
        output_res = self.dense(output_res)
        return output_res


class cnn_SE(nn.Module):
    def __init__(self, num_of_bands, num_of_class, patch_size):
        super(cnn_SE, self).__init__()
        self.num_of_bands = num_of_bands
        self.num_of_class = num_of_class
        self.cnn1 = pyramid_Block(self.num_of_bands, self.num_of_bands)
        self.avg_pool = nn.AvgPool2d((patch_size - 2, patch_size - 2))
        self.dense = nn.Linear(self.num_of_bands, self.num_of_class)
        self.SE_block = se_block(self.num_of_bands)

    def forward(self, x1, x2=None):
        x1 = torch.concat([x1, x2], dim=1)
        x1_cnn = self.cnn1(x1)
        x1_SE_block = self.SE_block(x1_cnn)
        output_res = self.avg_pool(x1_SE_block)
        output_res = output_res.contiguous().view(output_res.size(0), output_res.size(1))
        output_res = self.dense(output_res)
        return output_res


class single_transformer_S(nn.Module):
    def __init__(self, num_of_bands, num_of_class, patch_size, head, dim_head):
        super(single_transformer_S, self).__init__()
        self.num_of_bands = num_of_bands
        self.num_of_class = num_of_class
        self.transformer_S = ViT_2(
            image_size=patch_size,
            num_patches=patch_size ** 2,
            num_classes=self.num_of_class,
            dim=num_of_bands * 2,
            depth=1,
            heads=head,
            dim_head=dim_head,
            mlp_dim=8,
            dropout=0.1,
            emb_dropout=0.1,
            mode=ViT,
            MLP_model='MLP'
        )
        self.transformer_decoder_S = TransformerDecoder(
            dim=num_of_bands * 2,
            depth=1,
            heads=head,
            dim_head=dim_head,
            mlp_dim=8,
            dropout=0.
        )

        self.avg_pool = nn.AvgPool2d((patch_size, patch_size))
        self.dense = nn.Linear(self.num_of_bands * 2, num_of_class)

    def forward(self, x1, x2=None):
        x1 = torch.concat([x1, x2], dim=1)
        _, _, h, w = x1.shape
        x2_embedded = rearrange(x1, 'b n h w -> b n (h w)').transpose(1, 2)
        x1_transformer = self.transformer_S(x2_embedded)
        x2_decoder = self.transformer_decoder_S(x2_embedded, x1_transformer)

        x1_transformer = rearrange(x2_decoder.transpose(1, 2), 'b n (h w) -> b n h w', h=h, w=w)
        x1_cnn = self.avg_pool(x1_transformer)
        x1_cnn = x1_cnn.contiguous().view(x1_cnn.size(0), -1)
        x1_transformer = self.dense(x1_cnn)
        return x1_transformer
