import torch
import torch.nn as nn

from torch_cnn.blocks.conv import make_conv_block
from torch_cnn.models.stages import compose_to_stage_1, compose_to_stage_2, compose_to_stage_3


class ReferenceNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        widths=(64, 128, 256),
        depths=(2, 2, 2),
        *,
        conv_kind: str = "dynamic",
        norm: str = "bn",
        act: str = "relu",
        use_attention: bool = True,
        attn_kwargs: dict | None = None,
        dynamic_K: int = 4,
        dynamic_reduction: int = 4,
        dynamic_temperature: float = 1.0,
    ):
        super().__init__()

        w1, w2, w3 = map(int, widths)
        d1, d2, d3 = map(int, depths)

        self.stem = make_conv_block(
            conv_kind,
            in_channels,
            w1,
            k=3,
            stride=1,
            norm=norm,
            act=act,
            K=dynamic_K,
            reduction=dynamic_reduction,
            temperature=dynamic_temperature,
        )

        self.stage1 = compose_to_stage_1(
            cin=w1,
            cout=w1,
            depth=d1,
            conv_kind=conv_kind,
            norm=norm,
            act=act,
            use_attention=use_attention,
            attn_kwargs=attn_kwargs,
            dynamic_K=dynamic_K,
            dynamic_reduction=dynamic_reduction,
            dynamic_temperature=dynamic_temperature,
        )

        self.stage2 = compose_to_stage_2(
            cin=w1,
            cout=w2,
            depth=d2,
            conv_kind=conv_kind,
            norm=norm,
            act=act,
            use_attention=use_attention,
            attn_kwargs=attn_kwargs,
            dynamic_K=dynamic_K,
            dynamic_reduction=dynamic_reduction,
            dynamic_temperature=dynamic_temperature,
        )

        self.stage3 = compose_to_stage_3(
            cin=w2,
            cout=w3,
            depth=d3,
            conv_kind=conv_kind,
            norm=norm,
            act=act,
            use_attention=use_attention,
            attn_kwargs=attn_kwargs,
            dynamic_K=dynamic_K,
            dynamic_reduction=dynamic_reduction,
            dynamic_temperature=dynamic_temperature,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(w3, int(num_classes))

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    attn_kwargs = dict(
        ca_reduction=16,
        ca_fusion="softmax",
        ca_gate="sigmoid",
        ca_temperature=0.9,
        coord_norm="gn",
        coord_dilation=2,
    )
    model = ReferenceNet(
        in_channels=3,
        num_classes=10,
        widths=(64, 128, 256),
        depths=(2, 2, 2),
        conv_kind="dynamic",
        norm="bn",
        act="relu",
        use_attention=True,
        attn_kwargs=attn_kwargs,
        dynamic_K=4,
        dynamic_reduction=4,
        dynamic_temperature=1.0,
    ).to(device)

    x = torch.randn(8, 3, 32, 32, device=device)
    y = model(x)
    print("out:", y.shape)