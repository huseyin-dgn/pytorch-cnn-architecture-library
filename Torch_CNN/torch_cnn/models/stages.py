import torch.nn as nn

from torch_cnn.blocks.conv import make_conv_block
from torch_cnn.blocks.residual import PreActResidualConvBlock
from torch_cnn.blocks.attention import CBAMChannelPlusCoord


def compose_to_stage_1(
    cin: int,
    cout: int,
    depth: int,
    *,
    conv_kind: str = "standard",
    norm: str = "bn",
    act: str = "relu",
    use_attention: bool = False,
    attn_kwargs: dict | None = None,
    dynamic_K: int = 4,
    dynamic_reduction: int = 4,
    dynamic_temperature: float = 1.0,
):
    layers: list[nn.Module] = []

    layers.append(
        make_conv_block(
            conv_kind,
            cin,
            cout,
            k=3,
            stride=1,
            norm=norm,
            act=act,
            K=dynamic_K,
            reduction=dynamic_reduction,
            temperature=dynamic_temperature,
        )
    )

    for _ in range(int(depth)):
        layers.append(
            PreActResidualConvBlock(
                cin=cout,
                cout=cout,
                stride=1,
                norm=norm,
                act=act,
                alpha=1.0,
                skip_norm=False,
            )
        )

    if use_attention:
        kw = dict(attn_kwargs or {})
        kw.setdefault("residual", False)      
        kw.setdefault("return_maps", False)   
        layers.append(CBAMChannelPlusCoord(channels=cout, **kw))

    return nn.Sequential(*layers)

def compose_to_stage_2(
    cin: int,
    cout: int,
    depth: int,
    *,
    conv_kind: str = "standard",
    norm: str = "bn",
    act: str = "relu",
    use_attention: bool = False,
    attn_kwargs: dict | None = None,
    dynamic_K: int = 4,
    dynamic_reduction: int = 4,
    dynamic_temperature: float = 1.0,
):
    layers: list[nn.Module] = []
    layers.append(
        make_conv_block(
            conv_kind,
            cin,
            cout,
            k=3,
            stride=2,
            norm=norm,
            act=act,
            K=dynamic_K,
            reduction=dynamic_reduction,
            temperature=dynamic_temperature,
        )
    )

    for _ in range(int(depth)):
        layers.append(
            PreActResidualConvBlock(
                cin=cout,
                cout=cout,
                stride=1,
                norm=norm,
                act=act,
                alpha=1.0,
                skip_norm=False,
            )
        )

    if use_attention:
        kw = dict(attn_kwargs or {})
        kw.setdefault("residual", False)
        kw.setdefault("return_maps", False)
        layers.append(CBAMChannelPlusCoord(channels=cout, **kw))

    return nn.Sequential(*layers)

def compose_to_stage_3(
    cin: int,
    cout: int,
    depth: int,
    *,
    conv_kind: str = "standard",
    norm: str = "bn",
    act: str = "relu",
    use_attention: bool = False,
    attn_kwargs: dict | None = None,
    dynamic_K: int = 4,
    dynamic_reduction: int = 4,
    dynamic_temperature: float = 1.0,
):
    layers: list[nn.Module] = []

    layers.append(
        make_conv_block(
            conv_kind,
            cin,
            cout,
            k=3,
            stride=2,
            norm=norm,
            act=act,
            K=dynamic_K,
            reduction=dynamic_reduction,
            temperature=dynamic_temperature,
        )
    )

    for _ in range(int(depth)):
        layers.append(
            PreActResidualConvBlock(
                cin=cout,
                cout=cout,
                stride=1,
                norm=norm,
                act=act,
                alpha=1.0,
                skip_norm=False,
            )
        )

    if use_attention:
        kw = dict(attn_kwargs or {})
        kw.setdefault("residual", False)
        kw.setdefault("return_maps", False)
        layers.append(CBAMChannelPlusCoord(channels=cout, **kw))

    return nn.Sequential(*layers)