import pdb
import re

import torch
import torch.nn.functional as F


class EmptyLayer(torch.nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


def parse_config(fpath):
    """
    TODO
    """

    with open(fpath, "r") as f:
        # Ignore lines consisting only of whitespace or commented lines.
        lines = [
            line.strip() for line in f.readlines()
            if not (line.isspace() or line.startswith("#"))
        ]

    # Each block begins with a line of the form "[type]", with the block type
    # (eg, "convolutional") enclosed in square brackets. Chunk config text
    # into blocks.
    block_start_lines = [
        line_num for line_num, line in enumerate(lines) if line.startswith("[")
    ]
    block_start_lines.append(len(lines))

    text_blocks = []
    for i in range(1, len(block_start_lines)):
        block_start, block_end = block_start_lines[i-1], block_start_lines[i]
        text_blocks.append(lines[block_start:block_end])

    def str2type(raw_val):
        """
        Helper function to convert a string input to the appropriate
        type (str, int, or float).
        """
        try:
            return int(raw_val)
        except ValueError:
            pass

        try:
            return float(raw_val)
        except ValueError:
            return raw_val


    blocks = []
    net_info = None
    for text_block in text_blocks:
        block = {"type": text_block[0][1:-1]}
        for line in text_block[1:]:
            key, raw_val = line.split("=")

            # Convert fields with multiple comma-separated values into lists.
            if "," in raw_val:
                val = [str2type(item.strip()) for item in raw_val.split(",")]
            else:
                val = str2type(raw_val.strip())

            # If this is a "yolo" block, it contains an "anchors" field
            # consisting of pairs of anchors; group anchors into chunks of two.
            key = key.strip()
            if key == "anchors":
                val = [val[i:i+2] for i in range(0, len(val), 2)]

            block[key] = val

        if block["type"] == "net":
            net_info = block
        else:
            blocks.append(block)

    return blocks, net_info


def blocks2modules(blocks, net_info):
    modules = torch.nn.ModuleList()
    
    curr_out_channels = None
    prev_layer_out_channels = net_info["channels"]
    out_channels_list = []

    for i, block in enumerate(blocks):
        module = torch.nn.Sequential()

        if block["type"] == "convolutional":
            batch_normalize = "batch_normalize" in block
            bias = not batch_normalize
            kernel_size = block["size"]
            padding = (kernel_size - 1) // 2 if "pad" in block else 0
            in_channels = prev_layer_out_channels
            out_channels = block["filters"]

            conv = torch.nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=block["stride"],
                padding=padding, bias=bias
            )
            module.add_module("conv_{}".format(i), conv)

            if batch_normalize:
                bn = torch.nn.BatchNorm2d(num_features=out_channels)
                module.add_module("batch_norm_{}".format(i), bn)

            if block["activation"] == "leaky":
                acti = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)
                module.add_module("leaky_{}".format(i), acti)

            # Update the number of current (output) channels.
            curr_out_channels = out_channels

        elif block["type"] == "route":
            module.add_module("route_{}".format(i), EmptyLayer())

            # If the value of "layers" is a list (of multiple layers),
            # concatenate the number of channels from each specified layer.
            # Else the value of "layers" will be an int corresponding to a
            # single layer.
            if isinstance(block["layers"], list):
                out_channels = sum(
                    out_channels_list[layer_idx] for layer_idx in block["layers"]
                )
            else:
                out_channels = out_channels_list[block["layers"]]
            curr_out_channels = out_channels

        elif block["type"] == "shortcut":
            module.add_module("shortcut_{}".format(i), EmptyLayer())

        elif block["type"] == "upsample":
            # TODO: Upsample is deprecated in favor of Interpolate; consider
            # using this and/or other interpolation methods?
            upsample = torch.nn.Upsample(
                scale_factor=block["stride"], mode="nearest"
            )
            module.add_module("upsample_{}".format(i), upsample)

        elif block["type"] == "yolo":
            import pdb; pdb.set_trace()

        modules.append(module)
        prev_layer_out_channels = curr_out_channels
        out_channels_list.append(curr_out_channels)
    print(out_channels_list)

    return None


if __name__ == "__main__":
    blocks, net_info = parse_config("yolov3-tiny.cfg")
    modules = blocks2modules(blocks, net_info)
