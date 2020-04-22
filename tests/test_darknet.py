import itertools
import os

import pytest
import torch

from darknet import Darknet


def get_test_data():
    models = ("yolov3", "yolov3-tiny", "yolov3-spp")
    image_dims = (320, 416, 608)
    batch_sizes = (1, 2)
    return list(itertools.product(models, image_dims, batch_sizes))


@pytest.mark.parametrize("model,image_dim,batch_size", get_test_data())
def test_yolov3(model, image_dim, batch_size):
    model_dir = "models"
    config_path = os.path.join(model_dir, model + ".cfg")
    weights_path = os.path.join(model_dir, model + ".weights")
    net = Darknet(config_path, device="cpu")
    net.load_weights(weights_path)

    x = torch.rand(batch_size, 3, image_dim, image_dim)
    y = net.forward(x)
