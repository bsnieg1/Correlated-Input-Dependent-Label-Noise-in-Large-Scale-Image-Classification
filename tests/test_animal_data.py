import numpy as np
import torch

from src.animal_data import RECORD, make_transforms, read_bin
from src.models import ResNet18Small


def write_bin(path, ids, labels, images, order="<"):
    rec = np.zeros((len(ids), RECORD), dtype=np.uint8)
    rec[:, :4] = np.asarray(ids, dtype=f"{order}i4").view(np.uint8).reshape(-1, 4)
    rec[:, 4:8] = np.asarray(labels, dtype=f"{order}i4").view(np.uint8).reshape(-1, 4)
    rec[:, 8:] = images.reshape(len(ids), -1)
    rec.tofile(path)


def test_read_bin_both_byte_orders(tmp_path):
    images = np.random.default_rng(0).integers(0, 256, (3, 3, 64, 64), dtype=np.uint8)
    for order in ("<", ">"):
        p = tmp_path / f"b{order == '<'}.bin"
        write_bin(p, [7, 8, 9], [0, 9, 4], images, order)
        ids, labels, x = read_bin(p)
        assert ids.tolist() == [7, 8, 9] and labels.tolist() == [0, 9, 4]
        assert np.array_equal(x, images)


def test_train_transform_is_a_shifted_crop():
    x = torch.arange(2 * 3 * 64 * 64, dtype=torch.float32).view(2, 3, 64, 64).to(torch.uint8)
    train_tf, eval_tf = make_transforms(torch.zeros(3), torch.ones(3), pad=4)
    out = train_tf(x)
    assert out.shape == (2, 3, 64, 64) and out.dtype == torch.float32
    assert torch.equal(eval_tf(x), x.float())


def test_resnet18_small_shapes():
    net = ResNet18Small()
    assert net(torch.randn(2, 3, 64, 64)).shape == (2, 512)
