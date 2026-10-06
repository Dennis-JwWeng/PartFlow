"""Loss normalization, sample filtering, endpoint and camera regression checks."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from trellis.trainers.flow_matching.mask_loss_utils import (
    masked_mse_velocity, filter_keep_mask,
)
from trellis.trainers.flow_matching.render_loss import (
    predict_clean_features, load_target_image, RenderLossConfig,
)
from trellis.trainers.flow_matching.ss_render_loss import render_occupancy
from trellis.datasets.pxform_training import latent_delta_masks


class TrainingLossTests(unittest.TestCase):
    def test_explicit_all_edit_types_survives_config_loading(self):
        self.assertIsNone(RenderLossConfig.from_dict({'allowed_edit_types': None}).allowed_edit_types)

    def test_dense_mask_channel_normalization(self):
        pred = torch.ones(2, 8, 3, 3, 3, requires_grad=True)
        mask = torch.ones(2, 1, 3, 3, 3)
        loss = masked_mse_velocity(pred, torch.zeros_like(pred), mask)
        self.assertAlmostEqual(loss.item(), 1.0)
        loss.backward()
        self.assertGreater(pred.grad.abs().sum().item(), 0)

    def test_global_filter_is_per_sample(self):
        ref = torch.zeros(2, 8, 2, 2, 2)
        m = filter_keep_mask(torch.ones(2, 1, 2, 2, 2), ['global', 'scale'], True, ref)
        self.assertEqual(m[0].sum().item(), 0)
        self.assertEqual(m[1].sum().item(), 8)
        sparse = filter_keep_mask(torch.ones(5), ['global', 'scale'], True,
                                 torch.zeros(5, 8), [slice(0, 2), slice(2, 5)])
        self.assertEqual(sparse.tolist(), [0, 0, 1, 1, 1])

    def test_endpoint_matches_flow_for_nonzero_sigma(self):
        torch.manual_seed(0)
        x0, eps = torch.randn(3, 8), torch.randn(3, 8)
        t, sigma = torch.tensor([[0.0], [0.4], [1.0]]), 0.1
        xt = (1-t)*x0 + (sigma+(1-sigma)*t)*eps
        velocity = (1-sigma)*eps-x0
        torch.testing.assert_close(predict_clean_features(xt, velocity, t, sigma), x0)

    def test_rgba_render_target_uses_configured_background(self):
        with tempfile.TemporaryDirectory() as d:
            Image.fromarray(np.zeros((8, 8, 4), dtype=np.uint8), 'RGBA').save(Path(d)/'after.png')
            image = load_target_image(d, RenderLossConfig(background=(1, 1, 1)), 8)
            self.assertEqual(image.dtype, torch.float32)
            torch.testing.assert_close(image, torch.ones_like(image))

    def test_silhouette_camera_and_gradient(self):
        volume = torch.zeros(1, 1, 16, 16, 16)
        volume[:, :, 5:11, 5:11, 5:11] = 0.5
        volume.requires_grad_()
        ext = torch.eye(4)
        ext[2, 3] = 2  # camera at world z=-2, looking +z
        intr = torch.tensor([[1., 0, .5], [0, 1., .5], [0, 0, 1.]])
        silhouette = render_occupancy(volume, ext, intr, 32, 64)
        self.assertGreater(silhouette[16, 16].item(), .5)
        self.assertEqual(silhouette[0, 0].item(), 0)
        silhouette.sum().backward()
        self.assertTrue(torch.isfinite(volume.grad).all())
        self.assertGreater(volume.grad.abs().sum().item(), 0)

    def test_estimated_masks_never_preserve_new_empty_positions(self):
        c = np.array([[16, 16, 16], [48, 48, 48]], dtype=np.int32)
        b = dict(slat_coords=c[:1], slat_feats=np.zeros((1, 8), np.float32))
        a = dict(slat_coords=c, slat_feats=np.zeros((2, 8), np.float32))
        ss, slat = latent_delta_masks(b, a, .3)
        self.assertEqual(slat.tolist(), [1, 0])
        self.assertEqual(ss[12, 12, 12].item(), 0)
        self.assertEqual(ss[4, 4, 4].item(), 1)


if __name__ == '__main__':
    unittest.main()
