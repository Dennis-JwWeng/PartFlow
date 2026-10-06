"""Public release layout, camera conversion and safe archive preparation."""
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import zstandard
from PIL import Image

from download_data import extract_archive, verify
from trellis.datasets.pxform_training import PxformSSDataset
from trellis.trainers.flow_matching.render_loss import load_render_camera, RenderLossConfig


class PublicDataTests(unittest.TestCase):
    def test_published_camera_matches_legacy(self):
        frame = {'transform_matrix': np.eye(4).tolist(), 'camera_angle_x': .7}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/'meta.json').write_text(json.dumps({'camera': frame}))
            public = load_render_camera(d, RenderLossConfig())
            (p/'view.meta.json').write_text(json.dumps({'frame': frame}))
            legacy = load_render_camera(d, RenderLossConfig())
            for a, b in zip(public, legacy):
                torch.testing.assert_close(a, b)

    def test_public_train_split_and_automatic_sidecars(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            training = root/'training'
            record = dict(edit_type='deletion', shard='00', obj_id='o', edit_id='e')
            case = training/'deletion/00/o/e'
            case.mkdir(parents=True)
            arrays = dict(ss=np.zeros((8,16,16,16), np.float32),
                          slat_coords=np.array([[4,4,4]], np.int32),
                          slat_feats=np.zeros((1,8), np.float32))
            for name in ['before','after']:
                np.savez(case/f'{name}.npz', **arrays)
            Image.new('RGB', (16,16)).save(case/'after.png')
            (case/'meta.json').write_text(json.dumps({'camera': {
                'transform_matrix': np.eye(4).tolist(), 'camera_angle_x': .7}}))
            (training/'manifests').mkdir()
            (training/'manifests/all.jsonl').write_text(json.dumps(record)+'\n')
            (training/'release.json').write_text(json.dumps({'split':'train'}))
            mask = root/'mask_sidecars/deletion/00/o/e.npz'
            mask.parent.mkdir(parents=True)
            np.savez(mask, mask_keep_ss=np.ones((16,16,16),np.uint8),
                     mask_keep_slat=np.ones(1,np.uint8))
            verify(root)
            dataset = PxformSSDataset(training)
            self.assertEqual(len(dataset), 1)
            self.assertGreater(dataset[0]['mask_keep_ss'].sum(), 0)
            with self.assertRaises(FileNotFoundError):
                PxformSSDataset(training, split='val')
            linked = root/'links'
            linked.mkdir()
            (linked/'training').symlink_to(training, target_is_directory=True)
            (linked/'mask_sidecars').symlink_to(root/'mask_sidecars',target_is_directory=True)
            self.assertGreater(PxformSSDataset(linked/'training')[0]['mask_keep_ss'].sum(),0)

    def test_streamed_archive_rejects_traversal_and_links(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name, typ in [('deletion/00/file.txt',tarfile.REGTYPE),
                              ('../escape.txt',tarfile.REGTYPE),
                              ('link',tarfile.SYMTYPE)]:
                buf = io.BytesIO()
                with tarfile.open(fileobj=buf,mode='w') as tar:
                    member = tarfile.TarInfo(name)
                    member.type = typ
                    if typ == tarfile.REGTYPE:
                        member.size = 3
                        tar.addfile(member, io.BytesIO(b'abc'))
                    else:
                        member.linkname = '/tmp'
                        tar.addfile(member)
                archive = root/'case.tar.zst'
                archive.write_bytes(zstandard.ZstdCompressor().compress(buf.getvalue()))
                if name.startswith('deletion'):
                    extract_archive(archive,root/'expanded')
                    self.assertEqual((root/'expanded'/name).read_bytes(), b'abc')
                else:
                    with self.assertRaises(ValueError):
                        extract_archive(archive,root/'expanded')


if __name__ == '__main__':
    unittest.main()
