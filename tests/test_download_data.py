"""Downloader integration using the published archive layout and tiny files."""
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import zstandard
import download_data


class DownloadDataTests(unittest.TestCase):
    def test_download_prepare_and_resume_matching_shards(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            remote = root/'remote'
            remote.mkdir()
            calls = []
            def put(name, content):
                p = remote/name
                p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(content)
                return p
            record = dict(edit_type='deletion',shard='00',obj_id='obj',edit_id='edit')
            prefix = 'deletion/00/obj/edit/'
            camera = dict(camera={'transform_matrix': [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],'camera_angle_x': .7})
            def archive(name, files):
                buf = io.BytesIO()
                with tarfile.open(fileobj=buf,mode='w') as tar:
                    for path, data in files.items():
                        member = tarfile.TarInfo('./'+path)
                        member.size = len(data)
                        tar.addfile(member,io.BytesIO(data))
                return put(name,zstandard.ZstdCompressor().compress(buf.getvalue()))
            pairs = archive('data/train_shards/pairs.tar.zst',{
                prefix+'before.npz': b'npz',prefix+'after.npz': b'npz',
                prefix+'after.png': b'png',prefix+'meta.json': json.dumps(camera).encode()})
            masks = archive('data/train_mask_sidecars/masks.tar.zst',{
                'mask_sidecars/deletion/00/obj/edit.npz': b'mask'})
            for kind, p in [('train_shards',pairs),('train_mask_sidecars',masks)]:
                put(f'data/{kind}/index.json',json.dumps({'shards':{'00':{'archive':p.name,'bytes':p.stat().st_size}}}).encode())
            put('data/train_manifests/release_summary.json',b'{"split":"train"}')
            put('data/train_manifests/by_shard/00.jsonl',(json.dumps(record)+'\n').encode())
            def fetch(repo_id, filename, **kwargs):
                calls.append(filename)
                self.assertEqual(kwargs['revision'],'commit123')
                return str(remote/filename)
            from types import SimpleNamespace
            api = SimpleNamespace(dataset_info=lambda *args,**kwargs: SimpleNamespace(sha='commit123'))
            args = ['download_data.py','--data_dir',str(root/'prepared'),'--shards','00']
            with patch('huggingface_hub.HfApi',return_value=api),patch('huggingface_hub.hf_hub_download',side_effect=fetch),patch.object(sys,'argv',args):
                download_data.main()
                download_data.main()
            self.assertEqual(calls.count('data/train_shards/pairs.tar.zst'),1)
            self.assertEqual(calls.count('data/train_mask_sidecars/masks.tar.zst'),1)
            self.assertTrue((root/'prepared/mask_sidecars/deletion/00/obj/edit.npz').is_file())
            manifest = root/'prepared/training/manifests/all.jsonl'
            self.assertEqual(len(manifest.read_text().splitlines()),1)


if __name__ == '__main__':
    unittest.main()
