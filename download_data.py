"""Download and prepare the published Pxform training shards inside PartFlow."""
import argparse
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
REPO = 'ART-3D/Pxform_v1'


def extract_archive(archive, destination):
    """Stream tar.zst; permit only regular files/directories inside destination."""
    import zstandard
    destination = Path(destination).resolve()
    with open(archive, 'rb') as raw, zstandard.ZstdDecompressor().stream_reader(raw) as stream:
        with tarfile.open(fileobj=stream, mode='r|') as tar:
            for member in tar:
                parts = PurePosixPath(member.name)
                if parts.is_absolute() or '..' in parts.parts:
                    raise ValueError(f'Unsafe archive path: {member.name}')
                target = destination.joinpath(*parts.parts)
                if not target.resolve().is_relative_to(destination):
                    raise ValueError(f'Archive path escapes destination: {member.name}')
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, open(target, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                else:
                    raise ValueError(f'Unsupported archive entry: {member.name}')


def verify(root):
    """Check all prepared records, masks and render cameras without loading latents."""
    training = root / 'training'
    counts = {}
    errors = []
    with open(training / 'manifests/all.jsonl') as f:
        for line in f:
            r = json.loads(line)
            case = training / r['edit_type'] / str(r['shard']) / r['obj_id'] / r['edit_id']
            counts[r['edit_type']] = counts.get(r['edit_type'], 0) + 1
            for name in ('before.npz', 'after.npz', 'after.png'):
                if not (case / name).is_file():
                    errors.append(f'{case}/{name}')
            pose = case / 'view.meta.json'
            if not pose.is_file():
                pose = case / 'meta.json'
            try:
                meta = json.loads(pose.read_text())
                frame = meta.get('frame', meta.get('camera', meta))
                if len(frame['transform_matrix']) != 4 or not 0 < float(frame['camera_angle_x']) < 3.141593:
                    raise ValueError('Invalid camera')
            except (OSError, ValueError, KeyError, TypeError):
                errors.append(f'{pose}: missing/invalid camera')
            mask = root / 'mask_sidecars' / r['edit_type'] / str(r['shard']) / r['obj_id'] / (r['edit_id'] + '.npz')
            if r['edit_type'] != 'global' and not mask.is_file():
                errors.append(str(mask))
    if not counts or errors:
        raise RuntimeError(f'Data verification failed: {len(errors)} errors; first entries: {errors[:10]}')
    print(json.dumps({'verified_records': sum(counts.values()), 'by_type': counts}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', type=Path, default=ROOT / 'data/Pxform_v1')
    parser.add_argument('--shards', nargs='+', default=['00'], help='00 ... 09, or all (default: 00)')
    parser.add_argument('--revision', default='main', help='HF branch or commit; resolved to a pinned commit')
    parser.add_argument('--verify_only', action='store_true')
    args = parser.parse_args()
    root = args.data_dir.resolve()
    if args.verify_only:
        verify(root)
        return
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    revision = api.dataset_info(REPO, revision=args.revision).sha
    downloads = root / 'downloads'
    def fetch(path):
        return Path(hf_hub_download(REPO, path, repo_type='dataset', revision=revision,
                                    local_dir=downloads))
    indexes = {kind: json.loads(fetch(f'data/{kind}/index.json').read_text())
               for kind in ('train_shards', 'train_mask_sidecars')}
    shards = sorted(indexes['train_shards']['shards']) if args.shards == ['all'] else sorted(set(args.shards))
    for shard in shards:
        if shard not in indexes['train_shards']['shards'] or shard not in indexes['train_mask_sidecars']['shards']:
            parser.error(f'Unknown shard {shard}; use two digits, e.g. 00')
    training = root / 'training'
    previous_release = training / 'release.json'
    if previous_release.is_file():
        previous_revision = json.loads(previous_release.read_text()).get('revision')
        if previous_revision and previous_revision != revision:
            parser.error('Destination contains another release commit; use that --revision or a fresh --data_dir')
    if training.is_symlink() or (training.is_dir() and any(p.is_symlink() for p in training.iterdir())):
        parser.error('Refusing to overwrite a linked local dataset; use a fresh --data_dir')
    manifests = training / 'manifests/by_shard'
    manifests.mkdir(parents=True, exist_ok=True)
    release = json.loads(fetch('data/train_manifests/release_summary.json').read_text())
    (training / 'release.json').write_text(json.dumps({**release, 'repo_id': REPO, 'revision': revision}, indent=2)+'\n')
    for shard in shards:
        for kind, destination in (('train_shards', training), ('train_mask_sidecars', root)):
            name = indexes[kind]['shards'][shard]['archive']
            marker = root / '.prepared' / f'{kind}_{shard}.json'
            if marker.is_file() and json.loads(marker.read_text()).get('revision') == revision:
                print(f'Already extracted: {kind}/{shard}', flush=True)
                continue
            archive = fetch(f'data/{kind}/{name}')
            expected = indexes[kind]['shards'][shard]['bytes']
            if archive.stat().st_size != expected:
                raise ValueError(f'Archive size mismatch: {archive}')
            print(f'Extracting {name}', flush=True)
            extract_archive(archive, destination)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({'revision': revision, 'archive': name})+'\n')
        shutil.copyfile(fetch(f'data/train_manifests/by_shard/{shard}.jsonl'), manifests / f'{shard}.jsonl')
    # Rebuild from all locally prepared shards, including earlier invocations.
    with open(training / 'manifests/all.jsonl', 'wb') as out:
        for manifest in sorted(manifests.glob('*.jsonl')):
            with open(manifest, 'rb') as src:
                shutil.copyfileobj(src, out)
    verify(root)
    print(f'Ready: python train.py --config configs/train_stage1_ss.json --data_dir {training} --output_dir outputs/train_ss')


if __name__ == '__main__':
    main()
