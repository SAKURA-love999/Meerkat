#!/usr/bin/env python
"""Find candidate clips absent from registered analysis tables, without outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

CALL_LABELS = {'sn', 'cc', 'ld', 'mo', 'al', 'soc', 'agg', 'oth'}
IDENTIFIERS = ['_clip', 'source_id', 'clip_id', 'recording_id', 'call_id', 'sample_id', 'source_wav']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identify(values, clip_ids, call_to_clip):
    values = values.dropna().astype(str)
    direct = set(values) & clip_ids
    calls = set(values.map(call_to_clip).dropna())
    stems = values.str.replace('\\', '/', regex=False).str.rsplit('/', n=1).str[-1].str.replace(r'\.(wav|h5)$', '', regex=True)
    return direct | calls | (set(stems) & clip_ids)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--scan-root', type=Path, action='append', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261106)
    args = p.parse_args()
    # Deliberately do not load next-event labels, probabilities or acoustic values.
    cols = ['call_id', 'source_id', 'focal', 'label', 'start_time_sec', 'end_time_sec']
    manifest = pd.read_csv(args.manifest, usecols=cols)
    if manifest.call_id.duplicated().any():
        raise ValueError('Manifest call IDs are not unique.')
    clips = set(manifest.source_id.astype(str))
    mapping = manifest.set_index('call_id').source_id.to_dict()
    focal = manifest.focal.astype(str).str.lower().isin(['true', '1'])
    eligible = manifest.loc[focal & manifest.label.isin(CALL_LABELS)
        & manifest.start_time_sec.ge(.4) & manifest.end_time_sec.le(9.6)]
    counts = eligible.groupby('source_id').size().rename('n_focal_400_eligible')
    output = args.output_dir.resolve()
    paths = sorted({p.resolve() for root in args.scan_root for p in root.rglob('*.csv')
                    if output not in p.resolve().parents and p.resolve() != args.manifest.resolve()})
    used, rows = set(), []
    for path in paths:
        row = dict(path=str(path), status='unread', id_columns='', matched_clips=0)
        try:
            header = pd.read_csv(path, nrows=0).columns
            id_cols = [c for c in IDENTIFIERS if c in header]
            row['id_columns'] = ','.join(id_cols)
            if not id_cols:
                row['status'] = 'no_registered_id_column'
            else:
                found = set()
                for chunk in pd.read_csv(path, usecols=id_cols, dtype=str, chunksize=100000):
                    for col in id_cols:
                        found |= identify(chunk[col], clips, mapping)
                used |= found
                row.update(status='scanned_ids_only', matched_clips=len(found), sha256=digest(path))
        except Exception as exc:
            row.update(status='failed', error=repr(exc))
        rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage = pd.DataFrame(rows)
    coverage.to_csv(args.output_dir / 'table_coverage.csv', index=False)
    pd.DataFrame({'source_id': sorted(used)}).to_csv(args.output_dir / 'registered_used_clips.csv', index=False)
    pool = counts.loc[~counts.index.isin(used)].reset_index()
    pool['blinded_order_key'] = pool.source_id.map(lambda x: hashlib.sha256((str(args.seed) + '|' + x).encode()).hexdigest())
    pool = pool.sort_values('blinded_order_key')
    pool.to_csv(args.output_dir / 'candidate_clip_pool.csv', index=False)
    summary = dict(status='CANDIDATE_POOL_NOT_CERTIFIED_UNTOUCHED',
        n_manifest_clips=len(clips), n_400_eligible_clips=len(counts),
        n_registered_used_clips=len(used), n_candidate_clips=len(pool),
        n_candidate_focal_eligible=int(pool.n_focal_400_eligible.sum()),
        n_csv_tables=len(paths), n_failed_tables=int(coverage.status.eq('failed').sum()),
        seed=args.seed, manifest_sha256=digest(args.manifest), script_sha256=digest(__file__),
        scope='Excludes any clip found in identifier-bearing CSVs under declared roots. No future-activity outcomes read.',
        limits=['Prior aggregate analyses or external/not-CSV experiments may have seen candidates.',
                'Absence from these tables does not certify untouched data or new callers/sessions.',
                'Do not select a primary claim or inspect confirmation outcomes before freezing a protocol.'],
        roots=[str(p.resolve()) for p in args.scan_root])
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
