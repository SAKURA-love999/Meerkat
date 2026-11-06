#!/usr/bin/env python
"""Audit/freeze causal waveform supports; optionally extract matched animal2vec arms.

No model is imported unless --extract is explicitly supplied. The three arms share
timing/padding, not biological observability: masking cannot remove annotation or
duration artifacts. Focal-frame mapping reuses the legacy proportional convention.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from audit_egemaps_selection_on_librosa import digest

MODES = ['call_only', 'past_context', 'context_only']


def support_plan(start_s, end_s, sample_rate, n_file_samples, minimum_ms=500.):
    start, end = int(round(start_s*sample_rate)), int(round(end_s*sample_rate))
    if not (0 <= start < end <= n_file_samples):
        raise ValueError('Invalid call boundaries; do not silently truncate a focal call.')
    n_minimum = int(round(minimum_ms*sample_rate/1000.))
    window_start = min(start, end-n_minimum)
    source_start, source_end = max(0, window_start), end
    focal_start, focal_end = start-window_start, end-window_start
    return dict(window_start_sample=window_start, window_end_sample=end,
        source_start_sample=source_start, source_end_sample=source_end,
        left_pad_samples=max(0,-window_start), right_pad_samples=0,
        n_samples=end-window_start, focal_start_sample=focal_start,
        focal_end_sample=focal_end, call_start_sample=start, call_end_sample=end,
        future_samples_allowed=0)


def masked_normalize(window, focal_start, focal_end, left_pad, mode):
    if mode not in MODES:
        raise ValueError(mode)
    allowed = np.ones(len(window), dtype=bool)
    allowed[:left_pad] = False
    if mode == 'call_only':
        allowed[:] = False
        allowed[focal_start:focal_end] = True
    elif mode == 'context_only':
        allowed[focal_start:focal_end] = False
    values = np.asarray(window, dtype=np.float32)[allowed]
    mean = float(np.mean(values)) if len(values) else 0.
    std = float(np.std(values)) if len(values) else 0.
    result = np.zeros(len(window), dtype=np.float32)
    informative = bool(len(values)>0 and np.isfinite(std) and std>1e-8)
    if informative:
        result[allowed] = (values-mean)/std
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite normalized input.')
    assert np.all(result[~allowed] == 0)
    return result, dict(normalization_n_samples=int(allowed.sum()),
        normalization_mean=mean, normalization_std=std, informative_input=informative,
        masked_fraction=float(1-allowed.mean()))


def read_allowed_window(path, plan):
    # Seek/read only through focal offset, rather than reading the entire 10-s clip.
    with sf.SoundFile(str(path)) as source:
        source.seek(plan['source_start_sample'])
        audio = source.read(plan['source_end_sample']-plan['source_start_sample'],
                            dtype='float32', always_2d=True).mean(axis=1)
    audio = np.pad(audio,(plan['left_pad_samples'],0))
    if len(audio) != plan['n_samples']:
        raise ValueError('Audio support does not match plan.')
    return audio.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path(r'C:\学校\MeerKAT\repertoire_geometry'))
    p.add_argument('--cohort-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--context-ms',type=float,default=500.)
    p.add_argument('--limit-rows',type=int)
    p.add_argument('--extract',action='store_true')
    p.add_argument('--animal2vec-root',type=Path)
    p.add_argument('--checkpoint-path',type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--chunk-size',type=int,default=128)
    args = p.parse_args()
    if args.context_ms <= 0 or args.chunk_size < 1:
        raise ValueError('Positive context/chunk size required.')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest_path = args.project_root/'data_manifest/call_manifest.csv'
    cohort_path = args.cohort_dir/'cohort_membership.csv'
    manifest = pd.read_csv(manifest_path)
    cohort = pd.read_csv(cohort_path)
    manifest = cohort[['call_id','fold']].merge(manifest,on='call_id',validate='one_to_one').sort_values('call_id')
    if len(manifest)!=len(cohort):
        raise ValueError('Cohort has calls absent from manifest.')
    if args.limit_rows:
        manifest = manifest.head(args.limit_rows)
    info_cache, rows = {}, []
    for row in manifest.to_dict('records'):
        path = str(row['source_wav'])
        if path not in info_cache:
            info_cache[path] = sf.info(path)
        info = info_cache[path]
        plan = support_plan(row['start_time_sec'],row['end_time_sec'],info.samplerate,info.frames,args.context_ms)
        # Audit the legacy centered 500-ms arm, separately from the new causal arm.
        mid = (plan['call_start_sample']+plan['call_end_sample'])//2
        old_start = mid-int(round(args.context_ms*info.samplerate/1000.))//2
        old_end = old_start+int(round(args.context_ms*info.samplerate/1000.))
        old_future = max(0,min(old_end,info.frames)-plan['call_end_sample'])
        rows.append({**row,**plan,'actual_sample_rate':info.samplerate,
            'legacy_centered_future_audio_ms':1000*old_future/info.samplerate,
            'legacy_centered_truncates_focal':old_start>plan['call_start_sample'] or old_end<plan['call_end_sample'],
            'normalization_rule':'Allowed samples only; mask and padding stay exactly zero.'})
    audit = pd.DataFrame(rows)
    audit.to_csv(args.output_dir/'input_support_audit.csv',index=False)
    protocol = dict(context_ms=args.context_ms,limit_rows=args.limit_rows,modes=MODES,
        script_sha256=digest(Path(__file__)),manifest_sha256=digest(manifest_path),
        cohort_sha256=digest(cohort_path),chunk_size=args.chunk_size,
        boundary_rule='End at focal offset; start=min(call onset, offset-minimum context). Full call retained.',
        sample_rate_rule='Native sample rate, same as legacy; checkpoint compatibility checked on extraction.',
        normalization='Mode-specific allowed samples only; zero positions remain zero.',
        pooling='Mean in focal positions, top 12 layers, same proportional frame mapping as legacy.',
        caveat='Identical timing/padding; context-only call positions contain zeros but attend to history.',
        pre_caveat='NF_pre is retrodiction; past_context/context_only can directly observe its time range. Not a clean PRE negative control.')
    pp = args.output_dir/'input_protocol.json'
    if pp.exists() and json.loads(pp.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Audit protocol changed; choose another directory.')
    pp.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    summary = dict(n_calls=len(audit),n_clips=audit.source_id.nunique(),
        n_legacy_future_exposed=int(audit.legacy_centered_future_audio_ms.gt(0).sum()),
        legacy_future_ms_median=float(audit.legacy_centered_future_audio_ms.median()),
        n_legacy_focal_truncated=int(audit.legacy_centered_truncates_focal.sum()),
        new_future_samples_allowed=0,n_left_padded=int(audit.left_pad_samples.gt(0).sum()),
        n_context_only_no_real_context=int(audit.focal_start_sample.eq(audit.left_pad_samples).sum()),
        extraction_started=bool(args.extract))
    (args.output_dir/'audit_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    if not args.extract:
        print('Support audit only; no encoder loaded. Add --extract for the long task.')
        return
    if not args.animal2vec_root or not args.checkpoint_path:
        p.error('--extract requires --animal2vec-root and --checkpoint-path')
    from extract_animal2vec_embeddings import load_animal2vec_model, extract_frame_embeddings, pool_focal_frames
    extraction = dict(checkpoint=str(args.checkpoint_path.resolve()),
        checkpoint_sha256=digest(args.checkpoint_path),source_root=str(args.animal2vec_root.resolve()),
        device=args.device)
    ep = args.output_dir/'extraction_protocol.json'
    if ep.exists() and json.loads(ep.read_text(encoding='utf-8')) != extraction:
        raise ValueError('Checkpoint/source/device changed; choose another output directory.')
    ep.write_text(json.dumps(extraction,indent=2),encoding='utf-8')
    model, torch = load_animal2vec_model(args.animal2vec_root,args.checkpoint_path,args.device)
    checkpoint_rate = getattr(getattr(model,'cfg',None),'sample_rate',None)
    if checkpoint_rate is not None and set(audit.actual_sample_rate)!={int(checkpoint_rate)}:
        raise ValueError('Checkpoint sample rate mismatch; do not silently resample.')
    chunks = args.output_dir/'chunks'
    chunks.mkdir(exist_ok=True)
    for start in range(0,len(audit),args.chunk_size):
        batch = audit.iloc[start:start+args.chunk_size]
        name = f'batch_{start:06d}'
        if all((chunks/f'{name}_{mode}.npz').exists() for mode in MODES) and (chunks/f'{name}_metadata.csv').exists():
            for mode in MODES:
                existing = np.load(chunks/f'{name}_{mode}.npz',allow_pickle=False)
                assert np.array_equal(existing['call_ids'],batch.call_id.to_numpy(str))
            print(f'reused {start+len(batch)}/{len(audit)}',flush=True)
            continue
        vectors, metadata = {mode:[] for mode in MODES}, []
        for row in batch.to_dict('records'):
            window = read_allowed_window(row['source_wav'],row)
            for mode in MODES:
                audio, norm = masked_normalize(window,row['focal_start_sample'],
                    row['focal_end_sample'],row['left_pad_samples'],mode)
                frames = extract_frame_embeddings(model,torch,audio,args.device,None,12)
                pooled, sf0, sf1 = pool_focal_frames(frames,row['focal_start_sample'],row['focal_end_sample'],len(audio),'mean')
                if not np.isfinite(pooled).all():
                    raise ValueError('Nonfinite embedding: '+row['call_id'])
                vectors[mode].append(pooled)
                metadata.append({**row,**norm,'mode':mode,'n_frames':len(frames),
                    'focal_start_frame':sf0,'focal_end_frame':sf1,
                    'focal_frame_ratio':(sf1-sf0)/len(frames)})
        for mode in MODES:
            np.savez_compressed(chunks/f'{name}_{mode}.npz',X=np.stack(vectors[mode]),
                call_ids=batch.call_id.to_numpy(str),labels=batch.label.to_numpy(str))
        pd.DataFrame(metadata).to_csv(chunks/f'{name}_metadata.csv',index=False)
        print(f'extracted {start+len(batch)}/{len(audit)} x 3 modes',flush=True)
    for mode in MODES:
        files = sorted(chunks.glob(f'batch_*_{mode}.npz'))
        arrays = [np.load(f,allow_pickle=False) for f in files]
        x = np.concatenate([a['X'] for a in arrays])
        ids = np.concatenate([a['call_ids'] for a in arrays])
        assert np.array_equal(ids,audit.call_id.to_numpy(str))
        np.savez_compressed(args.output_dir/f'animal2vec_causal_{mode}.npz',X=x,call_ids=ids,
            labels=audit.label.to_numpy(str),amplitude_variants=np.full(len(ids),'masked_zscore'))
    pd.concat([pd.read_csv(f) for f in sorted(chunks.glob('*_metadata.csv'))],ignore_index=True).to_csv(
        args.output_dir/'extraction_metadata.csv',index=False)
    print('Extraction complete. Compare using the same frozen clip folds; no temporal-response claim.')


if __name__ == '__main__':
    main()
