"""Export allowlisted aggregate results and minimal native-run provenance only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


RUNS = {
    'b3': ('baseline_capacity_b3_v1','MK-BASECAP-B3-FIT-01','20261104T110805Z_bf1830d4',
        ['comparison_metrics_and_ci.csv','paired_changes_vs_linear.csv','fold_metrics.csv','verification.json']),
    'nested_c': ('nested_logistic_regularization400_v1','MK-NESTED-C400-FIT-01','20261105T044235Z_79dd31a9',
        ['comparison_metrics_and_ci.csv','paired_changes_vs_fixed.csv','verification.json']),
    'pca': ('pca_sensitivity400_v1','MK-PCA-S1-FIT-01','20261105T075923Z_79f7da56',
        ['comparison_metrics_and_ci.csv','paired_changes_and_ci.csv','verification.json']),
    'config': ('config_source_attribution400_v1','MK-CONFIG-ATTR-FIT-01','20261105T172717Z_05bbf0c3',
        ['paired_metrics_and_ci.csv','three_model_metrics.csv','fold_metrics.csv','verification.json']),
    'reverse_prepare': ('reverse_config_ablation400_v1','MK-REVERSE-CONFIG-PREPARE-01','20261106T055152Z_2c2b1096',
        ['prepare_check.json']),
    'reverse': ('reverse_config_ablation400_v1','MK-REVERSE-CONFIG-FIT-01','20261106T061256Z_1e85d716',
        ['paired_metrics_and_ci.csv','four_model_metrics.csv','predictive_interaction_and_ci.csv',
         'fold_metrics.csv','verification.json']),
}
AUDITS = {
    'b3_audit': ('baseline_capacity_b3_analysis','MK-BASECAP-B3-ANALYSIS-01','20261104T131756Z_a74b219e'),
    'nested_c_audit': ('nested_logistic_regularization400_analysis','MK-NESTED-C400-ANALYSIS-01','20261105T051609Z_24a85998'),
    'pca_audit': ('pca_sensitivity400_analysis','MK-PCA-S1-ANALYSIS-01','20261105T112504Z_bfd6e016'),
    'config_audit': ('config_source_attribution400_analysis','MK-CONFIG-ATTR-ANALYSIS-01','20261105T175316Z_e4ef62b6'),
    'reverse_audit': ('reverse_config_ablation400_analysis','MK-REVERSE-CONFIG-ANALYSIS-01','20261106T062124Z_635e885d'),
}
PROTOCOL_KEYS = ['kind','version','window_ms','n_calls','n_clips','n_valid','families',
    'learner_configs','measured_controls','numeric_by_target','acoustic_columns','M1_fields',
    'config_fields','remaining_fields','input_columns_M78','removed_config_fields','pca',
    'modes','new_modes','comparisons','predictive_interaction','bootstrap','evaluation',
    'interpretation','limitation','exploratory','practical_margin','independent_confirmation','environment']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_aggregate(path):
    if path.suffix=='.csv':
        with path.open(encoding='utf-8-sig',newline='') as handle:
            columns = next(csv.reader(handle))
        forbidden = {'call_id','_clip','clip_id','source_id','recording','individual','y_true',
            'm0_pred','m1_pred','mconfig_pred','m78_pred','path','metadata_path','reference_path'}
        if forbidden.intersection(columns):
            raise ValueError('Row-level/source-path export refused: '+path.name)
    text = path.read_text(encoding='utf-8-sig')
    if any(s in text for s in ['C:\\Users\\','C:/Users/','C:\\\\Users\\\\','BEGIN PRIVATE KEY']):
        raise ValueError('Local user path/private material export refused: '+path.name)


def native_record(private,run_id,execution):
    receipt = json.loads((private/'logged_runs'/run_id/execution/'receipt.json').read_text(encoding='utf-8'))
    if receipt['status']!='COMPLETED' or receipt['exit_code']!=0:
        raise ValueError('Require successful native receipt: '+run_id)
    keys = ['run_id','execution_id','kind','status','exit_code','command_started_utc',
        'command_finished_utc','command_started_jst','command_finished_jst']
    return {k:receipt[k] for k in keys if k in receipt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Use a new snapshot output;do not overwrite published evidence.')
    records,copies = [],[]
    for name,(folder,run_id,execution,files) in RUNS.items():
        source = args.private_root/'experiments'/folder
        protocol_path = source/'protocol.json'
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        record = native_record(args.private_root,run_id,execution)
        record.update(name=name,source_protocol_sha256=digest(protocol_path),artifacts=[])
        for filename in files:
            path = source/filename
            validate_aggregate(path)
            relative = name+'/'+filename
            record['artifacts'].append(dict(file=relative,sha256=digest(path),bytes=path.stat().st_size))
            copies.append((path,relative))
        record['protocol_excerpt'] = {k:protocol[k] for k in PROTOCOL_KEYS if k in protocol}
        record['source_code_hashes'] = {Path(p).name:h for p,h in protocol.get('source_hashes',{}).items()}
        records.append(record)
    for name,(folder,run_id,execution) in AUDITS.items():
        path = args.private_root/'experiments'/folder/'verification.json'
        validate_aggregate(path)
        record = native_record(args.private_root,run_id,execution)
        relative = name+'/verification.json'
        record.update(name=name,artifacts=[dict(file=relative,sha256=digest(path),bytes=path.stat().st_size)])
        records.append(record)
        copies.append((path,relative))
    for path,relative in copies:
        dest = args.output_dir/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
        if digest(dest)!=digest(path):
            raise ValueError('Export hash mismatch.')
    manifest = dict(scope='Aggregate research snapshot;not a clean-machine or independent replication.',
        includes_raw_data=False,includes_row_level_OOF=False,includes_private_logs=False,
        caveat='Native receipts time the named invocation,not original fits of reused predictions. '
        'Protocol excerpts are not the full frozen protocol. Code hashes refer to original file bytes; '
        'Git checkout line endings can change byte hashes.',runs=records)
    (args.output_dir/'provenance.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
    print(json.dumps(dict(exported_files=len(copies),native_runs=len(records),output=str(args.output_dir)),indent=2))


if __name__=='__main__':
    main()
