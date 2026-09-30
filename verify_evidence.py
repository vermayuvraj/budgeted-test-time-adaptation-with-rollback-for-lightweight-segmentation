"""Independent CPU audit of recorded metrics; no model or test threshold tuning."""
from pathlib import Path
import json,hashlib,math,statistics

BASE=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
def check_counts(r):
    tp,pred,fg,n=(r[k] for k in ['intersection','predicted','foreground','pixels'])
    assert 0<=tp<=min(pred,fg)<=max(pred,fg)<=n
    close(r['dice'],(2*tp+1e-6)/(pred+fg+1e-6))
    close(r['iou'],(tp+1e-6)/(pred+fg-tp+1e-6))

def main():
    audits={}
    for domain,counts in [('clinic',(489,61,62)),('mass',(137,4,10))]:
        run=BASE/'runs'/f'{domain}_seed42'
        config=read(run/'config.json');result=read(run/'results.json')
        history=[json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
        assert [r['epoch'] for r in history]==list(range(1,201))
        assert config['epochs']==200 and config['seed']==42
        assert result['parameter_count']==315566
        assert config['architecture_sha256']==hashlib.sha256((BASE/'MK-UNet/mkunet_network.py').read_bytes()).hexdigest()
        assert all(math.isfinite(r[k]) for r in history for k in ['loss','val_dice','val_iou','seconds'])
        best=max(history,key=lambda r:r['val_dice'])
        assert result['best_epoch']==best['epoch'];close(result['best_val_dice'],best['val_dice'])
        checkpoint_sha=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()
        assert checkpoint_sha==result['checkpoint_sha256']
        manifest=read(run/'data_manifest.json')
        hashes=[];files_checked=0
        for split,n in zip(['train','val','test'],counts):
            assert len(manifest[split])==n
            ids={r['image_sha256'] for r in manifest[split]}
            assert len(ids)==n
            assert all(ids.isdisjoint(other) for other in hashes)
            hashes.append(ids)
            for record in manifest[split]:
                for kind in ['image','mask']:
                    path=BASE/record[kind].replace('\\','/')
                    if path.exists():
                        assert hashlib.sha256(path.read_bytes()).hexdigest()==record[f'{kind}_sha256']
                        files_checked+=1
        for variant in ['native_raw','native_minmax']:
            metric=result[variant];records=metric['per_image']
            assert len(records)==counts[2]
            for r in records:check_counts(r)
            close(metric['mean_dice'],statistics.mean(r['dice'] for r in records))
            close(metric['mean_iou'],statistics.mean(r['iou'] for r in records))
            tp=sum(r['intersection'] for r in records);p=sum(r['predicted'] for r in records);g=sum(r['foreground'] for r in records)
            close(metric['global_iou'],(tp+1e-6)/(p+g-tp+1e-6))
        pilot=read(run/'pilot/summary.json');cases=read(run/'pilot/per_case.json')
        provenance=read(run/'pilot/provenance.json')
        assert provenance['checkpoint_sha256']==checkpoint_sha
        assert provenance['script_sha256']==hashlib.sha256((BASE/'prototype_evaluation.py').read_bytes()).hexdigest()
        assert len(cases)==counts[2]*5
        protocol=pilot['protocol']
        native={r['name']:r['dice'] for r in result['native_raw']['per_image']}
        native_counts={r['name']:r for r in result['native_raw']['per_image']}
        for summary in pilot['conditions']:
            rows=[r for r in cases if r['condition']==summary['condition']]
            assert len(rows)==counts[2]
            assert {r['name'] for r in rows}==set(native)
            for r in rows:
                expected=(r['js_after']<=r['js_before']-protocol['accept_js_improvement'] and r['area_change']<=protocol['max_area_change'])
                assert r['accepted']==expected
                for variant in ['baseline','candidate','gated']:check_counts(r[variant])
                assert r['gated']==r['candidate' if expected else 'baseline']
                if summary['condition']=='clean':
                    # Evaluation batch sizes differ; binary masks can vary at roundoff-sensitive pixels.
                    assert r['baseline']['foreground']==native_counts[r['name']]['foreground']
                    assert r['baseline']['pixels']==native_counts[r['name']]['pixels']
                    assert abs(r['baseline']['dice']-native[r['name']])<=1e-3
            for variant in ['baseline','candidate','gated']:
                close(summary[f'{variant}_dice'],statistics.mean(r[variant]['dice'] for r in rows))
            assert summary['accepted']==sum(r['accepted'] for r in rows)
            assert summary['harmful_accepted']==sum(r['accepted'] and r['candidate']['dice']<r['baseline']['dice']-protocol['harm_dice_threshold'] for r in rows)
            assert summary['harmful_accepted_any']==sum(r['accepted'] and r['candidate']['dice']<r['baseline']['dice']-1e-4 for r in rows)
            assert summary['harmful_candidates']==sum(r['candidate']['dice']<r['baseline']['dice']-protocol['harm_dice_threshold'] for r in rows)
            close(summary['delta_ci']['mean'],summary['gated_dice']-summary['baseline_dice'])
        clean_delta=max(abs(r['baseline']['dice']-native[r['name']]) for r in cases if r['condition']=='clean')
        audits[domain]=dict(status='PASS',epochs=200,split_sizes=counts,best_epoch=best['epoch'],checkpoint_sha256=checkpoint_sha,data_files_hashed=files_checked,metric_records_checked=counts[2]*2,pilot_cases_checked=len(cases),max_clean_batching_dice_difference=clean_delta,clean_batching_tolerance=1e-3,checks=['complete epoch sequence','validation-only checkpoint selection','checkpoint and architecture hashes','exact-image split disjointness','available dataset-file hashes','native Dice and IoU from integer counts','aggregate metrics','label-free gate decisions','gated output selection','identical clean ground-truth counts','clean pilot versus main evaluation within recorded numerical tolerance','harm counts'])
    (BASE/'evidence_audit.json').write_text(json.dumps(audits,indent=2))
    print(json.dumps(audits,indent=2))

if __name__=='__main__':main()
