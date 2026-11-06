"""Strictly nested external-only Experiment B. No production inference changes."""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np
import live_dead_baseline as a
import live_dead_temporal as t


def protocol():
    return dict(experiment='B', classifier=a.CONFIG, candidates=t.candidates(),
                objective='equal-weight inner-match mean dead interval F1 over IoUs 0.3–0.7',
                guardrail_absolute_macro_difference=t.MACRO_GUARDRAIL,
                tie_tolerance=t.TIE_TOLERANCE, tie_order=[c['id'] for c in t.candidates()],
                tie_rule='first eligible candidate within tolerance of highest eligible interval score',
                processing='LIVE gap fill snapshot, then DEAD burst removal snapshot; strict lengths; internal runs only',
                coverage='all external frames covered; reject unsupported invalid training labels',
                inner_training='five matches; both excluded match IDs absent from fit dictionary',
                outer_prediction='reuse checked Experiment A six-match OOF arrays',
                short_state_definition='reference state shorter than 25 frames; removed if no correctly predicted frame')


def fit_pair(sets, excluded, pipeline):
    from benchmark import _full_proba
    train = [m for m in a.MATCHES if m not in excluded]
    training = {m: sets[m] for m in train}
    if len(train)!=5 or any(m in training for m in excluded):
        raise ValueError('Invalid inner split')
    model = pipeline.fit('lightgbm', training, matches=train, weights=None)
    return {m: _full_proba(model, sets[m]['X']) for m in excluded}, train


CLASSIFIER_SOURCES = (
    "starter/features.py", "starter/external.py", "starter/external_pipeline.py",
    "starter/tactical.py", "starter/benchmark.py", "starter/baseline.py", "games.py",
    "evaluation/possession_metrics.py", "requirements.txt", "requirements-starter.txt",
)


def verify_classifier_sources(manifest):
    """Require classifier/scorer parity while allowing documentation and test maintenance."""
    for rel in CLASSIFIER_SOURCES:
        expected = manifest['source_sha256'].get(rel)
        if expected is None or a.sha256(a.ROOT / rel) != expected:
            raise ValueError('Experiment A classifier/scorer source changed: ' + rel)


def verify_a(directory):
    m = json.loads((directory/'manifest.json').read_text())
    metrics = json.loads((directory/'metrics.json').read_text())
    if m['status']!='complete' or metrics['status']!='complete' or m['config']!=a.CONFIG:
        raise ValueError('Experiment A protocol/status mismatch')
    # Evaluation-helper behavior is checked below by exact replay of every raw metric.
    # Documentation/test edits must not invalidate otherwise identical A artifacts.
    verify_classifier_sources(m)
    checks = json.loads((directory/'artifact_checksums.json').read_text())
    for rel in ('manifest.json', 'metrics.json'):
        if a.sha256(directory / rel) != checks.get(rel):
            raise ValueError('Experiment A artifact checksum mismatch: ' + rel)
    outer = {}
    for match in a.MATCHES:
        rel=f'posteriors/{match}.npz'
        if a.sha256(directory/rel)!=checks[rel]: raise ValueError('OOF checksum mismatch')
        with np.load(directory/rel) as z:
            outer[match] = {k:z[k] for k in ('P_had','y','period')}
        ref = outer[match]
        actual = a.evaluate_match(ref['y'],ref['P_had'],ref['period'])
        if actual!=metrics['per_match'][match]: raise ValueError('Raw score parity failure: '+match)
    for path,digest in m['data_sha256'].items():
        if a.sha256(path)!=digest: raise ValueError('Experiment A input hash mismatch: '+path)
    return outer, metrics, m


def write_csv(path, results):
    import csv
    rows=[dict(match=m, **a.flatten(r)) for m,r in results.items()]
    fields=['match']+sorted(set().union(*(r.keys() for r in rows))-{'match'})
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def execute(a_dir, output):
    output.mkdir(parents=True,exist_ok=False)
    a.write_json(output/'frozen_protocol.json',protocol())
    manifest=a.source_manifest()
    manifest.update(status='running', experiment='B', experiment_a=str(a_dir.resolve()),
                    protocol_sha256=a.sha256(output/'frozen_protocol.json'))
    for rel in ('starter/live_dead_temporal.py','starter/evaluate_live_dead_temporal.py',
                'tests/test_live_dead_temporal.py','experiments/live_dead_temporal.md'):
        manifest['source_sha256'][rel]=a.sha256(a.ROOT/rel)
    for rel,digest in manifest['source_sha256'].items():
        target=output/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(a.ROOT/rel,target)
    a.write_json(output/'manifest.json',manifest)
    try:
        outer, raw_metrics, a_manifest=verify_a(a_dir)
        print('Raw-score parity: all seven matches exact',flush=True)
        import external_pipeline as ep
        manifest['training_params']=ep.make_lightgbm().get_params()
        manifest['experiment_a_manifest_sha256']=a.sha256(a_dir/'manifest.json')
        manifest['data_sha256']=a_manifest['data_sha256']
        sets=a.load_sets(ep)
        for match,ref in sets.items():
            if ref['X'].shape!=(len(ref['y']),78) or not np.isfinite(ref['X']).all(): raise ValueError('Invalid features')
            if not np.isin(ref['y'],(0,1,2)).all(): raise ValueError('Invalid training references need an explicit coverage policy')
            if not np.array_equal(ref['y'],outer[match]['y']) or not np.array_equal(ref['period'],outer[match]['period']):
                raise ValueError('External reference/cache mismatch')
        cache=output/'inner_posteriors';cache.mkdir()
        pair_paths={}; split_manifest=[]
        for pair in itertools.combinations(a.MATCHES,2):
            print('Inner fit excludes: '+', '.join(pair),flush=True)
            posts,train=fit_pair(sets,pair,ep)
            pair_paths[pair]={}
            for held,P in posts.items():
                a.binary_probabilities(P)
                path=cache/('_'.join(pair)+'__'+held+'.npz')
                np.savez_compressed(path,P_had=P,excluded=np.array(pair),training=np.array(train))
                pair_paths[pair][held]=path
            split_manifest.append(dict(excluded=list(pair),training=train))
        a.write_json(output/'inner_splits.json',split_manifest)
        selections={}; inner_scores={}
        for held in a.MATCHES:
            posts={}
            for other in a.MATCHES:
                if other==held: continue
                pair=tuple(m for m in a.MATCHES if m in (held,other))
                with np.load(pair_paths[pair][other]) as z:
                    posts[other]=z['P_had'][:,2]
            references={m:dict(y=sets[m]['y'],period=sets[m]['period'],coverage=np.ones(len(sets[m]['y']),dtype=bool)) for m in posts}
            rows,chosen=t.score_inner(held,posts,references)
            inner_scores[held]=rows
            family={}
            for name in ('runs','smooth','hysteresis'):
                configs=[c for c in t.candidates() if c['family'] in ('raw',name)]
                family[name]=t.select_config(rows,configs)
            selections[held]=dict(selected=chosen,family_comparisons=family,
                                  inner_matches=list(posts),selected_inner_scores=rows[chosen['id']])
            print('Selected '+held+': '+chosen['id'],flush=True)
        # Freeze every choice before evaluating any outer outcome.
        a.write_json(output/'inner_scores.json',inner_scores)
        a.write_json(output/'selections.json',selections)
        del sets
        processed={}; families={k:{} for k in ('runs','smooth','hysteresis')}; diag={}; deltas={}
        predictions=output/'processed_predictions';predictions.mkdir()
        for held,ref in outer.items():
            y,P,period=ref['y'],ref['P_had'],ref['period']
            coverage=np.ones(len(y),dtype=bool)
            chosen=selections[held]['selected']
            pred=t.process(P[:,2],period,coverage,chosen)
            processed[held]=a.evaluate_labels(y,pred,period)
            rawpred=P[:,2]>=.5
            rawdiag=t.diagnostics(y,rawpred,period);procdiag=t.diagnostics(y,pred,period)
            # Distinguish newly removed genuine states from pre-existing raw misses.
            newly={'dead':0,'live':0}
            for s,e in t.spans(period,coverage):
                for b,c,state in a.pm.segments(y==2,[(s,e)]):
                    if c-b<25 and np.any(rawpred[b:c]==state) and not np.any(pred[b:c]==state):
                        newly['dead' if state else 'live']+=1
            diag[held]=dict(raw=rawdiag,processed=procdiag,newly_removed_short_states=newly)
            before=a.flatten(raw_metrics['per_match'][held]);after=a.flatten(processed[held])
            deltas[held]={k:after[k]-v for k,v in before.items() if k in after}
            np.savez_compressed(predictions/(held+'.npz'),dead=pred,raw_dead=rawpred,period=period,coverage=coverage)
            for family,config in selections[held]['family_comparisons'].items():
                codes=t.process(P[:,2],period,coverage,config)
                families[family][held]=a.evaluate_labels(y,codes,period)
        pooled_diag={key:{k:sum(d[key][k] for d in diag.values()) for k in next(iter(diag.values()))[key]}
                     for key in ('raw','processed','newly_removed_short_states')}
        result=dict(status='complete',raw=raw_metrics,processed=dict(per_match=processed,aggregate=a.aggregate(processed)),
                    family_comparisons={name:dict(per_match=rs,aggregate=a.aggregate(rs)) for name,rs in families.items()},
                    per_match_changes=deltas,diagnostics=dict(per_match=diag,pooled=pooled_diag))
        a.write_json(output/'metrics.json',result)
        write_csv(output/'per_match_processed.csv',processed)
        manifest.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat(),
                        selections_sha256=a.sha256(output/'selections.json'),raw_parity='exact across all seven matches',
                        inner_fits=21,inner_prediction_arrays=42)
        for command,name in (([sys.executable,'-m','pip','freeze'],'pip-freeze.txt'),):
            (output/name).write_text(subprocess.check_output(command,text=True))
        return 0
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc))
        (output/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        a.write_json(output/'manifest.json',manifest)
        if manifest['status'] == 'complete':
            a.write_artifact_checksums(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-a',type=Path,default=a.ROOT/'data/experiment_a/football_ml_external_run_001')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(execute(args.experiment_a,args.output))
