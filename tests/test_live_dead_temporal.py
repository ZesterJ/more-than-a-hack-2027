"""Synthetic tests of temporal transforms, scoring and nested split isolation."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'starter'))
import live_dead_baseline as a
import live_dead_temporal as t
import evaluate_live_dead_temporal as runner


def run(p, config, period=None, coverage=None):
    n=len(p)
    return t.process(p,np.ones(n) if period is None else period,
                     np.ones(n,dtype=bool) if coverage is None else coverage,config)


def runs(d=0,l=0):
    return dict(family='runs',dead_min=d,live_min=l)


def test_frozen_grid():
    cs=t.candidates()
    assert len(cs)==22 and len({c['id'] for c in cs})==22
    assert sum(c['family']=='runs' for c in cs)==15
    assert cs[0]['id']=='raw'
    assert t.TIE_TOLERANCE==1e-6 and t.MACRO_GUARDRAIL==.001


def test_strict_length_and_edges():
    assert run([0,1,1,0],runs(d=3)).tolist()==[False]*4
    assert run([0,1,1,1,0],runs(d=3)).tolist()==[False,True,True,True,False]
    assert run([1,0,0,1],runs(l=3)).tolist()==[True]*4
    assert run([1,0,1],runs(d=9)).tolist()==[True,False,True]
    assert run([0,1,0],runs(l=10)).tolist()==[False,True,False]


def test_snapshot_passes_live_first_then_dead():
    # The LIVE fill creates a longer DEAD run; second pass sees that snapshot.
    assert run([0,1,0,1,0],runs(d=3,l=3)).tolist()==[False,True,True,True,False]
    assert run([0,1,0,1,0],runs(d=5,l=3)).tolist()==[False]*5
    # One snapshot pass changes both separate bursts, without depending on visit order.
    np.testing.assert_array_equal(t.replace_internal(np.array([0,1,0,1,0],dtype=bool),True,3),False)


@pytest.mark.parametrize('config',[runs(d=9,l=10),dict(family='smooth',window=9),
                                  dict(family='hysteresis',exit=.3,enter=.7)])
def test_period_and_coverage_resets(config):
    p=np.array([.9,.9,.4,.4,.9,.9])
    result=run(p,config,period=[1,1,2,2,3,3])
    expected=np.r_[run(p[:2],config),run(p[2:4],config),run(p[4:],config)]
    np.testing.assert_array_equal(result,expected)
    covered=run(p,config,coverage=[1,1,0,1,1,1])
    expected=np.r_[run(p[:2],config),p[2]>=.5,run(p[3:],config)]
    np.testing.assert_array_equal(covered,expected)


def test_smoothing_exact_truncated_windows():
    p=[0,1,0,0,1]
    expected=np.array([.5,1/3,1/3,1/3,.5])>=.5
    np.testing.assert_array_equal(run(p,dict(family='smooth',window=3)),expected)


def test_hysteresis_inclusive_thresholds_and_initialisation():
    p=[.5,.5,.4,.6,.5,.4]
    assert run(p,dict(family='hysteresis',exit=.4,enter=.6)).tolist()==[True,True,False,True,True,False]
    assert run([.49,.59,.6],dict(family='hysteresis',exit=.4,enter=.6)).tolist()==[False,False,True]


def test_no_mutation_and_bad_inputs():
    p=np.array([.1,.9,.1]);original=p.copy()
    for c in t.candidates():run(p,c)
    np.testing.assert_array_equal(p,original)
    assert len(run([],dict(family='raw')))==0
    for p in ([np.nan],[1.1],[-.1]):
        with pytest.raises(ValueError):run(p,dict(family='raw'))
    with pytest.raises(ValueError):t.spans([1,1],[True])


def test_hard_score_parity_and_fast_selection_parity():
    rng=np.random.RandomState(2)
    for _ in range(8):
        y=rng.randint(0,4,200);P=rng.dirichlet([1,1,1],200);period=np.repeat([1,2],100)
        r=a.evaluate_match(y,P,period)
        assert a.evaluate_labels(y,P[:,2]>=.5,period)==r
        fast=t.selection_metrics(y,P[:,2]>=.5,period)
        assert fast['macro_f1']==r['macro_f1']
        assert fast['interval_f1']==r['dead_interval_mean_f1']


def test_guardrail_and_effective_ties():
    cs=t.candidates();rows={c['id']:dict(macro_f1=.99,interval_f1=.8) for c in cs}
    rows[cs[1]['id']]['interval_f1']=.8+5e-7
    assert t.select_config(rows)['id']=='raw'
    rows[cs[1]['id']]['interval_f1']=.9
    rows[cs[1]['id']]['macro_f1']=.988
    assert t.select_config(rows)['id']=='raw'
    rows[cs[1]['id']]['macro_f1']=.989
    assert t.select_config(rows)['id']==cs[1]['id']


def test_inner_selection_rejects_outer_reference():
    ref=dict(y=np.array([0,2,0]),period=np.ones(3),coverage=np.ones(3))
    with pytest.raises(ValueError):t.score_inner('held',{'held':np.array([0,1,0])},{'held':ref})
    scores,chosen=t.score_inner('held',{'inner':np.array([0,1,0])},{'inner':ref})
    assert scores['raw']['macro_f1']==1 and chosen['id']=='raw'


def test_all_inner_pair_fits_exclude_both_matches_and_mirrors():
    import itertools
    sets={m:dict(X=np.full((6,78),i),aug=[('mirror_'+m,[])]) for i,m in enumerate(a.MATCHES)}
    seen=[]
    def fit(name, training, matches, weights):
        assert name=='lightgbm' and weights is None
        assert set(training)==set(matches) and len(matches)==5
        seen.append(set(matches))
        return SimpleNamespace(classes_=np.arange(3),predict_proba=lambda X:np.tile([.2,.3,.5],(len(X),1)))
    for pair in itertools.combinations(a.MATCHES,2):
        posts,train=runner.fit_pair(sets,pair,SimpleNamespace(fit=fit))
        assert set(posts)==set(pair)
        assert not set(pair)&set(train)
        for x in posts.values():assert x.shape==(6,3)
    assert len(seen)==21 and len({tuple(sorted(s)) for s in seen})==21


def test_diagnostics_distinguish_bursts_gaps_and_removed_short_states():
    y=np.array([0,0,0,2,2,2,0,2,0])
    pred=np.array([0,1,0,1,0,1,0,0,0],dtype=bool)
    r=t.diagnostics(y,pred,np.ones(len(y)))
    assert r['false_dead_internal_bursts']==1
    assert r['false_live_internal_gaps']==1
    assert r['true_live_runs_split']==1 and r['true_dead_runs_split']==1
    assert r['genuine_short_dead_removed']==1


def test_temporal_source_has_no_competition_entrypoints():
    import ast
    banned={'load_gt','load_tracking','read_label_csv','ball_unseen_pattern','target_features',
            'domain_weights','decode_pipeline','tune_decoder','fetch_idsse','predict_and_write'}
    for module in (t,runner):
        tree=ast.parse(Path(module.__file__).read_text())
        calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree)
               if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
        assert not calls&banned


def test_provenance_allows_docs_edits_but_rejects_classifier_changes(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr(a, 'ROOT', tmp_path)
    hashes = {}
    for rel in runner.CLASSIFIER_SOURCES:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original classifier source\n")
        hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes['experiments/live_dead_baseline.md'] = 'historical documentation hash'
    manifest = {'source_sha256': hashes}
    runner.verify_classifier_sources(manifest)
    doc = tmp_path / 'experiments/live_dead_baseline.md'
    doc.parent.mkdir(parents=True)
    doc.write_text('Updated measured results; classifier unchanged.\n')
    runner.verify_classifier_sources(manifest)
    (tmp_path / runner.CLASSIFIER_SOURCES[0]).write_text('changed classifier\n')
    with pytest.raises(ValueError, match='classifier/scorer source changed'):
        runner.verify_classifier_sources(manifest)
