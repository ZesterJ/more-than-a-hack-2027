"""Fixed, independent temporal families for external-only Experiment B."""
import numpy as np
import live_dead_baseline as a

TIE_TOLERANCE = 1e-6
MACRO_GUARDRAIL = .001


def candidates():
    configs = [dict(id="raw", family="raw")]
    filters = [(d, l) for d in (0, 3, 5, 9) for l in (0, 3, 5, 10) if d or l]
    for d, l in sorted(filters, key=lambda x: (sum(x), max(x), x)):
        configs.append(dict(id=f"runs_d{d}_l{l}", family="runs", dead_min=d, live_min=l))
    configs.extend(dict(id=f"smooth_{w}", family="smooth", window=w) for w in (3, 5, 9))
    configs.extend(dict(id=f"hysteresis_{lo}_{hi}", family="hysteresis", exit=lo, enter=hi)
                   for lo, hi in ((.4, .6), (.3, .7), (.2, .8)))
    return configs


def spans(period, coverage):
    """Observable processing barriers; never takes reference state identities."""
    period, coverage = np.asarray(period), np.asarray(coverage, dtype=bool)
    if period.ndim != 1 or period.shape != coverage.shape:
        raise ValueError("period and coverage must be aligned vectors")
    if not len(period):
        return []
    start = coverage & np.r_[True, (~coverage[:-1]) | (period[1:] != period[:-1])]
    stop = coverage & np.r_[(~coverage[1:]) | (period[1:] != period[:-1]), True]
    return list(zip(np.flatnonzero(start).tolist(), (np.flatnonzero(stop)+1).tolist()))


def replace_internal(codes, state, threshold):
    out = codes.copy()
    if threshold <= 1 or not len(codes):
        return out
    cuts = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1])+1, len(codes)]
    for i in range(1, len(cuts)-2):
        s, e = cuts[i:i+2]
        if codes[s] == state and e-s < threshold:
            out[s:e] = not state
    return out


def process(p_dead, period, coverage, config):
    p = np.asarray(p_dead)
    if p.ndim != 1 or p.shape != np.asarray(period).shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("Invalid probability vector")
    result = p >= .5
    for s, e in spans(period, coverage):
        q = p[s:e]
        codes = q >= .5
        family = config['family']
        if family == 'runs':
            codes = replace_internal(codes, False, config['live_min'])
            codes = replace_internal(codes, True, config['dead_min'])
        elif family == 'smooth':
            h = config['window']//2
            ix = np.arange(len(q)); lo = np.maximum(0, ix-h); hi = np.minimum(len(q), ix+h+1)
            c = np.r_[0., np.cumsum(q)]
            codes = (c[hi]-c[lo])/(hi-lo) >= .5
        elif family == 'hysteresis':
            state = bool(codes[0])
            for i in range(1, len(q)):
                if state and q[i] <= config['exit']:
                    state = False
                elif not state and q[i] >= config['enter']:
                    state = True
                codes[i] = state
        elif family != 'raw':
            raise ValueError("Unknown temporal family")
        result[s:e] = codes
    return result


def selection_metrics(y, pred, period):
    """Fast inner scoring: exactly the two frozen selection metrics."""
    y = np.asarray(y); pred = np.asarray(pred, dtype=bool)
    valid = np.isin(y, (0, 1, 2)); truth = y == 2
    cm = np.bincount(truth[valid].astype(int)*2+pred[valid], minlength=4).reshape(2,2)
    macro = a.frame_metrics(cm)['macro_f1']
    counts = np.zeros((5,3), dtype=int)
    for s,e in spans(period, valid):
        runs = [(s,e)]
        gs = [x for x in a.pm.segments(np.where(truth,2,0), runs) if x[2]==2]
        ps = [x for x in a.pm.segments(np.where(pred,2,0), runs) if x[2]==2]
        overlaps = a.pm.spell_overlaps(gs, ps)
        for i,t in enumerate(a.IOUS):
            counts[i] += a.pm.match_spells(overlaps,len(gs),len(ps),t)
    return dict(macro_f1=macro, interval_f1=float(np.mean([a.prf(*map(int,c))['f1'] for c in counts])))


def select_config(rows, configs=None):
    """Rows contain inner scores only; list order is the frozen simplicity tie-break."""
    configs = candidates() if configs is None else configs
    raw = rows['raw']['macro_f1']
    eligible = [c for c in configs if abs(rows[c['id']]['macro_f1']-raw) <= MACRO_GUARDRAIL+1e-12]
    best = max(rows[c['id']]['interval_f1'] for c in eligible)
    return next(c for c in eligible if rows[c['id']]['interval_f1'] >= best-TIE_TOLERANCE)


def score_inner(outer, inner_posts, references):
    if outer in inner_posts or outer in references or set(inner_posts)!=set(references):
        raise ValueError("Outer match must be excluded from inner selection")
    scores = {c['id']: [] for c in candidates()}
    for match, p in inner_posts.items():
        ref = references[match]
        for c in candidates():
            pred = process(p, ref['period'], ref['coverage'], c)
            scores[c['id']].append(selection_metrics(ref['y'], pred, ref['period']))
    rows = {cid: {metric: float(np.mean([r[metric] for r in rs]))
                  for metric in ('macro_f1','interval_f1')} for cid,rs in scores.items()}
    return rows, select_config(rows)


def diagnostics(y, pred, period):
    y, pred = np.asarray(y), np.asarray(pred,dtype=bool)
    truth = y==2
    result = dict(false_dead_internal_bursts=0, false_live_internal_gaps=0,
                  true_dead_runs_split=0, true_live_runs_split=0,
                  genuine_short_dead_removed=0, genuine_short_live_removed=0,
                  genuine_dead_removed=0, genuine_live_removed=0)
    for s,e in spans(period,np.isin(y,(0,1,2))):
        for x,w,state in a.pm.segments(truth,[(s,e)]):
            parts = a.pm.segments(pred,[(x,w)])
            correct = sum(bool(k)==bool(state) for _,_,k in parts)
            kind = 'dead' if state else 'live'
            if not correct:
                result[f'genuine_{kind}_removed'] += 1
                if w-x<25: result[f'genuine_short_{kind}_removed'] += 1
            if correct>=2: result[f'true_{kind}_runs_split'] += 1
            for b,c,k in parts:
                if bool(k)!=bool(state) and b>x and c<w:
                    result['false_live_internal_gaps' if state else 'false_dead_internal_bursts'] += 1
    return result
