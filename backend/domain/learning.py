from functools import lru_cache
from hashlib import sha256
from json import dumps, loads
from random import Random

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from domain.states import ANOMALY_ARTIFACTS, BASELINE_MODEL_VERSION, Artifact

FEATURES = ('edge_loss', 'color_error', 'flicker_index', 'frame_var')
MODEL_VERSION = BASELINE_MODEL_VERSION
TRAIN_SEED = 411
TRAIN_SIZE = 55
HOLDOUT_SEED = 276
HOLDOUT_SIZE = 15
CENTERS = {
    Artifact.CLEAN: (.10,.11,.10,.09),
    Artifact.SOFT: (.76,.12,.13,.19),
    Artifact.SHIFT: (.13,.77,.18,.20),
    Artifact.FLICKER: (.18,.15,.81,.70),
}


def simulated_data(seed, size):
    rng = Random(seed)
    rows, labels = [], []
    for label, center in CENTERS.items():
        for _ in range(size):
            rows.append([max(0,min(1,rng.gauss(mean, .065))) for mean in center])
            labels.append(str(label))
    return rows, labels


def holdout_data():
    """The single acceptance holdout used by both sides of a replacement."""
    return simulated_data(HOLDOUT_SEED, HOLDOUT_SIZE)


def make_classifier():
    return ExtraTreesClassifier(n_estimators=96,max_depth=7,min_samples_leaf=2,
                                random_state=221,n_jobs=1)


def fit_model(samples, labels):
    model = make_classifier()
    model.fit(samples, labels)
    return model


def holdout_digest(check=None, known=None):
    if check is None or known is None:
        check, known = holdout_data()
    raw = dumps({'x': check, 'y': [str(x) for x in known]}, sort_keys=True,
                separators=(',', ':')).encode()
    return sha256(raw).hexdigest()[:16]


def evaluate_report(model, check=None, known=None):
    if check is None or known is None:
        check, known = holdout_data()
    predicted = model.predict(check)
    labels = [str(label) for label in model.classes_]
    matrix = confusion_matrix(known, predicted, labels=model.classes_).tolist()
    missed = false_negative_counts({'labels': labels, 'matrix': matrix})
    return {
        'training': getattr(model, 'training_rows_', None),
        'holdout': len(check),
        'holdout_seed': HOLDOUT_SEED,
        'holdout_size': HOLDOUT_SIZE,
        'holdout_digest': holdout_digest(check, known),
        'accuracy': float(accuracy_score(known, predicted)),
        'labels': labels,
        'matrix': matrix,
        'false_negatives': missed,
        'missed': missed,
    }


def false_negative_counts(report):
    """Missed cases per true class: row total minus its diagonal cell."""
    labels = [str(label) for label in report['labels']]
    matrix = report['matrix']
    result = {}
    for index, label in enumerate(labels):
        row = matrix[index]
        result[label] = int(sum(row) - row[index])
    return result


def compare_holdout(candidate_report, active_report):
    candidate_fn = false_negative_counts(candidate_report)
    active_fn = false_negative_counts(active_report)
    checks = {}
    regressions = []
    for label in map(str, ANOMALY_ARTIFACTS):
        old = int(active_fn.get(label, 0))
        new = int(candidate_fn.get(label, 0))
        increased = new > old
        checks[label] = {'active': old, 'candidate': new, 'before': old, 'after': new,
                         'increase': new - old, 'delta': new - old,
                         'increased': increased, 'regressed': increased}
        if increased:
            regressions.append(label)
    candidate_digest = candidate_report.get('holdout_digest')
    active_digest = active_report.get('holdout_digest')
    if candidate_digest and active_digest:
        same_set = candidate_digest == active_digest
    else:
        # Older report payloads predate the digest.  Both built-in and new
        # versions use the same fixed seed/size, so matching counts/seeds are
        # sufficient when no digest was persisted.
        same_set = (candidate_report.get('holdout', 60) == active_report.get('holdout', 60)
                    and candidate_report.get('holdout_seed', HOLDOUT_SEED)
                    == active_report.get('holdout_seed', HOLDOUT_SEED))
    eligible = same_set and not regressions
    return {
        'eligible': eligible,
        'allowed': eligible,
        'passed': eligible,
        'can_activate': eligible,
        'can_enable': eligible,
        'activation_allowed': eligible,
        'blocked': not eligible,
        'same_holdout': same_set,
        'regressions': regressions,
        'false_negative_regressions': regressions,
        'checks': checks,
        'false_negatives': {'active': active_fn, 'candidate': candidate_fn},
        'matrices': {'active': active_report.get('matrix'),
                     'candidate': candidate_report.get('matrix')},
        'accuracy': {'active': active_report.get('accuracy'),
                     'candidate': candidate_report.get('accuracy')},
        'active_accuracy': active_report.get('accuracy'),
        'candidate_accuracy': candidate_report.get('accuracy'),
        'reason': ('同一留出集上任一异常类漏判增加，已阻止启用'
                  if regressions else '同一留出集上各异常类漏判未增加'),
    }


def train_bundle(custom_rows=(), custom_labels=(), include_baseline=True):
    samples, labels = simulated_data(TRAIN_SEED, TRAIN_SIZE) if include_baseline else ([], [])
    samples.extend([[float(value) for value in row] for row in custom_rows])
    labels.extend([str(label) for label in custom_labels])
    if not samples or not labels:
        raise ValueError('至少需要一条训练读数')
    model = fit_model(samples, labels)
    report = evaluate_report(model)
    report['training'] = len(samples)
    report['baseline_training'] = 220 if include_baseline else 0
    report['new_samples'] = len(custom_rows)
    snapshot = {'include_baseline': include_baseline,
                'rows': [[float(value) for value in row] for row in custom_rows],
                'labels': [str(label) for label in custom_labels]}
    return {'model': model, 'report': report, 'snapshot': snapshot}


def compare_reports(candidate_report, active_report):
    return compare_holdout(candidate_report, active_report)


def train_model(readings=None, labels=None, include_baseline=True):
    """Flexible training helper used by both HTTP and direct unit callers."""
    if readings is None:
        rows, text_labels = [], []
    elif labels is not None:
        rows = [list(row) for row in readings]
        text_labels = list(labels)
    elif readings and isinstance(readings[0], dict):
        rows = [[float(reading[field]) for field in FEATURES] for reading in readings]
        text_labels = [str(reading.get('artifact') or reading.get('label')
                           or reading.get('actual_artifact')) for reading in readings]
    else:
        # A sequence of (feature_row, label) pairs is also accepted.
        rows = [list(item[0]) for item in readings]
        text_labels = [str(item[1]) for item in readings]
    bundle = train_bundle(rows, text_labels, include_baseline=include_baseline)
    return bundle['model'], bundle['report']


@lru_cache(maxsize=1)
def fitted():
    bundle = train_bundle()
    return bundle['model'], bundle['report']


@lru_cache(maxsize=64)
def bundle_from_snapshot(version, snapshot_json):
    snapshot = loads(snapshot_json)
    return train_bundle(snapshot.get('rows', []), snapshot.get('labels', []),
                        snapshot.get('include_baseline', True))


def calculate(payload, model_version=None, snapshot=None):
    version = model_version or payload.get('model_version') or MODEL_VERSION
    if snapshot is None and version == MODEL_VERSION:
        model, report = fitted()
    elif snapshot is not None:
        if isinstance(snapshot, str):
            bundle = bundle_from_snapshot(version, snapshot)
        else:
            bundle = train_bundle(snapshot.get('rows', []), snapshot.get('labels', []),
                                  snapshot.get('include_baseline', True))
        model, report = bundle['model'], bundle['report']
    else:
        # Allow the historical one-argument worker entry point to resolve a
        # pinned non-baseline version from the registry.
        from domain.quality import snapshot_for_calculation
        stored = snapshot_for_calculation(version)
        if stored is None:
            raise LookupError(f'model version unavailable: {version}')
        encoded = dumps(stored, sort_keys=True, separators=(',', ':'))
        model, report = bundle_from_snapshot(version, encoded)
    values = [[payload['sensors'][field] for field in FEATURES]]
    chances = model.predict_proba(values)[0].tolist()
    top = max(range(len(chances)),key=chances.__getitem__)
    return {'model_version': version, 'artifact': str(model.classes_[top]),
            'probability':chances[top],
            'votes':{str(label): chance for label, chance in zip(model.classes_,chances,strict=True)},
            'holdout':report}
