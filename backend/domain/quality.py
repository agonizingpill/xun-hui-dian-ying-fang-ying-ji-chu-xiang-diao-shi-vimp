from datetime import datetime, timezone
from json import dumps
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import func, select

from core.audit import record
from core.db import transaction
from domain.booth_tables import measurements
from domain.learning import (MODEL_VERSION, calculate, compare_holdout, fitted,
                            train_bundle)
from domain.quality_tables import model_activation, model_versions, training_readings
from domain.states import Artifact, BASELINE_MODEL_VERSION

FEATURE_KEYS = ('edge_loss', 'color_error', 'flicker_index', 'frame_var')
FEATURE_ALIASES = {
    'edge_loss': ('edge_loss', 'contour_loss', 'outline_loss', 'edge'),
    'color_error': ('color_error', 'colour_error', 'color', 'colour'),
    'flicker_index': ('flicker_index', 'flicker'),
    'frame_var': ('frame_var', 'frame_variation', 'frame_fluctuation', 'frame'),
}
LABEL_KEYS = ('artifact', 'label', 'actual_artifact', 'true_artifact',
              'actual_label', 'true_label', 'actual', 'category', 'class_name', 'class')
LABEL_VALUES = {str(label): str(label) for label in Artifact}
LABEL_VALUES.update({
    'soft': Artifact.SOFT,
    'shift': Artifact.SHIFT,
    'color': Artifact.SHIFT,
    'flickering': Artifact.FLICKER,
    'pass': Artifact.CLEAN,
    'ok': Artifact.CLEAN,
    '合格': Artifact.CLEAN,
    '画质通过': Artifact.CLEAN,
    '软焦': Artifact.SOFT,
    '色偏': Artifact.SHIFT,
    '色彩偏差': Artifact.SHIFT,
    '闪变': Artifact.FLICKER,
    '亮度闪变': Artifact.FLICKER,
})
LIST_KEYS = ('readings', 'samples', 'items', 'data', 'training_readings',
             'labeled_readings', 'new_readings', 'new_samples', 'records', 'rows')
NESTED_KEYS = ('sensors', 'features', 'reading', 'values', 'metrics')


def now():
    return datetime.now(timezone.utc)


def iso(value):
    return value.isoformat() if isinstance(value, datetime) else value


def serialize(row, snapshot=True):
    if row is None:
        return None
    data = dict(row)
    for key in ('created_at', 'activated_at', 'updated_at'):
        if key in data:
            data[key] = iso(data[key])
    if 'version' in data:
        data['id'] = data['version']
        data['model_version'] = data['version']
    if 'artifact' in data:
        data.setdefault('label', data['artifact'])
        data.setdefault('actual_artifact', data['artifact'])
    if not snapshot:
        data.pop('training_snapshot', None)
    return data


def normalize_label(value):
    if value is None:
        return None
    text = str(value).strip()
    found = LABEL_VALUES.get(text) or LABEL_VALUES.get(text.lower())
    return str(found) if found is not None else None


def _first_key(data, keys):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def extract_sample(data):
    if not isinstance(data, dict):
        raise HTTPException(422, '每条训练读数必须是对象')
    source = data
    for key in NESTED_KEYS:
        nested = data.get(key)
        if isinstance(nested, dict):
            source = {**nested, **data}
            break
        if isinstance(nested, (list, tuple)) and len(nested) >= len(FEATURE_KEYS):
            source = {**dict(zip(FEATURE_KEYS, nested[:len(FEATURE_KEYS)], strict=True)), **data}
            break
    label = normalize_label(_first_key(data, LABEL_KEYS))
    values = {}
    for field in FEATURE_KEYS:
        raw = _first_key(source, FEATURE_ALIASES[field])
        if raw is None:
            values[field] = None
            continue
        if isinstance(raw, bool):
            raise HTTPException(422, f'{field} 必须是 0 到 1 的数值')
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise HTTPException(422, f'{field} 必须是 0 到 1 的数值')
        if not 0 <= value <= 1:
            raise HTTPException(422, f'{field} 必须在 0 到 1 之间')
        values[field] = value
    if label is None:
        raise HTTPException(422, '训练读数缺少有效标签 artifact/label')
    source_code = (_first_key(data, ('measurement_code', 'source_measurement',
                                     'source_measurement_code')))
    code = data.get('code')
    return {'code': code, 'source_measurement': source_code, 'artifact': label, **values}


def extract_samples(body):
    if body is None:
        return []
    if isinstance(body, list):
        if body and not isinstance(body[0], dict):
            raise HTTPException(422, '读数列表项必须是带标签的对象')
        return [extract_sample(item) for item in body]
    if not isinstance(body, dict):
        raise HTTPException(422, '请求体必须是训练读数或读数列表')
    labels = body.get('labels') or body.get('artifact_labels') or body.get('targets')
    for key in ('feature_rows', 'rows', 'values', 'samples', 'readings', 'features'):
        rows = body.get(key)
        if (isinstance(rows, list) and rows and isinstance(rows[0], (list, tuple))
                and isinstance(labels, list) and len(labels) == len(rows)):
            return [extract_sample({**dict(zip(FEATURE_KEYS, row, strict=True)),
                                    'label': label})
                    for row, label in zip(rows, labels, strict=True)]
    for key in LIST_KEYS:
        value = body.get(key)
        if isinstance(value, list):
            return [extract_sample(item) for item in value]
        if isinstance(value, dict):
            return [extract_sample(value)]
    if _first_key(body, LABEL_KEYS) is not None or any(
            _first_key(body, FEATURE_ALIASES[field]) is not None for field in FEATURE_KEYS):
        return [extract_sample(body)]
    return []


def ensure_registry(conn, actor_name='system'):
    existing = conn.execute(select(model_versions).where(
        model_versions.c.version == MODEL_VERSION)).mappings().first()
    if existing is None:
        _, report = fitted()
        gate = compare_holdout(report, report)
        timestamp = now()
        conn.execute(model_versions.insert().values(
            version=MODEL_VERSION, status='active', training_count=report['training'],
            new_sample_count=0, holdout_count=report['holdout'], metrics=report, gate=gate,
            training_snapshot={'include_baseline': True, 'rows': [], 'labels': []},
            sample_ids=[], replaced_version=None, note='初始内置 ExtraTrees 版本',
            created_by=actor_name, created_at=timestamp, activated_at=timestamp,
            updated_at=timestamp))
    state = conn.execute(select(model_activation).where(
        model_activation.c.id == 1)).mappings().first()
    if state is None:
        timestamp = now()
        conn.execute(model_activation.insert().values(
            id=1, active_version=MODEL_VERSION, updated_by=actor_name, updated_at=timestamp))
    return active_version(conn)


def active_version(conn, lock=False):
    statement = select(model_activation.c.active_version).where(model_activation.c.id == 1)
    if lock:
        statement = statement.with_for_update()
    version = conn.execute(statement).scalar_one_or_none()
    if version is None:
        ensure_registry(conn)
        version = conn.execute(statement).scalar_one()
    return version


def _model_row(conn, version, lock=False):
    statement = select(model_versions).where(model_versions.c.version == version)
    if lock:
        statement = statement.with_for_update()
    return conn.execute(statement).mappings().first()


def _require_model(conn, version, lock=False):
    row = _model_row(conn, version, lock)
    if row is None:
        raise HTTPException(404, f'模型版本 {version} 不存在')
    return row


def _complete_source(conn, item):
    source_code = item.get('source_measurement')
    code = item.get('code')
    if source_code is None and code:
        measurement = conn.execute(select(measurements).where(
            measurements.c.code == code)).mappings().first()
        if measurement is not None:
            source_code = code
    if source_code and any(item[field] is None for field in FEATURE_KEYS):
        measurement = conn.execute(select(measurements).where(
            measurements.c.code == source_code)).mappings().first()
        if measurement is None:
            raise HTTPException(404, f'原检测单 {source_code} 不存在')
        for field in FEATURE_KEYS:
            if item.get(field) is None:
                item[field] = float(measurement[field])
    missing = [field for field in FEATURE_KEYS if item.get(field) is None]
    if missing:
        raise HTTPException(422, f'训练读数缺少 {", ".join(missing)}')
    item['source_measurement'] = source_code
    return item


def add_readings(body, user):
    items = extract_samples(body)
    if not items:
        raise HTTPException(422, '请提供至少一条带标签的新读数')
    created = []
    with transaction() as conn:
        for raw in items:
            item = _complete_source(conn, raw)
            values = {field: item[field] for field in FEATURE_KEYS}
            result = conn.execute(training_readings.insert().values(
                code=item.get('code'), artifact=item['artifact'],
                source_measurement=item.get('source_measurement'),
                created_by=user['sub'], created_at=now(), **values))
            sample_id = result.inserted_primary_key[0]
            row = conn.execute(select(training_readings).where(
                training_readings.c.id == sample_id)).mappings().one()
            created.append(serialize(dict(row)))
        record(conn, user, f'model-sample:{len(created)}')
    return {'samples': created, 'readings': created, 'items': created,
            'created': len(created), 'count': len(created),
            'training_samples': count_samples()}


def count_samples():
    with transaction() as conn:
        return int(conn.scalar(select(func.count()).select_from(training_readings)) or 0)


def _load_samples(conn, body):
    ids = body.get('sample_ids') or body.get('ids') or []
    if ids:
        rows = conn.execute(select(training_readings).where(
            training_readings.c.id.in_([int(value) for value in ids])
        ).order_by(training_readings.c.id)).mappings().all()
        found = {row['id'] for row in rows}
        missing = sorted(set(int(value) for value in ids) - found)
        if missing:
            raise HTTPException(404, f'训练读数不存在: {missing}')
    else:
        rows = conn.execute(select(training_readings).order_by(
            training_readings.c.id)).mappings().all()
    return [dict(row) for row in rows]


def train_candidate(body, user):
    body = body if isinstance(body, dict) else {}
    inline = extract_samples(body)
    version = body.get('version') or body.get('model_version')
    if version is not None:
        version = str(version)
        if not 3 <= len(version) <= 64:
            raise HTTPException(422, '模型版本号长度需为 3 到 64 个字符')
    include_baseline = body.get('include_baseline', body.get('baseline', True))
    if isinstance(include_baseline, str):
        include_baseline = include_baseline.lower() not in ('false', '0', 'no', '否')

    with transaction() as conn:
        ensure_registry(conn, user['sub'])
        if inline:
            for raw in inline:
                item = _complete_source(conn, raw)
                conn.execute(training_readings.insert().values(
                    code=item.get('code'), artifact=item['artifact'],
                    source_measurement=item.get('source_measurement'),
                    created_by=user['sub'], created_at=now(),
                    **{field: item[field] for field in FEATURE_KEYS}))
        samples = _load_samples(conn, body)
        rows = [[sample[field] for field in FEATURE_KEYS] for sample in samples]
        labels = [sample['artifact'] for sample in samples]
        bundle = train_bundle(rows, labels, include_baseline=bool(include_baseline))
        current_version = active_version(conn)
        active = _require_model(conn, current_version)
        gate = compare_holdout(bundle['report'], active['metrics'])
        if version is None:
            version = f"cand-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:4]}"
        if _model_row(conn, version) is not None:
            raise HTTPException(409, f'模型版本 {version} 已存在')
        timestamp = now()
        conn.execute(model_versions.insert().values(
            version=version, status='candidate',
            training_count=bundle['report']['training'], new_sample_count=len(samples),
            holdout_count=bundle['report']['holdout'], metrics=bundle['report'], gate=gate,
            training_snapshot=bundle['snapshot'],
            sample_ids=[sample['id'] for sample in samples],
            note=str(body.get('note') or '')[:200] or None,
            created_by=user['sub'], created_at=timestamp, updated_at=timestamp))
        record(conn, user, f'model-train:{version}:{len(samples)}')
        row = _require_model(conn, version)
        result = serialize(dict(row))
        if body.get('activate') or body.get('auto_activate') or body.get('enable'):
            activated = _activate(conn, version, user)
            result['activation'] = activated
        result['candidate'] = {k: value for k, value in result.items()
                               if k not in ('candidate', 'model')}
        result['model'] = result['candidate']
        return result


def list_samples(limit=200):
    with transaction() as conn:
        ensure_registry(conn)
        rows = conn.execute(select(training_readings).order_by(
            training_readings.c.id.desc()).limit(limit)).mappings().all()
        total = conn.scalar(select(func.count()).select_from(training_readings))
        rows_out = [serialize(dict(row)) for row in rows]
        return {'samples': rows_out, 'readings': rows_out, 'items': rows_out,
                'total': int(total or 0), 'count': int(total or 0)}


def quality_board():
    with transaction() as conn:
        ensure_registry(conn)
        version = active_version(conn)
        rows = conn.execute(select(model_versions).order_by(
            model_versions.c.created_at.desc())).mappings().all()
        sample_rows = conn.execute(select(training_readings).order_by(
            training_readings.c.id.desc()).limit(50)).mappings().all()
        sample_total = conn.scalar(select(func.count()).select_from(training_readings))
        models = [serialize(dict(row), snapshot=False) for row in rows]
        active = next((item for item in models if item['version'] == version), None)
        candidates = [item for item in models if item['status'] == 'candidate']
        sample_out = [serialize(dict(row)) for row in sample_rows]
        return {'active_version': version, 'active': active, 'current': active,
                'active_model': active, 'candidates': candidates,
                'baseline_version': BASELINE_MODEL_VERSION,
                'models': models, 'versions': models,
                'training_samples': sample_out, 'readings': sample_out,
                'sample_count': int(sample_total or 0)}


def get_version(version):
    with transaction() as conn:
        ensure_registry(conn)
        row = _require_model(conn, version)
        return serialize(dict(row))


def gate_for_version(version):
    with transaction() as conn:
        ensure_registry(conn)
        current = active_version(conn)
        candidate = _require_model(conn, version)
        active = _require_model(conn, current)
        gate = compare_holdout(candidate['metrics'], active['metrics'])
        return {'candidate_version': version, 'active_version': current,
                'gate': gate, **gate}


def _activate(conn, version, user):
    ensure_registry(conn, user['sub'])
    state_current = active_version(conn, lock=True)
    candidate = _require_model(conn, version, lock=True)
    if state_current == version and candidate['status'] == 'active':
        return {'version': version, 'active_version': version,
                'previous_version': candidate['replaced_version'],
                'model': serialize(dict(candidate)), 'changed': False}
    if candidate['status'] not in ('candidate', 'withdrawn', 'retired'):
        raise HTTPException(409, f'版本 {version} 当前状态为 {candidate["status"]}，不能启用')
    active = _require_model(conn, state_current, lock=True)
    gate = compare_holdout(candidate['metrics'], active['metrics'])
    if not gate['eligible']:
        conn.execute(model_versions.update().where(model_versions.c.version == version)
                     .values(gate=gate, updated_at=now()))
        raise HTTPException(409, detail={'message': '候选版未通过留出集漏判门槛，继续使用旧版',
                                         'gate': gate})
    timestamp = now()
    previous = state_current
    conn.execute(model_versions.update().where(model_versions.c.version == version).values(
        status='active', replaced_version=previous, activated_at=timestamp,
        updated_at=timestamp, gate=gate))
    conn.execute(model_versions.update().where(model_versions.c.version == previous).values(
        status='retired', updated_at=timestamp))
    conn.execute(model_activation.update().where(model_activation.c.id == 1).values(
        active_version=version, updated_by=user['sub'], updated_at=timestamp))
    # One global commit switches every booth's next accepted order together.
    from domain.booth_tables import booths
    conn.execute(booths.update().values(model_version=version))
    record(conn, user, f'model-activate:{version}:{previous}')
    row = _require_model(conn, version)
    model_data = serialize(dict(row))
    return {'version': version, 'active_version': version,
            'previous_version': previous,
            'replaced_version': previous, 'model': model_data,
            'active': model_data, 'current': model_data,
            'changed': True, 'gate': gate}


def activate_version(version, user, body=None):
    with transaction() as conn:
        result = _activate(conn, version, user)
        return result


def activate_named(body, user):
    body = body if isinstance(body, dict) else {}
    version = (body.get('version') or body.get('model_version')
               or body.get('candidate_version') or body.get('id') or body.get('name')
               or body.get('active_version'))
    if not version:
        with transaction() as conn:
            row = conn.execute(select(model_versions).where(
                model_versions.c.status == 'candidate').order_by(
                model_versions.c.created_at.desc())).mappings().first()
            if row is None:
                raise HTTPException(422, '请提供要启用的 version')
            version = row['version']
    return activate_version(str(version), user)


def _withdraw(conn, version, user):
    ensure_registry(conn, user['sub'])
    current = active_version(conn, lock=True)
    row = _require_model(conn, version, lock=True)
    if current != version:
        if row['status'] == 'candidate':
            conn.execute(model_versions.update().where(model_versions.c.version == version)
                         .values(status='withdrawn', updated_at=now()))
            record(conn, user, f'model-withdraw-candidate:{version}')
            return {'version': version, 'active_version': current,
                    'withdrawn_version': version, 'status': 'withdrawn', 'changed': False}
        if row['status'] == 'withdrawn':
            return {'version': version, 'active_version': current,
                    'withdrawn_version': version, 'status': 'withdrawn', 'changed': False}
        if row['status'] in ('retired', 'active'):
            # Rollback endpoints may name the replaced version to restore.
            outgoing = current
            target = version
            timestamp = now()
            conn.execute(model_versions.update().where(
                model_versions.c.version == outgoing).values(
                status='withdrawn', updated_at=timestamp))
            conn.execute(model_versions.update().where(
                model_versions.c.version == target).values(
                status='active', activated_at=timestamp, updated_at=timestamp))
            conn.execute(model_activation.update().where(model_activation.c.id == 1).values(
                active_version=target, updated_by=user['sub'], updated_at=timestamp))
            from domain.booth_tables import booths
            conn.execute(booths.update().values(model_version=target))
            record(conn, user, f'model-rollback-to:{target}:{outgoing}')
            target_row = _require_model(conn, target)
            target_data = serialize(dict(target_row))
            return {'version': target, 'active_version': target, 'restored_version': target,
                    'replaced_version': target, 'withdrawn_version': outgoing,
                    'model': target_data, 'active': target_data, 'current': target_data,
                    'changed': True}
        raise HTTPException(409, '只能撤回当前启用的新版或尚未启用的候选版')
    target = row['replaced_version']
    if not target or target == version or _model_row(conn, target) is None:
        raise HTTPException(409, '当前版本没有可恢复的被替下版本')
    timestamp = now()
    conn.execute(model_versions.update().where(model_versions.c.version == version).values(
        status='withdrawn', updated_at=timestamp))
    conn.execute(model_versions.update().where(model_versions.c.version == target).values(
        status='active', activated_at=timestamp, updated_at=timestamp))
    conn.execute(model_activation.update().where(model_activation.c.id == 1).values(
        active_version=target, updated_by=user['sub'], updated_at=timestamp))
    from domain.booth_tables import booths
    conn.execute(booths.update().values(model_version=target))
    record(conn, user, f'model-withdraw:{version}:{target}')
    target_row = _require_model(conn, target)
    target_data = serialize(dict(target_row))
    return {'version': target, 'active_version': target, 'restored_version': target,
            'replaced_version': target, 'withdrawn_version': version,
            'model': target_data, 'active': target_data, 'current': target_data,
            'changed': True}


def withdraw_version(version, user, body=None):
    with transaction() as conn:
        return _withdraw(conn, version, user)


def withdraw_named(body, user):
    body = body if isinstance(body, dict) else {}
    version = (body.get('version') or body.get('model_version')
               or body.get('withdraw_version') or body.get('current_version')
               or body.get('candidate_version') or body.get('id') or body.get('name')
               or body.get('target_version') or body.get('restore_version'))
    if not version:
        with transaction() as conn:
            version = active_version(conn)
    return withdraw_version(str(version), user)


def snapshot_for_calculation(version):
    if not version or version == MODEL_VERSION:
        return None
    with transaction() as conn:
        snapshot = conn.execute(select(model_versions.c.training_snapshot).where(
            model_versions.c.version == version)).scalar_one_or_none()
    if snapshot is None:
        raise LookupError(f'model version unavailable: {version}')
    return snapshot


def calculate_version(payload, version):
    snapshot = snapshot_for_calculation(version)
    if snapshot is not None:
        snapshot = dumps(snapshot, sort_keys=True, separators=(',', ':'))
    return calculate(payload, model_version=version or MODEL_VERSION, snapshot=snapshot)


# Compatibility names for callers/tests that use the issue's verbs.
add_reading = add_readings
add_labeled_reading = add_readings
create_training_reading = add_readings
train = train_candidate
train_model = train_candidate
train_model_registry = train_candidate
create_candidate = train_candidate
activate = activate_version
enable = activate_version
activate_model = activate_version
enable_model = activate_version
withdraw = withdraw_version
rollback = withdraw_version
withdraw_model = withdraw_version
rollback_model = withdraw_version
board = quality_board
