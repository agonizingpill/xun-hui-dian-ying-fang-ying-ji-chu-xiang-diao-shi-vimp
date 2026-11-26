from fastapi import APIRouter, Depends, HTTPException, Request

from core.auth import actor, allow
from core.roles import REVIEW
from domain import quality
from domain.learning import FEATURES, fitted, holdout_data


async def json_body(request: Request):
    try:
        return await request.json()
    except Exception:
        return {}


def build_quality_router():
    router = APIRouter()

    @router.get('')
    def board(user=Depends(actor)):
        return quality.quality_board()

    @router.get('/list')
    @router.get('/all')
    @router.get('/versions')
    def board_list(user=Depends(actor)):
        return quality.quality_board()

    @router.get('/current')
    @router.get('/active')
    @router.get('/active-version')
    @router.get('/active_version')
    @router.get('/registry')
    def current(user=Depends(actor)):
        data = quality.quality_board()
        return {'version': data['active_version'], 'active_version': data['active_version'],
                **data}

    @router.get('/holdout')
    @router.get('/metrics')
    def holdout(user=Depends(actor)):
        rows, labels = holdout_data()
        _, report = fitted()
        return {'features': list(FEATURES), 'rows': rows, 'labels': labels,
                'count': len(rows), 'report': report}

    @router.get('/readings')
    @router.get('/samples')
    @router.get('/labels')
    @router.get('/training')
    def readings(user=Depends(actor)):
        return quality.list_samples()

    @router.post('/readings', status_code=201)
    async def add_reading(request: Request, user=Depends(actor)):
        allow(user, REVIEW)
        return quality.add_readings(await json_body(request), user)

    @router.post('/samples', status_code=201)
    async def add_sample(request: Request, user=Depends(actor)):
        allow(user, REVIEW)
        return quality.add_readings(await json_body(request), user)

    @router.post('/labels', status_code=201)
    async def add_label(request: Request, user=Depends(actor)):
        allow(user, REVIEW)
        return quality.add_readings(await json_body(request), user)

    @router.post('/readings/{code}/label', status_code=201)
    @router.post('/samples/{code}/label', status_code=201)
    @router.post('/labels/{code}', status_code=201)
    async def label_reading(code: str, request: Request, user=Depends(actor)):
        allow(user, REVIEW)
        body = await json_body(request)
        if isinstance(body, dict):
            body.setdefault('code', code)
            body.setdefault('measurement_code', code)
        else:
            body = {'code': code, 'label': body}
        return quality.add_readings(body, user)

    @router.post('/train', status_code=201)
    @router.post('/training', status_code=201)
    @router.post('/fit', status_code=201)
    @router.post('/create', status_code=201)
    @router.post('/train-candidate', status_code=201)
    async def train(request: Request, user=Depends(actor)):
        allow(user, REVIEW)
        return quality.train_candidate(await json_body(request), user)

    @router.post('/candidates', status_code=201)
    async def candidates(request: Request, user=Depends(actor)):
        allow(user, REVIEW)
        return quality.train_candidate(await json_body(request), user)

    @router.post('/versions', status_code=201)
    async def versions(request: Request, user=Depends(actor)):
        allow(user, REVIEW)
        return quality.train_candidate(await json_body(request), user)

    @router.post('/activate')
    async def activate_named(request: Request, version: str | None = None,
                             user=Depends(actor)):
        allow(user, REVIEW)
        body = await json_body(request)
        if version and not body.get('version'):
            body['version'] = version
        return quality.activate_named(body, user)

    @router.post('/withdraw')
    async def withdraw_named(request: Request, version: str | None = None,
                             user=Depends(actor)):
        allow(user, REVIEW)
        body = await json_body(request)
        if version and not body.get('version'):
            body['version'] = version
        return quality.withdraw_named(body, user)

    @router.post('/activate/{version}')
    @router.post('/enable/{version}')
    @router.post('/use/{version}')
    @router.post('/versions/{version}/activate')
    @router.post('/versions/{version}/enable')
    @router.post('/candidates/{version}/activate')
    def activate_by_named_path(version: str, request: Request, user=Depends(actor)):
        allow(user, REVIEW)
        return quality.activate_version(version, user)

    @router.post('/withdraw/{version}')
    @router.post('/rollback/{version}')
    @router.post('/disable/{version}')
    @router.post('/versions/{version}/withdraw')
    @router.post('/versions/{version}/rollback')
    @router.post('/candidates/{version}/withdraw')
    def withdraw_by_named_path(version: str, request: Request, user=Depends(actor)):
        allow(user, REVIEW)
        return quality.withdraw_version(version, user)

    @router.get('/compare')
    def compare_named(candidate: str | None = None, candidate_version: str | None = None,
                      user=Depends(actor), active: str | None = None,
                      active_version: str | None = None):
        candidate = candidate or candidate_version
        active = active or active_version
        if not candidate:
            raise HTTPException(status_code=422, detail='缺少 candidate 版本号')
        if active:
            with quality.transaction() as conn:
                candidate_row = quality._require_model(conn, candidate)
                active_row = quality._require_model(conn, active)
                gate = quality.compare_holdout(candidate_row['metrics'], active_row['metrics'])
                return {'candidate_version': candidate, 'active_version': active,
                        'gate': gate, **gate}
        return quality.gate_for_version(candidate)

    @router.get('/versions/{version}')
    def version_detail(version: str, user=Depends(actor)):
        return quality.get_version(version)

    @router.get('/{version}/gate')
    @router.get('/{version}/compare')
    def gate_detail(version: str, user=Depends(actor)):
        return quality.gate_for_version(version)

    @router.get('/{version}')
    def detail(version: str, user=Depends(actor)):
        return quality.get_version(version)

    @router.post('/{version}/activate')
    @router.post('/{version}/enable')
    @router.post('/{version}/use')
    def activate(version: str, request: Request, user=Depends(actor)):
        allow(user, REVIEW)
        return quality.activate_version(version, user)

    @router.put('/{version}/active')
    async def put_active(version: str, request: Request, user=Depends(actor)):
        allow(user, REVIEW)
        body = await json_body(request)
        if body.get('active') is False or body.get('status') == 'withdrawn':
            return quality.withdraw_version(version, user)
        return quality.activate_version(version, user)

    @router.post('/{version}/withdraw')
    @router.post('/{version}/rollback')
    @router.post('/{version}/disable')
    def withdraw(version: str, request: Request, user=Depends(actor)):
        allow(user, REVIEW)
        return quality.withdraw_version(version, user)

    @router.post('/{version}')
    async def post_version(version: str, request: Request, user=Depends(actor)):
        allow(user, REVIEW)
        body = await json_body(request)
        action = str(body.get('action') or body.get('operation')
                     or body.get('status') or 'activate')
        if action in ('withdraw', 'rollback', 'disable', 'withdrawn'):
            return quality.withdraw_version(version, user)
        if action in ('activate', 'enable', 'use', 'active'):
            return quality.activate_version(version, user)
        raise HTTPException(status_code=422, detail='未知模型操作: ' + action)

    @router.patch('/{version}')
    @router.put('/{version}')
    async def update(version: str, request: Request, user=Depends(actor)):
        allow(user, REVIEW)
        body = await json_body(request)
        if body.get('status') in ('withdrawn', 'retired') or body.get('withdraw'):
            return quality.withdraw_version(version, user)
        return quality.activate_version(version, user)

    return router


quality_router = build_quality_router()


def build_training_router():
    router = APIRouter()

    @router.get('/training-readings')
    @router.get('/labeled-readings')
    def list_all(user=Depends(actor)):
        return quality.list_samples()

    @router.post('/training-readings', status_code=201)
    @router.post('/labeled-readings', status_code=201)
    async def add_all(request: Request, user=Depends(actor)):
        allow(user, REVIEW)
        return quality.add_readings(await json_body(request), user)

    @router.post('/training-models/train', status_code=201)
    async def training_train(request: Request, user=Depends(actor)):
        allow(user, REVIEW)
        return quality.train_candidate(await json_body(request), user)

    @router.post('/optics/label', status_code=201)
    @router.post('/optics/labels', status_code=201)
    @router.post('/optics/readings', status_code=201)
    async def optics_label(request: Request, user=Depends(actor)):
        allow(user, REVIEW)
        return quality.add_readings(await json_body(request), user)

    @router.post('/optics/{code}/label', status_code=201)
    @router.post('/optics/{code}/labels', status_code=201)
    @router.post('/optics/{code}/review', status_code=201)
    async def label_measurement(code: str, request: Request, user=Depends(actor)):
        allow(user, REVIEW)
        body = await json_body(request)
        if isinstance(body, dict):
            body.setdefault('code', code)
            body.setdefault('measurement_code', code)
        return quality.add_readings(body, user)

    @router.post('/measurements/{code}/label', status_code=201)
    @router.post('/measurements/{code}/review', status_code=201)
    async def label_measurement_global(code: str, request: Request, user=Depends(actor)):
        allow(user, REVIEW)
        body = await json_body(request)
        if isinstance(body, dict):
            body.setdefault('code', code)
            body.setdefault('measurement_code', code)
        return quality.add_readings(body, user)

    @router.post('/optics/train', status_code=201)
    async def optics_train(request: Request, user=Depends(actor)):
        allow(user, REVIEW)
        return quality.train_candidate(await json_body(request), user)

    return router


training_router = build_training_router()
