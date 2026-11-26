from sqlalchemy import inspect, text

from domain.states import BASELINE_MODEL_VERSION


def ensure_columns(engine):
    """Add/widen columns introduced after the initial snapshot without data loss."""
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if 'projection_booths' not in tables:
        return
    booth_columns = {column['name']: column for column in inspector.get_columns('projection_booths')}
    if 'model_version' not in booth_columns:
        with engine.begin() as conn:
            conn.execute(text(
                "ALTER TABLE projection_booths ADD COLUMN model_version VARCHAR(64) "
                f"NOT NULL DEFAULT '{BASELINE_MODEL_VERSION}'"))

    if engine.dialect.name != 'mysql':
        return

    alterations = []
    booth = booth_columns.get('model_version')
    if booth is not None and getattr(booth['type'], 'length', 64) < 64:
        alterations.append(
            "MODIFY COLUMN model_version VARCHAR(64) NOT NULL "
            f"DEFAULT '{BASELINE_MODEL_VERSION}'")
    if 'card_measurements' in tables:
        measurement_columns = {c['name']: c for c in inspector.get_columns('card_measurements')}
        version = measurement_columns.get('model_version')
        if version is not None and getattr(version['type'], 'length', 64) < 64:
            alterations.append("ALTER TABLE card_measurements MODIFY COLUMN model_version VARCHAR(64) NOT NULL")
    if alterations:
        with engine.begin() as conn:
            for alteration in alterations:
                if alteration.startswith('ALTER TABLE'):
                    conn.execute(text(alteration))
                else:
                    conn.execute(text('ALTER TABLE projection_booths ' + alteration))
