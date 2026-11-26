from sqlalchemy import (CheckConstraint, Column, DateTime, Float, ForeignKey, Integer,
                        JSON, String, Table)

from core.db import metadata
from domain.states import BASELINE_MODEL_VERSION

training_readings = Table(
    'training_readings', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('code', String(34)),
    Column('edge_loss', Float, nullable=False),
    Column('color_error', Float, nullable=False),
    Column('flicker_index', Float, nullable=False),
    Column('frame_var', Float, nullable=False),
    Column('artifact', String(24), nullable=False),
    Column('source_measurement', String(24)),
    Column('created_by', String(100), nullable=False),
    Column('created_at', DateTime(timezone=True), nullable=False),
    CheckConstraint("artifact IN ('clean','soft_focus','color_shift','flicker')"),
    CheckConstraint('edge_loss BETWEEN 0 AND 1'),
    CheckConstraint('color_error BETWEEN 0 AND 1'),
    CheckConstraint('flicker_index BETWEEN 0 AND 1'),
    CheckConstraint('frame_var BETWEEN 0 AND 1'),
)

model_versions = Table(
    'model_versions', metadata,
    Column('version', String(64), primary_key=True),
    Column('status', String(16), nullable=False, default='candidate'),
    Column('training_count', Integer, nullable=False),
    Column('new_sample_count', Integer, nullable=False, default=0),
    Column('holdout_count', Integer, nullable=False),
    Column('metrics', JSON, nullable=False),
    Column('gate', JSON, nullable=False),
    Column('training_snapshot', JSON, nullable=False),
    Column('sample_ids', JSON, nullable=False, default=list),
    Column('replaced_version', String(24)),
    Column('note', String(200)),
    Column('created_by', String(100), nullable=False),
    Column('created_at', DateTime(timezone=True), nullable=False),
    Column('activated_at', DateTime(timezone=True)),
    Column('updated_at', DateTime(timezone=True)),
    CheckConstraint("status IN ('candidate','active','retired','withdrawn')"),
)

model_activation = Table(
    'model_activation', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('active_version', String(64),
           ForeignKey('model_versions.version'), nullable=False),
    Column('updated_by', String(100), nullable=False),
    Column('updated_at', DateTime(timezone=True), nullable=False),
    CheckConstraint('id = 1'),
)
