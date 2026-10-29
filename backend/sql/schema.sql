-- Reference schema for the Face Shape AI backend.
--
-- The FastAPI app creates this schema automatically on startup via
-- SQLAlchemy (see app/main.py's lifespan + app/models.py). This file exists
-- so the schema can be reviewed, applied manually, or used as a starting
-- point for a real migration tool (e.g. Alembic) — it is not executed by
-- the application itself.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TYPE user_role AS ENUM ('member', 'barber', 'admin');
CREATE TYPE face_shape AS ENUM ('oval', 'round', 'square', 'heart', 'oblong');

-- ---------------------------------------------------------------------------
CREATE TABLE users (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email                VARCHAR(255) NOT NULL UNIQUE,
    hashed_password      VARCHAR(255) NOT NULL,
    display_name         VARCHAR(120) NOT NULL,
    role                 user_role NOT NULL DEFAULT 'member',
    barber_verification_status VARCHAR(16) NOT NULL DEFAULT 'none',
    barber_legal_name    VARCHAR(160),
    barber_phone         VARCHAR(40),
    barber_years_experience INTEGER NOT NULL DEFAULT 0,
    barber_specialties  JSONB NOT NULL DEFAULT '[]',
    barber_application_note TEXT,
    barber_review_note  TEXT,
    barber_reviewed_at  TIMESTAMPTZ,
    barber_reviewed_by_id UUID REFERENCES users(id) ON DELETE SET NULL,
    notify_on_complete   BOOLEAN NOT NULL DEFAULT TRUE,
    theme_accent         VARCHAR(9) NOT NULL DEFAULT '#4274D9',
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_users_email ON users (email);

-- ---------------------------------------------------------------------------
-- Scan history — corresponds to the front-end's "face_analytics" mock table
-- and the "ประวัติการสแกน" (history) screen.
CREATE TABLE face_analytics (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    face_shape        face_shape NOT NULL,
    face_shape_name   VARCHAR(120) NOT NULL,
    confidence        DOUBLE PRECISION NOT NULL,
    gender_pref       VARCHAR(10),
    -- Demo-only: base64 data URL stored directly in the DB. In production,
    -- upload to object storage (S3/GCS) and store just the URL/key here.
    image_data_url    TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_face_analytics_user_id ON face_analytics (user_id);
CREATE INDEX ix_face_analytics_created_at ON face_analytics (created_at);

-- ---------------------------------------------------------------------------
CREATE TABLE favorites (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    face_shape        face_shape NOT NULL,
    face_shape_name   VARCHAR(120) NOT NULL,
    confidence        VARCHAR(16) NOT NULL, -- pre-formatted, e.g. "92.3%"
    image_data_url    TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_favorites_user_id ON favorites (user_id);

-- ---------------------------------------------------------------------------
-- Golden Ratio Analysis snapshots. The metrics are computed client-side in
-- real time from MediaPipe landmarks; this table just persists ones the
-- user chose to save.
CREATE TABLE golden_ratio_results (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id               UUID NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    forehead_pct          DOUBLE PRECISION NOT NULL,
    midface_pct           DOUBLE PRECISION NOT NULL,
    lower_pct             DOUBLE PRECISION NOT NULL,
    length_width_ratio    DOUBLE PRECISION NOT NULL,
    ratio_score           DOUBLE PRECISION NOT NULL,
    symmetry_score        DOUBLE PRECISION NOT NULL,
    overall_score         DOUBLE PRECISION NOT NULL,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_golden_ratio_results_user_id ON golden_ratio_results (user_id);

-- Feedback / contact-admin messages (guests allowed: user_id is NULL for them)
CREATE TABLE IF NOT EXISTS feedback (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID REFERENCES users (id) ON DELETE SET NULL,
    contact_name    VARCHAR(120) NOT NULL,
    contact_email   VARCHAR(255) NOT NULL,
    category        VARCHAR(16)  NOT NULL CHECK (category IN ('suggestion','bug','contact','other')),
    subject         VARCHAR(160) NOT NULL,
    message         TEXT         NOT NULL,
    page            VARCHAR(60),
    status          VARCHAR(16)  NOT NULL DEFAULT 'new' CHECK (status IN ('new','read','resolved')),
    admin_note      TEXT,
    handled_by_id   UUID REFERENCES users (id) ON DELETE SET NULL,
    handled_at      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_feedback_status ON feedback (status);
CREATE INDEX IF NOT EXISTS ix_feedback_created_at ON feedback (created_at);

-- "สร้างภาพสมจริง" (AI hair) generations, for the daily per-member limit
CREATE TABLE IF NOT EXISTS hair_generations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    provider    VARCHAR(16) NOT NULL,
    style_id    VARCHAR(40) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_hair_generations_user_id ON hair_generations (user_id);
CREATE INDEX IF NOT EXISTS ix_hair_generations_created_at ON hair_generations (created_at);
