# Brewery Manager - Multi-user Streamlit app (Postgres/Neon)
import streamlit as st
import bcrypt
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
import re
import os
import tempfile

# -----------------------------
# AUTH + CONFIGURAÇÃO DO BANCO DE DADOS (Multiusuário)
# -----------------------------
# ✅ Leitura para todos os usuários autenticados
# ✅ Escrita APENAS para o usuário com role="admin"
#
# IMPORTANTE:
# - Configure usuários/senhas (hash bcrypt) e cookie em .streamlit/secrets.toml
# - Configure DATABASE_URL (Postgres recomendado). Ex:
#   DATABASE_URL="postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME"
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

st.set_page_config(
    page_title="Brewery Manager",
    page_icon="⭐",
    layout="wide"
)

def render_status_badge(status: str | None) -> str:
    """Return a small HTML badge for statuses like Active/Inactive/Planned/Completed."""
    s = (status or "").strip()
    if not s:
        s = "N/A"
    s_low = s.lower()


def _to_python_scalar(value):
    """Convert numpy/pandas scalar types to plain Python types (psycopg2 can't adapt numpy.*)."""
    if value is None:
        return None
    try:
        # Treat NaN/NaT as None
        if pd.isna(value):
            return None
    except Exception:
        pass

    # numpy scalar types
    try:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (np.datetime64,)):
            v = pd.to_datetime(value, errors="coerce")
            return None if pd.isna(v) else v.to_pydatetime()
    except Exception:
        pass

    # pandas Timestamp
    try:
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
    except Exception:
        pass

    # datetime/date are fine; strings/floats/ints fine
    return value

    # Map common statuses to CSS classes
    if s_low in {"active", "enabled", "open", "in stock", "available"}:
        cls = "badge-green"
    elif s_low in {"inactive", "disabled", "closed", "out of stock", "unavailable"}:
        cls = "badge-gray"
    elif s_low in {"planned", "draft"}:
        cls = "badge-blue"
    elif s_low in {"in progress", "brewing", "fermenting"}:
        cls = "badge-orange"
    elif s_low in {"completed", "done", "finished"}:
        cls = "badge-green"
    elif s_low in {"cancelled", "canceled", "error"}:
        cls = "badge-red"
    else:
        cls = "badge-gray"

    return f"<span class='status-badge {cls}'>{s}</span>"
def _auth_users():
    # Expected structure in secrets:
    # [auth]
    #   [auth.credentials]
    #     [auth.credentials.usernames]
    #       [auth.credentials.usernames.<username>]
    #       name = "..."
    #       password = "$2b$12$..."  # bcrypt hash
    #       role = "admin" | "viewer"
    return st.secrets["auth"]["credentials"]["usernames"]

def _check_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

def require_login():
    # If already logged in, show quick status + logout
    if st.session_state.get("logged_in"):
        with st.sidebar:
            st.caption(f"Signed in as: {st.session_state.get('auth_name')} ({st.session_state.get('auth_role')})")
            if st.button("🚪 Sign out"):
                for k in ["logged_in","auth_user","auth_name","auth_role","view_mode"]:
                    st.session_state.pop(k, None)
                st.rerun()
        return

    st.title("🔐 Login")
    st.write("Enter your username and password to access the app.")

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", value="", placeholder="ex: admin")
        password = st.text_input("Password", value="", type="password")
        submitted = st.form_submit_button("Sign in")



    # Public read-only mode (no login required)
    view_mode = st.button("👀 ENTER VISUALIZATION MODE", use_container_width=True)
    if view_mode:
        st.session_state["logged_in"] = True
        st.session_state["auth_user"] = "viewer"
        st.session_state["auth_name"] = "View mode"
        st.session_state["auth_role"] = "viewer"
        st.session_state["view_mode"] = True
        st.rerun()
    if not submitted:
        st.stop()

    users = _auth_users()
    if username not in users:
        st.error("Username ou senha inválidos.")
        st.stop()

    user_cfg = users[username]
    if not _check_password(password, user_cfg.get("password","")):
        st.error("Username ou senha inválidos.")
        st.stop()

    st.session_state["logged_in"] = True
    st.session_state["auth_user"] = username
    st.session_state["auth_name"] = user_cfg.get("name", username)
    st.session_state["auth_role"] = user_cfg.get("role", "viewer")
    st.success("Signed in!")
    st.rerun()

DEFAULT_SQLITE_FILE = os.getenv("SQLITE_FILE", "brewery_database.db")

def get_database_url() -> str:
    """Retorna a URL do banco (Postgres recomendado)."""
    # Prioridade: secrets -> env -> sqlite local
    if hasattr(st, "secrets") and "DATABASE_URL" in st.secrets:
        return st.secrets["DATABASE_URL"]
    if os.getenv("DATABASE_URL"):
        return os.getenv("DATABASE_URL")
    return f"sqlite:///{DEFAULT_SQLITE_FILE}"

@st.cache_resource

def get_engine() -> Engine:
    """Engine compartilhado (pool de conexões) para suportar múltiplos usuários."""
    db_url = get_database_url()
    # pool_pre_ping evita conexões “mortas”
    return create_engine(db_url, pool_pre_ping=True)


# -----------------------------
# PERFORMANCE HELPERS
# -----------------------------

def _get_db_version() -> int:
    return int(st.session_state.get("db_version", 0) or 0)

def bump_db_version() -> None:
    st.session_state["db_version"] = _get_db_version() + 1

@st.cache_data(ttl=3600, show_spinner=False)
def _get_table_columns_cached(table_name: str, dialect: str) -> list[str]:
    """Cache table columns to avoid repeated information_schema/PRAGMA hits."""
    engine = get_engine()
    cols: list[str] = []
    try:
        with engine.connect() as conn:
            if dialect in {"postgresql", "postgres"}:
                rows = conn.execute(
                    sql_text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema = current_schema() AND table_name = :t"
                    ),
                    {"t": table_name},
                ).fetchall()
                cols = [r[0] for r in rows]
            else:
                rows = conn.execute(sql_text(f"PRAGMA table_info({table_name})")).fetchall()
                cols = [r[1] for r in rows]
    except Exception:
        cols = []
    return cols

def get_table_columns(table_name: str) -> list[str]:
    engine = get_engine()
    dialect = engine.dialect.name.lower()
    cols = _get_table_columns_cached(table_name, dialect)
    return cols if cols else []

# Backwards-compat helper
# Some parts of the app historically called `get_table_columns_cached(table_name, dialect)`.
# We keep a thin wrapper so older code paths don't crash.
def get_table_columns_cached(table_name: str, dialect: str | None = None) -> list[str]:
    engine = get_engine()
    d = (dialect or engine.dialect.name).lower()
    cols = _get_table_columns_cached(table_name, d)
    return cols if cols else []

def is_admin() -> bool:
    return st.session_state.get("auth_role") == "admin"


def is_viewer() -> bool:
    return not is_admin()

def require_admin_action():
    """Bloqueio forte: viewers nunca escrevem no banco."""
    if not is_admin():
        st.error("🔒 Admin-only action.")
        st.stop()

def _require_auth_config():
    if "auth" not in st.secrets:
        st.error(
            "Missing authentication configuration. "
            "Add an [auth] section in Streamlit Secrets."
        )
        st.stop()

def _translate_sqlite_to_postgres(ddl: str) -> str:
    """Tradução simples do DDL do SQLite para Postgres."""
    ddl_pg = ddl
    ddl_pg = re.sub(r"INTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT", "SERIAL PRIMARY KEY", ddl_pg, flags=re.I)
    ddl_pg = re.sub(r"REAL\b", "DOUBLE PRECISION", ddl_pg, flags=re.I)
    ddl_pg = ddl_pg.replace("AUTOINCREMENT", "")
    # Clear vírgulas finais antes de ')'
    ddl_pg = re.sub(r",\s*\)", "\n)", ddl_pg)
    return ddl_pg


def _ensure_columns(table_name: str, columns_sql: dict[str, str]) -> None:
    """Add missing columns to an existing table (best-effort, safe for fresh deploys).
    columns_sql: {column_name: SQL_TYPE or 'SQL_TYPE DEFAULT ...'}
    """
    engine = get_engine()
    dialect = engine.dialect.name.lower()

    existing: set[str] = set()
    try:
        with engine.connect() as conn:
            if dialect in {"postgresql", "postgres"}:
                rows = conn.execute(
                    sql_text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema = current_schema() AND table_name = :t"
                    ),
                    {"t": table_name},
                ).fetchall()
                existing = {r[0] for r in rows}
            else:
                rows = conn.execute(sql_text(f"PRAGMA table_info({table_name})")).fetchall()
                # PRAGMA table_info: (cid, name, type, notnull, dflt_value, pk)
                existing = {r[1] for r in rows}
    except Exception:
        return

    missing = [c for c in columns_sql.keys() if c not in existing]
    if not missing:
        return

    with engine.begin() as conn:
        for col in missing:
            col_def = columns_sql[col]
            conn.execute(sql_text(f"ALTER TABLE {table_name} ADD COLUMN {col} {col_def}"))


def init_database():
    """Create tables if they don't exist (Postgres or SQLite).
    This project is deployed on Postgres (Neon), but we keep SQLite support for local runs.
    """
    engine = get_engine()
    dialect = engine.dialect.name.lower()

    if dialect in {"postgresql", "postgres"}:
        ddl_blocks = [
            """CREATE TABLE IF NOT EXISTS breweries (
                id_brewery BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT,
                address TEXT,
                city TEXT,
                state TEXT,
                country TEXT,
                postal_code TEXT,
                contact_person TEXT,
                contact_phone TEXT,
                contact_email TEXT,
                default_batch_size DOUBLE PRECISION,
                annual_capacity_hl DOUBLE PRECISION,
                status TEXT DEFAULT 'Active',
                license_number TEXT,
                established_date DATE,
                has_lab INTEGER DEFAULT 0,
                description TEXT,
                created_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS ingredients (
                id_ingredient BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT,
                supplier_id BIGINT,
                supplier_name TEXT,
                supplier TEXT,
                origin TEXT,
                unit TEXT,
                cost_per_unit DOUBLE PRECISION,
                quantity_in_stock DOUBLE PRECISION,
                reorder_level DOUBLE PRECISION,
                notes TEXT,
                status TEXT DEFAULT 'Active',
                created_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS suppliers (
                id_supplier BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                contact_person TEXT,
                phone TEXT,
                email TEXT,
                address TEXT,
                city TEXT,
                country TEXT,
                website TEXT,
                notes TEXT,
                status TEXT DEFAULT 'Active',
                created_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS recipes (
                id_recipe BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                style TEXT,
                abv DOUBLE PRECISION,
                ibu DOUBLE PRECISION,
                srm DOUBLE PRECISION,
                batch_size DOUBLE PRECISION,
                notes TEXT,
                created_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS recipe_items (
                id_recipe_item BIGSERIAL PRIMARY KEY,
                recipe_id TEXT,
                ingredient_name TEXT,
                quantity DOUBLE PRECISION,
                unit TEXT,
                notes TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS purchases (
                id_purchase BIGSERIAL PRIMARY KEY,
                ingredient_id TEXT,
                ingredient_name TEXT,
                supplier_id TEXT,
                supplier_name TEXT,
                quantity DOUBLE PRECISION,
                unit TEXT,
                total_cost DOUBLE PRECISION,
                purchase_date DATE,
                notes TEXT,
                created_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )""",

            # New (2026): purchase header + line items (purchase by order, not by item)
            """CREATE TABLE IF NOT EXISTS purchase_orders (
                id_purchase_order BIGSERIAL PRIMARY KEY,
                transaction_type TEXT DEFAULT 'Purchase',
                supplier TEXT,
                order_number TEXT,
                date DATE,
                freight_total DOUBLE PRECISION DEFAULT 0,
                notes TEXT,
                recorded_by TEXT,
                created_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS purchase_order_items (
                id_purchase_item BIGSERIAL PRIMARY KEY,
                purchase_order_id BIGINT,
                ingredient TEXT,
                quantity DOUBLE PRECISION,
                unit TEXT,
                unit_price DOUBLE PRECISION,
                freight_per_unit DOUBLE PRECISION,
                effective_unit_cost DOUBLE PRECISION,
                total_cost DOUBLE PRECISION,
                created_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS production_orders (
                id_order BIGSERIAL PRIMARY KEY,
                id_recipe TEXT,
                brewery_id TEXT,
                planned_volume DOUBLE PRECISION,
                status TEXT DEFAULT 'Planned',
                start_date DATE,
                end_date DATE,
                equipment TEXT,
                batch_id TEXT,
                notes TEXT,
                created_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS calendar_events (
                id_event BIGSERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                event_type TEXT,
                start_date DATE,
                end_date DATE,
                equipment TEXT,
                batch_id TEXT,
                notes TEXT,
                created_by TEXT,
                created_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS equipment (
                id_equipment BIGSERIAL PRIMARY KEY,
                brewery_id TEXT,
                name TEXT NOT NULL,
                type TEXT,
                capacity_liters DOUBLE PRECISION,
                unit TEXT,
                manufacturer TEXT,
                model TEXT,
                serial_number TEXT,
                material TEXT,
                status TEXT DEFAULT 'Active',
                install_date DATE,
                next_maintenance DATE,
                cleaning_frequency TEXT,
                cleaning_due DATE,
                pressure_rating DOUBLE PRECISION,
                has_jacket INTEGER DEFAULT 0,
                has_sight_glass INTEGER DEFAULT 0,
                notes TEXT,
                created_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS team_members (
                id_member BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT,
                email TEXT,
                phone TEXT,
                status TEXT DEFAULT 'Active',
                notes TEXT,
                created_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )""",
        ]
    else:
        ddl_blocks = [
            """CREATE TABLE IF NOT EXISTS breweries (
                id_brewery INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT,
                address TEXT,
                city TEXT,
                state TEXT,
                country TEXT,
                postal_code TEXT,
                contact_person TEXT,
                contact_phone TEXT,
                contact_email TEXT,
                default_batch_size REAL,
                annual_capacity_hl REAL,
                status TEXT DEFAULT 'Active',
                license_number TEXT,
                established_date DATE,
                has_lab INTEGER DEFAULT 0,
                description TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS ingredients (
                id_ingredient INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT,
                supplier_id INTEGER,
                supplier_name TEXT,
                supplier TEXT,
                origin TEXT,
                unit TEXT,
                cost_per_unit REAL,
                quantity_in_stock REAL,
                reorder_level REAL,
                notes TEXT,
                status TEXT DEFAULT 'Active',
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS suppliers (
                id_supplier INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                contact_person TEXT,
                phone TEXT,
                email TEXT,
                address TEXT,
                city TEXT,
                country TEXT,
                website TEXT,
                notes TEXT,
                status TEXT DEFAULT 'Active',
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS recipes (
                id_recipe INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                style TEXT,
                abv REAL,
                ibu REAL,
                srm REAL,
                batch_size REAL,
                notes TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS recipe_items (
                id_recipe_item INTEGER PRIMARY KEY AUTOINCREMENT,
                recipe_id TEXT,
                ingredient_name TEXT,
                quantity REAL,
                unit TEXT,
                notes TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS purchases (
                id_purchase INTEGER PRIMARY KEY AUTOINCREMENT,
                ingredient_id TEXT,
                ingredient_name TEXT,
                supplier_id TEXT,
                supplier_name TEXT,
                quantity REAL,
                unit TEXT,
                total_cost REAL,
                purchase_date DATE,
                notes TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",

            # New (2026): purchase header + line items (purchase by order, not by item)
            """CREATE TABLE IF NOT EXISTS purchase_orders (
                id_purchase_order INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_type TEXT DEFAULT 'Purchase',
                supplier TEXT,
                order_number TEXT,
                date DATE,
                freight_total REAL DEFAULT 0,
                notes TEXT,
                recorded_by TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS purchase_order_items (
                id_purchase_item INTEGER PRIMARY KEY AUTOINCREMENT,
                purchase_order_id INTEGER,
                ingredient TEXT,
                quantity REAL,
                unit TEXT,
                unit_price REAL,
                freight_per_unit REAL,
                effective_unit_cost REAL,
                total_cost REAL,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS production_orders (
                id_order INTEGER PRIMARY KEY AUTOINCREMENT,
                id_recipe TEXT,
                brewery_id TEXT,
                planned_volume REAL,
                status TEXT DEFAULT 'Planned',
                start_date DATE,
                end_date DATE,
                equipment TEXT,
                batch_id TEXT,
                notes TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS calendar_events (
                id_event INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                event_type TEXT,
                start_date DATE,
                end_date DATE,
                equipment TEXT,
                batch_id TEXT,
                notes TEXT,
                created_by TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS equipment (
                id_equipment INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT,
                capacity REAL,
                unit TEXT,
                status TEXT DEFAULT 'Active',
                notes TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS team_members (
                id_member INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                role TEXT,
                email TEXT,
                phone TEXT,
                status TEXT DEFAULT 'Active',
                notes TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
        ]

    with engine.begin() as conn:
        for ddl in ddl_blocks:
            conn.execute(sql_text(ddl))

    # Auto-sync schema with UI (adds missing columns safely)
    _ensure_columns(
        "breweries",
        {
            "type": "TEXT",
            "address": "TEXT",
            "city": "TEXT",
            "state": "TEXT",
            "country": "TEXT",
            "postal_code": "TEXT",
            "contact_person": "TEXT",
            "contact_phone": "TEXT",
            "contact_email": "TEXT",
            "default_batch_size": "DOUBLE PRECISION",
            "annual_capacity_hl": "DOUBLE PRECISION",
            "status": "TEXT",
            "license_number": "TEXT",
            "established_date": "DATE",
            "has_lab": "INTEGER DEFAULT 0",
            "description": "TEXT",
        },
    )

    _ensure_columns(
        "ingredients",
        {
            # Newer UI fields (keep legacy columns too)
            "manufacturer": "TEXT",
            "category": "TEXT",
            "supplier": "TEXT",
            "supplier_id": "BIGINT",
            "supplier_name": "TEXT",
            "origin": "TEXT",
            "unit": "TEXT",
            "stock": "DOUBLE PRECISION DEFAULT 0",
            "unit_cost": "DOUBLE PRECISION DEFAULT 0",
            "low_stock_threshold": "DOUBLE PRECISION DEFAULT 0",
            "alpha_acid": "DOUBLE PRECISION DEFAULT 0",
            "lot_number": "TEXT",
            "expiry_date": "DATE",
            "last_updated": "TIMESTAMPTZ",
            "cost_per_unit": "DOUBLE PRECISION",
            "quantity_in_stock": "DOUBLE PRECISION",
            "reorder_level": "DOUBLE PRECISION",
            "notes": "TEXT",
            "status": "TEXT",
        },
    )

    # Purchase-by-order tables
    _ensure_columns(
        "purchase_orders",
        {
            "transaction_type": "TEXT",
            "supplier": "TEXT",
            "order_number": "TEXT",
            "date": "DATE",
            "freight_total": "DOUBLE PRECISION DEFAULT 0",
            "notes": "TEXT",
            "recorded_by": "TEXT",
        },
    )
    _ensure_columns(
        "purchase_order_items",
        {
            "purchase_order_id": "BIGINT",
            "ingredient": "TEXT",
            "quantity": "DOUBLE PRECISION",
            "unit": "TEXT",
            "unit_price": "DOUBLE PRECISION",
            "freight_per_unit": "DOUBLE PRECISION",
            "effective_unit_cost": "DOUBLE PRECISION",
            "total_cost": "DOUBLE PRECISION",
        },
    )

    _ensure_columns(
        "suppliers",
        {
            "contact_person": "TEXT",
            "phone": "TEXT",
            "email": "TEXT",
            "address": "TEXT",
            "city": "TEXT",
            "country": "TEXT",
            "website": "TEXT",
            "notes": "TEXT",
            "status": "TEXT",
        },
    )

    _ensure_columns(
        "recipes",
        {
            "style": "TEXT",
            "abv": "DOUBLE PRECISION",
            "ibu": "DOUBLE PRECISION",
            "srm": "DOUBLE PRECISION",
            "batch_size": "DOUBLE PRECISION",
            "notes": "TEXT",
        },
    )

    _ensure_columns(
        "recipe_items",
        {
            "recipe_id": "TEXT",
            "ingredient_name": "TEXT",
            "quantity": "DOUBLE PRECISION",
            "unit": "TEXT",
            "notes": "TEXT",
        },
    )

    _ensure_columns(
        "purchases",
        {
            "ingredient_id": "TEXT",
            "ingredient_name": "TEXT",
            "supplier_id": "TEXT",
            "supplier_name": "TEXT",
            "quantity": "DOUBLE PRECISION",
            "unit": "TEXT",
            "total_cost": "DOUBLE PRECISION",
            "purchase_date": "DATE",
            "notes": "TEXT",
        },
    )

    _ensure_columns(
        "production_orders",
        {
            "id_recipe": "TEXT",
            "brewery_id": "TEXT",
            "planned_volume": "DOUBLE PRECISION",
            "status": "TEXT",
            "start_date": "DATE",
            "end_date": "DATE",
            "equipment": "TEXT",
            "batch_id": "TEXT",
            "notes": "TEXT",
        },
    )

    _ensure_columns(
        "calendar_events",
        {
            "event_type": "TEXT",
            "start_date": "DATE",
            "end_date": "DATE",
            "equipment": "TEXT",
            "batch_id": "TEXT",
            "notes": "TEXT",
            "created_by": "TEXT",
        },
    )

    _ensure_columns(
        "equipment",
        {
            "brewery_id": "TEXT",
            "type": "TEXT",
            "capacity_liters": "DOUBLE PRECISION",
            "unit": "TEXT",
            "manufacturer": "TEXT",
            "model": "TEXT",
            "serial_number": "TEXT",
            "material": "TEXT",
            "status": "TEXT",
            "install_date": "DATE",
            "next_maintenance": "DATE",
            "cleaning_frequency": "TEXT",
            "cleaning_due": "DATE",
            "pressure_rating": "DOUBLE PRECISION",
            "has_jacket": "INTEGER DEFAULT 0",
            "has_sight_glass": "INTEGER DEFAULT 0",
            "notes": "TEXT",
        },
    )

    _ensure_columns(
        "team_members",
        {
            "role": "TEXT",
            "email": "TEXT",
            "phone": "TEXT",
            "status": "TEXT",
            "notes": "TEXT",
        },
    )

    # Suppliers: keep schema compatible across older deploys
    _ensure_columns(
        "suppliers",
        {
            "name": "TEXT",
            "contact_person": "TEXT",
            "phone": "TEXT",
            "email": "TEXT",
            "address": "TEXT",
            "city": "TEXT",
            "country": "TEXT",
            "website": "TEXT",
            "notes": "TEXT",
            "status": "TEXT",
        },
    )


def query_to_df(query: str, params: dict | None = None) -> pd.DataFrame:
    """Executa SELECT e retorna DateFrame."""
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql_query(sql_text(query), conn, params=params or {})

def execute_query(query: str, params: dict | None = None):
    """Executa INSERT/UPDATE/DELETE (admin only)."""
    require_admin_action()
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(sql_text(query), params or {})
    # cache-bust for reads
    bump_db_version()
    return result.rowcount



def get_table_data(table_name: str) -> pd.DataFrame:
    return query_to_df(f"SELECT * FROM {table_name}")


def _singularize_table_name(table_name: str) -> str:
    """Best-effort singularization for common English plurals (for PK naming)."""
    name = table_name.strip().lower()
    if name.endswith("ies") and len(name) > 3:
        return name[:-3] + "y"   # breweries -> brewery
    if name.endswith("ses") and len(name) > 3:
        return name[:-2]         # classes -> class (rough)
    if name.endswith("s") and len(name) > 1:
        return name[:-1]         # suppliers -> supplier
    return name

def insert_data(table_name: str, data_dict: dict):
    """Insert a row into a table.
    - Filters out keys that are not real columns (prevents 'undefined column' errors).
    - For Postgres, uses RETURNING to get the inserted PK when possible.
    """
    require_admin_action()
    engine = get_engine()
    dialect = engine.dialect.name.lower()
    # Fetch actual columns (cached)
    try:
        actual_cols = get_table_columns_cached(table_name)
    except Exception:
        actual_cols = list(data_dict.keys())

    # Keep only keys that exist as columns
    filtered = {k: _to_python_scalar(v) for k, v in data_dict.items() if k in set(actual_cols)}

    if not filtered:
        raise ValueError(f"No valid columns to insert into '{table_name}'. Check schema vs UI fields.")

    cols = ", ".join(filtered.keys())
    placeholders = ", ".join([f":{k}" for k in filtered.keys()])

    # Determine PK column for RETURNING
    singular = _singularize_table_name(table_name)
    pk_candidates = ["id", f"id_{singular}"]
    pk_col = pk_candidates[0]
    for cand in pk_candidates:
        if cand in set(actual_cols):
            pk_col = cand
            break

    if dialect in {"postgresql", "postgres"}:
        sql = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders}) RETURNING {pk_col}"
        try:
            with engine.begin() as conn:
                res = conn.execute(sql_text(sql), filtered)
                row = res.fetchone()
                bump_db_version()
                return row[0] if row else None
        except Exception as e:
            # retry without RETURNING (some schemas/PKs may differ)
            sql2 = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"
            try:
                with engine.begin() as conn:
                    conn.execute(sql_text(sql2), filtered)
                    bump_db_version()
                return None
            except Exception as e2:
                # Friendly, safe error message for admins (Streamlit redacts the original)
                try:
                    import sqlalchemy
                    from sqlalchemy import exc as sa_exc
                    orig = getattr(e2, "orig", None)
                    if orig is not None:
                        msg = str(orig)
                        # Auto-fix missing column if Postgres says UndefinedColumn (SQLSTATE 42703)
                        pgcode = getattr(orig, "pgcode", None)
                        if pgcode == "42703":
                            mcol = re.search(r'column "([^"]+)"', msg)
                            if mcol:
                                missing_col = mcol.group(1)
                                _ensure_columns(table_name, {missing_col: "TEXT"})
                                # retry once
                                with engine.begin() as conn:
                                    conn.execute(sql_text(sql2), filtered)
                                return None
                        st.error(f"Database error: {type(orig).__name__}: {msg}")
                    else:
                        st.error(f"Database error: {type(e2).__name__}: {e2}")
                except Exception:
                    # If anything goes wrong while showing the message, just re-raise
                    pass
                raise
    else:
        sql = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"
        with engine.begin() as conn:
            res = conn.execute(sql_text(sql), filtered)
            # SQLite may provide lastrowid
            try:
                rid = res.lastrowid
            except Exception:
                rid = None
            bump_db_version()
            return rid

def update_data(table_name: str, data_dict: dict, where_clause: str, where_params: dict):
    """Update rows (admin only). Filters unknown columns and coerces numpy/pandas scalars."""
    require_admin_action()
    data_dict = data_dict or {}
    where_params = where_params or {}

    engine = get_engine()
    # Fetch actual columns (cached) to avoid "UndefinedColumn" errors
    dialect = engine.dialect.name.lower()
    actual_cols = get_table_columns_cached(table_name, dialect) or list(data_dict.keys())

    # Keep only columns that exist
    filtered_updates = {k: _to_python_scalar(v) for k, v in data_dict.items() if k in set(actual_cols)}
    if not filtered_updates:
        return 0

    set_clause = ", ".join([f"{k} = :set_{k}" for k in filtered_updates.keys()])
    params = {f"set_{k}": v for k, v in filtered_updates.items()}
    params.update({k: _to_python_scalar(v) for k, v in where_params.items()})

    sql = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"

    try:
        with engine.begin() as conn:
            res = conn.execute(sql_text(sql), params)
            bump_db_version()
        return res.rowcount
    except Exception as e:
        # If Postgres says a column doesn't exist, auto-add it as TEXT and retry once.
        orig = getattr(e, "orig", None)
        msg = str(orig) if orig is not None else str(e)
        pgcode = getattr(orig, "pgcode", None) if orig is not None else None
        if pgcode == "42703":  # UndefinedColumn
            mcol = re.search(r'column "([^"]+)"', msg)
            if mcol:
                missing_col = mcol.group(1)
                _ensure_columns(table_name, {missing_col: "TEXT"})
                with engine.begin() as conn:
                    res = conn.execute(sql_text(sql), params)
                    bump_db_version()
            return res.rowcount

        # Show a safe error message (Streamlit otherwise redacts)
        st.error(f"Database error: {type(orig).__name__ if orig is not None else type(e).__name__}: {msg}")
        raise

def delete_data(table_name: str, where_clause: str, where_params: dict):
    """Deleta dados (admin only)."""
    require_admin_action()
    sql = f"DELETE FROM {table_name} WHERE {where_clause}"
    engine = get_engine()
    with engine.begin() as conn:
        res = conn.execute(sql_text(sql), where_params or {})
    bump_db_version()
    return res.rowcount

def _get_all_data_uncached() -> dict[str, pd.DataFrame]:
    """Loads all tables from DB (no cache)."""
    table_names = [
        'ingredients',
        # legacy (kept for backwards compatibility)
        'purchases',
        # purchase-by-order (preferred)
        'purchase_orders', 'purchase_order_items',
        'suppliers', 'recipes', 'recipe_items',
        'breweries', 'equipment', 'production_orders', 'calendar_events', 'team_members'
    ]
    data: dict[str, pd.DataFrame] = {}
    for table in table_names:
        data[table] = get_table_data(table)
    return data

@st.cache_data(ttl=60, show_spinner=False)
def _get_all_data_cached(db_version: int) -> dict[str, pd.DataFrame]:
    # db_version is only used to bust cache when writes happen
    return _get_all_data_uncached()

def get_all_data() -> dict[str, pd.DataFrame]:
    """Cached all-tables snapshot.

    This keeps the UI snappy when switching tabs / selectboxes, while still updating
    quickly after writes (we bump db_version on every successful write).
    """
    return _get_all_data_cached(_get_db_version())

# -----------------------------
# FUNÇÕES DE MIGRAÇÃO E BACKUP
# -----------------------------
def migrate_excel_to_sqlite(excel_file):
    """Migra dados do Excel para o banco (admin only).
    Observação: isso é opcional — os usuários NÃO precisam fazer upload para ver dados.
    """
    require_admin_action()
    try:
        xls = pd.ExcelFile(excel_file)

        sheet_table_map = {
            "Ingredients": "ingredients",
            "Purchases": "purchases",
            "Suppliers": "suppliers",
            "Recipes": "recipes",
            "Recipe Items": "recipe_items",
            "Breweries": "breweries",
            "Equipment": "equipment",
            "Production Orders": "production_orders",
            "Calendar Events": "calendar_events",
            "Team Members": "team_members"
        }

        engine = get_engine()

        for sheet_name, table_name in sheet_table_map.items():
            if sheet_name not in xls.sheet_names:
                continue

            df = pd.read_excel(xls, sheet_name=sheet_name)
            if df.empty:
                continue

            # Converter possíveis datas
            for col in df.columns:
                if "date" in str(col).lower():
                    df[col] = pd.to_datetime(df[col], errors="coerce")

            # Clear tabela existente e inserir dados
            with engine.begin() as conn:
                conn.execute(sql_text(f"DELETE FROM {table_name}"))
            # to_sql funciona com SQLAlchemy (Postgres/SQLite)
            df.to_sql(table_name, engine, if_exists="append", index=False)

        return True
    except Exception as e:
        st.error(f"Erro na migração: {e}")
        return False

def export_to_excel():
    """Exporta todos os dados para Excel"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        data = get_all_data()
        for table_name, df in data.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=table_name, index=False)
    
    output.seek(0)
    return output

# -----------------------------
# FUNÇÕES ESPECÍFICAS DO NEGÓCIO
# -----------------------------
def update_stock_from_purchase(ingredient_name, quantity):
    """Atualiza o estoque após uma compra"""
    return execute_query(
        "UPDATE ingredients SET stock = stock + :quantity, last_updated = CURRENT_TIMESTAMP WHERE name = :ingredient_name",
        {"quantity": quantity, "ingredient_name": ingredient_name}
    )


def update_stock_and_cost_from_purchase(ingredient_name: str, quantity: float, effective_unit_cost: float):
    """Atualiza estoque + custo unitário (média ponderada) após uma compra.

    O effective_unit_cost já deve incluir o rateio do frete por unidade.
    """
    try:
        current = query_to_df(
            "SELECT COALESCE(stock, 0) AS stock, COALESCE(unit_cost, 0) AS unit_cost FROM ingredients WHERE name = :n",
            {"n": ingredient_name},
        )
        if current.empty:
            # fallback: só atualiza estoque
            return update_stock_from_purchase(ingredient_name, quantity)

        cur_stock = float(current.iloc[0]["stock"] or 0)
        cur_cost = float(current.iloc[0]["unit_cost"] or 0)
        new_stock = cur_stock + float(quantity)

        # média ponderada por estoque
        if new_stock > 0:
            new_cost = ((cur_stock * cur_cost) + (float(quantity) * float(effective_unit_cost))) / new_stock
        else:
            new_cost = float(effective_unit_cost)

        return execute_query(
            """UPDATE ingredients
               SET stock = stock + :q,
                   unit_cost = :c,
                   last_updated = CURRENT_TIMESTAMP
               WHERE name = :n""",
            {"q": float(quantity), "c": float(new_cost), "n": ingredient_name},
        )
    except Exception:
        # never block purchase flow
        return update_stock_from_purchase(ingredient_name, quantity)



def save_purchase_order_fast(
    transaction_type: str,
    supplier: str,
    order_number: str,
    po_date: date,
    freight_total: float,
    notes: str,
    recorded_by: str,
    preview_rows: list[dict],
) -> int:
    """Persist a purchase order + items in a single transaction (much faster).

    - Inserts header and all items with executemany
    - Updates ingredient stock and (for purchases) unit_cost by weighted average
    """
    require_admin_action()
    engine = get_engine()
    dialect = engine.dialect.name.lower()

    header = {
        "transaction_type": transaction_type,
        "supplier": supplier,
        "order_number": order_number,
        "date": po_date,
        "freight_total": float(freight_total) if transaction_type == "Purchase" else 0.0,
        "notes": notes,
        "recorded_by": recorded_by,
    }

    with engine.begin() as conn:
        if dialect in {"postgresql", "postgres"}:
            res = conn.execute(
                sql_text(
                    """
                    INSERT INTO purchase_orders (transaction_type, supplier, order_number, date, freight_total, notes, recorded_by)
                    VALUES (:transaction_type, :supplier, :order_number, :date, :freight_total, :notes, :recorded_by)
                    RETURNING id_purchase_order
                    """
                ),
                header,
            )
            po_id = int(res.scalar_one())
        else:
            res = conn.execute(
                sql_text(
                    """
                    INSERT INTO purchase_orders (transaction_type, supplier, order_number, date, freight_total, notes, recorded_by)
                    VALUES (:transaction_type, :supplier, :order_number, :date, :freight_total, :notes, :recorded_by)
                    """
                ),
                header,
            )
            po_id = int(getattr(res, "lastrowid", 0) or 0)

        items_payload = []
        for row in preview_rows:
            items_payload.append(
                {
                    "purchase_order_id": po_id,
                    "ingredient": str(row["Ingredient"]),
                    "quantity": float(row["Quantity"]),
                    "unit": str(row.get("Unit", "") or ""),
                    "unit_price": float(row["Unit Price"]),
                    "freight_per_unit": float(row["Freight / Unit"]),
                    "effective_unit_cost": float(row["Effective Unit Cost"]),
                    "total_cost": float(row["Line Total"]),
                }
            )

        if items_payload:
            conn.execute(
                sql_text(
                    """
                    INSERT INTO purchase_order_items (
                        purchase_order_id, ingredient, quantity, unit,
                        unit_price, freight_per_unit, effective_unit_cost, total_cost
                    ) VALUES (
                        :purchase_order_id, :ingredient, :quantity, :unit,
                        :unit_price, :freight_per_unit, :effective_unit_cost, :total_cost
                    )
                    """
                ),
                items_payload,
            )

        # --- Stock / cost updates ---
        # group by ingredient (in case user repeated an item)
        agg = {}
        for it in items_payload:
            name = it["ingredient"]
            a = agg.setdefault(name, {"qty": 0.0, "cost_total": 0.0})
            a["qty"] += float(it["quantity"])
            a["cost_total"] += float(it["quantity"]) * float(it["effective_unit_cost"])

        names = list(agg.keys())
        if names:
            # fetch current stock/cost in a single query
            params = {f"n{i}": n for i, n in enumerate(names)}
            in_clause = ", ".join([f":n{i}" for i in range(len(names))])
            cur_rows = conn.execute(
                sql_text(
                    f"SELECT name, COALESCE(stock,0) AS stock, COALESCE(unit_cost,0) AS unit_cost FROM ingredients WHERE name IN ({in_clause})"
                ),
                params,
            ).fetchall()
            cur_map = {r[0]: (float(r[1] or 0), float(r[2] or 0)) for r in cur_rows}

            updates = []
            for name in names:
                cur_stock, cur_cost = cur_map.get(name, (0.0, 0.0))
                add_qty = float(agg[name]["qty"])

                if transaction_type == "Purchase":
                    add_cost_total = float(agg[name]["cost_total"])
                    new_stock = cur_stock + add_qty
                    if new_stock > 0:
                        new_cost = ((cur_stock * cur_cost) + add_cost_total) / new_stock
                    else:
                        new_cost = (add_cost_total / add_qty) if add_qty > 0 else cur_cost
                    updates.append({"n": name, "q": add_qty, "c": float(new_cost)})
                else:
                    # Non-purchase transactions reduce stock
                    updates.append({"n": name, "q": -add_qty, "c": cur_cost})

            if updates:
                if transaction_type == "Purchase":
                    conn.execute(
                        sql_text(
                            """
                            UPDATE ingredients
                               SET stock = COALESCE(stock,0) + :q,
                                   unit_cost = :c,
                                   last_updated = CURRENT_TIMESTAMP
                             WHERE name = :n
                            """
                        ),
                        updates,
                    )
                else:
                    conn.execute(
                        sql_text(
                            """
                            UPDATE ingredients
                               SET stock = COALESCE(stock,0) + :q,
                                   last_updated = CURRENT_TIMESTAMP
                             WHERE name = :n
                            """
                        ),
                        [{"n": u["n"], "q": u["q"]} for u in updates],
                    )

    bump_db_version()
    return po_id

def update_stock_from_usage(ingredient_name, quantity):
    """Atualiza o estoque após uso em produção"""
    return execute_query(
        "UPDATE ingredients SET stock = stock - :quantity, last_updated = CURRENT_TIMESTAMP WHERE name = :ingredient_name",
        {"quantity": quantity, "ingredient_name": ingredient_name}
    )

def check_ingredient_usage(ingredient_id):
    """Verifica se um ingrediente está em uso em receitas"""
    result = query_to_df(
        "SELECT COUNT(*) as count FROM recipe_items WHERE id_ingredient = :ingredient_id",
        {"ingredient_id": ingredient_id}
    )
    return result.iloc[0]['count'] > 0

def check_supplier_usage(supplier_name):
    """Verifica se um fornecedor tem compras associadas"""
    # Support both legacy purchases and the new purchase-by-order model
    result = query_to_df(
        """SELECT
               (SELECT COUNT(*) FROM purchases WHERE supplier = :supplier_name) +
               (SELECT COUNT(*) FROM purchase_orders WHERE supplier = :supplier_name)
               AS count""",
        {"supplier_name": supplier_name},
    )
    return (not result.empty) and (result.iloc[0]["count"] > 0)

# -----------------------------
# UI CONFIG
# -----------------------------
# Exigir login (viewer/admin) antes de carregar o app
require_login()


# Estilos CSS
st.markdown("""
<style>
.section-box {
    background: #FFFFFF10;
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #ffffff22;
    margin-bottom: 1.5rem;
}
.tank-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.equipment-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
.brewery-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: bold;
    margin-right: 0.5rem;
}
.status-empty { background-color: #10b981; color: white; }
.status-in-use { background-color: #3b82f6; color: white; }
.status-cleaning { background-color: #f59e0b; color: white; }
.status-maintenance { background-color: #ef4444; color: white; }
.status-ready { background-color: #8b5cf6; color: white; }
.status-active { background-color: #10b981; color: white; }
.status-inactive { background-color: #6b7280; color: white; }
.alert-box {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.5rem;
}
.alert-critical {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
}
.alert-warning {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
}
.alert-info {
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
}
.calendar-day {
    border: 1px solid #e0e0e0;
    padding: 0.5rem;
    height: 100px;
    overflow-y: auto;
}
.calendar-day.today {
    background-color: #e3f2fd;
    border: 2px solid #2196f3;
}
.calendar-day.weekend {
    background-color: #f8f9fa !important;
}
.calendar-day.weekend.today {
    background-color: #e3f2fd !important;
    border: 2px solid #2196f3 !important;
}
.calendar-event {
    background-color: #4caf50;
    color: white;
    padding: 2px 5px;
    border-radius: 3px;
    font-size: 0.8rem;
    margin-bottom: 2px;
}
.delete-confirmation {
    background-color: #fee;
    border: 2px solid #f99;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}
.delete-btn {
    background-color: #ff4444 !important;
    color: white !important;
    border: 1px solid #cc0000 !important;
}
.delete-btn:hover {
    background-color: #cc0000 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)



@st.cache_resource(show_spinner=False)
def ensure_db_initialized() -> bool:
    """Run schema creation/migrations once per process for speed."""
    init_database()
    return True
# -----------------------------
# INICIALIZAÇÃO
# -----------------------------
# Inicializar banco de dados (cria tabelas se não existirem)
ensure_db_initialized()

# Carregar dados do banco
data = get_all_data()

# Inicializar session_state
if 'delete_confirmation' not in st.session_state:
    st.session_state.delete_confirmation = {"type": None, "id": None, "name": None}

if 'ingredient_rows' not in st.session_state:
    st.session_state.ingredient_rows = 1

if 'recipe_ingredient_count' not in st.session_state:
    st.session_state.recipe_ingredient_count = 1

if 'edit_equipment' not in st.session_state:
    st.session_state.edit_equipment = None

if 'edit_supplier' not in st.session_state:
    st.session_state.edit_supplier = None

if 'edit_event' not in st.session_state:
    st.session_state.edit_event = None

if 'recipe_to_brew' not in st.session_state:
    st.session_state.recipe_to_brew = None

if 'edit_recipe' not in st.session_state:
    st.session_state.edit_recipe = None

if 'delete_recipe' not in st.session_state:
    st.session_state.delete_recipe = None

if 'update_batch' not in st.session_state:
    st.session_state.update_batch = None

if 'transfer_source' not in st.session_state:
    st.session_state.transfer_source = None

if 'selected_brewery' not in st.session_state:
    st.session_state.selected_brewery = None

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📘 Brewery Manager")

st.sidebar.markdown("---")
st.sidebar.subheader("📤 Export")
if st.sidebar.button("📥 Export to Excel (XLSX)", use_container_width=True):
    output = export_to_excel()
    
    st.sidebar.download_button(
        label="📥 Download Excel (XLSX)",
        data=output,
        file_name="brewery_backup.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Reset database
st.sidebar.markdown("---")

def get_alerts():
    alerts = []
    
    # Verificar estoque baixo
    ingredients_df = data.get('ingredients', pd.DataFrame())
    if not ingredients_df.empty:
        if "low_stock_threshold" in ingredients_df.columns:
            low_stock = ingredients_df[ingredients_df["stock"] < ingredients_df["low_stock_threshold"]]
        else:
            low_stock = ingredients_df[ingredients_df["stock"] < 10]
        
        for _, item in low_stock.iterrows():
            alerts.append({
                "type": "critical",
                "title": "⚠️ Low Stock Alert",
                "message": f"{item['name']} is below threshold: {item['stock']} {item['unit']}",
                "time": datetime.now().strftime("%H:%M")
            })
    
    # Verificar equipamentos com limpeza vencida
    equipment_df = data.get('equipment', pd.DataFrame())
    if not equipment_df.empty and "cleaning_due" in equipment_df.columns:
        try:
            equipment_copy = equipment_df.copy()
            equipment_copy["cleaning_due"] = pd.to_datetime(equipment_copy["cleaning_due"])
            overdue_cleaning = equipment_copy[equipment_copy["cleaning_due"] < datetime.now()]
            for _, eq in overdue_cleaning.iterrows():
                alerts.append({
                    "type": "warning",
                    "title": "🧹 Cleaning Overdue",
                    "message": f"{eq['name']} ({eq['type']}) is overdue for cleaning",
                    "time": datetime.now().strftime("%H:%M")
                })
        except:
            pass
    
    # Verificar ordens agendadas para hoje
    orders_df = data.get('production_orders', pd.DataFrame())
    if not orders_df.empty:
        today = datetime.now().date()
        scheduled_today = orders_df[
            (orders_df["status"] == "Scheduled") &
            (orders_df["scheduled_date"].notna())
        ].copy()
        
        if "scheduled_date" in scheduled_today.columns:
            scheduled_today["scheduled_date"] = pd.to_datetime(scheduled_today["scheduled_date"])
            scheduled_today_today = scheduled_today[scheduled_today["scheduled_date"].dt.date == today]
            
            for _, order in scheduled_today_today.iterrows():
                alerts.append({
                    "type": "info",
                    "title": "🏭 Production Scheduled Today",
                    "message": f"Order #{order['id_order']}: {order.get('recipe_name', 'Unknown')}",
                    "time": order["scheduled_date"].strftime("%H:%M") if pd.notna(order["scheduled_date"]) else ""
                })
    
    return alerts

# -----------------------------
# HANDLERS DE CONFIRMAÇÃO DE DELETE
# -----------------------------
def handle_delete_confirmation():
    """Lida com confirmações de delete"""
    delete_type = st.session_state.delete_confirmation["type"]
    delete_id = st.session_state.delete_confirmation["id"]
    delete_name = st.session_state.delete_confirmation["name"]
    
    if delete_type == "ingredient":
        st.markdown(f"""
        <div class="delete-confirmation">
            <h3>⚠️ Confirm Ingredient Deletion</h3>
            <p>Are you sure you want to delete <strong>'{delete_name}'</strong>?</p>
            <p>This action cannot be undone.</p>
        </div>
        """, unsafe_allow_html=True)
        
    elif delete_type == "supplier":
        st.markdown(f"""
        <div class="delete-confirmation">
            <h3>⚠️ Confirm Supplier Deletion</h3>
            <p>Are you sure you want to delete <strong>'{delete_name}'</strong>?</p>
            <p>This supplier will be removed from the system.</p>
        </div>
        """, unsafe_allow_html=True)
    elif delete_type == "brewery":
        st.markdown(f"""
        <div class="delete-confirmation">
            <h3>⚠️ Confirm Brewery Deletion</h3>
            <p>Are you sure you want to delete <strong>'{delete_name}'</strong>?</p>
            <p>This will also delete all equipment associated with this brewery.</p>
        </div>
        """, unsafe_allow_html=True)
    
    col_confirm1, col_confirm2, col_confirm3 = st.columns([1, 1, 2])
    with col_confirm1:
        if st.button("✅ Yes, Delete", type="primary", use_container_width=True):
            if delete_type == "ingredient":
                # Verificar se o ingrediente está em uso
                if check_ingredient_usage(delete_id):
                    st.error("Failed to delete ingredient. It may be used in recipes.")
                else:
                    delete_data("ingredients", "id = :id", {"id": delete_id})
                    st.success(f"Ingredient '{delete_name}' deleted successfully!")
            
            elif delete_type == "supplier":
                # Verificar se há compras associadas
                if check_supplier_usage(delete_name):
                    st.error("Failed to delete supplier. It may have associated purchases.")
                else:
                    delete_data("suppliers", "id_supplier = :id_supplier", {"id_supplier": delete_id})
                    st.success(f"Supplier '{delete_name}' deleted successfully!")
            
            elif delete_type == "brewery":
                # Deletar equipamentos associados primeiro
                delete_data("equipment", "brewery_id = :brewery_id", {"brewery_id": delete_id})
                delete_data("breweries", "id_brewery = :id_brewery", {"id_brewery": delete_id})
                st.success(f"Brewery '{delete_name}' deleted successfully!")
            
            # Clear confirmação
            st.session_state.delete_confirmation = {"type": None, "id": None, "name": None}
            data = get_all_data()
            st.rerun()
    
    with col_confirm2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.delete_confirmation = {"type": None, "id": None, "name": None}
            st.rerun()

# Verificar se há confirmação pendente
if st.session_state.delete_confirmation["type"] in ["ingredient", "supplier", "brewery"]:
    handle_delete_confirmation()

# -----------------------------
# Navegação
# -----------------------------
page = st.sidebar.radio("Navigation", [
    "Dashboard", "Breweries", "Ingredients", "Purchases", 
    "Recipes", "Production Orders", "Calendar"
], key="page")
st.sidebar.markdown("---")
st.sidebar.info(f"👤 Role: {st.session_state.get('auth_role','viewer')}")
if st.session_state.get('view_mode'):
    st.sidebar.caption('👀 View-only mode')

# -----------------------------
# Dashboard Page
# -----------------------------
if page == "Dashboard":
    st.title("🏭 Brewery Dashboard")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_breweries = len(data["breweries"])
        st.metric("Breweries", total_breweries)
    with col2:
        total_equipment = len(data["equipment"])
        st.metric("Equipment", total_equipment)
    with col3:
        orders_df = data["production_orders"]
        if not orders_df.empty:
            active_batches = len(orders_df[orders_df["status"].isin(["Fermenting", "Conditioning", "Packaging", "Brewing"])])
            st.metric("Active Batches", active_batches)
        else:
            st.metric("Active Batches", 0)
    with col4:
        total_ingredients = len(data["ingredients"])
        st.metric("Ingredients", total_ingredients)
    
    # Layout com duas colunas
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # CALENDÁRIO
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("📅 Production Calendar")
        
        today = datetime.now()
        col_cal1, col_cal2 = st.columns(2)
        with col_cal1:
            selected_month = st.selectbox("Month", range(1, 13), index=today.month-1, format_func=lambda x: calendar.month_name[x])
        with col_cal2:
            selected_year = st.selectbox("Year", range(today.year-1, today.year+2), index=1)
        
        # Criar calendário
        cal = calendar.monthcalendar(selected_year, selected_month)
        
        # Cabeçalho
        days_header = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        cols = st.columns(7)
        for i, day in enumerate(days_header):
            cols[i].write(f"**{day}**")
        
        # Dias
        for week in cal:
            cols = st.columns(7)
            for i, day in enumerate(week):
                if day == 0:
                    cols[i].write("")
                else:
                    current_date = date(selected_year, selected_month, day)
                    is_today = (current_date == today.date())
                    
                    day_class = "calendar-day today" if is_today else "calendar-day"
                    
                    with cols[i].container():
                        st.markdown(f'<div class="{day_class}">', unsafe_allow_html=True)
                        st.write(f"**{day}**")
                        
                        # Verificar eventos
                        events_today = pd.DataFrame()
                        events_df = data.get("calendar_events", pd.DataFrame())
                        if not events_df.empty:
                            events_df = events_df.copy()
                            if "start_date" in events_df.columns:
                                events_df["start_date"] = pd.to_datetime(events_df["start_date"]).dt.date
                                events_today = events_df[events_df["start_date"] == current_date]
                        
                        for _, event in events_today.iterrows():
                            event_color = {
                                "Brewing": "#4caf50",
                                "Transfer": "#2196f3",
                                "Packaging": "#ff9800"
                            }.get(event.get("event_type", "Other"), "#757575")
                            
                            st.markdown(f'<div class="calendar-event" style="background-color: {event_color};">{event.get("title", "Event")}</div>', unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add new event
        with st.expander("➕ Add New Event"):
            with st.form("add_event_form", clear_on_submit=True):
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    event_title = st.text_input("Event Title")
                    event_type = st.selectbox("Event Type", ["Brewing", "Fermentation", "Packaging", "Cleaning", "Maintenance", "Meeting", "Other"])
                with col_e2:
                    event_date = st.date_input("Event Date", today)
                    equipment = st.text_input("Equipment (Optional)")
                event_notes = st.text_area("Notes")
                submitted = st.form_submit_button("Add Event", type="primary", use_container_width=True)
            if submitted and event_title:
                new_event = {
                    "title": event_title,
                    "event_type": event_type,
                    "start_date": event_date,
                    "end_date": event_date,
                    "equipment": equipment,
                    "batch_id": "",
                    "notes": event_notes,
                    "created_by": "User",
                }
                insert_data("calendar_events", new_event)
                data = get_all_data()
                st.success("Event added!")
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # PRÓXIMAS ATIVIDADES
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("📋 Upcoming Activities")
        
        events_df = data.get("calendar_events", pd.DataFrame())
        if not events_df.empty:
            events_df = events_df.copy()
            if "start_date" in events_df.columns:
                events_df["start_date"] = pd.to_datetime(events_df["start_date"])
                upcoming_events = events_df[events_df["start_date"] >= pd.Timestamp(today.date())].sort_values("start_date").head(5)
                
                if len(upcoming_events) > 0:
                    for _, event in upcoming_events.iterrows():
                        col_a1, col_a2, col_a3 = st.columns([3, 2, 1])
                        with col_a1:
                            st.write(f"**{event['title']}**")
                            if pd.notna(event.get("notes")) and event["notes"]:
                                st.caption(event["notes"][:50] + "..." if len(event["notes"]) > 50 else event["notes"])
                        with col_a2:
                            event_date = event["start_date"].date()
                            days_until = (event_date - today.date()).days
                            if days_until == 0:
                                st.write("**Today**")
                            elif days_until == 1:
                                st.write("**Tomorrow**")
                            else:
                                st.write(f"In **{days_until} days**")
                        with col_a3:
                            st.write(f"_{event.get('event_type', 'Event')}_")
                        st.markdown("---")
                else:
                    st.info("No upcoming events scheduled")
            else:
                st.info("No events with dates")
        else:
            st.info("No events in calendar")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_right:
        # ALERTAS
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("🔔 Today's Alerts")
        
        alerts = get_alerts()
        
        if alerts:
            for alert in alerts:
                alert_class = f"alert-box alert-{alert['type']}"
                st.markdown(f"""
                <div class="{alert_class}">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div>
                            <strong>{alert['title']}</strong>
                            <p style="margin: 0.5rem 0 0 0;">{alert['message']}</p>
                        </div>
                        <small>{alert['time']}</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No critical alerts for today!")
        
        # ESTOQUE CRÍTICO
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("📦 Critical Stock")
        
        ingredients_df = data.get("ingredients", pd.DataFrame())
        if not ingredients_df.empty:
            if "low_stock_threshold" in ingredients_df.columns:
                critical_stock = ingredients_df[ingredients_df["stock"] < ingredients_df["low_stock_threshold"]]
            else:
                critical_stock = ingredients_df.nsmallest(5, "stock")
            
            if len(critical_stock) > 0:
                for _, item in critical_stock.head(3).iterrows():
                    stock_percentage = (item["stock"] / item["low_stock_threshold"] * 100) if "low_stock_threshold" in item else None
                    
                    col_s1, col_s2 = st.columns([3, 1])
                    with col_s1:
                        st.write(f"**{item['name']}**")
                        if stock_percentage:
                            st.progress(stock_percentage / 100)
                    with col_s2:
                        st.write(f"{item['stock']} {item['unit']}")
                    st.markdown("---")
                
                if len(critical_stock) > 3:
                    st.info(f"... and {len(critical_stock) - 3} more items")
            else:
                st.success("✅ All items have sufficient stock")
        else:
            st.info("No ingredient data available")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # PRODUÇÃO ATIVA
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.subheader("🏭 Active Production")
    
    orders_df = data.get("production_orders", pd.DataFrame())
    if not orders_df.empty:
        active_orders = orders_df[orders_df["status"].isin(["Fermenting", "Conditioning", "Packaging", "Brewing"])]
        
        if len(active_orders) > 0:
            cols = st.columns(min(len(active_orders), 4))
            
            for idx, (_, order) in enumerate(active_orders.iterrows()):
                with cols[idx % len(cols)]:
                    recipe_name = order.get('recipe_name', f"Order #{order['id_order']}")
                    
                    # Calcular progresso
                    progress = {
                        "Brewing": 0.25,
                        "Fermenting": 0.5,
                        "Conditioning": 0.75,
                        "Packaging": 0.9
                    }.get(order["status"], 0.1)
                    
                    st.metric(f"Order #{order['id_order']}", recipe_name)
                    st.write(f"**Volume:** {order['volume']}L")
                    st.write(f"**Status:** {order['status']}")
                    st.progress(progress)
                    
                    if "start_date" in order and pd.notna(order["start_date"]):
                        days_ago = (datetime.now() - pd.to_datetime(order["start_date"])).days
                        st.caption(f"Started {days_ago} days ago")
        else:
            st.info("No active production orders")
    else:
        st.info("No production orders")
    
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Breweries Page
# -----------------------------
elif page == "Breweries":
    st.title("🏭 Breweries & Equipment Management")


    if is_viewer():
        st.info("👀 View-only mode: you can browse and generate reports, but you can't create/edit/delete anything.")
        breweries_df = data.get('breweries', pd.DataFrame())
        equipment_df = data.get('equipment', pd.DataFrame())
        st.subheader('📍 Breweries')
        if breweries_df is None or breweries_df.empty:
            st.info('No breweries yet.')
        else:
            st.dataframe(breweries_df, use_container_width=True, height=320)
        st.subheader('⚙️ Equipment')
        if equipment_df is None or equipment_df.empty:
            st.info('No equipment yet.')
        else:
            st.dataframe(equipment_df, use_container_width=True, height=320)
        st.stop()
    
    brew_tabs = ["📍 Breweries", "⚙️ Equipment", "📊 Overview"]
    selected_brew_tab = st.radio("", brew_tabs, horizontal=True, key="breweries_tab")
    if selected_brew_tab == "📍 Breweries":
        # Add Nova Beerria
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("➕ Add New Brewery / Production Location")
        
        with st.form("add_brewery_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                brewery_name = st.text_input("Brewery Name*", key="new_brewery_name")
                brewery_type = st.selectbox(
                    "Brewery Type*",
                    ["Production Brewery", "Pilot System", "Taproom Brewery", "Contract Brewery", "Experimental", "Other"],
                    key="new_brewery_type"
                )
                address = st.text_input("Address", key="new_brewery_address")
                city = st.text_input("City", key="new_brewery_city")
                state = st.text_input("State/Province", key="new_brewery_state")
                
            with col2:
                country = st.text_input("Country", value="Norway", key="new_brewery_country")
                postal_code = st.text_input("Postal Code", key="new_brewery_postal")
                contact_person = st.text_input("Contact Person", key="new_brewery_contact")
                contact_phone = st.text_input("Contact Phone", placeholder="+47 XXX XX XXX", key="new_brewery_phone")
                contact_email = st.text_input("Contact Email", key="new_brewery_email")
            
            # Configurações
            st.markdown("---")
            st.subheader("⚙️ Brewery Configuration")
            
            col_config1, col_config2, col_config3 = st.columns(3)
            with col_config1:
                default_batch_size = st.number_input(
                    "Default Batch Size (L)*",
                    min_value=10.0,
                    max_value=10000.0,
                    value=1000.0,
                    step=50.0,
                    key="new_brewery_batch"
                )
                annual_capacity = st.number_input(
                    "Annual Capacity (hL)",
                    min_value=1.0,
                    value=100.0,
                    step=10.0,
                    key="new_brewery_capacity"
                )
            
            with col_config2:
                status = st.selectbox("Status*", ["Active", "Inactive", "Under Construction", "Maintenance"], key="new_brewery_status")
                established_date = st.date_input("Established Date", datetime.now().date(), key="new_brewery_established")
            
            with col_config3:
                has_lab = st.checkbox("Quality Lab Available", key="new_brewery_lab")
            
            description = st.text_area("Description / Notes", key="new_brewery_desc")
            
            submitted = st.form_submit_button("🏭 Add Brewery", type="primary", use_container_width=True)
            if submitted:
                if not brewery_name:
                    st.error("Brewery name is required!")
                else:
                    new_brewery = {
                        "name": brewery_name,
                        "type": brewery_type,
                        "address": address,
                        "city": city,
                        "state": state,
                        "country": country,
                        "postal_code": postal_code,
                        "contact_person": contact_person,
                        "contact_phone": contact_phone,
                        "contact_email": contact_email,
                        "default_batch_size": default_batch_size,
                        "annual_capacity_hl": annual_capacity,
                        "status": status,
"established_date": established_date,
                        "has_lab": 1 if has_lab else 0,
                        "description": description
                    }
                    
                    insert_data("breweries", new_brewery)
                    data = get_all_data()
                    st.success(f"✅ Brewery '{brewery_name}' added successfully!")
                    st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Listar Beerrias Existentes
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("📋 Existing Breweries")
        
        breweries_df = data.get("breweries", pd.DataFrame())
        equipment_df_for_counts = data.get("equipment", pd.DataFrame())
        equipment_counts = {}
        if not equipment_df_for_counts.empty and "brewery_id" in equipment_df_for_counts.columns:
            try:
                equipment_counts = equipment_df_for_counts.groupby("brewery_id").size().to_dict()
            except Exception:
                equipment_counts = {}

        if not breweries_df.empty:
            # Filtros
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            with col_filter1:
                status_filter = st.selectbox("Filter by Status", ["All", "Active", "Inactive", "Under Construction", "Maintenance"], key="status_filter")
            with col_filter2:
                type_filter = st.selectbox("Filter by Type", ["All"] + sorted(breweries_df["type"].unique().tolist()), key="type_filter")
            with col_filter3:
                search_name = st.text_input("Search by Name", key="search_brewery")
            
            # Aplicar filtros
            filtered_breweries = breweries_df.copy()
            
            if status_filter != "All":
                filtered_breweries = filtered_breweries[filtered_breweries["status"] == status_filter]
            
            if type_filter != "All":
                filtered_breweries = filtered_breweries[filtered_breweries["type"] == type_filter]
            
            if search_name:
                filtered_breweries = filtered_breweries[filtered_breweries["name"].str.contains(search_name, case=False, na=False)]
            
            # Mostrar
            for _, brewery in filtered_breweries.iterrows():
                with st.expander(f"🏭 {brewery['name']} - {brewery['type']}", expanded=False):
                    col_info1, col_info2, col_info3 = st.columns(3)
                    
                    with col_info1:
                        st.write("**Location Info**")
                        st.write(f"📍 {brewery.get('address', '')}")
                        st.write(f"🏙️ {brewery.get('city', '')}, {brewery.get('state', '')}")
                        st.write(f"🇳🇴 {brewery.get('country', '')}")
                        st.write(f"📞 {brewery.get('contact_person', '')}")
                        st.write(f"📧 {brewery.get('contact_email', '')}")
                    
                    with col_info2:
                        st.write("**Production Info**")
                        st.write(f"📦 Default Batch: {brewery.get('default_batch_size', 0):.0f}L")
                        st.write(f"🏭 Annual Capacity: {brewery.get('annual_capacity_hl', 0):.0f} hL")
                        est = brewery.get("established_date")
                        est_str = "N/A"
                        if est is not None and not (isinstance(est, float) and pd.isna(est)) and not pd.isna(est):
                            try:
                                est_dt = pd.to_datetime(est, errors="coerce")
                                if pd.notna(est_dt):
                                    est_str = est_dt.date().isoformat()
                                else:
                                    # if it's already a date-like object
                                    if hasattr(est, "isoformat"):
                                        est_str = est.isoformat()
                                    else:
                                        est_str = str(est)
                            except Exception:
                                if hasattr(est, "isoformat"):
                                    est_str = est.isoformat()
                                else:
                                    est_str = str(est)
                        st.write(f"📅 Established: {est_str}")
                        eq_count = 0
                        try:
                            eq_count = int(equipment_counts.get(brewery.get("id_brewery"), equipment_counts.get(str(brewery.get("id_brewery")), 0)))
                        except Exception:
                            eq_count = 0
                        st.write(f"⚙️ Equipment: {eq_count}")
                    
                    with col_info3:
                        st.write("**Features**")
                        if brewery.get('has_lab'):
                            st.write("🔬 Quality Lab")
                        
                        st.write("**Status**")
                        st.markdown(render_status_badge(brewery['status']), unsafe_allow_html=True)
                    
                    # Buttons
                    col_edit, col_delete = st.columns(2)
                    with col_edit:
                        if st.button("Edit", key=f"edit_{brewery['id_brewery']}", use_container_width=True):
                            st.session_state['edit_brewery'] = brewery['id_brewery']
                    with col_delete:
                        if st.button("🗑️ Delete", key=f"del_{brewery['id_brewery']}", use_container_width=True, type="secondary"):
                            st.session_state.delete_confirmation = {"type": "brewery", "id": brewery['id_brewery'], "name": brewery['name']}
                            st.rerun()

                    
                    if brewery.get('description'):
                        st.markdown("---")
                        st.write("**Description:**")
                        st.write(brewery['description'])
        else:
            st.info("No breweries registered yet.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    if selected_brew_tab == "⚙️ Equipment":
        # Modo de edição de equipamento
        if st.session_state.get('edit_equipment'):
            edit_eq_id = st.session_state['edit_equipment']
            
            # Buscar equipamento no banco
            eq_df = query_to_df("SELECT * FROM equipment WHERE id_equipment = :id_equipment", {"id_equipment": edit_eq_id})
            
            if not eq_df.empty:
                eq_data = eq_df.iloc[0]
                
                st.markdown("<div class='section-box'>", unsafe_allow_html=True)
                st.subheader(f"📝 Editing Equipment: {eq_data['name']}")
                
                # Formulário de edição
                col_eq_edit1, col_eq_edit2, col_eq_edit3 = st.columns(3)
                
                with col_eq_edit1:
                    equipment_name = st.text_input("Equipment Name*", value=eq_data['name'], key="edit_eq_name")
                    equipment_type = st.selectbox(
                        "Equipment Type*",
                        [
                            "Fermentation Tank", 
                            "Bright Beer Tank (BBT)", 
                            "Serving Tank", 
                            "Brew Kettle", 
                            "Mash Tun", 
                            "Whirlpool", 
                            "Heat Exchanger",
                            "Hot Liquor Tank (HLT)",
                            "Cold Liquor Tank (CLT)",
                            "Glycol Unit", 
                            "Pump", 
                            "Filter",
                            "Centrifuge",
                            "Carbonation Stone",
                            "Yeast Brink",
                            "CIP System",
                            "Other"
                        ],
                        index=[
                            "Fermentation Tank", 
                            "Bright Beer Tank (BBT)", 
                            "Serving Tank", 
                            "Brew Kettle", 
                            "Mash Tun", 
                            "Whirlpool", 
                            "Heat Exchanger",
                            "Hot Liquor Tank (HLT)",
                            "Cold Liquor Tank (CLT)",
                            "Glycol Unit", 
                            "Pump", 
                            "Filter",
                            "Centrifuge",
                            "Carbonation Stone",
                            "Yeast Brink",
                            "CIP System",
                            "Other"
                        ].index(eq_data['type']) if eq_data['type'] in [
                            "Fermentation Tank", 
                            "Bright Beer Tank (BBT)", 
                            "Serving Tank", 
                            "Brew Kettle", 
                            "Mash Tun", 
                            "Whirlpool", 
                            "Heat Exchanger",
                            "Hot Liquor Tank (HLT)",
                            "Cold Liquor Tank (CLT)",
                            "Glycol Unit", 
                            "Pump", 
                            "Filter",
                            "Centrifuge",
                            "Carbonation Stone",
                            "Yeast Brink",
                            "CIP System",
                            "Other"
                        ] else 0,
                        key="edit_eq_type"
                    )
                    
                    capacity = st.number_input(
                        "Capacity (L)*",
                        min_value=1.0,
                        value=float(eq_data['capacity_liters']),
                        step=1.0,
                        key="edit_eq_capacity"
                    )
                
                with col_eq_edit2:
                    manufacturer = st.text_input("Manufacturer", value=eq_data.get('manufacturer', ''), key="edit_eq_manufacturer")
                    model = st.text_input("Model", value=eq_data.get('model', ''), key="edit_eq_model")
                    serial_number = st.text_input("Serial Number", value=eq_data.get('serial_number', ''), key="edit_eq_serial")
                    
                    if "Tank" in equipment_type:
                        status_options = ["Empty", "In Use", "Cleaning", "Ready", "Maintenance", "Out of Service"]
                    else:
                        status_options = ["Operational", "Maintenance", "Out of Service", "In Use", "Standby"]
                    
                    current_status = eq_data.get('status', 'Operational')
                    status_index = status_options.index(current_status) if current_status in status_options else 0
                    status = st.selectbox("Status*", status_options, index=status_index, key="edit_eq_status")
                
                with col_eq_edit3:
                    # Converter datas
                    install_date = eq_data.get('install_date')
                    if pd.notna(install_date):
                        install_date_value = pd.to_datetime(install_date).date()
                    else:
                        install_date_value = datetime.now().date()
                    
                    install_date_edit = st.date_input("Installation Date", value=install_date_value, key="edit_eq_install")
                    
                    next_maintenance = eq_data.get('next_maintenance')
                    if pd.notna(next_maintenance):
                        next_maintenance_value = pd.to_datetime(next_maintenance).date()
                    else:
                        next_maintenance_value = datetime.now().date() + timedelta(days=90)
                    
                    next_maintenance_edit = st.date_input("Next Maintenance Due", value=next_maintenance_value, key="edit_eq_maintenance")
                    
                    if "Tank" in equipment_type:
                        material_options = ["Stainless Steel", "Glass-Lined Steel", "Plastic", "Wood", "Other"]
                        current_material = eq_data.get('material', 'Stainless Steel')
                        material_index = material_options.index(current_material) if current_material in material_options else 0
                        material = st.selectbox("Material", material_options, index=material_index, key="edit_eq_material")
                        
                        pressure_rating = st.number_input(
                            "Pressure Rating (psi)", 
                            min_value=0.0, 
                            value=float(eq_data.get('pressure_rating', 15.0)), 
                            step=0.5, 
                            key="edit_eq_pressure"
                        )
                        has_jacket = st.checkbox("Temperature Jacket", value=bool(eq_data.get('has_jacket', True)), key="edit_eq_jacket")
                        has_sight_glass = st.checkbox("Sight Glass", value=bool(eq_data.get('has_sight_glass', True)), key="edit_eq_sight")
                    else:
                        material_options = ["Stainless Steel", "Copper", "Plastic", "Other"]
                        current_material = eq_data.get('material', 'Stainless Steel')
                        material_index = material_options.index(current_material) if current_material in material_options else 0
                        material = st.selectbox("Material", material_options, index=material_index, key="edit_eq_material_other")
                
                cleaning_options = ["After each use", "Daily", "Weekly", "Monthly", "As needed"]
                current_cleaning = eq_data.get('cleaning_frequency', 'After each use')
                cleaning_index = cleaning_options.index(current_cleaning) if current_cleaning in cleaning_options else 0
                cleaning_frequency = st.selectbox(
                    "Cleaning Frequency",
                    cleaning_options,
                    index=cleaning_index,
                    key="edit_eq_cleaning"
                )
                
                current_volume = st.number_input(
                    "Current Volume (L)",
                    min_value=0.0,
                    max_value=capacity,
                    value=float(eq_data.get('current_volume', 0.0)),
                    step=1.0,
                    key="edit_eq_current_volume"
                )
                
                equipment_notes = st.text_area("Notes / Special Instructions", value=eq_data.get('notes', ''), key="edit_eq_notes")
                
                # Botões de ação
                col_edit_btn1, col_edit_btn2, col_edit_btn3 = st.columns(3)
                
                with col_edit_btn1:
                    if st.button("💾 Save Changes", type="primary", use_container_width=True, key="save_edit_eq"):
                        updates = {
                            "name": equipment_name,
                            "type": equipment_type,
                            "capacity_liters": capacity,
                            "manufacturer": manufacturer,
                            "model": model,
                            "serial_number": serial_number,
                            "material": material,
                            "status": status,
                            "current_volume": current_volume,
                            "install_date": install_date_edit,
                            "next_maintenance": next_maintenance_edit,
                            "cleaning_frequency": cleaning_frequency,
                            "notes": equipment_notes
                        }
                        
                        if "Tank" in equipment_type:
                            updates["pressure_rating"] = pressure_rating
                            updates["has_jacket"] = 1 if has_jacket else 0
                            updates["has_sight_glass"] = 1 if has_sight_glass else 0
                        
                        update_data("equipment", updates, "id_equipment = :id_equipment", {"id_equipment": edit_eq_id})
                        data = get_all_data()
                        st.success(f"✅ Equipment '{equipment_name}' updated successfully!")
                        
                        del st.session_state['edit_equipment']
                        st.rerun()
                
                with col_edit_btn2:
                    if st.button("❌ Cancel", use_container_width=True, key="cancel_edit_eq"):
                        del st.session_state['edit_equipment']
                        st.rerun()
                
                with col_edit_btn3:
                    if st.button("🗑️ Delete Equipment", type="secondary", use_container_width=True, key="delete_edit_eq"):
                        delete_data("equipment", "id_equipment = :id_equipment", {"id_equipment": edit_eq_id})
                        data = get_all_data()
                        st.warning(f"Equipment '{eq_data['name']}' deleted!")
                        
                        del st.session_state['edit_equipment']
                        st.rerun()
                
                st.markdown("</div>", unsafe_allow_html=True)
                st.stop()
        
        # SEÇÃO PRINCIPAL DE EQUIPAMENTOS
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("⚙️ Manage Equipment")
        
        col_eq1, col_eq2 = st.columns(2)
        with col_eq1:
            breweries_df = data.get("breweries", pd.DataFrame())
            if not breweries_df.empty:
                brewery_options = breweries_df["name"].tolist()

                # If we arrived here from a Brewery card ("View Equipment"), preselect that brewery.
                preferred_id = st.session_state.get("selected_brewery")
                if preferred_id is not None and preferred_id in set(breweries_df["id_brewery"].values):
                    preferred_name = breweries_df.loc[breweries_df["id_brewery"] == preferred_id, "name"].iloc[0]
                    st.session_state["equipment_brewery_select"] = preferred_name

                selected_brewery = st.selectbox(
                    "Select Brewery for Equipment*",
                    brewery_options,
                    key="equipment_brewery_select",
                )

                match = breweries_df[breweries_df["name"] == selected_brewery]
                brewery_id = match["id_brewery"].iloc[0] if not match.empty else None
                if brewery_id is None:
                    st.error("Selected brewery not found in the database.")
                    st.stop()
            else:
                st.warning("⚠️ Please add a brewery first!")
                brewery_id = None
        
        with col_eq2:
            equipment_action = st.radio(
                "Action",
                ["Add New Equipment", "View/Edit Existing"],
                horizontal=True
            )
        
        if equipment_action == "Add New Equipment" and brewery_id:
            st.markdown("---")
            st.subheader(f"➕ Add Equipment to {selected_brewery}")

            with st.form("add_equipment_form", clear_on_submit=True):
                col_eq_form1, col_eq_form2, col_eq_form3 = st.columns(3)

                with col_eq_form1:
                    equipment_name = st.text_input("Equipment Name*", key="new_eq_name")
                    equipment_type = st.selectbox(
                        "Equipment Type*",
                        [
                            "Fermentation Tank",
                            "Bright Beer Tank (BBT)",
                            "Serving Tank",
                            "Brew Kettle",
                            "Mash Tun",
                            "Whirlpool",
                            "Heat Exchanger",
                            "Hot Liquor Tank (HLT)",
                            "Cold Liquor Tank (CLT)",
                            "Glycol Unit",
                            "Pump",
                            "Chiller",
                            "Compressor",
                            "Keg Washer",
                            "Can Filler",
                            "Bottling Line",
                            "Labeler",
                            "Filter",
                            "Centrifuge",
                            "Carbonation Stone",
                            "Yeast Brink",
                            "CIP System",
                            "Other",
                        ],
                        key="new_eq_type",
                    )
                    capacity = st.number_input(
                        "Capacity (L)*",
                        min_value=1.0,
                        value=100.0,
                        step=1.0,
                        key="new_eq_capacity",
                    )

                with col_eq_form2:
                    manufacturer = st.text_input("Manufacturer", key="new_eq_manufacturer")
                    model = st.text_input("Model", key="new_eq_model")
                    serial_number = st.text_input("Serial Number", key="new_eq_serial")

                    if "Tank" in equipment_type:
                        status_options = ["Empty", "In Use", "Cleaning", "Ready", "Maintenance", "Out of Service"]
                    else:
                        status_options = ["Operational", "Maintenance", "Out of Service", "In Use", "Standby"]

                    status = st.selectbox("Status*", status_options, key="new_eq_status")

                with col_eq_form3:
                    install_date = st.date_input("Installation Date", datetime.now().date(), key="new_eq_install")
                    next_maintenance = st.date_input(
                        "Next Maintenance",
                        datetime.now().date() + timedelta(days=90),
                        key="new_eq_maintenance",
                    )

                    if "Tank" in equipment_type:
                        material = st.selectbox(
                            "Material",
                            ["Stainless Steel", "Coated Steel", "Plastic", "Wood", "Other"],
                            key="new_eq_material",
                        )
                        pressure_rating = st.number_input(
                            "Pressure Rating (psi)",
                            min_value=0.0,
                            value=15.0,
                            step=0.5,
                            key="new_eq_pressure",
                        )
                        has_jacket = st.checkbox("Temperature Jacket", value=True, key="new_eq_jacket")
                        has_sight_glass = st.checkbox("Sight Glass", value=True, key="new_eq_sight")
                    else:
                        material = st.selectbox(
                            "Material",
                            ["Stainless Steel", "Copper", "Plastic", "Other"],
                            key="new_eq_material_other",
                        )
                        pressure_rating = None
                        has_jacket = False
                        has_sight_glass = False

                    cleaning_frequency = st.selectbox(
                        "Cleaning Frequency",
                        ["After each use", "Daily", "Weekly", "Monthly", "As needed"],
                        key="new_eq_cleaning",
                    )

                equipment_notes = st.text_area("Notes / Special Instructions", key="new_eq_notes")

                submitted = st.form_submit_button("⚙️ Add Equipment", type="primary", use_container_width=True)

            if submitted:
                if not equipment_name:
                    st.error("Equipment name is required!")
                else:
                    new_equipment = {
                        "brewery_id": brewery_id,
                        "name": equipment_name,
                        "type": equipment_type,
                        "capacity_liters": capacity,
                        "manufacturer": manufacturer,
                        "model": model,
                        "serial_number": serial_number,
                        "material": material,
                        "status": status,
                        "install_date": install_date,
                        "next_maintenance": next_maintenance,
                        "cleaning_frequency": cleaning_frequency,
                        "cleaning_due": datetime.now().date() + timedelta(days=7),
                        "notes": equipment_notes,
                    }

                    if "Tank" in equipment_type and pressure_rating is not None:
                        new_equipment["pressure_rating"] = pressure_rating
                        new_equipment["has_jacket"] = 1 if has_jacket else 0
                        new_equipment["has_sight_glass"] = 1 if has_sight_glass else 0

                    insert_data("equipment", new_equipment)
                    data = get_all_data()
                    st.success(f"✅ Equipment '{equipment_name}' added to {selected_brewery}!")
                    st.rerun()

        elif equipment_action == "View/Edit Existing":
            st.markdown("---")
            st.subheader("📋 Existing Equipment")
            
            equipment_df = data.get("equipment", pd.DataFrame())
            if equipment_df.empty:
                st.info("ℹ️ No equipment registered yet. Use **Add New Equipment** to create the first one.")
            else:
                # Ensure optional columns exist (prevents KeyError)
                if "current_volume" not in equipment_df.columns:
                    equipment_df["current_volume"] = 0.0
                if "capacity_liters" not in equipment_df.columns:
                    equipment_df["capacity_liters"] = 0.0
                if "manufacturer" not in equipment_df.columns:
                    equipment_df["manufacturer"] = ""
                if "model" not in equipment_df.columns:
                    equipment_df["model"] = ""
                # Filtros
                col_eq_filter1, col_eq_filter2, col_eq_filter3 = st.columns(3)
                
                with col_eq_filter1:
                    eq_type_filter = st.selectbox(
                        "Filter by Type", 
                        ["All"] + sorted(equipment_df["type"].unique().tolist()),
                        key="eq_type_filter"
                    )
                
                with col_eq_filter2:
                    eq_status_filter = st.selectbox(
                        "Filter by Status",
                        ["All", "Empty", "In Use", "Ready", "Cleaning", "Maintenance", "Operational"],
                        key="eq_status_filter"
                    )
                
                with col_eq_filter3:
                    if not breweries_df.empty:
                        brewery_filter = st.selectbox(
                            "Filter by Brewery",
                            ["All"] + breweries_df["name"].tolist(),
                            key="eq_brewery_filter"
                        )
                    else:
                        brewery_filter = "All"
                
                # Aplicar filtros
                filtered_equipment = equipment_df.copy()
                if 'current_volume' not in filtered_equipment.columns:
                    filtered_equipment['current_volume'] = 0.0
                if 'capacity_liters' not in filtered_equipment.columns:
                    filtered_equipment['capacity_liters'] = 0.0
                if 'manufacturer' not in filtered_equipment.columns:
                    filtered_equipment['manufacturer'] = ''
                if 'model' not in filtered_equipment.columns:
                    filtered_equipment['model'] = ''
                
                if eq_type_filter != "All":
                    filtered_equipment = filtered_equipment[filtered_equipment["type"] == eq_type_filter]
                
                if eq_status_filter != "All":
                    filtered_equipment = filtered_equipment[filtered_equipment["status"] == eq_status_filter]
                
                if brewery_filter != "All" and not breweries_df.empty:
                    brewery_id_filter = breweries_df[breweries_df["name"] == brewery_filter]["id_brewery"].iloc[0]
                    filtered_equipment = filtered_equipment[filtered_equipment["brewery_id"] == brewery_id_filter]
                
                if len(filtered_equipment) > 0:
                    # Mostrar em cards
                    cols = st.columns(2)
                    for idx, (_, eq) in enumerate(filtered_equipment.iterrows()):
                        with cols[idx % 2]:
                            brewery_name = ""
                            if not breweries_df.empty and eq["brewery_id"] in breweries_df["id_brewery"].values:
                                brewery_name = breweries_df[breweries_df["id_brewery"] == eq["brewery_id"]]["name"].iloc[0]
                            
                            occupancy_pct = (eq.get("current_volume", 0) / eq["capacity_liters"]) * 100 if eq["capacity_liters"] > 0 else 0
                            
                            status_colors = {
                                "Empty": "#10b981",
                                "In Use": "#3b82f6", 
                                "Ready": "#8b5cf6",
                                "Cleaning": "#f59e0b",
                                "Maintenance": "#ef4444",
                                "Operational": "#10b981"
                            }
                            card_color = status_colors.get(eq["status"], "#6b7280")
                            
                            with st.container():
                                st.markdown(f"""
                                <div style="border: 2px solid {card_color}; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
                                    <h4>{eq['name']}</h4>
                                    <p><strong>Type:</strong> {eq['type']}</p>
                                    <p><strong>Brewery:</strong> {brewery_name}</p>
                                    <p><strong>Capacity:</strong> {eq['capacity_liters']:,}L</p>
                                    <p><strong>Current:</strong> {eq['current_volume']:,}L ({occupancy_pct:.1f}%)</p>
                                    <p><strong>Status:</strong> {render_status_badge(eq['status'])}</p>
                                    <p><strong>Manufacturer:</strong> {eq.get('manufacturer', 'N/A')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.progress(occupancy_pct / 100)
                                
                                col_edit, col_dup, col_transfer = st.columns(3)
                                with col_edit:
                                    if st.button("📝 Edit", key=f"edit_eq_{eq['id_equipment']}", use_container_width=True):
                                        st.session_state['edit_equipment'] = eq['id_equipment']
                                        st.rerun()
                                with col_dup:
                                    if st.button("📄 Duplicate", key=f"dup_eq_{eq['id_equipment']}", use_container_width=True):
                                        require_admin_action()
                                        dup = eq.to_dict()
                                        dup.pop('id_equipment', None)
                                        dup.pop('created_date', None)
                                        dup['name'] = "Copy of " + str(eq['name'])
                                        insert_data('equipment', dup)
                                        data = get_all_data()
                                        st.success(f"✅ Duplicated equipment as '{dup['name']}'.")
                                        st.rerun()
                                with col_transfer:
                                    if st.button("🔄 Transfer", key=f"transfer_eq_{eq['id_equipment']}", use_container_width=True):
                                        st.session_state['transfer_source'] = eq['name']
                    
                    # Tabela detalhada
                    st.markdown("---")
                    st.subheader("Detailed View")
                    display_cols = ["name", "type", "capacity_liters", "current_volume", "status", "manufacturer", "model"]
                    existing_cols = [c for c in display_cols if c in filtered_equipment.columns]
                    st.dataframe(
                        filtered_equipment[existing_cols].rename(columns={
                            "name": "Name",
                            "type": "Type", 
                            "capacity_liters": "Capacity (L)",
                            "current_volume": "Current (L)",
                            "status": "Status",
                            "manufacturer": "Manufacturer",
                            "model": "Model"
                        }),
                        use_container_width=True
                    )
                else:
                    if brewery_filter != "All":
                        st.info(f"ℹ️ No equipment registered for **{brewery_filter}** yet.")
                    else:
                        st.info("ℹ️ No equipment found with selected filters.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    if selected_brew_tab == "📊 Overview":
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("📊 Breweries & Equipment Overview")
        
        breweries_df = data.get("breweries", pd.DataFrame())
        equipment_df = data.get("equipment", pd.DataFrame())
        
        if not breweries_df.empty:
            # Estatísticas
            col_over1, col_over2, col_over3, col_over4 = st.columns(4)
            
            with col_over1:
                total_breweries = len(breweries_df)
                active_breweries = len(breweries_df[breweries_df["status"] == "Active"])
                st.metric("Total Breweries", total_breweries, f"{active_breweries} active")
            
            with col_over2:
                total_equipment = len(equipment_df)
                tanks = len(equipment_df[equipment_df["type"].str.contains("Tank", na=False)])
                st.metric("Total Equipment", total_equipment, f"{tanks} tanks")
            
            with col_over3:
                total_capacity = breweries_df["default_batch_size"].sum() if "default_batch_size" in breweries_df.columns else 0
                st.metric("Total Batch Capacity", f"{total_capacity:,.0f}L")
            
            with col_over4:
                if not equipment_df.empty:
                    equipment_in_use = len(equipment_df[equipment_df["status"] == "In Use"])
                    utilization = (equipment_in_use / len(equipment_df) * 100) if len(equipment_df) > 0 else 0
                    st.metric("Equipment Utilization", f"{utilization:.1f}%")
                else:
                    st.metric("Equipment Utilization", "0%")
            
            # Gráfico de distribuição
            st.markdown("---")
            st.subheader("🏭 Equipment Distribution by Brewery")
            
            if not equipment_df.empty:
                equipment_by_brewery = []
                for _, brewery in breweries_df.iterrows():
                    brewery_eq = equipment_df[equipment_df["brewery_id"] == brewery["id_brewery"]]
                    
                    equipment_by_brewery.append({
                        "Brewery": brewery["name"],
                        "Total Equipment": len(brewery_eq),
                        "Tanks": len(brewery_eq[brewery_eq["type"].str.contains("Tank", na=False)]),
                        "Fermenters": len(brewery_eq[brewery_eq["type"] == "Fermentation Tank"]),
                        "BBTs": len(brewery_eq[brewery_eq["type"] == "Bright Beer Tank (BBT)"]),
                        "Serving Tanks": len(brewery_eq[brewery_eq["type"] == "Serving Tank"])
                    })
                
                eq_df = pd.DataFrame(equipment_by_brewery)
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=eq_df["Brewery"],
                    y=eq_df["Total Equipment"],
                    name="Total Equipment",
                    marker_color="#4caf50"
                ))
                
                fig.add_trace(go.Bar(
                    x=eq_df["Brewery"],
                    y=eq_df["Tanks"],
                    name="Tanks",
                    marker_color="#2196f3"
                ))
                
                fig.update_layout(
                    title="Equipment Count by Brewery",
                    xaxis_title="Brewery",
                    yaxis_title="Number of Equipment",
                    barmode="group",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabela de capacidade
                st.markdown("---")
                st.subheader("📈 Capacity Overview")
                
                capacity_data = []
                for _, brewery in breweries_df.iterrows():
                    brewery_eq = equipment_df[equipment_df["brewery_id"] == brewery["id_brewery"]]
                    
                    capacity_data.append({
                        "Brewery": brewery["name"],
                        "Type": brewery["type"],
                        "Status": brewery["status"],
                        "Default Batch (L)": brewery.get("default_batch_size", 0),
                        "Total Tank Capacity (L)": brewery_eq[brewery_eq["type"].str.contains("Tank", na=False)]["capacity_liters"].sum(),
                        "Equipment Count": len(brewery_eq),
                        "Active Tanks": len(brewery_eq[(brewery_eq["type"].str.contains("Tank", na=False)) & (brewery_eq["status"] == "In Use")])
                    })
                
                capacity_df = pd.DataFrame(capacity_data)
                st.dataframe(capacity_df, use_container_width=True)
            else:
                st.info("Add equipment to see distribution charts")
            
            # Próximas manutenções
            st.markdown("---")
            st.subheader("🛠️ Upcoming Maintenance")
            
            if not equipment_df.empty and "next_maintenance" in equipment_df.columns:
                equipment_copy = equipment_df.copy()
                equipment_copy["next_maintenance"] = pd.to_datetime(equipment_copy["next_maintenance"])
                
                upcoming_maintenance = equipment_copy[
                    (equipment_copy["next_maintenance"] >= pd.Timestamp(datetime.now().date())) &
                    (equipment_copy["next_maintenance"] <= pd.Timestamp(datetime.now().date() + timedelta(days=30)))
                ].sort_values("next_maintenance")
                
                if len(upcoming_maintenance) > 0:
                    for _, eq in upcoming_maintenance.iterrows():
                        days_until = (eq["next_maintenance"].date() - datetime.now().date()).days
                        
                        brewery_name = ""
                        if not breweries_df.empty and eq["brewery_id"] in breweries_df["id_brewery"].values:
                            brewery_name = breweries_df[breweries_df["id_brewery"] == eq["brewery_id"]]["name"].iloc[0]
                        
                        col_m1, col_m2, col_m3 = st.columns([3, 2, 1])
                        with col_m1:
                            st.write(f"**{eq['name']}** ({brewery_name})")
                            st.write(f"Type: {eq['type']}")
                        with col_m2:
                            st.write(f"**Due:** {eq['next_maintenance'].date()}")
                        with col_m3:
                            if days_until <= 7:
                                st.error(f"**{days_until} days**")
                            elif days_until <= 14:
                                st.warning(f"**{days_until} days**")
                            else:
                                st.info(f"**{days_until} days**")
                        st.markdown("---")
                else:
                    st.success("✅ No maintenance scheduled for the next 30 days")
            else:
                st.info("No maintenance data available")
        else:
            st.info("Add breweries and equipment to see overview statistics")
        
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Ingredients Page
# -----------------------------
elif page == "Ingredients":
    st.title("🌾 Ingredients Management")


    if is_viewer():
        st.info('👀 View-only mode: read-only. To edit ingredients, log in as admin.')
        ingredients_df = data.get('ingredients', pd.DataFrame())
        if ingredients_df is None or ingredients_df.empty:
            st.info('No ingredients yet.')
            st.stop()
        view_cols = [c for c in ['name','manufacturer','category','stock','unit','unit_cost','low_stock_threshold','last_updated'] if c in ingredients_df.columns]
        view_df = ingredients_df.copy()
        st.subheader('📦 Ingredients')
        st.dataframe(view_df[view_cols] if view_cols else view_df, use_container_width=True, height=520)
        csv = (view_df[view_cols] if view_cols else view_df).to_csv(index=False).encode('utf-8')
        st.download_button('📥 Export (CSV)', csv, file_name='ingredients.csv', mime='text/csv')
        st.stop()
    
    # Tabs para Ingredients
    tab_stock, tab_add, tab_categories, tab_history = st.tabs([
        "📦 Current Stock",
        "➕ Add/Edit Ingredients",
        "📊 Categories & Analytics",
        "📜 Stock History"
    ])
    
    with tab_stock:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("📦 Current Ingredient Stock")
        
        ingredients_df = data.get("ingredients", pd.DataFrame())
        if not ingredients_df.empty:
            # Filtros
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            with col_filter1:
                category_filter = st.selectbox(
                    "Filter by Category",
                    ["All"] + sorted(ingredients_df["category"].dropna().unique().tolist()),
                    key="ing_category_filter"
                )
            with col_filter2:
                manufacturer_filter = st.selectbox(
                    "Filter by Manufacturer",
                    ["All"] + sorted(ingredients_df["manufacturer"].dropna().unique().tolist()),
                    key="ing_manufacturer_filter"
                )
            with col_filter3:
                low_stock_only = st.checkbox("Show Low Stock Only", key="low_stock_check")
            
            # Aplicar filtros
            filtered_ingredients = ingredients_df.copy()
            
            if category_filter != "All":
                filtered_ingredients = filtered_ingredients[filtered_ingredients["category"] == category_filter]
            
            if manufacturer_filter != "All":
                filtered_ingredients = filtered_ingredients[filtered_ingredients["manufacturer"] == manufacturer_filter]
            
            if low_stock_only:
                if "low_stock_threshold" in filtered_ingredients.columns:
                    filtered_ingredients = filtered_ingredients[
                        filtered_ingredients["stock"] < filtered_ingredients["low_stock_threshold"]
                    ]
                else:
                    filtered_ingredients = filtered_ingredients[filtered_ingredients["stock"] < 10]
            
            # Estatísticas rápidas
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                total_items = len(filtered_ingredients)
                st.metric("Total Items", total_items)
            with col_stat2:
                total_stock_value = (filtered_ingredients["stock"] * filtered_ingredients["unit_cost"]).sum()
                st.metric("Total Stock Value", f"${total_stock_value:,.2f}")
            with col_stat3:
                low_stock_count = len(filtered_ingredients[
                    filtered_ingredients["stock"] < filtered_ingredients["low_stock_threshold"]
                ]) if "low_stock_threshold" in filtered_ingredients.columns else 0
                st.metric("Low Stock Items", low_stock_count)
            with col_stat4:
                avg_stock_value = total_stock_value / total_items if total_items > 0 else 0
                st.metric("Avg. Value per Item", f"${avg_stock_value:,.2f}")
            
            # Tabela de estoque
            st.markdown("---")
            st.subheader("Stock Details")
            
            # Preparar dados para exibição
            display_df = filtered_ingredients.copy()
            
            # Add coluna de status
            def get_stock_status(stock, threshold):
                if stock <= 0:
                    return "❌ Out of Stock"
                elif stock < threshold * 0.3:
                    return "⚠️ Very Low"
                elif stock < threshold * 0.5:
                    return "⚠️ Low"
                elif stock < threshold:
                    return "📉 Below Threshold"
                else:
                    return "✅ Good"
            
            if "low_stock_threshold" in display_df.columns:
                display_df["Status"] = display_df.apply(
                    lambda row: get_stock_status(row["stock"], row["low_stock_threshold"]), 
                    axis=1
                )
            
            # Add valor total
            display_df["Total Value"] = display_df["stock"] * display_df["unit_cost"]
            
            # Selecionar colunas para exibição
            display_cols = ["name", "manufacturer", "category", "stock", "unit", 
                          "unit_cost", "Total Value"]
            if "low_stock_threshold" in display_df.columns:
                display_cols.insert(5, "low_stock_threshold")
            if "Status" in display_df.columns:
                display_cols.append("Status")
            
            # Renomear colunas
            column_mapping = {
                "name": "Ingredient",
                "manufacturer": "Manufacturer",
                "category": "Category",
                "stock": "Current Stock",
                "unit": "Unit",
                "unit_cost": "Unit Cost",
                "low_stock_threshold": "Low Stock Threshold",
                "Total Value": "Total Value",
                "Status": "Status"
            }
            
            display_df = display_df[display_cols].rename(columns=column_mapping)
            
            # Formatar valores
            display_df["Unit Cost"] = display_df["Unit Cost"].apply(lambda x: f"${x:.2f}")
            display_df["Total Value"] = display_df["Total Value"].apply(lambda x: f"${x:.2f}")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
            
            # Export dados
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Stock Date (CSV)",
                data=csv,
                file_name="ingredient_stock.csv",
                mime="text/csv",
                key="export_stock_csv"
            )
            
            # Gráfico de estoque por categoria
            st.markdown("---")
            st.subheader("📊 Stock Distribution by Category")
            
            if len(filtered_ingredients) > 0:
                category_stock = filtered_ingredients.groupby("category").agg({
                    "stock": "sum",
                    "name": "count"
                }).reset_index()
                category_stock.columns = ["Category", "Total Stock", "Item Count"]
                
                fig1 = make_subplots(rows=1, cols=2, subplot_titles=("Total Stock by Category", "Number of Items by Category"))
                
                fig1.add_trace(
                    go.Bar(
                        x=category_stock["Category"],
                        y=category_stock["Total Stock"],
                        name="Total Stock",
                        marker_color="#4caf50"
                    ),
                    row=1, col=1
                )
                
                fig1.add_trace(
                    go.Bar(
                        x=category_stock["Category"],
                        y=category_stock["Item Count"],
                        name="Item Count",
                        marker_color="#2196f3"
                    ),
                    row=1, col=2
                )
                
                fig1.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig1, use_container_width=True)
            
        else:
            st.info("No ingredients registered yet. Use the 'Add/Edit Ingredients' tab to add new ingredients.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab_add:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("➕ Add/Edit Ingredients")
        
        action = st.radio(
            "Select Action",
            ["Add New Ingredient", "Edit Existing Ingredient"],
            horizontal=True,
            key="ing_action"
        )
        

        if action == "Add New Ingredient":
            # Supplier selection is managed OUTSIDE the ingredient form
            # (Streamlit forbids st.button inside st.form)
            suppliers_df = data.get("suppliers", pd.DataFrame())
            supplier_options = (
                suppliers_df["name"].dropna().astype(str).tolist() if not suppliers_df.empty else []
            )

            selected_supplier = st.selectbox(
                "Supplier*",
                ["Select Supplier"] + supplier_options,
                key="new_ing_supplier_select",
            )

            if st.button("➕ Add new supplier", key="new_ing_add_supplier_btn", use_container_width=True):
                st.session_state["new_ing_show_add_supplier"] = True

            if st.session_state.get("new_ing_show_add_supplier"):
                st.markdown("**Add new supplier**")
                with st.form("new_ing_add_supplier_form", clear_on_submit=True):
                    sup_name = st.text_input("Supplier Name*", key="new_ing_new_sup_name")
                    contact_person = st.text_input("Contact Person", key="new_ing_new_sup_contact")
                    email = st.text_input("Email", key="new_ing_new_sup_email")
                    phone = st.text_input("Phone", key="new_ing_new_sup_phone")
                    notes_sup = st.text_area("Notes", key="new_ing_new_sup_notes")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        submitted_sup = st.form_submit_button(
                            "Save supplier", type="primary", use_container_width=True
                        )
                    with col_b:
                        cancelled_sup = st.form_submit_button("Cancel", use_container_width=True)

                if submitted_sup:
                    if not sup_name:
                        st.error("Supplier name is required!")
                    else:
                        # Avoid duplicates (case-insensitive)
                        if (
                            not suppliers_df.empty
                            and (suppliers_df["name"].astype(str).str.lower() == str(sup_name).lower()).any()
                        ):
                            canonical = suppliers_df.loc[
                                suppliers_df["name"].astype(str).str.lower() == str(sup_name).lower(),
                                "name",
                            ].iloc[0]
                            st.session_state["new_ing_supplier_select"] = str(canonical)
                        else:
                            insert_data(
                                "suppliers",
                                {
                                    "name": sup_name,
                                    "contact_person": contact_person,
                                    "email": email,
                                    "phone": phone,
                                    "notes": notes_sup,
                                },
                            )
                            st.session_state["new_ing_supplier_select"] = str(sup_name)

                        st.session_state["new_ing_show_add_supplier"] = False
                        data = get_all_data()
                        st.success(f"✅ Supplier '{sup_name}' added!")
                        st.rerun()

                if cancelled_sup:
                    st.session_state["new_ing_show_add_supplier"] = False
                    st.rerun()

            # Ingredient details form (ONLY st.form_submit_button inside)
            with st.form("add_ingredient_form", clear_on_submit=True):
                col_form1, col_form2 = st.columns(2)

                with col_form1:
                    ing_name = st.text_input("Ingredient Name*", key="new_ing_name")
                    manufacturer = st.text_input("Manufacturer", key="new_ing_manufacturer")

                    category_options = [
                        "Grain",
                        "Malt Extract",
                        "Hops",
                        "Yeast",
                        "Sugar",
                        "Water Treatment",
                        "Spices",
                        "Fruits",
                        "Other",
                    ]
                    category = st.selectbox("Category*", category_options, key="new_ing_category")

                    unit_options = ["kg", "g", "lb", "oz", "L", "mL", "pkg", "unit"]
                    unit = st.selectbox("Unit*", unit_options, key="new_ing_unit")

                    stock = st.number_input(
                        "Initial Stock*", min_value=0.0, value=0.0, step=0.1, key="new_ing_stock"
                    )

                with col_form2:
                    low_stock_threshold = st.number_input(
                        "Low Stock Threshold*",
                        min_value=0.0,
                        value=10.0,
                        step=0.1,
                        key="new_ing_threshold",
                    )

                # Additional fields (inside the form, outside columns)
                alpha_acid = st.number_input(
                    "Alpha Acid % (for hops)",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.1,
                    key="new_ing_alpha",
                )
                lot_number = st.text_input("Lot/Batch Number", key="new_ing_lot")
                expiry_date = st.date_input("Expiry Date (optional)", key="new_ing_expiry")
                notes = st.text_area("Notes", key="new_ing_notes")

                submitted = st.form_submit_button(
                    "➕ Add Ingredient", type="primary", use_container_width=True
                )

                if submitted:
                    if not ing_name or (selected_supplier == "Select Supplier"):
                        st.error("Ingredient name and supplier are required!")
                    else:
                        # Resolve supplier_id from suppliers table (best effort)
                        suppliers_df2 = data.get("suppliers", pd.DataFrame())
                        sid = None
                        if (
                            not suppliers_df2.empty
                            and selected_supplier
                            and selected_supplier != "Select Supplier"
                        ):
                            id_col = (
                                "id_supplier"
                                if "id_supplier" in suppliers_df2.columns
                                else ("id" if "id" in suppliers_df2.columns else None)
                            )
                            if id_col:
                                try:
                                    sid = int(
                                        suppliers_df2.loc[
                                            suppliers_df2["name"].astype(str) == str(selected_supplier),
                                            id_col,
                                        ].iloc[0]
                                    )
                                except Exception:
                                    sid = None

                        new_ingredient = {
                            "name": ing_name,
                            "manufacturer": manufacturer,
                            "category": category,
                            "unit": unit,
                            "stock": stock,
                            # Prices/costs are calculated from Purchases (incl. freight allocation)
                            "unit_cost": 0.0,
                            "low_stock_threshold": low_stock_threshold,
                            "alpha_acid": alpha_acid if category == "Hops" else 0.0,
                            "lot_number": lot_number,
                            "expiry_date": expiry_date if expiry_date else None,
                            "supplier_id": sid,
                            "supplier_name": None
                            if selected_supplier == "Select Supplier"
                            else str(selected_supplier),
                            "supplier": None
                            if selected_supplier == "Select Supplier"
                            else str(selected_supplier),
                            "notes": notes,
                        }

                        insert_data("ingredients", new_ingredient)
                        data = get_all_data()
                        st.success(f"✅ Ingredient '{ing_name}' added successfully!")
                        st.rerun()
        
        else:  # Edit Existing Ingredient
            ingredients_df = data.get("ingredients", pd.DataFrame())
            if not ingredients_df.empty:
                ingredient_options = ingredients_df["name"].tolist()
                selected_ingredient = st.selectbox(
                    "Select Ingredient to Edit",
                    ingredient_options,
                    key="edit_ing_select"
                )
                
                if selected_ingredient:
                    # Obter dados atuais
                    ing_data = ingredients_df[ingredients_df["name"] == selected_ingredient].iloc[0]
                    
                    col_edit1, col_edit2 = st.columns(2)
                    
                    with col_edit1:
                        new_name = st.text_input("Ingredient Name", value=ing_data["name"], key="edit_ing_name")
                        new_manufacturer = st.text_input("Manufacturer", value=ing_data.get("manufacturer", ""), key="edit_ing_manufacturer")
                        
                        category_options = [
                            "Grain", "Malt Extract", "Hops", "Yeast", "Sugar", 
                            "Water Treatment", "Spices", "Fruits", "Other"
                        ]
                        current_category = ing_data.get("category", "Grain")
                        new_category = st.selectbox("Category", category_options, 
                                                  index=category_options.index(current_category) if current_category in category_options else 0,
                                                  key="edit_ing_category")
                        
                        unit_options = ["kg", "g", "lb", "oz", "L", "mL", "pkg", "unit"]
                        current_unit = ing_data.get("unit", "kg")
                        new_unit = st.selectbox("Unit", unit_options, 
                                              index=unit_options.index(current_unit) if current_unit in unit_options else 0,
                                              key="edit_ing_unit")
                        
                        new_stock = st.number_input("Current Stock", 
                                                  value=float(ing_data.get("stock", 0)), 
                                                  min_value=0.0, step=0.1, key="edit_ing_stock")
                    
                    with col_edit2:
                        # Unit cost is calculated from Purchases (incl. freight allocation)
                        current_cost = float(ing_data.get("unit_cost", 0) or 0)
                        st.metric("Current Unit Cost (calculated)", f"${current_cost:,.4f}")

                        # Supplier relationship
                        suppliers_df = data.get("suppliers", pd.DataFrame())
                        supplier_options = suppliers_df["name"].dropna().astype(str).tolist() if not suppliers_df.empty else []
                        current_supplier = str(ing_data.get("supplier_name") or ing_data.get("supplier") or "")
                        sup_choices = ["Select Supplier"] + supplier_options
                        sup_index = sup_choices.index(current_supplier) if current_supplier in sup_choices else 0
                        new_supplier = st.selectbox("Supplier", sup_choices, index=sup_index, key="edit_ing_supplier")

                        new_threshold = st.number_input(
                            "Low Stock Threshold",
                            value=float(ing_data.get("low_stock_threshold", 10) or 0),
                            min_value=0.0,
                            step=0.1,
                            key="edit_ing_threshold",
                        )

                        # Campos específicos para lúpulo
                        if new_category == "Hops":
                            new_alpha = st.number_input("Alpha Acid %", 
                                                       value=float(ing_data.get("alpha_acid", 0)), 
                                                       min_value=0.0, max_value=100.0, step=0.1, key="edit_ing_alpha")
                        else:
                            new_alpha = ing_data.get("alpha_acid", 0)
                        
                        new_lot = st.text_input("Lot/Batch Number", value=ing_data.get("lot_number", ""), key="edit_ing_lot")
                        
                        # Date de validade
                        expiry = ing_data.get("expiry_date")
                        if pd.notna(expiry):
                            new_expiry = st.date_input("Expiry Date", value=pd.to_datetime(expiry).date(), key="edit_ing_expiry")
                        else:
                            new_expiry = st.date_input("Expiry Date (optional)", key="edit_ing_expiry_none")
                    
                    new_notes = st.text_area("Notes", value=ing_data.get("notes", ""), key="edit_ing_notes")
                    
                    col_btn1, col_btn2, col_btn3 = st.columns(3)
                    with col_btn1:
                        if st.button("💾 Save Changes", use_container_width=True, key="save_ing_btn"):
                            # Atualizar dados
                            # Resolve supplier_id (best effort)
                            suppliers_df_edit = data.get("suppliers", pd.DataFrame())
                            new_supplier_id = None
                            if not suppliers_df_edit.empty and new_supplier and new_supplier != "Select Supplier":
                                id_col = "id_supplier" if "id_supplier" in suppliers_df_edit.columns else ("id" if "id" in suppliers_df_edit.columns else None)
                                if id_col:
                                    try:
                                        new_supplier_id = int(
                                            suppliers_df_edit.loc[
                                                suppliers_df_edit["name"].astype(str) == str(new_supplier),
                                                id_col,
                                            ].iloc[0]
                                        )
                                    except Exception:
                                        new_supplier_id = None

                            updates = {
                                "name": new_name,
                                "manufacturer": new_manufacturer,
                                "category": new_category,
                                "unit": new_unit,
                                "stock": new_stock,
                                "supplier_id": new_supplier_id,
                                "supplier_name": None if new_supplier == "Select Supplier" else str(new_supplier),
                                "supplier": None if new_supplier == "Select Supplier" else str(new_supplier),
                                "low_stock_threshold": new_threshold,
                                "alpha_acid": new_alpha,
                                "lot_number": new_lot,
                                "notes": new_notes
                            }
                            
                            if 'new_expiry' in locals():
                                updates["expiry_date"] = new_expiry if new_expiry else None
                            
                            update_data("ingredients", updates, "id = :id", {"id": ing_data["id"]})
                            data = get_all_data()
                            st.success(f"✅ Ingredient '{new_name}' updated successfully!")
                            st.rerun()
                    
                    with col_btn2:
                        if st.button("🗑️ Delete Ingredient", use_container_width=True, type="secondary", key="delete_ing_btn"):
                            # Verificar se o ingrediente está em uso em receitas
                            in_use = check_ingredient_usage(ing_data["id"])
                            
                            if in_use:
                                st.error("Cannot delete ingredient that is used in recipes! Remove it from recipes first.")
                            else:
                                st.session_state.delete_confirmation = {"type": "ingredient", "id": ing_data["id"], "name": selected_ingredient}
                                st.rerun()
                    
                    with col_btn3:
                        if st.button("🔄 Reset Stock", use_container_width=True, key="reset_stock_btn"):
                            # Apenas resetar o estoque para 0
                            update_data("ingredients", {"stock": 0}, "id = :id", {"id": ing_data["id"]})
                            data = get_all_data()
                            st.warning(f"⚠️ Stock for '{selected_ingredient}' reset to 0!")
                            st.rerun()
            else:
                st.info("No ingredients available to edit.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab_categories:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("📊 Categories & Analytics")
        
        ingredients_df = data.get("ingredients", pd.DataFrame())
        if not ingredients_df.empty:
            # Análise por categoria
            category_analysis = ingredients_df.groupby("category").agg({
                "name": "count",
                "stock": "sum",
                "unit_cost": "mean"
            }).reset_index()
            
            # Calcular o valor total separadamente
            category_analysis["Total Value"] = category_analysis["stock"] * category_analysis["unit_cost"]
            
            # Renomear colunas
            category_analysis.columns = ["Category", "Item Count", "Total Stock", "Avg Unit Cost", "Total Value"]
            
            # Formatar
            category_analysis["Avg Unit Cost"] = category_analysis["Avg Unit Cost"].round(2)
            category_analysis["Total Value"] = category_analysis["Total Value"].round(2)
            
            col_cat1, col_cat2 = st.columns(2)
            
            with col_cat1:
                st.subheader("📈 Category Overview")
                st.dataframe(
                    category_analysis.sort_values("Total Value", ascending=False),
                    use_container_width=True
                )
            
            with col_cat2:
                st.subheader("💰 Value Distribution")
                
                fig2 = go.Figure(data=[go.Pie(
                    labels=category_analysis["Category"],
                    values=category_analysis["Total Value"],
                    hole=0.3,
                    textinfo='label+percent'
                )])
                
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Análise de custos
            st.markdown("---")
            st.subheader("💵 Cost Analysis")
            
            # Top 10 ingredientes mais valiosos
            ingredients_df["total_value"] = ingredients_df["stock"] * ingredients_df["unit_cost"]
            top_valuable = ingredients_df.nlargest(10, "total_value")
            
            fig3 = go.Figure(data=[
                go.Bar(
                    x=top_valuable["name"],
                    y=top_valuable["total_value"],
                    marker_color="#ff6b6b",
                    text=top_valuable["total_value"].round(2),
                    textposition='auto'
                )
            ])
            
            fig3.update_layout(
                title="Top 10 Most Valuable Ingredients (by stock value)",
                xaxis_title="Ingredient",
                yaxis_title="Total Value ($)",
                height=400
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # Análise de fornecedores
            st.markdown("---")
            st.subheader("🏭 Manufacturer Analysis")
            
            if "manufacturer" in ingredients_df.columns:
                manufacturer_analysis = ingredients_df.groupby("manufacturer").agg({
                    "name": "count",
                    "total_value": "sum"
                }).reset_index()
                
                manufacturer_analysis.columns = ["Manufacturer", "Item Count", "Total Value"]
                manufacturer_analysis = manufacturer_analysis.sort_values("Total Value", ascending=False)
                
                st.dataframe(
                    manufacturer_analysis,
                    use_container_width=True
                )
        
        else:
            st.info("Add ingredients to see category analytics.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab_history:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("📜 Stock Movement History")
        
        # Mostrar as compras relacionadas
        purchases_df = data.get("purchases", pd.DataFrame())
        if not purchases_df.empty:
            # Filtrar apenas compras
            purchase_history = purchases_df[purchases_df["transaction_type"] == "Purchase"].copy()
            
            if len(purchase_history) > 0:
                # Filtros
                col_hist1, col_hist2 = st.columns(2)
                with col_hist1:
                    hist_ingredient = st.selectbox(
                        "Filter by Ingredient",
                        ["All"] + sorted(purchase_history["ingredient"].dropna().unique().tolist()),
                        key="hist_ing_filter"
                    )
                with col_hist2:
                    date_range = st.date_input(
                        "Date Range",
                        [datetime.now().date() - timedelta(days=30), datetime.now().date()],
                        key="hist_date_range"
                    )
                
                # Aplicar filtros
                filtered_history = purchase_history.copy()
                
                if hist_ingredient != "All":
                    filtered_history = filtered_history[filtered_history["ingredient"] == hist_ingredient]
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    if "date" in filtered_history.columns:
                        filtered_history["date"] = pd.to_datetime(filtered_history["date"])
                        filtered_history = filtered_history[
                            (filtered_history["date"] >= pd.Timestamp(start_date)) &
                            (filtered_history["date"] <= pd.Timestamp(end_date))
                        ]
                
                # Mostrar histórico
                if len(filtered_history) > 0:
                    display_cols = ["date", "ingredient", "quantity", "unit", "total_cost", "order_number", "notes"]
                    display_df = filtered_history[display_cols].copy()
                    display_df["date"] = pd.to_datetime(display_df["date"]).dt.date
                    
                    st.dataframe(
                        display_df.rename(columns={
                            "date": "Date",
                            "ingredient": "Ingredient",
                            "quantity": "Quantity",
                            "unit": "Unit",
                            "total_cost": "Total Cost",
                            "order_number": "Order Number",
                            "notes": "Notes"
                        }),
                        use_container_width=True
                    )
                    
                    # Estatísticas do período
                    st.markdown("---")
                    st.subheader("📊 Period Summary")
                    
                    col_sum1, col_sum2, col_sum3 = st.columns(3)
                    with col_sum1:
                        total_quantity = filtered_history["quantity"].sum()
                        st.metric("Total Quantity", f"{total_quantity:,.1f}")
                    with col_sum2:
                        total_cost = filtered_history["total_cost"].sum()
                        st.metric("Total Cost", f"${total_cost:,.2f}")
                    with col_sum3:
                        avg_cost_per_unit = total_cost / total_quantity if total_quantity > 0 else 0
                        st.metric("Avg. Cost per Unit", f"${avg_cost_per_unit:,.2f}")
                else:
                    st.info("No purchase history found for the selected filters.")
            else:
                st.info("No purchase history available.")
        else:
            st.info("No purchase data available. Purchases will appear here once recorded.")
        
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Purchases Page
# -----------------------------
elif page == "Purchases":
    st.title("🛒 Purchases & Inventory Management")


    if is_viewer():
        st.info('👀 View-only mode: read-only. To record purchases/suppliers, log in as admin.')
        po = data.get('purchase_orders', pd.DataFrame())
        poi = data.get('purchase_order_items', pd.DataFrame())
        legacy = data.get('purchases', pd.DataFrame())
        st.subheader('🧾 Purchase Orders')
        if po is None or po.empty:
            st.info('No purchase orders yet.')
        else:
            st.dataframe(po, use_container_width=True, height=260)
        st.subheader('📦 Purchase Order Items')
        if poi is None or poi.empty:
            st.info('No purchase order items yet.')
        else:
            st.dataframe(poi, use_container_width=True, height=260)
        if legacy is not None and not legacy.empty:
            st.subheader('🗂️ Legacy Purchases')
            st.dataframe(legacy, use_container_width=True, height=220)
        st.stop()
    
    # Tabs para Purchases
    tab_new, tab_history, tab_suppliers, tab_reports = st.tabs([
        "🛍️ New Purchase",
        "📜 Purchase History",
        "🏭 Suppliers",
        "📊 Reports & Analytics"
    ])
    
    with tab_new:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("🧾 Record Purchase (by Order)")
        st.caption(
            "Supplier + order number first, then add multiple items. "
            "The *freight total* is automatically allocated per unit across all items and included in the effective unit cost."
        )

        ingredients_df = data.get("ingredients", pd.DataFrame())
        suppliers_df = data.get("suppliers", pd.DataFrame())

        # --- Header ---
        col_h1, col_h2, col_h3 = st.columns(3)
        with col_h1:
            transaction_type = st.selectbox(
                "Transaction Type*",
                ["Purchase", "Return", "Adjustment", "Sample", "Other"],
                key="po_type",
            )
        with col_h2:
            supplier_options = (
                suppliers_df["name"].dropna().astype(str).tolist() if not suppliers_df.empty else []
            )

            supplier = st.selectbox(
                "Supplier*",
                ["Select Supplier"] + supplier_options,
                key="po_supplier",
            )

            # Supplier management happens through Purchases
            if st.button("➕ Add new supplier", key="po_add_supplier_btn", use_container_width=True):
                st.session_state["po_show_add_supplier"] = True

            if st.session_state.get("po_show_add_supplier"):
                st.markdown("**Add new supplier**")
                with st.form("po_add_supplier_form", clear_on_submit=True):
                    sup_name = st.text_input("Supplier Name*", key="po_new_sup_name")
                    contact_person = st.text_input("Contact Person", key="po_new_sup_contact")
                    email = st.text_input("Email", key="po_new_sup_email")
                    phone = st.text_input("Phone", key="po_new_sup_phone")
                    notes = st.text_area("Notes", key="po_new_sup_notes")
                    submitted = st.form_submit_button("Save supplier", type="primary", use_container_width=True)

                if submitted:
                    if not sup_name:
                        st.error("Supplier name is required!")
                    else:
                        # If supplier already exists, just select it
                        if suppliers_df is not None and not suppliers_df.empty and (suppliers_df["name"].astype(str).str.lower() == str(sup_name).lower()).any():
                            # Pick the canonical stored name
                            canonical = suppliers_df.loc[
                                suppliers_df["name"].astype(str).str.lower() == str(sup_name).lower(),
                                "name",
                            ].iloc[0]
                            st.session_state["po_supplier"] = str(canonical)
                        else:
                            insert_data(
                                "suppliers",
                                {
                                    "name": sup_name,
                                    "contact_person": contact_person,
                                    "email": email,
                                    "phone": phone,
                                    "notes": notes,
                                },
                            )
                            st.session_state["po_supplier"] = str(sup_name)

                        st.session_state["po_show_add_supplier"] = False
                        data = get_all_data()
                        st.success(f"✅ Supplier '{sup_name}' added!")
                        st.rerun()

                if st.button("Cancel", key="po_cancel_add_supplier", use_container_width=True):
                    st.session_state["po_show_add_supplier"] = False
                    st.rerun()
        with col_h3:

            order_number = st.text_input("Order / Purchase Number", key="po_order_number")

        col_h4, col_h5 = st.columns(2)
        with col_h4:
            po_date = st.date_input("Purchase Date", datetime.now().date(), key="po_date")
        with col_h5:
            freight_total = st.number_input(
                "Freight Total",
                min_value=0.0,
                value=0.0,
                step=0.01,
                format="%.2f",
                disabled=(transaction_type != "Purchase"),
                key="po_freight_total",
            )
        notes = st.text_area("Notes", placeholder="Any additional notes for this order…", key="po_notes")

        st.markdown("---")
        st.subheader("🧺 Items")

        if ingredients_df.empty:
            st.warning("⚠️ No ingredients available. Please add ingredients first.")
        else:
            # Filter ingredients by supplier relationship
            eligible_df = ingredients_df.copy()

            if supplier and supplier != "Select Supplier" and not suppliers_df.empty:
                # Try to resolve supplier id from suppliers table
                id_col = "id_supplier" if "id_supplier" in suppliers_df.columns else ("id" if "id" in suppliers_df.columns else None)
                supplier_id = None
                if id_col:
                    try:
                        supplier_id = int(
                            suppliers_df.loc[suppliers_df["name"].astype(str) == str(supplier), id_col].iloc[0]
                        )
                    except Exception:
                        supplier_id = None

                # Backward-compatible matching: supplier_id OR supplier_name OR legacy supplier text
                mask = pd.Series([False] * len(eligible_df), index=eligible_df.index)

                if supplier_id is not None and "supplier_id" in eligible_df.columns:
                    try:
                        mask = mask | (eligible_df["supplier_id"].astype(object) == supplier_id)
                        mask = mask | (eligible_df["supplier_id"].astype(str) == str(supplier_id))
                    except Exception:
                        pass

                if "supplier_name" in eligible_df.columns:
                    try:
                        mask = mask | (eligible_df["supplier_name"].astype(str).str.lower() == str(supplier).lower())
                    except Exception:
                        pass

                if "supplier" in eligible_df.columns:
                    try:
                        mask = mask | (eligible_df["supplier"].astype(str).str.lower() == str(supplier).lower())
                    except Exception:
                        pass

                eligible_df = eligible_df[mask]

            if supplier and supplier != "Select Supplier" and eligible_df.empty:
                st.warning("⚠️ This supplier has no linked ingredients yet. Add ingredients and link them to this supplier first.")

            ingredient_options = sorted(eligible_df["name"].dropna().astype(str).tolist())

            # --- Items entry (real dropdowns + unit auto from ingredient registry) ---
            # We avoid st.data_editor here because some Streamlit versions render SelectboxColumn like a text input.
            unit_by_name = {}
            try:
                if ingredients_df is not None and not ingredients_df.empty:
                    if "name" in ingredients_df.columns:
                        # best-effort for unit column naming
                        unit_col = None
                        for c in ["unit", "uom", "Unit", "UOM"]:
                            if c in ingredients_df.columns:
                                unit_col = c
                                break
                        if unit_col:
                            unit_by_name = dict(
                                zip(
                                    ingredients_df["name"].astype(str).tolist(),
                                    ingredients_df[unit_col].astype(str).fillna("").tolist(),
                                )
                            )
            except Exception:
                unit_by_name = {}

            if "po_items" not in st.session_state:
                st.session_state.po_items = [
                    {"Ingredient": "", "Quantity": 0.0, "Unit": "", "Unit Price": 0.0}
                ]

            add_col, _ = st.columns([1, 5])
            with add_col:
                if st.button("➕ Add item", use_container_width=True, key="po_add_item_btn"):
                    st.session_state.po_items.append(
                        {"Ingredient": "", "Quantity": 0.0, "Unit": "", "Unit Price": 0.0}
                    )

            # Render rows
            rows_to_remove = []
            for i, row in enumerate(list(st.session_state.po_items)):
                c1, c2, c3, c4, c5 = st.columns([5, 2, 2, 2, 1])

                # Ingredient dropdown (with arrow)
                current_ing = str(row.get("Ingredient", "") or "")
                ing_options = [""] + ingredient_options
                try:
                    ing_idx = ing_options.index(current_ing) if current_ing in ing_options else 0
                except Exception:
                    ing_idx = 0
                with c1:
                    ing_val = st.selectbox(
                        "Ingredient",
                        ing_options,
                        index=ing_idx,
                        key=f"po_item_ing_{i}",
                        label_visibility="collapsed",
                    )

                # Quantity
                with c2:
                    qty_val = st.number_input(
                        "Quantity",
                        min_value=0.0,
                        value=float(row.get("Quantity", 0.0) or 0.0),
                        step=0.1,
                        format="%.3f",
                        key=f"po_item_qty_{i}",
                        label_visibility="collapsed",
                    )

                # Unit (auto from ingredient registry; read-only unless missing)
                unit_val = str(unit_by_name.get(str(ing_val), "") or "")
                with c3:
                    if ing_val and unit_val:
                        st.text_input(
                            "Unit",
                            value=unit_val,
                            disabled=True,
                            key=f"po_item_unit_{i}",
                            label_visibility="collapsed",
                        )
                    else:
                        # fallback if ingredient has no unit configured
                        fallback_units = ["", "kg", "g", "lb", "oz", "L", "mL", "unit"]
                        unit_val = st.selectbox(
                            "Unit",
                            fallback_units,
                            index=fallback_units.index(str(row.get("Unit", "") or ""))
                            if str(row.get("Unit", "") or "") in fallback_units
                            else 0,
                            key=f"po_item_unit_sel_{i}",
                            label_visibility="collapsed",
                        )

                # Unit price
                with c4:
                    price_val = st.number_input(
                        "Unit Price",
                        min_value=0.0,
                        value=float(row.get("Unit Price", 0.0) or 0.0),
                        step=0.01,
                        format="%.2f",
                        key=f"po_item_price_{i}",
                        label_visibility="collapsed",
                    )

                # Remove
                with c5:
                    if st.button("🗑️", key=f"po_item_rm_{i}"):
                        rows_to_remove.append(i)

                # Persist row
                st.session_state.po_items[i] = {
                    "Ingredient": str(ing_val),
                    "Quantity": float(qty_val),
                    "Unit": str(unit_val),
                    "Unit Price": float(price_val),
                }

            # Apply removals (reverse so indices don't shift)
            for idx in sorted(rows_to_remove, reverse=True):
                try:
                    st.session_state.po_items.pop(idx)
                except Exception:
                    pass

            # Build a dataframe for downstream preview/save logic
            edited_items = pd.DataFrame(
                st.session_state.po_items,
                columns=["Ingredient", "Quantity", "Unit", "Unit Price"],
            )

            # --- Preview / freight allocation ---
            items = edited_items.copy()
            items = items.fillna({"Ingredient": "", "Quantity": 0.0, "Unit Price": 0.0})
            items = items[(items["Ingredient"].astype(str).str.len() > 0) & (items["Quantity"] > 0)]

            total_qty = float(items["Quantity"].sum()) if len(items) else 0.0
            freight_per_unit = (float(freight_total) / total_qty) if (transaction_type == "Purchase" and total_qty > 0) else 0.0

            preview_rows = []
            for _, r in items.iterrows():
                ing = str(r["Ingredient"])
                qty = float(r["Quantity"])
                unit_price = float(r["Unit Price"])
                # Unit is sourced from the ingredient registry during entry.
                # Use it directly to avoid schema/name mismatches.
                unit = str(r.get("Unit", "") or "")
                eff_unit_cost = unit_price + freight_per_unit
                preview_rows.append(
                    {
                        "Ingredient": ing,
                        "Quantity": qty,
                        "Unit": unit,
                        "Unit Price": unit_price,
                        "Freight / Unit": freight_per_unit,
                        "Effective Unit Cost": eff_unit_cost,
                        "Line Total": qty * eff_unit_cost,
                    }
                )

            if preview_rows:
                preview_df = pd.DataFrame(preview_rows)

                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    st.metric("Total Units", f"{total_qty:,.3f}")
                with col_p2:
                    st.metric("Freight / Unit", f"${freight_per_unit:,.4f}")
                with col_p3:
                    st.metric("Order Total", f"${preview_df['Line Total'].sum():,.2f}")

                st.dataframe(preview_df, use_container_width=True, height=300)
            else:
                st.info("Add at least one item (Ingredient + Quantity) to preview the allocation.")

            # --- Save ---
            if st.button("💾 Record Purchase Order", type="primary", use_container_width=True, key="po_save_btn"):
                if (not supplier) or (supplier == "Select Supplier"):
                    st.error("Please select a supplier.")
                elif not preview_rows:
                    st.error("Please add at least one valid item.")
                else:
                    po_id = save_purchase_order_fast(
                        transaction_type=transaction_type,
                        supplier=supplier,
                        order_number=order_number,
                        po_date=po_date,
                        freight_total=float(freight_total),
                        notes=notes,
                        recorded_by="User",
                        preview_rows=preview_rows,
                    )

                    # Reset items UI
                    st.session_state.po_items = [
                        {"Ingredient": "", "Quantity": 0.0, "Unit": "", "Unit Price": 0.0}
                    ]
                    # Clear per-row widget state so the UI doesn't keep old values
                    for k in list(st.session_state.keys()):
                        if str(k).startswith("po_item_"):
                            try:
                                del st.session_state[k]
                            except Exception:
                                pass
                    data = get_all_data()
                    st.success(f"✅ Order saved! #{order_number if order_number else po_id}")
                    st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab_history:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("📜 Purchase History")

        # Prefer purchase-by-order tables
        orders_df = data.get("purchase_orders", pd.DataFrame())
        items_df = data.get("purchase_order_items", pd.DataFrame())
        legacy_purchases_df = data.get("purchases", pd.DataFrame())

        if not orders_df.empty and not items_df.empty:
            # Build a line-level dataframe (items + header)
            merged = items_df.merge(
                orders_df,
                left_on="purchase_order_id",
                right_on="id_purchase_order",
                how="left",
                suffixes=("_item", ""),
            )
            merged["date"] = pd.to_datetime(merged["date"], errors="coerce")

            # Filters
            col_hist1, col_hist2, col_hist3 = st.columns(3)
            with col_hist1:
                hist_type = st.selectbox(
                    "Transaction Type",
                    ["All"] + sorted(merged["transaction_type"].dropna().unique().tolist()),
                    key="hist_type_filter",
                )
            with col_hist2:
                hist_ingredient = st.selectbox(
                    "Ingredient",
                    ["All"] + sorted(merged["ingredient"].dropna().unique().tolist()),
                    key="hist_ing_filter_main",
                )
            with col_hist3:
                hist_supplier = st.selectbox(
                    "Supplier",
                    ["All"] + sorted(merged["supplier"].dropna().unique().tolist()),
                    key="hist_supplier_filter",
                )

            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_date = st.date_input(
                    "Start Date",
                    datetime.now().date() - timedelta(days=30),
                    key="hist_start_date",
                )
            with date_col2:
                end_date = st.date_input(
                    "End Date",
                    datetime.now().date(),
                    key="hist_end_date",
                )

            filtered = merged.copy()
            if pd.notna(filtered["date"]).any():
                filtered = filtered[
                    (filtered["date"] >= pd.Timestamp(start_date))
                    & (filtered["date"] <= pd.Timestamp(end_date) + pd.Timedelta(days=1))
                ]
            if hist_type != "All":
                filtered = filtered[filtered["transaction_type"] == hist_type]
            if hist_ingredient != "All":
                filtered = filtered[filtered["ingredient"] == hist_ingredient]
            if hist_supplier != "All":
                filtered = filtered[filtered["supplier"] == hist_supplier]

            # Stats
            st.markdown("---")
            st.subheader("📊 Summary Statistics")
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                total_orders = filtered["purchase_order_id"].nunique()
                st.metric("Total Orders", int(total_orders))
            with col_stat2:
                total_lines = len(filtered)
                st.metric("Total Line Items", int(total_lines))
            with col_stat3:
                total_quantity = float(filtered["quantity"].sum()) if "quantity" in filtered.columns else 0.0
                st.metric("Total Quantity", f"{total_quantity:,.3f}")
            with col_stat4:
                total_cost = float(filtered["total_cost"].sum()) if "total_cost" in filtered.columns else 0.0
                st.metric("Total Cost", f"${total_cost:,.2f}")

            st.markdown("---")
            st.subheader("🧾 Orders")

            # Order summary table
            order_summary = (
                filtered.groupby(["purchase_order_id", "date", "transaction_type", "supplier", "order_number", "freight_total"], dropna=False)
                .agg(
                    items=("ingredient", "count"),
                    order_total=("total_cost", "sum"),
                )
                .reset_index()
            )
            order_summary["date"] = pd.to_datetime(order_summary["date"], errors="coerce").dt.date
            order_summary = order_summary.sort_values(by=["date", "purchase_order_id"], ascending=False)

            st.dataframe(
                order_summary.rename(
                    columns={
                        "purchase_order_id": "Order ID",
                        "date": "Date",
                        "transaction_type": "Type",
                        "supplier": "Supplier",
                        "order_number": "Order #",
                        "freight_total": "Freight Total",
                        "items": "# Items",
                        "order_total": "Order Total",
                    }
                ),
                use_container_width=True,
                height=280,
            )

            # Expanders with items
            st.markdown("---")
            st.subheader("📦 Order Details")
            for _, o in order_summary.head(25).iterrows():
                oid = o["purchase_order_id"]
                title = f"{o.get('Date', o.get('date'))} • {o.get('supplier', '')} • #{o.get('order_number', oid)}"
                with st.expander(title, expanded=False):
                    sub = filtered[filtered["purchase_order_id"] == oid].copy()
                    sub["date"] = pd.to_datetime(sub["date"], errors="coerce").dt.date
                    show_cols = [
                        "ingredient",
                        "quantity",
                        "unit",
                        "unit_price",
                        "freight_per_unit",
                        "effective_unit_cost",
                        "total_cost",
                    ]
                    sub2 = sub[show_cols].copy()
                    st.dataframe(
                        sub2.rename(
                            columns={
                                "ingredient": "Ingredient",
                                "quantity": "Quantity",
                                "unit": "Unit",
                                "unit_price": "Unit Price",
                                "freight_per_unit": "Freight/Unit",
                                "effective_unit_cost": "Effective Unit Cost",
                                "total_cost": "Line Total",
                            }
                        ),
                        use_container_width=True,
                        height=240,
                    )

            # Export line level
            st.markdown("---")
            csv = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export filtered line items (CSV)",
                data=csv,
                file_name="purchase_order_items.csv",
                mime="text/csv",
                key="export_po_items_csv",
            )

        elif not legacy_purchases_df.empty:
            st.info("Showing legacy purchase history (single-item purchases).")
            st.dataframe(legacy_purchases_df, use_container_width=True, height=450)
        else:
            st.info("No purchase history available yet. Record your first purchase in the 'New Purchase' tab.")
        
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_suppliers:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("🏭 Suppliers")
        st.caption("Manage suppliers here. Ingredients are linked to a supplier, and Purchases can only include items from the selected supplier.")

        suppliers_df = data.get("suppliers", pd.DataFrame()).copy()

        # Normalize id column (SQLite/Postgres)
        sup_id_col = None
        for cand in ["id_supplier", "id"]:
            if cand in suppliers_df.columns:
                sup_id_col = cand
                break

        # --- List ---
        if suppliers_df.empty:
            st.info("No suppliers yet. Add your first supplier below.")
        else:
            col_l1, col_l2 = st.columns([2, 1])
            with col_l1:
                q = st.text_input("Search suppliers", key="sup_search", placeholder="Type a name, contact, email...")
            with col_l2:
                status_filter = st.selectbox(
                    "Status",
                    ["All"] + sorted(suppliers_df.get("status", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
                    key="sup_status_filter",
                )

            view_df = suppliers_df.copy()
            if q:
                ql = q.lower()
                hay = (
                    view_df.get("name", "").astype(str)
                    + " " + view_df.get("contact_person", "").astype(str)
                    + " " + view_df.get("email", "").astype(str)
                    + " " + view_df.get("phone", "").astype(str)
                ).str.lower()
                view_df = view_df[hay.str.contains(ql, na=False)]
            if status_filter != "All" and "status" in view_df.columns:
                view_df = view_df[view_df["status"].astype(str) == status_filter]

            show_cols = [c for c in [sup_id_col, "name", "contact_person", "email", "phone", "status", "notes"] if c and c in view_df.columns]
            st.dataframe(
                view_df[show_cols].rename(
                    columns={
                        sup_id_col: "ID" if sup_id_col else "ID",
                        "name": "Supplier",
                        "contact_person": "Contact Person",
                        "email": "Email",
                        "phone": "Phone",
                        "status": "Status",
                        "notes": "Notes",
                    }
                ),
                use_container_width=True,
                height=280,
            )

        st.markdown("---")
        st.subheader("➕ Add supplier")
        with st.form("sup_add_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                sup_name = st.text_input("Supplier Name*", key="sup_add_name")
                contact_person = st.text_input("Contact Person", key="sup_add_contact")
            with c2:
                email = st.text_input("Email", key="sup_add_email")
                phone = st.text_input("Phone", key="sup_add_phone")
            with c3:
                status = st.selectbox("Status", ["Active", "Inactive"], key="sup_add_status")
                website = st.text_input("Website", key="sup_add_website")

            address = st.text_input("Address", key="sup_add_address")
            col_city, col_country = st.columns(2)
            with col_city:
                city = st.text_input("City", key="sup_add_city")
            with col_country:
                country = st.text_input("Country", key="sup_add_country")
            notes = st.text_area("Notes", key="sup_add_notes")
            add_ok = st.form_submit_button("Save supplier", type="primary", use_container_width=True)

        if add_ok:
            if not sup_name:
                st.error("Supplier name is required!")
            else:
                # Basic uniqueness by case-insensitive name
                existing = (
                    suppliers_df.get("name", pd.Series(dtype=str)).dropna().astype(str).str.lower().tolist()
                    if not suppliers_df.empty else []
                )
                if str(sup_name).lower() in existing:
                    st.warning("Supplier already exists. Use the edit section below to update it.")
                else:
                    insert_data(
                        "suppliers",
                        {
                            "name": sup_name,
                            "contact_person": contact_person,
                            "email": email,
                            "phone": phone,
                            "status": status,
                            "website": website,
                            "address": address,
                            "city": city,
                            "country": country,
                            "notes": notes,
                        },
                    )
                    st.success(f"✅ Supplier '{sup_name}' added!")
                    data = get_all_data()
                    st.rerun()

        st.markdown("---")
        st.subheader("✏️ Edit supplier")

        suppliers_df = data.get("suppliers", pd.DataFrame()).copy()
        if suppliers_df.empty:
            st.info("Add a supplier first to edit it.")
        else:
            sup_names = suppliers_df["name"].dropna().astype(str).sort_values().tolist() if "name" in suppliers_df.columns else []
            selected = st.selectbox("Select supplier", ["Select..."] + sup_names, key="sup_edit_select")

            if selected and selected != "Select...":
                row = suppliers_df.loc[suppliers_df["name"].astype(str) == str(selected)].iloc[0]
                supplier_pk = None
                if "id_supplier" in suppliers_df.columns:
                    supplier_pk = row.get("id_supplier")
                elif "id" in suppliers_df.columns:
                    supplier_pk = row.get("id")

                with st.form("sup_edit_form", clear_on_submit=False):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        e_name = st.text_input("Supplier Name*", value=str(row.get("name", "")), key="sup_edit_name")
                        e_contact = st.text_input("Contact Person", value=str(row.get("contact_person", "") or ""), key="sup_edit_contact")
                    with c2:
                        e_email = st.text_input("Email", value=str(row.get("email", "") or ""), key="sup_edit_email")
                        e_phone = st.text_input("Phone", value=str(row.get("phone", "") or ""), key="sup_edit_phone")
                    with c3:
                        e_status = st.selectbox(
                            "Status",
                            ["Active", "Inactive"],
                            index=0 if str(row.get("status", "Active")) == "Active" else 1,
                            key="sup_edit_status",
                        )
                        e_website = st.text_input("Website", value=str(row.get("website", "") or ""), key="sup_edit_website")

                    e_address = st.text_input("Address", value=str(row.get("address", "") or ""), key="sup_edit_address")
                    col_city2, col_country2 = st.columns(2)
                    with col_city2:
                        e_city = st.text_input("City", value=str(row.get("city", "") or ""), key="sup_edit_city")
                    with col_country2:
                        e_country = st.text_input("Country", value=str(row.get("country", "") or ""), key="sup_edit_country")
                    e_notes = st.text_area("Notes", value=str(row.get("notes", "") or ""), key="sup_edit_notes")

                    edit_ok = st.form_submit_button("Update supplier", type="primary", use_container_width=True)

                if edit_ok:
                    if not e_name:
                        st.error("Supplier name is required!")
                    else:
                        where_clause = "id_supplier = :id" if "id_supplier" in suppliers_df.columns else "id = :id"
                        update_data(
                            "suppliers",
                            {
                                "name": e_name,
                                "contact_person": e_contact,
                                "email": e_email,
                                "phone": e_phone,
                                "status": e_status,
                                "website": e_website,
                                "address": e_address,
                                "city": e_city,
                                "country": e_country,
                                "notes": e_notes,
                            },
                            where_clause,
                            {"id": supplier_pk},
                        )
                        st.success("✅ Supplier updated!")
                        data = get_all_data()
                        st.rerun()

                # Delete (outside any form)
                st.markdown("##### 🗑️ Delete")
                st.warning("Deleting a supplier does not automatically update linked ingredients. Consider setting it to Inactive instead.")
                confirm_del = st.checkbox("I understand the consequences", key="sup_delete_confirm")
                if st.button("Delete supplier", key="sup_delete_btn", use_container_width=True, disabled=not confirm_del):
                    where_clause = "id_supplier = :id" if "id_supplier" in suppliers_df.columns else "id = :id"
                    delete_data("suppliers", where_clause, {"id": supplier_pk})
                    st.success("✅ Supplier deleted!")
                    data = get_all_data()
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with tab_reports:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("📊 Purchase Reports & Analytics")

        # Normalize purchases for analytics:
        # Prefer the purchase-by-order model (effective_unit_cost already includes freight/unit).
        orders_df = data.get("purchase_orders", pd.DataFrame())
        items_df = data.get("purchase_order_items", pd.DataFrame())
        if not orders_df.empty and not items_df.empty:
            merged = items_df.merge(
                orders_df,
                left_on="purchase_order_id",
                right_on="id_purchase_order",
                how="left",
                suffixes=("_item", ""),
            )
            purchases_df = pd.DataFrame(
                {
                    "transaction_type": merged.get("transaction_type"),
                    "ingredient": merged.get("ingredient"),
                    "supplier": merged.get("supplier"),
                    "quantity": merged.get("quantity"),
                    "unit": merged.get("unit"),
                    "unit_cost": merged.get("effective_unit_cost"),
                    "total_cost": merged.get("total_cost"),
                    "order_number": merged.get("order_number"),
                    "date": merged.get("date"),
                    "notes": merged.get("notes"),
                }
            )
        else:
            purchases_df = data.get("purchases", pd.DataFrame())
        if not purchases_df.empty:
            # Reports disponíveis
            report_type = st.selectbox(
                "Select Report",
                ["Monthly Spending", "Supplier Performance", "Ingredient Cost Trends", "Low Stock Alerts", "Purchase Forecast"],
                key="report_type"
            )
            
            if report_type == "Monthly Spending":
                st.markdown("### 📈 Monthly Spending Report")
                
                # Agrupar por mês
                purchases_copy = purchases_df.copy()
                purchases_copy["date"] = pd.to_datetime(purchases_copy["date"])
                purchases_copy["month"] = purchases_copy["date"].dt.to_period("M").astype(str)
                
                monthly_spending = purchases_copy.groupby("month").agg({
                    "total_cost": "sum"
                }).reset_index()
                monthly_spending.columns = ["Month", "Total Spending"]
                
                # Add contagem de compras separadamente
                purchase_counts = purchases_copy.groupby("month").size().reset_index()
                purchase_counts.columns = ["Month", "Number of Purchases"]
                
                # Combinar
                monthly_spending = pd.merge(monthly_spending, purchase_counts, on="Month")
                
                # Gráfico
                fig_report = go.Figure()
                fig_report.add_trace(go.Bar(
                    x=monthly_spending["Month"],
                    y=monthly_spending["Total Spending"],
                    name="Spending",
                    marker_color="#4caf50"
                ))
                
                fig_report.add_trace(go.Scatter(
                    x=monthly_spending["Month"],
                    y=monthly_spending["Number of Purchases"],
                    name="Purchase Count",
                    yaxis="y2",
                    line=dict(color="#ff6b6b", width=3)
                ))
                
                fig_report.update_layout(
                    title="Monthly Spending & Purchase Count",
                    xaxis_title="Month",
                    yaxis_title="Spending ($)",
                    yaxis2=dict(
                        title="Purchase Count",
                        overlaying="y",
                        side="right"
                    ),
                    height=500
                )
                
                st.plotly_chart(fig_report, use_container_width=True)
                
                # Tabela de dados
                st.dataframe(
                    monthly_spending.sort_values("Total Spending", ascending=False),
                    use_container_width=True
                )
            
            elif report_type == "Supplier Performance":
                st.markdown("### 🏭 Supplier Performance Report")
                
                # Calcular gastos por fornecedor
                supplier_spending = purchases_df.groupby("supplier").agg({
                    "total_cost": "sum",
                    "quantity": "sum"
                }).reset_index()
                supplier_spending.columns = ["Supplier", "Total Spent", "Total Quantity"]
                
                # Contar transações por fornecedor separadamente
                supplier_counts = purchases_df.groupby("supplier").size().reset_index()
                supplier_counts.columns = ["Supplier", "Purchase Count"]
                
                # Combinar
                supplier_performance = pd.merge(supplier_spending, supplier_counts, on="Supplier")
                supplier_performance = supplier_performance.sort_values("Total Spent", ascending=False)
                
                # Gráfico
                fig_supplier = go.Figure(data=[
                    go.Bar(
                        x=supplier_performance["Supplier"],
                        y=supplier_performance["Total Spent"],
                        text=supplier_performance["Total Spent"].round(2),
                        textposition='auto',
                        marker_color="#2196f3"
                    )
                ])
                
                fig_supplier.update_layout(
                    title="Top Suppliers by Total Spending",
                    xaxis_title="Supplier",
                    yaxis_title="Total Spent ($)",
                    height=500
                )
                
                st.plotly_chart(fig_supplier, use_container_width=True)
                
                # Tabela detalhada
                st.dataframe(
                    supplier_performance,
                    use_container_width=True
                )
            
            elif report_type == "Ingredient Cost Trends":
                st.markdown("### 🌾 Ingredient Cost Trends")
                
                # Top 10 ingredientes por gasto
                ingredient_spending = purchases_df.groupby("ingredient").agg({
                    "total_cost": "sum",
                    "unit_cost": "mean",
                    "quantity": "sum"
                }).reset_index()
                
                ingredient_spending.columns = ["Ingredient", "Total Spent", "Avg Unit Cost", "Total Quantity"]
                ingredient_spending = ingredient_spending.sort_values("Total Spent", ascending=False).head(10)
                
                fig_ing = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Top 10 Ingredients by Total Spending", "Average Unit Cost")
                )
                
                fig_ing.add_trace(
                    go.Bar(
                        x=ingredient_spending["Ingredient"],
                        y=ingredient_spending["Total Spent"],
                        name="Total Spent",
                        marker_color="#ff9800"
                    ),
                    row=1, col=1
                )
                
                fig_ing.add_trace(
                    go.Bar(
                        x=ingredient_spending["Ingredient"],
                        y=ingredient_spending["Avg Unit Cost"],
                        name="Avg Unit Cost",
                        marker_color="#9c27b0"
                    ),
                    row=2, col=1
                )
                
                fig_ing.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig_ing, use_container_width=True)
            
            elif report_type == "Low Stock Alerts":
                st.markdown("### ⚠️ Low Stock Alert Report")
                
                # Combinar dados de ingredientes e compras
                ingredients_df = data.get("ingredients", pd.DataFrame())
                if not ingredients_df.empty:
                    low_stock_items = []
                    
                    for _, ing in ingredients_df.iterrows():
                        if ing["stock"] < ing.get("low_stock_threshold", 10):
                            # Calcular dias desde a última compra
                            last_purchase = None
                            if not purchases_df.empty:
                                ing_purchases = purchases_df[purchases_df["ingredient"] == ing["name"]]
                                if len(ing_purchases) > 0:
                                    last_purchase = ing_purchases["date"].max()
                            
                            low_stock_items.append({
                                "Ingredient": ing["name"],
                                "Current Stock": ing["stock"],
                                "Unit": ing["unit"],
                                "Low Threshold": ing.get("low_stock_threshold", 10),
                                "Status": "Critical" if ing["stock"] == 0 else "Low",
                                "Last Purchase": last_purchase.date() if last_purchase else "Never",
                                "Suggested Order": max(ing.get("low_stock_threshold", 10) * 2 - ing["stock"], 0)
                            })
                    
                    if len(low_stock_items) > 0:
                        low_stock_df = pd.DataFrame(low_stock_items)
                        
                        # Colorir por status
                        def color_status(val):
                            if val == "Critical":
                                return 'background-color: #ffcccc'
                            elif val == "Low":
                                return 'background-color: #fff3cd'
                            return ''
                        
                        styled_df = low_stock_df.style.applymap(color_status, subset=['Status'])
                        
                        st.dataframe(
                            styled_df,
                            use_container_width=True
                        )
                        
                        # Botão para criar pedido de compra
                        if st.button("🛒 Create Purchase Orders for All", use_container_width=True, key="create_all_po"):
                            st.info("This would create purchase orders for all low stock items in a real system.")
                    else:
                        st.success("✅ No low stock items! All ingredients are above their thresholds.")
                else:
                    st.info("No ingredient data available.")
            
            elif report_type == "Purchase Forecast":
                st.markdown("### 🔮 Purchase Forecast")
                
                # Previsão simples baseada em histórico
                st.info("""
                **Purchase Forecast Analysis**  
                Based on your purchase history and current stock levels,  
                this report helps predict when you'll need to reorder ingredients.
                """)
                
                # Análise simples
                ingredients_df = data.get("ingredients", pd.DataFrame())
                if not ingredients_df.empty and not purchases_df.empty:
                    forecast_data = []
                    
                    for _, ing in ingredients_df.iterrows():
                        # Calcular taxa de uso média
                        ing_purchases = purchases_df[purchases_df["ingredient"] == ing["name"]]
                        
                        if len(ing_purchases) > 1:
                            # Calcular taxa de uso (compras por mês)
                            ing_purchases["date"] = pd.to_datetime(ing_purchases["date"])
                            monthly_purchases = ing_purchases.groupby(
                                ing_purchases["date"].dt.to_period("M")
                            )["quantity"].sum().mean()
                            
                            # Dias até esgotar
                            if monthly_purchases > 0:
                                daily_usage = monthly_purchases / 30
                                days_remaining = ing["stock"] / daily_usage if daily_usage > 0 else 999
                                
                                forecast_data.append({
                                    "Ingredient": ing["name"],
                                    "Current Stock": ing["stock"],
                                    "Unit": ing["unit"],
                                    "Avg Monthly Usage": round(monthly_purchases, 2),
                                    "Days Until Empty": round(days_remaining, 1),
                                    "Status": "Critical" if days_remaining < 7 else 
                                            "Warning" if days_remaining < 14 else 
                                            "Good" if days_remaining < 30 else "Excellent"
                                })
                    
                    if len(forecast_data) > 0:
                        forecast_df = pd.DataFrame(forecast_data)
                        st.dataframe(
                            forecast_df.sort_values("Days Until Empty"),
                            use_container_width=True
                        )
                    else:
                        st.info("Insufficient purchase history for forecasting.")
                else:
                    st.info("Need more purchase history to generate forecasts.")
        
        else:
            st.info("""
            ## 📊 Reports & Analytics
            
            Once you start recording purchases, you'll be able to generate various reports:
            
            - **Monthly Spending**: Track your spending over time
            - **Supplier Performance**: Compare suppliers by cost and reliability  
            - **Ingredient Cost Trends**: Monitor price changes for key ingredients
            - **Low Stock Alerts**: Automatic alerts for items needing reorder
            - **Purchase Forecast**: Predict when you'll need to reorder
            
            Start by recording purchases in the 'New Purchase' tab!
            """)
        
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Recipes Page
# -----------------------------
elif page == "Recipes":
    st.title("📋 Recipe Management")


    if is_viewer():
        st.info('👀 View-only mode: read-only. To create/edit recipes, log in as admin.')
        recipes_df = data.get('recipes', pd.DataFrame())
        items_df = data.get('recipe_items', pd.DataFrame())
        st.subheader('📖 Recipes')
        if recipes_df is None or recipes_df.empty:
            st.info('No recipes yet.')
        else:
            st.dataframe(recipes_df, use_container_width=True, height=280)
        st.subheader('🧾 Recipe Items')
        if items_df is None or items_df.empty:
            st.info('No recipe items yet.')
        else:
            st.dataframe(items_df, use_container_width=True, height=320)
        st.stop()
    
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.subheader("🍺 Recipe Datebase")
    
    # Verificar se temos dados de receitas
    recipes_df = data.get("recipes", pd.DataFrame())
    if recipes_df.empty:
        st.info("""
        ## 📋 No Recipes Found
        
        To get started with Recipe Management:
        
        1. **Add recipes manually** using the form below
        2. **Import from Excel** if you have existing recipes
        3. **Clone existing recipes** to create variations
        
        Start by creating your first recipe!
        """)
    
    # Tabs para gerenciamento de receitas
    tab_view, tab_create, tab_import, tab_analyze = st.tabs([
        "📖 View Recipes",
        "➕ Create Recipe",
        "📤 Import Recipes",
        "📊 Analyze Recipes"
    ])
    
    with tab_view:
        st.subheader("📖 Recipe Library")
        
        if not recipes_df.empty:
            # Filtros
            col1, col2, col3 = st.columns(3)
            with col1:
                recipe_search = st.text_input("Search Recipes", key="recipe_search")
            with col2:
                style_filter = st.selectbox(
                    "Filter by Style",
                    ["All Styles"] + sorted(recipes_df['style'].dropna().unique().tolist()),
                    key="style_filter"
                )
            with col3:
                brewery_filter = st.selectbox(
                    "Filter by Brewery",
                    ["All Breweries"] + sorted(recipes_df['brewery_name'].dropna().unique().tolist()) if 'brewery_name' in recipes_df.columns else ["All Breweries"],
                    key="brewery_filter_recipe"
                )
            
            # Aplicar filtros
            filtered_recipes = recipes_df.copy()
            if recipe_search:
                filtered_recipes = filtered_recipes[
                    filtered_recipes['name'].str.contains(recipe_search, case=False, na=False) |
                    filtered_recipes['style'].str.contains(recipe_search, case=False, na=False)
                ]
            if style_filter != "All Styles":
                filtered_recipes = filtered_recipes[filtered_recipes['style'] == style_filter]
            if brewery_filter != "All Breweries" and 'brewery_name' in filtered_recipes.columns:
                filtered_recipes = filtered_recipes[filtered_recipes['brewery_name'] == brewery_filter]
            
            # Mostrar receitas
            if not filtered_recipes.empty:
                for idx, recipe in filtered_recipes.iterrows():
                    with st.expander(f"🍺 {recipe['name']} - {recipe.get('style', 'N/A')}", expanded=False):
                        col_left, col_right = st.columns([2, 1])
                        
                        with col_left:
                            # Informações básicas
                            st.write(f"**Batch Size:** {recipe.get('batch_volume', 'N/A')}L")
                            st.write(f"**Efficiency:** {recipe.get('efficiency', 'N/A')}%")
                            st.write(f"**Target Brewery:** {recipe.get('brewery_name', 'N/A')}")
                            
                            # Estatísticas da cerveja
                            if all(k in recipe for k in ['og', 'fg', 'ibus', 'ebc']):
                                st.write("**Beer Stats:**")
                                col_stats1, col_stats2 = st.columns(2)
                                with col_stats1:
                                    st.write(f"OG: {recipe['og']}°P")
                                    st.write(f"FG: {recipe['fg']}°P")
                                with col_stats2:
                                    st.write(f"IBU: {recipe['ibus']}")
                                    st.write(f"Color: {recipe['ebc']} EBC")
                                
                                # Calcular ABV
                                if recipe['og'] and recipe['fg']:
                                    abv = (recipe['og'] - recipe['fg']) * 0.524
                                    st.write(f"**ABV:** {abv:.1f}%")
                            
                            # Descrição
                            if recipe.get('description'):
                                st.write("**Description:**")
                                st.write(recipe['description'])
                        
                        with col_right:
                            # Ações
                            st.write("**Actions:**")
                            if st.button("📋 Create Batch", key=f"brew_{recipe['id_receipt']}", use_container_width=True):
                                st.session_state['recipe_to_brew'] = recipe['id_receipt']
                                st.rerun()
                            
                            if st.button("📝 Edit", key=f"edit_{recipe['id_receipt']}", use_container_width=True):
                                st.session_state['edit_recipe'] = recipe['id_receipt']
                            
                            if st.button("🗑️ Delete", key=f"delete_{recipe['id_receipt']}", use_container_width=True):
                                st.session_state.delete_confirmation = {"type": "recipe", "id": recipe['id_receipt'], "name": recipe['name']}
                                st.rerun()
                        
                        # Ingredients
                        st.write("**Ingredients:**")
                        recipe_items_df = data.get("recipe_items", pd.DataFrame())
                        if not recipe_items_df.empty:
                            recipe_items = recipe_items_df[recipe_items_df['id_receipt'] == recipe['id_receipt']]
                            if not recipe_items.empty:
                                ingredients_df = data.get("ingredients", pd.DataFrame())
                                for _, item in recipe_items.iterrows():
                                    # Obter nome do ingrediente
                                    ingredient_name = "Unknown"
                                    if not ingredients_df.empty:
                                        ing = ingredients_df[ingredients_df['id'] == item['id_ingredient']]
                                        if not ing.empty:
                                            ingredient_name = ing.iloc[0]['name']
                                    
                                    st.write(f"- {ingredient_name}: {item['quantity']} units")
                            else:
                                st.write("No ingredients defined")
        else:
            st.info("No recipes available. Create your first recipe!")
    
    with tab_create:
        st.subheader("➕ Create New Recipe")
        
        # Formulário em duas colunas
        col_left, col_right = st.columns(2)
        
        with col_left:
            recipe_name = st.text_input("Recipe Name*", key="new_recipe_name")
            recipe_style = st.selectbox(
                "Beer Style*",
                ["American Pale Ale", "IPA", "Stout", "Porter", "Lager", "Pilsner", 
                 "Wheat Beer", "Sour", "Belgian Ale", "Other"],
                key="new_recipe_style"
            )
            
            # Selecionar cervejaria
            breweries_df = data.get("breweries", pd.DataFrame())
            if not breweries_df.empty:
                brewery_options = {row['id_brewery']: row['name'] for _, row in breweries_df.iterrows()}
                selected_brewery = st.selectbox(
                    "Target Brewery*",
                    options=list(brewery_options.keys()),
                    format_func=lambda x: brewery_options[x],
                    key="new_recipe_brewery"
                )
                brewery_name = brewery_options[selected_brewery]
            else:
                selected_brewery = 1
                brewery_name = "Main Brewery"
                st.warning("No breweries found. Using default.")
            
            batch_volume = st.number_input("Batch Volume (L)*", min_value=1.0, value=20.0, step=0.5, key="new_batch_volume")
            efficiency = st.slider("Brewing Efficiency (%)", 50, 100, 75, key="new_efficiency")
        
        with col_right:
            # Parâmetros da cerveja
            st.write("**Beer Parameters:**")
            og = st.number_input("Original Gravity (°P)*", min_value=1.0, max_value=30.0, value=12.0, step=0.1, key="new_og")
            fg = st.number_input("Final Gravity (°P)*", min_value=0.0, max_value=20.0, value=3.0, step=0.1, key="new_fg")
            
            # Calcular ABV automaticamente
            if og and fg:
                abv = (og - fg) * 0.524
                st.info(f"**Estimated ABV:** {abv:.1f}%")
            
            col_params1, col_params2 = st.columns(2)
            with col_params1:
                ibus = st.number_input("IBUs", min_value=0, value=30, step=1, key="new_ibus")
            with col_params2:
                ebc = st.number_input("Color (EBC)", min_value=0, value=20, step=1, key="new_ebc")
        
        # Descrição
        description = st.text_area("Recipe Description", 
                                 placeholder="Describe the beer style, flavor profile, brewing notes...",
                                 height=100,
                                 key="new_description")
        
        # Seção de ingredientes
        st.markdown("---")
        st.subheader("🍻 Recipe Ingredients")
        
        ingredients_df = data.get("ingredients", pd.DataFrame())
        if not ingredients_df.empty:
            # Controle dinâmico de ingredientes
            if 'recipe_ingredient_count' not in st.session_state:
                st.session_state.recipe_ingredient_count = 1
            
            ingredient_list = []
            
            for i in range(st.session_state.recipe_ingredient_count):
                st.write(f"**Ingredient {i+1}**")
                col_ing1, col_ing2, col_ing3, col_ing4 = st.columns([3, 2, 2, 1])
                
                with col_ing1:
                    # Agrupar ingredientes por categoria
                    ingredient_categories = sorted(ingredients_df['category'].dropna().unique())
                    selected_category = st.selectbox(
                        "Category",
                        ["All"] + ingredient_categories,
                        key=f"ing_cat_{i}"
                    )
                    
                    # Filtrar ingredientes por categoria
                    if selected_category == "All":
                        available_ingredients = ingredients_df['name'].tolist()
                    else:
                        available_ingredients = ingredients_df[ingredients_df['category'] == selected_category]['name'].tolist()
                    
                    selected_ingredient = st.selectbox(
                        "Ingredient",
                        [""] + available_ingredients,
                        key=f"ing_{i}"
                    )
                
                with col_ing2:
                    if selected_ingredient:
                        # Obter unidade do ingrediente
                        ing_info = ingredients_df[ingredients_df['name'] == selected_ingredient].iloc[0]
                        unit = ing_info['unit']
                        quantity = st.number_input(
                            f"Amount ({unit})",
                            min_value=0.0,
                            value=0.0,
                            step=0.1,
                            key=f"qty_{i}"
                        )
                    else:
                        quantity = 0.0
                        unit = ""
                
                with col_ing3:
                    if selected_ingredient and quantity > 0:
                        # Mostrar custo estimado
                        unit_cost = ing_info.get('unit_cost', 0)
                        total_cost = unit_cost * quantity
                        st.write(f"**Cost:** ${total_cost:.2f}")
                        
                        # Verificar estoque
                        stock = ing_info.get('stock', 0)
                        if quantity > stock:
                            st.error(f"⚠️ Low stock: {stock} {unit}")
                        else:
                            st.success("✓ In stock")
                
                with col_ing4:
                    # Botão para remover ingrediente
                    if i > 0:  # No remover o primeiro
                        if st.button("🗑️", key=f"remove_{i}"):
                            st.session_state.recipe_ingredient_count -= 1
                            st.rerun()
                
                if selected_ingredient and quantity > 0:
                    ingredient_list.append({
                        'id_ingredient': ing_info['id'],
                        'name': selected_ingredient,
                        'quantity': quantity,
                        'unit': unit,
                        'unit_cost': unit_cost,
                        'total_cost': total_cost
                    })
            
            # Botão para adicionar mais ingredientes
            if st.button("➕ Add Another Ingredient", key="add_another_ing"):
                st.session_state.recipe_ingredient_count += 1
                st.rerun()
            
            # Summary dos ingredientes
            if ingredient_list:
                st.markdown("---")
                st.subheader("📋 Ingredients Summary")
                
                total_cost = sum(item['total_cost'] for item in ingredient_list)
                total_ingredients = len(ingredient_list)
                
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                with col_sum1:
                    st.metric("Total Ingredients", total_ingredients)
                with col_sum2:
                    st.metric("Total Recipe Cost", f"${total_cost:.2f}")
                with col_sum3:
                    cost_per_liter = total_cost / batch_volume if batch_volume > 0 else 0
                    st.metric("Cost per Liter", f"${cost_per_liter:.2f}")
                
                # Tabela de resumo
                summary_df = pd.DataFrame([
                    {
                        'Ingredient': item['name'],
                        'Amount': f"{item['quantity']} {item['unit']}",
                        'Unit Cost': f"${item['unit_cost']:.2f}",
                        'Total Cost': f"${item['total_cost']:.2f}"
                    }
                    for item in ingredient_list
                ])
                st.dataframe(summary_df, use_container_width=True)
        else:
            st.warning("No ingredients available. Please add ingredients first in the Ingredients page.")
            ingredient_list = []
        
        # Botão para criar receita
        if st.button("📋 Create Recipe", type="primary", use_container_width=True, key="create_recipe_final"):
            if not recipe_name:
                st.error("Recipe name is required!")
            elif not ingredient_list:
                st.error("Please add at least one ingredient!")
            elif fg >= og:
                st.error("Final Gravity must be lower than Original Gravity!")
            else:
                # Criar registro da receita
                new_recipe = {
                    'name': recipe_name,
                    'style': recipe_style,
                    'batch_volume': batch_volume,
                    'efficiency': efficiency,
                    'brewery_id': selected_brewery,
                    'brewery_name': brewery_name,
                    'og': og,
                    'fg': fg,
                    'ibus': ibus,
                    'ebc': ebc,
                    'abv': (og - fg) * 0.524,
                    'description': description
                }
                
                # Inserir receita
                recipe_id = insert_data("recipes", new_recipe)
                
                # Add ingredientes à tabela recipe_items
                for item in ingredient_list:
                    new_recipe_item = {
                        'id_receipt': recipe_id,
                        'id_ingredient': item['id_ingredient'],
                        'quantity': item['quantity']
                    }
                    
                    insert_data("recipe_items", new_recipe_item)
                
                # Clear estado
                st.session_state.recipe_ingredient_count = 1
                
                # Atualizar dados
                data = get_all_data()
                st.success(f"✅ Recipe '{recipe_name}' created successfully!")
                st.rerun()
    
    with tab_import:
        st.subheader("📤 Import Recipes")
        
        st.info("""
        **Import recipes from Excel or CSV files**
        
        Supported formats:
        - Excel files (.xlsx, .xls) with recipe sheets
        - CSV files with recipe data
        - BeerXML files (coming soon)
        """)
        
        uploaded_recipe_file = st.file_uploader(
            "Choose recipe file",
            type=['xlsx', 'xls', 'csv'],
            key="recipe_upload"
        )
        
        if uploaded_recipe_file:
            try:
                if uploaded_recipe_file.name.endswith('.csv'):
                    recipe_df = pd.read_csv(uploaded_recipe_file)
                else:
                    # Para Excel, tentar ler a primeira sheet
                    recipe_df = pd.read_excel(uploaded_recipe_file)
                
                st.success(f"Successfully loaded {len(recipe_df)} recipes")
                
                # Pré-visualização
                st.write("**Preview of imported recipes:**")
                st.dataframe(recipe_df.head(), use_container_width=True)
                
                # Mapeamento de colunas
                st.write("**Column Mapping**")
                st.write("Map your file columns to the required recipe fields:")
                
                required_fields = ['name', 'style', 'batch_volume', 'og', 'fg']
                file_columns = recipe_df.columns.tolist()
                
                column_mapping = {}
                for field in required_fields:
                    column_mapping[field] = st.selectbox(
                        f"Select column for '{field}'",
                        [""] + file_columns,
                        key=f"map_{field}"
                    )
                
                # Botão de importação
                if st.button("Import Recipes", type="primary", key="import_recipes_btn"):
                    if all(column_mapping.values()):
                        # Processar importação
                        imported_count = 0
                        
                        for _, row in recipe_df.iterrows():
                            # Criar registro da receita
                            recipe_data = {
                                'name': row[column_mapping['name']],
                                'style': row[column_mapping['style']],
                                'batch_volume': float(row[column_mapping['batch_volume']]),
                                'og': float(row[column_mapping['og']]),
                                'fg': float(row[column_mapping['fg']])
                            }
                            
                            # Add campos opcionais se disponíveis
                            optional_fields = ['ibus', 'ebc', 'description', 'efficiency']
                            for field in optional_fields:
                                if field in file_columns:
                                    recipe_data[field] = row[field]
                            
                            # Inserir receita
                            insert_data("recipes", recipe_data)
                            imported_count += 1
                        
                        data = get_all_data()
                        st.success(f"✅ Successfully imported {imported_count} recipes!")
                        st.rerun()
                    else:
                        st.error("Please map all required fields!")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    with tab_analyze:
        st.subheader("📊 Recipe Analysis")
        
        if not recipes_df.empty:
            # Estatísticas gerais
            col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
            with col_stats1:
                total_recipes = len(recipes_df)
                st.metric("Total Recipes", total_recipes)
            with col_stats2:
                unique_styles = recipes_df['style'].nunique()
                st.metric("Unique Styles", unique_styles)
            with col_stats3:
                avg_batch_size = recipes_df['batch_volume'].mean()
                st.metric("Avg Batch Size", f"{avg_batch_size:.1f}L")
            with col_stats4:
                avg_abv = ((recipes_df['og'] - recipes_df['fg']) * 0.524).mean()
                st.metric("Avg ABV", f"{avg_abv:.1f}%")
            
            # Gráfico de distribuição por estilo
            st.markdown("---")
            st.write("**Distribution by Beer Style**")
            
            style_dist = recipes_df['style'].value_counts()
            fig_style = go.Figure(data=[
                go.Bar(
                    x=style_dist.index,
                    y=style_dist.values,
                    marker_color='#4caf50'
                )
            ])
            fig_style.update_layout(
                title="Recipes by Beer Style",
                xaxis_title="Beer Style",
                yaxis_title="Number of Recipes",
                height=400
            )
            st.plotly_chart(fig_style, use_container_width=True)
            
            # Análise de parâmetros
            st.markdown("---")
            st.write("**Beer Parameters Analysis**")
            
            col_param1, col_param2 = st.columns(2)
            with col_param1:
                # Scatter plot OG vs FG
                fig_og_fg = go.Figure(data=[
                    go.Scatter(
                        x=recipes_df['og'],
                        y=recipes_df['fg'],
                        mode='markers',
                        marker=dict(size=10, color=recipes_df['ibus'], colorscale='Viridis'),
                        text=recipes_df['name'],
                        hovertemplate='<b>%{text}</b><br>OG: %{x}°P<br>FG: %{y}°P<extra></extra>'
                    )
                ])
                fig_og_fg.update_layout(
                    title="OG vs FG Scatter Plot",
                    xaxis_title="Original Gravity (°P)",
                    yaxis_title="Final Gravity (°P)",
                    height=400
                )
                st.plotly_chart(fig_og_fg, use_container_width=True)
            
            with col_param2:
                # Histograma de ABV
                abv_values = (recipes_df['og'] - recipes_df['fg']) * 0.524
                fig_abv = go.Figure(data=[
                    go.Histogram(
                        x=abv_values,
                        nbinsx=20,
                        marker_color='#2196f3'
                    )
                ])
                fig_abv.update_layout(
                    title="ABV Distribution",
                    xaxis_title="ABV (%)",
                    yaxis_title="Number of Recipes",
                    height=400
                )
                st.plotly_chart(fig_abv, use_container_width=True)
            
            # Tabela de custos (se houver ingredientes)
            recipe_items_df = data.get("recipe_items", pd.DataFrame())
            if not recipe_items_df.empty:
                st.markdown("---")
                st.write("**Recipe Cost Analysis**")
                
                # Calcular custos estimados para cada receita
                recipe_costs = []
                for _, recipe in recipes_df.iterrows():
                    recipe_id = recipe['id_receipt']
                    recipe_items = recipe_items_df[recipe_items_df['id_receipt'] == recipe_id]
                    
                    total_cost = 0
                    for _, item in recipe_items.iterrows():
                        ingredients_df = data.get("ingredients", pd.DataFrame())
                        if not ingredients_df.empty:
                            ing = ingredients_df[ingredients_df['id'] == item['id_ingredient']]
                            if not ing.empty:
                                unit_cost = ing.iloc[0].get('unit_cost', 0)
                                total_cost += unit_cost * item['quantity']
                    
                    recipe_costs.append({
                        'Recipe': recipe['name'],
                        'Batch Volume (L)': recipe['batch_volume'],
                        'Total Cost': total_cost,
                        'Cost per Liter': total_cost / recipe['batch_volume'] if recipe['batch_volume'] > 0 else 0
                    })
                
                if recipe_costs:
                    cost_df = pd.DataFrame(recipe_costs)
                    st.dataframe(cost_df.sort_values('Cost per Liter', ascending=False), use_container_width=True)
        else:
            st.info("No recipes available for analysis. Create some recipes first!")
    
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Production Orders Page
# -----------------------------
elif page == "Production Orders":
    st.title("🏭 Production Management")


    if is_viewer():
        st.info('👀 View-only mode: read-only. To create/edit production orders, log in as admin.')
        orders_df = data.get('production_orders', pd.DataFrame())
        st.subheader('🏭 Production Orders')
        if orders_df is None or orders_df.empty:
            st.info('No production orders yet.')
        else:
            st.dataframe(orders_df, use_container_width=True, height=520)
        st.stop()
    
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.subheader("⚙️ Production Planning & Tracking")
    
    # Tabs para gerenciamento de produção
    tab_plan, tab_active, tab_history, tab_schedule = st.tabs([
        "📋 Plan Production",
        "🏭 Active Batches",
        "📜 Production History",
        "📅 Production Schedule"
    ])
    
    with tab_plan:
        st.subheader("📋 Plan New Production Batch")
        
        # Verificar requisitos
        recipes_df = data.get("recipes", pd.DataFrame())
        if recipes_df.empty:
            st.error("No recipes available. Please create recipes first!")
            st.stop()
        
        breweries_df = data.get("breweries", pd.DataFrame())
        if breweries_df.empty:
            st.error("No breweries available. Please add breweries first!")
            st.stop()
        
        # Formulário de planejamento
        col_plan1, col_plan2 = st.columns(2)
        
        with col_plan1:
            # Selecionar receita
            recipe_options = {row['id_receipt']: f"{row['name']} ({row.get('style', 'N/A')})" 
                            for _, row in recipes_df.iterrows()}
            selected_recipe_id = st.selectbox(
                "Select Recipe*",
                options=list(recipe_options.keys()),
                format_func=lambda x: recipe_options[x],
                key="plan_recipe"
            )
            
            # Obter detalhes da receita
            if selected_recipe_id:
                recipe_data = recipes_df[recipes_df['id_receipt'] == selected_recipe_id].iloc[0]
                
                st.info(f"""
                **Selected Recipe: {recipe_data['name']}**
                - Style: {recipe_data.get('style', 'N/A')}
                - Original Batch: {recipe_data['batch_volume']}L
                - OG: {recipe_data.get('og', 'N/A')}°P
                - FG: {recipe_data.get('fg', 'N/A')}°P
                - Est. ABV: {(recipe_data.get('og', 0) - recipe_data.get('fg', 0)) * 0.524:.1f}%
                """)
        
        with col_plan2:
            # Configuração do lote
            batch_volume = st.number_input(
                "Batch Volume (L)*",
                min_value=1.0,
                value=float(recipe_data['batch_volume']) if selected_recipe_id else 20.0,
                step=0.5,
                key="plan_volume"
            )
            
            # Selecionar cervejaria
            brewery_options = {row['id_brewery']: row['name'] for _, row in breweries_df.iterrows()}
            selected_brewery_id = st.selectbox(
                "Brewery*",
                options=list(brewery_options.keys()),
                format_func=lambda x: brewery_options[x],
                key="plan_brewery"
            )
            
            # Prioridade
            priority = st.select_slider(
                "Priority",
                options=["Low", "Medium", "High", "Critical"],
                value="Medium",
                key="plan_priority"
            )
        
        # Date de início planejada
        st.write("**Planned Start Date**")
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            planned_date = st.date_input(
                "Date",
                datetime.now().date() + timedelta(days=1),
                key="plan_date"
            )
        with col_date2:
            planned_time = st.time_input(
                "Time",
                datetime.now().time().replace(hour=8, minute=0),
                key="plan_time"
            )
        
        planned_datetime = datetime.combine(planned_date, planned_time)
        
        # Verificação de ingredientes
        st.markdown("---")
        st.subheader("🧪 Ingredient Availability Check")
        
        if selected_recipe_id:
            recipe_items_df = data.get("recipe_items", pd.DataFrame())
            if not recipe_items_df.empty:
                recipe_items = recipe_items_df[recipe_items_df['id_receipt'] == selected_recipe_id]
                
                if not recipe_items.empty:
                    # Calcular fator de escala
                    scale_factor = batch_volume / recipe_data['batch_volume']
                    
                    all_available = True
                    missing_ingredients = []
                    
                    for _, item in recipe_items.iterrows():
                        required_qty = item['quantity'] * scale_factor
                        
                        # Encontrar ingrediente
                        ingredients_df = data.get("ingredients", pd.DataFrame())
                        if not ingredients_df.empty:
                            ing = ingredients_df[ingredients_df['id'] == item['id_ingredient']]
                            if not ing.empty:
                                ing_data = ing.iloc[0]
                                available_qty = ing_data.get('stock', 0)
                                unit = ing_data.get('unit', 'units')
                                
                                col_check1, col_check2, col_check3, col_check4 = st.columns([3, 2, 2, 1])
                                with col_check1:
                                    st.write(f"**{ing_data['name']}**")
                                with col_check2:
                                    st.write(f"Required: {required_qty:.2f} {unit}")
                                with col_check3:
                                    st.write(f"Available: {available_qty:.2f} {unit}")
                                with col_check4:
                                    if available_qty >= required_qty:
                                        st.success("✓")
                                    else:
                                        st.error("✗")
                                        all_available = False
                                        missing_ingredients.append({
                                            'name': ing_data['name'],
                                            'required': required_qty,
                                            'available': available_qty,
                                            'unit': unit,
                                            'shortage': required_qty - available_qty
                                        })
                    
                    if not all_available:
                        st.warning("⚠️ Insufficient ingredients for this batch!")
                        
                        with st.expander("🛒 Purchase Recommendations"):
                            st.write("Create purchase order for missing ingredients:")
                            for item in missing_ingredients:
                                st.write(f"- **{item['name']}**: Need {item['shortage']:.2f} {item['unit']}")
                    else:
                        st.success("✅ All ingredients are available!")
                else:
                    st.info("No ingredients defined for this recipe.")
            else:
                st.warning("Recipe ingredients not available.")
        
        # Notas e instruções
        st.markdown("---")
        production_notes = st.text_area(
            "Production Notes",
            placeholder="Special instructions, modifications, observations...",
            height=100,
            key="plan_notes"
        )
        
        # Botão para criar ordem de produção
        if st.button("🏭 Create Production Order", type="primary", use_container_width=True, key="create_production_order"):
            if not selected_recipe_id or not selected_brewery_id or batch_volume <= 0:
                st.error("Please fill all required fields!")
            else:
                # Obter nomes para display
                recipe_name = recipe_data['name']
                brewery_name = brewery_options[selected_brewery_id]
                
                # Criar ordem de produção
                new_order = {
                    'id_receipt': selected_recipe_id,
                    'recipe_name': recipe_name,
                    'brewery_id': selected_brewery_id,
                    'brewery_name': brewery_name,
                    'volume': batch_volume,
                    'batch_size': recipe_data['batch_volume'],
                    'status': 'Planned',
                    'priority': priority,
                    'scheduled_date': planned_datetime,
                    'current_step': 'Planning',
                    'notes': production_notes,
                    'created_by': 'User'
                }
                
                # Add à tabela
                order_id = insert_data("production_orders", new_order)
                
                # Atualizar dados
                data = get_all_data()
                st.success(f"✅ Production Order #{order_id} created successfully!")
                st.info(f"📅 **Planned for:** {planned_datetime.strftime('%Y-%m-%d %H:%M')}")
                st.rerun()
    
    with tab_active:
        st.subheader("🏭 Active Production Batches")
        
        orders_df = data.get("production_orders", pd.DataFrame())
        if not orders_df.empty:
            # Filtrar ordens ativas
            active_statuses = ['Brewing', 'Fermenting', 'Conditioning', 'Packaging', 'Transferring']
            active_orders = orders_df[orders_df["status"].isin(active_statuses)]
            
            if not active_orders.empty:
                # Filtros
                col_filter1, col_filter2 = st.columns(2)
                with col_filter1:
                    status_filter = st.multiselect(
                        "Filter by Status",
                        active_statuses,
                        default=active_statuses,
                        key="active_status_filter"
                    )
                with col_filter2:
                    breweries_df = data.get("breweries", pd.DataFrame())
                    if not breweries_df.empty:
                        brewery_filter = st.multiselect(
                            "Filter by Brewery",
                            breweries_df["name"].unique(),
                            key="active_brewery_filter"
                        )
                    else:
                        brewery_filter = []
                
                # Aplicar filtros
                filtered_orders = active_orders.copy()
                if status_filter:
                    filtered_orders = filtered_orders[filtered_orders["status"].isin(status_filter)]
                if brewery_filter:
                    filtered_orders = filtered_orders[filtered_orders["brewery_name"].isin(brewery_filter)]
                
                # Estatísticas
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Active Batches", len(filtered_orders))
                with col_stat2:
                    total_volume = filtered_orders['volume'].sum()
                    st.metric("Total Volume", f"{total_volume:.0f}L")
                with col_stat3:
                    avg_days = 0
                    if 'start_date' in filtered_orders.columns:
                        filtered_orders['start_date'] = pd.to_datetime(filtered_orders['start_date'])
                        avg_days = (datetime.now() - filtered_orders['start_date'].min()).days if not filtered_orders.empty else 0
                    st.metric("Avg. Days Active", avg_days)
                
                # Lista de ordens ativas
                st.markdown("---")
                for _, order in filtered_orders.iterrows():
                    # Determinar cor baseada no status
                    status_colors = {
                        'Brewing': '#4caf50',
                        'Fermenting': '#2196f3',
                        'Conditioning': '#9c27b0',
                        'Packaging': '#ff9800',
                        'Transferring': '#ff5722'
                    }
                    
                    with st.expander(f"🏭 Batch #{order['id_order']} - {order['recipe_name']} ({order['status']})", expanded=False):
                        col_info1, col_info2 = st.columns(2)
                        
                        with col_info1:
                            st.write("**Batch Details**")
                            st.write(f"**Recipe:** {order['recipe_name']}")
                            st.write(f"**Volume:** {order['volume']}L")
                            st.write(f"**Brewery:** {order.get('brewery_name', 'N/A')}")
                            st.write(f"**Priority:** {order.get('priority', 'Medium')}")
                            
                            if pd.notna(order.get('start_date')):
                                start_date = pd.to_datetime(order['start_date']).date()
                                days_active = (datetime.now().date() - start_date).days
                                st.write(f"**Started:** {start_date} ({days_active} days ago)")
                        
                        with col_info2:
                            st.write("**Current Status**")
                            st.markdown(f'<span style="background-color: {status_colors.get(order["status"], "#757575")}; '
                                      f'color: white; padding: 5px 10px; border-radius: 5px;">{order["status"]}</span>', 
                                      unsafe_allow_html=True)
                            
                            st.write(f"**Current Step:** {order.get('current_step', 'N/A')}")
                            
                            # Barra de progresso baseada no status
                            progress_map = {
                                'Brewing': 0.3,
                                'Fermenting': 0.6,
                                'Conditioning': 0.8,
                                'Packaging': 0.9,
                                'Transferring': 0.95
                            }
                            progress = progress_map.get(order['status'], 0.1)
                            st.progress(progress)
                            
                            # Botões de ação
                            col_action1, col_action2, col_action3 = st.columns(3)
                            with col_action1:
                                if order['status'] == 'Brewing':
                                    if st.button("→ Fermentation", key=f"to_ferment_{order['id_order']}", use_container_width=True):
                                        # Atualizar status
                                        updates = {
                                            'status': 'Fermenting',
                                            'current_step': 'Primary Fermentation'
                                        }
                                        update_data("production_orders", updates, "id_order = :id_order", {"id_order": order['id_order']})
                                        data = get_all_data()
                                        st.success("Batch moved to fermentation!")
                                        st.rerun()
                            with col_action2:
                                if st.button("📝 Update", key=f"update_{order['id_order']}", use_container_width=True):
                                    st.session_state['update_batch'] = order['id_order']
                            with col_action3:
                                if st.button("⏸️ Pause", key=f"pause_{order['id_order']}", use_container_width=True):
                                    updates = {'status': 'On Hold'}
                                    update_data("production_orders", updates, "id_order = :id_order", {"id_order": order['id_order']})
                                    data = get_all_data()
                                    st.warning("Batch paused!")
                                    st.rerun()
                        
                        # Notas
                        if order.get('notes'):
                            st.write("**Notes:**")
                            st.write(order['notes'])
            else:
                st.info("No active production batches. Plan a new batch in the 'Plan Production' tab.")
        else:
            st.info("No production orders created yet.")
    
    with tab_history:
        st.subheader("📜 Production History")
        
        orders_df = data.get("production_orders", pd.DataFrame())
        if not orders_df.empty:
            # Filtrar ordens concluídas
            completed_orders = orders_df[orders_df["status"] == 'Completed']
            
            if not completed_orders.empty:
                # Filtros de data
                col_date1, col_date2 = st.columns(2)
                with col_date1:
                    start_date = st.date_input(
                        "From Date",
                        datetime.now().date() - timedelta(days=30),
                        key="history_start"
                    )
                with col_date2:
                    end_date = st.date_input(
                        "To Date",
                        datetime.now().date(),
                        key="history_end"
                    )
                
                # Filtrar por data
                if 'end_date' in completed_orders.columns:
                    completed_orders['end_date'] = pd.to_datetime(completed_orders['end_date'])
                    filtered_history = completed_orders[
                        (completed_orders['end_date'] >= pd.Timestamp(start_date)) &
                        (completed_orders['end_date'] <= pd.Timestamp(end_date))
                    ]
                else:
                    filtered_history = completed_orders
                
                # Estatísticas
                col_hist1, col_hist2, col_hist3 = st.columns(3)
                with col_hist1:
                    st.metric("Total Batches", len(filtered_history))
                with col_hist2:
                    total_volume = filtered_history['volume'].sum()
                    st.metric("Total Volume", f"{total_volume:.0f}L")
                with col_hist3:
                    avg_duration = 0
                    if 'start_date' in filtered_history.columns and 'end_date' in filtered_history.columns:
                        filtered_history['start_date'] = pd.to_datetime(filtered_history['start_date'])
                        filtered_history['end_date'] = pd.to_datetime(filtered_history['end_date'])
                        durations = (filtered_history['end_date'] - filtered_history['start_date']).dt.days
                        avg_duration = durations.mean() if not durations.empty else 0
                    st.metric("Avg. Duration", f"{avg_duration:.1f} days")
                
                # Tabela de histórico
                st.markdown("---")
                st.write("**Production History**")
                
                display_cols = ['id_order', 'recipe_name', 'brewery_name', 'volume', 'start_date', 'end_date', 'priority']
                display_df = filtered_history[display_cols].copy()
                
                # Formatar datas
                if 'start_date' in display_df.columns:
                    display_df['start_date'] = pd.to_datetime(display_df['start_date']).dt.date
                if 'end_date' in display_df.columns:
                    display_df['end_date'] = pd.to_datetime(display_df['end_date']).dt.date
                
                display_df.columns = ['Order #', 'Recipe', 'Brewery', 'Volume (L)', 'Start Date', 'End Date', 'Priority']
                st.dataframe(display_df, use_container_width=True)
                
                # Gráfico de produção mensal
                st.markdown("---")
                st.write("**Monthly Production Volume**")
                
                # Agrupar por mês
                if 'end_date' in filtered_history.columns:
                    filtered_history['month'] = filtered_history['end_date'].dt.to_period('M').astype(str)
                    monthly_production = filtered_history.groupby('month')['volume'].sum().reset_index()
                    
                    fig_monthly = go.Figure(data=[
                        go.Bar(
                            x=monthly_production['month'],
                            y=monthly_production['volume'],
                            marker_color='#4caf50'
                        )
                    ])
                    fig_monthly.update_layout(
                        title="Monthly Production Volume",
                        xaxis_title="Month",
                        yaxis_title="Volume (L)",
                        height=400
                    )
                    st.plotly_chart(fig_monthly, use_container_width=True)
            else:
                st.info("No completed production batches in history.")
        else:
            st.info("No production history available.")
    
    with tab_schedule:
        st.subheader("📅 Production Schedule")
        
        # Calendar de produção
        today = datetime.now()
        col_sched1, col_sched2 = st.columns(2)
        with col_sched1:
            schedule_month = st.selectbox(
                "Month",
                range(1, 13),
                index=today.month - 1,
                format_func=lambda x: calendar.month_name[x],
                key="sched_month"
            )
        with col_sched2:
            schedule_year = st.selectbox(
                "Year",
                range(today.year - 1, today.year + 2),
                index=1,
                key="sched_year"
            )
        
        # Obter ordens agendadas para o mês
        orders_df = data.get("production_orders", pd.DataFrame())
        if not orders_df.empty:
            scheduled_orders = orders_df[
                (orders_df['status'].isin(['Planned', 'Scheduled'])) &
                (orders_df['scheduled_date'].notna())
            ].copy()
            
            if not scheduled_orders.empty:
                scheduled_orders['scheduled_date'] = pd.to_datetime(scheduled_orders['scheduled_date'])
                
                # Filtrar para o mês selecionado
                month_scheduled = scheduled_orders[
                    (scheduled_orders['scheduled_date'].dt.month == schedule_month) &
                    (scheduled_orders['scheduled_date'].dt.year == schedule_year)
                ]
                
                if not month_scheduled.empty:
                    # Criar calendário
                    cal = calendar.monthcalendar(schedule_year, schedule_month)
                    
                    # Cabeçalho
                    days_header = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                    cols = st.columns(7)
                    for i, day in enumerate(days_header):
                        cols[i].write(f"**{day}**")
                    
                    # Dias
                    for week in cal:
                        cols = st.columns(7)
                        for i, day in enumerate(week):
                            if day == 0:
                                cols[i].write("")
                            else:
                                current_date = date(schedule_year, schedule_month, day)
                                is_today = (current_date == today.date())
                                
                                day_class = "calendar-day today" if is_today else "calendar-day"
                                
                                with cols[i].container():
                                    st.markdown(f'<div class="{day_class}">', unsafe_allow_html=True)
                                    st.write(f"**{day}**")
                                    
                                    # Ordens para este dia
                                    day_orders = month_scheduled[
                                        month_scheduled['scheduled_date'].dt.date == current_date
                                    ]
                                    
                                    if not day_orders.empty:
                                        for _, order in day_orders.iterrows():
                                            order_time = order['scheduled_date'].strftime('%H:%M')
                                            st.markdown(
                                                f'<div class="calendar-event" style="background-color: #9c27b0;">'
                                                f'#{order["id_order"]} - {order["recipe_name"][:10]}... ({order_time})'
                                                f'</div>',
                                                unsafe_allow_html=True
                                            )
                                    
                                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Lista detalhada de ordens agendadas
                    st.markdown("---")
                    st.write("**Scheduled Production Orders**")
                    
                    for _, order in month_scheduled.sort_values('scheduled_date').iterrows():
                        col_list1, col_list2, col_list3 = st.columns([3, 2, 1])
                        
                        with col_list1:
                            st.write(f"**Order #{order['id_order']}** - {order['recipe_name']}")
                            st.write(f"Volume: {order['volume']}L | Brewery: {order.get('brewery_name', 'N/A')}")
                        
                        with col_list2:
                            scheduled_time = order['scheduled_date'].strftime('%Y-%m-%d %H:%M')
                            days_until = (order['scheduled_date'].date() - today.date()).days
                            
                            if days_until == 0:
                                st.write("**Today**")
                            elif days_until == 1:
                                st.write("**Tomorrow**")
                            else:
                                st.write(f"In **{days_until} days**")
                            
                            st.write(f"_{scheduled_time}_")
                        
                        with col_list3:
                            if st.button("▶️ Start", key=f"start_sched_{order['id_order']}", use_container_width=True):
                                # Iniciar produção
                                updates = {
                                    'status': 'Brewing',
                                    'start_date': datetime.now(),
                                    'current_step': 'Mashing'
                                }
                                update_data("production_orders", updates, "id_order = :id_order", {"id_order": order['id_order']})
                                
                                # Deduzir ingredientes do estoque
                                recipe_items_df = data.get("recipe_items", pd.DataFrame())
                                if not recipe_items_df.empty:
                                    recipe_items = recipe_items_df[recipe_items_df['id_receipt'] == order['id_receipt']]
                                    for _, item in recipe_items.iterrows():
                                        ingredients_df = data.get("ingredients", pd.DataFrame())
                                        if not ingredients_df.empty:
                                            ing = ingredients_df[ingredients_df['id'] == item['id_ingredient']]
                                            if not ing.empty:
                                                ing_data = ing.iloc[0]
                                                scaled_qty = item['quantity'] * (order['volume'] / order['batch_size'])
                                                update_stock_from_usage(ing_data['name'], scaled_qty)
                                
                                data = get_all_data()
                                st.success(f"Production Order #{order['id_order']} started!")
                                st.rerun()
                else:
                    st.info(f"No production scheduled for {calendar.month_name[schedule_month]} {schedule_year}.")
            else:
                st.info("No scheduled production orders. Schedule orders in the 'Plan Production' tab.")
        else:
            st.info("No production orders available.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Calendar Page
# -----------------------------
elif page == "Calendar":
    st.title("📅 Production Calendar")


    if is_viewer():
        st.info('👀 View-only mode: read-only. To add/edit events, log in as admin.')
        events_df = data.get('calendar_events', pd.DataFrame())
        st.subheader('📅 Calendar Events')
        if events_df is None or events_df.empty:
            st.info('No calendar events yet.')
        else:
            st.dataframe(events_df, use_container_width=True, height=520)
        st.stop()
    
    tab_calendar, tab_events, tab_tasks = st.tabs([
        "📅 Calendar View",
        "📋 Events List",
        "✅ Task Manager"
    ])
    
    with tab_calendar:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("📅 Interactive Calendar")
        
        # Seletor de mês/ano
        today = datetime.now()
        col_cal1, col_cal2 = st.columns(2)
        with col_cal1:
            cal_month = st.selectbox("Month", range(1, 13), index=today.month-1, 
                                   format_func=lambda x: calendar.month_name[x], key="cal_month")
        with col_cal2:
            cal_year = st.selectbox("Year", range(today.year-1, today.year+2), index=1, key="cal_year")
        
        # Criar calendário
        cal = calendar.monthcalendar(cal_year, cal_month)
        
        # Cabeçalho
        days_header = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        cols = st.columns(7)
        for i, day in enumerate(days_header):
            cols[i].write(f"**{day}**")
        
        # Dias com eventos
        for week in cal:
            cols = st.columns(7)
            for i, day in enumerate(week):
                if day == 0:
                    cols[i].write("")
                else:
                    current_date = date(cal_year, cal_month, day)
                    is_today = (current_date == today.date())
                    
                    day_class = "calendar-day today" if is_today else "calendar-day"
                    
                    with cols[i].container():
                        st.markdown(f'<div class="{day_class}">', unsafe_allow_html=True)
                        st.write(f"**{day}**")
                        
                        # Events do calendário
                        calendar_events = pd.DataFrame()
                        events_df = data.get("calendar_events", pd.DataFrame())
                        if not events_df.empty:
                            events_df = events_df.copy()
                            if "start_date" in events_df.columns:
                                events_df["start_date"] = pd.to_datetime(events_df["start_date"]).dt.date
                                calendar_events = events_df[events_df["start_date"] == current_date]
                        
                        # Ordens de produção
                        production_events = pd.DataFrame()
                        orders_df = data.get("production_orders", pd.DataFrame())
                        if not orders_df.empty:
                            orders_df = orders_df.copy()
                            if "start_date" in orders_df.columns:
                                orders_df["start_date"] = pd.to_datetime(orders_df["start_date"]).dt.date
                                production_events = orders_df[orders_df["start_date"] == current_date]
                        
                        # Mostrar eventos
                        all_events = list(calendar_events.iterrows()) + list(production_events.iterrows())
                        
                        for _, event in all_events:
                            if "title" in event:  # Evento do calendário
                                event_color = {
                                    "Brewing": "#4caf50",
                                    "Transfer": "#2196f3",
                                    "Packaging": "#ff9800",
                                    "Cleaning": "#f59e0b",
                                    "Maintenance": "#ef4444",
                                    "Meeting": "#9c27b0",
                                    "Other": "#757575"
                                }.get(event.get("event_type", "Other"), "#757575")
                                
                                event_text = event["title"][:20] + "..." if len(event["title"]) > 20 else event["title"]
                                st.markdown(f'<div class="calendar-event" style="background-color: {event_color};">{event_text}</div>', unsafe_allow_html=True)
                            
                            elif "id_order" in event:  # Ordem de produção
                                st.markdown(f'<div class="calendar-event" style="background-color: #9c27b0;">Order #{event["id_order"]}</div>', unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
        
        # Legenda
        st.markdown("---")
        st.subheader("🎨 Legend")
        
        col_leg1, col_leg2, col_leg3 = st.columns(3)
        with col_leg1:
            st.markdown('<div class="calendar-event" style="background-color: #4caf50;">Brewing</div>', unsafe_allow_html=True)
            st.markdown('<div class="calendar-event" style="background-color: #2196f3;">Transfer</div>', unsafe_allow_html=True)
        with col_leg2:
            st.markdown('<div class="calendar-event" style="background-color: #ff9800;">Packaging</div>', unsafe_allow_html=True)
            st.markdown('<div class="calendar-event" style="background-color: #f59e0b;">Cleaning</div>', unsafe_allow_html=True)
        with col_leg3:
            st.markdown('<div class="calendar-event" style="background-color: #ef4444;">Maintenance</div>', unsafe_allow_html=True)
            st.markdown('<div class="calendar-event" style="background-color: #9c27b0;">Production Order</div>', unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab_events:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("📋 Events Management")
        
        # Add novo evento
        with st.expander("➕ Add New Event", expanded=False):
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                event_title = st.text_input("Event Title", key="new_event_title")
                event_type = st.selectbox("Event Type", ["Brewing", "Transfer", "Packaging", "Cleaning", "Maintenance", "Meeting", "Other"], key="new_event_type")
                event_date = st.date_input("Event Date", datetime.now().date(), key="new_event_date")
            
            with col_e2:
                equipment_df = data.get("equipment", pd.DataFrame())
                if not equipment_df.empty:
                    equipment_options = equipment_df["name"].tolist()
                    equipment = st.multiselect("Equipment", equipment_options, key="new_event_equipment")
                else:
                    equipment = st.text_input("Equipment", key="new_event_eq_text")
                
                orders_df = data.get("production_orders", pd.DataFrame())
                if not orders_df.empty:
                    order_options = orders_df["id_order"].tolist()
                    batch_id = st.selectbox("Related Production Order", ["None"] + [f"Order #{oid}" for oid in order_options], key="new_event_batch")
                else:
                    batch_id = st.text_input("Batch ID", key="new_event_batch_text")
            
            event_notes = st.text_area("Notes", key="new_event_notes")
            
            if st.button("Add Event", key="add_event_btn"):
                if event_title:
                    new_event = {
                        "title": event_title,
                        "event_type": event_type,
                        "start_date": event_date,
                        "end_date": event_date,
                        "equipment": ", ".join(equipment) if isinstance(equipment, list) else equipment,
                        "batch_id": batch_id if batch_id != "None" else "",
                        "notes": event_notes,
                        "created_by": "User"
                    }
                    
                    insert_data("calendar_events", new_event)
                    data = get_all_data()
                    st.success("✅ Event added successfully!")
                    st.rerun()
                else:
                    st.error("Event title is required!")
        
        # Lista de eventos
        st.markdown("---")
        st.subheader("Upcoming Events")
        
        events_df = data.get("calendar_events", pd.DataFrame())
        if not events_df.empty:
            events_df = events_df.copy()
            events_df["start_date"] = pd.to_datetime(events_df["start_date"])
            
            # Filtrar eventos futuros
            upcoming_events = events_df[events_df["start_date"] >= pd.Timestamp(datetime.now().date())].sort_values("start_date")
            
            if len(upcoming_events) > 0:
                for _, event in upcoming_events.iterrows():
                    col_ev1, col_ev2, col_ev3 = st.columns([3, 2, 1])
                    
                    with col_ev1:
                        st.write(f"**{event['title']}**")
                        if event["notes"]:
                            st.caption(event["notes"][:100] + "..." if len(event["notes"]) > 100 else event["notes"])
                        
                        if event["equipment"]:
                            st.caption(f"Equipment: {event['equipment']}")
                    
                    with col_ev2:
                        event_date = event["start_date"].date()
                        days_until = (event_date - datetime.now().date()).days
                        
                        if days_until == 0:
                            st.write("**Today**")
                        elif days_until == 1:
                            st.write("**Tomorrow**")
                        else:
                            st.write(f"In **{days_until} days**")
                        
                        st.write(f"_{event['event_type']}_")
                    
                    with col_ev3:
                        col_btn1, col_btn2 = st.columns(2)
                        with col_btn1:
                            if st.button("✏️", key=f"edit_ev_{event['id_event']}"):
                                st.session_state['edit_event'] = event['id_event']
                        with col_btn2:
                            if st.button("🗑️", key=f"delete_ev_{event['id_event']}"):
                                delete_data("calendar_events", "id_event = :id_event", {"id_event": event['id_event']})
                                data = get_all_data()
                                st.success("Event deleted!")
                                st.rerun()
                    
                    st.markdown("---")
            else:
                st.info("No upcoming events scheduled.")
        else:
            st.info("No events in calendar. Add your first event above!")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab_tasks:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.subheader("✅ Task Manager")
        
        # Gerenciamento de tarefas
        st.info("""
        **Task Management Features:**
        - Track cleaning schedules
        - Maintenance reminders
        - Quality control checks
        - Team assignments
        """)
        
        # Tarefas de limpeza
        st.markdown("---")
        st.subheader("🧹 Cleaning Tasks")
        
        equipment_df = data.get("equipment", pd.DataFrame())
        if not equipment_df.empty:
            # Filtros para tarefas de limpeza
            col_clean1, col_clean2 = st.columns(2)
            with col_clean1:
                cleaning_status = st.selectbox(
                    "Cleaning Status",
                    ["All", "Overdue", "Due Today", "Upcoming", "Completed"],
                    key="cleaning_status"
                )
            with col_clean2:
                breweries_df = data.get("breweries", pd.DataFrame())
                if not breweries_df.empty:
                    cleaning_brewery = st.selectbox(
                        "Filter by Brewery",
                        ["All"] + breweries_df["name"].tolist(),
                        key="cleaning_brewery"
                    )
                else:
                    cleaning_brewery = "All"
            
            # Calcular tarefas de limpeza
            equipment_copy = equipment_df.copy()
            if "cleaning_due" in equipment_copy.columns:
                equipment_copy["cleaning_due"] = pd.to_datetime(equipment_copy["cleaning_due"])
                
                # Aplicar filtros
                cleaning_tasks = equipment_copy.copy()
                
                if cleaning_brewery != "All":
                    brewery_id = breweries_df[breweries_df["name"] == cleaning_brewery]["id_brewery"].iloc[0]
                    cleaning_tasks = cleaning_tasks[cleaning_tasks["brewery_id"] == brewery_id]
                
                # Classificar por status
                today = datetime.now().date()
                overdue_tasks = cleaning_tasks[cleaning_tasks["cleaning_due"].dt.date < today]
                due_today = cleaning_tasks[cleaning_tasks["cleaning_due"].dt.date == today]
                upcoming_tasks = cleaning_tasks[cleaning_tasks["cleaning_due"].dt.date > today]
                
                # Selecionar tarefas baseadas no filtro
                if cleaning_status == "Overdue":
                    display_tasks = overdue_tasks
                elif cleaning_status == "Due Today":
                    display_tasks = due_today
                elif cleaning_status == "Upcoming":
                    display_tasks = upcoming_tasks.head(10)  # Limitar a 10 próximas
                elif cleaning_status == "Completed":
                    # Para simplificar, assumimos que equipamentos sem data de limpeza estão "limpos"
                    display_tasks = cleaning_tasks[cleaning_tasks["cleaning_due"].isna()]
                else:
                    display_tasks = pd.concat([overdue_tasks, due_today, upcoming_tasks.head(10)])
                
                if len(display_tasks) > 0:
                    for _, eq in display_tasks.iterrows():
                        col_task1, col_task2, col_task3 = st.columns([3, 2, 1])
                        
                        with col_task1:
                            st.write(f"**{eq['name']}**")
                            st.write(f"Type: {eq['type']}")
                            
                            # Obter nome da cervejaria
                            brewery_name = ""
                            if not breweries_df.empty and eq["brewery_id"] in breweries_df["id_brewery"].values:
                                brewery_name = breweries_df[breweries_df["id_brewery"] == eq["brewery_id"]]["name"].iloc[0]
                            st.write(f"Brewery: {brewery_name}")
                        
                        with col_task2:
                            if pd.notna(eq.get("cleaning_due")):
                                due_date = eq["cleaning_due"].date()
                                days_diff = (due_date - today).days
                                
                                if days_diff < 0:
                                    st.error(f"**Overdue by {abs(days_diff)} days**")
                                elif days_diff == 0:
                                    st.warning("**Due today!**")
                                elif days_diff <= 3:
                                    st.warning(f"**Due in {days_diff} days**")
                                else:
                                    st.info(f"Due: {due_date}")
                            
                            st.write(f"Frequency: {eq.get('cleaning_frequency', 'As needed')}")
                        
                        with col_task3:
                            if st.button("✓ Complete", key=f"complete_{eq['id_equipment']}", use_container_width=True):
                                # Marcar como limpo (próxima limpeza em 7 dias)
                                updates = {
                                    "cleaning_due": today + timedelta(days=7)
                                }
                                update_data("equipment", updates, "id_equipment = :id_equipment", {"id_equipment": eq["id_equipment"]})
                                data = get_all_data()
                                st.success(f"Cleaning completed for {eq['name']}!")
                                st.rerun()
                        
                        st.markdown("---")
                else:
                    st.success("✅ No cleaning tasks found with selected filters!")
            else:
                st.info("No cleaning schedule data available for equipment.")
        else:
            st.info("No equipment available for cleaning tasks.")
        
        # Tarefas de manutenção
        st.markdown("---")
        st.subheader("🛠️ Maintenance Tasks")
        
        if not equipment_df.empty and "next_maintenance" in equipment_df.columns:
            equipment_copy = equipment_df.copy()
            equipment_copy["next_maintenance"] = pd.to_datetime(equipment_copy["next_maintenance"])
            
            # Filtrar manutenções próximas (próximos 30 dias)
            today = datetime.now().date()
            upcoming_maintenance = equipment_copy[
                (equipment_copy["next_maintenance"].dt.date >= today) &
                (equipment_copy["next_maintenance"].dt.date <= today + timedelta(days=30))
            ].sort_values("next_maintenance")
            
            if len(upcoming_maintenance) > 0:
                for _, eq in upcoming_maintenance.iterrows():
                    due_date = eq["next_maintenance"].date()
                    days_until = (due_date - today).days
                    
                    col_maint1, col_maint2, col_maint3 = st.columns([3, 2, 1])
                    
                    with col_maint1:
                        st.write(f"**{eq['name']}**")
                        st.write(f"Type: {eq['type']}")
                        
                        # Obter nome da cervejaria
                        brewery_name = ""
                        breweries_df = data.get("breweries", pd.DataFrame())
                        if not breweries_df.empty and eq["brewery_id"] in breweries_df["id_brewery"].values:
                            brewery_name = breweries_df[breweries_df["id_brewery"] == eq["brewery_id"]]["name"].iloc[0]
                        st.write(f"Brewery: {brewery_name}")
                    
                    with col_maint2:
                        if days_until <= 7:
                            st.error(f"**Due in {days_until} days**")
                        elif days_until <= 14:
                            st.warning(f"**Due in {days_until} days**")
                        else:
                            st.info(f"Due: {due_date}")
                        
                        if eq.get("notes"):
                            st.caption(f"Notes: {eq['notes'][:50]}...")
                    
                    with col_maint3:
                        if st.button("✓ Done", key=f"maint_{eq['id_equipment']}", use_container_width=True):
                            # Marcar manutenção como concluída (próxima em 90 dias)
                            updates = {
                                "next_maintenance": today + timedelta(days=90)
                            }
                            update_data("equipment", updates, "id_equipment = :id_equipment", {"id_equipment": eq["id_equipment"]})
                            data = get_all_data()
                            st.success(f"Maintenance completed for {eq['name']}!")
                            st.rerun()
                    
                    st.markdown("---")
            else:
                st.success("✅ No maintenance scheduled for the next 30 days!")
        else:
            st.info("No maintenance data available for equipment.")
        
        # Criar nova tarefa
        st.markdown("---")
        st.subheader("➕ Create New Task")
        
        task_type = st.selectbox("Task Type", ["Cleaning", "Maintenance", "Quality Check", "Inventory", "Other"], key="new_task_type")
        
        col_task_new1, col_task_new2 = st.columns(2)
        with col_task_new1:
            task_title = st.text_input("Task Title", key="new_task_title")
            
            if not equipment_df.empty:
                task_equipment = st.selectbox(
                    "Related Equipment",
                    ["None"] + equipment_df["name"].tolist(),
                    key="new_task_equipment"
                )
            else:
                task_equipment = st.text_input("Related Equipment", key="new_task_eq_text")
        
        with col_task_new2:
            due_date = st.date_input("Due Date", datetime.now().date() + timedelta(days=1), key="new_task_due")
            priority = st.select_slider("Priority", options=["Low", "Medium", "High"], value="Medium", key="new_task_priority")
        
        task_description = st.text_area("Task Description", key="new_task_desc")
        
        if st.button("➕ Create Task", type="primary", use_container_width=True, key="create_task_btn"):
            if task_title:
                # Em um sistema completo, isso criaria uma entrada em uma tabela de tarefas
                st.success(f"Task '{task_title}' created successfully!")
                st.info("In a complete system, this would save to a 'tasks' table in the database.")
            else:
                st.error("Task title is required!")
    
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# RODAPÉ
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Brewery Manager v2.0 • Multiusuário (Postgres/SQLite)")
