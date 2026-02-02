# Brewery Manager - Multi-user Streamlit app (Postgres/Neon)
import streamlit as st
import bcrypt
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import io
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
import html
import re
import os
import tempfile
import json

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
    """Return a small HTML badge for common statuses."""
    s = (status or "").strip() or "N/A"
    s_low = s.lower()

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
    """Return configured users from Streamlit secrets (optional).

    If auth is not configured, returns an empty dict and the app will run
    in guest visualization mode only.
    """
    try:
        return st.secrets["auth"]["credentials"]["usernames"]
    except Exception:
        return {}

def _check_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

def auth_sidebar():
    """Optional admin login + session mode control.

    Default behavior (no login):
      - Guest access
      - Visualization mode ON (read-only)

    If an admin logs in:
      - Visualization mode can be toggled off to enable edits
    """

    # Default to guest visualization mode
    if not st.session_state.get("logged_in"):
        st.session_state.setdefault("visualization_mode", True)

    with st.sidebar:
        st.markdown("### Session")

        if st.session_state.get("logged_in"):
            role = st.session_state.get("auth_role", "viewer")
            name = st.session_state.get("auth_name", "user")
            mode = "Visualization" if is_visualization_mode() else "Edit"

            st.caption(f"Signed in as: {name} ({role})")
            st.caption(f"Mode: {mode}")

            if role == "admin":
                st.toggle(
                    "👁️ Visualization mode",
                    key="visualization_mode",
                    help="Read-only: disables all database writes for this session.",
                )
            else:
                st.session_state["visualization_mode"] = True

            if st.button("🚪 Sign out"):
                for k in ["logged_in", "auth_user", "auth_name", "auth_role", "visualization_mode"]:
                    st.session_state.pop(k, None)
                # back to guest visualization
                st.session_state["visualization_mode"] = True
                st.rerun()

            # -----------------------------
            # Admin tools: upload / restore Excel backup
            # -----------------------------
            if role == "admin":
                st.divider()
                st.markdown("### Admin tools")

                with st.expander("📦 Upload / restore backup (.xlsx)", expanded=False):
                    st.caption("Upload an Excel backup (same format as Export). You can save it to the server and/or restore the database from it.")

                    uploaded = st.file_uploader(
                        "Backup file (.xlsx)",
                        type=["xlsx"],
                        key="admin_backup_uploader",
                        help="Tip: edit prices in the Excel file, then upload and restore.",
                    )

                    note = st.text_input("Note (optional)", value="", key="admin_backup_note")

                    confirm = st.checkbox(
                        "I understand this will overwrite the database tables from the backup.",
                        value=False,
                        key="admin_backup_confirm",
                    )

                    col_a, col_b = st.columns(2)
                    if uploaded is not None:
                        try:
                            xls = pd.ExcelFile(uploaded)
                            st.write("Sheets detected:", ", ".join(xls.sheet_names))
                        except Exception:
                            st.warning("Could not read the Excel file preview. You can still try saving/restoring.")

                        with col_a:
                            if st.button("💾 Save backup to server", use_container_width=True):
                                require_admin_action()
                                backup_id = save_admin_backup_to_db(uploaded.getvalue(), uploaded.name, note.strip() or None)
                                st.success(f"Backup saved (id={backup_id}).")

                        with col_b:
                            if st.button("♻️ Restore database now", use_container_width=True, disabled=not confirm):
                                require_admin_action()
                                backup_id = save_admin_backup_to_db(uploaded.getvalue(), uploaded.name, note.strip() or None)
                                result = restore_from_excel_backup_bytes(uploaded.getvalue())
                                st.success(f"Restore completed. Saved backup id={backup_id}.")
                                st.json(result)
                                st.rerun()

                    st.markdown("#### Saved backups")
                    backups_df = list_admin_backups()
                    if backups_df.empty:
                        st.caption("No backups saved yet.")
                    else:
                        options = [
                            (
                                int(row["id_backup"]),
                                f'#{int(row["id_backup"])} — {row["filename"]} — {row.get("uploaded_at", "")}'
                            )
                            for _, row in backups_df.iterrows()
                        ]
                        sel_label = st.selectbox(
                            "Choose a saved backup",
                            options=[lbl for _, lbl in options],
                            key="admin_backup_select",
                        )
                        sel_id = int([bid for bid, lbl in options if lbl == sel_label][0])

                        if st.button("♻️ Restore selected backup", disabled=not confirm, use_container_width=True):
                            require_admin_action()
                            b = load_admin_backup_bytes(sel_id)
                            result = restore_from_excel_backup_bytes(b)
                            st.success("Restore completed.")
                            st.json(result)
                            st.rerun()


        else:
            # Guest
            st.caption("Signed in as: Guest")
            st.caption("Mode: Visualization (read-only)")

            users = _auth_users()
            if users:
                with st.expander("Admin login", expanded=False):
                    with st.form("admin_login_form", clear_on_submit=False):
                        username = st.text_input("Username", value="", placeholder="ex: admin")
                        password = st.text_input("Password", value="", type="password")
                        submitted = st.form_submit_button("Sign in")

                    if submitted:
                        if username not in users:
                            st.error("Invalid username or password.")
                            st.stop()
                        user_cfg = users[username]
                        if not _check_password(password, user_cfg.get("password", "")):
                            st.error("Invalid username or password.")
                            st.stop()

                        role = user_cfg.get("role", "viewer")
                        st.session_state["logged_in"] = True
                        st.session_state["auth_user"] = username
                        st.session_state["auth_name"] = user_cfg.get("name", username)
                        st.session_state["auth_role"] = role

                        # viewers remain read-only; admins default to edit mode
                        st.session_state["visualization_mode"] = (role != "admin")
                        st.success("Signed in!")
                        st.rerun()
            else:
                st.info("Admin login is not configured (missing [auth] in secrets).")

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


def _first_existing(cols: list[str], candidates: list[str]) -> str | None:
    """Return the first candidate that exists in cols (case-insensitive)."""
    if not cols:
        return None
    s = {str(c).lower() for c in cols}
    for cand in candidates:
        if str(cand).lower() in s:
            # return the original col name from cols to preserve case
            for c in cols:
                if str(c).lower() == str(cand).lower():
                    return c
            return cand
    return None

def _normalize_id_value(v):
    """Normalize ids for safe comparisons (e.g., 1.0 -> '1', UUID/text unchanged)."""
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    # floats like 1.0 coming from pandas when ints have NaNs
    try:
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
    except Exception:
        pass
    return str(v).strip()

def get_recipe_items_for_recipe(recipe_id) -> pd.DataFrame:
    """Fetch recipe_items for a recipe directly from DB (bypasses get_all_data cache)."""
    engine = get_engine()
    dialect = engine.dialect.name.lower()
    try:
        recipe_items_cols = get_table_columns_cached('recipe_items', dialect) or []
        fk_col = _first_existing(recipe_items_cols, ['recipe_id', 'id_recipe', 'id_receipt', 'recipe', 'recipe_fk', 'fk_recipe']) or 'recipe_id'
        # safest: compare as text
        q = f"SELECT * FROM recipe_items WHERE CAST({fk_col} AS TEXT) = :rid"
        return query_to_df(q, {"rid": str(recipe_id)})
    except Exception:
        # fallback to cached snapshot
        data = get_all_data()
        df = data.get("recipe_items", pd.DataFrame())
        if df is None or df.empty:
            return pd.DataFrame()
        # best effort filter
        fk = _col(df, 'recipe_id', 'id_recipe', 'id_receipt')
        if fk:
            ridn = _normalize_id_value(recipe_id)
            return df[df[fk].apply(_normalize_id_value) == ridn].copy()
        return df


def _secrets_read_only_default() -> bool:
    """Optional default read-only mode via Streamlit secrets/env.

    Streamlit Cloud secrets example:

        [app]
        read_only = true

    Env vars accepted:
      - VISUALIZATION_ONLY=1
      - READ_ONLY=1
    """
    try:
        if "app" in st.secrets and isinstance(st.secrets["app"], dict):
            v = st.secrets["app"].get("read_only")
            if isinstance(v, bool):
                return v
    except Exception:
        pass

    env = (os.getenv("VISUALIZATION_ONLY") or os.getenv("READ_ONLY") or "").strip().lower()
    return env in {"1", "true", "yes", "y", "on"}


def is_visualization_mode() -> bool:
    """Session-level visualization (read-only) mode."""
    default = _secrets_read_only_default()
    return bool(st.session_state.get("visualization_mode", default))


def is_admin_role() -> bool:
    """Whether the authenticated user *role* is admin (independent of mode)."""
    return st.session_state.get("auth_role") == "admin"


def can_write() -> bool:
    """True only when the user is an admin AND not in visualization mode."""
    return is_admin_role() and (not is_visualization_mode())


def is_admin() -> bool:
    """Backwards-compat: in this app, `is_admin()` means 'has write permissions'."""
    return can_write()


def require_admin_action():
    """Strong guard: blocks all DB writes unless `can_write()` is True."""
    if is_visualization_mode():
        st.error("🔒 Visualization mode: changes are disabled in this session.")
        st.stop()
    if not is_admin_role():
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

    IMPORTANT:
    - We clear the cached column list after ALTER TABLE so insert/update helpers
      don't silently drop newly-added fields.
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

    try:
        with engine.begin() as conn:
            for col in missing:
                col_def = columns_sql[col]
                conn.execute(sql_text(f"ALTER TABLE {table_name} ADD COLUMN {col} {col_def}"))
    finally:
        # Clear cached column metadata so subsequent writes can see new columns
        try:
            _get_table_columns_cached.clear()
        except Exception:
            pass


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
"""CREATE TABLE IF NOT EXISTS stock_movements (
                id_stock_move BIGSERIAL PRIMARY KEY,
                movement_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                ingredient_name TEXT,
                delta_qty DOUBLE PRECISION,
                direction TEXT,
                unit TEXT,
                reason TEXT,
                source TEXT,
                destination TEXT,
                ref_table TEXT,
                ref_id BIGINT,
                batch_id BIGINT,
                prod_event_id BIGINT,
                order_id BIGINT,
                order_number TEXT,
                created_by TEXT,
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

            """CREATE TABLE IF NOT EXISTS production_batches (
                id_batch BIGSERIAL PRIMARY KEY,
                batch_code TEXT,
                recipe_id TEXT,
                recipe_name TEXT,
                brewery_id TEXT,
                planned_date DATE,
                planned_volume_l DOUBLE PRECISION,
                brewhouse TEXT,
                status TEXT DEFAULT 'Planned',
                stage TEXT DEFAULT 'Production Order',
                current_vessel TEXT,
                volume_remaining_l DOUBLE PRECISION,
                notes TEXT,
                created_by TEXT,
                created_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS production_events (
                id_prod_event BIGSERIAL PRIMARY KEY,
                batch_id BIGINT,
                event_type TEXT,
                event_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                from_vessel TEXT,
                to_vessel TEXT,
                notes TEXT,
                created_by TEXT,
                created_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS production_consumptions (
                id_consumption BIGSERIAL PRIMARY KEY,
                batch_id BIGINT,
                prod_event_id BIGINT,
                ingredient_id TEXT,
                ingredient_name TEXT,
                quantity DOUBLE PRECISION,
                unit TEXT,
                unit_cost DOUBLE PRECISION,
                total_cost DOUBLE PRECISION,
                meta TEXT,
                created_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS production_keg_runs (
                id_keg_run BIGSERIAL PRIMARY KEY,
                batch_id BIGINT,
                product_name TEXT,
                keg_size_l DOUBLE PRECISION,
                keg_count INTEGER,
                beer_volume_l DOUBLE PRECISION,
                packaging_item TEXT,
                packaging_qty DOUBLE PRECISION,
                warehouse TEXT,
                estimated_cost DOUBLE PRECISION,
                actual_cost DOUBLE PRECISION,
                notes TEXT,
                created_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )""",

            """CREATE TABLE IF NOT EXISTS composite_products (
                id_composite BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                sku TEXT,
                beer_recipe_id TEXT,
                beer_recipe_name TEXT,
                output_unit TEXT DEFAULT 'unit',
                beer_liters_per_unit DOUBLE PRECISION DEFAULT 0,
                notes TEXT,
                status TEXT DEFAULT 'Active',
                created_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS composite_product_items (
                id_composite_item BIGSERIAL PRIMARY KEY,
                composite_id BIGINT,
                component_type TEXT,
                component_name TEXT,
                quantity DOUBLE PRECISION,
                unit TEXT,
                notes TEXT,
                created_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS composite_inventory (
                id_composite_inventory BIGSERIAL PRIMARY KEY,
                composite_id BIGINT,
                composite_name TEXT,
                warehouse TEXT,
                quantity_units DOUBLE PRECISION,
                created_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
             )""",

            """CREATE TABLE IF NOT EXISTS deposits (
                id_deposit BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                address TEXT,
                city TEXT,
                country TEXT,
                notes TEXT,
                status TEXT DEFAULT 'Active',
                created_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS customers (
                id_customer BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                org_no TEXT,
                email TEXT,
                phone TEXT,
                billing_address TEXT,
                delivery_address TEXT,
                notes TEXT,
                status TEXT DEFAULT 'Active',
                created_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS sales_orders (
                id_sales_order BIGSERIAL PRIMARY KEY,
                order_no TEXT,
                order_date DATE,
                delivery_date DATE,
                customer_id BIGINT,
                customer_name TEXT,
                deposit_id BIGINT,
                deposit_name TEXT,
                status TEXT DEFAULT 'Draft',
                currency TEXT DEFAULT 'NOK',
                subtotal DOUBLE PRECISION DEFAULT 0,
                total DOUBLE PRECISION DEFAULT 0,
                confirmed_date DATE,
                fulfilled_date DATE,
                notes TEXT,
                created_by TEXT,
                created_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS sales_order_items (
                id_sales_order_item BIGSERIAL PRIMARY KEY,
                sales_order_id BIGINT,
                product_id BIGINT,
                product_name TEXT,
                quantity DOUBLE PRECISION,
                unit TEXT,
                unit_price DOUBLE PRECISION,
                line_total DOUBLE PRECISION,
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
            """CREATE TABLE IF NOT EXISTS admin_backups (
                id_backup BIGSERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                uploaded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                uploaded_by TEXT,
                note TEXT,
                content BYTEA NOT NULL
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
"""CREATE TABLE IF NOT EXISTS stock_movements (
                id_stock_move INTEGER PRIMARY KEY AUTOINCREMENT,
                movement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ingredient_name TEXT,
                delta_qty REAL,
                direction TEXT,
                unit TEXT,
                reason TEXT,
                source TEXT,
                destination TEXT,
                ref_table TEXT,
                ref_id INTEGER,
                batch_id INTEGER,
                prod_event_id INTEGER,
                order_id INTEGER,
                order_number TEXT,
                created_by TEXT,
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

            """CREATE TABLE IF NOT EXISTS production_batches (
                id_batch INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_code TEXT,
                recipe_id TEXT,
                recipe_name TEXT,
                brewery_id TEXT,
                planned_date DATE,
                planned_volume_l REAL,
                brewhouse TEXT,
                status TEXT DEFAULT 'Planned',
                stage TEXT DEFAULT 'Production Order',
                current_vessel TEXT,
                volume_remaining_l REAL,
                notes TEXT,
                created_by TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS production_events (
                id_prod_event INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER,
                event_type TEXT,
                event_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                from_vessel TEXT,
                to_vessel TEXT,
                notes TEXT,
                created_by TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS production_consumptions (
                id_consumption INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER,
                prod_event_id INTEGER,
                ingredient_id TEXT,
                ingredient_name TEXT,
                quantity REAL,
                unit TEXT,
                unit_cost REAL,
                total_cost REAL,
                meta TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS production_keg_runs (
                id_keg_run INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER,
                product_name TEXT,
                keg_size_l REAL,
                keg_count INTEGER,
                beer_volume_l REAL,
                packaging_item TEXT,
                packaging_qty REAL,
                warehouse TEXT,
                estimated_cost REAL,
                actual_cost REAL,
                notes TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
             )""",

            """CREATE TABLE IF NOT EXISTS composite_products (
                id_composite INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                sku TEXT,
                recipe_id TEXT,
                recipe_name TEXT,
                output_unit TEXT DEFAULT 'unit',
                beer_liters_per_unit REAL DEFAULT 0,
                notes TEXT,
                status TEXT DEFAULT 'Active',
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS composite_product_items (
                id_composite_item INTEGER PRIMARY KEY AUTOINCREMENT,
                composite_id INTEGER,
                component_type TEXT,
                component_name TEXT,
                quantity REAL,
                unit TEXT,
                notes TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS composite_inventory (
                id_composite_inventory INTEGER PRIMARY KEY AUTOINCREMENT,
                composite_id INTEGER,
                composite_name TEXT,
                warehouse TEXT,
                quantity_units REAL,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",

            """CREATE TABLE IF NOT EXISTS deposits (
                id_deposit INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                address TEXT,
                city TEXT,
                country TEXT,
                notes TEXT,
                status TEXT DEFAULT 'Active',
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS customers (
                id_customer INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                org_no TEXT,
                email TEXT,
                phone TEXT,
                billing_address TEXT,
                delivery_address TEXT,
                notes TEXT,
                status TEXT DEFAULT 'Active',
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS sales_orders (
                id_sales_order INTEGER PRIMARY KEY AUTOINCREMENT,
                order_no TEXT,
                order_date DATE,
                delivery_date DATE,
                customer_id INTEGER,
                customer_name TEXT,
                deposit_id INTEGER,
                deposit_name TEXT,
                status TEXT DEFAULT 'Draft',
                currency TEXT DEFAULT 'NOK',
                subtotal REAL DEFAULT 0,
                total REAL DEFAULT 0,
                confirmed_date DATE,
                fulfilled_date DATE,
                notes TEXT,
                created_by TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS sales_order_items (
                id_sales_order_item INTEGER PRIMARY KEY AUTOINCREMENT,
                sales_order_id INTEGER,
                product_id INTEGER,
                product_name TEXT,
                quantity REAL,
                unit TEXT,
                unit_price REAL,
                line_total REAL,
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
            "use_manual_cost": "INTEGER DEFAULT 0",
            "unit_cost_manual": "DOUBLE PRECISION",
            "manual_cost_notes": "TEXT",
            "manual_cost_updated": "TIMESTAMPTZ",
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
            # Keep legacy keys + new brew-recipe stats
            "name": "TEXT",
            "recipe_name": "TEXT",
            "style": "TEXT",
            "beer_style": "TEXT",
            "description": "TEXT",
            "notes": "TEXT",
            "brewery_id": "TEXT",
            "id_brewery": "TEXT",
            "target_brewery_id": "TEXT",
            "brewery_name": "TEXT",
            "target_brewery_name": "TEXT",
            "batch_size": "DOUBLE PRECISION",
            "batch_volume": "DOUBLE PRECISION",
            "abv": "DOUBLE PRECISION",
            "ibu": "DOUBLE PRECISION",
            "ibus": "DOUBLE PRECISION",
            "srm": "DOUBLE PRECISION",
            "ebc": "DOUBLE PRECISION",
            "color_ebc": "DOUBLE PRECISION",
            "efficiency": "DOUBLE PRECISION",
            "brewhouse_efficiency": "DOUBLE PRECISION",
            "brew_efficiency": "DOUBLE PRECISION",
            "bh_efficiency": "DOUBLE PRECISION",
            "efficiency_pct": "DOUBLE PRECISION",
            "efficiency_percent": "DOUBLE PRECISION",
            "og": "DOUBLE PRECISION",
            "target_og": "DOUBLE PRECISION",
            "og_target": "DOUBLE PRECISION",
            "original_gravity": "DOUBLE PRECISION",
            "fg": "DOUBLE PRECISION",
            "target_fg": "DOUBLE PRECISION",
            "fg_target": "DOUBLE PRECISION",
            "final_gravity": "DOUBLE PRECISION",
            "created_date": "TIMESTAMPTZ",
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
        "production_batches",
        {
            "batch_code": "TEXT",
            "recipe_id": "TEXT",
            "recipe_name": "TEXT",
            "brewery_id": "TEXT",
            "planned_date": "DATE",
            "planned_volume_l": "DOUBLE PRECISION",
            "brewhouse": "TEXT",
            "status": "TEXT",
            "stage": "TEXT",
            "current_vessel": "TEXT",
            "volume_remaining_l": "DOUBLE PRECISION",
            "notes": "TEXT",
            "created_by": "TEXT",
        },
    )
    _ensure_columns(
        "production_events",
        {
            "batch_id": "BIGINT",
            "event_type": "TEXT",
            "event_date": "TIMESTAMPTZ",
            "from_vessel": "TEXT",
            "to_vessel": "TEXT",
            "notes": "TEXT",
            "created_by": "TEXT",
        },
    )
    _ensure_columns(
        "production_consumptions",
        {
            "batch_id": "BIGINT",
            "prod_event_id": "BIGINT",
            "ingredient_id": "TEXT",
            "ingredient_name": "TEXT",
            "quantity": "DOUBLE PRECISION",
            "unit": "TEXT",
            "unit_cost": "DOUBLE PRECISION",
            "total_cost": "DOUBLE PRECISION",
            "meta": "TEXT",
        },
    )
    _ensure_columns(
        "production_keg_runs",
        {
            "batch_id": "BIGINT",
            "product_name": "TEXT",
            "keg_size_l": "DOUBLE PRECISION",
            "keg_count": "INTEGER",
            "beer_volume_l": "DOUBLE PRECISION",
            "packaging_item": "TEXT",
            "packaging_qty": "DOUBLE PRECISION",
            "warehouse": "TEXT",
            "estimated_cost": "DOUBLE PRECISION",
            "actual_cost": "DOUBLE PRECISION",
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




    # --- Composite products (finished goods) ---
    _ensure_columns(
        "composite_products",
        {
            "name": "TEXT",
            "recipe_id": "TEXT",
            "recipe_name": "TEXT",
            "output_unit": "TEXT",
            "notes": "TEXT",
            "status": "TEXT",
        },
    )
    _ensure_columns(
        "composite_product_items",
        {
            "composite_id": "BIGINT",
            "component_type": "TEXT",
            "component_name": "TEXT",
            "quantity": "DOUBLE PRECISION",
            "unit": "TEXT",
        },
    )
    _ensure_columns(
        "composite_inventory",
        {
            "composite_id": "BIGINT",
            "composite_name": "TEXT",
            "warehouse": "TEXT",
            "quantity_units": "DOUBLE PRECISION",
        },
    )



    # --- Sales / Orders ---
    _ensure_columns(
        "deposits",
        {
            "name": "TEXT",
            "address": "TEXT",
            "city": "TEXT",
            "country": "TEXT",
            "notes": "TEXT",
            "status": "TEXT",
        },
    )
    _ensure_columns(
        "customers",
        {
            "name": "TEXT",
            "org_no": "TEXT",
            "email": "TEXT",
            "phone": "TEXT",
            "billing_address": "TEXT",
            "delivery_address": "TEXT",
            "notes": "TEXT",
            "status": "TEXT",
        },
    )
    _ensure_columns(
        "sales_orders",
        {
            "order_no": "TEXT",
            "order_date": "DATE",
            "delivery_date": "DATE",
            "customer_id": "BIGINT",
            "customer_name": "TEXT",
            "deposit_id": "BIGINT",
            "deposit_name": "TEXT",
            "status": "TEXT",
            "currency": "TEXT",
            "subtotal": "DOUBLE PRECISION DEFAULT 0",
            "total": "DOUBLE PRECISION DEFAULT 0",
            "confirmed_date": "DATE",
            "fulfilled_date": "DATE",
            "notes": "TEXT",
            "created_by": "TEXT",
        },
    )
    _ensure_columns(
        "sales_order_items",
        {
            "sales_order_id": "BIGINT",
            "product_id": "BIGINT",
            "product_name": "TEXT",
            "quantity": "DOUBLE PRECISION",
            "unit": "TEXT",
            "unit_price": "DOUBLE PRECISION",
            "line_total": "DOUBLE PRECISION",
            "notes": "TEXT",
        },
    )
    # --- Production extensions ---
    _ensure_columns(
        "production_batches",
        {
            "og": "DOUBLE PRECISION",
            "efficiency": "DOUBLE PRECISION",
            "brewed_volume_l": "DOUBLE PRECISION",
            "loss_l": "DOUBLE PRECISION DEFAULT 0",
            "finished_date": "DATE",
            # cancellation tracking (admin action: Cancel Order)
            "cancelled_reason": "TEXT",
            "cancelled_by": "TEXT",
            "cancelled_at": "TIMESTAMPTZ",
        },
    )
    _ensure_columns(
        "production_events",
        {
            "meta": "TEXT",
        },
    )
    _ensure_columns(
        "production_keg_runs",
        {
            "composite_id": "BIGINT",
            "composite_name": "TEXT",
            "units_produced": "DOUBLE PRECISION",
            "output_unit": "TEXT",
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


    # Fallback: many tables use custom PK names like id_prod_event, id_ingredient, etc.
    if pk_col not in set(actual_cols):
        try:
            id_like = [c for c in actual_cols if str(c).lower().startswith("id_")]
            if id_like:
                sing = str(singular).lower()
                best = [c for c in id_like if sing and sing in str(c).lower()]
                pk_col = best[0] if best else id_like[0]
        except Exception:
            pass
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
    cols_set = set(actual_cols)

    # Keep only columns that exist (but be resilient to stale cache after schema changes)
    filtered_updates = {k: _to_python_scalar(v) for k, v in data_dict.items() if k in cols_set}

    # If cache is stale (common on Streamlit Cloud across deploys), refresh once
    if data_dict and (not filtered_updates or len(filtered_updates) < len(data_dict)):
        try:
            get_table_columns_cached.clear()
        except Exception:
            try:
                _get_table_columns_cached.clear()
            except Exception:
                pass
        actual_cols = get_table_columns_cached(table_name, dialect) or list(data_dict.keys())
        cols_set = set(actual_cols)
        filtered_updates = {k: _to_python_scalar(v) for k, v in data_dict.items() if k in cols_set}

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
        'stock_movements',
        # legacy (kept for backwards compatibility)
        'purchases',
        # purchase-by-order (preferred)
        'purchase_orders', 'purchase_order_items',
        'suppliers', 'recipes', 'recipe_items',
        'breweries', 'equipment',
        # production (new pipeline)
        'production_batches', 'production_events', 'production_consumptions', 'production_keg_runs',
        # finished goods
        'composite_products', 'composite_product_items', 'composite_inventory',
        # sales / orders
        'deposits', 'customers', 'sales_orders', 'sales_order_items',
        # legacy
        'production_orders', 'calendar_events', 'team_members'
    ]
    data: dict[str, pd.DataFrame] = {}
    for table in table_names:
        data[table] = get_table_data(table)
    return data
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
            "Production": "production_orders",
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
        st.error(f"Migration error: {e}")
        return False

def _excel_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df that is safe to write to Excel.

    - Excel does not support timezone-aware datetimes
    - Some object columns (dict/list/etc.) need to be stringified
    """
    df = df.copy()

    # 1) Timezone-aware datetimes -> timezone-naive (Excel limitation)
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64tz_dtype(s):
            # Convert to naive (drops timezone); keeps absolute time
            df[col] = s.dt.tz_convert(None)
        elif s.dtype == "object":
            # Handle cases where tz-aware datetimes are stored as objects
            try:
                dt = pd.to_datetime(s, errors="raise")
                if pd.api.types.is_datetime64tz_dtype(dt):
                    df[col] = dt.dt.tz_convert(None)
            except Exception:
                pass

    # 2) Non-scalar objects -> string (avoids writer errors)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(
                lambda x: str(x) if isinstance(x, (dict, list, set, tuple)) else x
            )

    return df


def export_to_excel():
    """Exporta todos os dados para Excel"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        data = get_all_data()
        for table_name, df in data.items():
            if df is None or df.empty:
                continue

            # Excel sheet name limit is 31 chars
            safe_sheet_name = str(table_name)[:31]

            try:
                safe_df = _excel_safe_df(df)
                safe_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
            except Exception as e:
                # Make it easy to see which table caused the export to fail
                raise ValueError(f"Erro ao exportar a tabela '{table_name}': {e}") from e

    output.seek(0)
    return output


# -----------------------------
# BACKUP UPLOAD / RESTORE (ADMIN)
# -----------------------------
def _read_excel_backup_bytes(xlsx_bytes: bytes) -> dict[str, pd.DataFrame]:
    """Read an Excel backup (export format) into a dict {table_name: df}."""
    xls = pd.ExcelFile(io.BytesIO(xlsx_bytes))
    data: dict[str, pd.DataFrame] = {}
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        if df is None:
            continue
        # Normalize NaN to None for DB inserts
        df = df.replace({np.nan: None})
        data[str(sheet)] = df
    return data


def _truncate_or_delete_table(conn, table_name: str) -> None:
    """Clear a table in a Postgres-safe way."""
    # NOTE: We keep this very small to avoid accidental SQL injection.
    safe = re.sub(r"[^a-zA-Z0-9_]+", "", table_name)
    if safe != table_name:
        raise ValueError(f"Invalid table name: {table_name}")
    dialect = conn.engine.dialect.name.lower()
    if dialect in {"postgresql", "postgres"}:
        conn.execute(sql_text(f'TRUNCATE TABLE "{table_name}" RESTART IDENTITY CASCADE'))
    else:
        conn.execute(sql_text(f'DELETE FROM {table_name}'))


def _reset_serial_sequences(conn, table_name: str, df: pd.DataFrame) -> None:
    """After inserting explicit IDs, ensure Postgres sequences are aligned."""
    dialect = conn.engine.dialect.name.lower()
    if dialect not in {"postgresql", "postgres"}:
        return

    # Try to reset any serial sequence associated with id_* columns present in df
    for col in df.columns:
        if not str(col).startswith("id_"):
            continue
        # pg_get_serial_sequence returns the sequence name if this column is backed by a sequence
        seq = conn.execute(
            sql_text("SELECT pg_get_serial_sequence(:t, :c)"),
            {"t": table_name, "c": str(col)},
        ).scalar()
        if not seq:
            continue
        conn.execute(
            sql_text(
                "SELECT setval(:seq, COALESCE((SELECT MAX({col}) FROM \"{t}\"), 1), true)".format(
                    col=str(col),
                    t=table_name,
                )
            ),
            {"seq": seq},
        )


def save_admin_backup_to_db(xlsx_bytes: bytes, filename: str, note: str | None = None) -> int:
    """Persist an uploaded backup into the database (BYTEA) so it survives restarts."""
    require_admin_action()
    engine = get_engine()
    uploaded_by = st.session_state.get("auth_user") or st.session_state.get("auth_name") or "admin"
    with engine.begin() as conn:
        result = conn.execute(
            sql_text(
                """INSERT INTO admin_backups (filename, uploaded_by, note, content)
                   VALUES (:filename, :uploaded_by, :note, :content)
                   RETURNING id_backup"""
            ),
            {
                "filename": filename,
                "uploaded_by": uploaded_by,
                "note": note,
                "content": xlsx_bytes,
            },
        )
        backup_id = int(result.scalar())
    return backup_id


def list_admin_backups() -> pd.DataFrame:
    engine = get_engine()
    try:
        return query_to_df(
            "SELECT id_backup, filename, uploaded_at, uploaded_by, note FROM admin_backups ORDER BY uploaded_at DESC",
            {},
        )
    except Exception:
        # If table doesn't exist yet (first run), return empty
        return pd.DataFrame(columns=["id_backup", "filename", "uploaded_at", "uploaded_by", "note"])


def load_admin_backup_bytes(backup_id: int) -> bytes:
    require_admin_action()
    engine = get_engine()
    with engine.begin() as conn:
        b = conn.execute(
            sql_text("SELECT content FROM admin_backups WHERE id_backup = :id"),
            {"id": int(backup_id)},
        ).scalar()
    if b is None:
        raise ValueError("Backup not found.")
    return bytes(b)


def restore_from_excel_backup_bytes(xlsx_bytes: bytes) -> dict[str, int]:
    """Restore database tables from an Excel backup. Returns {table: rows_inserted}."""
    require_admin_action()
    data = _read_excel_backup_bytes(xlsx_bytes)

    # Restore order helps if foreign keys exist (even if most tables don't enforce them)
    preferred_order = [
        "breweries",
        "suppliers",
        "ingredients",
        "stock_movements",
        "purchase_orders",
        "purchase_order_items",
        "recipes",
        "recipe_items",
        "composite_products",
        "composite_product_items",
        "equipment",
        "production_batches",
        "production_events",
        "production_consumptions",
        "calendar_events",
    ]
    tables = [t for t in preferred_order if t in data] + [t for t in data.keys() if t not in preferred_order]

    engine = get_engine()
    inserted: dict[str, int] = {}

    with engine.begin() as conn:
        # Clear in reverse order to reduce FK issues
        for table in reversed(tables):
            _truncate_or_delete_table(conn, table)

        # Insert in order
        for table in tables:
            df = data[table]
            if df is None or df.empty:
                inserted[table] = 0
                continue

            # Replace pandas NA with None for SQLAlchemy
            df2 = df.where(pd.notnull(df), None)

            # Write
            df2.to_sql(table, con=conn, if_exists="append", index=False, method="multi")

            # Align sequences if needed
            _reset_serial_sequences(conn, table, df2)

            inserted[table] = int(len(df2))

    bump_db_version()
    return inserted

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


        # --- Stock movement history (best-effort) ---
        try:
            unit_map = {}
            for it in items_payload:
                nm = str(it.get('ingredient') or '')
                if nm and nm not in unit_map:
                    unit_map[nm] = str(it.get('unit') or '')
            mv_rows = []
            for u in updates:
                nm = str(u.get('n') or '')
                dq = float(u.get('q') if 'q' in u else u.get('d') or 0)
                if abs(dq) <= 1e-12 or not nm:
                    continue
                direction = 'IN' if dq > 0 else 'OUT'
                src = supplier if direction == 'IN' else 'Inventory'
                dst = 'Inventory' if direction == 'IN' else supplier
                mv_rows.append({
                    'movement_date': po_date,
                    'ingredient_name': nm,
                    'delta_qty': float(dq),
                    'direction': direction,
                    'unit': unit_map.get(nm, ''),
                    'reason': str(transaction_type),
                    'source': src,
                    'destination': dst,
                    'ref_table': 'purchase_orders',
                    'ref_id': int(po_id),
                    'order_id': int(po_id),
                    'order_number': str(order_number),
                    'created_by': str(recorded_by),
                    'notes': (notes or ''),
                })
            if mv_rows:
                conn.execute(
                    sql_text(
                        """
                        INSERT INTO stock_movements
                            (movement_date, ingredient_name, delta_qty, direction, unit,
                             reason, source, destination, ref_table, ref_id, order_id, order_number, created_by, notes)
                        VALUES
                            (:movement_date, :ingredient_name, :delta_qty, :direction, :unit,
                             :reason, :source, :destination, :ref_table, :ref_id, :order_id, :order_number, :created_by, :notes)
                        """
                    ),
                    mv_rows,
                )
        except Exception:
            pass
    bump_db_version()
    return po_id


def _recalc_unit_cost_for_ingredients(conn, ingredient_names: list[str]):
    """Recalculate ingredients.unit_cost as a weighted average of effective_unit_cost.

    Semantics:
    - 'Purchase' adds quantity and cost
    - 'Return' subtracts quantity and cost (undoes/partially undoes a purchase)
    - Other transaction types (Adjustment/Sample/Other) are ignored for unit_cost

    This keeps unit_cost consistent if users correct mistakes via Return instead of editing/deleting.
    """
    ingredient_names = [str(n) for n in ingredient_names if str(n).strip()]
    if not ingredient_names:
        return

    for ing in sorted(set(ingredient_names)):
        row = conn.execute(
            sql_text(
                """
                SELECT
                    SUM(
                        CASE
                            WHEN o.transaction_type = 'Return' THEN -1 * (i.quantity * i.effective_unit_cost)
                            ELSE (i.quantity * i.effective_unit_cost)
                        END
                    ) AS cost_total,
                    SUM(
                        CASE
                            WHEN o.transaction_type = 'Return' THEN -1 * i.quantity
                            ELSE i.quantity
                        END
                    ) AS qty_total
                FROM purchase_order_items i
                JOIN purchase_orders o ON o.id_purchase_order = i.purchase_order_id
                WHERE o.transaction_type IN ('Purchase', 'Return')
                  AND i.ingredient = :ing
                """
            ),
            {"ing": ing},
        ).fetchone()

        cost_total = float(row[0] or 0.0) if row else 0.0
        qty_total = float(row[1] or 0.0) if row else 0.0

        if qty_total > 0:
            avg_cost = cost_total / qty_total
            conn.execute(
                sql_text(
                    """
                    UPDATE ingredients
                       SET unit_cost = :c,
                           last_updated = CURRENT_TIMESTAMP
                     WHERE name = :n
                    """
                ),
                {"c": float(avg_cost), "n": ing},
            )
        else:
            # If no purchase history remains, keep unit_cost as-is (don't zero it out automatically)
            pass


def update_purchase_order_fast(
    po_id: int,
    transaction_type: str,
    supplier: str,
    order_number: str,
    po_date: date,
    freight_total: float,
    notes: str,
    recorded_by: str,
    preview_rows: list[dict],
):
    """Update a purchase order (header + items) and adjust inventory deltas.

    We compute the *inventory impact* of the old order vs the new order and apply the delta to ingredients.stock.
    Then we recalc unit_cost as weighted avg of purchase history for affected ingredients.
    """
    require_admin_action()
    engine = get_engine()

    with engine.begin() as conn:
        # Load old header + items
        old_hdr = conn.execute(
            sql_text("SELECT transaction_type, supplier, order_number FROM purchase_orders WHERE id_purchase_order = :id"),
            {"id": int(po_id)},
        ).fetchone()
        old_type = str(old_hdr[0] if old_hdr else "Purchase")

        old_items = conn.execute(
            sql_text(
                """
                SELECT ingredient, quantity
                  FROM purchase_order_items
                 WHERE purchase_order_id = :id
                """
            ),
            {"id": int(po_id)},
        ).fetchall()

        # Aggregate old/new impacts
        def impact_sign(t: str) -> float:
            return 1.0 if str(t) == "Purchase" else -1.0

        old_imp = {}
        for ing, qty in old_items:
            ing = str(ing)
            old_imp[ing] = old_imp.get(ing, 0.0) + float(qty or 0.0) * impact_sign(old_type)

        new_imp = {}
        for row in preview_rows:
            ing = str(row.get("Ingredient", "") or "")
            qty = float(row.get("Quantity", 0.0) or 0.0)
            if ing and qty:
                new_imp[ing] = new_imp.get(ing, 0.0) + qty * impact_sign(transaction_type)

        all_ings = sorted(set(list(old_imp.keys()) + list(new_imp.keys())))

        # Header update
        conn.execute(
            sql_text(
                """
                UPDATE purchase_orders
                   SET transaction_type = :transaction_type,
                       supplier = :supplier,
                       order_number = :order_number,
                       date = :date,
                       freight_total = :freight_total,
                       notes = :notes,
                       recorded_by = :recorded_by
                 WHERE id_purchase_order = :id
                """
            ),
            {
                "transaction_type": transaction_type,
                "supplier": supplier,
                "order_number": order_number,
                "date": po_date,
                "freight_total": float(freight_total) if transaction_type == "Purchase" else 0.0,
                "notes": notes,
                "recorded_by": recorded_by,
                "id": int(po_id),
            },
        )

        # Replace items
        conn.execute(
            sql_text("DELETE FROM purchase_order_items WHERE purchase_order_id = :id"),
            {"id": int(po_id)},
        )

        items_payload = []
        for row in preview_rows:
            items_payload.append(
                {
                    "purchase_order_id": int(po_id),
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

        # Apply stock deltas
        updates = []
        for ing in all_ings:
            delta = float(new_imp.get(ing, 0.0) - old_imp.get(ing, 0.0))
            if abs(delta) > 1e-12:
                updates.append({"n": ing, "d": delta})

        if updates:
            conn.execute(
                sql_text(
                    """
                    UPDATE ingredients
                       SET stock = COALESCE(stock,0) + :d,
                           last_updated = CURRENT_TIMESTAMP
                     WHERE name = :n
                    """
                ),
                updates,
            )


        # --- Stock movement history for PO edit (best-effort) ---
        try:
            mv_rows = []
            for u in updates:
                nm = str(u.get('n') or '')
                dq = float(u.get('d') or 0.0)
                if abs(dq) <= 1e-12 or not nm:
                    continue
                direction = 'IN' if dq > 0 else 'OUT'
                src = supplier if direction == 'IN' else 'Inventory'
                dst = 'Inventory' if direction == 'IN' else supplier
                mv_rows.append({
                    'movement_date': po_date,
                    'ingredient_name': nm,
                    'delta_qty': float(dq),
                    'direction': direction,
                    'unit': '',
                    'reason': f'PO Update ({transaction_type})',
                    'source': src,
                    'destination': dst,
                    'ref_table': 'purchase_orders',
                    'ref_id': int(po_id),
                    'order_id': int(po_id),
                    'order_number': str(order_number),
                    'created_by': str(recorded_by),
                    'notes': (notes or ''),
                })
            if mv_rows:
                conn.execute(
                    sql_text(
                        """
                        INSERT INTO stock_movements
                            (movement_date, ingredient_name, delta_qty, direction, unit,
                             reason, source, destination, ref_table, ref_id, order_id, order_number, created_by, notes)
                        VALUES
                            (:movement_date, :ingredient_name, :delta_qty, :direction, :unit,
                             :reason, :source, :destination, :ref_table, :ref_id, :order_id, :order_number, :created_by, :notes)
                        """
                    ),
                    mv_rows,
                )
        except Exception:
            pass

        # Recalc unit costs for affected ingredients
        _recalc_unit_cost_for_ingredients(conn, all_ings)

    bump_db_version()




def delete_stock_movement_fast(stock_move_id: int, *, also_reverse_stock: bool = True) -> tuple[bool, str]:
    """Delete one row from stock_movements.

    If also_reverse_stock=True, it will ALSO reverse its inventory impact by applying:
        ingredients.stock = ingredients.stock - movement.delta_qty

    Notes:
      - This is an admin-only action.
      - If the movement references another entity (PO / production event), deleting it will NOT delete that entity.
    """
    require_admin_action()
    engine = get_engine()
    dialect = engine.dialect.name.lower()

    # Resolve PK column for stock_movements
    sm_cols = set(get_table_columns_cached('stock_movements', dialect) or [])
    sm_id_col = 'id_stock_move' if 'id_stock_move' in sm_cols else ('id' if 'id' in sm_cols else None)

    # Resolve stock column for ingredients
    ing_cols = set(get_table_columns_cached('ingredients', dialect) or [])
    ing_stock_col = 'stock' if 'stock' in ing_cols else ('quantity_in_stock' if 'quantity_in_stock' in ing_cols else None)
    has_last_updated = ('last_updated' in ing_cols)

    with engine.begin() as conn:
        # Fetch movement row (best effort across schemas)
        row = None
        try:
            if sm_id_col:
                row = conn.execute(
                    sql_text(f"SELECT * FROM stock_movements WHERE {sm_id_col} = :id"),
                    {'id': int(stock_move_id)},
                ).mappings().fetchone()
        except Exception:
            row = None

        # Fallback if schema uses a different PK name
        if row is None and not sm_id_col:
            try:
                row = conn.execute(
                    sql_text("SELECT * FROM stock_movements WHERE id_stock_move = :id"),
                    {'id': int(stock_move_id)},
                ).mappings().fetchone()
            except Exception:
                row = None

        if row is None:
            return False, "Movement not found."

        ing_name = str(row.get('ingredient_name') or row.get('ingredient') or '')
        try:
            delta = float(row.get('delta_qty') or row.get('qty') or row.get('quantity') or 0.0)
        except Exception:
            delta = 0.0

        # Reverse inventory impact
        if also_reverse_stock and ing_name and ing_stock_col and abs(delta) > 1e-12:
            try:
                if has_last_updated:
                    conn.execute(
                        sql_text(
                            f"UPDATE ingredients SET {ing_stock_col} = COALESCE({ing_stock_col},0) - :d, last_updated = CURRENT_TIMESTAMP WHERE name = :n"
                        ),
                        {'d': float(delta), 'n': ing_name},
                    )
                else:
                    conn.execute(
                        sql_text(
                            f"UPDATE ingredients SET {ing_stock_col} = COALESCE({ing_stock_col},0) - :d WHERE name = :n"
                        ),
                        {'d': float(delta), 'n': ing_name},
                    )
            except Exception:
                # If we can't update stock, we still allow deleting the movement row.
                pass

        # Delete movement row
        try:
            if sm_id_col:
                conn.execute(
                    sql_text(f"DELETE FROM stock_movements WHERE {sm_id_col} = :id"),
                    {'id': int(stock_move_id)},
                )
            else:
                conn.execute(
                    sql_text("DELETE FROM stock_movements WHERE id_stock_move = :id"),
                    {'id': int(stock_move_id)},
                )
        except Exception as e:
            return False, f"Delete failed: {e}"

    bump_db_version()
    return True, "Movement deleted."


def delete_purchase_order_fast(po_id: int):
    """Hard-delete a purchase order + items, and reverse its inventory impact."""
    require_admin_action()
    engine = get_engine()

    with engine.begin() as conn:
        hdr = conn.execute(
            sql_text("SELECT transaction_type FROM purchase_orders WHERE id_purchase_order = :id"),
            {"id": int(po_id)},
        ).fetchone()
        old_type = str(hdr[0] if hdr else "Purchase")
        supplier = str(hdr[1] if hdr and len(hdr) > 1 else "")
        order_number = str(hdr[2] if hdr and len(hdr) > 2 else "")

        items = conn.execute(
            sql_text(
                """
                SELECT ingredient, quantity
                  FROM purchase_order_items
                 WHERE purchase_order_id = :id
                """
            ),
            {"id": int(po_id)},
        ).fetchall()

        def impact_sign(t: str) -> float:
            return 1.0 if str(t) == "Purchase" else -1.0

        imp = {}
        for ing, qty in items:
            ing = str(ing)
            imp[ing] = imp.get(ing, 0.0) + float(qty or 0.0) * impact_sign(old_type)

        # Reverse impact
        updates = []
        for ing, signed_qty in imp.items():
            delta = -float(signed_qty)
            if abs(delta) > 1e-12:
                updates.append({"n": ing, "d": delta})

        if updates:
            conn.execute(
                sql_text(
                    """
                    UPDATE ingredients
                       SET stock = COALESCE(stock,0) + :d,
                           last_updated = CURRENT_TIMESTAMP
                     WHERE name = :n
                    """
                ),
                updates,
            )

        # --- Stock movement history for PO delete (best-effort) ---
        try:
            mv_rows = []
            for u in updates:
                nm = str(u.get('n') or '')
                dq = float(u.get('d') or 0.0)
                if abs(dq) <= 1e-12 or not nm:
                    continue
                direction = 'IN' if dq > 0 else 'OUT'
                src = supplier if direction == 'IN' else 'Inventory'
                dst = 'Inventory' if direction == 'IN' else supplier
                mv_rows.append({
                    'movement_date': pd.Timestamp.utcnow(),
                    'ingredient_name': nm,
                    'delta_qty': float(dq),
                    'direction': direction,
                    'unit': '',
                    'reason': f'PO Delete (was {old_type})',
                    'source': src,
                    'destination': dst,
                    'ref_table': 'purchase_orders',
                    'ref_id': int(po_id),
                    'order_id': int(po_id),
                    'order_number': str(order_number),
                    'created_by': str(st.session_state.get('auth_user', 'admin')),
                    'notes': 'Purchase order deleted',
                })
            if mv_rows:
                conn.execute(
                    sql_text(
                        """
                        INSERT INTO stock_movements
                            (movement_date, ingredient_name, delta_qty, direction, unit,
                             reason, source, destination, ref_table, ref_id, order_id, order_number, created_by, notes)
                        VALUES
                            (:movement_date, :ingredient_name, :delta_qty, :direction, :unit,
                             :reason, :source, :destination, :ref_table, :ref_id, :order_id, :order_number, :created_by, :notes)
                        """
                    ),
                    mv_rows,
                )
        except Exception:
            pass

        # Delete children then header
        conn.execute(
            sql_text("DELETE FROM purchase_order_items WHERE purchase_order_id = :id"),
            {"id": int(po_id)},
        )
        conn.execute(
            sql_text("DELETE FROM purchase_orders WHERE id_purchase_order = :id"),
            {"id": int(po_id)},
        )

        _recalc_unit_cost_for_ingredients(conn, list(imp.keys()))

    bump_db_version()

def update_stock_from_usage(ingredient_name, quantity):
    """Atualiza o estoque após uso em produção"""
    return execute_query(
        "UPDATE ingredients SET stock = stock - :quantity, last_updated = CURRENT_TIMESTAMP WHERE name = :ingredient_name",
        {"quantity": quantity, "ingredient_name": ingredient_name}
    )

def _quote_ident(name: str) -> str:
    """Safely double-quote a SQL identifier."""
    return '"' + str(name).replace('"', '""') + '"'


def check_ingredient_usage(ingredient_id):
    """Return True if ingredient appears in recipe_items (supports schema variations)."""
    # Try to discover the ingredient FK column in recipe_items
    try:
        cols_df = query_to_df("SELECT * FROM recipe_items LIMIT 0")
    except Exception:
        # If table doesn't exist / can't be queried, assume not in use
        return False

    ing_col = _col(
        cols_df,
        'id_ingredient', 'ingredient_id', 'id_ing', 'ingredients_id', 'id_ingredients',
        'fk_ingredient', 'ingredient', 'ingredient_fk'
    )
    if not ing_col:
        return False

    q = f"SELECT COUNT(*) as count FROM recipe_items WHERE {_quote_ident(ing_col)} = :ingredient_id"
    try:
        result = query_to_df(q, {"ingredient_id": ingredient_id})
        return bool(result.iloc[0]['count'] > 0)
    except Exception:
        # Fallback: try without quoting (some DBs don't like quoted lower-case identifiers)
        try:
            q2 = f"SELECT COUNT(*) as count FROM recipe_items WHERE {ing_col} = :ingredient_id"
            result = query_to_df(q2, {"ingredient_id": ingredient_id})
            return bool(result.iloc[0]['count'] > 0)
        except Exception:
            return False



def _col(df, *names):
    """Return first matching column name (case-insensitive) or None."""
    if df is None or df.empty:
        return None
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    return None



def _is_active_status(status_val) -> bool:
    """Consider a batch/order active unless explicitly completed/cancelled."""
    s = (str(status_val or '')).strip().lower()
    return s not in {'completed', 'cancelled', 'canceled'}


def _capacity_to_l(capacity_raw, volume_l: float | None = None) -> float:
    """Parse a capacity value and return liters.

    Accepts:
      - numbers (assumed liters by default)
      - strings like '2000', '2,000', '20 hL', '20hl', '30 L', '10 bbl', '100 gal'
    Heuristic:
      - if no unit is given and volume_l is provided, treat small capacities as hL when that makes sense.
    """
    try:
        import numpy as _np
        number_types = (int, float, _np.number)
    except Exception:
        number_types = (int, float)

    cap_num = 0.0
    unit = None

    if capacity_raw is None:
        cap_num = 0.0
    elif isinstance(capacity_raw, number_types):
        cap_num = float(capacity_raw)
    else:
        s = str(capacity_raw).strip().lower()
        # keep unit hint before stripping
        if 'hl' in s:
            unit = 'hl'
        elif 'bbl' in s:
            unit = 'bbl'
        elif 'gal' in s:
            unit = 'gal'
        elif re.search(r'\bl\b', s) or s.endswith('l') or ' liters' in s or ' litre' in s:
            unit = 'l'
        # normalize number formatting: '2,000' -> '2000', '2,5' -> '2.5'
        s_num = s.replace(' ', '')
        # if both comma and dot exist, assume comma is thousands sep
        if ',' in s_num and '.' in s_num:
            s_num = s_num.replace(',', '')
        else:
            s_num = s_num.replace(',', '.')
        m = re.search(r"[-+]?\d+(?:\.\d+)?", s_num)
        cap_num = float(m.group(0)) if m else 0.0

    # Unit conversions
    if unit == 'hl':
        cap_l = cap_num * 100.0
    elif unit == 'bbl':
        cap_l = cap_num * 117.347765  # US beer barrel
    elif unit == 'gal':
        cap_l = cap_num * 3.78541     # US gallon
    else:
        cap_l = cap_num  # assume liters

    # Heuristic: if no explicit unit and a volume is given, interpret as hL when it matches reality.
    if unit is None and volume_l is not None:
        try:
            v = float(volume_l)
        except Exception:
            v = None
        if v and cap_l > 0:
            # Example: cap=20, volume=2000 => likely 20 hL
            if cap_l < v and cap_l * 100.0 >= v - 1e-6:
                cap_l = cap_l * 100.0

    return float(cap_l)


def _pick_col_from_cols(actual_cols, *candidates, contains_all=None, contains_any=None):
    """Pick a column name from a list of column names.

    1) Try exact matches (case-insensitive) for the provided candidates.
    2) If none match, try a fuzzy search:
       - contains_all: all of these substrings must be present
       - contains_any: at least one of these substrings must be present

    Returns the real column name as it appears in the DB, or None.
    """
    if not actual_cols:
        return None
    cols_map = {str(c).lower(): c for c in actual_cols}

    # exact
    for cand in candidates:
        if cand is None:
            continue
        key = str(cand).lower()
        if key in cols_map:
            return cols_map[key]

    # fuzzy
    cols_lower = [str(c).lower() for c in actual_cols]
    contains_all = [s.lower() for s in (contains_all or [])]
    contains_any = [s.lower() for s in (contains_any or [])]

    for c_low, c_real in zip(cols_lower, actual_cols):
        if contains_all and not all(s in c_low for s in contains_all):
            continue
        if contains_any and not any(s in c_low for s in contains_any):
            continue
        if contains_all or contains_any:
            return c_real
    return None


def _ingredient_name_label_map(ingredients_df: pd.DataFrame) -> dict:
    """Return a mapping {ingredient_name: "Manufacturer — Name"} for UI.

    IMPORTANT: The app still stores and queries ingredients by *name* in many places
    (legacy tables like recipe_items and purchase_order_items). This helper only
    changes how options are displayed in dropdowns.
    """
    if ingredients_df is None or ingredients_df.empty:
        return {}

    name_col = _col(ingredients_df, 'name', 'ingredient_name', 'ingredient')
    if not name_col:
        return {}

    mfg_col = _col(ingredients_df, 'manufacturer', 'brand', 'producer', 'maltster', 'maker')

    out = {}
    for _, r in ingredients_df.iterrows():
        nm = r.get(name_col)
        if nm is None or str(nm).strip() == "":
            continue
        nm_s = str(nm).strip()
        mfg = str(r.get(mfg_col) or "").strip() if mfg_col else ""
        out[nm_s] = f"{mfg} — {nm_s}" if mfg else nm_s
    return out


def build_recipe_insert_dict(recipe_name, recipe_style, batch_volume, efficiency, og, fg, ibus, ebc, selected_brewery, brewery_name, description):
    """Build a recipes-row payload that matches the *actual* recipes schema.

    We look up the real column names and only set keys that exist. This prevents
    'nan' / 'N/A' later when the DB uses different column names than the UI.

    Note: UI inputs OG/FG are in °P. We store those values in whichever gravity
    columns exist in the schema (including Plato targets).
    """
    try:
        actual_cols = get_table_columns_cached('recipes')
    except Exception:
        actual_cols = []

    payload = {}

    # Name/style/description
    name_col = _pick_col_from_cols(actual_cols, 'name', 'recipe_name', 'title', contains_any=['name'])
    if name_col:
        payload[name_col] = recipe_name

    style_col = _pick_col_from_cols(actual_cols, 'style', 'beer_style', 'type_style', contains_any=['style'])
    if style_col:
        payload[style_col] = recipe_style

    desc_col = _pick_col_from_cols(actual_cols, 'description', 'notes', 'desc', contains_any=['desc','note'])
    if desc_col:
        payload[desc_col] = description

    # Batch volume
    batch_col = _pick_col_from_cols(
        actual_cols,
        'batch_volume', 'batch_size', 'batch_l', 'batch_volume_l', 'batch_volume_liters',
        'volume_l', 'volume',
        contains_any=['batch', 'volume']
    )
    if batch_col:
        payload[batch_col] = float(batch_volume)

    # Efficiency
    eff_col = _pick_col_from_cols(
        actual_cols,
        'efficiency', 'brewhouse_efficiency', 'brew_efficiency', 'bh_efficiency',
        'efficiency_pct', 'efficiency_percent',
        contains_any=['eff']
    )
    if eff_col:
        payload[eff_col] = float(efficiency)

    # Brewery id / name
    brew_id_col = _pick_col_from_cols(
        actual_cols,
        'brewery_id', 'id_brewery', 'target_brewery_id', 'id_target_brewery',
        contains_all=['brewery'], contains_any=['id']
    )
    if brew_id_col:
        payload[brew_id_col] = selected_brewery

    brew_name_col = _pick_col_from_cols(actual_cols, 'brewery_name', 'target_brewery_name', contains_all=['brewery'], contains_any=['name'])
    if brew_name_col:
        payload[brew_name_col] = brewery_name

    # Gravities (Plato)
    og_col = _pick_col_from_cols(
        actual_cols,
        'og', 'target_og', 'og_target', 'og_planned',
        'original_gravity', 'target_original_gravity',
        'og_plato', 'original_gravity_plato', 'target_og_plato', 'target_original_gravity_plato',
        contains_any=['og','original_gravity']
    )
    if og_col:
        payload[og_col] = float(og)

    fg_col = _pick_col_from_cols(
        actual_cols,
        'fg', 'target_fg', 'fg_target', 'fg_planned',
        'final_gravity', 'target_final_gravity',
        'fg_plato', 'final_gravity_plato', 'target_fg_plato', 'target_final_gravity_plato',
        contains_any=['fg','final_gravity']
    )
    if fg_col:
        payload[fg_col] = float(fg)

    # IBU & Color
    ibu_col = _pick_col_from_cols(actual_cols, 'ibus', 'ibu', 'target_ibu', 'ibu_target', contains_any=['ibu'])
    if ibu_col:
        payload[ibu_col] = float(ibus)

    col_col = _pick_col_from_cols(actual_cols, 'ebc', 'color_ebc', 'colour_ebc', 'target_ebc', 'ebc_target', 'srm', 'color', contains_any=['ebc','srm','color'])
    if col_col:
        payload[col_col] = float(ebc)

    # ABV (optional)
    abv_val = (float(og) - float(fg)) * 0.524
    abv_col = _pick_col_from_cols(actual_cols, 'abv', 'target_abv', contains_any=['abv'])
    if abv_col:
        payload[abv_col] = float(abv_val)

    return payload

def get_vessels_for_production(data: dict) -> list[str]:
    """Best-effort list of vessels (tanks/BBT/serving) from equipment table."""
    eq = data.get('equipment', None)
    if eq is None or eq.empty:
        return []
    tcol = _col(eq, 'type')
    ncol = _col(eq, 'name')
    if not ncol:
        return []
    if tcol:
        mask = eq[tcol].astype(str).str.lower().str.contains('tank|bbt|ferment|unitank|serv', regex=True, na=False)
        out = eq.loc[mask, ncol].astype(str).tolist()
        if out:
            return sorted(dict.fromkeys(out))
    out = eq[ncol].astype(str).tolist()
    return sorted(dict.fromkeys(out))


def _norm_ing_name(s: str) -> str:
    """Normalize ingredient names for matching (case/whitespace tolerant)."""
    try:
        s = str(s)
    except Exception:
        return ''
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _find_ingredient_row_by_name(ing_df: pd.DataFrame, ingredient_name: str):
    """Best-effort ingredient lookup by name.

    Tries exact match, then normalized (case/whitespace) match. Returns a Series or None.
    """
    if ing_df is None or ing_df.empty:
        return None
    ncol = _col(ing_df, 'name')
    if not ncol:
        return None
    # Exact match
    m = ing_df[ing_df[ncol].astype(str) == str(ingredient_name)]
    if not m.empty:
        return m.iloc[0]
    # Normalized match
    qn = _norm_ing_name(ingredient_name)
    if not qn:
        return None
    try:
        ser = ing_df[ncol].astype(str).map(_norm_ing_name)
        m2 = ing_df[ser == qn]
        if not m2.empty:
            return m2.iloc[0]
    except Exception:
        pass
    return None



def _ingredient_effective_unit_cost(row) -> float:
    """Return the unit cost that should be used in calculations.

    If the ingredient has a manual override enabled, that value wins.
    Falls back to unit_cost (or legacy cost_per_unit).
    """
    try:
        use_manual = row.get("use_manual_cost", 0) if hasattr(row, "get") else 0
        # Normalize booleans/strings
        if isinstance(use_manual, str):
            use_manual = 1 if use_manual.strip().lower() in {"1", "true", "yes", "y", "on"} else 0
        try:
            use_manual = int(float(use_manual or 0))
        except Exception:
            use_manual = 0

        if use_manual:
            v = None
            try:
                v = row.get("unit_cost_manual", None) if hasattr(row, "get") else None
            except Exception:
                v = None
            if v is not None and str(v).strip().lower() not in {"", "nan", "none"}:
                try:
                    fv = float(v)
                    if fv > 0:
                        return float(fv)
                except Exception:
                    pass
    except Exception:
        pass

    # Fallbacks (calculated costs)
    for k in ("unit_cost", "cost_per_unit"):
        try:
            v = row.get(k, None) if hasattr(row, "get") else None
            if v is None or str(v).strip().lower() in {"", "nan", "none"}:
                continue
            return float(v)
        except Exception:
            continue
    return 0.0



def get_ingredient_unit_and_cost(data: dict, ingredient_name: str):
    ing = data.get('ingredients', None)
    if ing is None or ing.empty:
        return None, 0.0
    ncol = _col(ing, 'name')
    ucol = _col(ing, 'unit')
    ccol = _col(ing, 'unit_cost', 'cost_per_unit')
    mcol = _col(ing, 'unit_cost_manual')
    umcol = _col(ing, 'use_manual_cost')
    if not ncol:
        return None, 0.0
    row = _find_ingredient_row_by_name(ing, ingredient_name)
    if row is None:
        return None, 0.0
    unit = row.get(ucol) if ucol and ucol in ing.columns else None
    try:
        cost = _ingredient_effective_unit_cost(row)
    except Exception:
        cost = 0.0
    return unit, float(cost or 0.0)



def log_stock_movement(
    *,
    ingredient_name: str,
    delta_qty: float,
    unit: str | None = None,
    reason: str | None = None,
    source: str | None = None,
    destination: str | None = None,
    ref_table: str | None = None,
    ref_id: int | None = None,
    batch_id: int | None = None,
    prod_event_id: int | None = None,
    order_id: int | None = None,
    order_number: str | None = None,
    notes: str | None = None,
    movement_date: datetime | date | None = None,
    created_by: str | None = None,
):
    """Append a row to stock_movements (best-effort; never raises)."""
    try:
        created_by = created_by or st.session_state.get('auth_user', 'admin')
    except Exception:
        created_by = created_by or 'admin'

    try:
        dt = movement_date
        if dt is None:
            dt = pd.Timestamp.utcnow()
        else:
            dt = pd.Timestamp(dt)
    except Exception:
        dt = pd.Timestamp.utcnow()

    direction = 'IN' if float(delta_qty) > 0 else 'OUT'
    payload = {
        'movement_date': dt,
        'ingredient_name': str(ingredient_name),
        'delta_qty': float(delta_qty),
        'direction': direction,
        'unit': (unit or ''),
        'reason': reason or 'Adjustment',
        'source': source or ('Unknown source' if direction == 'IN' else 'Inventory'),
        'destination': destination or ('Inventory' if direction == 'IN' else 'Unknown usage'),
        'ref_table': ref_table or '',
        'ref_id': int(ref_id) if ref_id is not None else None,
        'batch_id': int(batch_id) if batch_id is not None else None,
        'prod_event_id': int(prod_event_id) if prod_event_id is not None else None,
        'order_id': int(order_id) if order_id is not None else None,
        'order_number': str(order_number) if order_number is not None else '',
        'created_by': created_by,
        'notes': str(notes) if notes is not None else '',
    }
    # Insert best-effort (ignore failures to avoid blocking stock updates)
    try:
        insert_data('stock_movements', payload)
    except Exception:
        pass


def adjust_stock_for_ingredient(
    data: dict,
    ingredient_name: str,
    delta_qty: float,
    *,
    reason: str | None = None,
    source: str | None = None,
    destination: str | None = None,
    ref_table: str | None = None,
    ref_id: int | None = None,
    batch_id: int | None = None,
    prod_event_id: int | None = None,
    order_id: int | None = None,
    order_number: str | None = None,
    notes: str | None = None,
    movement_date: datetime | date | None = None,
):
    """delta_qty: negative to consume, positive to add. Also logs to stock_movements."""
    ing = data.get('ingredients', None)
    if ing is None or ing.empty:
        return
    ncol = _col(ing, 'name')
    scol = _col(ing, 'stock', 'quantity_in_stock')
    idcol = _col(ing, 'id_ingredient', 'ingredient_id', 'id')
    ucol = _col(ing, 'unit')
    if not (ncol and scol and idcol):
        return
    row = _find_ingredient_row_by_name(ing, ingredient_name)
    if row is None:
        return

    try:
        prev_stock = float(row.get(scol) or 0)
    except Exception:
        prev_stock = 0.0

    new_stock = prev_stock + float(delta_qty)
    update_data('ingredients', {scol: new_stock}, f"{idcol} = :id", {'id': row[idcol]})

    try:
        unit0 = str(row.get(ucol) or '') if ucol and ucol in ing.columns else ''
    except Exception:
        unit0 = ''

    # Log movement (best-effort)
    log_stock_movement(
        ingredient_name=str(ingredient_name),
        delta_qty=float(delta_qty),
        unit=unit0,
        reason=reason,
        source=source,
        destination=destination,
        ref_table=ref_table,
        ref_id=ref_id,
        batch_id=batch_id,
        prod_event_id=prod_event_id,
        order_id=order_id,
        order_number=order_number,
        notes=notes,
        movement_date=movement_date,
    )


def sum_production_costs(data: dict, batch_id: int):
    cons = data.get('production_consumptions', pd.DataFrame())
    if cons is None or cons.empty:
        return 0.0
    bid_col = _col(cons, 'batch_id')
    tcol = _col(cons, 'total_cost')
    if not (bid_col and tcol):
        return 0.0
    sub = cons[cons[bid_col] == batch_id]
    try:
        return float(sub[tcol].fillna(0).sum())
    except Exception:
        return 0.0


def generate_production_report_pdf_bytes(batch_id: int) -> bytes:
    """Build a production report PDF for the given batch and return raw PDF bytes (A4)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.platypus import Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from datetime import datetime

    # Optional: embed fermentation charts when readings exist
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    data2 = get_all_data()
    batches = data2.get('production_batches', pd.DataFrame())
    events = data2.get('production_events', pd.DataFrame())
    cons = data2.get('production_consumptions', pd.DataFrame())
    kegs = data2.get('production_keg_runs', pd.DataFrame())
    recipes = data2.get('recipes', pd.DataFrame())

    # Resolve batch row
    b: dict = {}
    if batches is not None and not batches.empty:
        bid_col = _col(batches, 'id_batch', 'batch_id', 'id')
        if bid_col:
            mm_b = batches[batches[bid_col] == batch_id]
            if not mm_b.empty:
                b = mm_b.iloc[0].to_dict()

    # Related tables
    ev = pd.DataFrame()
    if events is not None and not events.empty:
        e_bid = _col(events, 'batch_id')
        if e_bid:
            ev = events[events[e_bid] == batch_id].copy()

    co = pd.DataFrame()
    if cons is not None and not cons.empty:
        c_bid = _col(cons, 'batch_id')
        if c_bid:
            co = cons[cons[c_bid] == batch_id].copy()

    kg = pd.DataFrame()
    if kegs is not None and not kegs.empty:
        k_bid = _col(kegs, 'batch_id')
        if k_bid:
            kg = kegs[kegs[k_bid] == batch_id].copy()

    total_cost = sum_production_costs(data2, batch_id)

    # Pull some recipe stats if possible
    recipe_stats: dict = {}
    try:
        rid = b.get('recipe_id', None)
        if rid is not None and recipes is not None and not recipes.empty:
            rid_col = _col(recipes, 'id_recipe', 'recipe_id', 'id')
            if rid_col:
                mrec = recipes[recipes[rid_col].astype(str) == str(rid)]
                if not mrec.empty:
                    recipe_stats = mrec.iloc[0].to_dict()
    except Exception:
        recipe_stats = {}

    def _safe(v) -> str:
        if v is None:
            return ''
        s = str(v)
        return '' if s.lower() == 'nan' else s

    def _fmt_num(v, decimals=None):
        try:
            if v is None or str(v).lower() == 'nan' or str(v).strip() == '':
                return ''
            fv = float(v)
            if decimals is None:
                return f"{fv:g}"
            return f"{fv:.{decimals}f}"
        except Exception:
            return _safe(v)

    def _first_nonempty(*vals):
        for v in vals:
            if v is None:
                continue
            s = str(v).strip()
            if s and s.lower() != 'nan':
                return v
        return ''

    # Batch basics
    planned_vol = _first_nonempty(b.get('planned_volume_l'), b.get('planned_volume'))
    planned_date = ''
    try:
        cd = b.get('created_date')
        if cd is not None and str(cd).strip() and str(cd).lower() != 'nan':
            dtv = pd.to_datetime(cd, errors='coerce')
            if not pd.isna(dtv):
                planned_date = dtv.date().isoformat()
    except Exception:
        planned_date = _safe(b.get('created_date'))
    brewed_vol = _first_nonempty(b.get('brewed_volume_l'), b.get('volume_brewed_l'), b.get('actual_volume_l'))
    og = _first_nonempty(b.get('og'))
    eff = _first_nonempty(b.get('efficiency'))
    stage = _first_nonempty(b.get('stage'))
    status = _first_nonempty(b.get('status'))
    vessel = _first_nonempty(b.get('current_vessel'), b.get('vessel'), b.get('fermenter'))

    ibu = _first_nonempty(recipe_stats.get('ibu'), recipe_stats.get('IBU'))
    color_val = _first_nonempty(
        recipe_stats.get('color'), recipe_stats.get('color_ebc'), recipe_stats.get('srm'),
        recipe_stats.get('colour'), recipe_stats.get('color_srm')
    )

    # Sort events by date if possible
    if ev is not None and not ev.empty:
        dtcol = _col(ev, 'event_date')
        try:
            if dtcol:
                ev[dtcol] = pd.to_datetime(ev[dtcol], errors='coerce')
                ev = ev.sort_values(dtcol)
        except Exception:
            pass

    # PDF build
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18*mm,
        rightMargin=18*mm,
        topMargin=22*mm,
        bottomMargin=16*mm
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=18,
        spaceAfter=6
    )
    beer_title_style = ParagraphStyle(
        "BeerTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        spaceAfter=2
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=colors.grey,
        spaceAfter=10
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        spaceBefore=10,
        spaceAfter=6
    )
    small = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=10.5
    )
    small_grey = ParagraphStyle(
        "SmallGrey",
        parent=small,
        textColor=colors.grey
    )
    normal = ParagraphStyle(
        "NormalTight",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12
    )

    # Batch code (fallback to Batch ID)
    batch_code_txt = str(b.get('batch_code') or batch_id)

    logo_path = _find_logo_path()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    def _on_page(canvas, _doc):
        canvas.saveState()

        # Header line + logo
        y_top = A4[1] - 12*mm
        x_left = _doc.leftMargin
        x_right = A4[0] - _doc.rightMargin

        logo_w = 0
        if logo_path:
            try:
                # Keep a fixed height, preserve aspect ratio
                img_h = 10*mm
                # reportlab needs actual image dimensions to compute ratio; try to use RLImage helper
                tmp = RLImage(logo_path)
                iw, ih = tmp.imageWidth, tmp.imageHeight
                if ih and iw:
                    img_w = img_h * (iw/ih)
                else:
                    img_w = 24*mm
                canvas.drawImage(logo_path, x_left, y_top - img_h + 1*mm, width=img_w, height=img_h, mask='auto')
                logo_w = img_w + 4*mm
            except Exception:
                logo_w = 0

        canvas.setFont("Helvetica-Bold", 12)
        canvas.drawString(x_left + logo_w, y_top - 3*mm, f"Production Report — Batch #{batch_id} - ({batch_code_txt})")

        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(x_right, y_top - 3*mm, f"Generated: {generated_at}")

        # Footer
        canvas.setStrokeColor(colors.lightgrey)
        canvas.setLineWidth(0.5)
        canvas.line(x_left, 12*mm, x_right, 12*mm)
        canvas.setFillColor(colors.grey)

        # Signature + confidentiality (left)
        canvas.setFont("Helvetica", 7.5)
        t = canvas.beginText(x_left, 9*mm)
        t.textLine("Schouskjelleren — Brewing & Production Software | Made by Joao Alberto Kolling")
        t.textLine("CONFIDENTIAL: This document contains proprietary production data. For internal use only. Do not distribute without authorization.")
        canvas.drawText(t)

        # Page number (right)
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(x_right, 9*mm, f"Page {_doc.page}")
        canvas.restoreState()

    story = []
    story.append(Spacer(1, 8*mm))

    # Overview block
    # Title block: keep the report title in the header; make the beer name prominent in the body
    rec_name = _safe(b.get('recipe_name',''))
    if rec_name:
        story.append(Paragraph(f"<b>{rec_name}</b>", beer_title_style))
    story.append(Paragraph(f"Batch #{batch_id} — Batch code: {batch_code_txt}", subtitle_style))


    # Build a 2x4 grid (label/value pairs)
    brewery = _first_nonempty(b.get('brewery_name'), b.get('brewery_id'))
    # System: use the brewhouse/site where the current vessel is registered (equipment.brewery_id -> breweries.name)
    system = _first_nonempty(b.get('brewhouse'), b.get('system'))
    try:
        eq = data2.get('equipment', pd.DataFrame())
        br = data2.get('breweries', pd.DataFrame())
        vname = _safe(vessel).strip().lower()
        if vname and eq is not None and not eq.empty:
            ncol = _col(eq, 'name')
            bidcol = _col(eq, 'brewery_id')
            if ncol and bidcol:
                mm_eq = eq[eq[ncol].astype(str).str.strip().str.lower() == vname]
                if not mm_eq.empty:
                    vessel_brewery_id = str(mm_eq.iloc[0][bidcol])
                    if br is not None and not br.empty:
                        idcol = _col(br, 'id_brewery', 'brewery_id', 'id')
                        namecol = _col(br, 'name')
                        if idcol and namecol:
                            mm_br = br[br[idcol].astype(str) == vessel_brewery_id]
                            if not mm_br.empty:
                                system = mm_br.iloc[0][namecol]
    except Exception:
        pass

    left_pairs = [
        ("Batch code", _safe(batch_code_txt)),
        ("Brewery", _safe(brewery)),
        ("System", _safe(system)),
        ("Planned", f"{_safe(planned_date)}" if planned_date else "")
    ]
    right_pairs = [
        ("Brew day", f"{_fmt_num(brewed_vol)} L • OG {_fmt_num(og)}°P • Eff {_fmt_num(eff)}%" if any([brewed_vol, og, eff]) else ""),
        ("Stage / Status", f"{_safe(stage)} / {_safe(status)}" if stage or status else ""),
        ("Vessel", _safe(vessel)),
        ("Created by", _safe(b.get("created_by")))
    ]

    def _pair_row(p1, p2):
        return [
            Paragraph(f"<font color='#{'666666'}'>{p1[0]}</font>", small_grey),
            Paragraph(_safe(p1[1]) or "—", normal),
            Paragraph(f"<font color='#{'666666'}'>{p2[0]}</font>", small_grey),
            Paragraph(_safe(p2[1]) or "—", normal),
        ]

    overview_table = Table(
        [_pair_row(left_pairs[i], right_pairs[i]) for i in range(4)],
        colWidths=[26*mm, 60*mm, 26*mm, 60*mm]
    )
    overview_table.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("LINEBELOW", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("LEFTPADDING", (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
    ]))
    story.append(overview_table)

    # KPI cards row
    def _kpi_cell(label, value):
        v = _safe(value) if _safe(value) else "—"
        return Paragraph(f"<font size='8' color='#666666'>{label}</font><br/><font size='12'><b>{v}</b></font>", styles["Normal"])

    kpi_items = [
        ("Volume", f"{_fmt_num(brewed_vol)} L" if brewed_vol else ""),
        ("OG", f"{_fmt_num(og)}°P" if og else ""),
        ("Efficiency", f"{_fmt_num(eff)}%" if eff else ""),
        ("IBU", _fmt_num(ibu) if ibu else ""),
        ("Color (EBC)", _fmt_num(color_val) if color_val else ""),
    ]
    kpi_table = Table([[ _kpi_cell(l, v) for (l, v) in kpi_items ]], colWidths=[(A4[0]-36*mm)/5]*5)
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.whitesmoke),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("BOX", (0,0), (-1,-1), 0.5, colors.lightgrey),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
    ]))
    story.append(Spacer(1, 6*mm))
    story.append(kpi_table)

    # Events
    story.append(Paragraph("Events", h2))
    if ev is not None and not ev.empty:
        dtcol = _col(ev, 'event_date')
        rows = [["Date/Time", "Event", "Move", "Notes"]]
        for _, r in ev.iterrows():
            d_raw = r.get(dtcol, r.get('event_date','')) if dtcol else r.get('event_date','')
            try:
                d_txt = pd.to_datetime(d_raw).strftime("%Y-%m-%d %H:%M")
            except Exception:
                d_txt = str(d_raw)[:16]
            ev_type = _safe(r.get('event_type',''))
            move = f"{_safe(r.get('from_vessel',''))} → {_safe(r.get('to_vessel',''))}".strip()
            note = _safe(r.get('notes',''))
            rows.append([d_txt, ev_type or "—", move if move != "→" else "—", note or ""])
        t = Table(rows, colWidths=[28*mm, 34*mm, 40*mm, (A4[0]-36*mm-28*mm-34*mm-40*mm)])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
            ("TEXTCOLOR", (0,0), (-1,0), colors.black),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,0), 9),
            ("FONTSIZE", (0,1), (-1,-1), 8.8),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LINEBELOW", (0,0), (-1,0), 0.5, colors.lightgrey),
            ("INNERGRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
            ("BOX", (0,0), (-1,-1), 0.5, colors.lightgrey),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 6),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("<i>No events logged.</i>", small_grey))

    # Fermentation readings (curve)
    # Use the same Reading events (event_type == 'Reading') and plot Gravity/Temp/Pressure over time.
    try:
        dtcol = _col(ev, 'event_date') if (ev is not None and not ev.empty) else None
        etcol = _col(ev, 'event_type', 'type') if (ev is not None and not ev.empty) else None
        mcol = _col(ev, 'meta') if (ev is not None and not ev.empty) else None
        ncol = _col(ev, 'notes') if (ev is not None and not ev.empty) else None
        rpdf = pd.DataFrame()
        if ev is not None and not ev.empty and etcol:
            rpdf = ev[ev[etcol].astype(str).str.lower() == 'reading'].copy()
        if rpdf is not None and not rpdf.empty:
            if dtcol and dtcol in rpdf.columns:
                rpdf[dtcol] = pd.to_datetime(rpdf[dtcol], errors='coerce')
                rpdf = rpdf.sort_values(dtcol)

            def _safe_json(_s):
                try:
                    if _s is None:
                        return {}
                    if isinstance(_s, dict):
                        return _s
                    ss = str(_s)
                    if not ss or ss.strip().lower() in {'nan', 'none'}:
                        return {}
                    return json.loads(ss)
                except Exception:
                    return {}

            if mcol and mcol in rpdf.columns:
                metas = rpdf[mcol].apply(_safe_json)
                rpdf['Gravity (°P)'] = metas.apply(lambda d: d.get('gravity_plato') if isinstance(d, dict) else None)
                rpdf['Temp (°C)'] = metas.apply(lambda d: d.get('temp_c') if isinstance(d, dict) else None)
                rpdf['Pressure (bar)'] = metas.apply(lambda d: d.get('pressure_bar') if isinstance(d, dict) else None)
                rpdf['pH'] = metas.apply(lambda d: d.get('ph') if isinstance(d, dict) else None)
            if ncol and ncol in rpdf.columns:
                rpdf['Notes'] = rpdf[ncol]
            else:
                rpdf['Notes'] = ''

            story.append(Paragraph("Fermentation readings", h2))

            # Chart (optional): embed only if matplotlib is available
            if plt is not None and dtcol and dtcol in rpdf.columns:
                try:
                    cdf = rpdf[[dtcol, 'Gravity (°P)', 'Temp (°C)', 'Pressure (bar)']].copy()
                    cdf = cdf.dropna(subset=[dtcol])
                    if not cdf.empty:
                        for cc in ('Gravity (°P)', 'Temp (°C)', 'Pressure (bar)'):
                            if cc in cdf.columns:
                                cdf[cc] = pd.to_numeric(cdf[cc], errors='coerce')

                        fig = plt.figure(figsize=(7.2, 2.2))
                        ax = fig.add_subplot(111)
                        ax.plot(cdf[dtcol], cdf['Gravity (°P)'], label='Gravity (°P)')
                        if cdf['Temp (°C)'].notna().any():
                            ax.plot(cdf[dtcol], cdf['Temp (°C)'], label='Temp (°C)')
                        if cdf['Pressure (bar)'].notna().any():
                            ax.plot(cdf[dtcol], cdf['Pressure (bar)'], label='Pressure (bar)')
                        ax.set_xlabel('Date')
                        ax.grid(True, alpha=0.25)
                        ax.legend(fontsize=7, loc='best')
                        fig.autofmt_xdate(rotation=30, ha='right')

                        img_buf = io.BytesIO()
                        fig.savefig(img_buf, format='png', dpi=160, bbox_inches='tight')
                        plt.close(fig)
                        img_buf.seek(0)

                        img = RLImage(img_buf, width=(A4[0]-36*mm), height=42*mm)
                        story.append(img)
                        story.append(Spacer(1, 2*mm))
                except Exception:
                    pass

            # Table
            rows = [["Date/Time", "Gravity (°P)", "Temp (°C)", "Pressure (bar)", "pH", "Notes"]]
            for _, rr in rpdf.iterrows():
                d_raw = rr.get(dtcol, '') if dtcol else ''
                try:
                    d_txt = pd.to_datetime(d_raw).strftime('%Y-%m-%d %H:%M')
                except Exception:
                    d_txt = str(d_raw)[:16]
                rows.append([
                    d_txt,
                    _fmt_num(rr.get('Gravity (°P)', ''), 2),
                    _fmt_num(rr.get('Temp (°C)', ''), 1),
                    _fmt_num(rr.get('Pressure (bar)', ''), 2),
                    _fmt_num(rr.get('pH', ''), 2),
                    _safe(rr.get('Notes', '')),
                ])

            rt = Table(rows, colWidths=[28*mm, 22*mm, 20*mm, 24*mm, 14*mm, (A4[0]-36*mm-28*mm-22*mm-20*mm-24*mm-14*mm)])
            rt.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE", (0,0), (-1,0), 9),
                ("FONTSIZE", (0,1), (-1,-1), 8.8),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("ALIGN", (1,1), (4,-1), "RIGHT"),
                ("LINEBELOW", (0,0), (-1,0), 0.5, colors.lightgrey),
                ("INNERGRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
                ("BOX", (0,0), (-1,-1), 0.5, colors.lightgrey),
                ("LEFTPADDING", (0,0), (-1,-1), 6),
                ("RIGHTPADDING", (0,0), (-1,-1), 6),
                ("TOPPADDING", (0,0), (-1,-1), 4),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ]))
            for i in range(1, len(rows)):
                if i % 2 == 0:
                    rt.setStyle(TableStyle([("BACKGROUND", (0,i), (-1,i), colors.Color(0.98,0.98,0.98))]))
            story.append(rt)
        else:
            # Keep the section out if there are no readings
            pass
    except Exception:
        pass

    # Consumptions
    story.append(Paragraph("Consumptions", h2))
    if co is not None and not co.empty:
        # Determine columns
        name_col = _col(co, 'ingredient_name', 'name')
        qty_col = _col(co, 'quantity', 'qty', 'amount')
        unit_col = _col(co, 'unit')
        cost_col = _col(co, 'total_cost', 'cost')

        rows = [["Ingredient", "Qty", "Unit", "Ingredient Cost"]]
        total_cost_calc = 0.0
        for _, r in co.iterrows():
            ing = _safe(r.get(name_col,'') if name_col else r.get('ingredient_name',''))
            qty = r.get(qty_col,'') if qty_col else r.get('quantity','')
            unit = _safe(r.get(unit_col,'') if unit_col else r.get('unit',''))
            cost = r.get(cost_col,'') if cost_col else r.get('total_cost','')
            try:
                total_cost_calc += float(cost) if str(cost).strip() else 0.0
            except Exception:
                pass
            rows.append([ing or "—", _fmt_num(qty), unit or "—", _fmt_num(cost, 2)])
        rows.append(["", "", Paragraph("<b>Total</b>", normal), Paragraph(f"<b>{_fmt_num(total_cost_calc, 2)}</b>", normal)])

        t = Table(rows, colWidths=[78*mm, 22*mm, 18*mm, (A4[0]-36*mm-78*mm-22*mm-18*mm)])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,0), 9),
            ("FONTSIZE", (0,1), (-1,-2), 8.8),
            ("ALIGN", (1,1), (1,-1), "RIGHT"),
            ("ALIGN", (3,1), (3,-1), "RIGHT"),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LINEBELOW", (0,0), (-1,0), 0.5, colors.lightgrey),
            ("INNERGRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
            ("BOX", (0,0), (-1,-1), 0.5, colors.lightgrey),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 6),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        # Zebra striping
        for i in range(1, len(rows)-1):
            if i % 2 == 0:
                t.setStyle(TableStyle([("BACKGROUND", (0,i), (-1,i), colors.Color(0.98,0.98,0.98))]))
        story.append(t)
    else:
        story.append(Paragraph("<i>No consumptions logged.</i>", small_grey))

    # Kegging
    story.append(Paragraph("Kegging", h2))
    if kg is not None and not kg.empty:
        rows = [["Units", "Unit", "Beer used (L)", "Warehouse", "SKU"]]
        for _, r in kg.iterrows():
            rows.append([
                _safe(r.get('units_produced','')) or "—",
                _safe(r.get('output_unit','')) or "—",
                _fmt_num(r.get('beer_volume_l',''), 2) or "—",
                _safe(r.get('warehouse','')) or "—",
                _safe(r.get('composite_name','')) or "—"
            ])
        t = Table(rows, colWidths=[18*mm, 18*mm, 26*mm, 38*mm, (A4[0]-36*mm-18*mm-18*mm-26*mm-38*mm)])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,0), 9),
            ("FONTSIZE", (0,1), (-1,-1), 8.8),
            ("ALIGN", (0,1), (2,-1), "RIGHT"),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LINEBELOW", (0,0), (-1,0), 0.5, colors.lightgrey),
            ("INNERGRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
            ("BOX", (0,0), (-1,-1), 0.5, colors.lightgrey),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 6),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("<i>No kegging runs logged.</i>", small_grey))

    # Build
    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return buf.getvalue()

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
# Optional admin login (default: guest visualization mode)
auth_sidebar()


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
    transition: border-color 120ms ease, box-shadow 120ms ease, transform 120ms ease;
}
.calendar-day:hover {
    border-color: #bdbdbd;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    transform: translateY(-1px);
}
.calendar-day.selected {
    border: 2px solid #111827;
    background-color: #f3f4f6;
}
.calendar-day.today {
    background-color: #e3f2fd;
    border: 2px solid #2196f3;
}
.calendar-day.today.selected {
    border: 2px solid #111827;
}
.calendar-day.weekend {
    background-color: #f8f9fa !important;
}
.calendar-day.weekend.today {
    background-color: #e3f2fd !important;
    border: 2px solid #2196f3 !important;
}
a.calendar-day-link {
    text-decoration: none;
    color: inherit;
    display: block;
    height: 100%;
}
.calendar-event {
    background-color: #4caf50;
    color: white;
    padding: 2px 5px;
    border-radius: 3px;
    font-size: 0.8rem;
    margin-bottom: 2px;
    transition: transform 120ms ease, filter 120ms ease, box-shadow 120ms ease;
    cursor: pointer;
}
.calendar-event:hover {
    transform: scale(1.04);
    filter: brightness(1.05);
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}
.calendar-selected-panel {
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 0.75rem 1rem;
    background: #ffffff;
    margin-top: 0.75rem;
}
.calendar-selected-title {
    font-weight: 700;
    margin-bottom: 0.5rem;
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
    
    # (Dashboard) cleaning alerts disabled (user cleans tanks after use)
    
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

    elif delete_type == "recipe":
        st.markdown(f"""
        <div class=\"delete-confirmation\">
            <h3>⚠️ Confirm Recipe Deletion</h3>
            <p>Are you sure you want to delete <strong>'{delete_name}'</strong>?</p>
            <p>This will also delete all ingredient lines for this recipe.</p>
            <p>This action cannot be undone.</p>
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
                    # Ingredients table PK varies across schemas (id / id_ingredient / ingredient_id)
                    engine = get_engine()
                    dialect = engine.dialect.name.lower()
                    ing_cols = get_table_columns_cached('ingredients', dialect) or []

                    def _first_existing_ci(cols, candidates):
                        s = set([c.lower() for c in cols])
                        for cand in candidates:
                            if cand.lower() in s:
                                # return the actual-cased column name from cols
                                for c in cols:
                                    if c.lower() == cand.lower():
                                        return c
                                return cand
                        return None

                    ing_id_col = _first_existing_ci(ing_cols, ['id', 'id_ingredient', 'ingredient_id']) or 'id'

                    try:
                        delete_data("ingredients", f"{ing_id_col} = :rid", {"rid": delete_id})
                    except Exception:
                        # try string comparison if types mismatch
                        delete_data("ingredients", f"CAST({ing_id_col} AS TEXT) = :rid", {"rid": str(delete_id)})

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
            elif delete_type == "recipe":
                # Delete recipe items first, then the recipe itself (handle schema variations)
                engine = get_engine()
                dialect = engine.dialect.name.lower()

                recipes_cols = get_table_columns_cached('recipes', dialect) or []
                recipe_items_cols = get_table_columns_cached('recipe_items', dialect) or []

                def _first_existing(cols, candidates):
                    s = set([c.lower() for c in cols])
                    for cand in candidates:
                        if cand.lower() in s:
                            return cand
                    return None

                recipes_id_col = _first_existing(recipes_cols, ['id_recipe', 'id_receipt', 'id'])
                items_fk_col = _first_existing(recipe_items_cols, ['recipe_id', 'id_recipe', 'id_receipt'])

                # Fallbacks if schema cache couldn't see columns
                if not recipes_id_col:
                    recipes_id_col = 'id_recipe'
                if not items_fk_col:
                    items_fk_col = 'recipe_id'

                # Delete lines
                try:
                    delete_data('recipe_items', f"{items_fk_col} = :rid", {'rid': delete_id})
                except Exception:
                    # try string comparison if types mismatch
                    delete_data('recipe_items', f"CAST({items_fk_col} AS TEXT) = :rid", {'rid': str(delete_id)})

                # Delete recipe
                try:
                    delete_data('recipes', f"{recipes_id_col} = :rid", {'rid': delete_id})
                except Exception:
                    delete_data('recipes', f"CAST({recipes_id_col} AS TEXT) = :rid", {'rid': str(delete_id)})

                st.success(f"Recipe '{delete_name}' deleted successfully!")

            st.session_state.delete_confirmation = {"type": None, "id": None, "name": None}
            data = get_all_data()
            st.rerun()
    
    with col_confirm2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.delete_confirmation = {"type": None, "id": None, "name": None}
            st.rerun()

# Verificar se há confirmação pendente
if st.session_state.delete_confirmation["type"] in ["ingredient", "supplier", "brewery", "recipe"]:
    handle_delete_confirmation()


# -----------------------------
# Reset de estado ao trocar de aba (página)
# -----------------------------
# O Streamlit mantém st.session_state entre reruns. Isso é útil, mas aqui queremos
# que o estado de edição/formulários NÃO fique preso quando o usuário navega.
# Então, sempre que a página muda, apagamos as chaves não essenciais.
_PERSISTENT_SESSION_KEYS = {
    # auth
    "logged_in", "auth_user", "auth_name", "auth_role", "visualization_mode",
    # navegação
    "page", "_last_page",
}

def _reset_non_persistent_state():
    for k in list(st.session_state.keys()):
        if k not in _PERSISTENT_SESSION_KEYS:
            try:
                del st.session_state[k]
            except Exception:
                pass


def _on_page_change():
    prev = st.session_state.get("_last_page")
    cur = st.session_state.get("page")
    # Só limpa quando realmente mudou
    if prev is not None and cur != prev:
        _reset_non_persistent_state()
    st.session_state["_last_page"] = cur



def _find_logo_path():
    """Return path to a project logo image if present, else None.

    Looks for common locations like assets/logo.(png|jpg|jpeg|webp) or logo.(png|...).
    """
    try:
        base_dirs = []
        # Prefer directory of this file, then current working directory
        try:
            base_dirs.append(Path(__file__).resolve().parent)
        except Exception:
            pass
        try:
            base_dirs.append(Path.cwd())
        except Exception:
            pass

        candidates = [
            Path("assets") / "logo.png",
            Path("assets") / "logo.jpg",
            Path("assets") / "logo.jpeg",
            Path("assets") / "logo.webp",
            Path("logo.png"),
            Path("logo.jpg"),
            Path("logo.jpeg"),
            Path("logo.webp"),
        ]
        for bd in base_dirs:
            for rel in candidates:
                p = (bd / rel).resolve()
                if p.exists() and p.is_file():
                    return str(p)
    except Exception:
        return None
    return None


# -----------------------------
# Navegação
# -----------------------------
_logo_path = _find_logo_path()
if _logo_path:
    st.sidebar.image(_logo_path, width=140)

page = st.sidebar.radio("Navigation", [
    "Dashboard", "Breweries", "Ingredients", "Products", "Orders", "Purchases", 
    "Recipes", "Production", "Calendar"
], key="page", on_change=_on_page_change)
st.sidebar.markdown("---")
if "_last_page" not in st.session_state:
    st.session_state["_last_page"] = page
st.sidebar.info(f"👤 Role: {st.session_state.get('auth_role','viewer')} | Mode: {'Visualization' if is_visualization_mode() else 'Edit'}")

# -----------------------------
# Dashboard Page
# -----------------------------
if page == "Dashboard":
    logo_path = _find_logo_path()
    if logo_path:
        col_logo, col_title = st.columns([1, 6])
        with col_logo:
            st.image(logo_path, width=90)
        with col_title:
            st.title("🏭 Brewery Dashboard")
    else:
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
        # Hover-only calendar (no click selection)

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
                        # Build a single HTML cell so events render INSIDE the day box
                        event_badges = []

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
                                "Packaging": "#ff9800",
                                "Cleaning": "#f59e0b",
                                "Maintenance": "#ef4444",
                                "Meeting": "#9c27b0",
                                "Other": "#757575",
                            }.get(event.get("event_type", "Other"), "#757575")

                            full_title = str(event.get("title", "Event") or "Event")
                            short_title = full_title
                            if len(full_title) > 25:
                                short_title = full_title[:22] + "..."

                            ev_type = str(event.get("event_type", "") or "")
                            equipment = str(event.get("equipment", "") or "")
                            notes = str(event.get("notes", "") or "").strip()
                            if len(notes) > 260:
                                notes = notes[:260] + "..."

                            tooltip_parts = [
                                f"<div class='tt-title'>{html.escape(full_title)}</div>"
                            ]
                            meta = " • ".join([p for p in [ev_type, current_date.strftime('%Y-%m-%d')] if p])
                            if meta:
                                tooltip_parts.append(f"<div class='tt-meta'>{html.escape(meta)}</div>")
                            if equipment:
                                tooltip_parts.append(f"<div class='tt-line'><b>Equipment:</b> {html.escape(equipment)}</div>")
                            if notes:
                                tooltip_parts.append(f"<div class='tt-line'>{html.escape(notes)}</div>")

                            tooltip_html = "".join(tooltip_parts)

                            event_badges.append(
                                f"<div class='calendar-event' style='background-color:{event_color};'>"
                                f"{html.escape(short_title)}"
                                f"<div class='event-tooltip'>{tooltip_html}</div>"
                                f"</div>"
                            )

                        badges_html = "".join(event_badges)
                        cell_html = f"""
  <div class="{day_class}">
    <div style="font-weight:600; margin-bottom:4px;">{day}</div>
    {badges_html}
  </div>
"""
                        st.markdown(cell_html, unsafe_allow_html=True)
        
        # Add new event
        with st.expander("➕ Add New Event"):
            with st.form("add_event_form", clear_on_submit=True):
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    event_title = st.text_input("Event Title")
                    event_type = st.selectbox(
                        "Event Type",
                        ["Brewing", "Fermentation", "Packaging", "Cleaning", "Maintenance", "Meeting", "Other"],
                    )
                with col_e2:
                    event_date = st.date_input("Event Date", today)
                    equipment = st.text_input("Equipment (Optional)")
                event_notes = st.text_area("Notes")

                repeat = st.radio(
                    "Repeat",
                    ["No repeat", "Daily", "Weekly", "Monthly"],
                    horizontal=True,
                    index=0,
                )
                repeat_until = None
                if repeat != "No repeat":
                    default_until = event_date + timedelta(days=30 if repeat in ("Daily", "Weekly") else 90)
                    repeat_until = st.date_input("Repeat until", default_until)
                    st.caption("Creates multiple events up to the selected date.")

                submitted = st.form_submit_button("Add Event", type="primary", use_container_width=True)

            if submitted and event_title:
                def _add_month(d: date) -> date:
                    """Return the same day in the next month (or last valid day)."""
                    y = d.year + (d.month // 12)
                    m = (d.month % 12) + 1
                    last_day = calendar.monthrange(y, m)[1]
                    return date(y, m, min(d.day, last_day))

                dates_to_create = [event_date]
                if repeat != "No repeat" and repeat_until and repeat_until >= event_date:
                    max_occurrences = 400
                    current = event_date
                    while len(dates_to_create) < max_occurrences:
                        if repeat == "Daily":
                            current = current + timedelta(days=1)
                        elif repeat == "Weekly":
                            current = current + timedelta(weeks=1)
                        else:  # Monthly
                            current = _add_month(current)
                        if current > repeat_until:
                            break
                        dates_to_create.append(current)

                base_event = {
                    "title": event_title,
                    "event_type": event_type,
                    "end_date": None,
                    "equipment": equipment,
                    "batch_id": "",
                    "notes": event_notes,
                    "created_by": "User",
                }

                for d in dates_to_create:
                    new_event = dict(base_event)
                    new_event["start_date"] = d
                    new_event["end_date"] = d
                    insert_data("calendar_events", new_event)

                data = get_all_data()
                if len(dates_to_create) == 1:
                    st.success("Event added!")
                else:
                    st.success(f"{len(dates_to_create)} events added!")
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
                <div class="{alert_class}" style="color:#000; font-weight:700;">
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
                brewery_id = brewery.get('id_brewery', brewery.get('brewery_id', brewery.get('id')))
                if brewery_id is None:
                    brewery_id = _
                brewery_id_str = str(brewery_id)
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
                            eq_count = int(equipment_counts.get(brewery_id, equipment_counts.get(str(brewery_id), 0)))
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
                        if st.button("Edit", key=f"edit_{brewery_id_str}", use_container_width=True):
                            st.session_state['edit_brewery'] = brewery_id
                    with col_delete:
                        if st.button("🗑️ Delete", key=f"del_{brewery_id_str}", use_container_width=True, type="secondary"):
                            st.session_state.delete_confirmation = {"type": "brewery", "id": brewery_id, "name": brewery['name']}
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
                brewery_name_col = _col(breweries_df, 'name', 'brewery_name')
                brewery_id_col = _col(breweries_df, 'id_brewery', 'brewery_id', 'id')

                if not brewery_name_col:
                    st.error("Breweries table is missing a name column.")
                    st.stop()

                brewery_options = breweries_df[brewery_name_col].astype(str).tolist()

                # If we arrived here from a Brewery card ("View Equipment"), preselect that brewery.
                preferred_id = st.session_state.get("selected_brewery")
                if preferred_id is not None and brewery_id_col and preferred_id in set(breweries_df[brewery_id_col].astype(str).values):
                    preferred_name = breweries_df.loc[breweries_df[brewery_id_col].astype(str) == str(preferred_id), brewery_name_col].iloc[0]
                    st.session_state["equipment_brewery_select"] = str(preferred_name)

                selected_brewery = st.selectbox(
                    "Select Brewery for Equipment*",
                    brewery_options,
                    key="equipment_brewery_select",
                )

                match = breweries_df[breweries_df[brewery_name_col].astype(str) == str(selected_brewery)]
                brewery_id = match[brewery_id_col].iloc[0] if (brewery_id_col and not match.empty) else None
                if brewery_id is None:
                    st.error("Selected brewery not found in the database (missing/unknown brewery id column).")
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

                # Live occupancy (derived from active batches assigned to vessels)
                batches_live = data.get("production_batches", pd.DataFrame())
                vcol = _col(batches_live, "current_vessel", "vessel", "fermenter", "tank")
                scol = _col(batches_live, "status")
                volcol = _col(
                    batches_live,
                    "brewed_volume_l", "volume_brewed_l", "actual_volume_l",
                    "planned_volume_l", "batch_volume_l", "batch_size", "batch_volume"
                )
                codecol = _col(batches_live, "batch_code", "code", "batch")

                occ_map: dict[str, dict] = {}
                if batches_live is not None and not batches_live.empty and vcol:
                    for _, br in batches_live.iterrows():
                        if scol and not _is_active_status(br.get(scol)):
                            continue
                        vname = str(br.get(vcol) or "").strip()
                        if not vname:
                            continue
                        vol = 0.0
                        if volcol:
                            try:
                                vol = float(br.get(volcol) or 0.0)
                            except Exception:
                                vol = 0.0
                        bcode = ""
                        if codecol:
                            bcode = str(br.get(codecol) or "").strip()
                        if not bcode:
                            bcode = str(br.get("id_batch") or "").strip()
                        if vname not in occ_map:
                            occ_map[vname] = {"volume": 0.0, "batches": []}
                        occ_map[vname]["volume"] += vol
                        if bcode:
                            occ_map[vname]["batches"].append(bcode)

                # Normalize capacities to liters (handles 'hL' etc.)
                try:
                    equipment_df["capacity_liters"] = equipment_df["capacity_liters"].apply(lambda x: _capacity_to_l(x))
                except Exception:
                    pass

                # Override status/current volume for display based on active batches
                if "name" in equipment_df.columns:
                    equipment_df["active_batches"] = equipment_df["name"].apply(
                        lambda n: ", ".join(occ_map.get(str(n).strip(), {}).get("batches", []))
                    )
                    equipment_df["current_volume"] = equipment_df.apply(
                        lambda r: float(occ_map.get(str(r.get("name")).strip(), {}).get("volume", r.get("current_volume", 0.0)) or 0.0),
                        axis=1
                    )
                    equipment_df["status"] = equipment_df.apply(
                        lambda r: "In Use" if str(r.get("name")).strip() in occ_map else str(r.get("status") or "Empty"),
                        axis=1
                    )
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
                    brewery_name_col = _col(breweries_df, "name", "brewery_name")
                    brewery_id_col = _col(breweries_df, "id_brewery", "brewery_id", "id")
                    brewery_id_filter = None
                    if brewery_name_col and brewery_id_col:
                        m = breweries_df[breweries_df[brewery_name_col].astype(str) == str(brewery_filter)]
                        brewery_id_filter = m[brewery_id_col].iloc[0] if not m.empty else None
                    filtered_equipment = filtered_equipment[filtered_equipment["brewery_id"] == brewery_id_filter]
                
                if len(filtered_equipment) > 0:
                    # Mostrar em cards
                    cols = st.columns(2)
                    for idx, (_, eq) in enumerate(filtered_equipment.iterrows()):
                        with cols[idx % 2]:
                            brewery_name = ""
                            brewery_name_col = _col(breweries_df, "name", "brewery_name")
                            brewery_id_col = _col(breweries_df, "id_brewery", "brewery_id", "id")
                            if (not breweries_df.empty) and brewery_name_col and brewery_id_col and str(eq.get("brewery_id")) in set(breweries_df[brewery_id_col].astype(str).values):
                                brewery_name = breweries_df.loc[breweries_df[brewery_id_col].astype(str) == str(eq.get("brewery_id")), brewery_name_col].iloc[0]
                            
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
                                    <p><strong>Active batch(es):</strong> {eq.get('active_batches','') or '—'}</p>
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
                    brewery_id_col = _col(breweries_df, "id_brewery", "brewery_id", "id")
                    b_id = brewery.get(brewery_id_col) if brewery_id_col else brewery.get("id_brewery")
                    brewery_eq = equipment_df[equipment_df["brewery_id"].astype(str) == str(b_id)]
                    
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
                    brewery_id_col = _col(breweries_df, "id_brewery", "brewery_id", "id")
                    b_id = brewery.get(brewery_id_col) if brewery_id_col else brewery.get("id_brewery")
                    brewery_eq = equipment_df[equipment_df["brewery_id"].astype(str) == str(b_id)]
                    
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
            
            # (Dashboard) maintenance reminders disabled
        else:
            st.info("Add breweries and equipment to see overview statistics")
        
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Ingredients Page
# -----------------------------
elif page == "Ingredients":
    st.title("🌾 Ingredients Management")
    
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
            # Effective unit cost respects manual overrides (if enabled)
            if 'effective_unit_cost' not in filtered_ingredients.columns:
                try:
                    filtered_ingredients['effective_unit_cost'] = filtered_ingredients.apply(_ingredient_effective_unit_cost, axis=1)
                except Exception:
                    filtered_ingredients['effective_unit_cost'] = filtered_ingredients.get('unit_cost', 0)

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
                total_stock_value = (filtered_ingredients["stock"] * filtered_ingredients.get("effective_unit_cost", filtered_ingredients["unit_cost"])).sum()
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
            # Total value uses the effective unit cost (manual override if set)
            if 'effective_unit_cost' not in display_df.columns:
                try:
                    display_df['effective_unit_cost'] = display_df.apply(_ingredient_effective_unit_cost, axis=1)
                except Exception:
                    display_df['effective_unit_cost'] = display_df.get('unit_cost', 0)
            display_df["Total Value"] = display_df["stock"] * display_df['effective_unit_cost']
            
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
                "unit_cost": "Calculated Unit Cost",
                "effective_unit_cost": "Effective Unit Cost",
                "low_stock_threshold": "Low Stock Threshold",
                "Total Value": "Total Value",
                "Status": "Status"
            }
            
            display_df = display_df[display_cols].rename(columns=column_mapping)
            
            # Formatar valores
            display_df["Calculated Unit Cost"] = display_df["Calculated Unit Cost"].apply(lambda x: f"${float(str(x).replace('$','') or 0):.4f}") if "Calculated Unit Cost" in display_df.columns else display_df.get("Calculated Unit Cost")
            display_df["Effective Unit Cost"] = display_df["Effective Unit Cost"].apply(lambda x: f"${float(str(x).replace('$','') or 0):.4f}") if "Effective Unit Cost" in display_df.columns else display_df.get("Effective Unit Cost")
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
                        "Fermentable",
                        "Hops",
                        "Yeast",
                        "Packaging",
                        "Process",
                        "Spice",
                        "Fruit",
                        "Other",
                    ]
                    category = st.selectbox("Category*", category_options, key="new_ing_category")

                    unit_options = ["kg", "g", "lb", "oz", "L", "mL", "pkg", "unit"]
                    unit = st.selectbox("Unit*", unit_options, key="new_ing_unit")

                    stock = st.number_input(
                        "Initial Stock*", min_value=0.0, value=0.0, step=0.1, key="new_ing_stock"
                    )

                    opening_stock = st.checkbox(
                        "Opening stock (I already have this item in inventory)",
                        value=False,
                        help="Enable this if you're registering an ingredient that is already in stock and you want to set an initial unit cost baseline before the first purchase entry.",
                        key="new_ing_opening_stock",
                    )

                    opening_unit_cost = st.number_input(
                        "Unit cost for opening stock",
                        min_value=0.0,
                        value=0.0,
                        step=0.01,
                        help="Only used if Opening stock is checked. Cost per unit (same unit as the ingredient unit).",
                        key="new_ing_opening_unit_cost",
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
                            # Prices/costs are calculated from Purchases (incl. freight allocation).
                            # If user is setting opening stock, allow an initial cost baseline.
                            "unit_cost": float(opening_unit_cost) if opening_stock else 0.0,
                            "cost_per_unit": float(opening_unit_cost) if opening_stock else 0.0,
                            "use_manual_cost": 0,
                            "unit_cost_manual": None,
                            "manual_cost_notes": None,
                            "manual_cost_updated": None,
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

                        inserted_id = insert_data("ingredients", new_ingredient)

                        # Log opening stock as a stock movement (so Stock History is complete)
                        try:
                            if float(opening_stock or 0) > 0:
                                log_stock_movement(
                                    ingredient_name=str(ing_name),
                                    delta_qty=float(opening_stock),
                                    unit=str(unit),
                                    reason="Opening Stock",
                                    source="Opening Balance",
                                    destination="Inventory",
                                    ref_table="ingredients",
                                    ref_id=int(inserted_id) if inserted_id is not None else None,
                                    notes="Initial stock when ingredient was created",
                                )
                        except Exception:
                            pass

                        # If this ingredient is being registered with opening stock, persist the initial unit cost baseline.
                        # Some deployments have DB defaults/triggers that can overwrite inserted values; this explicit update
                        # ensures the cost is reflected immediately in the stock view.
                        if opening_stock and float(opening_unit_cost or 0.0) > 0:
                            c0 = float(opening_unit_cost)
                            # Use raw SQL updates to avoid any stale cached-schema filtering.
                            try:
                                execute_query(
                                    "UPDATE ingredients SET unit_cost = :c, last_updated = CURRENT_TIMESTAMP WHERE name = :n",
                                    {"c": c0, "n": ing_name},
                                )
                            except Exception:
                                pass
                            # Best-effort: keep legacy alias in sync if present
                            try:
                                execute_query(
                                    "UPDATE ingredients SET cost_per_unit = :c WHERE name = :n",
                                    {"c": c0, "n": ing_name},
                                )
                            except Exception:
                                pass

                        data = get_all_data()
                        st.success(f"✅ Ingredient '{ing_name}' added successfully!")
                        st.rerun()
        
        else:  # Edit Existing Ingredient
            ingredients_df = data.get("ingredients", pd.DataFrame())
            if not ingredients_df.empty:
                ing_label_map = _ingredient_name_label_map(ingredients_df)
                ingredient_options = ingredients_df["name"].tolist()
                selected_ingredient = st.selectbox(
                    "Select Ingredient to Edit",
                    ingredient_options,
                    format_func=lambda x: ing_label_map.get(str(x), str(x)),
                    key="edit_ing_select"
                )
                
                if selected_ingredient:
                    # Obter dados atuais
                    ing_data = ingredients_df[ingredients_df["name"] == selected_ingredient].iloc[0]
                    

                    # Primary key column (supports legacy schemas)
                    ing_pk_col = _col(ingredients_df, 'id_ingredient', 'id')
                    ing_pk_val = ing_data.get(ing_pk_col) if ing_pk_col else None
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


                        # Manual cost override (useful to correct historical wrong costs)
                        try:
                            use_manual0 = int(float(ing_data.get("use_manual_cost", 0) or 0))
                        except Exception:
                            use_manual0 = 0
                        try:
                            manual_cost0 = float(ing_data.get("unit_cost_manual", 0) or 0)
                        except Exception:
                            manual_cost0 = 0.0
                        manual_note0 = str(ing_data.get("manual_cost_notes", "") or "")

                        use_manual_new = st.checkbox(
                            "Use manual unit cost override",
                            value=bool(use_manual0),
                            key="edit_ing_use_manual_cost",
                            help="If enabled, this value will be used for recipe costing and stock valuation. Purchases will still update the calculated cost in the background.",
                        )

                        if use_manual_new:
                            default_cost = manual_cost0 if manual_cost0 > 0 else current_cost
                            manual_cost_new = st.number_input(
                                "Manual unit cost",
                                min_value=0.0,
                                value=float(default_cost),
                                step=0.01,
                                format="%.4f",
                                key="edit_ing_manual_cost",
                            )
                            manual_note_new = st.text_input(
                                "Manual cost note (optional)",
                                value=manual_note0,
                                key="edit_ing_manual_note",
                            )
                        else:
                            manual_cost_new = 0.0
                            manual_note_new = manual_note0

                        eff_cost_preview = float(manual_cost_new) if (use_manual_new and float(manual_cost_new or 0) > 0) else float(current_cost or 0)
                        st.caption(f"Effective unit cost used in calculations: **${eff_cost_preview:,.4f}**")

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
                                "use_manual_cost": 1 if bool(use_manual_new) else 0,
                                "unit_cost_manual": float(manual_cost_new) if bool(use_manual_new) and float(manual_cost_new or 0) > 0 else None,
                                "manual_cost_notes": str(manual_note_new).strip() if bool(use_manual_new) and str(manual_note_new).strip() else None,
                                "manual_cost_updated": pd.Timestamp.utcnow() if bool(use_manual_new) else None,
                                "alpha_acid": new_alpha,
                                "lot_number": new_lot,
                                "notes": new_notes
                            }
                            
                            if 'new_expiry' in locals():
                                updates["expiry_date"] = new_expiry if new_expiry else None
                            
                            # Log stock delta if user changed stock manually
                            try:
                                old_stock_val = float(ing_data.get("stock", 0) or 0)
                                delta_stock = float(new_stock) - old_stock_val
                                if abs(delta_stock) > 1e-12:
                                    log_stock_movement(
                                        ingredient_name=str(new_name),
                                        delta_qty=float(delta_stock),
                                        unit=str(new_unit),
                                        reason="Manual Stock Edit",
                                        source="Manual" if delta_stock > 0 else "Inventory",
                                        destination="Inventory" if delta_stock > 0 else "Manual",
                                        ref_table="ingredients",
                                        ref_id=int(ing_pk_val or 0),
                                        notes="Edited stock in Ingredients tab",
                                    )
                            except Exception:
                                pass

                            update_data("ingredients", updates, f"{ing_pk_col} = :id" if ing_pk_col else "id_ingredient = :id", {"id": ing_pk_val})
                            data = get_all_data()
                            st.success(f"✅ Ingredient '{new_name}' updated successfully!")
                            st.rerun()
                    
                    with col_btn2:
                        if st.button("🗑️ Delete Ingredient", use_container_width=True, type="secondary", key="delete_ing_btn"):
                            # Verificar se o ingrediente está em uso em receitas
                            in_use = check_ingredient_usage(ing_pk_val)
                            
                            if in_use:
                                st.error("Cannot delete ingredient that is used in recipes! Remove it from recipes first.")
                            else:
                                st.session_state.delete_confirmation = {"type": "ingredient", "id": ing_pk_val, "name": selected_ingredient}
                                st.rerun()
                    
                    with col_btn3:
                        if st.button("🔄 Reset Stock", use_container_width=True, key="reset_stock_btn"):
                            # Apenas resetar o estoque para 0
                            try:
                                old_stock_val = float(ing_data.get("stock", 0) or 0)
                                if old_stock_val != 0:
                                    log_stock_movement(
                                        ingredient_name=str(selected_ingredient),
                                        delta_qty=-float(old_stock_val),
                                        unit=str(ing_data.get("unit") or ""),
                                        reason="Reset Stock",
                                        source="Inventory",
                                        destination="Manual",
                                        ref_table="ingredients",
                                        ref_id=int(ing_pk_val or 0),
                                        notes="Reset stock to 0 in Ingredients tab",
                                    )
                            except Exception:
                                pass
                            update_data("ingredients", {"stock": 0}, f"{ing_pk_col} = :id" if ing_pk_col else "id_ingredient = :id", {"id": ing_pk_val})
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
            # Ensure required columns exist and compute effective cost/value before aggregations
            ingredients_df = ingredients_df.copy()
            if "category" not in ingredients_df.columns:
                ingredients_df["category"] = "Uncategorized"

            # Unit cost used everywhere (manual override > calculated)
            try:
                ingredients_df["effective_unit_cost"] = ingredients_df.apply(_ingredient_effective_unit_cost, axis=1)
            except Exception:
                # Fallback if something goes sideways (should be rare)
                base = ingredients_df["unit_cost"] if "unit_cost" in ingredients_df.columns else (
                    ingredients_df["cost_per_unit"] if "cost_per_unit" in ingredients_df.columns else 0.0
                )
                ingredients_df["effective_unit_cost"] = base

            ingredients_df["effective_unit_cost"] = pd.to_numeric(ingredients_df["effective_unit_cost"], errors="coerce").fillna(0.0)
            ingredients_df["stock"] = pd.to_numeric(ingredients_df.get("stock", 0.0), errors="coerce").fillna(0.0)
            ingredients_df["total_value"] = ingredients_df["stock"] * ingredients_df["effective_unit_cost"]

            # Análise por categoria (safe even if a column was missing before)
            category_analysis = (
                ingredients_df.groupby("category", dropna=False)
                .agg(
                    **{
                        "Item Count": ("name", "count"),
                        "Total Stock": ("stock", "sum"),
                        "Avg Unit Cost": ("effective_unit_cost", "mean"),
                        "Total Value": ("total_value", "sum"),
                    }
                )
                .reset_index()
                .rename(columns={"category": "Category"})
            )

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
            ingredients_df["effective_unit_cost"] = ingredients_df.apply(_ingredient_effective_unit_cost, axis=1)
            ingredients_df["total_value"] = ingredients_df["stock"] * ingredients_df["effective_unit_cost"]
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

        movements_df = data.get("stock_movements", pd.DataFrame())

        if movements_df is None or movements_df.empty:
            st.info("No stock movements logged yet. Movements will appear here after Purchases, Brews, Dry Hops, Kegging, manual stock edits, etc.")
            # Legacy fallback: show purchases if available
            purchases_df = data.get("purchases", pd.DataFrame())
            if purchases_df is not None and not purchases_df.empty:
                st.caption("Legacy purchases table (fallback):")
                st.dataframe(purchases_df.sort_values(by=purchases_df.columns[0], ascending=False), use_container_width=True)
        else:
            # Normalize columns
            dcol = _col(movements_df, "movement_date", "date", "created_date")
            ingcol = _col(movements_df, "ingredient_name", "ingredient")
            dircol = _col(movements_df, "direction", "type", "in_out")
            qtycol = _col(movements_df, "delta_qty", "qty", "quantity", "delta")
            unitcol = _col(movements_df, "unit")
            reasoncol = _col(movements_df, "reason")
            srccol = _col(movements_df, "source", "from_location", "from")
            dstcol = _col(movements_df, "destination", "to_location", "to")
            batchcol = _col(movements_df, "batch_id")
            orderncol = _col(movements_df, "order_number", "order_no")
            bycol = _col(movements_df, "created_by", "recorded_by", "user")
            notescol = _col(movements_df, "notes")

            dfh = movements_df.copy()

            # Parse date for sorting/filtering
            if dcol and dcol in dfh.columns:
                dfh[dcol] = pd.to_datetime(dfh[dcol], errors="coerce")

            with st.expander("Filters", expanded=False):
                c1, c2, c3 = st.columns(3)
                with c1:
                    ing_opts = sorted(dfh[ingcol].dropna().astype(str).unique().tolist()) if ingcol else []
                    sel_ings = st.multiselect("Ingredient", ing_opts, default=[], key="stock_hist_filter_ing")
                with c2:
                    dir_opts = ["All", "IN", "OUT"]
                    sel_dir = st.selectbox("Direction", dir_opts, index=0, key="stock_hist_filter_dir")
                with c3:
                    if dcol and dfh[dcol].notna().any():
                        min_d = dfh[dcol].min().date()
                        max_d = dfh[dcol].max().date()
                    else:
                        min_d = date.today() - timedelta(days=30)
                        max_d = date.today()
                    dr = st.date_input("Date range", value=(min_d, max_d), key="stock_hist_filter_dates")

            # Apply filters
            if sel_ings and ingcol:
                dfh = dfh[dfh[ingcol].astype(str).isin([str(x) for x in sel_ings])]

            if sel_dir != "All" and dircol:
                dfh = dfh[dfh[dircol].astype(str).str.upper() == sel_dir]

            try:
                if dcol and isinstance(dr, (tuple, list)) and len(dr) == 2:
                    d0, d1 = dr
                    dfh = dfh[(dfh[dcol] >= pd.Timestamp(d0)) & (dfh[dcol] <= pd.Timestamp(d1) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]
            except Exception:
                pass


            # --- Admin tools: manual ADJUSTMENT + delete movement ---
            idcol = _col(dfh, "id_stock_move", "id", "id_movement")

            if can_write():
                with st.expander("🧾 Manual stock movement: ADJUSTMENT (IN/OUT)", expanded=False):
                    ingredients_df2 = data.get("ingredients", pd.DataFrame())
                    ncol2 = _col(ingredients_df2, "name") if ingredients_df2 is not None else None
                    if ingredients_df2 is None or ingredients_df2.empty or not ncol2:
                        st.info("No ingredients found.")
                    else:
                        ing_label_map2 = _ingredient_name_label_map(ingredients_df2)
                        ing_list = ingredients_df2[ncol2].dropna().astype(str).tolist()

                        with st.form("stock_manual_adjustment_form", clear_on_submit=True):
                            c1a, c2a, c3a = st.columns(3)
                            with c1a:
                                sel_ing = st.selectbox(
                                    "Ingredient",
                                    ing_list,
                                    format_func=lambda x: ing_label_map2.get(str(x), str(x)),
                                )
                            with c2a:
                                direction_adj = st.selectbox("Direction", ["IN", "OUT"], index=0)
                            with c3a:
                                qty_adj = st.number_input(
                                    "Quantity",
                                    min_value=0.0,
                                    value=0.0,
                                    step=0.1,
                                    format="%.3f",
                                )
                            d_adj = st.date_input("Date", date.today())
                            notes_adj = st.text_area("Notes", height=80)
                            submit_adj = st.form_submit_button(
                                "Record ADJUSTMENT", type="primary", use_container_width=True
                            )

                        if submit_adj:
                            require_admin_action()
                            if (not sel_ing) or (float(qty_adj or 0) <= 0):
                                st.error("Select an ingredient and set a quantity > 0.")
                            else:
                                delta = float(qty_adj) if direction_adj == "IN" else -float(qty_adj)
                                src = "ADJUSTMENT" if delta > 0 else "Inventory"
                                dst = "Inventory" if delta > 0 else "ADJUSTMENT"
                                adjust_stock_for_ingredient(
                                    get_all_data(),
                                    str(sel_ing),
                                    float(delta),
                                    reason="ADJUSTMENT",
                                    source=src,
                                    destination=dst,
                                    ref_table="manual",
                                    notes=str(notes_adj or ""),
                                    movement_date=pd.Timestamp(d_adj),
                                )
                                st.success("✅ ADJUSTMENT recorded.")
                                st.rerun()

                with st.expander("🗑️ Delete a stock movement (admin)", expanded=False):
                    if not idcol or idcol not in dfh.columns:
                        st.warning(
                            "This stock_movements table has no id column loaded, so delete-by-id is unavailable."
                        )
                    else:
                        # Use the currently filtered view (max 500 rows)
                        try:
                            pick = dfh.copy()
                            if dcol and dcol in pick.columns:
                                pick = pick.sort_values(by=dcol, ascending=False)
                            pick = pick.head(500)
                        except Exception:
                            pick = dfh.head(500)

                        recs = pick.to_dict("records")

                        def _mv_label(r):
                            _id = r.get(idcol, "")
                            _date = ""
                            try:
                                if dcol and r.get(dcol) is not None and str(r.get(dcol)) != "nan":
                                    _date = str(pd.to_datetime(r.get(dcol), errors="coerce"))[:10]
                            except Exception:
                                _date = str(r.get(dcol) or "")[:10] if dcol else ""
                            _ing = str(r.get(ingcol) or "")
                            _dir = str(r.get(dircol) or "").upper()
                            _qty = r.get(qtycol) if qtycol else ""
                            try:
                                _qty = f"{float(_qty):g}"
                            except Exception:
                                _qty = str(_qty)
                            _unit = str(r.get(unitcol) or "")
                            _reason = str(r.get(reasoncol) or "")
                            return f"#{_id} • {_date} • {_ing} • {_dir} {_qty} {_unit} • {_reason}"

                        sel_mv = st.selectbox(
                            "Select movement",
                            recs,
                            format_func=_mv_label,
                            key="stock_hist_delete_sel",
                        )
                        also_reverse = st.checkbox(
                            "Also reverse its inventory impact",
                            value=True,
                            key="stock_hist_delete_reverse",
                        )

                        confirm_key = "stock_hist_delete_confirm"
                        if st.button(
                            "🗑️ Delete selected movement",
                            type="secondary",
                            use_container_width=True,
                            key="stock_hist_delete_btn",
                        ):
                            st.session_state[confirm_key] = True

                        if st.session_state.get(confirm_key, False):
                            st.warning(
                                "This will permanently delete the selected movement. "
                                "If it came from a Purchase Order or Production event, deleting it won't delete the original transaction."
                            )
                            cdel1, cdel2 = st.columns(2)
                            with cdel1:
                                if st.button(
                                    "✅ Yes, delete it",
                                    type="primary",
                                    use_container_width=True,
                                    key="stock_hist_delete_yes",
                                ):
                                    require_admin_action()
                                    mv_id = sel_mv.get(idcol)
                                    try:
                                        mv_id_int = int(mv_id)
                                    except Exception:
                                        mv_id_int = None

                                    if mv_id_int is None:
                                        st.error("Invalid movement id.")
                                    else:
                                        ok, msg = delete_stock_movement_fast(
                                            mv_id_int, also_reverse_stock=bool(also_reverse)
                                        )
                                        if ok:
                                            st.success("✅ " + str(msg))
                                            st.session_state[confirm_key] = False
                                            st.rerun()
                                        else:
                                            st.error(str(msg))
                            with cdel2:
                                if st.button(
                                    "Cancel",
                                    use_container_width=True,
                                    key="stock_hist_delete_cancel",
                                ):
                                    st.session_state[confirm_key] = False
                                    st.rerun()
            else:
                st.info("Admin edit mode required to record ADJUSTMENT or delete movements.")


            # Display (friendly column order)
            show_cols = []
            for c in [dcol, ingcol, dircol, qtycol, unitcol, reasoncol, srccol, dstcol, batchcol, orderncol, bycol, notescol]:
                if c and c in dfh.columns and c not in show_cols:
                    show_cols.append(c)

            if dcol:
                dfh = dfh.sort_values(by=dcol, ascending=False)

            st.dataframe(dfh[show_cols] if show_cols else dfh, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Purchases Page
# -----------------------------
elif page == "Products":
    st.title("📦 Products")
    st.caption("Define composite products (finished goods).")

    data = get_all_data()
    recipes_df = data.get('recipes', pd.DataFrame())
    ingredients_df = data.get('ingredients', pd.DataFrame())
    composites_df = data.get('composite_products', pd.DataFrame())
    composite_items_df = data.get('composite_product_items', pd.DataFrame())
    composite_inv_df = data.get('composite_inventory', pd.DataFrame())

    # Helpers
    recipe_name_col = _col(recipes_df, 'name')
    recipe_id_col = _col(recipes_df, 'id_recipe', 'recipe_id', 'id')
    ing_name_col = _col(ingredients_df, 'name')
    ing_cat_col = _col(ingredients_df, 'category')

    tab1 = st.tabs(["Composite Products"])[0]

    with tab1:
        st.subheader("Composite Products")

        if is_admin():
            with st.form('create_composite_product', clear_on_submit=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    comp_name = st.text_input("Product name", placeholder="e.g., Beer A — 30L KeyKeg")
                    output_unit = st.text_input("Output unit", value="keg")
                with c2:
                    recipe_options = []
                    if recipes_df is not None and not recipes_df.empty and recipe_name_col:
                        recipe_options = recipes_df[recipe_name_col].astype(str).tolist()
                    recipe_name = st.selectbox("Beer / Recipe", recipe_options, index=0 if recipe_options else None)
                    beer_l_per_unit = st.number_input("Beer (L) per unit", min_value=0.0, value=30.0, step=1.0)
                with c3:
                    notes = st.text_area("Notes", height=90)

                st.markdown("**Packaging / other components (per 1 unit)**")
                packaging_lines = []
                ing_options = ingredients_df[ing_name_col].astype(str).tolist() if ingredients_df is not None and not ingredients_df.empty and ing_name_col else []
                # Optional: prioritize packaging in dropdown
                if ing_cat_col and ingredients_df is not None and not ingredients_df.empty:
                    pkg = ingredients_df[ingredients_df[ing_cat_col].astype(str).str.lower().str.contains('pack', na=False)]
                    pkg_opts = pkg[ing_name_col].astype(str).tolist() if not pkg.empty else []
                    ing_options = list(dict.fromkeys(pkg_opts + ing_options))

                for i in range(5):
                    cc1, cc2 = st.columns([3, 1])
                    with cc1:
                        ing = st.selectbox(f"Component {i+1}", [''] + ing_options, index=0, key=f"comp_ing_{i}")
                    with cc2:
                        qty = st.number_input("Qty", min_value=0.0, value=0.0, step=1.0, key=f"comp_qty_{i}")
                    if ing and qty > 0:
                        unit, _ = get_ingredient_unit_and_cost(data, ing)
                        packaging_lines.append((ing, float(qty), unit or ''))

                submit = st.form_submit_button("Create composite product", type='primary', use_container_width=True)

            if submit:
                require_admin_action()
                if not comp_name.strip():
                    st.error("Product name is required.")
                elif not recipe_name:
                    st.error("Please select a recipe.")
                elif beer_l_per_unit <= 0:
                    st.error("Beer (L) per unit must be greater than 0.")
                else:
                    # Resolve recipe_id
                    rid = None
                    if recipe_name and recipes_df is not None and not recipes_df.empty and recipe_name_col and recipe_id_col:
                        m = recipes_df[recipes_df[recipe_name_col].astype(str) == str(recipe_name)]
                        if not m.empty:
                            rid = str(m.iloc[0][recipe_id_col])

                    comp_id = insert_data('composite_products', {
                        'name': comp_name.strip(),
                        'recipe_id': rid,
                        'recipe_name': recipe_name,
                        'output_unit': (output_unit or 'unit').strip() or 'unit',
                        'notes': notes,
                        'status': 'Active',
                    })

                    # Beer component
                    insert_data('composite_product_items', {
                        'composite_id': int(comp_id) if comp_id is not None else None,
                        'component_type': 'Beer',
                        'component_name': recipe_name,
                        'quantity': float(beer_l_per_unit),
                        'unit': 'L',
                    })

                    # Packaging components
                    for ing, qty, unit in packaging_lines:
                        insert_data('composite_product_items', {
                            'composite_id': int(comp_id) if comp_id is not None else None,
                            'component_type': 'Ingredient',
                            'component_name': ing,
                            'quantity': float(qty_stock),
                            'unit': unit,
                        })

                    st.success("✅ Composite product created.")
                    st.rerun()

        st.markdown('---')
        if composites_df is None or composites_df.empty:
            st.info("No composite products yet.")
        else:
            cid_col = _col(composites_df, 'id_composite', 'id')
            cname_col = _col(composites_df, 'name')
            out_unit_col = _col(composites_df, 'output_unit')
            rname_col = _col(composites_df, 'recipe_name')

            # Show list + BOM
            for _, c in composites_df.sort_values(cid_col or composites_df.columns[0], ascending=False).iterrows():
                with st.expander(f"{c.get(cname_col,'')}  (#{c.get(cid_col,'')})"):
                    st.write(f"**Beer / Recipe:** {c.get(rname_col,'')}")
                    st.write(f"**Output unit:** {c.get(out_unit_col,'unit')}")
                    if composite_items_df is not None and not composite_items_df.empty and cid_col:
                        icid = _col(composite_items_df, 'composite_id')
                        if icid:
                            bom = composite_items_df[composite_items_df[icid] == c.get(cid_col)]
                            if not bom.empty:
                                st.dataframe(bom[[col for col in bom.columns if col not in ('created_date',)]], use_container_width=True)

elif page == "Orders":
    st.title("🧾 Orders")

    deposits_df = data.get('deposits', pd.DataFrame())
    customers_df = data.get('customers', pd.DataFrame())
    products_df = data.get('composite_products', pd.DataFrame())
    inv_df = data.get('composite_inventory', pd.DataFrame())
    orders_df = data.get('sales_orders', pd.DataFrame())
    items_df = data.get('sales_order_items', pd.DataFrame())

    def _safe_str(x):
        return '' if x is None else str(x)

    def _available_units(composite_id: int, warehouse: str) -> float:
        if inv_df is None or inv_df.empty:
            return 0.0
        cid = _col(inv_df, 'composite_id')
        wcol = _col(inv_df, 'warehouse')
        qcol = _col(inv_df, 'quantity_units')
        if not (cid and wcol and qcol):
            return 0.0
        v = inv_df.copy()
        try:
            v = v[(v[cid].astype(float) == float(composite_id)) & (v[wcol].astype(str) == str(warehouse))]
            return float(v[qcol].sum() or 0)
        except Exception:
            return 0.0

    def _generate_order_pdf(order_row: dict, lines: pd.DataFrame, customer_row: dict | None, deposit_row: dict | None) -> bytes:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import mm

        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        w, h = A4
        y = h - 18*mm

        order_no = _safe_str(order_row.get('order_no') or f"#{order_row.get('id_sales_order','')}")
        status = _safe_str(order_row.get('status'))
        currency = _safe_str(order_row.get('currency') or 'NOK')

        c.setFont('Helvetica-Bold', 15)
        c.drawString(18*mm, y, 'ORDER CONFIRMATION')
        y -= 8*mm
        c.setFont('Helvetica', 10)
        c.drawString(18*mm, y, f"Order: {order_no}    Status: {status}")
        y -= 6*mm
        c.drawString(18*mm, y, f"Order date: {_safe_str(order_row.get('order_date'))}    Delivery: {_safe_str(order_row.get('delivery_date'))}")
        y -= 8*mm

        cust_name = _safe_str(order_row.get('customer_name') or (customer_row or {}).get('name'))
        dep_name = _safe_str(order_row.get('deposit_name') or (deposit_row or {}).get('name'))

        c.setFont('Helvetica-Bold', 11)
        c.drawString(18*mm, y, 'Customer')
        y -= 5*mm
        c.setFont('Helvetica', 10)
        c.drawString(18*mm, y, cust_name)
        y -= 5*mm
        if customer_row:
            if customer_row.get('org_no'):
                c.drawString(18*mm, y, f"Org: {_safe_str(customer_row.get('org_no'))}")
                y -= 5*mm
            if customer_row.get('delivery_address'):
                for line in _safe_str(customer_row.get('delivery_address')).split('\n'):
                    c.drawString(18*mm, y, line[:100])
                    y -= 4.5*mm
            if customer_row.get('email') or customer_row.get('phone'):
                c.drawString(18*mm, y, f"Email: {_safe_str(customer_row.get('email'))}   Phone: {_safe_str(customer_row.get('phone'))}")
                y -= 5*mm
        y -= 2*mm

        c.setFont('Helvetica-Bold', 11)
        c.drawString(18*mm, y, 'Dispatch / Deposit')
        y -= 5*mm
        c.setFont('Helvetica', 10)
        c.drawString(18*mm, y, dep_name)
        y -= 8*mm

        # Table header
        c.setFont('Helvetica-Bold', 10)
        c.drawString(18*mm, y, 'Product')
        c.drawString(120*mm, y, 'Qty')
        c.drawString(145*mm, y, f"Unit ({currency})")
        c.drawString(175*mm, y, f"Total ({currency})")
        y -= 4*mm
        c.line(18*mm, y, 195*mm, y)
        y -= 6*mm

        # Lines
        c.setFont('Helvetica', 10)
        subtotal = 0.0
        if lines is not None and not lines.empty:
            for _, r in lines.iterrows():
                pname = _safe_str(r.get('product_name'))
                qty = float(r.get('quantity') or 0)
                unit_price = float(r.get('unit_price') or 0)
                total = float(r.get('line_total') or (qty * unit_price))
                subtotal += total

                # wrap product name if long
                pname_lines = [pname[i:i+52] for i in range(0, len(pname), 52)] or ['']
                c.drawString(18*mm, y, pname_lines[0])
                c.drawRightString(140*mm, y, f"{qty:g}")
                c.drawRightString(170*mm, y, f"{unit_price:.2f}")
                c.drawRightString(195*mm, y, f"{total:.2f}")
                y -= 5*mm
                for extra in pname_lines[1:]:
                    c.drawString(18*mm, y, extra)
                    y -= 4.5*mm

                if y < 25*mm:
                    c.showPage()
                    y = h - 18*mm
                    c.setFont('Helvetica', 10)

        y -= 2*mm
        c.line(120*mm, y, 195*mm, y)
        y -= 6*mm
        c.setFont('Helvetica-Bold', 11)
        c.drawRightString(170*mm, y, 'Subtotal:')
        c.drawRightString(195*mm, y, f"{subtotal:.2f}")
        y -= 6*mm
        total_val = float(order_row.get('total') or subtotal)
        c.drawRightString(170*mm, y, 'Total:')
        c.drawRightString(195*mm, y, f"{total_val:.2f}")
        y -= 10*mm

        notes = _safe_str(order_row.get('notes'))
        if notes:
            c.setFont('Helvetica-Bold', 10)
            c.drawString(18*mm, y, 'Notes')
            y -= 5*mm
            c.setFont('Helvetica', 10)
            for line in notes.split('\n'):
                c.drawString(18*mm, y, line[:110])
                y -= 4.5*mm
                if y < 20*mm:
                    c.showPage()
                    y = h - 18*mm
                    c.setFont('Helvetica', 10)

        c.showPage()
        c.save()
        buf.seek(0)
        return buf.read()

    tab_dep, tab_cust, tab_new, tab_list = st.tabs([
        "📦 Deposits",
        "👥 Customers",
        "🧾 New Order",
        "📜 Orders",
    ])

    with tab_dep:
        st.subheader("Deposits")
        if not can_write():
            st.info("Guest mode: you can view deposits, but only admin can create/edit.")

        # Create
        with st.form('deposit_create', clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                dep_name = st.text_input('Name*')
                dep_city = st.text_input('City')
            with c2:
                dep_country = st.text_input('Country')
                dep_status = st.selectbox('Status', ['Active', 'Inactive'], index=0)
            dep_address = st.text_area('Address', height=80)
            dep_notes = st.text_area('Notes', height=80)
            submit = st.form_submit_button('Add deposit', type='primary', use_container_width=True)

        if submit:
            require_admin_action()
            if not dep_name.strip():
                st.error('Name is required.')
            else:
                insert_data('deposits', {
                    'name': dep_name.strip(),
                    'address': dep_address,
                    'city': dep_city,
                    'country': dep_country,
                    'status': dep_status,
                    'notes': dep_notes,
                })
                st.success('✅ Deposit added.')
                st.rerun()

        st.markdown('---')
        if deposits_df is None or deposits_df.empty:
            st.info('No deposits yet.')
        else:
            d_id = _col(deposits_df, 'id_deposit', 'id')
            d_name = _col(deposits_df, 'name')
            view = deposits_df.copy()
            cols = [c for c in view.columns if c not in ('created_date',)]
            st.dataframe(view[cols], use_container_width=True)

            if can_write() and d_id and d_name:
                st.markdown('#### Edit deposit')
                recs = view.sort_values(d_name).to_dict('records')
                sel = st.selectbox('Select', recs, format_func=lambda r: f"{r.get(d_name,'')} (#{r.get(d_id,'')})")
                with st.form('deposit_edit', clear_on_submit=False):
                    n = st.text_input('Name', value=_safe_str(sel.get(d_name)))
                    addr = st.text_area('Address', value=_safe_str(sel.get('address')), height=80)
                    city = st.text_input('City', value=_safe_str(sel.get('city')))
                    country = st.text_input('Country', value=_safe_str(sel.get('country')))
                    status = st.selectbox('Status', ['Active','Inactive'], index=0 if _safe_str(sel.get('status','Active'))!='Inactive' else 1)
                    notes = st.text_area('Notes', value=_safe_str(sel.get('notes')), height=80)
                    save = st.form_submit_button('Save changes', type='primary', use_container_width=True)

                if save:
                    require_admin_action()
                    update_data('deposits', {
                        'name': n.strip(),
                        'address': addr,
                        'city': city,
                        'country': country,
                        'status': status,
                        'notes': notes,
                    }, f"{d_id} = :id", {'id': int(sel.get(d_id))})
                    st.success('✅ Updated.')
                    st.rerun()

    with tab_cust:
        st.subheader("Customers")
        if not can_write():
            st.info("Guest mode: you can view customers, but only admin can create/edit.")

        with st.form('customer_create', clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                cust_name = st.text_input('Name*')
                cust_org = st.text_input('Org no')
            with c2:
                cust_email = st.text_input('Email')
                cust_phone = st.text_input('Phone')
            with c3:
                cust_status = st.selectbox('Status', ['Active','Inactive'], index=0)
            bill = st.text_area('Billing address', height=80)
            deliver = st.text_area('Delivery address', height=80)
            notes = st.text_area('Notes', height=80)
            submit = st.form_submit_button('Add customer', type='primary', use_container_width=True)

        if submit:
            require_admin_action()
            if not cust_name.strip():
                st.error('Name is required.')
            else:
                insert_data('customers', {
                    'name': cust_name.strip(),
                    'org_no': cust_org,
                    'email': cust_email,
                    'phone': cust_phone,
                    'billing_address': bill,
                    'delivery_address': deliver,
                    'status': cust_status,
                    'notes': notes,
                })
                st.success('✅ Customer added.')
                st.rerun()

        st.markdown('---')
        if customers_df is None or customers_df.empty:
            st.info('No customers yet.')
        else:
            c_id = _col(customers_df, 'id_customer', 'id')
            c_name = _col(customers_df, 'name')
            view = customers_df.copy()
            cols = [c for c in view.columns if c not in ('created_date',)]
            st.dataframe(view[cols], use_container_width=True)

            if can_write() and c_id and c_name:
                st.markdown('#### Edit customer')
                recs = view.sort_values(c_name).to_dict('records')
                sel = st.selectbox('Select', recs, format_func=lambda r: f"{r.get(c_name,'')} (#{r.get(c_id,'')})")
                with st.form('customer_edit', clear_on_submit=False):
                    n = st.text_input('Name', value=_safe_str(sel.get(c_name)))
                    org = st.text_input('Org no', value=_safe_str(sel.get('org_no')))
                    email = st.text_input('Email', value=_safe_str(sel.get('email')))
                    phone = st.text_input('Phone', value=_safe_str(sel.get('phone')))
                    status = st.selectbox('Status', ['Active','Inactive'], index=0 if _safe_str(sel.get('status','Active'))!='Inactive' else 1)
                    bill = st.text_area('Billing address', value=_safe_str(sel.get('billing_address')), height=80)
                    deliver = st.text_area('Delivery address', value=_safe_str(sel.get('delivery_address')), height=80)
                    notes = st.text_area('Notes', value=_safe_str(sel.get('notes')), height=80)
                    save = st.form_submit_button('Save changes', type='primary', use_container_width=True)

                if save:
                    require_admin_action()
                    update_data('customers', {
                        'name': n.strip(),
                        'org_no': org,
                        'email': email,
                        'phone': phone,
                        'billing_address': bill,
                        'delivery_address': deliver,
                        'status': status,
                        'notes': notes,
                    }, f"{c_id} = :id", {'id': int(sel.get(c_id))})
                    st.success('✅ Updated.')
                    st.rerun()

    with tab_new:
        st.subheader('Create order')
        if not can_write():
            st.warning('You are in visualization mode. To create orders, sign in as admin and turn off visualization mode.')

        # Options
        dep_name_col = _col(deposits_df, 'name')
        dep_id_col = _col(deposits_df, 'id_deposit', 'id')
        cust_name_col = _col(customers_df, 'name')
        cust_id_col = _col(customers_df, 'id_customer', 'id')

        prod_name_col = _col(products_df, 'name')
        prod_id_col = _col(products_df, 'id_composite', 'id')
        prod_unit_col = _col(products_df, 'output_unit')

        dep_records = deposits_df[deposits_df.get('status','Active')!='Inactive'].to_dict('records') if deposits_df is not None and not deposits_df.empty else []
        cust_records = customers_df[customers_df.get('status','Active')!='Inactive'].to_dict('records') if customers_df is not None and not customers_df.empty else []

        if not dep_records:
            st.info('Add at least one deposit first.')
        if not cust_records:
            st.info('Add at least one customer first.')

        # Stable labels for selectboxes (avoid dict objects as options)
        dep_label_to_rec = {}
        dep_labels = []
        if dep_records and dep_id_col and dep_name_col:
            for r in dep_records:
                lab = f"{_safe_str(r.get(dep_name_col))} (#{_safe_str(r.get(dep_id_col))})"
                dep_labels.append(lab)
                dep_label_to_rec[lab] = r

        cust_label_to_rec = {}
        cust_labels = []
        if cust_records and cust_id_col and cust_name_col:
            for r in cust_records:
                lab = f"{_safe_str(r.get(cust_name_col))} (#{_safe_str(r.get(cust_id_col))})"
                cust_labels.append(lab)
                cust_label_to_rec[lab] = r

        product_label_to_row = {}
        product_labels = []
        if products_df is not None and (not products_df.empty) and prod_name_col and prod_id_col:
            for _, r in products_df.sort_values(prod_name_col).iterrows():
                name = _safe_str(r.get(prod_name_col))
                pid = _safe_str(r.get(prod_id_col))
                unit = _safe_str(r.get(prod_unit_col) or 'unit')
                lab = f"{name} [{unit}] (#{pid})"
                product_labels.append(lab)
                product_label_to_row[lab] = r

        def _reset_order_form():
            # Header
            for k in [
                'so_customer_label', 'so_deposit_label', 'so_order_date', 'so_delivery_date',
                'so_currency', 'so_notes'
            ]:
                st.session_state.pop(k, None)
            # Lines
            count = int(st.session_state.get('so_line_count', 1) or 1)
            for i in range(max(count, 1)):
                for k in [f'so_prod_{i}', f'so_qty_{i}', f'so_price_{i}']:
                    st.session_state.pop(k, None)
            st.session_state['so_line_count'] = 1

        st.markdown('---')
        st.markdown('### Order header')
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            cust_label = st.selectbox('Customer*', [''] + cust_labels, key='so_customer_label')
        with h2:
            dep_label = st.selectbox('Deposit*', [''] + dep_labels, key='so_deposit_label')
        with h3:
            order_date = st.date_input('Order date', value=st.session_state.get('so_order_date', date.today()), key='so_order_date')
        with h4:
            delivery_date = st.date_input('Delivery date', value=st.session_state.get('so_delivery_date', date.today()), key='so_delivery_date')

        h5, h6 = st.columns([1, 3])
        with h5:
            currency = st.selectbox('Currency', ['NOK', 'EUR', 'USD'], index=0, key='so_currency')
        with h6:
            notes = st.text_area('Notes', height=80, key='so_notes')

        st.markdown('---')
        st.subheader('🧾 Items')

        # Dynamic line items (similar to Recipes -> Ingredients)
        st.session_state.setdefault('so_line_count', 1)

        line_items = []

        for i in range(int(st.session_state.get('so_line_count', 1) or 1)):
            st.write(f"**Item {i+1}**")
            col1, col2, col3, col4 = st.columns([4, 2, 2, 1])

            with col1:
                prod_label = st.selectbox(
                    'Product',
                    [''] + product_labels,
                    key=f'so_prod_{i}',
                )

                # Stock hint (when deposit + product selected)
                if prod_label and dep_label:
                    try:
                        prow = product_label_to_row.get(prod_label)
                        dep_rec = dep_label_to_rec.get(dep_label)
                        if prow is not None and dep_rec is not None:
                            pid = int(float(prow.get(prod_id_col)))
                            warehouse = _safe_str(dep_rec.get(dep_name_col))
                            avail = _available_units(pid, warehouse)
                            st.caption(f"Available in deposit: {avail:g} units")
                            # Warn if product is not available in this deposit
                            if avail <= 0:
                                st.warning("⚠️ This item has no stock in the selected warehouse. You can add it to the order, but you won’t be able to issue/consume it from this warehouse until stock is available.")
                    except Exception:
                        pass

            with col2:
                qty = st.number_input('Quantity', min_value=0.0, value=float(st.session_state.get(f'so_qty_{i}', 0.0) or 0.0), step=1.0, key=f'so_qty_{i}')

                # Warn if quantity exceeds available stock (when deposit + product selected)
                if prod_label and dep_label and qty > 0:
                    try:
                        prow = product_label_to_row.get(prod_label)
                        dep_rec = dep_label_to_rec.get(dep_label)
                        if prow is not None and dep_rec is not None:
                            pid = int(float(prow.get(prod_id_col)))
                            warehouse = _safe_str(dep_rec.get(dep_name_col))
                            avail = _available_units(pid, warehouse)
                            if qty > avail + 1e-9:
                                st.warning(f"⚠️ Requested quantity ({qty:g}) is greater than available stock ({avail:g}) in this warehouse.")
                    except Exception:
                        pass

            with col3:
                unit_price = st.number_input('Unit price', min_value=0.0, value=float(st.session_state.get(f'so_price_{i}', 0.0) or 0.0), step=1.0, format='%.2f', key=f'so_price_{i}')

                if prod_label and qty > 0:
                    st.write(f"**Line total:** {(qty * unit_price):.2f} {currency}")

            with col4:
                if i > 0:
                    if st.button('🗑️', key=f'so_remove_{i}'):
                        # Remove the last line (simple + predictable)
                        st.session_state['so_line_count'] = max(1, int(st.session_state.get('so_line_count', 1)) - 1)
                        st.rerun()

            if prod_label and qty > 0:
                prow = product_label_to_row.get(prod_label)
                pname = _safe_str(prow.get(prod_name_col)) if prow is not None else prod_label
                unit = _safe_str(prow.get(prod_unit_col) or 'unit') if prow is not None else 'unit'
                line_items.append({
                    'product_label': prod_label,
                    'product_name': pname,
                    'product_id': int(float(prow.get(prod_id_col))) if (prow is not None and prod_id_col) else None,
                    'unit': unit,
                    'quantity': float(qty),
                    'unit_price': float(unit_price),
                    'line_total': float(qty * unit_price),
                })

        if st.button('➕ Add another product', key='so_add_line'):
            st.session_state['so_line_count'] = int(st.session_state.get('so_line_count', 1) or 1) + 1
            st.rerun()

        subtotal_preview = float(sum(li['line_total'] for li in line_items) or 0.0)
        st.caption(f"Subtotal preview: {subtotal_preview:.2f} {currency}")

        a1, a2 = st.columns([2, 1])
        with a1:
            create = st.button('Create order', type='primary', use_container_width=True, key='so_create_btn')
        with a2:
            if st.button('Reset form', use_container_width=True, key='so_reset_btn'):
                _reset_order_form()
                st.rerun()

        if create:
            require_admin_action()

            if not cust_label or not dep_label:
                st.error('Customer and deposit are required.')
                st.stop()
            if not line_items:
                st.error('Add at least one item (product + quantity).')
                st.stop()

            cust = cust_label_to_rec.get(cust_label)
            dep = dep_label_to_rec.get(dep_label)
            cust_id = int(cust.get(cust_id_col)) if (cust and cust_id_col and cust.get(cust_id_col) is not None) else None
            cust_name = _safe_str(cust.get(cust_name_col)) if (cust and cust_name_col) else ''
            dep_id = int(dep.get(dep_id_col)) if (dep and dep_id_col and dep.get(dep_id_col) is not None) else None
            dep_name = _safe_str(dep.get(dep_name_col)) if (dep and dep_name_col) else ''

            # Order number
            try:
                next_id = 1
                if orders_df is not None and not orders_df.empty:
                    oid = _col(orders_df, 'id_sales_order', 'id')
                    if oid:
                        next_id = int(pd.to_numeric(orders_df[oid], errors='coerce').max() or 0) + 1
                order_no = f"SO-{date.today().strftime('%Y%m%d')}-{next_id:04d}"
            except Exception:
                order_no = f"SO-{date.today().strftime('%Y%m%d')}-{int(datetime.utcnow().timestamp())%100000:05d}"

            subtotal = float(sum(li['line_total'] for li in line_items) or 0.0)

            order_id = insert_data('sales_orders', {
                'order_no': order_no,
                'order_date': order_date,
                'delivery_date': delivery_date,
                'customer_id': cust_id,
                'customer_name': cust_name,
                'deposit_id': dep_id,
                'deposit_name': dep_name,
                'status': 'Draft',
                'currency': currency,
                'subtotal': subtotal,
                'total': subtotal,
                'notes': notes,
                'created_by': st.session_state.get('auth_user', 'admin'),
            })

            for li in line_items:
                insert_data('sales_order_items', {
                    'sales_order_id': int(order_id) if order_id is not None else None,
                    'product_id': li.get('product_id'),
                    'product_name': li.get('product_name'),
                    'quantity': li.get('quantity'),
                    'unit': li.get('unit', 'unit'),
                    'unit_price': li.get('unit_price'),
                    'line_total': li.get('line_total'),
                })

            st.success(f"✅ Order created: {order_no}")
            _reset_order_form()
            st.rerun()

    with tab_list:
        st.subheader('Orders list')
        if orders_df is None or orders_df.empty:
            st.info('No orders yet.')
        else:
            oid = _col(orders_df, 'id_sales_order', 'id')
            ono = _col(orders_df, 'order_no')
            ocust = _col(orders_df, 'customer_name')
            odep = _col(orders_df, 'deposit_name')
            ostatus = _col(orders_df, 'status')
            ototal = _col(orders_df, 'total')

            f1, f2, f3 = st.columns(3)
            with f1:
                status_filter = st.selectbox('Status', ['All'] + sorted(list(set([_safe_str(s) for s in orders_df[ostatus].dropna().tolist()]))) if ostatus else ['All'], index=0)
            with f2:
                cust_filter = st.selectbox('Customer', ['All'] + sorted(list(set([_safe_str(s) for s in orders_df[ocust].dropna().tolist()]))) if ocust else ['All'], index=0)
            with f3:
                dep_filter = st.selectbox('Deposit', ['All'] + sorted(list(set([_safe_str(s) for s in orders_df[odep].dropna().tolist()]))) if odep else ['All'], index=0)

            view = orders_df.copy()
            if ostatus and status_filter != 'All':
                view = view[view[ostatus].astype(str) == status_filter]
            if ocust and cust_filter != 'All':
                view = view[view[ocust].astype(str) == cust_filter]
            if odep and dep_filter != 'All':
                view = view[view[odep].astype(str) == dep_filter]

            show_cols = [c for c in [ono, 'order_date', 'delivery_date', ocust, odep, ostatus, ototal] if c and c in view.columns]
            st.dataframe(view[show_cols].sort_values(view.columns[0], ascending=False), use_container_width=True)

            # Select order to view details
            recs = view.sort_values(oid or view.columns[0], ascending=False).to_dict('records')
            sel = st.selectbox('Open order', recs, format_func=lambda r: f"{_safe_str(r.get(ono) or r.get(oid))} — {_safe_str(r.get(ocust))} ({_safe_str(r.get(ostatus))})")

            order_id = int(sel.get(oid)) if oid else None
            lines = pd.DataFrame()
            if items_df is not None and not items_df.empty and order_id is not None:
                sid = _col(items_df, 'sales_order_id')
                if sid:
                    lines = items_df[items_df[sid].astype(float) == float(order_id)].copy()

            st.markdown('---')
            st.markdown(f"### {_safe_str(sel.get(ono) or ('Order #' + str(order_id))) }")
            st.caption(f"Customer: {_safe_str(sel.get(ocust))} | Deposit: {_safe_str(sel.get(odep))} | Status: {_safe_str(sel.get(ostatus))}")

            if lines is None or lines.empty:
                st.info('No items.')
            else:
                pcol = _col(lines, 'product_name')
                qcol = _col(lines, 'quantity')
                ucol = _col(lines, 'unit')
                upcol = _col(lines, 'unit_price')
                ltcol = _col(lines, 'line_total')
                disp_cols = [c for c in [pcol, qcol, ucol, upcol, ltcol] if c]
                st.dataframe(lines[disp_cols], use_container_width=True)

            # PDF
            cust_row = None
            dep_row = None
            try:
                if customers_df is not None and not customers_df.empty:
                    cid = _col(customers_df, 'id_customer', 'id')
                    if cid and sel.get('customer_id') is not None:
                        m = customers_df[customers_df[cid].astype(float) == float(sel.get('customer_id'))]
                        cust_row = m.iloc[0].to_dict() if not m.empty else None
                if deposits_df is not None and not deposits_df.empty:
                    did = _col(deposits_df, 'id_deposit', 'id')
                    if did and sel.get('deposit_id') is not None:
                        m = deposits_df[deposits_df[did].astype(float) == float(sel.get('deposit_id'))]
                        dep_row = m.iloc[0].to_dict() if not m.empty else None
            except Exception:
                pass

            pdf_bytes = _generate_order_pdf(sel, lines, cust_row, dep_row)
            fname = f"order_{_safe_str(sel.get(ono) or order_id)}.pdf".replace(' ', '_')
            st.download_button('📄 Download Order Confirmation (PDF)', data=pdf_bytes, file_name=fname, mime='application/pdf', use_container_width=True)

            # Admin actions
            if can_write():
                st.markdown('---')
                st.subheader('Admin actions')
                a1, a2, a3 = st.columns(3)

                with a1:
                    if st.button('✅ Confirm order', use_container_width=True):
                        require_admin_action()
                        update_data('sales_orders', {
                            'status': 'Confirmed',
                            'confirmed_date': date.today(),
                        }, f"{oid} = :id", {'id': order_id})
                        st.success('Confirmed.')
                        st.rerun()

                with a2:
                    if st.button('🚚 Fulfill (dispatch + stock out)', use_container_width=True):
                        require_admin_action()
                        # stock check
                        warehouse = _safe_str(sel.get(odep))
                        insufficient = []
                        if lines is not None and not lines.empty:
                            pidc = _col(lines, 'product_id')
                            pnc = _col(lines, 'product_name')
                            qc = _col(lines, 'quantity')
                            for _, r in lines.iterrows():
                                try:
                                    pid = int(float(r.get(pidc))) if pidc and r.get(pidc) is not None else None
                                except Exception:
                                    pid = None
                                pname = _safe_str(r.get(pnc))
                                qty = float(r.get(qc) or 0)
                                if pid is None:
                                    # fall back: try map by name
                                    if products_df is not None and not products_df.empty and prod_name_col and prod_id_col:
                                        m = products_df[products_df[prod_name_col].astype(str) == pname]
                                        if not m.empty:
                                            pid = int(m.iloc[0][prod_id_col])
                                if pid is None:
                                    insufficient.append((pname, qty, 0.0))
                                    continue
                                avail = _available_units(pid, warehouse)
                                if qty > avail + 1e-9:
                                    insufficient.append((pname, qty, avail))

                        if insufficient:
                            msg = "\n".join([f"- {n}: need {q:g}, available {a:g}" for n, q, a in insufficient])
                            st.error("Insufficient stock in this deposit:\n" + msg)
                        else:
                            # stock out via negative rows
                            warehouse = _safe_str(sel.get(odep))
                            pidc = _col(lines, 'product_id')
                            pnc = _col(lines, 'product_name')
                            qc = _col(lines, 'quantity')
                            for _, r in lines.iterrows():
                                pid = None
                                try:
                                    pid = int(float(r.get(pidc))) if pidc and r.get(pidc) is not None else None
                                except Exception:
                                    pid = None
                                pname = _safe_str(r.get(pnc))
                                qty = float(r.get(qc) or 0)
                                if pid is None:
                                    if products_df is not None and not products_df.empty and prod_name_col and prod_id_col:
                                        m = products_df[products_df[prod_name_col].astype(str) == pname]
                                        if not m.empty:
                                            pid = int(m.iloc[0][prod_id_col])
                                insert_data('composite_inventory', {
                                    'composite_id': pid,
                                    'composite_name': pname,
                                    'warehouse': warehouse,
                                    'quantity_units': -float(qty),
                                })

                            update_data('sales_orders', {
                                'status': 'Fulfilled',
                                'fulfilled_date': date.today(),
                            }, f"{oid} = :id", {'id': order_id})
                            st.success('Fulfilled and stock updated.')
                            st.rerun()

                with a3:
                    if st.button('🛑 Cancel order', use_container_width=True):
                        require_admin_action()
                        update_data('sales_orders', {
                            'status': 'Cancelled',
                        }, f"{oid} = :id", {'id': order_id})
                        st.success('Cancelled.')
                        st.rerun()


elif page == "Purchases":
    st.title("🛒 Purchases & Inventory Management")
    
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

            # UI-only labels: show Manufacturer — Name in ingredient dropdowns
            ing_label_map = _ingredient_name_label_map(ingredients_df)

            # --- Items entry (Add items + real dropdowns) ---
            # We avoid st.data_editor here because in some Streamlit versions the SelectboxColumn
            # can behave like a plain text input, which breaks the ‘pick from list’ workflow.
            unit_by_name = {}
            try:
                if ingredients_df is not None and not ingredients_df.empty and "name" in ingredients_df.columns:
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

            # Render item rows
            rows_to_remove = []
            for i, row in enumerate(list(st.session_state.po_items)):
                c1, c2, c3, c4, c5 = st.columns([5, 2, 2, 2, 1])

                # Ingredient dropdown
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
                        format_func=lambda x: ing_label_map.get(str(x), str(x)) if str(x) else "",
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

                # Remove row
                with c5:
                    if st.button("🗑️", key=f"po_item_rm_{i}"):
                        rows_to_remove.append(i)

                # Persist row values
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
                # Unit is sourced during entry (auto from ingredient registry when available)
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


            # Quick drill-down: pick an order and see all items that impacted stock in that entry
            st.markdown("### 🔎 Open an order")
            if order_summary is not None and not order_summary.empty:
                order_recs = order_summary.to_dict("records")
                def _po_label(r):
                    try:
                        dtv = r.get("date", "")
                        dtv = str(dtv)[:10] if dtv is not None else ""
                    except Exception:
                        dtv = ""
                    sup = str(r.get("supplier", "") or "")
                    on = str(r.get("order_number", "") or r.get("purchase_order_id", ""))
                    oid0 = r.get("purchase_order_id", "")
                    typ = str(r.get("transaction_type", "") or "")
                    return f"{dtv} • {sup} • #{on} • {typ} (ID {oid0})"

                sel_order = st.selectbox(
                    "Select an order to view its items",
                    options=order_recs,
                    format_func=_po_label,
                    key="po_open_order_select",
                )
                sel_oid = int(sel_order.get("purchase_order_id") or 0)

                sub = filtered[filtered["purchase_order_id"] == sel_oid].copy()
                if sub is None or sub.empty:
                    st.info("No line items found for this order.")
                else:
                    # Show the exact items that were added/removed in that order
                    show_cols = [c for c in ["ingredient", "quantity", "unit", "unit_price", "freight_per_unit", "effective_unit_cost", "total_cost"] if c in sub.columns]
                    st.dataframe(sub[show_cols].reset_index(drop=True), use_container_width=True, height=260)
            else:
                st.info("No orders in the selected date range.")


            # --- Corrections: Return / Adjustment (no editing the original order) ---
            st.markdown("---")
            st.subheader("🛠️ Corrections (Return / Adjustment)")
            st.caption(
                "Instead of editing/deleting a recorded purchase, create a Return or an Adjustment to correct mistakes. "
                "Returns reduce stock *and* undo unit-cost impact; Adjustments only change stock (no financial impact)."
            )

            if order_summary is not None and not order_summary.empty:
                # Build labels for selection
                _oids = [int(x) for x in order_summary["purchase_order_id"].tolist()]
                _labels = []
                for _, _r in order_summary.iterrows():
                    _oid = int(_r["purchase_order_id"])
                    _dt = _r.get("date", "")
                    _sup = str(_r.get("supplier", "") or "")
                    _on = str(_r.get("order_number", "") or _oid)
                    _labels.append(f"{_dt} • {_sup} • #{_on} (ID {_oid})")

                sel_pos = st.selectbox(
                    "Select an order",
                    options=list(range(len(_oids))),
                    format_func=lambda i: _labels[i],
                    index=0,
                    key="po_correction_select",
                )
                sel_oid = int(_oids[int(sel_pos)])

                # Header + items
                _hdr = orders_df[orders_df["id_purchase_order"] == sel_oid].iloc[0] if (
                    orders_df is not None and not orders_df.empty
                    and "id_purchase_order" in orders_df.columns
                    and (orders_df["id_purchase_order"] == sel_oid).any()
                ) else None
                _items_df = items_df[items_df["purchase_order_id"] == sel_oid].copy() if (
                    items_df is not None and not items_df.empty and "purchase_order_id" in items_df.columns
                ) else pd.DataFrame()

                cur_type = str(_hdr.get("transaction_type", "") or "") if _hdr is not None else ""
                cur_supplier = str(_hdr.get("supplier", "") or "") if _hdr is not None else ""
                cur_order = str(_hdr.get("order_number", "") or "") if _hdr is not None else ""
                try:
                    cur_date = pd.to_datetime(_hdr.get("date", datetime.now().date()), errors="coerce").date() if _hdr is not None else datetime.now().date()
                except Exception:
                    cur_date = datetime.now().date()

                if _hdr is not None:
                    st.write(
                        f"**Type:** {cur_type}  •  **Supplier:** {cur_supplier}  •  **Order #:** {cur_order if cur_order else sel_oid}"
                    )
                    st.write(f"**Date:** {cur_date}")
                    if str(_hdr.get("notes", "") or "").strip():
                        st.caption(str(_hdr.get("notes", "")))

                # Map original quantities to help validate returns
                original_qty = {}
                if _items_df is not None and not _items_df.empty and "ingredient" in _items_df.columns and "quantity" in _items_df.columns:
                    try:
                        original_qty = (
                            _items_df.groupby("ingredient")["quantity"].sum().astype(float).to_dict()
                        )
                    except Exception:
                        original_qty = {}

                if _items_df is not None and not _items_df.empty:
                    show_cols = [c for c in ["ingredient", "quantity", "unit", "unit_price", "effective_unit_cost", "total_cost"] if c in _items_df.columns]
                    st.dataframe(
                        _items_df[show_cols].rename(
                            columns={
                                "ingredient": "Ingredient",
                                "quantity": "Quantity",
                                "unit": "Unit",
                                "unit_price": "Unit Price",
                                "effective_unit_cost": "Effective Unit Cost",
                                "total_cost": "Line Total",
                            }
                        ),
                        use_container_width=True,
                        height=220,
                    )
                else:
                    st.info("No items found for this order.")

                a1, a2, a3, _ = st.columns([1, 1, 1, 5])
                with a1:
                    if st.button("↩️ Create Return", key=f"po_correction_return_btn_{sel_oid}"):
                        st.session_state["po_correction_order_id"] = sel_oid
                        st.session_state["po_correction_mode"] = "Return"
                with a2:
                    if st.button("⚙️ Create Adjustment", key=f"po_correction_adjust_btn_{sel_oid}"):
                        st.session_state["po_correction_order_id"] = sel_oid
                        st.session_state["po_correction_mode"] = "Adjustment"
                with a3:
                    if st.button("🧨 Void order", key=f"po_correction_void_btn_{sel_oid}"):
                        st.session_state["po_correction_order_id"] = sel_oid
                        st.session_state["po_correction_mode"] = "Void"

                # --- Correction panel ---
                if st.session_state.get("po_correction_order_id") == sel_oid and st.session_state.get("po_correction_mode") in {"Return", "Adjustment", "Void"}:
                    mode = str(st.session_state.get("po_correction_mode"))
                    st.markdown("---")
                    st.subheader(f"🧾 {mode} for selected order")

                    # Supplier: fixed to original when available
                    if cur_supplier:
                        supplier_for_correction = cur_supplier
                        st.text_input("Supplier", value=cur_supplier, disabled=True, key=f"po_corr_supplier_view_{sel_oid}")
                    else:
                        sup_opts = suppliers_df["name"].dropna().astype(str).tolist() if (
                            suppliers_df is not None and not suppliers_df.empty and "name" in suppliers_df.columns
                        ) else []
                        supplier_for_correction = st.selectbox(
                            "Supplier*",
                            ["Select Supplier"] + sup_opts,
                            key=f"po_corr_supplier_pick_{sel_oid}",
                        )

                    default_prefix = {"Return": "RETURN", "Adjustment": "ADJ", "Void": "VOID"}.get(mode, "CORR")
                    default_order_num = f"{default_prefix}-{cur_order if cur_order else sel_oid}"

                    c1, c2 = st.columns(2)
                    with c1:
                        corr_date = st.date_input(
                            "Correction Date",
                            value=datetime.now().date(),
                            key=f"po_corr_date_{sel_oid}",
                        )
                    with c2:
                        corr_order_number = st.text_input(
                            "Reference #",
                            value=default_order_num,
                            key=f"po_corr_order_{sel_oid}",
                        )

                    default_note = f"{mode} created to correct order #{cur_order if cur_order else sel_oid} (ID {sel_oid})."
                    corr_notes = st.text_area(
                        "Notes",
                        value=default_note,
                        key=f"po_corr_notes_{sel_oid}",
                    )

                    # Items state
                    corr_items_key = f"po_corr_items_state_{sel_oid}"
                    if corr_items_key not in st.session_state:
                        init = []
                        if _items_df is not None and not _items_df.empty:
                            for _, r in _items_df.iterrows():
                                init.append(
                                    {
                                        "Ingredient": str(r.get("ingredient", "") or ""),
                                        "Quantity": float(r.get("quantity", 0.0) or 0.0),
                                        "Unit": str(r.get("unit", "") or ""),
                                        "Unit Price": float(r.get("unit_price", r.get("effective_unit_cost", 0.0)) or 0.0),
                                    }
                                )
                        if not init:
                            init = [{"Ingredient": "", "Quantity": 0.0, "Unit": "", "Unit Price": 0.0}]
                        st.session_state[corr_items_key] = init

                    # Ingredient options: restrict to original items for Return/Void, full list for Adjustment
                    if mode in {"Return", "Void"} and original_qty:
                        ing_options = [""] + sorted([str(k) for k in original_qty.keys()])
                    else:
                        ing_options = [""] + (
                            sorted(ingredients_df["name"].dropna().astype(str).tolist())
                            if ingredients_df is not None and not ingredients_df.empty and "name" in ingredients_df.columns
                            else []
                        )

                    # UI-only labels: show Manufacturer — Name when possible
                    corr_ing_label_map = _ingredient_name_label_map(ingredients_df)

                    lock_items = (mode == "Void")

                    addc1, _ = st.columns([1, 6])
                    with addc1:
                        if st.button("➕ Add item", key=f"po_corr_add_item_{sel_oid}", disabled=lock_items):
                            st.session_state[corr_items_key].append(
                                {"Ingredient": "", "Quantity": 0.0, "Unit": "", "Unit Price": 0.0}
                            )

                    rows_to_remove = []
                    for j, it in enumerate(list(st.session_state[corr_items_key])):
                        r1, r2, r3, r4, r5 = st.columns([5, 2, 2, 2, 1])
                        cur_ing = str(it.get("Ingredient", "") or "")
                        with r1:
                            ing_val = st.selectbox(
                                "Ingredient",
                                ing_options,
                                index=(ing_options.index(cur_ing) if cur_ing in ing_options else 0),
                                format_func=lambda x: corr_ing_label_map.get(str(x), str(x)) if str(x) else "",
                                key=f"po_corr_ing_{sel_oid}_{j}",
                                label_visibility="collapsed",
                                disabled=lock_items,
                            )
                        with r2:
                            qty_val = st.number_input(
                                "Quantity",
                                min_value=0.0,
                                value=float(it.get("Quantity", 0.0) or 0.0),
                                step=0.1,
                                format="%.3f",
                                key=f"po_corr_qty_{sel_oid}_{j}",
                                label_visibility="collapsed",
                                disabled=lock_items,
                            )
                        with r3:
                            unit_val = st.text_input(
                                "Unit",
                                value=str(it.get("Unit", "") or ""),
                                key=f"po_corr_unit_{sel_oid}_{j}",
                                label_visibility="collapsed",
                                disabled=lock_items,
                            )
                        with r4:
                            # Adjustments: no financial impact by default
                            disable_price = lock_items or (mode == "Adjustment")
                            price_val = st.number_input(
                                "Unit Price",
                                min_value=0.0,
                                value=float(it.get("Unit Price", 0.0) or 0.0),
                                step=0.01,
                                format="%.2f",
                                key=f"po_corr_price_{sel_oid}_{j}",
                                label_visibility="collapsed",
                                disabled=disable_price,
                            )
                        with r5:
                            if st.button("🗑️", key=f"po_corr_rm_{sel_oid}_{j}", disabled=lock_items):
                                rows_to_remove.append(j)

                        st.session_state[corr_items_key][j] = {
                            "Ingredient": str(ing_val),
                            "Quantity": float(qty_val),
                            "Unit": str(unit_val),
                            "Unit Price": float(price_val),
                        }

                    for idx in sorted(rows_to_remove, reverse=True):
                        try:
                            st.session_state[corr_items_key].pop(idx)
                        except Exception:
                            pass

                    # Validation + preview
                    items = pd.DataFrame(
                        st.session_state[corr_items_key],
                        columns=["Ingredient", "Quantity", "Unit", "Unit Price"],
                    ).fillna({"Ingredient": "", "Quantity": 0.0, "Unit Price": 0.0})
                    items = items[(items["Ingredient"].astype(str).str.len() > 0) & (items["Quantity"] > 0)]

                    warnings = []
                    if mode in {"Return", "Void"} and original_qty:
                        for _, r in items.iterrows():
                            ing = str(r["Ingredient"])
                            qty = float(r["Quantity"])
                            max_qty = float(original_qty.get(ing, 0.0) or 0.0)
                            if qty > max_qty + 1e-9:
                                warnings.append(f"Return qty for '{ing}' ({qty:,.3f}) is greater than original ({max_qty:,.3f}).")

                    if warnings:
                        st.warning("\n".join(warnings))

                    preview_rows = []
                    for _, r in items.iterrows():
                        ing = str(r["Ingredient"])
                        qty = float(r["Quantity"])
                        unit = str(r.get("Unit", "") or "")
                        unit_price = float(r.get("Unit Price", 0.0) or 0.0)

                        # Costs: Return/Void are negative (refund). Adjustment has zero financial impact.
                        if mode in {"Return", "Void"}:
                            line_total = -1.0 * qty * unit_price
                        else:
                            line_total = 0.0

                        preview_rows.append(
                            {
                                "Ingredient": ing,
                                "Quantity": qty,
                                "Unit": unit,
                                "Unit Price": unit_price,
                                "Freight / Unit": 0.0,
                                "Effective Unit Cost": unit_price,
                                "Line Total": float(line_total),
                            }
                        )

                    if preview_rows:
                        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, height=220)
                    else:
                        st.info("Add at least one valid item to preview.")

                    s1, s2, _ = st.columns([1, 1, 6])
                    with s1:
                        if st.button("💾 Save correction", type="primary", key=f"po_corr_save_{sel_oid}"):
                            if (not supplier_for_correction) or (supplier_for_correction == "Select Supplier"):
                                st.error("Please select a supplier.")
                            elif not preview_rows:
                                st.error("Please add at least one valid item.")
                            else:
                                try:
                                    tx_type = "Return" if mode in {"Return", "Void"} else "Adjustment"
                                    save_purchase_order_fast(
                                        transaction_type=tx_type,
                                        supplier=str(supplier_for_correction),
                                        order_number=str(corr_order_number),
                                        po_date=corr_date,
                                        freight_total=0.0,
                                        notes=str(corr_notes),
                                        recorded_by="User",
                                        preview_rows=preview_rows,
                                    )

                                    # Returns should also undo unit-cost impact
                                    if tx_type == "Return":
                                        try:
                                            eng = get_engine()
                                            with eng.begin() as conn:
                                                _recalc_unit_cost_for_ingredients(conn, [r["Ingredient"] for r in preview_rows])
                                        except Exception:
                                            pass

                                    # Clear state
                                    st.session_state.pop(corr_items_key, None)
                                    st.session_state.pop("po_correction_order_id", None)
                                    st.session_state.pop("po_correction_mode", None)
                                    data = get_all_data()
                                    st.success("✅ Correction saved!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Erro ao salvar: {e}")
                    with s2:
                        if st.button("✖ Cancel", key=f"po_corr_cancel_{sel_oid}"):
                            st.session_state.pop(corr_items_key, None)
                            st.session_state.pop("po_correction_order_id", None)
                            st.session_state.pop("po_correction_mode", None)

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
                    "effective_unit_cost": "mean",
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

        # Edit panel (shows when a recipe has been selected)
        if st.session_state.get('edit_recipe') is not None:
            edit_id = st.session_state.get('edit_recipe')

            # Locate the recipe row
            rid_col = _col(recipes_df, 'id_recipe', 'id_receipt', 'id')
            recipe_row = None
            if rid_col and not recipes_df.empty:
                match = recipes_df[recipes_df[rid_col].astype(str) == str(edit_id)]
                if not match.empty:
                    recipe_row = match.iloc[0]
            if recipe_row is None and not recipes_df.empty:
                # fallback: try index
                try:
                    recipe_row = recipes_df.loc[edit_id]
                except Exception:
                    recipe_row = None

            if recipe_row is None:
                st.warning('Selected recipe not found (it may have been deleted).')
                st.session_state.edit_recipe = None
            else:
                with st.expander('📝 Edit Recipe', expanded=True):
                    # Current values (with schema fallbacks)
                    name_col = _col(recipes_df, 'name', 'recipe_name', 'title')
                    style_col = _col(recipes_df, 'style', 'beer_style', 'type_style')
                    desc_col = _col(recipes_df, 'description', 'notes', 'desc')
                    batch_col = _col(recipes_df, 'batch_volume', 'batch_size', 'batch_l', 'volume_l', 'volume')
                    eff_col = _col(recipes_df, 'efficiency', 'brewhouse_efficiency', 'brew_efficiency', 'bh_efficiency', 'efficiency_pct', 'efficiency_percent')
                    og_col = _col(recipes_df, 'og', 'original_gravity', 'target_original_gravity', 'target_og', 'og_target', 'og_planned', 'og_plato', 'original_gravity_plato', 'target_og_plato', 'target_original_gravity_plato', 'original_gravity_p', 'og_p')
                    fg_col = _col(recipes_df, 'fg', 'final_gravity', 'target_final_gravity', 'target_fg', 'fg_target', 'fg_planned', 'fg_plato', 'final_gravity_plato', 'target_fg_plato', 'target_final_gravity_plato', 'final_gravity_p', 'fg_p')
                    ibu_col = _col(recipes_df, 'ibus', 'ibu', 'target_ibu', 'ibu_target')
                    ebc_col = _col(recipes_df, 'ebc', 'color_ebc', 'srm', 'target_ebc', 'ebc_target', 'color')

                    cur_name = recipe_row.get(name_col) if name_col else recipe_row.get('name', recipe_row.get('recipe_name', ''))
                    cur_style = recipe_row.get(style_col) if style_col else recipe_row.get('style', recipe_row.get('beer_style', ''))
                    cur_desc = recipe_row.get(desc_col) if desc_col else recipe_row.get('description', recipe_row.get('notes', ''))
                    cur_batch = recipe_row.get(batch_col) if batch_col else recipe_row.get('batch_volume', recipe_row.get('batch_size', None))
                    cur_eff = recipe_row.get(eff_col) if eff_col else recipe_row.get('efficiency', None)
                    cur_og = recipe_row.get(og_col) if og_col else recipe_row.get('og', recipe_row.get('original_gravity', recipe_row.get('og_plato', None)))
                    cur_fg = recipe_row.get(fg_col) if fg_col else recipe_row.get('fg', recipe_row.get('final_gravity', recipe_row.get('fg_plato', None)))
                    cur_ibu = recipe_row.get(ibu_col) if ibu_col else recipe_row.get('ibus', recipe_row.get('ibu', None))
                    cur_ebc = recipe_row.get(ebc_col) if ebc_col else recipe_row.get('ebc', recipe_row.get('color_ebc', recipe_row.get('srm', None)))
                    # Brewery selection
                    breweries_df = data.get('breweries', pd.DataFrame())
                    brewery_id_col = _col(breweries_df, 'id_brewery', 'brewery_id', 'id')
                    brewery_name_col = _col(breweries_df, 'name', 'brewery_name')

                    # attempt to find current brewery id
                    cur_brewery_id = recipe_row.get('brewery_id', recipe_row.get('id_brewery', recipe_row.get('target_brewery_id', recipe_row.get('id_target_brewery', recipe_row.get('brewery_target_id', recipe_row.get('id_brewery_target', None))))))

                    # Sometimes legacy schemas store the brewery as a *name* instead of an id
                    cur_brewery_name = recipe_row.get('brewery_name', recipe_row.get('target_brewery_name', recipe_row.get('brewery_target_name', recipe_row.get('target_brewery', recipe_row.get('brewery_target', recipe_row.get('brewery', None))))))

                    with st.form(key=f'edit_recipe_form_{str(edit_id)}'):
                        c1, c2 = st.columns(2)
                        with c1:
                            new_name = st.text_input('Recipe Name*', value=str(cur_name) if cur_name is not None else '')
                            new_style = st.text_input('Beer Style', value=str(cur_style) if cur_style is not None else '')

                            if not breweries_df.empty and brewery_id_col and brewery_name_col:
                                brewery_options = {row[brewery_id_col]: row[brewery_name_col] for _, row in breweries_df.iterrows()}
                                ids = list(brewery_options.keys())
                                # Determine index (supports id stored as int/uuid/str OR stored as name)
                                idx = 0
                                try:
                                    ids_str = [str(x) for x in ids]
                                    if cur_brewery_id is not None and str(cur_brewery_id) in ids_str:
                                        idx = ids_str.index(str(cur_brewery_id))
                                    elif cur_brewery_name is not None:
                                        names = [str(brewery_options.get(_id, '')).strip() for _id in ids]
                                        name_map = {nm.lower(): i for i, nm in enumerate(names) if nm}
                                        idx = name_map.get(str(cur_brewery_name).strip().lower(), 0)
                                except Exception:
                                    idx = 0
                                new_brewery = st.selectbox('Target Brewery', options=ids, index=idx,
                                                          format_func=lambda x: brewery_options.get(x, str(x)))
                                new_brewery_name = brewery_options.get(new_brewery)
                            else:
                                new_brewery = cur_brewery_id
                                new_brewery_name = recipe_row.get('brewery_name', None)
                                st.info('Brewery list not available; keeping current brewery.')

                            new_batch = st.number_input('Batch Volume (L)', min_value=0.0, value=float(cur_batch) if cur_batch not in [None, '', 'N/A'] and str(cur_batch) != 'nan' else 0.0, step=0.5)
                            new_eff = st.number_input('Efficiency (%)', min_value=0.0, max_value=100.0, value=float(cur_eff) if cur_eff not in [None, '', 'N/A'] and str(cur_eff) != 'nan' else 0.0, step=1.0)

                        with c2:
                            new_og = st.number_input('OG (°P)', min_value=0.0, value=float(cur_og) if cur_og not in [None, '', 'N/A'] and str(cur_og) != 'nan' else 0.0, step=0.1)
                            new_fg = st.number_input('FG (°P)', min_value=0.0, value=float(cur_fg) if cur_fg not in [None, '', 'N/A'] and str(cur_fg) != 'nan' else 0.0, step=0.1)
                            new_ibu = st.number_input('IBU', min_value=0.0, value=float(cur_ibu) if cur_ibu not in [None, '', 'N/A'] and str(cur_ibu) != 'nan' else 0.0, step=1.0)
                            new_ebc = st.number_input('Color (EBC)', min_value=0.0, value=float(cur_ebc) if cur_ebc not in [None, '', 'N/A'] and str(cur_ebc) != 'nan' else 0.0, step=1.0)

                        new_desc = st.text_area('Description / Notes', value=str(cur_desc) if cur_desc is not None else '', height=120)

                        colb1, colb2, colb3 = st.columns([1,1,3])
                        with colb1:
                            save = st.form_submit_button('💾 Save', type='primary', use_container_width=True)
                        with colb2:
                            cancel = st.form_submit_button('✖ Cancel', use_container_width=True)



                    # -----------------------------
                    # Ingredients editor
                    # -----------------------------
                    st.markdown("### 🧪 Recipe Ingredients")

                    ingredients_df = data.get("ingredients", pd.DataFrame())
                    ing_id_col = _col(ingredients_df, 'id_ingredient', 'ingredient_id', 'id')
                    ing_name_col = _col(ingredients_df, 'name', 'ingredient_name', 'ingredient')
                    ing_unit_col = _col(ingredients_df, 'unit')

                    recipe_items_df = data.get("recipe_items", pd.DataFrame())
                    ri_recipe_col = _col(recipe_items_df, 'recipe_id', 'id_recipe', 'id_receipt')
                    ri_ing_id_col = _col(recipe_items_df, 'id_ingredient', 'ingredient_id')
                    ri_ing_name_col = _col(recipe_items_df, 'ingredient_name', 'ingredient')
                    ri_qty_col = _col(recipe_items_df, 'quantity')
                    ri_unit_col = _col(recipe_items_df, 'unit')

                    edit_items_key = f"edit_recipe_items_{str(edit_id)}"
                    edit_count_key = f"edit_recipe_ing_count_{str(edit_id)}"

                    # Load items once per edit session
                    if edit_items_key not in st.session_state:
                        items = []
                        if recipe_items_df is not None and not recipe_items_df.empty and ri_recipe_col:
                            try:
                                _f = recipe_items_df[recipe_items_df[ri_recipe_col].astype(str) == str(edit_id)].copy()
                            except Exception:
                                _f = pd.DataFrame()
                            if not _f.empty:
                                for _, r in _f.iterrows():
                                    nm = r.get(ri_ing_name_col) if ri_ing_name_col else None
                                    iid = r.get(ri_ing_id_col) if ri_ing_id_col else None
                                    qty = r.get(ri_qty_col) if ri_qty_col else 0.0
                                    unit = r.get(ri_unit_col) if ri_unit_col else None
                                    items.append({
                                        'id_ingredient': iid,
                                        'ingredient_name': nm,
                                        'quantity': float(qty) if qty not in [None, '', 'N/A'] and str(qty) != 'nan' else 0.0,
                                        'unit': unit,
                                    })
                        st.session_state[edit_items_key] = items
                        st.session_state[edit_count_key] = max(1, len(items) if items else 1)

                    # Ensure count exists
                    if edit_count_key not in st.session_state:
                        st.session_state[edit_count_key] = max(1, len(st.session_state.get(edit_items_key, [])) or 1)

                    # Ingredient options
                    ingredient_options = []
                    ing_lookup = {}
                    if ingredients_df is not None and not ingredients_df.empty and ing_id_col and ing_name_col:
                        ing_mfg_col = _col(ingredients_df, 'manufacturer', 'brand', 'producer', 'maltster', 'maker')
                        for _, r in ingredients_df.iterrows():
                            _iid = r.get(ing_id_col)
                            _nm = r.get(ing_name_col)
                            if _iid is None or _nm is None:
                                continue
                            ingredient_options.append(_iid)
                            _nm_s = str(_nm)
                            _mfg = str(r.get(ing_mfg_col) or '').strip() if ing_mfg_col else ''
                            _label = f"{_mfg} — {_nm_s}" if _mfg else _nm_s
                            ing_lookup[_iid] = {
                                'label': _label,
                                'name': _label,
                                'raw_name': _nm_s,
                                'unit': str(r.get(ing_unit_col) or ''),
                            }

                    # Render rows
                    working_items = []
                    row_count = int(st.session_state.get(edit_count_key, 1) or 1)
                    for i in range(row_count):
                        base = None
                        existing = st.session_state.get(edit_items_key, [])
                        if i < len(existing):
                            base = existing[i]
                        default_ing_id = base.get('id_ingredient') if base else None

                        # If we only have a name, try to map to an id
                        if default_ing_id is None and base and base.get('ingredient_name') and ing_lookup:
                            name_to_id = {v['name']: k for k, v in ing_lookup.items()}
                            default_ing_id = name_to_id.get(str(base.get('ingredient_name')))

                        cols_ing = st.columns([5, 3, 2, 1])
                        with cols_ing[0]:
                            if ingredient_options:
                                try:
                                    idx = ingredient_options.index(default_ing_id) if default_ing_id in ingredient_options else 0
                                except Exception:
                                    idx = 0
                                sel_ing = st.selectbox(
                                    "Ingredient",
                                    options=ingredient_options,
                                    index=idx,
                                    format_func=lambda x: ing_lookup.get(x, {}).get('name', str(x)),
                                    key=f"edit_ing_{str(edit_id)}_{i}",
                                )
                            else:
                                # fallback: free text
                                sel_ing = None
                                st.text_input(
                                    "Ingredient",
                                    value=str(base.get('ingredient_name') if base else ''),
                                    key=f"edit_ing_name_{str(edit_id)}_{i}",
                                )

                        with cols_ing[1]:
                            planned_unit = ''
                            if sel_ing is not None and sel_ing in ing_lookup:
                                planned_unit = ing_lookup[sel_ing].get('unit', '')
                            else:
                                planned_unit = str(base.get('unit') if base else '')

                            qty_default = float(base.get('quantity') or 0.0) if base else 0.0
                            qty_val = st.number_input(
                                f"Quantity ({planned_unit or 'units'})",
                                min_value=0.0,
                                value=qty_default,
                                step=0.1,
                                key=f"edit_qty_{str(edit_id)}_{i}",
                            )

                        with cols_ing[2]:
                            st.text_input(
                                "Unit",
                                value=planned_unit,
                                disabled=True,
                                key=f"edit_unit_{str(edit_id)}_{i}",
                            )

                        with cols_ing[3]:
                            if st.button("🗑️", key=f"edit_rm_{str(edit_id)}_{i}"):
                                # remove item i
                                items = st.session_state.get(edit_items_key, [])
                                if i < len(items):
                                    items.pop(i)
                                    st.session_state[edit_items_key] = items
                                st.session_state[edit_count_key] = max(1, int(st.session_state.get(edit_count_key, 1)) - 1)
                                st.rerun()

                        # collect row
                        if ingredient_options:
                            ing_name = ing_lookup.get(sel_ing, {}).get('raw_name') if sel_ing is not None else ''
                            unit_val = ing_lookup.get(sel_ing, {}).get('unit') if sel_ing is not None else planned_unit
                            working_items.append({
                                'id_ingredient': sel_ing,
                                'ingredient_name': ing_name,
                                'quantity': float(qty_val),
                                'unit': unit_val,
                            })
                        else:
                            ing_name = st.session_state.get(f"edit_ing_name_{str(edit_id)}_{i}", '')
                            working_items.append({
                                'id_ingredient': None,
                                'ingredient_name': ing_name,
                                'quantity': float(qty_val),
                                'unit': planned_unit,
                            })

                    st.session_state[edit_items_key] = working_items

                    if st.button("➕ Add Ingredient", key=f"edit_add_ing_{str(edit_id)}"):
                        st.session_state[edit_count_key] = int(st.session_state.get(edit_count_key, 1) or 1) + 1
                        st.rerun()

                    if not st.session_state.get(edit_items_key):
                        st.warning("No ingredients in this recipe yet. Add at least one ingredient.")

                    # Keep the confirmation UI *right below* the Cancel button
                    confirm_flag_key = f"confirm_cancel_recipe_edit_{str(edit_id)}"
                    if confirm_flag_key not in st.session_state:
                        st.session_state[confirm_flag_key] = False

                    if cancel:
                        st.session_state[confirm_flag_key] = True
                        st.rerun()

                    if st.session_state.get(confirm_flag_key):
                        st.warning("Confirm cancel? Unsaved changes will be lost.")
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            if st.button("✅ Yes, discard", key=f"confirm_cancel_yes_{str(edit_id)}", use_container_width=True):
                                st.session_state[confirm_flag_key] = False
                                st.session_state.edit_recipe = None
                                st.rerun()
                        with cc2:
                            if st.button("↩️ No, keep editing", key=f"confirm_cancel_no_{str(edit_id)}", use_container_width=True):
                                st.session_state[confirm_flag_key] = False
                                st.rerun()

                    if save:
                        # reset any pending cancel confirmation
                        try:
                            st.session_state[confirm_flag_key] = False
                        except Exception:
                            pass
                        # Build update dict with common columns (update_data will ignore unknown cols)
                        update_dict = {
                            'name': new_name,
                            'recipe_name': new_name,
                            'style': new_style,
                            'beer_style': new_style,
                            'description': new_desc,
                            'notes': new_desc,
                            'batch_volume': new_batch,
                            'batch_size': new_batch,
                            'efficiency': new_eff,
                            'brewhouse_efficiency': new_eff,
                            'brew_efficiency': new_eff,
                            'bh_efficiency': new_eff,
                            'efficiency_pct': new_eff,
                            'efficiency_percent': new_eff,
                            'og': new_og,
                            'target_og': new_og,
                            'og_target': new_og,
                            'og_planned': new_og,
                            'original_gravity': new_og,
                            'og_plato': new_og,
                            'fg': new_fg,
                            'target_fg': new_fg,
                            'fg_target': new_fg,
                            'fg_planned': new_fg,
                            'final_gravity': new_fg,
                            'fg_plato': new_fg,
                            'ibus': new_ibu,
                            'target_ibu': new_ibu,
                            'ibu_target': new_ibu,
                            'ibu': new_ibu,
                            'ebc': new_ebc,
                            'target_ebc': new_ebc,
                            'ebc_target': new_ebc,
                            'color_ebc': new_ebc,
                            'srm': new_ebc,
                            'brewery_id': new_brewery,
                            'id_brewery': new_brewery,
                            'target_brewery_id': new_brewery,
                            'id_target_brewery': new_brewery,
                            'brewery_target_id': new_brewery,
                            'id_brewery_target': new_brewery,
                            'target_brewery': new_brewery_name,  # some schemas store name here
                            'brewery_target': new_brewery_name,
                            'brewery_name': new_brewery_name,
                            'target_brewery_name': new_brewery_name,
                            'brewery_target_name': new_brewery_name,
                        }

                        # Compute and store ABV (rounded to 1 decimal) when possible
                        try:
                            if new_og is not None and new_fg is not None:
                                _ogv = float(new_og)
                                _fgv = float(new_fg)
                                _abv = (_ogv - _fgv) * 0.524
                                _abv = round(_abv, 1)
                                update_dict.update({
                                    'abv': _abv,
                                    'abv_pct': _abv,
                                    'abv_percent': _abv,
                                    'alcohol_by_volume': _abv,
                                })
                        except Exception:
                            pass

                        # Determine recipes table id column
                        engine = get_engine()
                        dialect = engine.dialect.name.lower()
                        recipes_cols = get_table_columns_cached('recipes', dialect) or []
                        recipes_id_col = None
                        for cand in ['id_recipe', 'id_receipt', 'id']:
                            if cand.lower() in set([c.lower() for c in recipes_cols]):
                                recipes_id_col = cand
                                break
                        if not recipes_id_col:
                            recipes_id_col = 'id_recipe'

                        

                        # --- Update recipe_items to match edited ingredient list ---
                        try:
                            recipe_items_cols = get_table_columns_cached('recipe_items', dialect) or []
                            items_fk_col = _first_existing(recipe_items_cols, ['recipe_id', 'id_recipe', 'id_receipt', 'recipe', 'recipe_fk', 'fk_recipe']) or 'recipe_id'
                            items_ing_name_col = _first_existing(recipe_items_cols, ['ingredient_name', 'ingredient', 'name', 'item_name', 'ingredient_desc', 'ingredient_description']) or 'ingredient_name'
                            items_ing_id_col = _first_existing(recipe_items_cols, ['id_ingredient', 'ingredient_id', 'ing_id', 'fk_ingredient'])
                            items_qty_col = _first_existing(recipe_items_cols, ['quantity']) or 'quantity'
                            items_unit_col = _first_existing(recipe_items_cols, ['unit']) or 'unit'

                            # Remove existing items for this recipe
                            try:
                                delete_data('recipe_items', f"{items_fk_col} = :rid", {'rid': edit_id})
                            except Exception:
                                delete_data('recipe_items', f"CAST({items_fk_col} AS TEXT) = :rid", {'rid': str(edit_id)})

                            # Re-insert current list
                            _items = st.session_state.get(f"edit_recipe_items_{str(edit_id)}", []) or []
                            for it in _items:
                                if not it or float(it.get('quantity') or 0) <= 0:
                                    continue
                                row = {
                                    items_fk_col: edit_id,
                                    items_ing_name_col: it.get('ingredient_name'),
                                    items_qty_col: float(it.get('quantity') or 0),
                                    items_unit_col: it.get('unit') or '',
                                }
                                if items_ing_id_col:
                                    row[items_ing_id_col] = it.get('id_ingredient')
                                insert_data('recipe_items', row)
                        except Exception as _e:
                            st.error(f"Could not update recipe ingredients: {_e}")

                        # Update recipe row (handle id-type differences across DBs)
                        updated_rows = 0
                        try:
                            updated_rows = update_data('recipes', update_dict, f"{recipes_id_col} = :rid", {'rid': edit_id})
                        except Exception:
                            updated_rows = 0

                        if updated_rows == 0:
                            # Fallback: compare as TEXT (helps when DB column is numeric but edit_id is a string)
                            updated_rows = update_data('recipes', update_dict, f"CAST({recipes_id_col} AS TEXT) = :rid", {'rid': str(edit_id)})
                        st.success('Recipe updated!')
                        st.session_state.edit_recipe = None
                        # refresh
                        data = get_all_data()
                        st.rerun()

        
        if not recipes_df.empty:
            # Filtros
            col1, col2, col3 = st.columns(3)
            with col1:
                recipe_search = st.text_input("Search Recipes", key="recipe_search")
            with col2:
                style_filter = st.selectbox(
                    "Filter by Style",
                    (["All Styles"] + sorted(recipes_df[_col(recipes_df,'style','beer_style')].dropna().unique().tolist()) if _col(recipes_df,'style','beer_style') else ["All Styles"]),
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
            name_col = _col(filtered_recipes, 'name', 'recipe_name')
            style_col = _col(filtered_recipes, 'style', 'beer_style')
            if recipe_search and (name_col or style_col):
                mask = False
                if name_col:
                    mask = mask | filtered_recipes[name_col].astype(str).str.contains(recipe_search, case=False, na=False)
                if style_col:
                    mask = mask | filtered_recipes[style_col].astype(str).str.contains(recipe_search, case=False, na=False)
                filtered_recipes = filtered_recipes[mask]
            if style_filter != "All Styles" and style_col:
                filtered_recipes = filtered_recipes[filtered_recipes[style_col] == style_filter]
            if brewery_filter != "All Breweries" and 'brewery_name' in filtered_recipes.columns:
                filtered_recipes = filtered_recipes[filtered_recipes['brewery_name'] == brewery_filter]
            
            # Mostrar receitas
            if not filtered_recipes.empty:
                for idx, recipe in filtered_recipes.iterrows():
                    recipe_name = recipe.get('name', recipe.get('recipe_name', 'Recipe'))
                    recipe_style = recipe.get('style', recipe.get('beer_style', 'N/A'))
                    with st.expander(f"🍺 {recipe_name} - {recipe_style}", expanded=False):
                        col_left, col_right = st.columns([2, 1])
                        
                        with col_left:
                            # Informações básicas
                            # Batch size (robust against schema differences + NaN)
                            batch_val = recipe.get('batch_volume', recipe.get('batch_size'))
                            if batch_val is None or str(batch_val) == 'nan':
                                for _k in ['batch_l','batch_volume_l','batch_volume_liters','volume_l','volume']:
                                    if _k in recipe.index and recipe.get(_k) is not None and str(recipe.get(_k)) != 'nan':
                                        batch_val = recipe.get(_k)
                                        break
                            batch_txt = f"{batch_val}L" if batch_val is not None and str(batch_val) != 'nan' else 'N/A'
                            st.write(f"**Batch Size:** {batch_txt}")

                            eff_val = recipe.get('efficiency', recipe.get('brewhouse_efficiency', recipe.get('brew_efficiency', recipe.get('bh_efficiency'))))
                            eff_txt = f"{eff_val}%" if eff_val is not None and str(eff_val) != 'nan' else 'N/A'
                            st.write(f"**Efficiency:** {eff_txt}")

                            # Brewery name: use stored name if present, else map from breweries table
                            brew_name = recipe.get('brewery_name', recipe.get('target_brewery_name', recipe.get('brewery_target_name', recipe.get('target_brewery', recipe.get('brewery_target')))))
                            if brew_name is None or str(brew_name) == 'nan' or str(brew_name).strip() == '':
                                bid = recipe.get('brewery_id', recipe.get('id_brewery', recipe.get('target_brewery_id', recipe.get('id_target_brewery', recipe.get('brewery_target_id', recipe.get('id_brewery_target'))))))
                                breweries_df = data.get('breweries', pd.DataFrame())
                                b_id_col = _col(breweries_df, 'id_brewery', 'brewery_id', 'id')
                                b_name_col = _col(breweries_df, 'name', 'brewery_name')
                                if bid is not None and not breweries_df.empty and b_id_col and b_name_col:
                                    mm = breweries_df[breweries_df[b_id_col].astype(str) == str(bid)]
                                    if not mm.empty:
                                        brew_name = mm.iloc[0][b_name_col]
                            st.write(f"**Target Brewery:** {brew_name if brew_name is not None and str(brew_name) != 'nan' else 'N/A'}")
                            
                            # Estatísticas da cerveja
                            if any(k in recipe.index for k in ['og','fg','ibus','ibu','ebc','srm','original_gravity','final_gravity','og_plato','fg_plato']):
                                st.write("**Beer Stats:**")
                                col_stats1, col_stats2 = st.columns(2)
                                with col_stats1:
                                    og_val = recipe.get('og', recipe.get('original_gravity', recipe.get('og_plato')))
                                    if og_val is not None and str(og_val) != 'nan':
                                        st.write(f"OG: {og_val}°P")
                                    fg_val = recipe.get('fg', recipe.get('final_gravity', recipe.get('fg_plato')))
                                    if fg_val is not None and str(fg_val) != 'nan':
                                        st.write(f"FG: {fg_val}°P")
                                with col_stats2:
                                    ibu_val = recipe.get('ibus', recipe.get('ibu'))
                                    if ibu_val is not None and str(ibu_val) != 'nan':
                                        st.write(f"IBU: {ibu_val}")
                                    col_val = recipe.get('ebc', recipe.get('color_ebc', recipe.get('srm')))
                                    if col_val is not None and str(col_val) != 'nan':
                                        st.write(f"Color: {col_val} EBC")                                # ABV (supports either stored 'abv' or derived from OG/FG)
                                abv_stored = recipe.get('abv')
                                if abv_stored is not None and str(abv_stored) != 'nan':
                                    try:
                                        st.write(f"**ABV:** {float(abv_stored):.1f}%")
                                    except Exception:
                                        st.write(f"**ABV:** {abv_stored}%")
                                else:
                                    og_calc = recipe.get('og', recipe.get('original_gravity', recipe.get('og_plato')))
                                    fg_calc = recipe.get('fg', recipe.get('final_gravity', recipe.get('fg_plato')))
                                    if og_calc and fg_calc:
                                        abv = (float(og_calc) - float(fg_calc)) * 0.524
                                        st.write(f"**ABV:** {abv:.1f}%")
                            # Description / notes (supports Postgres legacy 'notes' column)
                            desc = recipe.get('description', recipe.get('notes', ''))
                            if desc is not None and str(desc) != 'nan' and str(desc).strip() != '':
                                st.write("**Description / Notes:**")
                                st.write(desc)
                        with col_right:
                            # Ações
                            st.write("**Actions:**")

                            # Robust recipe id (supports older typo `id_receipt` and proper `id_recipe`)
                            recipe_id = recipe.get('id_recipe', recipe.get('id_receipt', recipe.get('id')))
                            if recipe_id is None:
                                recipe_id = idx
                            recipe_id_str = str(recipe_id)

                            if st.button("📋 Create Batch", key=f"brew_{recipe_id_str}", use_container_width=True):
                                st.session_state['recipe_to_brew'] = recipe_id
                                st.rerun()

                            if st.button("📝 Edit", key=f"edit_{recipe_id_str}", use_container_width=True):
                                st.session_state['edit_recipe'] = recipe_id
                                st.rerun()

                            if st.button("🗑️ Delete", key=f"delete_{recipe_id_str}", use_container_width=True):
                                st.session_state.delete_confirmation = {"type": "recipe", "id": recipe_id, "name": recipe.get('name', '')}
                                st.rerun()

                        # Ingredients
                        st.write("**Ingredients:**")
                        recipe_items = get_recipe_items_for_recipe(recipe_id)

                        if recipe_items is not None and not recipe_items.empty:
                            ingredients_df = data.get("ingredients", pd.DataFrame())
                            ing_id_col = _col(ingredients_df, 'id_ingredient', 'ingredient_id', 'id')
                            ing_name_col = _col(ingredients_df, 'name', 'ingredient_name', 'ingredient')

                            it_ing_name_col = _col(recipe_items, 'ingredient_name', 'ingredient', 'name', 'item_name', 'ingredient_desc', 'ingredient_description')
                            it_ing_id_col = _col(recipe_items, 'id_ingredient', 'ingredient_id', 'ing_id', 'fk_ingredient')
                            it_qty_col = _col(recipe_items, 'quantity')
                            it_unit_col = _col(recipe_items, 'unit')

                            for _, item in recipe_items.iterrows():
                                ingredient_name = "Unknown"

                                if it_ing_name_col and pd.notna(item.get(it_ing_name_col)) and str(item.get(it_ing_name_col)).strip():
                                    ingredient_name = str(item.get(it_ing_name_col))
                                elif ingredients_df is not None and not ingredients_df.empty and it_ing_id_col and ing_id_col and ing_name_col:
                                    ing = ingredients_df[ingredients_df[ing_id_col].astype(str) == str(item.get(it_ing_id_col))]
                                    if not ing.empty:
                                        ingredient_name = str(ing.iloc[0][ing_name_col])

                                qty = item.get(it_qty_col) if it_qty_col else ''
                                unit = str(item.get(it_unit_col) or 'units') if it_unit_col else 'units'
                                st.write(f"- {ingredient_name}: {qty} {unit}")
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
                brewery_id_col = _col(breweries_df, 'id_brewery', 'brewery_id', 'id')
                brewery_name_col = _col(breweries_df, 'name', 'brewery_name')
                if not (brewery_id_col and brewery_name_col):
                    st.error('Breweries table is missing id/name columns.')
                    st.stop()
                brewery_options = {row[brewery_id_col]: row[brewery_name_col] for _, row in breweries_df.iterrows()}
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
            ing_label_map = _ingredient_name_label_map(ingredients_df)
            # Controle dinâmico de ingredientes
            if 'recipe_ingredient_count' not in st.session_state:
                st.session_state.recipe_ingredient_count = 1
            
            ingredient_list = []
            
            for i in range(st.session_state.recipe_ingredient_count):
                st.write(f"**Ingredient {i+1}**")
                col_ing1, col_ing2, col_ing3, col_ing4 = st.columns([3, 2, 2, 1])
                
                with col_ing1:
                    # Agrupar ingredientes por categoria
                    ing_cat_col = _col(ingredients_df, 'category', 'type', 'ingredient_type')
                    ing_name_col = _col(ingredients_df, 'name', 'ingredient', 'ingredient_name')
                    if ing_cat_col:
                        ingredient_categories = sorted(ingredients_df[ing_cat_col].dropna().unique())
                    else:
                        ingredient_categories = []
                    selected_category = st.selectbox(
                        "Category",
                        ["All"] + ingredient_categories,
                        key=f"ing_cat_{i}"
                    )
                    
                    # Filtrar ingredientes por categoria
                    if selected_category == "All":
                        if ing_name_col:
                            available_ingredients = ingredients_df[ing_name_col].astype(str).tolist()
                        else:
                            available_ingredients = []
                    else:
                        if ing_name_col and ing_cat_col:
                            available_ingredients = ingredients_df[ingredients_df[ing_cat_col].astype(str) == str(selected_category)][ing_name_col].astype(str).tolist()
                        else:
                            available_ingredients = []
                    
                    selected_ingredient = st.selectbox(
                        "Ingredient",
                        [""] + available_ingredients,
                        format_func=lambda x: ing_label_map.get(str(x), str(x)) if str(x) else "",
                        key=f"ing_{i}"
                    )
                
                with col_ing2:
                    if selected_ingredient:
                        # Obter unidade do ingrediente
                        if ing_name_col:
                            ing_info = ingredients_df[ingredients_df[ing_name_col].astype(str) == str(selected_ingredient)].iloc[0]
                        else:
                            ing_info = pd.Series()
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
                new_recipe = build_recipe_insert_dict(
                    recipe_name=recipe_name,
                    recipe_style=recipe_style,
                    batch_volume=batch_volume,
                    efficiency=efficiency,
                    og=og,
                    fg=fg,
                    ibus=ibus,
                    ebc=ebc,
                    selected_brewery=selected_brewery,
                    brewery_name=brewery_name,
                    description=description,
                )

                # Inserir receita
                recipe_id = insert_data("recipes", new_recipe)
                
                # Add ingredientes à tabela recipe_items
                for item in ingredient_list:
                    new_recipe_item = {
                        # support both legacy (typo) and current schemas
                        'recipe_id': recipe_id,
                        'id_recipe': recipe_id,
                        'id_receipt': recipe_id,
                        'ingredient_name': item.get('name'),
                        'id_ingredient': item.get('id_ingredient'),
                        'quantity': item.get('quantity'),
                        'unit': item.get('unit'),
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
                style_col = _col(recipes_df, 'style', 'beer_style')
                og_col = _col(recipes_df, 'og', 'original_gravity', 'og_plato')
                fg_col = _col(recipes_df, 'fg', 'final_gravity', 'fg_plato')
                ibu_col = _col(recipes_df, 'ibus', 'ibu')
                name_col = _col(recipes_df, 'name', 'recipe_name')

                unique_styles = recipes_df[style_col].nunique() if style_col else 0
                st.metric("Unique Styles", unique_styles)
            with col_stats3:
                batch_col = next((c for c in ['batch_volume', 'batch_size', 'batch_l', 'volume_l', 'volume'] if c in recipes_df.columns), None)
                if batch_col is not None:
                    avg_batch_size = pd.to_numeric(recipes_df[batch_col], errors='coerce').mean()
                    st.metric("Avg Batch Size", f"{avg_batch_size:.1f}L" if pd.notna(avg_batch_size) else "N/A")
                else:
                    st.metric("Avg Batch Size", "N/A")
            with col_stats4:
                avg_abv = (((pd.to_numeric(recipes_df[og_col], errors='coerce') - pd.to_numeric(recipes_df[fg_col], errors='coerce')) * 0.524).mean() if (og_col and fg_col) else float('nan'))
                st.metric("Avg ABV", f"{avg_abv:.1f}%")
            
            # Gráfico de distribuição por estilo
            st.markdown("---")
            st.write("**Distribution by Beer Style**")
            
            style_dist = (recipes_df[style_col].value_counts() if style_col else pd.Series(dtype=int))
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
                        x=pd.to_numeric(recipes_df[og_col], errors='coerce') if og_col else None,
                        y=pd.to_numeric(recipes_df[fg_col], errors='coerce') if fg_col else None,
                        mode='markers',
                        marker=dict(size=10, color=pd.to_numeric(recipes_df[ibu_col], errors='coerce') if ibu_col else None, colorscale='Viridis'),
                        text=recipes_df[name_col] if name_col else None,
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
                abv_values = ((pd.to_numeric(recipes_df[og_col], errors='coerce') - pd.to_numeric(recipes_df[fg_col], errors='coerce')) * 0.524) if (og_col and fg_col) else pd.Series(dtype=float)
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
                    recipe_id = recipe.get('id_recipe', recipe.get('id_receipt', recipe.get('id')))
                    recipe_id_str = str(recipe_id)
                    ri_recipe_col = _col(recipe_items_df, 'recipe_id', 'id_recipe', 'id_receipt')
                    recipe_items = recipe_items_df.copy()
                    if ri_recipe_col:
                        recipe_items = recipe_items[recipe_items[ri_recipe_col].astype(str) == recipe_id_str]
                    else:
                        recipe_items = pd.DataFrame()
                    
                    total_cost = 0
                    for _, item in recipe_items.iterrows():
                        ingredients_df = data.get("ingredients", pd.DataFrame())
                        if not ingredients_df.empty:
                            ing_id_col = _col(ingredients_df, 'id_ingredient', 'id')
                            ing_name_col = _col(ingredients_df, 'name')
                            it_ing_id_col = _col(recipe_items_df, 'id_ingredient', 'ingredient_id')
                            it_ing_name_col = _col(recipe_items_df, 'ingredient_name', 'ingredient')
                            if it_ing_name_col and ing_name_col:
                                ing = ingredients_df[ingredients_df[ing_name_col].astype(str) == str(item.get(it_ing_name_col))]
                            elif it_ing_id_col and ing_id_col:
                                ing = ingredients_df[ingredients_df[ing_id_col].astype(str) == str(item.get(it_ing_id_col))]
                            else:
                                ing = pd.DataFrame()
                            if not ing.empty:
                                unit_cost = _ingredient_effective_unit_cost(ing.iloc[0])
                                total_cost += float(unit_cost or 0) * float(item.get('quantity', 0) or 0)
                    
                    batch_vol = pd.to_numeric(recipe.get('batch_volume', recipe.get('batch_size', 0)), errors='coerce')
                    batch_vol = float(batch_vol) if pd.notna(batch_vol) else 0.0

                    recipe_costs.append({
                        'Recipe': recipe['name'],
                        'Batch Volume (L)': batch_vol,
                        'Total Cost': total_cost,
                        'Cost per Liter': total_cost / batch_vol if batch_vol > 0 else 0
                    })
                
                if recipe_costs:
                    cost_df = pd.DataFrame(recipe_costs)
                    st.dataframe(cost_df.sort_values('Cost per Liter', ascending=False), use_container_width=True)
        else:
            st.info("No recipes available for analysis. Create some recipes first!")
    
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------

# -----------------------------
# -----------------------------
# Production Page (Orders + Actions)
# -----------------------------
elif page == "Production":
    st.title("🏭 Production")

    data = get_all_data()

    if not is_admin():
        st.info("👀 View-only mode: you can browse and generate reports, but you can't create/edit/delete anything.")

    # Data
    batches_df = data.get('production_batches', pd.DataFrame())
    events_df = data.get('production_events', pd.DataFrame())
    cons_df = data.get('production_consumptions', pd.DataFrame())
    keg_runs_df = data.get('production_keg_runs', pd.DataFrame())
    composites_df = data.get('composite_products', pd.DataFrame())
    composite_items_df = data.get('composite_product_items', pd.DataFrame())
    equipment_df = data.get('equipment', pd.DataFrame())
    recipes_df = data.get('recipes', pd.DataFrame())
    recipe_items_df = data.get('recipe_items', pd.DataFrame())
    breweries_df = data.get('breweries', pd.DataFrame())
    ingredients_df = data.get('ingredients', pd.DataFrame())

    # Column helpers
    b_id_col = _col(batches_df, 'id_batch', 'batch_id', 'id')
    b_recipe_id_col = _col(batches_df, 'recipe_id')
    b_recipe_name_col = _col(batches_df, 'recipe_name')
    b_planned_date_col = _col(batches_df, 'planned_date')
    b_planned_vol_col = _col(batches_df, 'planned_volume_l')
    b_stage_col = _col(batches_df, 'stage')
    b_status_col = _col(batches_df, 'status')
    b_vessel_col = _col(batches_df, 'current_vessel')
    b_rem_col = _col(batches_df, 'volume_remaining_l')
    b_loss_col = _col(batches_df, 'loss_l')
    b_code_col = _col(batches_df, 'batch_code')

    recipe_name_col = _col(recipes_df, 'name')
    recipe_id_col = _col(recipes_df, 'id_recipe', 'recipe_id', 'id')
    recipe_batch_col = _col(recipes_df, 'batch_size', 'batch_volume')

    brewery_name_col = _col(breweries_df, 'name')
    brewery_id_col = _col(breweries_df, 'id_brewery', 'brewery_id', 'id')

    eq_name_col = _col(equipment_df, 'name')
    eq_type_col = _col(equipment_df, 'type')
    eq_cap_col = _col(equipment_df, 'capacity_liters', 'capacity')

    def _brewhouse_system_options():
        """Return equipment names suitable as brewhouse / brewing system."""
        if equipment_df is None or equipment_df.empty or not eq_name_col:
            return []
        df = equipment_df.copy()
        if eq_type_col and eq_type_col in df.columns:
            t = df[eq_type_col].astype(str).str.lower()
            mask = (
                t.str.contains('brewhouse', na=False)
                | t.str.contains('brew', na=False)
                | t.str.contains('system', na=False)
                | t.str.contains('pilot', na=False)
            )
            cand = df[mask]
            if cand.empty:
                cand = df
        else:
            cand = df
        return sorted(pd.unique(cand[eq_name_col].astype(str)))


    # Helpers
    def _is_active_batch_status(s: str | None) -> bool:
        s = (s or '').strip().lower()
        return s not in {'completed', 'cancelled', 'canceled'}

    def _vessel_is_free(vessel_name: str, current_batch_id: int) -> bool:
        if not vessel_name or batches_df is None or batches_df.empty or not b_id_col or not b_vessel_col:
            return True
        for _, r in batches_df.iterrows():
            if int(r.get(b_id_col)) == int(current_batch_id):
                continue
            if str(r.get(b_vessel_col) or '') == str(vessel_name) and _is_active_batch_status(str(r.get(b_status_col) or '')):
                return False
        return True

    def _vessel_capacity_ok(vessel_name: str, volume_l: float) -> bool:
        if not vessel_name or equipment_df is None or equipment_df.empty or not (eq_name_col and eq_cap_col):
            return True
        m = equipment_df[equipment_df[eq_name_col].astype(str) == str(vessel_name)]
        if m.empty:
            return True
        try:
            cap_raw = m.iloc[0][eq_cap_col]
        except Exception:
            cap_raw = 0
        cap_l = _capacity_to_l(cap_raw, volume_l=volume_l)
        # If capacity is missing/zero, don't block the action.
        return (cap_l <= 0) or (float(volume_l) <= cap_l + 1e-9)

    def _recipe_scale_factor(recipe_id: str | None, batch_volume_l: float) -> float:
        if not recipe_id or recipes_df is None or recipes_df.empty or not recipe_batch_col or not recipe_id_col:
            return 1.0
        m = recipes_df[recipes_df[recipe_id_col].astype(str) == str(recipe_id)]
        if m.empty:
            return 1.0
        try:
            base = float(m.iloc[0][recipe_batch_col] or 0)
        except Exception:
            base = 0
        if base <= 0:
            return 1.0
        return float(batch_volume_l) / base

    # Layout
    tab_orders, tab_reports = st.tabs(["Orders & Actions", "Reports"])

    with tab_orders:
        left, right = st.columns([1, 2], gap='large')

        with left:
            st.subheader("Production Orders")

            if is_admin():
                with st.expander("➕ Create production order", expanded=True):
                    with st.form('create_production_order', clear_on_submit=True):
                        c1, c2 = st.columns(2)
                        with c1:
                            recipe_options = recipes_df[recipe_name_col].astype(str).tolist() if recipes_df is not None and not recipes_df.empty and recipe_name_col else []
                            recipe_name = st.selectbox('Beer / Recipe', recipe_options, index=0 if recipe_options else None)
                            planned_date = st.date_input('Planned brew date', date.today())
                            planned_volume = st.number_input('Planned volume (L)', min_value=0.0, value=100.0, step=10.0)
                        with c2:
                            brewery_options = breweries_df[brewery_name_col].astype(str).tolist() if breweries_df is not None and not breweries_df.empty and brewery_name_col else []
                            brewery_name = st.selectbox('Brewery / location', brewery_options, index=0 if brewery_options else None)
                            brewhouse_opts = _brewhouse_system_options()
                            brewhouse_choice = st.selectbox(
                                'Brewhouse / system',
                                [''] + brewhouse_opts + (['(Custom...)'] if brewhouse_opts else ['(Custom...)']),
                                index=0,
                            )
                            if brewhouse_choice == '(Custom...)':
                                brewhouse = st.text_input('Custom brewhouse / system', placeholder='e.g., Pilot system')
                            else:
                                brewhouse = brewhouse_choice
                            st.caption("Batch code: you can type one manually, or leave it blank to auto-assign (uses the Batch ID).")
                            batch_code = st.text_input("Batch code (optional)", value="", placeholder="Leave blank to auto-assign", key="prod_create_batch_code")
                        notes = st.text_area('Notes')
                        submit = st.form_submit_button('Create order', type='primary', use_container_width=True)

                    if submit:
                        require_admin_action()

                        # Validate manual batch code (when provided)
                        bc_in = (batch_code or '').strip()
                        if bc_in:
                            try:
                                if batches_df is not None and not batches_df.empty and (b_code_col or 'batch_code') in batches_df.columns and b_id_col:
                                    if (batches_df[(b_code_col or 'batch_code')].astype(str) == bc_in).any():
                                        st.error("Batch code already exists. Please choose a unique code.")
                                        st.stop()
                            except Exception:
                                pass

                        rid = None
                        if recipe_name and recipes_df is not None and not recipes_df.empty and recipe_name_col and recipe_id_col:
                            m = recipes_df[recipes_df[recipe_name_col].astype(str) == str(recipe_name)]
                            if not m.empty:
                                rid = str(m.iloc[0][recipe_id_col])
                        bid = None
                        if brewery_name and breweries_df is not None and not breweries_df.empty and brewery_name_col and brewery_id_col:
                            m = breweries_df[breweries_df[brewery_name_col].astype(str) == str(brewery_name)]
                            if not m.empty:
                                bid = str(m.iloc[0][brewery_id_col])

                        batch_id = insert_data('production_batches', {
                            'batch_code': batch_code,
                            'recipe_id': rid,
                            'recipe_name': recipe_name,
                            'brewery_id': bid or brewery_name,
                            'planned_date': planned_date,
                            'planned_volume_l': float(planned_volume),
                            'brewhouse': brewhouse,
                            'status': 'Planned',
                            'stage': 'Production Order',
                            'current_vessel': '',
                            'volume_remaining_l': float(planned_volume),
                            'loss_l': 0,
                            'notes': notes,
                            'created_by': st.session_state.get('auth_user', 'admin'),
                        })

                        # Auto-assign batch code only when user left it blank
                        if not str(batch_code or '').strip():
                            try:
                                _id_col = b_id_col if 'b_id_col' in locals() and b_id_col else 'id'
                                update_data('production_batches', {'batch_code': str(batch_id)}, f"{_id_col} = :id", {'id': batch_id})
                            except Exception:
                                pass

                        # Mark calendar
                        try:
                            insert_data('calendar_events', {
                                'title': f"Production: {recipe_name}",
                                'event_type': 'Production',
                                'start_date': planned_date,
                                'end_date': planned_date,
                                'equipment': brewhouse,
                                'batch_id': str(batch_id),
                                'notes': notes,
                                'created_by': st.session_state.get('auth_user', 'admin'),
                            })
                        except Exception:
                            pass

                        st.success('✅ Production order created.')
                        st.rerun()

            st.markdown('---')
            if batches_df is None or batches_df.empty:
                st.info('No production orders yet.')
                selected_batch = None
            else:
                # Sort by planned date desc when available
                view = batches_df.copy()
                if b_planned_date_col:
                    try:
                        view[b_planned_date_col] = pd.to_datetime(view[b_planned_date_col], errors='coerce')
                        view = view.sort_values(b_planned_date_col, ascending=False)
                    except Exception:
                        pass

                def _label(r):
                    bid = r.get(b_id_col)
                    pdx = r.get(b_planned_date_col)
                    pdx_s = str(pdx)[:10] if pdx is not None else ''
                    code = (r.get(b_code_col) if b_code_col else r.get("batch_code"))
                    code_s = str(code) if code is not None else ""
                    code_s = code_s.strip()
                    code_part = f" ({code_s})" if code_s else ""
                    return f"#{bid}{code_part} — {pdx_s} — {r.get('recipe_name','')} — {r.get(b_stage_col,'')}"

                records = view.to_dict('records')
                # Show only active (not completed/cancelled) orders here.
                active_records = records
                try:
                    if b_status_col and b_status_col in view.columns:
                        active_records = [r for r in records if _is_active_batch_status(str(r.get(b_status_col) or ''))]
                except Exception:
                    active_records = records

                if not active_records:
                    st.info('No active production orders. Finished/cancelled orders are available in the "Reports" tab.')
                    selected_batch = None
                else:
                    selected_batch = st.selectbox('Select order', active_records, format_func=_label)

        with right:
            st.subheader("Order details")

            if not selected_batch:
                st.info("Select a production order to see details and add actions.")
            else:
                batch_id = int(selected_batch.get(b_id_col))
                planned_vol = float(selected_batch.get(b_planned_vol_col) or 0)
                remaining_vol = float(selected_batch.get(b_rem_col) or planned_vol or 0)
                stage = str(selected_batch.get(b_stage_col) or '')
                status = str(selected_batch.get(b_status_col) or '')
                current_vessel = str(selected_batch.get(b_vessel_col) or '')

                c1, c2, c3, c4 = st.columns(4)
                c1.metric('Planned (L)', f"{planned_vol:g}")
                c2.metric('Remaining (L)', f"{remaining_vol:g}")
                c3.metric('Stage', stage or '-')
                c4.metric('Vessel', current_vessel or '-')

                st.markdown('---')
                # Tabs: View vs Actions
                if is_admin():
                    view_tab, actions_tab = st.tabs(["View", "Actions"])
                else:
                    view_tab, = st.tabs(["View"])
                    actions_tab = None

                with view_tab:
                    # ---- BATCH CODE (editable) ----
                    try:
                        _bcode_val = str(selected_batch.get(b_code_col) if b_code_col else selected_batch.get('batch_code') or '')
                    except Exception:
                        _bcode_val = ''
                    st.markdown("#### Batch code")
                    if _bcode_val.strip():
                        st.write(f"**Current:** `{_bcode_val.strip()}`")
                    else:
                        st.write("**Current:** (blank)")

                    if is_admin():
                        with st.form(f"edit_batch_code_{batch_id}", clear_on_submit=False):
                            new_code = st.text_input(
                                "Edit batch code",
                                value=_bcode_val.strip(),
                                max_chars=50,
                                help="Tip: use a unique, human-friendly code (e.g. 2026-01-PILS).",
                            )
                            save_code = st.form_submit_button("Save batch code", type="primary", use_container_width=True)

                        if save_code:
                            require_admin_action()
                            nc = (new_code or '').strip()
                            if not nc:
                                st.error("Batch code can't be empty.")
                                st.stop()
                            # Uniqueness check
                            try:
                                if batches_df is not None and not batches_df.empty and b_id_col and b_code_col:
                                    _dups = batches_df[
                                        (batches_df[b_id_col].astype(int) != int(batch_id))
                                        & (batches_df[b_code_col].astype(str) == nc)
                                    ]
                                    if not _dups.empty:
                                        st.error("This batch code is already used by another batch. Choose a unique code.")
                                        st.stop()
                            except Exception:
                                pass

                            update_data(
                                'production_batches',
                                {(b_code_col or 'batch_code'): nc},
                                f"{b_id_col} = :id",
                                {'id': batch_id},
                            )
                            st.success("✅ Batch code updated.")
                            st.rerun()

                    st.markdown('---')
                    st.subheader("Timeline")
                    if events_df is None or events_df.empty:
                        st.caption("No events yet.")
                    else:
                        e_bid = _col(events_df, 'batch_id')
                        e_date = _col(events_df, 'event_date')
                        if e_bid:
                            ev = events_df[events_df[e_bid] == batch_id].copy()
                            if not ev.empty:
                                if e_date:
                                    try:
                                        ev[e_date] = pd.to_datetime(ev[e_date], errors='coerce')
                                        ev = ev.sort_values(e_date)
                                    except Exception:
                                        pass
                                st.dataframe(ev[[c for c in ev.columns if c not in ('created_date',)]], use_container_width=True)
                            else:
                                st.caption("No events for this batch.")

                    # Fermentation readings (multiple readings over time)
                    st.markdown("### Fermentation readings")
                    def _safe_json_load(_s):
                        try:
                            if _s is None:
                                return {}
                            if isinstance(_s, dict):
                                return _s
                            _ss = str(_s)
                            if not _ss or _ss.strip().lower() in {"nan", "none"}:
                                return {}
                            return json.loads(_ss)
                        except Exception:
                            return {}

                    try:
                        _etype = _col(events_df, 'event_type', 'type')
                        _emeta = _col(events_df, 'meta')
                        if events_df is None or events_df.empty or not e_bid or not _etype:
                            rview = pd.DataFrame()
                        else:
                            rview = events_df[events_df[e_bid] == batch_id].copy()
                            rview = rview[rview[_etype].astype(str).str.lower() == 'reading']
                            if not rview.empty and e_date:
                                rview[e_date] = pd.to_datetime(rview[e_date], errors='coerce')
                                rview = rview.sort_values(e_date)
                            if not rview.empty and _emeta and _emeta in rview.columns:
                                metas = rview[_emeta].apply(_safe_json_load)
                                rview['Gravity (°P)'] = metas.apply(lambda d: d.get('gravity_plato') if isinstance(d, dict) else None)
                                rview['Temp (°C)'] = metas.apply(lambda d: d.get('temp_c') if isinstance(d, dict) else None)
                                rview['pH'] = metas.apply(lambda d: d.get('ph') if isinstance(d, dict) else None)
                                rview['Pressure (bar)'] = metas.apply(lambda d: d.get('pressure_bar') if isinstance(d, dict) else None)
                            elif not rview.empty:
                                for _c in ('Gravity (°P)', 'Temp (°C)', 'pH', 'Pressure (bar)'):
                                    if _c not in rview.columns:
                                        rview[_c] = None
                    except Exception:
                        rview = pd.DataFrame()

                    if rview is None or rview.empty:
                        st.caption("No fermentation readings yet.")
                    else:
                        # Chart: gravity + temperature over time (two series)
                        try:
                            if e_date and e_date in rview.columns:
                                chart_df = rview[[e_date, 'Gravity (°P)', 'Temp (°C)']].copy()
                                chart_df = chart_df.rename(columns={e_date: 'Date'}).set_index('Date')
                                for _c in list(chart_df.columns):
                                    chart_df[_c] = pd.to_numeric(chart_df[_c], errors='coerce')
                                st.line_chart(chart_df)

                                # Optional: pressure chart
                                try:
                                    if 'Pressure (bar)' in rview.columns:
                                        p_df = rview[[e_date, 'Pressure (bar)']].copy()
                                        p_df = p_df.rename(columns={e_date: 'Date'}).set_index('Date')
                                        p_df['Pressure (bar)'] = pd.to_numeric(p_df['Pressure (bar)'], errors='coerce')
                                        if p_df['Pressure (bar)'].notna().any():
                                            st.line_chart(p_df)
                                except Exception:
                                    pass

                        except Exception:
                            pass

                        cols_show = []
                        if e_date and e_date in rview.columns:
                            cols_show.append(e_date)
                        for _c in ('Gravity (°P)', 'Temp (°C)', 'pH', 'Pressure (bar)'):
                            if _c in rview.columns:
                                cols_show.append(_c)
                        st.dataframe(rview[cols_show], use_container_width=True)

                    st.markdown('---')
                    st.subheader("Material consumption")
                    if cons_df is None or cons_df.empty:
                        st.caption("No consumption records yet.")
                    else:
                        c_bid = _col(cons_df, 'batch_id')
                        if c_bid:
                            cview = cons_df[cons_df[c_bid] == batch_id].copy()
                            if not cview.empty:
                                st.dataframe(cview[[c for c in cview.columns if c not in ('created_date',)]], use_container_width=True)
                            else:
                                st.caption("No consumption records for this batch.")

                    st.markdown('---')
                    st.subheader("Kegging runs")
                    if keg_runs_df is None or keg_runs_df.empty:
                        st.caption("No kegging runs yet.")
                    else:
                        k_bid = _col(keg_runs_df, 'batch_id')
                        if k_bid:
                            kview = keg_runs_df[keg_runs_df[k_bid] == batch_id].copy()
                            if not kview.empty:
                                st.dataframe(kview[[c for c in kview.columns if c not in ('created_date',)]], use_container_width=True)
                            else:
                                st.caption("No kegging runs for this batch.")

                    st.markdown('---')
                    st.subheader("Reports")
                    # Production Report is available only after Brew is recorded for this batch
                    brew_done = False
                    brew_dt = None

                    # Detect via batch fields
                    try:
                        if float(selected_batch.get('brewed_volume_l') or 0) > 0 or float(selected_batch.get('og') or 0) > 0 or float(selected_batch.get('efficiency') or 0) > 0:
                            brew_done = True
                    except Exception:
                        pass

                    # Detect via events table
                    try:
                        if not brew_done and events_df is not None and not events_df.empty:
                            _e_bid = _col(events_df, 'batch_id')
                            _e_type = _col(events_df, 'event_type', 'type')
                            _e_date = _col(events_df, 'event_date')
                            if _e_bid and _e_type:
                                _ev2 = events_df[events_df[_e_bid] == batch_id].copy()
                                if not _ev2.empty:
                                    _brew_rows = _ev2[_ev2[_e_type].astype(str).str.lower() == 'brew']
                                    if not _brew_rows.empty:
                                        brew_done = True
                                        if _e_date and _e_date in _brew_rows.columns:
                                            try:
                                                brew_dt = pd.to_datetime(_brew_rows.iloc[-1][_e_date], errors='coerce')
                                            except Exception:
                                                brew_dt = None
                    except Exception:
                        pass

                    if not brew_done:
                        st.info("Production Report becomes available after you record the **Brew** action.")
                    else:
                        if brew_dt is not None and not pd.isna(brew_dt):
                            st.caption(f"Brew recorded on {str(brew_dt)[:10]}.")
                        _pdf_state_key = f"prod_pdf_bytes_{batch_id}"
                        if st.button('Generate Production Report (PDF)', key=f'gen_prod_pdf_{batch_id}', use_container_width=True):
                            st.session_state[_pdf_state_key] = generate_production_report_pdf_bytes(batch_id)
                        if _pdf_state_key in st.session_state:
                            st.download_button(
                                '⬇️ Download Production Report',
                                data=st.session_state[_pdf_state_key],
                                file_name=f'production_report_batch_{batch_id}.pdf',
                                mime='application/pdf',
                                use_container_width=True,
                                key=f'dl_prod_pdf_{batch_id}',
                            )

                if actions_tab is not None:
                    with actions_tab:
                        st.subheader("Actions")
                        
                        # Detect whether Brew already exists for this batch (used by Undo Brew + disabling Brew)
                        brew_event_exists = False
                        brew_event_id = None
                        brew_event_date = None
                        try:
                            if events_df is not None and not events_df.empty:
                                _e_bid = _col(events_df, 'batch_id')
                                _e_type = _col(events_df, 'event_type', 'type')
                                _e_id = _col(events_df, 'id_prod_event', 'id_event', 'id')
                                _e_date = _col(events_df, 'event_date')
                                if _e_bid and _e_type:
                                    _ev0 = events_df[events_df[_e_bid] == batch_id].copy()
                                    if not _ev0.empty:
                                        _brew0 = _ev0[_ev0[_e_type].astype(str).str.lower() == 'brew']
                                        if not _brew0.empty:
                                            brew_event_exists = True
                                            if _e_id and _e_id in _brew0.columns:
                                                try:
                                                    brew_event_id = int(_brew0.iloc[-1][_e_id])
                                                except Exception:
                                                    brew_event_id = None
                                            if _e_date and _e_date in _brew0.columns:
                                                try:
                                                    brew_event_date = pd.to_datetime(_brew0.iloc[-1][_e_date], errors='coerce')
                                                except Exception:
                                                    brew_event_date = None
                        except Exception:
                            pass

                        # ---- CANCEL ORDER ----
                        st.markdown("### 🛑 Cancel Order")
                        _status_low = (str(status or '').strip().lower())
                        if _status_low in {'cancelled', 'canceled'}:
                            # Show reason if available
                            try:
                                cr = str(selected_batch.get('cancelled_reason') or '').strip()
                            except Exception:
                                cr = ''
                            if cr:
                                st.info(f"This order is already cancelled. Reason: {cr}")
                            else:
                                st.info("This order is already cancelled.")
                        elif _status_low in {'completed', 'done', 'finished'}:
                            st.info("Completed orders can't be cancelled.")
                        else:
                            # Only allow cancelling if no production events exist yet (safe: no inventory to revert)
                            try:
                                _e_bid2 = _col(events_df, 'batch_id')
                                _has_events = False
                                if events_df is not None and not events_df.empty and _e_bid2:
                                    _has_events = not events_df[events_df[_e_bid2] == batch_id].empty
                            except Exception:
                                _has_events = False

                            if _has_events:
                                st.warning(
                                    "Cancel Order is disabled because this batch already has actions recorded. "
                                    "(Cancelling after actions would require inventory/process reversal.)"
                                )
                            else:
                                with st.form(f"cancel_order_form_{batch_id}", clear_on_submit=True):
                                    reason = st.text_area(
                                        "Reason (max 200 characters)*",
                                        max_chars=200,
                                        height=80,
                                        placeholder="Why are you cancelling this production order?",
                                    )
                                    submit_cancel = st.form_submit_button(
                                        "🛑 Cancel order",
                                        type="primary",
                                        use_container_width=True,
                                    )

                                if submit_cancel:
                                    require_admin_action()
                                    r = (reason or '').strip()
                                    if not r:
                                        st.error("Cancellation reason is required (max 200 characters).")
                                        st.stop()
                                    if len(r) > 200:
                                        st.error("Cancellation reason must be at most 200 characters.")
                                        st.stop()

                                    now_ts = datetime.now()

                                    # Update batch
                                    # Keep notes, but append cancellation note for quick visibility
                                    try:
                                        existing_notes = str(selected_batch.get('notes') or '').strip()
                                    except Exception:
                                        existing_notes = ''
                                    cancel_line = f"Cancelled on {now_ts.date().isoformat()}: {r}"
                                    merged_notes = (existing_notes + "\n" + cancel_line).strip() if existing_notes else cancel_line

                                    update_data(
                                        'production_batches',
                                        {
                                            'status': 'Cancelled',
                                            'stage': 'Cancelled',
                                            'current_vessel': '',
                                            'cancelled_reason': r,
                                            'cancelled_by': st.session_state.get('auth_user', 'admin'),
                                            'cancelled_at': now_ts,
                                            'notes': merged_notes,
                                        },
                                        f"{b_id_col} = :id",
                                        {'id': batch_id},
                                    )

                                    # Log as production event
                                    try:
                                        insert_data('production_events', {
                                            'batch_id': int(batch_id),
                                            'event_type': 'Cancel Order',
                                            'event_date': now_ts,
                                            'from_vessel': str(current_vessel or ''),
                                            'to_vessel': '',
                                            'notes': r,
                                            'created_by': st.session_state.get('auth_user', 'admin'),
                                            'meta': json.dumps({'action': 'Cancel Order', 'reason': r}, ensure_ascii=False),
                                        })
                                    except Exception:
                                        pass

                                    st.success("✅ Production order cancelled.")
                                    st.rerun()

                        st.markdown("---")

                        # ---- UNDO BREW ----
                        if brew_event_exists and brew_event_id is not None:
                            st.markdown("### 🔁 Undo Brew")
                            try:
                                _ev2 = events_df.copy() if events_df is not None else pd.DataFrame()
                                _bid = _col(_ev2, 'batch_id')
                                _typ = _col(_ev2, 'event_type')
                                _eid = _col(_ev2, 'id_prod_event', 'prod_event_id', 'id')
                                if _bid and _typ and not _ev2.empty:
                                    _sub = _ev2[_ev2[_bid] == batch_id]
                                    _other = _sub[_sub[_typ].astype(str).str.lower() != 'brew']
                                else:
                                    _other = pd.DataFrame()
                            except Exception:
                                _other = pd.DataFrame()

                            if _other is not None and not _other.empty:
                                other_types = sorted(_other[_col(_other, 'event_type')].astype(str).unique().tolist()) if _col(_other, 'event_type') else []
                                st.info("Undo Brew is disabled because this batch already has other actions recorded: " + ", ".join(other_types))
                            else:
                                _confirm_key = f"undo_brew_confirm_{batch_id}"
                                if not st.session_state.get(_confirm_key, False):
                                    if st.button("↩️ Undo Brew", key=f"undo_brew_btn_{batch_id}", use_container_width=True, type="secondary"):
                                        st.session_state[_confirm_key] = True
                                        st.rerun()
                                else:
                                    st.warning("This will delete the **Brew** event, remove its logged consumptions, and return all consumed ingredients back to inventory.")
                                    c_undo1, c_undo2 = st.columns(2)
                                    with c_undo1:
                                        if st.button("✅ Yes, undo Brew", key=f"undo_brew_yes_{batch_id}", use_container_width=True, type="primary"):
                                            require_admin_action()
                                            data_u = get_all_data()
                                            cons_u = data_u.get('production_consumptions', pd.DataFrame())
                                            c_bid = _col(cons_u, 'batch_id')
                                            c_eid = _col(cons_u, 'prod_event_id')
                                            c_ing = _col(cons_u, 'ingredient_name')
                                            c_qty = _col(cons_u, 'quantity')
                                            # Return stock
                                            if cons_u is not None and not cons_u.empty and c_eid and c_ing and c_qty:
                                                try:
                                                    sub = cons_u.copy()
                                                    # Robust numeric filtering (handles DBs where ids are loaded as strings/objects)
                                                    if c_bid and c_bid in sub.columns:
                                                        sub = sub[pd.to_numeric(sub[c_bid], errors='coerce') == float(batch_id)]
                                                    if c_eid and c_eid in sub.columns:
                                                        sub = sub[pd.to_numeric(sub[c_eid], errors='coerce') == float(brew_event_id)]
                                                except Exception:
                                                    sub = cons_u.copy()
                                                if sub is None or sub.empty:
                                                    # Fallback: some DB backends may not return the Brew event id during insert,
                                                    # leaving prod_event_id NULL in the consumption lines. In that case, recover by batch + meta.
                                                    try:
                                                        sub_f = cons_u.copy()
                                                        if c_bid and c_bid in sub_f.columns:
                                                            sub_f = sub_f[pd.to_numeric(sub_f[c_bid], errors='coerce') == float(batch_id)]
                                                        # Prefer NULL prod_event_id (or matching id if present)
                                                        if c_eid and c_eid in sub_f.columns:
                                                            ce = pd.to_numeric(sub_f[c_eid], errors='coerce')
                                                            sub_f = sub_f[ce.isna() | (ce == float(brew_event_id))]
                                                        mcol = _col(sub_f, 'meta')
                                                        if mcol and mcol in sub_f.columns:
                                                            sub_f = sub_f[sub_f[mcol].astype(str).str.contains('"event"\s*:\s*"Brew"', na=False)]
                                                        if sub_f is not None and not sub_f.empty:
                                                            sub = sub_f
                                                    except Exception:
                                                        pass

                                                if sub is None or sub.empty:
                                                    st.error("I couldn't find any logged consumption lines for this Brew, so I can't return inventory safely. **Undo Brew was aborted** — nothing was deleted.")
                                                    st.session_state[_confirm_key] = False
                                                    st.stop()

                                                for _, rr in sub.iterrows():
                                                    ingn = str(rr.get(c_ing) or '')
                                                    try:
                                                        qty = float(rr.get(c_qty) or 0)
                                                    except Exception:
                                                        qty = 0.0
                                                    if ingn and qty > 0:
                                                        adjust_stock_for_ingredient(
                                                            get_all_data(),
                                                            ingn,
                                                            float(qty),
                                                            reason='Undo Brew',
                                                            source=f"Batch #{batch_id} (Undo Brew)",
                                                            destination='Inventory',
                                                            ref_table='production_events',
                                                            ref_id=int(brew_event_id),
                                                            batch_id=int(batch_id),
                                                            prod_event_id=int(brew_event_id),
                                                        )
                                            # Delete consumptions + event
                                            try:
                                                delete_data('production_consumptions', "prod_event_id = :eid AND batch_id = :bid", {'eid': int(brew_event_id), 'bid': int(batch_id)})
                                            except Exception:
                                                try:
                                                    delete_data('production_consumptions', "prod_event_id = :eid", {'eid': int(brew_event_id)})
                                                except Exception:
                                                    pass
                                            try:
                                                # Delete Brew event
                                                ev_df3 = get_all_data().get('production_events', pd.DataFrame())
                                                e_id_col2 = _col(ev_df3, 'id_prod_event', 'id')
                                                if e_id_col2:
                                                    delete_data('production_events', f"{e_id_col2} = :id", {'id': int(brew_event_id)})
                                                else:
                                                    delete_data('production_events', "id_prod_event = :id", {'id': int(brew_event_id)})
                                            except Exception:
                                                pass

                                            # Reset batch fields back to pre-brew state
                                            try:
                                                planned_col = _col(batches_df, 'planned_volume_l', 'planned_volume', 'batch_volume_l', 'volume_l')
                                                planned_vol = float(selected_batch.get(planned_col) or 0) if planned_col else float(selected_batch.get('planned_volume_l') or 0)
                                            except Exception:
                                                planned_vol = 0.0
                                            update_data('production_batches', {
                                                'status': 'Planned',
                                                'stage': 'Production Order',
                                                'current_vessel': '',
                                                'volume_remaining_l': planned_vol if planned_vol > 0 else None,
                                                'brewed_volume_l': None,
                                                'og': None,
                                                'efficiency': None,
                                            }, f"{b_id_col} = :id", {'id': batch_id})

                                            # Clear any cached PDF bytes
                                            try:
                                                _pdf_state_key = f"prod_pdf_bytes_{batch_id}"
                                                if _pdf_state_key in st.session_state:
                                                    del st.session_state[_pdf_state_key]
                                            except Exception:
                                                pass

                                            st.session_state[_confirm_key] = False
                                            st.success("✅ Brew undone. Ingredients were returned to inventory.")
                                            st.rerun()
                                    with c_undo2:
                                        if st.button("Cancel", key=f"undo_brew_cancel_{batch_id}", use_container_width=True):
                                            st.session_state[_confirm_key] = False
                                            st.rerun()

                            st.markdown("---")


                        # ---- RECOVER ORPHANED BREW CONSUMPTIONS (safety net) ----
                        # If a Brew was undone or partially deleted in older versions and the consumption rows remain,
                        # this lets admins restore inventory + log movements.
                        if not brew_event_exists:
                            try:
                                data_r = get_all_data()
                                cons_r = data_r.get('production_consumptions', pd.DataFrame())
                                if cons_r is not None and not cons_r.empty:
                                    c_bid2 = _col(cons_r, 'batch_id')
                                    c_meta2 = _col(cons_r, 'meta')
                                    c_eid2 = _col(cons_r, 'prod_event_id')
                                    c_ing2 = _col(cons_r, 'ingredient_name')
                                    c_qty2 = _col(cons_r, 'quantity')
                                    c_pk2 = _col(cons_r, 'id_consumption', 'id')
                                    sub_orph = cons_r.copy()
                                    if c_bid2 and c_bid2 in sub_orph.columns:
                                        sub_orph = sub_orph[pd.to_numeric(sub_orph[c_bid2], errors='coerce') == float(batch_id)]
                                    if c_meta2 and c_meta2 in sub_orph.columns:
                                        sub_orph = sub_orph[sub_orph[c_meta2].astype(str).str.contains('"event"\s*:\s*"Brew"', case=False, regex=True, na=False)]
                                    # pick the most recent prod_event_id if present
                                    if sub_orph is not None and not sub_orph.empty and c_eid2 and c_eid2 in sub_orph.columns:
                                        pe = pd.to_numeric(sub_orph[c_eid2], errors='coerce')
                                        if pe.notna().any():
                                            latest_eid = int(pe.dropna().max())
                                            sub_orph = sub_orph[pd.to_numeric(sub_orph[c_eid2], errors='coerce') == float(latest_eid)]
                                        else:
                                            latest_eid = None
                                    else:
                                        latest_eid = None

                                    if sub_orph is not None and not sub_orph.empty:
                                        st.markdown("### 🛠️ Recover inventory (orphaned Brew)")
                                        st.caption("Found Brew consumption rows without a Brew event. You can restore inventory and log Stock History.")
                                        rec_key = f"recover_orphan_brew_{batch_id}"
                                        if not st.session_state.get(rec_key, False):
                                            if st.button("Restore inventory from orphaned Brew rows", key=f"recover_orphan_btn_{batch_id}", use_container_width=True):
                                                st.session_state[rec_key] = True
                                                st.rerun()
                                        else:
                                            st.warning("This will **add back** the consumed quantities to inventory and then delete those orphaned consumption rows.")
                                            c1r, c2r = st.columns(2)
                                            with c1r:
                                                if st.button("✅ Yes, restore inventory", key=f"recover_orphan_yes_{batch_id}", use_container_width=True, type="primary"):
                                                    require_admin_action()
                                                    # Return stock + log movement
                                                    if c_ing2 and c_qty2:
                                                        for _, rr in sub_orph.iterrows():
                                                            ingn = str(rr.get(c_ing2) or '')
                                                            try:
                                                                qty = float(rr.get(c_qty2) or 0)
                                                            except Exception:
                                                                qty = 0.0
                                                            if ingn and qty > 0:
                                                                adjust_stock_for_ingredient(
                                                                    get_all_data(),
                                                                    ingn,
                                                                    float(qty),
                                                                    reason='Recovery (orphaned Brew)',
                                                                    source=f"Batch #{batch_id} (Recovery)",
                                                                    destination='Inventory',
                                                                    ref_table='production_consumptions',
                                                                    ref_id=int(latest_eid) if latest_eid is not None else None,
                                                                    batch_id=int(batch_id),
                                                                    prod_event_id=int(latest_eid) if latest_eid is not None else None,
                                                                )
                                                    # Delete those consumption rows by PK (safe)
                                                    if c_pk2 and c_pk2 in sub_orph.columns:
                                                        for _id in sub_orph[c_pk2].dropna().tolist():
                                                            try:
                                                                delete_data('production_consumptions', f"{c_pk2} = :id", {'id': int(_id)})
                                                            except Exception:
                                                                pass
                                                    st.session_state[rec_key] = False
                                                    st.success("✅ Inventory restored from orphaned Brew rows.")
                                                    st.rerun()
                                            with c2r:
                                                if st.button("Cancel", key=f"recover_orphan_cancel_{batch_id}", use_container_width=True):
                                                    st.session_state[rec_key] = False
                                                    st.rerun()
                            except Exception:
                                pass

                        action = st.selectbox("Add action", [
                            'Brew', 'Reading', 'Dry Hop', 'Add Adjunct', 'Conditioning', 'Transfer', 'Kegging', 'Finish Batch'
                        ], key=f'action_type_{batch_id}')

                        # ---- BREW ----
                        if action == 'Brew':
                            # Prevent accidental double-consumption: Brew already recorded will disable "Record Brew"
                            # (brew_event_exists / brew_event_id / brew_event_date computed above)
                            if brew_event_exists:
                                msg = "⚠️ Brew already recorded for this batch."
                                if brew_event_date is not None and not pd.isna(brew_event_date):
                                    msg += f" (Date: {str(brew_event_date)[:10]})"
                                st.warning(msg)
                                st.caption("Brew is disabled to avoid consuming inventory twice. You can still correct OG/Efficiency/Volume below (no inventory changes).")

                                with st.form(f'brew_update_stats_{batch_id}', clear_on_submit=False):
                                    c1, c2, c3 = st.columns(3)
                                    with c1:
                                        new_vol = st.number_input('Brewed volume (L)', min_value=0.0, value=float(selected_batch.get('brewed_volume_l') or planned_vol or 0), step=1.0)
                                    with c2:
                                        new_og = st.number_input('OG (°P)', min_value=0.0, value=float(selected_batch.get('og') or 0), step=0.1)
                                    with c3:
                                        new_eff = st.number_input('Efficiency (%)', min_value=0.0, max_value=100.0, value=float(selected_batch.get('efficiency') or 0), step=0.5)
                                    notes2 = st.text_area('Notes (optional)', height=80)
                                    upd = st.form_submit_button('Update Brew Stats', type='primary', use_container_width=True)

                                if upd:
                                    require_admin_action()
                                    update_data('production_batches', {
                                        'brewed_volume_l': float(new_vol),
                                        'volume_remaining_l': float(new_vol),
                                        'og': float(new_og) if new_og is not None else None,
                                        'efficiency': float(new_eff) if new_eff is not None else None,
                                    }, f"{b_id_col} = :id", {'id': batch_id})

                                    if brew_event_id is not None:
                                        try:
                                            update_data('production_events', {
                                                'meta': json.dumps({
                                                    'action': 'Brew',
                                                    'og': float(new_og) if new_og is not None else None,
                                                    'efficiency': float(new_eff) if new_eff is not None else None,
                                                    'volume_l': float(new_vol),
                                                    'stats_updated': True,
                                                    'notes_update': str(notes2 or ''),
                                                }, ensure_ascii=False)
                                            }, "id_prod_event = :eid", {'eid': brew_event_id})
                                        except Exception:
                                            pass

                                    st.success("✅ Brew stats updated.")
                                    st.rerun()
                            # Equipment options
                            eq_names = equipment_df[eq_name_col].astype(str).tolist() if equipment_df is not None and not equipment_df.empty and eq_name_col else []
                            fermenter_opts = eq_names
                            if eq_type_col and equipment_df is not None and not equipment_df.empty:
                                fm = equipment_df[equipment_df[eq_type_col].astype(str).str.lower().str.contains('ferment', na=False)]
                                fermenter_opts = fm[eq_name_col].astype(str).tolist() if not fm.empty else eq_names

                            
                            # Defaults (prefer existing batch values; fallback to recipe values when available)
                            default_og = 0.0
                            default_eff = 0.0
                            try:
                                default_og = float(selected_batch.get('og') or 0)
                            except Exception:
                                default_og = 0.0
                            try:
                                default_eff = float(selected_batch.get('efficiency') or 0)
                            except Exception:
                                default_eff = 0.0
                            if (default_og <= 0 or default_eff <= 0) and recipes_df is not None and not recipes_df.empty:
                                try:
                                    rid0 = str(selected_batch.get(b_recipe_id_col) or '')
                                    if rid0 and recipe_id_col:
                                        mrec = recipes_df[recipes_df[recipe_id_col].astype(str) == rid0]
                                        if not mrec.empty:
                                            rrow = mrec.iloc[0]
                                            if default_og <= 0:
                                                og_col = _col(recipes_df, 'og', 'original_gravity', 'og_plato')
                                                if og_col:
                                                    default_og = float(rrow.get(og_col) or 0)
                                            if default_eff <= 0:
                                                eff_col = _col(recipes_df, 'efficiency', 'brewhouse_efficiency')
                                                if eff_col:
                                                    default_eff = float(rrow.get(eff_col) or 0)
                                except Exception:
                                    pass

                            with st.form(f'brew_action_{batch_id}', clear_on_submit=True):
                                r1, r2, r3, r4 = st.columns(4)
                                with r1:
                                    brew_date = st.date_input('Brew date', date.today())
                                with r2:
                                    actual_volume = st.number_input('Volume (L)', min_value=0.0, value=float(planned_vol or 0), step=1.0)
                                with r3:
                                    og_brew = st.number_input('OG (°P)', min_value=0.0, value=float(default_og or 0), step=0.1)
                                with r4:
                                    efficiency_brew = st.number_input('Efficiency (%)', min_value=0.0, max_value=100.0, value=float(default_eff or 0), step=0.5)

                                c1, c2 = st.columns(2)
                                with c1:
                                    fermenter = st.selectbox('Fermenter', fermenter_opts if fermenter_opts else [''], index=0)
                                with c2:
                                    brewery = st.text_input('Brewery / location', value=str(selected_batch.get('brewery_id','') or ''))

                                notes = st.text_area('Notes', height=90)

                                st.markdown("**Ingredient consumption**")
                                st.caption("You can **skip** ingredients, substitute ingredients, or add extra ingredients for this brew. Quantities entered here will be used for batch costing.")

                                rid = str(selected_batch.get(b_recipe_id_col) or '')
                                scale = _recipe_scale_factor(rid, float(actual_volume))

                                # Ingredient options
                                ing_name_col = _col(ingredients_df, 'name')
                                all_ing_opts = []
                                if ingredients_df is not None and not ingredients_df.empty and ing_name_col:
                                    all_ing_opts = sorted(set(ingredients_df[ing_name_col].astype(str).dropna().tolist()))

                                # UI-only labels (Manufacturer — Name)
                                ing_label_map = _ingredient_name_label_map(ingredients_df)

                                ri_recipe_col = _col(recipe_items_df, 'recipe_id', 'id_recipe')
                                ri_ing_col = _col(recipe_items_df, 'ingredient_name', 'ingredient')
                                ri_qty_col = _col(recipe_items_df, 'quantity')
                                ri_unit_col = _col(recipe_items_df, 'unit')
                                ri_id_col = _col(recipe_items_df, 'id_recipe_item', 'id')

                                items = recipe_items_df.copy() if recipe_items_df is not None else pd.DataFrame()
                                if not items.empty and ri_recipe_col and rid:
                                    items = items[items[ri_recipe_col].astype(str) == str(rid)]

                                confirmations = []  # (checked, chosen_ing, qty, unit, original_ing)
                                if items.empty:
                                    st.warning("No recipe items found for this recipe.")
                                else:
                                    for i, row in enumerate(items.to_dict('records')):
                                        # Stable key base to avoid Streamlit state mismatches when the item order changes
                                        _row_id = None
                                        if ri_id_col:
                                            try:
                                                _row_id = row.get(ri_id_col)
                                            except Exception:
                                                _row_id = None
                                        try:
                                            _row_id = int(_row_id) if _row_id is not None and str(_row_id) != '' else None
                                        except Exception:
                                            _row_id = None
                                        if _row_id is None:
                                            _row_id = i
                                        original_ing = str(row.get(ri_ing_col) or '')
                                        base_qty = float(row.get(ri_qty_col) or 0)
                                        unit = str(row.get(ri_unit_col) or '')
                                        planned_qty = base_qty * float(scale)

                                        # Build options including original even if it's not in the ingredients list
                                        opts = list(all_ing_opts)
                                        if original_ing and original_ing not in opts:
                                            opts = [original_ing] + opts

                                        # If the recipe stores a different casing/spacing than the Ingredients table,
                                        # preselect the best exact normalized match so stock is adjusted by default.
                                        if original_ing and str(original_ing) not in set(all_ing_opts):
                                            try:
                                                norm_orig = _norm_ing_name(original_ing)
                                                ci_matches = [x for x in all_ing_opts if _norm_ing_name(x) == norm_orig]
                                                if len(ci_matches) == 1 and ci_matches[0] in opts:
                                                    # We'll keep original_ing for audit, but default the selectbox to the matched inventory name
                                                    pass
                                            except Exception:
                                                pass
                                        # Default selection: original name if present, else best normalized match (if unique)
                                        _default_name = original_ing
                                        if original_ing and str(original_ing) not in set(all_ing_opts):
                                            try:
                                                norm_orig = _norm_ing_name(original_ing)
                                                ci_matches = [x for x in all_ing_opts if _norm_ing_name(x) == norm_orig]
                                                if len(ci_matches) == 1:
                                                    _default_name = ci_matches[0]
                                            except Exception:
                                                _default_name = original_ing
                                        try:
                                            default_index = opts.index(_default_name) if _default_name else 0
                                        except Exception:
                                            default_index = 0

                                        cc1, cc2, cc3, cc4 = st.columns([1.1, 3, 2, 2])
                                        with cc1:
                                            checked = st.checkbox(
                                                "Use",
                                                value=True,
                                                key=f"brew_ing_use_{batch_id}_{_row_id}",
                                                help="Uncheck to skip this ingredient for this Brew (it won't be consumed or costed).",
                                            )
                                        with cc2:
                                            chosen_ing = st.selectbox(
                                                "Ingredient",
                                                opts if opts else [''],
                                                index=default_index if opts else 0,
                                                format_func=lambda x: ing_label_map.get(str(x), str(x)) if str(x) else "",
                                                key=f"brew_ing_pick_{batch_id}_{_row_id}",
                                                disabled=not checked,
                                            )
                                        with cc3:
                                            actual_qty = st.number_input(
                                                f"Qty ({unit or 'unit'})",
                                                min_value=0.0,
                                                value=float(planned_qty),
                                                step=0.1,
                                                key=f"brew_ing_qty_{batch_id}_{_row_id}",
                                                help=f"Planned: {planned_qty:g} {unit}",
                                                disabled=not checked,
                                            )
                                        with cc4:
                                            if not checked:
                                                st.caption("Skipped on this Brew")
                                            elif original_ing and chosen_ing and chosen_ing != original_ing:
                                                st.caption(f"Planned: {planned_qty:g} {unit}  •  Sub: {original_ing} → {chosen_ing}")
                                            else:
                                                st.caption(f"Planned: {planned_qty:g} {unit}")

                                            # Force substitutions for items that aren't registered in Ingredients
                                            if checked and chosen_ing and (str(chosen_ing) not in set(all_ing_opts)):
                                                st.caption("⚠️ This name isn't registered in **Ingredients**. Pick a substitute from the dropdown so inventory can be adjusted.")

                                        confirmations.append((checked, str(chosen_ing or original_ing), float(actual_qty), unit, original_ing))

                                st.markdown("**Add extra ingredients (optional)**")
                                st.caption("These items are consumed from stock during the brew and included in the batch cost.")
                                extra_lines = []  # (ing, qty)
                                for j in range(5):
                                    ec1, ec2 = st.columns([3, 2])
                                    with ec1:
                                        ex_ing = st.selectbox(
                                            f"Extra ingredient {j+1}",
                                            [''] + (all_ing_opts if all_ing_opts else []),
                                            index=0,
                                            format_func=lambda x: ing_label_map.get(str(x), str(x)) if str(x) else "",
                                            key=f"brew_extra_ing_{batch_id}_{j}",
                                        )
                                    with ec2:
                                        ex_qty = st.number_input(
                                            "Qty",
                                            min_value=0.0,
                                            value=0.0,
                                            step=0.1,
                                            key=f"brew_extra_qty_{batch_id}_{j}",
                                        )
                                    if ex_ing and ex_qty > 0:
                                        extra_lines.append((str(ex_ing), float(ex_qty)))
                                submit = st.form_submit_button('Record Brew', type='primary', use_container_width=True, disabled=brew_event_exists)

                            if submit:
                                require_admin_action()
                                # --- Stock validation (prevents negative stock by default) ---
                                required = {}

                                def _qty_to_stock_unit(qty_val: float, from_unit: str, to_unit: str) -> float:
                                    """Convert qty between common brewery units (g<->kg, ml<->l). Falls back to identity."""
                                    try:
                                        q = float(qty_val)
                                    except Exception:
                                        return 0.0
                                    fu = str(from_unit or '').strip().lower()
                                    tu = str(to_unit or '').strip().lower()
                                    if not fu or not tu or fu == tu:
                                        return q
                                    # grams / kilograms
                                    if fu in {'g', 'gram', 'grams'} and tu in {'kg', 'kilogram', 'kilograms'}:
                                        return q / 1000.0
                                    if fu in {'kg', 'kilogram', 'kilograms'} and tu in {'g', 'gram', 'grams'}:
                                        return q * 1000.0
                                    # milliliters / liters
                                    if fu in {'ml', 'milliliter', 'milliliters'} and tu in {'l', 'lt', 'liter', 'liters'}:
                                        return q / 1000.0
                                    if fu in {'l', 'lt', 'liter', 'liters'} and tu in {'ml', 'milliliter', 'milliliters'}:
                                        return q * 1000.0
                                    return q

                                for checked, ingn, qty, unit, orig_ing in confirmations:
                                    if checked and ingn and float(qty) > 0:
                                        _stock_unit = ''
                                        try:
                                            if ingredients_df is not None and not ingredients_df.empty:
                                                _u0 = _col(ingredients_df, 'unit')
                                                _r0 = _find_ingredient_row_by_name(ingredients_df, str(ingn))
                                                if _r0 is not None and _u0 and _u0 in ingredients_df.columns:
                                                    _stock_unit = str(_r0.get(_u0) or '')
                                        except Exception:
                                            _stock_unit = ''
                                        _need = _qty_to_stock_unit(float(qty), str(unit or ''), _stock_unit)
                                        required[str(ingn)] = float(required.get(str(ingn), 0.0)) + float(_need)
                                for ingn, qty in extra_lines:
                                    if ingn and float(qty) > 0:
                                        required[str(ingn)] = float(required.get(str(ingn), 0.0)) + float(qty)

                                insufficient = []
                                unknown = []
                                try:
                                    if required and ingredients_df is not None and not ingredients_df.empty:
                                        _n = _col(ingredients_df, 'name')
                                        _s = _col(ingredients_df, 'stock', 'quantity_in_stock')
                                        _u = _col(ingredients_df, 'unit')
                                        if _n and _s:
                                            for ingn, need in required.items():
                                                m_ing = ingredients_df[ingredients_df[_n].astype(str) == str(ingn)]
                                                if m_ing.empty:
                                                    unknown.append((ingn, need))
                                                    continue
                                                row = m_ing.iloc[0]
                                                have = float(row[_s] or 0)
                                                unit0 = str(row[_u]) if _u and _u in ingredients_df.columns and row[_u] is not None else ''
                                                if have + 1e-9 < float(need):
                                                    insufficient.append((ingn, have, need, unit0))
                                except Exception:
                                    pass

                                if insufficient:
                                    st.error("Insufficient inventory for this Brew (nothing was recorded):")
                                    for ingn, have, need, unit0 in insufficient:
                                        st.write(f"- {ingn}: have {have:g} {unit0} / need {need:g} {unit0}")
                                    st.stop()

                                if unknown:
                                    st.error("Some ingredients are not registered in **Ingredients**, so I can't adjust inventory. Please choose substitutes from the dropdown before recording Brew:")
                                    for ingn, need in unknown:
                                        st.write(f"- {ingn}: need {need:g}")
                                    st.stop()
                                if not fermenter:
                                    st.error('Please select a fermenter.')
                                elif not _vessel_is_free(fermenter, batch_id):
                                    st.error('This fermenter is already occupied by another active batch.')
                                elif not _vessel_capacity_ok(fermenter, float(actual_volume)):
                                    st.error('Selected fermenter capacity is lower than the batch volume.')
                                else:
                                    # Create event
                                    ev_id = insert_data('production_events', {
                                        'batch_id': batch_id,
                                        'event_type': 'Brew',
                                        'event_date': pd.Timestamp(brew_date),
                                        'from_vessel': '',
                                        'to_vessel': fermenter,
                                        'notes': notes,
                                        'created_by': st.session_state.get('auth_user', 'admin'),
                                        'meta': json.dumps({'action': 'Brew', 'og': float(og_brew) if og_brew is not None else None, 'efficiency': float(efficiency_brew) if efficiency_brew is not None else None, 'volume_l': float(actual_volume)}, ensure_ascii=False),
                                    })

                                    # Consume ingredients
                                    total_cost = 0.0
                                    for checked, ingn, qty, unit, orig_ing in confirmations:
                                        if not checked or not ingn or qty <= 0:
                                            continue
                                        unit0, unit_cost = get_ingredient_unit_and_cost(get_all_data(), ingn)
                                        qty_stock = _qty_to_stock_unit(float(qty), str(unit or ''), str(unit0 or unit or ''))
                                        line_cost = float(qty_stock) * float(unit_cost or 0)
                                        total_cost += line_cost
                                        adjust_stock_for_ingredient(get_all_data(), ingn, -float(qty_stock), reason='Brew', destination=f"Batch #{batch_id} (Brew)", ref_table='production_events', ref_id=int(ev_id) if ev_id is not None else None, batch_id=int(batch_id), prod_event_id=int(ev_id) if ev_id is not None else None)
                                        meta = {'event': 'Brew', 'source': 'recipe'}
                                        _ing_id_val = ''
                                        try:
                                            _idc0 = _col(ingredients_df, 'id_ingredient', 'ingredient_id', 'id')
                                            if _idc0:
                                                _r0 = _find_ingredient_row_by_name(ingredients_df, ingn)
                                                if _r0 is not None:
                                                    _ing_id_val = str(_r0.get(_idc0) or '')
                                        except Exception:
                                            _ing_id_val = ''
                                        if orig_ing and str(orig_ing) != str(ingn):
                                            meta = {'event': 'Brew', 'source': 'substitute', 'original_ingredient': orig_ing}
                                        insert_data('production_consumptions', {
                                            'batch_id': batch_id,
                                            'prod_event_id': int(ev_id) if ev_id is not None else None,
                                            'ingredient_id': str(_ing_id_val or ''),
                                            'ingredient_name': ingn,
                                            'quantity': float(qty),
                                            'unit': str(unit0 or unit or ''),
                                            'unit_cost': float(unit_cost or 0),
                                            'total_cost': float(line_cost),
                                            'meta': json.dumps(meta, ensure_ascii=False),
                                        })

                                    # Extra ingredients
                                    for ingn, qty in extra_lines:
                                        if not ingn or qty <= 0:
                                            continue
                                        unit0, unit_cost = get_ingredient_unit_and_cost(get_all_data(), ingn)
                                        line_cost = float(qty) * float(unit_cost or 0)
                                        total_cost += line_cost
                                        adjust_stock_for_ingredient(get_all_data(), ingn, -float(qty), reason='Brew', destination=f"Batch #{batch_id} (Brew)", ref_table='production_events', ref_id=int(ev_id) if ev_id is not None else None, batch_id=int(batch_id), prod_event_id=int(ev_id) if ev_id is not None else None)
                                        _ing_id_val = ''
                                        try:
                                            _idc0 = _col(ingredients_df, 'id_ingredient', 'ingredient_id', 'id')
                                            if _idc0:
                                                _r0 = _find_ingredient_row_by_name(ingredients_df, ingn)
                                                if _r0 is not None:
                                                    _ing_id_val = str(_r0.get(_idc0) or '')
                                        except Exception:
                                            _ing_id_val = ''
                                        insert_data('production_consumptions', {
                                            'batch_id': batch_id,
                                            'prod_event_id': int(ev_id) if ev_id is not None else None,
                                            'ingredient_id': str(_ing_id_val or ''),
                                            'ingredient_name': ingn,
                                            'quantity': float(qty),
                                            'unit': unit0 or '',
                                            'unit_cost': float(unit_cost or 0),
                                            'total_cost': float(line_cost),
                                            'meta': json.dumps({'event': 'Brew', 'source': 'extra'}, ensure_ascii=False),
                                        })
                                    update_data('production_batches', {
                                        'status': 'In Progress',
                                        'stage': 'Fermenting',
                                        'current_vessel': fermenter,
                                        'volume_remaining_l': float(actual_volume),
                                        'brewed_volume_l': float(actual_volume),
                                        'og': float(og_brew) if og_brew is not None else None,
                                        'efficiency': float(efficiency_brew) if efficiency_brew is not None else None,
                                    },  f"{b_id_col} = :id", {'id': batch_id})

                                    st.success(f"✅ Brew recorded. Materials cost logged: {total_cost:.2f}")
                                    st.rerun()


                                                # ---- READING (Fermentation) ----
                        if action == 'Reading':
                            st.caption("Log fermentation readings (add as many as you want over time).")

                            # Existing readings for this batch
                            def _safe_json_load(_s):
                                try:
                                    if _s is None:
                                        return {}
                                    if isinstance(_s, dict):
                                        return _s
                                    _ss = str(_s)
                                    if not _ss or _ss.strip().lower() in {"nan", "none"}:
                                        return {}
                                    return json.loads(_ss)
                                except Exception:
                                    return {}

                            try:
                                _etype = _col(events_df, 'event_type', 'type')
                                _edate = _col(events_df, 'event_date', 'date')
                                _emeta = _col(events_df, 'meta')
                                _ebid = _col(events_df, 'batch_id')
                                _enotes = _col(events_df, 'notes')
                                existing_r = pd.DataFrame()
                                if events_df is not None and not events_df.empty and _etype and _ebid:
                                    existing_r = events_df[events_df[_ebid] == batch_id].copy()
                                    existing_r = existing_r[existing_r[_etype].astype(str).str.lower() == 'reading']
                                    if not existing_r.empty and _edate:
                                        existing_r[_edate] = pd.to_datetime(existing_r[_edate], errors='coerce')
                                        existing_r = existing_r.sort_values(_edate)
                                    if not existing_r.empty and _emeta and _emeta in existing_r.columns:
                                        metas = existing_r[_emeta].apply(_safe_json_load)
                                        existing_r['Gravity (°P)'] = metas.apply(lambda d: d.get('gravity_plato') if isinstance(d, dict) else None)
                                        existing_r['Temp (°C)'] = metas.apply(lambda d: d.get('temp_c') if isinstance(d, dict) else None)
                                        existing_r['Pressure (bar)'] = metas.apply(lambda d: d.get('pressure_bar') if isinstance(d, dict) else None)
                                        existing_r['pH'] = metas.apply(lambda d: d.get('ph') if isinstance(d, dict) else None)
                                    if _enotes and _enotes in existing_r.columns:
                                        existing_r['Notes'] = existing_r[_enotes]
                                    else:
                                        existing_r['Notes'] = None
                            except Exception:
                                existing_r = pd.DataFrame()

                            if existing_r is not None and not existing_r.empty:
                                with st.expander("Existing readings", expanded=False):
                                    show_cols = []
                                    if '_edate' in locals() and _edate and _edate in existing_r.columns:
                                        show_cols.append(_edate)
                                    for _c in ('Gravity (°P)', 'Temp (°C)', 'Pressure (bar)', 'pH', 'Notes'):
                                        if _c in existing_r.columns:
                                            show_cols.append(_c)
                                    st.dataframe(existing_r[show_cols], use_container_width=True)

                                    # Delete a specific existing reading (with confirmation)
                                    try:
                                        _eid = _col(existing_r, 'id_event', 'event_id', 'id')
                                    except Exception:
                                        _eid = None
                                    if _eid and _eid in existing_r.columns:
                                        # Build labels for selection
                                        _labels = {}
                                        for _, _rr in existing_r.iterrows():
                                            rid = _rr.get(_eid)
                                            # Timestamp label
                                            dlabel = ""
                                            try:
                                                if _edate and _edate in existing_r.columns:
                                                    dlabel = pd.to_datetime(_rr.get(_edate), errors='coerce').strftime('%Y-%m-%d %H:%M')
                                            except Exception:
                                                dlabel = str(_rr.get(_edate, ''))[:16]
                                            g = _rr.get('Gravity (°P)', None)
                                            t = _rr.get('Temp (°C)', None)
                                            p = _rr.get('Pressure (bar)', None)
                                            parts = [dlabel]
                                            if g is not None and not pd.isna(g):
                                                parts.append(f"{g}°P")
                                            if t is not None and not pd.isna(t):
                                                parts.append(f"{t}°C")
                                            if p is not None and not pd.isna(p):
                                                parts.append(f"{p} bar")
                                            _labels[rid] = " — ".join([parts[0], ", ".join(parts[1:])]) if len(parts) > 1 else parts[0]

                                        st.markdown("**Delete a reading**")
                                        sel_id = st.selectbox(
                                            "Select an existing reading",
                                            options=list(_labels.keys()),
                                            format_func=lambda x: _labels.get(x, str(x)),
                                            key=f"del_reading_select_{batch_id}",
                                        )
                                        _confirm_key = f"del_reading_confirm_{batch_id}"
                                        if _confirm_key not in st.session_state:
                                            st.session_state[_confirm_key] = False

                                        if st.button("Delete selected reading", type="secondary", use_container_width=False, key=f"del_reading_btn_{batch_id}"):
                                            st.session_state[_confirm_key] = True

                                        if st.session_state.get(_confirm_key) and sel_id is not None:
                                            st.warning("This will permanently delete the selected reading. Continue?")
                                            cc1, cc2 = st.columns(2)
                                            with cc1:
                                                if st.button("Yes, delete", type="primary", use_container_width=True, key=f"del_reading_yes_{batch_id}"):
                                                    require_admin_action()
                                                    try:
                                                        delete_data('production_events', f"{_eid} = :id", {'id': sel_id})
                                                        st.success("✅ Reading deleted.")
                                                    except Exception as _ex:
                                                        st.error(f"Failed to delete reading: {_ex}")
                                                    st.session_state[_confirm_key] = False
                                                    st.rerun()
                                            with cc2:
                                                if st.button("Cancel", use_container_width=True, key=f"del_reading_cancel_{batch_id}"):
                                                    st.session_state[_confirm_key] = False
                                                    st.rerun()

                            st.markdown("**Add new readings**")
                            _editor_key = f"reading_editor_{batch_id}"
                            if _editor_key not in st.session_state:
                                st.session_state[_editor_key] = pd.DataFrame([{
                                    'Date': date.today(),
                                    'Time': datetime.now().time().replace(second=0, microsecond=0),
                                    'Gravity (°P)': None,
                                    'Temp (°C)': None,
                                    'Pressure (bar)': None,
                                    'pH': None,
                                    'Notes': '',
                                }])

                            new_readings_df = st.data_editor(
                                st.session_state[_editor_key],
                                num_rows="dynamic",
                                use_container_width=True,
                                key=f"reading_editor_widget_{batch_id}",
                            )

                            c1, c2 = st.columns([1, 3])
                            with c1:
                                save_readings = st.button("Save readings", type="primary", use_container_width=True)
                            with c2:
                                if st.button("Clear new rows (does not delete existing)", use_container_width=True):
                                    st.session_state[_editor_key] = pd.DataFrame([{
                                        'Date': date.today(),
                                        'Time': datetime.now().time().replace(second=0, microsecond=0),
                                        'Gravity (°P)': None,
                                        'Temp (°C)': None,
                                        'Pressure (bar)': None,
                                        'pH': None,
                                        'Notes': '',
                                    }])
                                    st.rerun()

                            if save_readings:
                                require_admin_action()
                                if new_readings_df is None or new_readings_df.empty:
                                    st.error("Please add at least one reading row.")
                                else:
                                    # Build a set of existing timestamps to avoid accidental duplicates
                                    existing_ts = set()
                                    try:
                                        if existing_r is not None and not existing_r.empty and '_edate' in locals() and _edate and _edate in existing_r.columns:
                                            for _v in pd.to_datetime(existing_r[_edate], errors='coerce').dropna().tolist():
                                                existing_ts.add(pd.Timestamp(_v).to_pydatetime().replace(tzinfo=None))
                                    except Exception:
                                        existing_ts = set()

                                    inserted = 0
                                    skipped = 0
                                    for _, row in new_readings_df.iterrows():
                                        # Skip fully empty rows
                                        vals = {
                                            'Gravity (°P)': row.get('Gravity (°P)', None),
                                            'Temp (°C)': row.get('Temp (°C)', None),
                                            'Pressure (bar)': row.get('Pressure (bar)', None),
                                            'pH': row.get('pH', None),
                                        }
                                        notes = row.get('Notes', '')
                                        is_all_blank = True
                                        for _vv in vals.values():
                                            try:
                                                if _vv is not None and not (isinstance(_vv, float) and pd.isna(_vv)) and str(_vv) != '':
                                                    is_all_blank = False
                                            except Exception:
                                                pass
                                        if isinstance(notes, str) and notes.strip():
                                            is_all_blank = False
                                        if is_all_blank:
                                            continue

                                        # Parse date/time
                                        d0 = row.get('Date', date.today())
                                        t0 = row.get('Time', datetime.now().time().replace(second=0, microsecond=0))
                                        try:
                                            if isinstance(d0, str):
                                                d0 = pd.to_datetime(d0, errors='coerce').date()
                                        except Exception:
                                            d0 = date.today()
                                        try:
                                            if isinstance(t0, str):
                                                _tt = pd.to_datetime(t0, errors='coerce')
                                                if pd.isna(_tt):
                                                    t0 = datetime.now().time().replace(second=0, microsecond=0)
                                                else:
                                                    t0 = _tt.time()
                                        except Exception:
                                            t0 = datetime.now().time().replace(second=0, microsecond=0)

                                        try:
                                            dt_val = datetime.combine(d0, t0)
                                        except Exception:
                                            dt_val = datetime.now()

                                        dt_plain = dt_val.replace(tzinfo=None)
                                        if dt_plain in existing_ts:
                                            skipped += 1
                                            continue

                                        meta = {
                                            'action': 'Reading',
                                            'gravity_plato': (float(row.get('Gravity (°P)')) if row.get('Gravity (°P)') is not None and str(row.get('Gravity (°P)')) != '' and not pd.isna(row.get('Gravity (°P)')) else None),
                                            'temp_c': (float(row.get('Temp (°C)')) if row.get('Temp (°C)') is not None and str(row.get('Temp (°C)')) != '' and not pd.isna(row.get('Temp (°C)')) else None),
                                            'pressure_bar': (float(row.get('Pressure (bar)')) if row.get('Pressure (bar)') is not None and str(row.get('Pressure (bar)')) != '' and not pd.isna(row.get('Pressure (bar)')) else None),
                                            'ph': (float(row.get('pH')) if row.get('pH') is not None and str(row.get('pH')) != '' and not pd.isna(row.get('pH')) else None),
                                        }

                                        insert_data('production_events', {
                                            'batch_id': batch_id,
                                            'event_type': 'Reading',
                                            'event_date': pd.Timestamp(dt_val),
                                            'from_vessel': current_vessel,
                                            'to_vessel': current_vessel,
                                            'notes': (notes if isinstance(notes, str) else ''),
                                            'created_by': st.session_state.get('auth_user', 'admin'),
                                            'meta': json.dumps(meta, ensure_ascii=False),
                                        })

                                        inserted += 1
                                        existing_ts.add(dt_plain)

                                    if inserted == 0 and skipped == 0:
                                        st.error("No valid reading rows to save.")
                                    else:
                                        msg = f"✅ Saved {inserted} reading(s)."
                                        if skipped:
                                            msg += f" Skipped {skipped} duplicate timestamp(s)."
                                        st.success(msg)

                                        # Reset the editor to one empty row
                                        st.session_state[_editor_key] = pd.DataFrame([{
                                            'Date': date.today(),
                                            'Time': datetime.now().time().replace(second=0, microsecond=0),
                                            'Gravity (°P)': None,
                                            'Temp (°C)': None,
                                            'Pressure (bar)': None,
                                            'pH': None,
                                            'Notes': '',
                                        }])
                                        st.rerun()
# ---- DRY HOP / ADJUNCT ----
                        if action in {'Dry Hop', 'Add Adjunct'}:
                            is_dryhop = (action == 'Dry Hop')
                            st.caption("Consumes ingredients from stock and logs an event.")

                            ing_name_col = _col(ingredients_df, 'name')
                            ing_cat_col = _col(ingredients_df, 'category')
                            options = []
                            if ingredients_df is not None and not ingredients_df.empty and ing_name_col:
                                if is_dryhop and ing_cat_col:
                                    m = ingredients_df[ingredients_df[ing_cat_col].astype(str).str.lower().str.contains('hop', na=False)]
                                    options = m[ing_name_col].astype(str).tolist() if not m.empty else ingredients_df[ing_name_col].astype(str).tolist()
                                else:
                                    options = ingredients_df[ing_name_col].astype(str).tolist()

                            # UI-only labels (Manufacturer — Name)
                            ing_label_map = _ingredient_name_label_map(ingredients_df)

                            with st.form(f'addition_action_{batch_id}_{action}', clear_on_submit=True):
                                d = st.date_input('Date', date.today())
                                notes = st.text_area('Notes', height=80)
                                st.markdown('**Additions**')
                                lines = []
                                for i in range(5):
                                    cc1, cc2 = st.columns([3, 1])
                                    with cc1:
                                        ing = st.selectbox(
                                            f"Ingredient {i+1}",
                                            [''] + options,
                                            index=0,
                                            format_func=lambda x: ing_label_map.get(str(x), str(x)) if str(x) else "",
                                            key=f"add_{batch_id}_{action}_{i}",
                                        )
                                    with cc2:
                                        qty = st.number_input("Qty", min_value=0.0, value=0.0, step=0.1, key=f"add_qty_{batch_id}_{action}_{i}")
                                    if ing and qty > 0:
                                        unit, _ = get_ingredient_unit_and_cost(get_all_data(), ing)
                                        lines.append((ing, float(qty), unit or ''))

                                submit = st.form_submit_button(f"Record {action}", type='primary', use_container_width=True)

                            if submit:
                                require_admin_action()
                                if not lines:
                                    st.error('Please add at least one ingredient line.')
                                else:
                                    ev_id = insert_data('production_events', {
                                        'batch_id': batch_id,
                                        'event_type': action,
                                        'event_date': pd.Timestamp(d),
                                        'from_vessel': current_vessel,
                                        'to_vessel': current_vessel,
                                        'notes': notes,
                                        'created_by': st.session_state.get('auth_user', 'admin'),
                                        'meta': json.dumps({'action': action}, ensure_ascii=False),
                                    })
                                    total_cost = 0.0
                                    for ing, qty, unit in lines:
                                        unit0, unit_cost = get_ingredient_unit_and_cost(get_all_data(), ing)
                                        line_cost = float(qty) * float(unit_cost or 0)
                                        total_cost += line_cost
                                        adjust_stock_for_ingredient(get_all_data(), ing, -float(qty), reason=str(action), destination=f"Batch #{batch_id} ({action})", ref_table='production_events', ref_id=int(ev_id) if ev_id is not None else None, batch_id=int(batch_id), prod_event_id=int(ev_id) if ev_id is not None else None)
                                        insert_data('production_consumptions', {
                                            'batch_id': batch_id,
                                            'prod_event_id': int(ev_id) if ev_id is not None else None,
                                            'ingredient_id': '',
                                            'ingredient_name': ing,
                                            'quantity': float(qty),
                                            'unit': unit0 or unit,
                                            'unit_cost': float(unit_cost or 0),
                                            'total_cost': float(line_cost),
                                            'meta': json.dumps({'event': action}, ensure_ascii=False),
                                        })

                                    st.success(f"✅ {action} recorded. Cost logged: {total_cost:.2f}")
                                    st.rerun()

                        # ---- CONDITIONING ----
                        if action == 'Conditioning':
                            with st.form(f'conditioning_{batch_id}', clear_on_submit=True):
                                d = st.date_input('Date', date.today())
                                target_temp = st.number_input('Target temperature (°C)', value=2.0, step=0.5)
                                notes = st.text_area('Notes', height=80)
                                submit = st.form_submit_button('Record Conditioning', type='primary', use_container_width=True)

                            if submit:
                                require_admin_action()
                                insert_data('production_events', {
                                    'batch_id': batch_id,
                                    'event_type': 'Conditioning',
                                    'event_date': pd.Timestamp(d),
                                    'from_vessel': current_vessel,
                                    'to_vessel': current_vessel,
                                    'notes': notes,
                                    'created_by': st.session_state.get('auth_user', 'admin'),
                                    'meta': json.dumps({'target_temp_c': float(target_temp)}, ensure_ascii=False),
                                })
                                update_data('production_batches', {
                                    'stage': 'Conditioning',
                                }, f"{b_id_col} = :id", {'id': batch_id})
                                st.success('✅ Conditioning recorded.')
                                st.rerun()

                        # ---- TRANSFER ----
                        if action == 'Transfer':
                            eq_names = equipment_df[eq_name_col].astype(str).tolist() if equipment_df is not None and not equipment_df.empty and eq_name_col else []
                            with st.form(f'transfer_{batch_id}', clear_on_submit=True):
                                d = st.date_input('Date', date.today())
                                c1, c2 = st.columns(2)
                                with c1:
                                    from_v = st.text_input('From vessel', value=current_vessel)
                                with c2:
                                    to_v = st.selectbox('To vessel', [''] + eq_names, index=0)
                                notes = st.text_area('Notes', height=80)
                                submit = st.form_submit_button('Record Transfer', type='primary', use_container_width=True)

                            if submit:
                                require_admin_action()
                                if not to_v:
                                    st.error('Please select a destination vessel.')
                                elif not _vessel_is_free(to_v, batch_id):
                                    st.error('Destination vessel is occupied by another active batch.')
                                elif not _vessel_capacity_ok(to_v, float(remaining_vol)):
                                    st.error('Destination vessel capacity is lower than the remaining volume.')
                                else:
                                    insert_data('production_events', {
                                        'batch_id': batch_id,
                                        'event_type': 'Transfer',
                                        'event_date': pd.Timestamp(d),
                                        'from_vessel': from_v,
                                        'to_vessel': to_v,
                                        'notes': notes,
                                        'created_by': st.session_state.get('auth_user', 'admin'),
                                        'meta': json.dumps({'action': 'Transfer'}, ensure_ascii=False),
                                    })
                                    update_data('production_batches', {
                                        'current_vessel': to_v,
                                    }, f"{b_id_col} = :id", {'id': batch_id})
                                    st.success('✅ Transfer recorded.')
                                    st.rerun()

                        # ---- KEGGING (via Composite Product) ----
                        if action == 'Kegging':
                            # Guard: you cannot keg if the batch has not been brewed (no wort/beer yet).
                            try:
                                _brew_cnt = query_to_df(
                                    "SELECT COUNT(*) AS cnt FROM production_events WHERE batch_id = :bid AND LOWER(event_type) = 'brew'",
                                    {"bid": batch_id},
                                )
                                brewed_ok = bool(_brew_cnt is not None and not _brew_cnt.empty and int(_brew_cnt.iloc[0].get('cnt', 0)) > 0)
                            except Exception:
                                brewed_ok = False

                            try:
                                remaining_ok = float(remaining_vol or 0) > 1e-9
                            except Exception:
                                remaining_ok = False

                            if not brewed_ok or not remaining_ok:
                                st.warning(
                                    "Kegging is only available after the batch has been **brewed** and has beer volume remaining. "
                                    "Record a **Brew** event first."
                                )
                            elif composites_df is None or composites_df.empty:
                                st.warning('No composite products found. Create one in Products → Composite Products.')
                            else:
                                cid_col = _col(composites_df, 'id_composite', 'id')
                                cname_col = _col(composites_df, 'name')
                                c_recipe_name_col = _col(composites_df, 'recipe_name')
                                c_out_unit_col = _col(composites_df, 'output_unit')

                                comp_records = composites_df.to_dict('records')

                                def _comp_label(r):
                                    return f"{r.get(cname_col,'')} (#{r.get(cid_col,'')})"

                                with st.form(f'kegging_{batch_id}', clear_on_submit=True):
                                    d = st.date_input('Date', date.today())
                                    comp = st.selectbox('Composite product', comp_records, format_func=_comp_label)
                                    units_out = st.number_input('Units produced', min_value=1.0, value=10.0, step=1.0)
                                    warehouse = st.text_input('Warehouse', value='Main')
                                    notes = st.text_area('Notes', height=80)
                                    submit = st.form_submit_button('Record Kegging', type='primary', use_container_width=True)

                                if submit:
                                    require_admin_action()
                                    comp_id = int(comp.get(cid_col))
                                    comp_name = str(comp.get(cname_col) or '')
                                    comp_recipe_name = str(comp.get(c_recipe_name_col) or '')

                                    # Safety: ensure same beer
                                    if comp_recipe_name and str(selected_batch.get(b_recipe_name_col) or '') and comp_recipe_name != str(selected_batch.get(b_recipe_name_col) or ''):
                                        st.error('This composite product is for a different beer/recipe than the selected batch.')
                                    else:
                                        # Load BOM
                                        icid = _col(composite_items_df, 'composite_id')
                                        ctype = _col(composite_items_df, 'component_type')
                                        cname = _col(composite_items_df, 'component_name')
                                        cqty = _col(composite_items_df, 'quantity')
                                        cunit = _col(composite_items_df, 'unit')

                                        bom = composite_items_df[composite_items_df[icid] == comp_id] if composite_items_df is not None and not composite_items_df.empty and icid else pd.DataFrame()
                                        if bom.empty:
                                            st.error('Composite product BOM is empty.')
                                        else:
                                            # Beer liters per unit
                                            beer_row = bom[bom[ctype].astype(str).str.lower() == 'beer'] if ctype else pd.DataFrame()
                                            if beer_row.empty:
                                                st.error('Composite product must include a Beer component.')
                                            else:
                                                try:
                                                    beer_l_per_unit = float(beer_row.iloc[0][cqty] or 0)
                                                except Exception:
                                                    beer_l_per_unit = 0

                                                liters_needed = float(units_out) * float(beer_l_per_unit)
                                                if liters_needed > float(remaining_vol) + 1e-9:
                                                    st.error('Not enough beer volume remaining in the batch for this kegging run.')
                                                else:
                                                    # Validate packaging stock
                                                    pkg_rows = bom[bom[ctype].astype(str).str.lower() == 'ingredient'] if ctype else pd.DataFrame()
                                                    missing = []
                                                    for _, r in pkg_rows.iterrows():
                                                        ingn = str(r.get(cname) or '')
                                                        per_unit = float(r.get(cqty) or 0)
                                                        need = per_unit * float(units_out)
                                                        # find current stock
                                                        ing_df2 = get_all_data().get('ingredients', pd.DataFrame())
                                                        ncol = _col(ing_df2, 'name')
                                                        scol = _col(ing_df2, 'stock', 'quantity_in_stock')
                                                        if not (ncol and scol) or ing_df2.empty:
                                                            continue
                                                        m = ing_df2[ing_df2[ncol].astype(str) == ingn]
                                                        if m.empty:
                                                            continue
                                                        try:
                                                            stock = float(m.iloc[0][scol] or 0)
                                                        except Exception:
                                                            stock = 0
                                                        if stock + 1e-9 < need:
                                                            missing.append(f"{ingn} (need {need:g})")

                                                    if missing:
                                                        st.error('Not enough packaging stock: ' + ', '.join(missing))
                                                    else:
                                                        # Log event
                                                        ev_id = insert_data('production_events', {
                                                            'batch_id': batch_id,
                                                            'event_type': 'Kegging',
                                                            'event_date': pd.Timestamp(d),
                                                            'from_vessel': current_vessel,
                                                            'to_vessel': current_vessel,
                                                            'notes': notes,
                                                            'created_by': st.session_state.get('auth_user', 'admin'),
                                                            'meta': json.dumps({'composite_id': comp_id, 'composite_name': comp_name, 'units': float(units_out)}, ensure_ascii=False),
                                                        })

                                                        # Consume packaging + log
                                                        total_pack_cost = 0.0
                                                        for _, r in pkg_rows.iterrows():
                                                            ingn = str(r.get(cname) or '')
                                                            per_unit = float(r.get(cqty) or 0)
                                                            need = per_unit * float(units_out)
                                                            unit = str(r.get(cunit) or '')
                                                            unit0, unit_cost = get_ingredient_unit_and_cost(get_all_data(), ingn)
                                                            line_cost = float(need) * float(unit_cost or 0)
                                                            total_pack_cost += line_cost
                                                            adjust_stock_for_ingredient(get_all_data(), ingn, -float(need), reason='Kegging', destination=f"Batch #{batch_id} (Kegging)", ref_table='production_events', ref_id=int(ev_id) if ev_id is not None else None, batch_id=int(batch_id), prod_event_id=int(ev_id) if ev_id is not None else None)
                                                            insert_data('production_consumptions', {
                                                                'batch_id': batch_id,
                                                                'prod_event_id': int(ev_id) if ev_id is not None else None,
                                                                'ingredient_id': '',
                                                                'ingredient_name': ingn,
                                                                'quantity': float(need),
                                                                'unit': unit0 or unit,
                                                                'unit_cost': float(unit_cost or 0),
                                                                'total_cost': float(line_cost),
                                                                'meta': json.dumps({'event': 'Kegging', 'composite_id': comp_id}, ensure_ascii=False),
                                                            })

                                                        # Add finished goods inventory
                                                        insert_data('composite_inventory', {
                                                            'composite_id': comp_id,
                                                            'composite_name': comp_name,
                                                            'warehouse': warehouse,
                                                            'quantity_units': float(units_out),
                                                        })

                                                        # Keg run record
                                                        insert_data('production_keg_runs', {
                                                            'batch_id': batch_id,
                                                            'product_name': str(selected_batch.get(b_recipe_name_col) or ''),
                                                            'beer_volume_l': float(liters_needed),
                                                            'warehouse': warehouse,
                                                            'actual_cost': float(sum_production_costs(get_all_data(), batch_id) + total_pack_cost),
                                                            'notes': notes,
                                                            'composite_id': comp_id,
                                                            'composite_name': comp_name,
                                                            'units_produced': float(units_out),
                                                            'output_unit': str(comp.get(c_out_unit_col) or 'unit'),
                                                        })

                                                        # Update batch volume
                                                        new_remaining = float(remaining_vol) - float(liters_needed)
                                                        upd = {
                                                            'volume_remaining_l': new_remaining,
                                                            'stage': 'Kegging',
                                                            'status': 'In Progress',
                                                        }
                                                        if new_remaining <= 1e-9:
                                                            upd['stage'] = 'Ready to Finish'
                                                        update_data('production_batches', upd, f"{b_id_col} = :id", {'id': batch_id})

                                                        st.success('✅ Kegging recorded and finished goods added to inventory.')
                                                        st.rerun()

                        # ---- FINISH BATCH ----
                        if action == 'Finish Batch':
                            remaining = float(remaining_vol)
                            loss_hint = f"{remaining:g} L" if remaining > 0 else "0 L"
                            st.warning(f"Remaining beer in tank: {loss_hint}")
                            with st.form(f'finish_{batch_id}', clear_on_submit=True):
                                d = st.date_input('Finish date', date.today())
                                mark_loss = st.checkbox(f"Finish batch and mark remaining volume as loss ({loss_hint})", value=True)
                                notes = st.text_area('Notes', height=80)
                                submit = st.form_submit_button('Finish batch', type='primary', use_container_width=True)

                            if submit:
                                require_admin_action()
                                loss_l = remaining if mark_loss else 0.0
                                insert_data('production_events', {
                                    'batch_id': batch_id,
                                    'event_type': 'Finish Batch',
                                    'event_date': pd.Timestamp(d),
                                    'from_vessel': current_vessel,
                                    'to_vessel': current_vessel,
                                    'notes': notes,
                                    'created_by': st.session_state.get('auth_user', 'admin'),
                                    'meta': json.dumps({'loss_l': float(loss_l)}, ensure_ascii=False),
                                })
                                # accumulate loss
                                prior_loss = float(selected_batch.get(b_loss_col) or 0) if b_loss_col else 0.0
                                update_data('production_batches', {
                                    'loss_l': float(prior_loss) + float(loss_l),
                                    'volume_remaining_l': 0.0,
                                    'status': 'Completed',
                                    'stage': 'Completed',
                                    'finished_date': d,
                                }, f"{b_id_col} = :id", {'id': batch_id})
                                st.success('✅ Batch completed.')
                                st.rerun()

    with tab_reports:
        st.subheader("Production reports")

        report_mode = st.radio(
            "Report view",
            ["Finished Orders", "Batch report"],
            horizontal=True,
            key="prod_reports_mode",
        )

        def _is_finished_status(s: str | None) -> bool:
            s0 = (s or "").strip().lower()
            return s0 in {"completed", "done", "finished", "cancelled", "canceled"}

        if report_mode == "Finished Orders":
            st.subheader("Finished / cancelled orders")

            if batches_df is None or batches_df.empty:
                st.info("No batches yet.")
            else:
                viewf = batches_df.copy()

                # Filter finished/cancelled when we have a status column
                if b_status_col and b_status_col in viewf.columns:
                    try:
                        viewf = viewf[viewf[b_status_col].astype(str).apply(_is_finished_status)]
                    except Exception:
                        pass
                else:
                    st.caption("Note: couldn't find a status column in production_batches, so showing all batches.")

                if viewf.empty:
                    st.info("No finished/cancelled orders yet.")
                else:
                    # Sort (finished_date/cancelled_at/planned_date) when available
                    sort_cols = []
                    for c in ["finished_date", "cancelled_at", b_planned_date_col]:
                        if c and c in viewf.columns:
                            sort_cols.append(c)
                    for c in sort_cols:
                        try:
                            viewf[c] = pd.to_datetime(viewf[c], errors="coerce")
                        except Exception:
                            pass
                    if sort_cols:
                        try:
                            viewf = viewf.sort_values(sort_cols[0], ascending=False)
                        except Exception:
                            pass

                    cols_show = []
                    for c in [b_id_col, b_code_col, "recipe_name", b_planned_date_col, "finished_date", "cancelled_at", b_status_col, "cancelled_reason"]:
                        if c and c in viewf.columns and c not in cols_show:
                            cols_show.append(c)
                    if not cols_show:
                        cols_show = list(viewf.columns)[:12]

                    st.dataframe(viewf[cols_show], use_container_width=True)

                    st.markdown("---")
                    records_f = viewf.to_dict("records")

                    # Only show selector + PDF if we can resolve an ID
                    if not b_id_col:
                        st.warning("Couldn't resolve the batch ID column, so PDF generation is disabled here.")
                    else:
                        sel_f = st.selectbox(
                            "Select finished/cancelled order",
                            records_f,
                            format_func=lambda r: f"#{r.get(b_id_col)} {r.get(b_code_col) if b_code_col else r.get('batch_code','')} {r.get('recipe_name','')}",
                            key="prod_finished_select",
                        )
                        if sel_f:
                            try:
                                fid = int(sel_f.get(b_id_col))
                            except Exception:
                                fid = None

                            if fid is None:
                                st.warning("Couldn't read this batch ID.")
                            else:
                                fstatus = str(sel_f.get(b_status_col) or "")
                                if fstatus:
                                    st.caption(f"Status: {fstatus}")

                                total_cost = sum_production_costs(get_all_data(), fid)
                                st.metric("Total logged materials cost", f"{total_cost:.2f}")

                                _pdf_key = f"prod_report_pdf_finished_{fid}"
                                if st.button("Generate PDF report", key=f"gen_pdf_finished_{fid}", use_container_width=True):
                                    st.session_state[_pdf_key] = generate_production_report_pdf_bytes(fid)
                                if _pdf_key in st.session_state:
                                    st.download_button(
                                        "⬇️ Download PDF",
                                        data=st.session_state[_pdf_key],
                                        file_name=f"production_report_batch_{fid}.pdf",
                                        mime="application/pdf",
                                        use_container_width=True,
                                        key=f"dl_pdf_finished_{fid}",
                                    )

        else:  # Batch report
            st.subheader("Batch report")

            if batches_df is None or batches_df.empty:
                st.info("No batches yet.")
            else:
                records = batches_df.to_dict("records")

                if not b_id_col:
                    st.warning("Couldn't resolve the batch ID column, so PDF generation is disabled here.")
                else:
                    selected = st.selectbox(
                        "Select batch",
                        records,
                        format_func=lambda r: f"#{r.get(b_id_col)} {r.get(b_code_col) if b_code_col else r.get('batch_code','')} {r.get('recipe_name','')}",
                        key="prod_batch_report_select",
                    )

                    try:
                        batch_id = int(selected.get(b_id_col))
                    except Exception:
                        batch_id = None

                    if batch_id is None:
                        st.warning("Couldn't read this batch ID.")
                    else:
                        total_cost = sum_production_costs(get_all_data(), batch_id)
                        st.metric("Total logged materials cost", f"{total_cost:.2f}")

                        _pdf_key = f"prod_report_pdf_{batch_id}"
                        if st.button("Generate PDF report", use_container_width=True, key=f"gen_pdf_any_{batch_id}"):
                            st.session_state[_pdf_key] = generate_production_report_pdf_bytes(batch_id)
                        if _pdf_key in st.session_state:
                            st.download_button(
                                "⬇️ Download PDF",
                                data=st.session_state[_pdf_key],
                                file_name=f"production_report_batch_{batch_id}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                                key=f"dl_pdf_any_{batch_id}",
                            )

elif page == "Calendar":
    st.title("📅 Production Calendar")
    
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
        # Hover-only calendar (no click selection)
        
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
                        # Build a single HTML cell so events render INSIDE the day box
                        event_badges = []

                        # Calendar events
                        calendar_events = pd.DataFrame()
                        events_df = data.get("calendar_events", pd.DataFrame())
                        if not events_df.empty:
                            events_df = events_df.copy()
                            if "start_date" in events_df.columns:
                                events_df["start_date"] = pd.to_datetime(events_df["start_date"]).dt.date
                                calendar_events = events_df[events_df["start_date"] == current_date]
                        for _, event in calendar_events.iterrows():
                            event_color = {
                                "Brewing": "#4caf50",
                                "Transfer": "#2196f3",
                                "Packaging": "#ff9800",
                                "Cleaning": "#f59e0b",
                                "Maintenance": "#ef4444",
                                "Meeting": "#9c27b0",
                                "Other": "#757575",
                            }.get(event.get("event_type", "Other"), "#757575")

                            full_title = str(event.get("title", "Event") or "Event")
                            short_title = full_title
                            if len(short_title) > 21:
                                short_title = short_title[:18] + "..."

                            ev_type = str(event.get("event_type", "") or "")
                            equipment = str(event.get("equipment", "") or "")
                            notes = str(event.get("notes", "") or "").strip()
                            if len(notes) > 260:
                                notes = notes[:260] + "..."

                            tooltip_parts = [f"<div class='tt-title'>{html.escape(full_title)}</div>"]
                            meta = " • ".join([p for p in [ev_type, current_date.strftime('%Y-%m-%d')] if p])
                            if meta:
                                tooltip_parts.append(f"<div class='tt-meta'>{html.escape(meta)}</div>")
                            if equipment:
                                tooltip_parts.append(f"<div class='tt-line'><b>Equipment:</b> {html.escape(equipment)}</div>")
                            if notes:
                                tooltip_parts.append(f"<div class='tt-line'>{html.escape(notes)}</div>")
                            tooltip_html = "".join(tooltip_parts)

                            event_badges.append(
                                f"<div class='calendar-event' style='background-color:{event_color};'>"
                                f"{html.escape(short_title)}"
                                f"<div class='event-tooltip'>{tooltip_html}</div>"
                                f"</div>"
                            )

                        # Production orders
                        orders_df = data.get("production_orders", pd.DataFrame())
                        production_events = pd.DataFrame()
                        if not orders_df.empty:
                            orders_df = orders_df.copy()
                            if "start_date" in orders_df.columns:
                                orders_df["start_date"] = pd.to_datetime(orders_df["start_date"]).dt.date
                                production_events = orders_df[orders_df["start_date"] == current_date]

                        for _, order in production_events.iterrows():
                            oid = order.get("id_order", "")
                            if oid != "":
                                event_badges.append(
    f"<div class='calendar-event' style='background-color:#9c27b0;'>"
    f"Order #{oid}"
    f"<div class='event-tooltip'>"
    f"<div class='tt-title'>Production Order #{oid}</div>"
    f"<div class='tt-meta'>{current_date.strftime('%Y-%m-%d')}</div>"
    f"</div>"
    f"</div>"
)

                        badges_html = "".join(event_badges)
                        cell_html = f"""
                          <div class="{day_class}">
                            <div style=\"font-weight:600; margin-bottom:4px;\">{day}</div>
                            {badges_html}
                          </div>
                        """
                        st.markdown(cell_html, unsafe_allow_html=True)


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
                repeat_mode = st.radio(
                    "Repeat",
                    ["No repeat", "Daily", "Weekly", "Monthly"],
                    horizontal=True,
                    key="new_event_repeat_mode"
                )
                if repeat_mode != "No repeat":
                    # Sensible default horizon: 30 days (daily) / 12 weeks (weekly) / 6 months (monthly)
                    if repeat_mode == "Daily":
                        default_until = event_date + timedelta(days=30)
                    elif repeat_mode == "Weekly":
                        default_until = event_date + timedelta(weeks=12)
                    else:
                        default_until = (pd.Timestamp(event_date) + pd.DateOffset(months=6)).date()
                    repeat_until = st.date_input("Repeat until", default_until, key="new_event_repeat_until")
                else:
                    repeat_until = event_date
            
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
                    # Build the list of occurrence dates
                    dates_to_create = [event_date]

                    if repeat_mode != "No repeat":
                        # Guardrails to avoid accidental huge inserts
                        max_occurrences = 400
                        occ_count = 1
                        cur = event_date

                        # Normalize repeat_until
                        try:
                            if repeat_until < event_date:
                                repeat_until = event_date
                        except Exception:
                            repeat_until = event_date

                        while occ_count < max_occurrences:
                            if repeat_mode == "Daily":
                                cur = cur + timedelta(days=1)
                            elif repeat_mode == "Weekly":
                                cur = cur + timedelta(weeks=1)
                            else:
                                cur = (pd.Timestamp(cur) + pd.DateOffset(months=1)).date()

                            if cur > repeat_until:
                                break

                            dates_to_create.append(cur)
                            occ_count += 1

                        if occ_count >= max_occurrences:
                            st.warning("Repeat series capped at 400 occurrences.")

                    # Insert occurrences
                    created = 0
                    for d in dates_to_create:
                        new_event = {
                            "title": event_title,
                            "event_type": event_type,
                            "start_date": d,
                            "end_date": d,
                            "equipment": ", ".join(equipment) if isinstance(equipment, list) else equipment,
                            "batch_id": batch_id if batch_id != "None" else "",
                            "notes": event_notes,
                            "created_by": "User",
                        }
                        insert_data("calendar_events", new_event)
                        created += 1

                    data = get_all_data()
                    if created == 1:
                        st.success("✅ Event added successfully!")
                    else:
                        st.success(f"✅ {created} events added successfully!")
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
                    brewery_name_col = _col(breweries_df, "name", "brewery_name")
                    brewery_id_col = _col(breweries_df, "id_brewery", "brewery_id", "id")
                    brewery_id = None
                    if brewery_name_col and brewery_id_col:
                        m = breweries_df[breweries_df[brewery_name_col].astype(str) == str(cleaning_brewery)]
                        brewery_id = m[brewery_id_col].iloc[0] if not m.empty else None
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
                            brewery_name_col = _col(breweries_df, "name", "brewery_name")
                            brewery_id_col = _col(breweries_df, "id_brewery", "brewery_id", "id")
                            if (not breweries_df.empty) and brewery_name_col and brewery_id_col and str(eq.get("brewery_id")) in set(breweries_df[brewery_id_col].astype(str).values):
                                brewery_name = breweries_df.loc[breweries_df[brewery_id_col].astype(str) == str(eq.get("brewery_id")), brewery_name_col].iloc[0]
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
                        brewery_name_col = _col(breweries_df, 'name', 'brewery_name')
                        brewery_id_col = _col(breweries_df, 'id_brewery', 'brewery_id', 'id')
                        if (not breweries_df.empty) and brewery_id_col and brewery_name_col and (eq.get('brewery_id') in set(breweries_df[brewery_id_col].values)):
                            brewery_name = breweries_df.loc[breweries_df[brewery_id_col] == eq.get('brewery_id'), brewery_name_col].iloc[0]
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
st.sidebar.caption("Brewery Manager v2.0 • Multi-user (Postgres/SQLite)")
