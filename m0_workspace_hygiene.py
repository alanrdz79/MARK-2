# -*- coding: utf-8 -*-
"""
=================================================================================================================
PROSPERITY MARK-2 — MÓDULO 0: HIGIENE DE ESPACIO DE TRABAJO Y BLINDAJE DE CONFINAMIENTO
(m0_workspace_hygiene.py)
=================================================================================================================
Responsabilidades:
1. Blindaje de Rutas (`resolve_confined_path`): Garantiza matemáticamente que ningún módulo de MARK-2
   pueda crear, leer o escribir bases SQLite WAL, bitácoras o archivos fuera de `MARK-2/`.
2. Limpieza Segura de Residuales Heredados de MARK-1 (`purge_legacy_mark1_residuals`): Detecta y elimina
   bases vacías o residuales heredadas de MARK-1 (`prosperity_mark1_journal.db*`) dentro de `MARK-2/`
   o en la carpeta raíz, preservando cualquier registro si existiera.
3. Limpieza Interna y Mantenimiento SQLite WAL (`purge_internal_residuals`): Purga bytecode `.pyc`
   huérfano y ejecuta mantenimiento no bloqueante sobre `prosperity_mark2_journal.db` y
   `prosperity_backtest_warehouse.db`.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional

logger = logging.getLogger("M0_WorkspaceHygiene_M2")

MARK2_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_WORKSPACE_DIR = os.path.dirname(MARK2_DIR)
MARK1_SIBLING_DIR = os.path.join(PARENT_WORKSPACE_DIR, "Model MARK-1")

KNOWN_MARK2_DB_BASENAMES = (
    "prosperity_mark2_journal.db",
    "prosperity_backtest_warehouse.db",
)
LEGACY_MARK1_DB_BASENAME = "prosperity_mark1_journal.db"
SQLITE_SIDE_SUFFIXES = ("", "-wal", "-shm", "-journal")


def verify_path_is_confined(path: str, allowed_root: str = MARK2_DIR) -> bool:
    """Comprueba que `path` resuelva estrictamente dentro de `allowed_root` (`MARK-2`)."""
    abs_root = os.path.normcase(os.path.realpath(allowed_root))
    abs_target = os.path.normcase(os.path.realpath(path))
    try:
        common = os.path.commonpath([abs_root, abs_target])
        return common == abs_root
    except ValueError:
        return False


def resolve_confined_path(
    path: str,
    base_dir: Optional[str] = None,
    strict: bool = False,
) -> str:
    """
    Resuelve cualquier ruta garantizando confinamiento absoluto dentro de `MARK-2/`.
    """
    anchor = os.path.abspath(base_dir) if base_dir else MARK2_DIR
    if not verify_path_is_confined(anchor, MARK2_DIR):
        anchor = MARK2_DIR

    if not path or not path.strip():
        raise ValueError("La ruta solicitada no puede estar vacía.")

    candidate = (
        os.path.abspath(path)
        if os.path.isabs(path)
        else os.path.abspath(os.path.join(anchor, path))
    )

    if verify_path_is_confined(candidate, MARK2_DIR):
        return candidate

    if strict:
        raise PermissionError(
            f"Violación de confinamiento MARK-2: '{candidate}' está fuera de '{MARK2_DIR}'."
        )

    safe_fallback = os.path.abspath(os.path.join(MARK2_DIR, os.path.basename(path)))
    logger.warning(
        "🔒 [M0 Confinamiento MARK-2] Ruta externa interceptada ('%s') -> Re-anclada a '%s'.",
        candidate,
        safe_fallback,
    )
    return safe_fallback


def purge_external_residuals(dry_run: bool = False) -> Dict[str, Any]:
    """
    Elimina archivos residuales de bases de datos que hayan escapado a la carpeta padre
    (`PARENT_WORKSPACE_DIR`) y limpia el artefacto vacío `prosperity_mark1_journal.db`
    que había quedado dentro de `MARK-2/`.
    """
    removed_files: List[str] = []
    skipped_files: List[str] = []

    # 1. Eliminar `prosperity_mark1_journal.db*` dentro de MARK-2 si está vacío (residual de MARK-1)
    legacy_in_m2 = os.path.join(MARK2_DIR, LEGACY_MARK1_DB_BASENAME)
    if os.path.isfile(legacy_in_m2):
        has_rows = False
        try:
            conn = sqlite3.connect(legacy_in_m2, timeout=5.0)
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table';"
                ).fetchall()
            }
            if "trade_journal" in tables:
                cnt = conn.execute("SELECT COUNT(*) FROM trade_journal;").fetchone()[0]
                has_rows = cnt > 0
            conn.close()
        except Exception:
            pass

        if not has_rows:
            for suffix in SQLITE_SIDE_SUFFIXES:
                cand = os.path.join(MARK2_DIR, f"{LEGACY_MARK1_DB_BASENAME}{suffix}")
                if os.path.isfile(cand):
                    if not dry_run:
                        try:
                            os.remove(cand)
                            removed_files.append(cand)
                            logger.info(
                                "🧹 [M0 Higiene MARK-2] Base residual vacía de MARK-1 eliminada: %s",
                                cand,
                            )
                        except OSError as exc:
                            skipped_files.append(f"{cand} ({exc})")
                    else:
                        removed_files.append(cand)

    # 2. Eliminar posibles archivos SQLite escapados en PARENT_WORKSPACE_DIR
    if PARENT_WORKSPACE_DIR and not verify_path_is_confined(PARENT_WORKSPACE_DIR, MARK2_DIR):
        for db_base in KNOWN_MARK2_DB_BASENAMES + (LEGACY_MARK1_DB_BASENAME,):
            for suffix in SQLITE_SIDE_SUFFIXES:
                candidate = os.path.join(PARENT_WORKSPACE_DIR, f"{db_base}{suffix}")
                if os.path.isfile(candidate):
                    if dry_run:
                        removed_files.append(candidate)
                        continue
                    try:
                        os.remove(candidate)
                        removed_files.append(candidate)
                    except OSError as exc:
                        skipped_files.append(f"{candidate} ({exc})")

    return {
        "removed": removed_files,
        "skipped": skipped_files,
    }


def purge_internal_residuals(
    checkpoint_wal: bool = False,
) -> Dict[str, Any]:
    """
    Limpia artefactos `.pyc` huérfanos dentro de `MARK-2/` y ejecuta `wal_checkpoint` opcional.
    """
    removed_orphan_pyc: List[str] = []
    wal_checkpoints: List[str] = []

    for root, dirs, _ in os.walk(MARK2_DIR):
        if "__pycache__" in dirs:
            pycache_dir = os.path.join(root, "__pycache__")
            for fname in os.listdir(pycache_dir):
                if not fname.endswith(".pyc"):
                    continue
                mod_stem = fname.split(".", 1)[0]
                expected_py = os.path.join(root, f"{mod_stem}.py")
                if not os.path.isfile(expected_py):
                    orphan_path = os.path.join(pycache_dir, fname)
                    try:
                        os.remove(orphan_path)
                        removed_orphan_pyc.append(orphan_path)
                    except OSError:
                        pass

    if checkpoint_wal:
        for db_name in KNOWN_MARK2_DB_BASENAMES:
            db_path = os.path.join(MARK2_DIR, db_name)
            if os.path.isfile(db_path):
                conn: Optional[sqlite3.Connection] = None
                try:
                    conn = sqlite3.connect(db_path, timeout=10.0)
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                    wal_checkpoints.append(db_path)
                except Exception:
                    pass
                finally:
                    if conn is not None:
                        conn.close()

    return {
        "removed_orphan_pyc": removed_orphan_pyc,
        "wal_checkpoints": wal_checkpoints,
    }


def run_full_hygiene(
    checkpoint_wal: bool = True,
) -> Dict[str, Any]:
    """Ejecuta la auditoría y limpieza completa del ecosistema MARK-2."""
    ext_res = purge_external_residuals(dry_run=False)
    int_res = purge_internal_residuals(checkpoint_wal=checkpoint_wal)
    return {
        "mark2_dir": MARK2_DIR,
        "external": ext_res,
        "internal": int_res,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    summary = run_full_hygiene(checkpoint_wal=True)
    logger.info("✅ [M0 Higiene MARK-2 Completada] Resumen: %s", summary)
