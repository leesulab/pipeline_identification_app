# pages/pipeline.py
# -*- coding:utf-8 -*-
"""
Module d'exécution de la pipeline d'identification MS directement depuis Streamlit.
Appelle les fonctions Python du package pipeline_identification_ms.
"""

import importlib.util
import sys
import time
import traceback
from pathlib import Path
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import fields

import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers pour ajouter le package pipeline au sys.path
# ---------------------------------------------------------------------------

def _ensure_pipeline_on_path():
    """Ajoute le dossier pipeline_identification_ms au sys.path si nécessaire."""
    # 1. Chercher en tant que voisin du dossier de l'app
    app_dir = Path(__file__).resolve().parent.parent  # pipeline_identification_app/
    pipeline_dir = app_dir.parent / "pipeline_identification_ms"

    # 2. Fallback : chemin configuré par l'utilisateur
    if not pipeline_dir.exists():
        custom = st.session_state.get("pipeline_ms_path", "")
        if custom:
            pipeline_dir = Path(custom)

    # 3. Fallback : à côté du working_dir
    if not pipeline_dir.exists():
        wd = st.session_state.get("working_dir", "")
        if wd:
            pipeline_dir = Path(wd).parent / "pipeline_identification_ms"
    if pipeline_dir.exists() and str(pipeline_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_dir))
    return pipeline_dir


def _get_reusable_pipeline_runner():
    """Charge le moteur réutilisable de pipeline_identification_ms."""
    pipeline_dir = _ensure_pipeline_on_path()
    main_file = pipeline_dir / "main.py"

    if not main_file.exists():
        raise FileNotFoundError(
            f"Point d'entrée de la pipeline introuvable : {main_file}"
        )

    module_name = "pipeline_identification_ms_main"
    module = sys.modules.get(module_name)

    if module is None:
        spec = importlib.util.spec_from_file_location(
            module_name,
            main_file,
        )

        if spec is None or spec.loader is None:
            raise ImportError(
                f"Impossible de charger le moteur depuis {main_file}"
            )

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    runner = getattr(module, "run_pipeline", None)

    if runner is None or not callable(runner):
        raise AttributeError(
            f"La fonction run_pipeline est absente de {main_file}"
        )

    return runner


def _get_pipeline_imports():
    """Importe dynamiquement les modules de la pipeline."""
    _ensure_pipeline_on_path()
    from scripts.config.config import (
        Config, PathConfig, PeakDetectionConfig, IntraClusteringConfig,
        FeatureAlignmentConfig, BlankSubtractionConfig, BlankReplicateConfig,
        ReplicateConfig, IdentificationConfig, MS2ExtractionConfig, ProcessingConfig,
    )
    from scripts.processing.peak_detection import PeakDetector
    from scripts.processing.blank_processing import BlankProcessor
    from scripts.processing.replicate_processing import ReplicateProcessor
    from scripts.processing.ccs_calibration import CCSCalibrator
    from scripts.processing.identification import CompoundIdentifier
    from scripts.processing.feature_matrix import FeatureProcessor
    from scripts.processing.ms2_extraction import MS2Extractor
    from scripts.processing.ms2_comparaison import MS2Comparator
    from scripts.utils.io_handlers import IOHandler
    from scripts.utils.replicate_handling import ReplicateHandler

    # Import visualization functions directly
    try:
        from scripts.visualization.plotting import (
            plot_unique_molecules_per_sample,
            plot_level1_molecules_per_sample,
            plot_sample_similarity_heatmap,
            plot_sample_similarity_heatmap_by_confidence,
            plot_level1_molecule_distribution_bubble,
            analyze_and_save_clusters,
            analyze_categories,
        )
        viz_funcs = {
            "plot_unique_molecules_per_sample": plot_unique_molecules_per_sample,
            "plot_level1_molecules_per_sample": plot_level1_molecules_per_sample,
            "plot_sample_similarity_heatmap": plot_sample_similarity_heatmap,
            "plot_sample_similarity_heatmap_by_confidence": plot_sample_similarity_heatmap_by_confidence,
            "plot_level1_molecule_distribution_bubble": plot_level1_molecule_distribution_bubble,
            "analyze_and_save_clusters": analyze_and_save_clusters,
            "analyze_categories": analyze_categories,
        }
    except ImportError:
        viz_funcs = {}

    return {
        "Config": Config,
        "PathConfig": PathConfig,
        "PeakDetectionConfig": PeakDetectionConfig,
        "IntraClusteringConfig": IntraClusteringConfig,
        "FeatureAlignmentConfig": FeatureAlignmentConfig,
        "BlankSubtractionConfig": BlankSubtractionConfig,
        "BlankReplicateConfig": BlankReplicateConfig,
        "ReplicateConfig": ReplicateConfig,
        "IdentificationConfig": IdentificationConfig,
        "MS2ExtractionConfig": MS2ExtractionConfig,
        "ProcessingConfig": ProcessingConfig,
        "PeakDetector": PeakDetector,
        "BlankProcessor": BlankProcessor,
        "ReplicateProcessor": ReplicateProcessor,
        "CCSCalibrator": CCSCalibrator,
        "CompoundIdentifier": CompoundIdentifier,
        "FeatureProcessor": FeatureProcessor,
        "MS2Extractor": MS2Extractor,
        "MS2Comparator": MS2Comparator,
        "IOHandler": IOHandler,
        "ReplicateHandler": ReplicateHandler,
        **viz_funcs,
    }


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_pipeline_state():
    defaults = {
        "working_dir": "",
        "pipeline_ms_path": "",
        "pipeline_logs": [],
        "pipeline_step": 0,
        "pipeline_results": {},
        "config_overrides": {},
        "pipeline_running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Configuration editor
# ---------------------------------------------------------------------------

_CONFIG_SECTIONS = [
    ("Détection des pics", "PeakDetectionConfig", "PEAK_DETECTION"),
    ("Clustering intra-échantillon", "IntraClusteringConfig", "INTRA_CLUSTERING"),
    ("Réplicats blanks", "BlankReplicateConfig", "BLANK_REPLICATE"),
    ("Alignement des features", "FeatureAlignmentConfig", "FEATURE_ALIGNMENT"),
    ("Soustraction du blank", "BlankSubtractionConfig", "BLANK_SUBTRACTION"),
    ("Réplicats échantillons", "ReplicateConfig", "REPLICATE"),
    ("Identification", "IdentificationConfig", "IDENTIFICATION"),
    ("Extraction MS2", "MS2ExtractionConfig", "MS2_EXTRACTION"),
    ("Traitement général", "ProcessingConfig", "PROCESSING"),
]


def _render_field_input(section_key: str, field_name: str, field_type, default_value, key_prefix: str):
    """Rend un widget Streamlit adapté au type du champ."""
    full_key = f"cfg_{key_prefix}_{field_name}"

    # Valeur actuelle (override ou défaut)
    current = st.session_state.get("config_overrides", {}).get(section_key, {}).get(field_name, default_value)

    if isinstance(default_value, bool):
        val = st.checkbox(field_name, value=current, key=full_key)
    elif isinstance(default_value, int):
        val = st.number_input(field_name, value=int(current), step=1, key=full_key)
    elif isinstance(default_value, float):
        val = st.number_input(field_name, value=float(current), format="%.6f", step=0.001, key=full_key)
    elif isinstance(default_value, str):
        val = st.text_input(field_name, value=str(current), key=full_key)
    elif isinstance(default_value, dict):
        st.markdown(f"**{field_name}** *(dict)*")
        val = {}
        for dk, dv in default_value.items():
            sub_key = f"{full_key}_{dk}"
            current_sub = current.get(dk, dv) if isinstance(current, dict) else dv
            if isinstance(dv, (int, float)):
                val[dk] = st.number_input(
                    f"  {dk}", value=float(current_sub) if isinstance(dv, float) else int(current_sub),
                    format="%.6f" if isinstance(dv, float) else "%d",
                    step=0.001 if isinstance(dv, float) else 1,
                    key=sub_key,
                )
            else:
                val[dk] = st.text_input(f"  {dk}", value=str(current_sub), key=sub_key)
    else:
        val = st.text_input(field_name, value=str(current), key=full_key)

    return val


def render_config_editor(imports: dict):
    """Affiche l'éditeur de configuration complet."""
    st.subheader("⚙️ Configuration de la pipeline")

    if "config_overrides" not in st.session_state:
        st.session_state["config_overrides"] = {}

    tabs = st.tabs([label for label, _, _ in _CONFIG_SECTIONS])

    for tab, (label, cls_name, attr_name) in zip(tabs, _CONFIG_SECTIONS):
        with tab:
            cls = imports[cls_name]
            defaults_instance = cls()
            new_overrides = {}
            for f in fields(defaults_instance):
                val = _render_field_input(attr_name, f.name, f.type, getattr(defaults_instance, f.name), attr_name)
                new_overrides[f.name] = val

            st.session_state["config_overrides"][attr_name] = new_overrides

    if st.button("🔄 Réinitialiser la configuration", key="reset_config"):
        st.session_state["config_overrides"] = {}
        st.rerun()


def _apply_config_overrides(Config):
    """Applique les overrides de session_state sur l'objet Config global."""
    overrides = st.session_state.get("config_overrides", {})
    for attr_name, values in overrides.items():
        config_obj = getattr(Config, attr_name, None)
        if config_obj is None:
            continue
        for field_name, field_value in values.items():
            if hasattr(config_obj, field_name):
                setattr(config_obj, field_name, field_value)


# ---------------------------------------------------------------------------
# Working directory & file discovery
# ---------------------------------------------------------------------------

def render_working_directory(imports: dict):
    """UI pour configurer le dossier de travail et le chemin du pipeline."""
    st.subheader("📂 Dossier de travail")

    Config = imports["Config"]

    # Chemin optionnel vers pipeline_identification_ms
    col_path1, col_path2 = st.columns(2)
    with col_path1:
        wd = st.text_input(
            "Chemin du dossier de travail (contenant data/input/samples, etc.)",
            value=st.session_state.get("working_dir", str(Config.PATHS.BASE_DIR)),
            key="wd_input",
        )
    with col_path2:
        ms_path = st.text_input(
            "Chemin vers pipeline_identification_ms (optionnel)",
            value=st.session_state.get("pipeline_ms_path", ""),
            key="ms_path_input",
            help="Laisser vide pour détection automatique (dossier voisin de l'app).",
        )

    if st.button("Appliquer le dossier de travail", key="apply_wd"):
        p = Path(wd)
        if p.exists():
            st.session_state["working_dir"] = str(p)
            # Mettre à jour les chemins de Config
            Config.PATHS.BASE_DIR = p
            Config.PATHS.DATA_DIR = p / "data"
            Config.PATHS.INPUT_DIR = p / "data" / "input"
            Config.PATHS.OUTPUT_DIR = p / "data" / "output"
            Config.PATHS.INTERMEDIATE_DIR = p / "data" / "intermediate"
            Config.PATHS.INPUT_SAMPLES = p / "data" / "input" / "samples"
            Config.PATHS.INPUT_BLANKS = p / "data" / "input" / "blanks"
            Config.PATHS.INPUT_CALIBRANTS = p / "data" / "input" / "calibration"
            Config.PATHS.INPUT_DATABASES = p / "data" / "input" / "databases"
            Config.PATHS.INTERMEDIATE_SAMPLES = p / "data" / "intermediate" / "samples"
            Config.PATHS.INTERMEDIATE_CALIBRANTS = p / "data" / "intermediate" / "calibrants"
            Config.setup_directories()
            st.success(f"✅ Dossier de travail : {p}")
        else:
            st.error(f"❌ Le dossier {p} n'existe pas.")

        if ms_path.strip():
            mp_ = Path(ms_path.strip())
            if mp_.exists():
                st.session_state["pipeline_ms_path"] = str(mp_)
                st.success(f"✅ Pipeline MS : {mp_}")
            else:
                st.error(f"❌ Le dossier {mp_} n'existe pas.")

    # Affichage de l'arborescence détectée
    wd_path = Path(st.session_state.get("working_dir", ""))
    if wd_path.exists():
        _show_detected_files(wd_path, Config)


def _show_detected_files(wd: Path, Config):
    """Affiche les fichiers détectés dans le dossier de travail."""
    cols = st.columns(3)

    samples_dir = Config.PATHS.INPUT_SAMPLES
    blanks_dir = Config.PATHS.INPUT_BLANKS
    calib_dir = Config.PATHS.INPUT_CALIBRANTS
    db_dir = Config.PATHS.INPUT_DATABASES

    with cols[0]:
        st.markdown("**🧪 Échantillons**")
        if samples_dir.exists():
            files = sorted(samples_dir.glob("*.parquet"))
            if files:
                for f in files:
                    st.caption(f"  • {f.name}")
                st.info(f"{len(files)} fichier(s)")
            else:
                st.warning("Aucun fichier .parquet")
        else:
            st.warning(f"Dossier absent : {samples_dir}")

    with cols[1]:
        st.markdown("**🫧 Blanks**")
        if blanks_dir.exists():
            files = sorted(blanks_dir.glob("*.parquet"))
            if files:
                for f in files:
                    st.caption(f"  • {f.name}")
                st.info(f"{len(files)} fichier(s)")
            else:
                st.warning("Aucun fichier .parquet")
        else:
            st.warning(f"Dossier absent : {blanks_dir}")

    with cols[2]:
        st.markdown("**📏 Calibration & DB**")
        if calib_dir.exists():
            for f in sorted(calib_dir.glob("*")):
                st.caption(f"  📏 {f.name}")
        if db_dir.exists():
            for f in sorted(db_dir.glob("*")):
                st.caption(f"  📚 {f.name}")


# ---------------------------------------------------------------------------
# Upload de fichiers
# ---------------------------------------------------------------------------

def render_file_upload(imports: dict):
    """UI pour uploader des fichiers dans le dossier de travail."""
    st.subheader("📤 Charger des fichiers")

    Config = imports["Config"]

    dest_options = {
        "Échantillons (.parquet)": Config.PATHS.INPUT_SAMPLES,
        "Blanks (.parquet)": Config.PATHS.INPUT_BLANKS,
        "Calibration CCS (.csv)": Config.PATHS.INPUT_CALIBRANTS,
        "Base de données (.h5)": Config.PATHS.INPUT_DATABASES,
    }

    dest = st.selectbox("Destination", list(dest_options.keys()), key="upload_dest")
    uploaded = st.file_uploader(
        "Sélectionner les fichiers",
        accept_multiple_files=True,
        key="pipeline_file_upload",
    )

    if uploaded and st.button("💾 Sauvegarder", key="save_uploads"):
        dest_dir = dest_options[dest]
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in uploaded:
            target = dest_dir / f.name
            target.write_bytes(f.getbuffer())
            st.success(f"✅ {f.name} → {target}")
        st.rerun()


# ---------------------------------------------------------------------------
# Pipeline steps execution
# ---------------------------------------------------------------------------

PIPELINE_STEPS = [
    ("calibration", "1. Chargement calibration CCS"),
    ("identification", "2. Initialisation identification"),
    ("file_discovery", "3. Détection des fichiers"),
    ("blanks", "4. Traitement des blanks"),
    (
        "samples",
        "5. Traitement des échantillons "
        "(réplicats + blank + CCS)",
    ),
    (
        "feature_alignment",
        "6. Alignement des features & identification",
    ),
    ("visualizations", "7. Génération des visualisations"),
    ("similarities", "8. Analyse des similarités"),
    ("categories", "9. Analyse des catégories"),
]


def _log(msg: str):
    """Ajoute un message au log pipeline."""
    st.session_state["pipeline_logs"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")


def _capture_output(func, *args, **kwargs):
    """Exécute une fonction en capturant stdout/stderr."""
    buf = StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            result = func(*args, **kwargs)
        output = buf.getvalue()
        if output.strip():
            for line in output.strip().split("\n"):
                _log(line)
        return result
    except Exception as e:
        output = buf.getvalue()
        if output.strip():
            for line in output.strip().split("\n"):
                _log(line)
        raise


def run_pipeline(imports: dict, selected_steps: list):
    """Exécute les étapes sélectionnées avec le moteur partagé."""
    Config = imports["Config"]
    _apply_config_overrides(Config)

    st.session_state["pipeline_running"] = True
    st.session_state["pipeline_logs"] = []

    progress_bar = st.progress(0.0)
    status_area = st.empty()
    log_area = st.empty()

    working_dir = st.session_state.get(
        "working_dir",
        str(Config.PATHS.BASE_DIR),
    )

    def refresh_logs():
        log_area.code(
            "\n".join(st.session_state["pipeline_logs"]),
            language="text",
        )

    def progress_callback(event):
        step = event.get("step", "pipeline")
        status = event.get("status", "running")
        message = event.get("message", "")
        progress = event.get("progress")

        _log(
            f"[{step}] {status}: {message}"
        )

        if progress is not None:
            progress_bar.progress(
                max(0.0, min(1.0, float(progress)))
            )

        if status == "error":
            status_area.error(message)
        elif status == "success":
            status_area.success(message)
        else:
            status_area.info(message)

        refresh_logs()

    try:
        engine_runner = _get_reusable_pipeline_runner()

        _log(
            "Étapes sélectionnées : "
            + ", ".join(selected_steps)
        )
        _log(f"Dossier de travail : {working_dir}")
        refresh_logs()

        output_buffer = StringIO()

        with redirect_stdout(output_buffer), redirect_stderr(
            output_buffer
        ):
            result = engine_runner(
                base_dir=working_dir,
                steps=selected_steps,
                progress_callback=progress_callback,
            )

        captured_output = output_buffer.getvalue()

        if captured_output.strip():
            for line in captured_output.strip().splitlines():
                _log(line)

        st.session_state["pipeline_results"] = result

        progress_bar.progress(1.0)
        status_area.success(
            "✅ Étapes sélectionnées terminées"
        )

        _log("=== Exécution terminée avec succès ===")
        refresh_logs()

    except Exception as error:
        captured_output = (
            output_buffer.getvalue()
            if "output_buffer" in locals()
            else ""
        )

        if captured_output.strip():
            for line in captured_output.strip().splitlines():
                _log(line)

        _log(f"❌ ERREUR : {error}")
        _log(traceback.format_exc())

        status_area.error(f"❌ Erreur : {error}")
        refresh_logs()

    finally:
        st.session_state["pipeline_running"] = False


# ---------------------------------------------------------------------------
# Interface principale de la page pipeline
# ---------------------------------------------------------------------------

def render_pipeline_page():
    """Point d'entrée pour la page Pipeline."""
    _init_pipeline_state()

    st.title("🚀 Pipeline d'identification MS")
    st.markdown("Exécutez la pipeline de traitement directement depuis cette interface.")

    try:
        imports = _get_pipeline_imports()
    except Exception as e:
        st.error(f"❌ Impossible de charger les modules de la pipeline : {e}")
        st.info("Vérifiez que le dossier `pipeline_identification_ms` est accessible depuis le dossier de travail.")
        st.code(traceback.format_exc(), language="text")
        return

    # --- Onglets principaux ---
    tab_dir, tab_config, tab_run, tab_logs = st.tabs([
        "📂 Dossier de travail",
        "⚙️ Configuration",
        "▶️ Exécution",
        "📋 Logs",
    ])

    with tab_dir:
        render_working_directory(imports)
        st.divider()
        render_file_upload(imports)

    with tab_config:
        render_config_editor(imports)

    with tab_run:
        render_execution_ui(imports)

    with tab_logs:
        render_logs()


def render_execution_ui(imports: dict):
    """UI pour lancer les étapes de la pipeline."""
    st.subheader("▶️ Exécution de la pipeline")

    # Sélection des étapes
    st.markdown("**Sélectionner les étapes à exécuter :**")
    selected = []
    cols = st.columns(2)

    for i, (step_id, step_label) in enumerate(
        PIPELINE_STEPS
    ):
        with cols[i % 2]:
            if st.checkbox(
                step_label,
                value=True,
                key=f"step_{step_id}",
            ):
                selected.append(step_id)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Lancer la pipeline", type="primary", key="run_pipeline",
                      disabled=st.session_state.get("pipeline_running", False)):
            if not selected:
                st.warning("Aucune étape sélectionnée.")
            else:
                run_pipeline(imports, selected)

    with col2:
        if st.button("🗑️ Réinitialiser les résultats", key="reset_results"):
            st.session_state["pipeline_results"] = {}
            st.session_state["pipeline_logs"] = []
            st.session_state["pipeline_step"] = 0
            st.rerun()

    # Affichage des résultats intermédiaires
    results = st.session_state.get("pipeline_results", {})
    if results:
        st.divider()
        st.subheader("📊 Résultats intermédiaires")

        if "sample_results" in results:
            sample_results = results["sample_results"]
            stats_data = []
            for name, df in sample_results.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    stats_data.append({
                        "Échantillon": name,
                        "Pics": len(df),
                        "CCS min": f"{df['CCS'].min():.1f}" if "CCS" in df.columns else "N/A",
                        "CCS max": f"{df['CCS'].max():.1f}" if "CCS" in df.columns else "N/A",
                        "m/z min": f"{df['mz'].min():.3f}",
                        "m/z max": f"{df['mz'].max():.3f}",
                    })
            if stats_data:
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

        # Lien vers les fichiers de sortie
        Config = imports["Config"]
        output_dir = Config.PATHS.OUTPUT_DIR / "feature_matrix"
        if output_dir.exists():
            output_files = list(output_dir.glob("*"))
            if output_files:
                st.markdown("**Fichiers de sortie :**")
                for f in sorted(output_files):
                    st.caption(f"  📄 {f.name} ({f.stat().st_size / 1024:.0f} KB)")


def render_logs():
    """Affiche les logs de la pipeline."""
    st.subheader("📋 Logs de la pipeline")

    logs = st.session_state.get("pipeline_logs", [])
    if logs:
        st.code("\n".join(logs), language="text")
        if st.button("📋 Copier les logs", key="copy_logs"):
            st.text_area("Logs (copier ci-dessous)", value="\n".join(logs), height=300, key="logs_textarea")
        if st.button("🗑️ Effacer les logs", key="clear_logs"):
            st.session_state["pipeline_logs"] = []
            st.rerun()
    else:
        st.info("Aucun log disponible. Lancez la pipeline pour voir les logs.")


# Supporte l'exécution directe en page Streamlit multipage (pages/pipeline.py).
# Lors d'un import depuis app.py, ce bloc n'est pas exécuté.
if __name__ == "__main__":
    render_pipeline_page()
