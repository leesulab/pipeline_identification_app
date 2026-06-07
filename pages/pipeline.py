# pages/pipeline.py
# -*- coding:utf-8 -*-
"""
Module d'exécution de la pipeline d'identification MS directement depuis Streamlit.
Appelle les fonctions Python du package pipeline_identification_ms.
"""

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
    "1. Chargement calibration CCS",
    "2. Initialisation identification",
    "3. Détection fichiers",
    "4. Traitement des blanks",
    "5. Traitement des échantillons (réplicats + blank + CCS)",
    "6. Alignement des features & identification",
    "7. Génération des visualisations",
    "8. Analyse des catégories",
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
    """Exécute les étapes sélectionnées de la pipeline."""
    Config = imports["Config"]
    _apply_config_overrides(Config)

    st.session_state["pipeline_running"] = True
    st.session_state["pipeline_logs"] = []
    results = st.session_state.get("pipeline_results", {})

    progress_bar = st.progress(0)
    status_area = st.empty()
    log_area = st.empty()

    total_steps = len(selected_steps)

    try:
        # --- Étape 1 : Calibration CCS ---
        if 0 in selected_steps:
            step_idx = selected_steps.index(0)
            progress_bar.progress((step_idx) / total_steps)
            status_area.info("📈 Chargement des données de calibration CCS...")
            _log("=== Chargement calibration CCS ===")

            calibration_file = Config.PATHS.INPUT_CALIBRANTS / "CCS_calibration_data.csv"
            if not calibration_file.exists():
                csv_files = list(Config.PATHS.INPUT_CALIBRANTS.glob("*.csv"))
                if csv_files:
                    calibration_file = csv_files[0]
                    _log(f"Fichier de calibration trouvé : {calibration_file.name}")
                else:
                    raise FileNotFoundError(f"Aucun fichier de calibration dans {Config.PATHS.INPUT_CALIBRANTS}")

            CCSCalibrator = imports["CCSCalibrator"]
            calibrator = _capture_output(CCSCalibrator, str(calibration_file))
            results["calibrator"] = calibrator
            _log(f"✓ Calibration chargée depuis {calibration_file.name}")
            log_area.code("\n".join(st.session_state["pipeline_logs"]), language="text")

        # --- Étape 2 : Initialisation identification ---
        if 1 in selected_steps:
            step_idx = selected_steps.index(1)
            progress_bar.progress((step_idx) / total_steps)
            status_area.info("📚 Initialisation de l'identification...")
            _log("=== Initialisation identification ===")

            CompoundIdentifier = imports["CompoundIdentifier"]
            identifier = _capture_output(CompoundIdentifier)
            results["identifier"] = identifier
            _log(f"✓ Base de données chargée ({len(identifier.db)} composés)")
            log_area.code("\n".join(st.session_state["pipeline_logs"]), language="text")

        # --- Étape 3 : Détection fichiers ---
        if 2 in selected_steps:
            step_idx = selected_steps.index(2)
            progress_bar.progress((step_idx) / total_steps)
            status_area.info("📁 Recherche des fichiers...")
            _log("=== Détection des fichiers ===")

            ReplicateHandler = imports["ReplicateHandler"]
            replicate_handler = ReplicateHandler()

            blank_files = sorted(Config.PATHS.INPUT_BLANKS.glob("*.parquet"))
            _log(f"Blanks trouvés : {len(blank_files)}")
            for f in blank_files:
                _log(f"  • {f.name}")
            results["blank_files"] = blank_files

            sample_files = sorted(Config.PATHS.INPUT_SAMPLES.glob("*.parquet"))
            if not sample_files:
                raise ValueError(f"Aucun échantillon dans {Config.PATHS.INPUT_SAMPLES}")

            replicate_groups = replicate_handler.group_replicates(sample_files)
            _log(f"Échantillons : {len(replicate_groups)} groupes")
            for name, reps in replicate_groups.items():
                _log(f"  • {name}: {len(reps)} réplicat(s)")

            results["sample_files"] = sample_files
            results["replicate_groups"] = replicate_groups
            log_area.code("\n".join(st.session_state["pipeline_logs"]), language="text")

        # --- Étape 4 : Traitement des blanks ---
        if 3 in selected_steps:
            step_idx = selected_steps.index(3)
            progress_bar.progress((step_idx) / total_steps)
            status_area.info("🫧 Traitement des blanks...")
            _log("=== Traitement des blanks ===")

            BlankProcessor = imports["BlankProcessor"]
            blank_processor = BlankProcessor()

            blank_files = results.get("blank_files", sorted(Config.PATHS.INPUT_BLANKS.glob("*.parquet")))

            blank_peaks = pd.DataFrame()
            if blank_files:
                blank_peaks = _capture_output(
                    blank_processor.process_blank_with_replicates,
                    blank_files[0].stem,
                    blank_files,
                    Config.PATHS.INTERMEDIATE_DIR / "blanks",
                )
                _log(f"✓ {len(blank_peaks)} pics blanks détectés")
            else:
                _log("ℹ️ Aucun blank trouvé, étape ignorée")

            results["blank_peaks"] = blank_peaks
            log_area.code("\n".join(st.session_state["pipeline_logs"]), language="text")

        # --- Étape 5 : Traitement des échantillons ---
        if 4 in selected_steps:
            step_idx = selected_steps.index(4)
            progress_bar.progress((step_idx) / total_steps)
            status_area.info("🧪 Traitement des échantillons...")
            _log("=== Traitement des échantillons ===")

            ReplicateProcessor = imports["ReplicateProcessor"]
            BlankProcessor = imports["BlankProcessor"]
            IOHandler = imports["IOHandler"]

            replicate_processor = ReplicateProcessor()
            blank_processor = BlankProcessor()
            io_handler = IOHandler()

            calibrator = results.get("calibrator")
            if calibrator is None:
                CCSCalibrator = imports["CCSCalibrator"]
                calibration_file = Config.PATHS.INPUT_CALIBRANTS / "CCS_calibration_data.csv"
                csv_files = list(Config.PATHS.INPUT_CALIBRANTS.glob("*.csv"))
                if csv_files:
                    calibration_file = csv_files[0]
                calibrator = CCSCalibrator(str(calibration_file))
                results["calibrator"] = calibrator

            blank_peaks = results.get("blank_peaks", pd.DataFrame())
            replicate_groups = results.get("replicate_groups")
            if replicate_groups is None:
                ReplicateHandler = imports["ReplicateHandler"]
                rh = ReplicateHandler()
                sample_files = sorted(Config.PATHS.INPUT_SAMPLES.glob("*.parquet"))
                replicate_groups = rh.group_replicates(sample_files)

            output_base_dir = Config.PATHS.INTERMEDIATE_SAMPLES
            sample_results = {}
            total = len(replicate_groups)

            for i, (base_name, replicates) in enumerate(replicate_groups.items()):
                _log(f"\n--- {base_name} ({i+1}/{total}) ---")
                progress_bar.progress((step_idx + (i / total)) / total_steps)

                try:
                    common_peaks = _capture_output(
                        replicate_processor.process_sample_with_replicates,
                        base_name, replicates, output_base_dir,
                    )

                    if common_peaks.empty:
                        _log(f"✗ Pas de pics pour {base_name}")
                        continue

                    if not blank_peaks.empty:
                        clean_peaks = _capture_output(
                            blank_processor.subtract_blank_peaks, common_peaks, blank_peaks
                        )
                    else:
                        clean_peaks = common_peaks

                    if clean_peaks.empty:
                        _log(f"✗ Pas de pics après blank pour {base_name}")
                        continue

                    peaks_with_ccs = _capture_output(calibrator.calculate_ccs, clean_peaks)

                    if not peaks_with_ccs.empty:
                        _log(f"✓ {base_name}: {len(peaks_with_ccs)} pics, CCS [{peaks_with_ccs['CCS'].min():.1f} - {peaks_with_ccs['CCS'].max():.1f}] Å²")
                        output_dir = output_base_dir / base_name / "ms1"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        io_handler.save_results(peaks_with_ccs, output_dir / "common_peaks.parquet")
                        sample_results[base_name] = peaks_with_ccs

                except Exception as e:
                    _log(f"❌ Erreur {base_name}: {e}")
                    _log(traceback.format_exc())

                log_area.code("\n".join(st.session_state["pipeline_logs"]), language="text")

            results["sample_results"] = sample_results
            _log(f"\n✓ {len(sample_results)}/{total} échantillons traités avec succès")
            log_area.code("\n".join(st.session_state["pipeline_logs"]), language="text")

        # --- Étape 6 : Feature matrix & identification ---
        if 5 in selected_steps:
            step_idx = selected_steps.index(5)
            progress_bar.progress((step_idx) / total_steps)
            status_area.info("📊 Alignement des features & identification...")
            _log("=== Alignement des features & identification ===")

            FeatureProcessor = imports["FeatureProcessor"]
            feature_processor = FeatureProcessor()

            identifier = results.get("identifier")
            if identifier is None:
                CompoundIdentifier = imports["CompoundIdentifier"]
                identifier = CompoundIdentifier()
                results["identifier"] = identifier

            _capture_output(
                feature_processor.create_feature_matrix,
                input_dir=Config.PATHS.INTERMEDIATE_SAMPLES,
                output_dir=Config.PATHS.OUTPUT_DIR / "feature_matrix",
                identifier=identifier,
            )
            _log("✓ Matrice des features créée et identifications terminées")
            log_area.code("\n".join(st.session_state["pipeline_logs"]), language="text")

        # --- Étape 7 : Visualisations ---
        if 6 in selected_steps:
            step_idx = selected_steps.index(6)
            progress_bar.progress((step_idx) / total_steps)
            status_area.info("📊 Génération des visualisations...")
            _log("=== Génération des visualisations ===")

            import matplotlib.pyplot as plt

            output_dir = Config.PATHS.OUTPUT_DIR
            identifications_file = output_dir / "feature_matrix" / "features_complete.parquet"
            if identifications_file.exists():
                try:
                    plot_funcs = {
                        "plot_unique_molecules_per_sample": imports.get("plot_unique_molecules_per_sample"),
                        "plot_level1_molecules_per_sample": imports.get("plot_level1_molecules_per_sample"),
                        "plot_sample_similarity_heatmap": imports.get("plot_sample_similarity_heatmap"),
                        "plot_level1_molecule_distribution_bubble": imports.get("plot_level1_molecule_distribution_bubble"),
                    }

                    for func_name, func in plot_funcs.items():
                        if func is not None:
                            try:
                                fig = _capture_output(func, output_dir)
                                if fig is not None:
                                    fig.savefig(
                                        output_dir / f"{func_name}.png",
                                        bbox_inches="tight", dpi=300,
                                    )
                                    plt.close(fig)
                                    _log(f"  ✓ {func_name}")
                            except Exception as ve:
                                _log(f"  ⚠️ {func_name}: {ve}")

                    # Heatmaps par niveaux de confiance
                    heatmap_func = imports.get("plot_sample_similarity_heatmap_by_confidence")
                    if heatmap_func is not None:
                        for levels, suffix in [
                            ([1], "level1"),
                            ([1, 2], "level12"),
                            ([1, 2, 3], "level123"),
                        ]:
                            try:
                                fig = _capture_output(
                                    heatmap_func, output_dir,
                                    confidence_levels=levels,
                                    title_suffix=f" - Niveaux {','.join(map(str, levels))}",
                                )
                                if fig is not None:
                                    fig.savefig(
                                        output_dir / f"sample_similarity_heatmap_{suffix}.png",
                                        bbox_inches="tight", dpi=300,
                                    )
                                    plt.close(fig)
                                    _log(f"  ✓ heatmap {suffix}")
                            except Exception as ve:
                                _log(f"  ⚠️ heatmap {suffix}: {ve}")

                    # Analyse des clusters
                    cluster_func = imports.get("analyze_and_save_clusters")
                    if cluster_func is not None:
                        try:
                            _capture_output(cluster_func, output_dir)
                            _log("  ✓ analyze_and_save_clusters")
                        except Exception as ve:
                            _log(f"  ⚠️ analyze_and_save_clusters: {ve}")

                    _log("✓ Visualisations générées")
                except Exception as e:
                    _log(f"⚠️ Erreur visualisations : {e}")
            else:
                _log("⚠️ Fichier features_complete.parquet non trouvé, visualisations ignorées")

            log_area.code("\n".join(st.session_state["pipeline_logs"]), language="text")

        # --- Étape 8 : Analyse des catégories ---
        if 7 in selected_steps:
            step_idx = selected_steps.index(7)
            progress_bar.progress((step_idx) / total_steps)
            status_area.info("📊 Analyse des catégories de molécules...")
            _log("=== Analyse des catégories ===")

            output_dir = Config.PATHS.OUTPUT_DIR
            cat_func = imports.get("analyze_categories")
            if cat_func is not None:
                try:
                    _capture_output(cat_func, output_dir)
                    _log("✓ Analyse des catégories terminée")
                except Exception as e:
                    _log(f"⚠️ Erreur analyse catégories : {e}")
            else:
                _log("⚠️ Fonction analyze_categories non disponible")

            log_area.code("\n".join(st.session_state["pipeline_logs"]), language="text")

        progress_bar.progress(1.0)
        status_area.success("✅ Pipeline terminée !")
        _log("\n=== Pipeline terminée ===")
        log_area.code("\n".join(st.session_state["pipeline_logs"]), language="text")

    except Exception as e:
        status_area.error(f"❌ Erreur : {e}")
        _log(f"❌ ERREUR : {e}")
        _log(traceback.format_exc())
        log_area.code("\n".join(st.session_state["pipeline_logs"]), language="text")

    finally:
        st.session_state["pipeline_running"] = False
        st.session_state["pipeline_results"] = results


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
    for i, step_name in enumerate(PIPELINE_STEPS):
        with cols[i % 2]:
            if st.checkbox(step_name, value=True, key=f"step_{i}"):
                selected.append(i)

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
