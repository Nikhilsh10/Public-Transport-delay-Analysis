"""Unit tests for src/config.py — configuration constants and paths."""

import pathlib
import src.config as cfg


class TestConfigConstants:
    """Verify that all expected configuration attributes are defined."""

    def test_base_dir_is_pathlib_path(self):
        assert isinstance(cfg.BASE_DIR, pathlib.Path)

    def test_data_path_attribute_exists(self):
        assert hasattr(cfg, "DATA_PATH")
        assert isinstance(cfg.DATA_PATH, pathlib.Path)

    def test_model_path_attribute_exists(self):
        assert hasattr(cfg, "MODEL_PATH")
        assert isinstance(cfg.MODEL_PATH, pathlib.Path)

    def test_random_state_is_int(self):
        assert isinstance(cfg.RANDOM_STATE, int)

    def test_n_estimators_is_positive_int(self):
        assert isinstance(cfg.N_ESTIMATORS, int)
        assert cfg.N_ESTIMATORS > 0

    def test_test_size_is_valid_fraction(self):
        assert 0 < cfg.TEST_SIZE < 1

    def test_stratify_is_bool(self):
        assert isinstance(cfg.STRATIFY, bool)

    def test_target_column_is_non_empty_string(self):
        assert isinstance(cfg.TARGET_COLUMN, str)
        assert len(cfg.TARGET_COLUMN) > 0

    def test_model_path_has_joblib_extension(self):
        assert cfg.MODEL_PATH.suffix == ".joblib"

    def test_data_path_ends_with_csv(self):
        assert cfg.DATA_PATH.suffix == ".csv"

    def test_base_dir_is_parent_of_src(self):
        """BASE_DIR should be the project root (parent of the src/ package)."""
        src_dir = pathlib.Path(cfg.__file__).parent
        assert cfg.BASE_DIR == src_dir.parent
