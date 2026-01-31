"""Test module for the explore() functionality of InstanceSpace.

This module tests the core framework for applying trained instance space models
to new test data (exploreIS pipeline).
"""

from datetime import datetime
from pathlib import Path
from typing import Self

import numpy as np
import pytest

from instancespace.data.metadata import Metadata, from_csv_file
from instancespace.data.model import ExploreResult
from instancespace.data.options import InstanceSpaceOptions, from_json_file
from instancespace.instance_space import InstanceSpace


script_dir = Path(__file__).parent


class TestExploreFramework:
    """Test suite for the explore() framework and core functionality."""

    @pytest.fixture(scope="class")
    def trained_instance_space(self: Self) -> InstanceSpace:
        """Create a trained InstanceSpace by loading data and running build().

        Returns
        -------
            InstanceSpace: A fully trained instance space model.
        """
        metadata_path = script_dir / "test_data/load_file/metadata.csv"
        options_path = script_dir / "test_data/load_file/options.json"

        metadata = from_csv_file(metadata_path)
        assert metadata is not None, "Failed to load metadata"

        options = from_json_file(options_path)
        assert options is not None, "Failed to load options"

        instance_space = InstanceSpace(metadata, options)
        instance_space.build()

        return instance_space

    @pytest.fixture(scope="class")
    def test_metadata(self: Self) -> Metadata:
        """Load test metadata (using same data as training for simplicity).

        In real usage, this would be new test data with the same features.

        Returns
        -------
            Metadata: Test metadata for explore().
        """
        metadata_path = script_dir / "test_data/load_file/metadata.csv"
        metadata = from_csv_file(metadata_path)
        assert metadata is not None, "Failed to load test metadata"
        return metadata

    @pytest.fixture()
    def untrained_instance_space(self: Self) -> InstanceSpace:
        """Create an InstanceSpace without calling build().

        Returns
        -------
            InstanceSpace: An untrained instance space.
        """
        metadata_path = script_dir / "test_data/load_file/metadata.csv"
        options_path = script_dir / "test_data/load_file/options.json"

        metadata = from_csv_file(metadata_path)
        assert metadata is not None, "Failed to load metadata"

        options = from_json_file(options_path)
        assert options is not None, "Failed to load options"

        return InstanceSpace(metadata, options)

    def test_explore_returns_result(
        self: Self,
        trained_instance_space: InstanceSpace,
        test_metadata: Metadata,
    ) -> None:
        """Test that explore() returns an ExploreResult object."""
        result = trained_instance_space.explore(test_metadata, dataset_id="test_1")

        assert result is not None
        assert isinstance(result, ExploreResult)

    def test_explore_dataset_id_provided(
        self: Self,
        trained_instance_space: InstanceSpace,
        test_metadata: Metadata,
    ) -> None:
        """Test that provided dataset_id is stored in result."""
        result = trained_instance_space.explore(test_metadata, dataset_id="my_test_set")

        assert result.dataset_id == "my_test_set"

    def test_explore_dataset_id_generated(
        self: Self,
        trained_instance_space: InstanceSpace,
        test_metadata: Metadata,
    ) -> None:
        """Test that dataset_id is auto-generated when not provided."""
        result = trained_instance_space.explore(test_metadata)

        assert result.dataset_id is not None
        assert result.dataset_id.startswith("explore_")

    def test_explore_timestamp_recorded(
        self: Self,
        trained_instance_space: InstanceSpace,
        test_metadata: Metadata,
    ) -> None:
        """Test that timestamp is recorded in result."""
        before = datetime.now()
        result = trained_instance_space.explore(test_metadata, dataset_id="test_1")
        after = datetime.now()

        assert result.timestamp is not None
        assert before <= result.timestamp <= after

    def test_explore_results_stored(
        self: Self,
        trained_instance_space: InstanceSpace,
        test_metadata: Metadata,
    ) -> None:
        """Test that explore results are stored in instance space."""
        initial_count = len(trained_instance_space.explore_results)

        trained_instance_space.explore(test_metadata, dataset_id="stored_test")
        assert len(trained_instance_space.explore_results) == initial_count + 1

    def test_explore_multiple_datasets(
        self: Self,
        trained_instance_space: InstanceSpace,
        test_metadata: Metadata,
    ) -> None:
        """Test that multiple explore calls accumulate results."""
        initial_count = len(trained_instance_space.explore_results)

        result1 = trained_instance_space.explore(test_metadata, dataset_id="multi_batch_1")
        result2 = trained_instance_space.explore(test_metadata, dataset_id="multi_batch_2")
        result3 = trained_instance_space.explore(test_metadata, dataset_id="multi_batch_3")

        assert len(trained_instance_space.explore_results) == initial_count + 3
        assert trained_instance_space.explore_results[-3].dataset_id == "multi_batch_1"
        assert trained_instance_space.explore_results[-2].dataset_id == "multi_batch_2"
        assert trained_instance_space.explore_results[-1].dataset_id == "multi_batch_3"

        # Verify returned results match stored results
        assert result1 is trained_instance_space.explore_results[-3]
        assert result2 is trained_instance_space.explore_results[-2]
        assert result3 is trained_instance_space.explore_results[-1]

    def test_explore_requires_training(
        self: Self,
        untrained_instance_space: InstanceSpace,
        test_metadata: Metadata,
    ) -> None:
        """Test that explore() raises error when build() not called."""
        with pytest.raises(RuntimeError, match="Must call build"):
            untrained_instance_space.explore(test_metadata)

    def test_explore_result_has_instance_labels(
        self: Self,
        trained_instance_space: InstanceSpace,
        test_metadata: Metadata,
    ) -> None:
        """Test that result contains instance labels from test metadata."""
        result = trained_instance_space.explore(test_metadata, dataset_id="test_1")

        assert result.inst_labels is not None
        assert len(result.inst_labels) == len(test_metadata.instance_labels)

    def test_explore_result_has_feature_matrix(
        self: Self,
        trained_instance_space: InstanceSpace,
        test_metadata: Metadata,
    ) -> None:
        """Test that result contains processed feature matrix."""
        result = trained_instance_space.explore(test_metadata, dataset_id="test_1")

        assert result.x is not None
        assert isinstance(result.x, np.ndarray)
        assert result.x.ndim == 2
        # First dimension should match number of instances
        assert result.x.shape[0] == len(test_metadata.instance_labels)

    def test_explore_result_has_coordinates(
        self: Self,
        trained_instance_space: InstanceSpace,
        test_metadata: Metadata,
    ) -> None:
        """Test that result contains 2D projected coordinates."""
        result = trained_instance_space.explore(test_metadata, dataset_id="test_1")

        assert result.z is not None
        assert isinstance(result.z, np.ndarray)
        assert result.z.shape == (len(test_metadata.instance_labels), 2)

    def test_explore_result_placeholder_pythia(
        self: Self,
        trained_instance_space: InstanceSpace,
        test_metadata: Metadata,
    ) -> None:
        """Test that PYTHIA outputs are None (placeholder implementation)."""
        result = trained_instance_space.explore(test_metadata, dataset_id="test_1")

        # Placeholder implementation returns None
        assert result.y_hat is None
        assert result.pr0_hat is None
        assert result.selection0 is None

    def test_explore_result_placeholder_trace(
        self: Self,
        trained_instance_space: InstanceSpace,
        test_metadata: Metadata,
    ) -> None:
        """Test that TRACE outputs are None (placeholder implementation)."""
        result = trained_instance_space.explore(test_metadata, dataset_id="test_1")

        # Placeholder implementation returns None
        assert result.in_good is None
        assert result.in_best is None
        assert result.in_space is None


class TestExploreValidation:
    """Test suite for explore() input validation."""

    @pytest.fixture(scope="class")
    def trained_instance_space(self: Self) -> InstanceSpace:
        """Create a trained InstanceSpace."""
        metadata_path = script_dir / "test_data/load_file/metadata.csv"
        options_path = script_dir / "test_data/load_file/options.json"

        metadata = from_csv_file(metadata_path)
        assert metadata is not None
        options = from_json_file(options_path)
        assert options is not None

        instance_space = InstanceSpace(metadata, options)
        instance_space.build()
        return instance_space

    @pytest.fixture()
    def metadata_missing_features(self: Self) -> Metadata:
        """Create metadata with missing features.

        This simulates test data that doesn't have all required features.
        """
        # Load valid metadata first
        metadata_path = script_dir / "test_data/load_file/metadata.csv"
        original = from_csv_file(metadata_path)
        assert original is not None

        # Create new metadata with fewer features (drop half of them)
        n_features_to_keep = len(original.feature_names) // 2
        reduced_features = original.features[:, :n_features_to_keep]
        reduced_feature_names = original.feature_names[:n_features_to_keep]

        return Metadata(
            feature_names=reduced_feature_names,
            algorithm_names=original.algorithm_names,
            instance_labels=original.instance_labels,
            instance_sources=original.instance_sources,
            features=reduced_features,
            algorithms=original.algorithms,
        )

    def test_explore_validates_features(
        self: Self,
        trained_instance_space: InstanceSpace,
        metadata_missing_features: Metadata,
    ) -> None:
        """Test that explore() raises error for missing features."""
        with pytest.raises(ValueError, match="missing features"):
            trained_instance_space.explore(metadata_missing_features)


class TestExploreResultDataclass:
    """Test suite for ExploreResult dataclass properties."""

    def test_explore_result_is_frozen(self: Self) -> None:
        """Test that ExploreResult is immutable (frozen dataclass)."""
        result = ExploreResult(
            dataset_id="test",
            timestamp=datetime.now(),
            x=np.array([[1.0, 2.0]]),
            z=np.array([[0.1, 0.2]]),
            y_hat=None,
            pr0_hat=None,
            selection0=None,
            in_good=None,
            in_best=None,
            in_space=None,
            inst_labels=None,  # type: ignore[arg-type]
        )

        with pytest.raises(AttributeError):
            result.dataset_id = "modified"  # type: ignore[misc]

    def test_explore_result_all_fields(self: Self) -> None:
        """Test that ExploreResult can be constructed with all fields."""
        import pandas as pd

        result = ExploreResult(
            dataset_id="full_test",
            timestamp=datetime.now(),
            x=np.array([[1.0, 2.0, 3.0]]),
            z=np.array([[0.1, 0.2]]),
            y_hat=np.array([[True, False]]),
            pr0_hat=np.array([[0.8, 0.2]]),
            selection0=np.array([0]),
            in_good=np.array([[True, False]]),
            in_best=np.array([[True, False]]),
            in_space=np.array([True]),
            inst_labels=pd.Series(["instance_1"]),
        )

        assert result.dataset_id == "full_test"
        assert result.x.shape == (1, 3)
        assert result.z.shape == (1, 2)
        assert result.y_hat is not None
        assert result.pr0_hat is not None
        assert result.selection0 is not None
        assert result.in_good is not None
        assert result.in_best is not None
        assert result.in_space is not None
