"""Tests for the Zarr-backed RasterStore."""

import numpy as np
import pytest
from bike_sim.state.raster_store import RasterStore


@pytest.fixture
def store(tmp_path):
    """Create a fresh RasterStore in a temporary directory."""
    return RasterStore.create(tmp_path / "rasters.zarr")


class TestWriteAndRead:
    """Write a layer, read it back — arrays match exactly."""

    def test_round_trip_float64(self, store):
        data = np.random.default_rng(0).random((512, 512))
        store.write_layer("geology", "heightmap", data, tick_number=1)
        result = store.read_layer("geology", "heightmap")
        np.testing.assert_array_equal(result, data)

    def test_round_trip_int32(self, store):
        data = np.random.default_rng(1).integers(0, 20, size=(512, 512), dtype=np.int32)
        store.write_layer("geology", "bedrock_type", data, tick_number=1)
        result = store.read_layer("geology", "bedrock_type")
        np.testing.assert_array_equal(result, data)
        assert result.dtype == np.int32


class TestTierIndependence:
    """Layers in different tiers are independent."""

    def test_same_layer_name_different_tiers(self, store):
        geo_data = np.ones((256, 256), dtype=np.float64) * 100.0
        eco_data = np.ones((256, 256), dtype=np.float64) * 0.5

        store.write_layer("geology", "heightmap", geo_data, tick_number=1)
        store.write_layer("ecology", "heightmap", eco_data, tick_number=1)

        np.testing.assert_array_equal(store.read_layer("geology", "heightmap"), geo_data)
        np.testing.assert_array_equal(store.read_layer("ecology", "heightmap"), eco_data)

    def test_different_tiers_different_layers(self, store):
        geo_data = np.zeros((128, 128), dtype=np.float64)
        eco_data = np.full((128, 128), 42.0, dtype=np.float64)

        store.write_layer("geology", "heightmap", geo_data, tick_number=1)
        store.write_layer("ecology", "density", eco_data, tick_number=5)

        np.testing.assert_array_equal(store.read_layer("geology", "heightmap"), geo_data)
        np.testing.assert_array_equal(store.read_layer("ecology", "density"), eco_data)


class TestListLayers:
    """list_layers returns correct names per tier."""

    def test_empty_tier(self, store):
        assert store.list_layers("geology") == []

    def test_single_layer(self, store):
        store.write_layer("geology", "heightmap", np.zeros((64, 64)), tick_number=0)
        assert store.list_layers("geology") == ["heightmap"]

    def test_multiple_layers_one_tier(self, store):
        store.write_layer("geology", "heightmap", np.zeros((64, 64)), tick_number=0)
        bedrock = np.zeros((64, 64), dtype=np.int32)
        store.write_layer("geology", "bedrock_type", bedrock, tick_number=0)
        result = sorted(store.list_layers("geology"))
        assert result == ["bedrock_type", "heightmap"]

    def test_layers_scoped_to_tier(self, store):
        store.write_layer("geology", "heightmap", np.zeros((64, 64)), tick_number=0)
        store.write_layer("ecology", "density", np.zeros((64, 64)), tick_number=0)

        assert store.list_layers("geology") == ["heightmap"]
        assert store.list_layers("ecology") == ["density"]


class TestOverwrite:
    """Overwriting a layer with a new tick replaces the data."""

    def test_overwrite_replaces_data(self, store):
        original = np.ones((256, 256), dtype=np.float64) * 10.0
        updated = np.ones((256, 256), dtype=np.float64) * 99.0

        store.write_layer("geology", "heightmap", original, tick_number=1)
        store.write_layer("geology", "heightmap", updated, tick_number=2)

        result = store.read_layer("geology", "heightmap")
        np.testing.assert_array_equal(result, updated)

    def test_overwrite_does_not_duplicate_in_list(self, store):
        store.write_layer("geology", "heightmap", np.zeros((64, 64)), tick_number=1)
        store.write_layer("geology", "heightmap", np.ones((64, 64)), tick_number=2)
        assert store.list_layers("geology") == ["heightmap"]


class TestDtypePreservation:
    """Different dtypes are preserved through write/read."""

    @pytest.mark.parametrize(
        "dtype",
        [np.float32, np.float64, np.int32, np.int64, np.uint8, np.uint16],
        ids=["float32", "float64", "int32", "int64", "uint8", "uint16"],
    )
    def test_dtype_preserved(self, store, dtype):
        rng = np.random.default_rng(42)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            data = rng.integers(info.min, info.max, size=(128, 128), dtype=dtype)
        else:
            data = rng.random((128, 128)).astype(dtype)

        store.write_layer("geology", "test_layer", data, tick_number=0)
        result = store.read_layer("geology", "test_layer")

        assert result.dtype == dtype
        np.testing.assert_array_equal(result, data)


class TestOpenExisting:
    """RasterStore.open can reopen a previously created store."""

    def test_open_reads_previously_written_data(self, tmp_path):
        path = tmp_path / "rasters.zarr"
        data = np.arange(100, dtype=np.float64).reshape(10, 10)

        store = RasterStore.create(path)
        store.write_layer("geology", "heightmap", data, tick_number=3)

        reopened = RasterStore.open(path)
        result = reopened.read_layer("geology", "heightmap")
        np.testing.assert_array_equal(result, data)
