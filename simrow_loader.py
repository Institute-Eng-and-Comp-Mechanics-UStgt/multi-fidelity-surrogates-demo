import numpy as np
from scipy import io as sio
from typing import Dict, List, Literal, Optional, Tuple, Callable
from dataclasses import dataclass
import os
from enum import Enum
import logging
from torch.utils.data import Dataset

from sklearn.discriminant_analysis import StandardScaler

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    train_ratio: float = 0.6
    val_ratio: float = 0.2

    def __post_init__(self):
        logger.debug(
            f"Initializing SplitConfig with train_ratio={self.train_ratio}, val_ratio={self.val_ratio}"
        )
        if not 0 <= self.train_ratio <= 1:
            logger.error(f"Invalid train_ratio: {self.train_ratio}")
            raise ValueError("Train ratio must be between 0 and 1")
        if not 0 <= self.val_ratio <= 1:
            logger.error(f"Invalid val_ratio: {self.val_ratio}")
            raise ValueError("Validation ratio must be between 0 and 1")
        if self.train_ratio + self.val_ratio >= 1.0:
            logger.error(
                f"Invalid split configuration: train_ratio={self.train_ratio}, val_ratio={self.val_ratio}"
            )
            raise ValueError("Train and validation ratios must sum to less than 1.0")
        self.test_ratio = 1.0 - (self.train_ratio + self.val_ratio)
        logger.info(
            f"Aimed split ratios: train={self.train_ratio:.3f}, val={self.val_ratio:.3f}, test={self.test_ratio:.3f}"
        )


@dataclass
class Split:
    """Data structure to hold train/val/test split information"""

    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray

    def __post_init__(self):
        logger.debug(
            f"Created Split with sizes: train={len(self.train_indices)}, "
            f"val={len(self.val_indices)}, test={len(self.test_indices)}"
        )

    def __add__(self, other: "Split") -> "Split":
        """Combine two splits by concatenating their respective indices

        Args:
            other: Another Split object to combine with this one

        Returns:
            A new Split object containing the combined indices
        """
        logger.debug("Combining two Split objects")
        return Split(
            train_indices=np.concatenate([self.train_indices, other.train_indices]),
            val_indices=np.concatenate([self.val_indices, other.val_indices]),
            test_indices=np.concatenate([self.test_indices, other.test_indices]),
        )

    @property
    def sizes(self) -> Tuple[int, int, int]:
        """Get the sizes of each split

        Returns:
            Tuple of (train_size, val_size, test_size)
        """
        return (len(self.train_indices), len(self.val_indices), len(self.test_indices))


class SimSplitter:
    """Helper class to manage simulation data splitting"""

    def __init__(self, sim_count: int, samples_per_sim: int, total_sample_count: int):
        logger.info(
            f"Initializing SimSplitter with {sim_count} simulations, "
            f"{samples_per_sim} samples per simulation, "
            f"{total_sample_count} total samples"
        )
        self.sim_count = sim_count
        self.samples_per_sim = samples_per_sim
        self.total_sample_count = total_sample_count

    def split_data(self, config: SplitConfig) -> Split:
        """Create train/val/test split of simulation and preliminary data"""
        # Split simulation indices
        sim_split = self._split_simulation_indices(config)

        # Split preliminary data indices
        prelim_split = self._split_preliminary_indices(config)

        combined_split = sim_split + prelim_split
        logger.info(
            f"Completed data split with sizes: {combined_split.sizes} ({', '.join('{0:.3f}'.format(size/self.total_sample_count) for size in combined_split.sizes)})"
        )
        return combined_split

    def _split_simulation_indices(self, config: SplitConfig) -> Split:
        """Split the simulation indices into train/val/test"""
        logger.debug("Splitting simulation indices")
        sim_indices = np.random.permutation(self.sim_count)

        # Calculate split sizes
        train_val_size = int(self.sim_count * (config.train_ratio + config.val_ratio))

        # Split indices
        train_val_sim_indices = sim_indices[:train_val_size]
        test_sim_indices = sim_indices[train_val_size:]

        logger.debug(
            f"Simulation split sizes: train+val={len(train_val_sim_indices)}, test={len(test_sim_indices)}"
        )

        train_val_indices = self._get_sample_indices_for_sims(train_val_sim_indices)
        train_val_indices = np.random.permutation(train_val_indices)
        # Calculate split size for train
        train_size = int(
            len(train_val_indices)
            * config.train_ratio
            / (config.train_ratio + config.val_ratio)
        )

        train_indices = train_val_indices[:train_size]
        val_indices = train_val_indices[train_size:]

        test_indices = self._get_sample_indices_for_sims(test_sim_indices)

        return Split(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        )

    def _split_preliminary_indices(self, config: SplitConfig) -> Split:
        """Split the preliminary data indices into train/val/test"""

        logger.debug("Splitting preliminary indices")
        sim_samples = self.sim_count * self.samples_per_sim
        prelim_indices = np.random.permutation(
            np.arange(sim_samples, self.total_sample_count)
        )

        # Calculate split sizes
        prelim_count = len(prelim_indices)
        train_size = int(prelim_count * config.train_ratio)
        val_size = int(prelim_count * config.val_ratio)

        logger.debug(f"Preliminary data count: {prelim_count}")
        return Split(
            train_indices=prelim_indices[:train_size],
            val_indices=prelim_indices[train_size : train_size + val_size],
            test_indices=prelim_indices[train_size + val_size :],
        )

    def _get_sample_indices_for_sims(self, sim_indices: np.ndarray) -> np.ndarray:
        """Convert simulation indices to sample indices"""
        if len(sim_indices) == 0:
            return np.array([], dtype=int)

        # Create sample indices for each simulation
        indices_list = [
            np.arange(
                sim_idx * self.samples_per_sim, (sim_idx + 1) * self.samples_per_sim
            )
            for sim_idx in sim_indices
        ]

        return np.concatenate(indices_list)

    @staticmethod
    def _combine_indices(*index_arrays: np.ndarray) -> np.ndarray:
        """Combine multiple index arrays"""
        return np.concatenate(index_arrays) if index_arrays else np.array([], dtype=int)


class ColumnType(Enum):
    Feature = "feature"
    Target = "target"


class SimrowDataset(Dataset):
    def __init__(
        self,
        feature_data: np.ndarray,
        target_data: np.ndarray,
        hf_flag: Optional[np.ndarray] = None,
        split: Optional[str] = None,
    ):
        self.feature_data = feature_data
        self.target_data = target_data
        self.hf_flag = hf_flag
        self.split = split

    def __len__(self):
        return len(self.feature_data)

    def __getitem__(self, idx):
        return self.feature_data[idx], self.target_data[idx]

    def __repr__(self):
        return f"SimrowDataset with {len(self)} samples and split={self.split}"

    def get_feature_dim(self):
        return self.feature_data.shape[1]

    def get_target_dim(self):
        return self.target_data.shape[1]


class SimrowLoader:
    def __init__(self, data_path: str, dtype: Optional[np.dtype] = np.float32):
        """Initialize SimrowLoader with path to .mat file

        Args:
            data_path: Path to the .mat file containing simulation data
        """

        logger.info(f"Initializing SimrowLoader with data path: {data_path}")
        if not os.path.isfile(data_path):
            logger.error(f"Data path does not exist: {data_path}")
            raise ValueError(f"Data path does not exist: {data_path}")

        self.data_path = data_path
        self.dtype = dtype
        self.raw_data: Dict[str, np.ndarray] = {}
        self.data: Dict[str, np.ndarray] = {}
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []
        self.transforms: List[Callable] = []
        self.split: Optional[Split] = None
        self.noise = None

        self._load_data()

    def __len__(self):
        return self.total_sample_count

    def _load_data(self):
        """Load data from .mat file and perform initial processing"""

        logger.info(f"Loading data from {self.data_path}")
        try:
            raw_data = sio.loadmat(self.data_path)
            logger.debug("Successfully loaded .mat file")
        except Exception as e:
            logger.error(f"Failed to load .mat file: {self.data_path}")
            raise IOError(f"Error loading data: {str(e)}")

        self.sim_count = int(raw_data["sim_count"][0, 0])
        self.samples_per_sim = int(raw_data["samples_per_sim"][0, 0])
        logger.info(
            f"Found {self.sim_count} simulations with {self.samples_per_sim} samples each"
        )

        # Remove metadata fields
        metadata_fields = [
            "__header__",
            "__version__",
            "__globals__",
            "sim_count",
            "samples_per_sim",
        ]
        for field in metadata_fields:
            raw_data.pop(field, None)

        # Convert to right dtype and validate
        for key in raw_data:
            if isinstance(raw_data[key], np.ndarray):
                raw_data[key] = raw_data[key].astype(self.dtype)
                if np.any(np.isnan(raw_data[key])) or np.any(np.isinf(raw_data[key])):
                    logger.warning(f"Found NaN or Inf values in column {key}")

        self.total_sample_count = len(next(iter(raw_data.values())))

        self.raw_data = raw_data
        logger.info(
            f"Loaded {len(raw_data)} columns with {self.total_sample_count} samples each"
        )

    def add_columns(self, column_type: ColumnType, *names: List[str]):
        """Add one or several new columns from raw data to the dataset

        Args:
            column_type: Type of column to add (feature or target)
            names: Column names to add
        """

        logger.info(f"Adding {len(names)} columns of type {column_type.value}")
        for name in names:
            if name in self.data:
                logger.error(f"Column {name} already exists in data")
                raise ValueError(f"Column {name} already exists in data")
            if name not in self.raw_data:
                logger.error(f"Column {name} not found in raw data")
                raise ValueError(f"Column {name} not found in raw data")

            self.data[name] = self.raw_data[name]
            logger.debug(f"Added column {name} as {column_type.value}")

            match column_type:
                case ColumnType.Feature:
                    self.feature_columns.append(name)
                case ColumnType.Target:
                    self.target_columns.append(name)
                case _:
                    logger.error(f"Invalid column type: {column_type}")
                    raise ValueError("Invalid column type")

    def remove_columns(self, column_type: ColumnType, *names: List[str]):
        """Remove one or several columns from the dataset

        Args:
            column_type: Type of column to remove (feature or target)
            names: Column names to remove
        """
        logger.info(f"Removing {len(names)} columns of type {column_type.value}")
        for name in names:
            if name not in self.data:
                logger.error(f"Column {name} not found in data")
                raise ValueError(f"Column {name} not found in data")

            self.data.pop(name)
            logger.debug(f"Removed column {name} as {column_type.value}")

            match column_type:
                case ColumnType.Feature:
                    self.feature_columns.remove(name)
                case ColumnType.Target:
                    self.target_columns.remove(name)
                case _:
                    logger.error(f"Invalid column type: {column_type}")
                    raise ValueError("Invalid column type")

    def add_calculated_column(
        self, column_type: ColumnType, name: str, calculation_fn: Callable
    ):
        """Add a new calculated column to the dataset

        Args:
            column_type: Type of column to add (feature or target)
            name: Name of the new column
            calculation_fn: Function that takes the raw data dict and returns np.ndarray
        """
        if name in self.data:
            raise ValueError(f"Column {name} already exists in data")

        logger.info(f"Adding calculated column {name} of type {column_type.value}")

        # Verify the function works and returns proper shape
        try:
            result = calculation_fn(self.data)
            if not isinstance(result, np.ndarray):
                raise ValueError("Calculation function must return numpy array")
            if result.shape[0] != self.total_sample_count:
                raise ValueError(
                    "Calculation function must return array with same length as samples"
                )
            # ensure the result is 2D
            if len(result.shape) == 1:
                result = result.reshape(-1, 1)

            # ensure correct data type
            result = result.astype(self.dtype)

            self.data[name] = result

            match column_type:
                case ColumnType.Feature:
                    self.feature_columns.append(name)
                case ColumnType.Target:
                    self.target_columns.append(name)
                case _:
                    raise ValueError("Invalid column type")

        except Exception as e:
            raise ValueError(f"Error in calculation function: {str(e)}")

    def add_transform(self, transform_fn: Callable):
        """Add a transform to be applied to the data

        Args:
            transform_fn: Function that takes a dict of arrays and returns transformed dict
        """
        self.transforms.append(transform_fn)

    def create_split(self, train_ratio: float = 0.6, val_ratio: float = 0.2):
        """Create train/val/test split of the data

        Args:
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            (remaining data will be used for testing)
        """
        config = SplitConfig(train_ratio=train_ratio, val_ratio=val_ratio)
        splitter = SimSplitter(
            self.sim_count, self.samples_per_sim, self.total_sample_count
        )
        self.split = splitter.split_data(config)

    def add_additive_noise(
        self,
        split: Literal["train", "val", "test"],
        column_type: ColumnType,
        noise_std: float,
        noise_mean: float = 0.0,
    ):
        """Add additive Gaussian noise to the data

        Args:
            split: Dataset split to add noise to
            noise_std: Standard deviation of the Gaussian noise
            noise_mean: Mean of the Gaussian noise
        """
        logger.info(
            f"Adding noise to {column_type.value} data for split {split} with std={noise_std}, mean={noise_mean}"
        )

        if self.split is None:
            logger.error("Attempted to add noise but no split has been created")
            raise ValueError("No split has been created")

        feature_data_shape = np.concatenate(
            [self.data[col] for col in self.feature_columns], axis=1
        ).shape
        target_data_shape = np.concatenate(
            [self.data[col] for col in self.target_columns], axis=1
        ).shape

        if self.noise is None:
            self.noise = np.zeros(feature_data_shape, dtype=self.dtype), np.zeros(
                target_data_shape, dtype=self.dtype
            )

        match split:
            case "train":
                indices = self.split.train_indices
            case "val":
                indices = self.split.val_indices
            case "test":
                indices = self.split.test_indices
            case _:
                logger.error(f"Invalid split type: {split}")
                raise ValueError(f"Invalid split type: {split}")

        if column_type == ColumnType.Feature:
            self.noise[0][indices] = np.random.normal(
                loc=noise_mean, scale=noise_std, size=feature_data_shape
            )[indices]
        elif column_type == ColumnType.Target:
            self.noise[1][indices] = np.random.normal(
                loc=noise_mean, scale=noise_std, size=target_data_shape
            )[indices]

    def apply_additive_noise(
        self, feature_data: np.ndarray, target_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply noise to the data

        Args:
            feature_data: Feature data to add noise to
            target_data: Target data to add noise to

        Returns:
            Tuple of (feature_data, target_data) with noise added
        """
        if self.noise is None:
            return feature_data, target_data

        feature_data_noise, target_data_noise = self.noise

        return feature_data + feature_data_noise, target_data + target_data_noise

    def get_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Get the processed data for training, validation, and testing as PyTorch Datasets

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if self.split is None:
            logger.error("Attempted to get split data but no split has been created")
            raise ValueError("No split has been created")

        train_dataset = self.get_dataset(split="train")
        val_dataset = self.get_dataset(split="val")
        test_dataset = self.get_dataset(split="test")

        return train_dataset, val_dataset, test_dataset

    def get_dataset(self, split: Literal["train", "val", "test"]) -> Dataset:
        """Get the processed data for a specific split as a PyTorch Dataset

        Args:
            split: Dataset split to return

        Returns:
            PyTorch Dataset object
        """
        feature_data, target_data, hf_flag = self.get_data(split=split)
        return SimrowDataset(feature_data, target_data, hf_flag, split=split)

    def get_data(
        self,
        split: Optional[Literal["train", "val", "test"]] = None,
        skip_transforms: bool = False,
        skip_noise: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the processed feature and target data.

        Args:
            split: Dataset split to return. If None, returns all data.
            skip_transforms: If True, skips applying transformations.
            skip_noise: If True, skips adding noise.

        Returns:
            Tuple of (feature_data, target_data, hf_flag) as numpy arrays
        """
        # Start with raw data as a single array

        logger.debug(
            f"Getting data for split={split}, skip_transforms={skip_transforms}"
        )

        feature_data = np.concatenate(
            [self.data[col] for col in self.feature_columns], axis=1
        )
        target_data = np.concatenate(
            [self.data[col] for col in self.target_columns], axis=1
        )

        logger.debug(
            f"Initial data shapes: features={feature_data.shape}, targets={target_data.shape}"
        )

        if not skip_transforms:
            for i, transform in enumerate(self.transforms):
                logger.debug(f"Applying transform {i+1}/{len(self.transforms)}")
                feature_data, target_data = transform(feature_data, target_data)

        if not skip_noise:
            feature_data, target_data = self.apply_additive_noise(
                feature_data, target_data
            )

        indices = range(len(feature_data))
        if split is not None:
            if self.split is None:
                logger.error(
                    "Attempted to get split data but no split has been created"
                )
                raise ValueError("No split has been created")

            match split:
                case "train":
                    indices = self.split.train_indices
                case "val":
                    indices = self.split.val_indices
                case "test":
                    indices = self.split.test_indices
                case _:
                    logger.error(f"Invalid split type: {split}")
                    raise ValueError(f"Invalid split type: {split}")

        feature_data, target_data = feature_data[indices], target_data[indices]
        logger.debug(
            f"Data shapes for split={split}: features={feature_data.shape}, targets={target_data.shape}"
        )
        hf_flag = indices < self.sim_count * self.samples_per_sim

        return feature_data, target_data, hf_flag

    def get_raw_data_column_names(self) -> List[str]:
        """Get all available column names in the raw data

        Returns:
            List of column names as strings
        """
        return list(self.raw_data.keys())

    def get_column_names(self) -> List[str]:
        """Get all available column names in the dataset

        Returns:
            List of column names as strings
        """
        return list(self.data.keys())


if __name__ == "__main__":
    # configure logging
    logging.basicConfig(level=logging.DEBUG)

    # Create a SimrowLoader object
    data_path = "surrogatetrainer/simrowloader/test/test_data.mat"
    simrow_loader = SimrowLoader(data_path)

    # Get the column names
    column_names = simrow_loader.get_raw_data_column_names()
    print(column_names)

    # Add columns to the dataset
    simrow_loader.add_columns(
        ColumnType.Feature,
        "relativePos_headrestobserver",
        "relativeVel_headrestobserver",
        "relativeOri6d_headrestobserver",
        "relativeOmega_headrestobserver",
    )

    simrow_loader.add_columns(
        ColumnType.Target,
        "force_headrestobserver",
    )

    # Create train/val/test split
    simrow_loader.create_split(train_ratio=0.6, val_ratio=0.2)
    unsc_feature_data, unsc_target_data = simrow_loader.get_data(split="train")

    scaler = StandardScaler()
    scaler.fit(unsc_feature_data)

    # include variance threshold
    scaler.scale_ = np.where(scaler.var_ < 1e-10, 1.0, scaler.scale_)

    def transform_data(
        feature_data: np.ndarray, target_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return scaler.transform(feature_data), target_data

    simrow_loader.add_transform(transform_data)

    # Get the processed data for training
    feature_data, target_data = simrow_loader.get_data(split="train")
    print("Feature data shape", feature_data.shape)
    print("Target data shape", target_data.shape)
