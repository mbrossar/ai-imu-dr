import pytest
import tempfile
import shutil
import os
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file in the temp directory."""
    def _create_temp_file(name="test_file.txt", content=""):
        file_path = temp_dir / name
        file_path.write_text(content)
        return file_path
    return _create_temp_file


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = MagicMock()
    config.data_path = "/mock/data/path"
    config.output_path = "/mock/output/path"
    config.batch_size = 32
    config.learning_rate = 0.001
    config.epochs = 10
    return config


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    import numpy as np
    
    return {
        'features': np.random.randn(100, 10),
        'labels': np.random.randint(0, 2, size=(100,)),
        'timestamps': np.arange(100),
    }


@pytest.fixture
def mock_dataset():
    """Create a mock dataset object."""
    dataset = Mock()
    dataset.data = []
    dataset.labels = []
    dataset.length = 0
    
    def add_sample(data, label):
        dataset.data.append(data)
        dataset.labels.append(label)
        dataset.length += 1
    
    dataset.add_sample = add_sample
    dataset.__len__ = lambda self: dataset.length
    dataset.__getitem__ = lambda self, idx: (dataset.data[idx], dataset.labels[idx])
    
    return dataset


@pytest.fixture
def numpy_array_fixture():
    """Provide common numpy arrays for testing."""
    import numpy as np
    
    return {
        'vector': np.array([1, 2, 3, 4, 5]),
        'matrix': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        'zeros': np.zeros((3, 3)),
        'ones': np.ones((3, 3)),
        'random': np.random.randn(10, 10),
    }


@pytest.fixture
def torch_tensor_fixture():
    """Provide common torch tensors for testing (if torch is available)."""
    try:
        import torch
        
        return {
            'vector': torch.tensor([1, 2, 3, 4, 5]),
            'matrix': torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            'zeros': torch.zeros(3, 3),
            'ones': torch.ones(3, 3),
            'random': torch.randn(10, 10),
        }
    except ImportError:
        pytest.skip("PyTorch not installed")


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    import random
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    
    try:
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    except ImportError:
        pass


@pytest.fixture
def capture_stdout():
    """Capture stdout for testing print statements."""
    from io import StringIO
    import sys
    
    captured_output = StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured_output
    
    yield captured_output
    
    sys.stdout = old_stdout


@pytest.fixture
def mock_file_operations(mocker):
    """Mock common file operations."""
    mock_open = mocker.mock_open()
    mocker.patch('builtins.open', mock_open)
    
    mock_exists = mocker.patch('os.path.exists')
    mock_exists.return_value = True
    
    mock_makedirs = mocker.patch('os.makedirs')
    
    return {
        'open': mock_open,
        'exists': mock_exists,
        'makedirs': mock_makedirs,
    }


@pytest.fixture
def time_tracker():
    """Track execution time for performance tests."""
    import time
    
    class TimeTracker:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.elapsed = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            self.elapsed = self.end_time - self.start_time
            return self.elapsed
    
    return TimeTracker()