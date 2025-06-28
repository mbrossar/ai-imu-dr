import pytest
import sys
import os
from pathlib import Path


class TestSetupValidation:
    """Validation tests to ensure the testing infrastructure is properly configured."""
    
    def test_project_structure_exists(self):
        """Test that the project structure is correctly set up."""
        project_root = Path(__file__).parent.parent
        
        # Check main directories exist
        assert project_root.exists()
        assert (project_root / "src").exists()
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "integration").exists()
        
        # Check configuration files exist
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / ".gitignore").exists()
    
    def test_src_in_python_path(self):
        """Test that src directory is in Python path."""
        src_path = str(Path(__file__).parent.parent / "src")
        assert any(src_path in path for path in sys.path)
    
    def test_pytest_imports(self):
        """Test that pytest and related packages can be imported."""
        import pytest
        import pytest_cov
        import pytest_mock
        
        assert pytest.__version__
        assert hasattr(pytest_cov, '__version__') or hasattr(pytest_cov, 'version')
        assert pytest_mock is not None
    
    def test_fixtures_available(self, temp_dir, mock_config, sample_data):
        """Test that custom fixtures are available and working."""
        # Test temp_dir fixture
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test mock_config fixture
        assert mock_config.batch_size == 32
        assert mock_config.learning_rate == 0.001
        
        # Test sample_data fixture
        assert 'features' in sample_data
        assert 'labels' in sample_data
        assert len(sample_data['features']) == 100
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit test marker is working."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration test marker is working."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow test marker is working."""
        import time
        time.sleep(0.1)  # Simulate slow test
        assert True
    
    def test_coverage_configuration(self):
        """Test that coverage is properly configured."""
        import coverage
        assert coverage.__version__
    
    def test_temp_file_fixture(self, temp_file):
        """Test that temp_file fixture creates files correctly."""
        test_file = temp_file("test.txt", "Hello, World!")
        assert test_file.exists()
        assert test_file.read_text() == "Hello, World!"
    
    def test_numpy_fixture(self, numpy_array_fixture):
        """Test numpy array fixture."""
        import numpy as np
        
        assert numpy_array_fixture['vector'].shape == (5,)
        assert numpy_array_fixture['matrix'].shape == (3, 3)
        assert np.all(numpy_array_fixture['zeros'] == 0)
        assert np.all(numpy_array_fixture['ones'] == 1)
    
    def test_mock_dataset_fixture(self, mock_dataset):
        """Test mock dataset fixture."""
        mock_dataset.add_sample([1, 2, 3], 0)
        mock_dataset.add_sample([4, 5, 6], 1)
        
        assert len(mock_dataset) == 2
        assert mock_dataset[0] == ([1, 2, 3], 0)
        assert mock_dataset[1] == ([4, 5, 6], 1)
    
    def test_capture_stdout_fixture(self, capsys):
        """Test stdout capture using pytest's built-in capsys fixture."""
        print("Test output")
        print("Another line")
        
        captured = capsys.readouterr()
        assert "Test output" in captured.out
        assert "Another line" in captured.out
    
    def test_time_tracker_fixture(self, time_tracker):
        """Test time tracker fixture."""
        import time
        
        time_tracker.start()
        time.sleep(0.1)
        elapsed = time_tracker.stop()
        
        assert elapsed >= 0.1
        assert elapsed < 0.2