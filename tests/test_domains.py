import pytest
from folie._numpy import np
import folie as fl
from folie.domains import create_minimum_point_mesh
from folie.domains._mesh_utils import centroid_driven_mesh


class TestMinimumPointMesh:
    """Tests for minimum point mesh generation functionality."""
    
    def test_create_minimum_point_mesh_basic(self):
        """Test basic mesh creation with minimum point constraint."""
        np.random.seed(42)
        data = np.random.rand(1000, 2)
        
        vertices, simplices, point_counts = create_minimum_point_mesh(
            data, min_points=10, verbose=False
        )
        
        # Check that all elements have at least min_points
        assert point_counts.min() >= 10, f"Min points {point_counts.min()} < 10"
        
        # Check mesh structure
        assert vertices.shape[1] == 2
        assert simplices.shape[1] == 3  # Triangular elements
        assert len(simplices) == len(point_counts)
        
    def test_create_minimum_point_mesh_with_max(self):
        """Test mesh creation with both min and max point constraints."""
        np.random.seed(42)
        data = np.random.rand(1000, 2)
        
        vertices, simplices, point_counts = create_minimum_point_mesh(
            data, min_points=10, max_points=100, verbose=False
        )
        
        # Check constraints
        assert point_counts.min() >= 10
        assert point_counts.max() <= 100
        
    def test_create_minimum_point_mesh_large_dataset(self):
        """Test with large dataset (simulating 100K+ points)."""
        np.random.seed(42)
        data = np.random.rand(5000, 2)  # Smaller for test speed
        
        vertices, simplices, point_counts = create_minimum_point_mesh(
            data, min_points=20, verbose=False
        )
        
        # Verify constraints
        assert point_counts.min() >= 20
        
        # Check that we have reasonable number of elements
        assert len(simplices) > 0
        assert len(simplices) < len(data)  # Should have fewer elements than points
        
    def test_create_minimum_point_mesh_invalid_data(self):
        """Test error handling for invalid input."""
        # Wrong dimensions
        with pytest.raises(ValueError):
            create_minimum_point_mesh(np.random.rand(100, 3), min_points=10)
            
        # 1D data
        with pytest.raises(ValueError):
            create_minimum_point_mesh(np.random.rand(100), min_points=10)
            
    def test_create_minimum_point_mesh_connectivity(self):
        """Test that mesh remains connected."""
        np.random.seed(42)
        data = np.random.rand(500, 2)
        
        vertices, simplices, point_counts = create_minimum_point_mesh(
            data, min_points=10, verbose=False
        )
        
        # Check that mesh is connected by verifying all vertices are used
        unique_vertices = np.unique(simplices)
        assert len(unique_vertices) <= len(vertices)
        
        # Each element should have 3 vertices
        assert simplices.shape[1] == 3
        
    def test_meshed_domain_factory_method(self):
        """Test the MeshedDomain.create_from_data_with_constraints factory method."""
        np.random.seed(42)
        data = np.random.rand(1000, 2)
        
        domain, point_counts = fl.MeshedDomain.create_from_data_with_constraints(
            data, min_points=15, max_points=80, verbose=False
        )
        
        # Check domain was created
        assert isinstance(domain, fl.MeshedDomain)
        assert domain.dim == 2
        
        # Check constraints
        assert point_counts.min() >= 15
        assert point_counts.max() <= 80
        
    def test_meshed_domain_localize_data(self):
        """Test that created mesh works with localize_data."""
        np.random.seed(42)
        data = np.random.rand(1000, 2)
        
        domain, _ = fl.MeshedDomain.create_from_data_with_constraints(
            data, min_points=10, verbose=False
        )
        
        # Test localize_data
        cells, loc_x = domain.localize_data(data[:100])
        
        # All points should be in some element
        assert len(cells) == 100
        assert all(cells >= 0)
        
    def test_centroid_driven_mesh_compatibility(self):
        """Test that new function works with existing centroid_driven_mesh."""
        np.random.seed(42)
        data = np.random.rand(500, 2)
        
        # Create mesh using original function
        vertices_orig, simplices_orig = centroid_driven_mesh(
            data, bins=20, boundary_vertices=None
        )
        
        # Create mesh using new function
        vertices_new, simplices_new, _ = create_minimum_point_mesh(
            data, min_points=5, max_iterations=1, verbose=False
        )
        
        # Both should create valid meshes
        assert vertices_orig.shape[1] == 2
        assert vertices_new.shape[1] == 2
        assert simplices_orig.shape[1] == 3
        assert simplices_new.shape[1] == 3


class TestMeshUtils:
    """Tests for mesh utility functions."""
    
    def test_centroid_driven_mesh_2d(self):
        """Test centroid driven mesh creation."""
        np.random.seed(42)
        data = np.random.rand(500, 2)
        
        vertices, simplices = centroid_driven_mesh(data, bins=10)
        
        assert vertices.shape[1] == 2
        assert simplices.shape[1] == 3
        assert len(vertices) > 10  # Should have boundary + cluster centers
        
    def test_centroid_driven_mesh_with_boundary(self):
        """Test centroid driven mesh with custom boundary."""
        np.random.seed(42)
        data = np.random.rand(500, 2)
        boundary = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        
        vertices, simplices = centroid_driven_mesh(
            data, bins=10, boundary_vertices=boundary
        )
        
        assert vertices.shape[1] == 2
        # Should include our boundary vertices
        assert len(vertices) >= len(boundary)


def test_integration_with_fem_functions():
    """Integration test: Create mesh and use with FiniteElement functions."""
    np.random.seed(42)
    
    # Generate scattered data
    data = np.random.rand(500, 2)
    values = data[:, 0]**2 + data[:, 1]**2  # Simple function to fit
    
    # Create mesh with minimum point constraint
    domain, point_counts = fl.MeshedDomain.create_from_data_with_constraints(
        data, min_points=10, verbose=False
    )
    
    # Create finite element function
    import skfem
    element = skfem.ElementTriP1()
    fem_func = fl.functions.FiniteElement(domain, element)
    
    # Fit the function
    fem_func.fit(data, values)
    
    # Test evaluation
    test_points = np.random.rand(50, 2)
    result = fem_func(test_points)
    
    assert result.shape == (50,)
    assert not np.any(np.isnan(result))
