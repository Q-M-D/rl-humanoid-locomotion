import torch

def quaternion_multiply(q1, q2):
    """Multiply two quaternions
    Args:
        q1: First quaternion (4D vector)
        q2: Second quaternion (4D vector)
    Returns:
        Quaternion product of q1 and q2
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return torch.tensor([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])

def quaternion_inverse(q):
    """Compute the inverse of a quaternion
    Args:
        q: Quaternion (4D vector)
    Returns:
        Inverse of the quaternion
    """
    w, x, y, z = q
    norm_sq = w*w + x*x + y*y + z*z
    return torch.tensor([w/norm_sq, -x/norm_sq, -y/norm_sq, -z/norm_sq])

def point_rotation(point, quaternion):
    """Rotate a point using a quaternion
    Args:
        point: 3D point (3D vector)
        quaternion: Quaternion (4D vector)
    Returns:
        Rotated point
    """
    q_point = torch.tensor([0] + list(point))
    q_conjugate = quaternion_inverse(quaternion)
    rotated_point = quaternion_multiply(quaternion_multiply(quaternion, q_point), q_conjugate)
    return rotated_point[1:]  # Return only the 3D part

def point_translation(point, translation):
    """Translate a point by a given translation vector
    Args:
        point: 3D point (3D vector)
        translation: Translation vector (3D vector)
    Returns:
        Translated point
    """
    device = point.device if point.is_cuda else translation.device
    return point.to(device) + translation.to(device)  # Ensure both tensors are on the same device

def transform_point(point, quaternion, translation):
    """Transform a point using a quaternion and a translation vector
    Args:
        point: 3D point (3D vector)
        quaternion: Quaternion (4D vector)
        translation: Translation vector (3D vector)
    Returns:
        Transformed point
    """
    rotated_point = point_rotation(point, quaternion)
    transformed_point = point_translation(rotated_point, translation)
    return transformed_point

def transform_points(points, quaternion, translation):
    """Transform multiple points using a quaternion and a translation vector
    Args:
        points: List of 3D points (list of 3D vectors)
        quaternion: Quaternion (4D vector)
        translation: Translation vector (3D vector)
    Returns:
        List of transformed points
    """
    return torch.stack([transform_point(point, quaternion, translation) for point in points])

def run_tests():
    """Run a series of tests for quaternion and transformation functions"""
    print("\n=== Running Standardized Tests ===\n")
    
    # Test quaternion multiplication (identity quaternion)
    q1 = torch.tensor([1, 0, 0, 0], dtype=torch.float32)  # Identity quaternion
    q2 = torch.tensor([0, 1, 0, 0], dtype=torch.float32)
    q_product = quaternion_multiply(q1, q2)
    print("Identity quaternion multiplication test:")
    assert torch.allclose(q_product, q2)
    print("✓ Passed")
    
    # Test quaternion inverse
    q = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)
    q_inv = quaternion_inverse(q)
    q_product = quaternion_multiply(q, q_inv)
    print("\nQuaternion inverse test:")
    assert torch.allclose(q_product, torch.tensor([1, 0, 0, 0], dtype=torch.float32))
    print("✓ Passed")
    
    # Test 90-degree rotation around z-axis
    point = torch.tensor([1, 0, 0], dtype=torch.float32)
    quat_z90 = torch.tensor([torch.cos(torch.tensor(torch.pi/4)), 0, 0, torch.sin(torch.tensor(torch.pi/4))], dtype=torch.float32)
    rotated_point = point_rotation(point, quat_z90)
    print("\nRotation (90° around z-axis) test:")
    assert torch.allclose(rotated_point, torch.tensor([0, 1, 0], dtype=torch.float32), atol=1e-6)
    print("✓ Passed")
    
    # Test translation
    point = torch.tensor([1, 2, 3], dtype=torch.float32)
    translation = torch.tensor([10, 20, 30], dtype=torch.float32)
    translated_point = point_translation(point, translation)
    print("\nTranslation test:")
    assert torch.equal(translated_point, torch.tensor([11, 22, 33], dtype=torch.float32))
    print("✓ Passed")
    
    # Test combined transformation
    point = torch.tensor([1, 0, 0], dtype=torch.float32)
    quat_x180 = torch.tensor([0, 1, 0, 0], dtype=torch.float32)  # 180 degrees around x-axis
    translation = torch.tensor([5, 5, 5], dtype=torch.float32)
    transformed_point = transform_point(point, quat_x180, translation)
    print("\nCombined transformation test:")
    assert torch.allclose(transformed_point, torch.tensor([6, 5, 5], dtype=torch.float32))
    print("✓ Passed")
    
    # Test multiple point transformation
    points = [torch.tensor([1, 0, 0], dtype=torch.float32), torch.tensor([0, 1, 0], dtype=torch.float32), torch.tensor([0, 0, 1], dtype=torch.float32)]
    transformed_points = transform_points(points, quat_x180, translation)
    expected = [torch.tensor([6, 5, 5], dtype=torch.float32), torch.tensor([5, 4, 5], dtype=torch.float32), torch.tensor([5, 5, 4], dtype=torch.float32)]
    print("\nMultiple points transformation test:")
    for i, (actual, expected_pt) in enumerate(zip(transformed_points, expected)):
        assert torch.allclose(actual, expected_pt)
        print(f"Point {i+1}: ✓ Passed")
    
    print("\n=== All tests passed successfully ===")

if __name__ == "__main__":
    print("Testing quaternion and point transformation functions")
    run_tests()
