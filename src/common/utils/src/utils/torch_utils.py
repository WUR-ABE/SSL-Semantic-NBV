import torch


def look_at_rotation(
    eye: torch.tensor,
    target: torch.tensor,
    ref: torch.tensor = torch.tensor([1.0, 0.0, 0.0]),
) -> torch.tensor:
    """
    Compute the quaternion rotation to look at a target from a given eye position
    :param eye: eye position
    :param target: target position
    :param ref: reference vector
    :return: quaternion rotation
    """
    dir = target - eye
    dir = dir / torch.norm(dir)
    ref = ref.to(dir.device).to(dir.dtype)
    # Determine the rotation axis and angle
    rot_axis = torch.cross(ref, dir)
    rot_axis = rot_axis / torch.norm(rot_axis)
    rot_angle = torch.arccos(torch.dot(ref, dir))
    # If the rotation axis is nan, return the identity quaternion
    if torch.isnan(rot_axis).any():
        return torch.tensor([1.0, 0.0, 0.0, 0.0], device=dir.device, dtype=dir.dtype)
    # Convert the rotation axis and angle to a quaternion
    quat = axangle2quat(rot_axis, rot_angle, True)
    return quat


def axangle2quat(vector: torch.tensor, theta: float, is_normalized=False):
    """
    Convert an axis-angle representation to a quaternion representation.
    :param vector: tensor of rotation axis (..., 3)
    :param theta: tensor of rotation angle (...)
    :param is_normalized: whether the rotation axis is already normalized
    :return: tensor of quaternions (..., 4) ordered (w, x, y, z)
    """
    if not is_normalized:
        vector = vector / torch.norm(vector)
    t2 = theta / 2.0
    st2 = torch.sin(t2)
    return torch.cat((torch.cos(t2)[None, ...], vector * st2))


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    :param quaternions: tensor of quaternions (..., 4) ordered (w, x, y, z)
    :return: tensor of rotation matrices (..., 3, 3)
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def transform_from_rotation_translation(
    quaternions: torch.Tensor, translations: torch.Tensor
) -> torch.Tensor:
    """
    Convert rotations given as quaternions and translation vectors to 4x4 transformation matrices.
    :param quaternions: tensor of quaternions (..., 4) ordered (w, x, y, z)
    :param translations: tensor of translation vectors (..., 3)
    :return: tensor of transformation matrices (..., 4, 4)
    """
    matrices = (
        torch.eye(4, device=quaternions.device)
        .unsqueeze(0)
        .repeat(quaternions.shape[0], 1, 1)
    )
    matrices[:, :3, :3] = quaternion_to_matrix(quaternions)
    matrices[:, :3, 3] = translations
    return matrices


def unravel_index(indices, shape):
    """
    Convert a flat index or tensor of flat indices into a tuple of coordinate arrays.
    :param indices: tensor of flat indices
    :param shape: shape of the array into which the indices point
    :return: tuple of coordinate arrays
    """
    shape = torch.tensor(shape, device=indices.device)
    strides = torch.flip(shape.cumprod(dim=0), [0])
    indices = indices.unsqueeze(-1)
    unravel_idx = (torch.floor_divide(indices, strides) % shape).long()
    return unravel_idx


def gaussian_kernel(kernel_size, sigma, device):
    """
    Create a 3D Gaussian kernel.
    :param kernel_size: size of the kernel
    :param sigma: standard deviation of the Gaussian
    :param device: device on which to create the kernel
    :return: 3D Gaussian kernel
    """
    # Create a 1D Gaussian kernel along each dimension
    kernel_1d = torch.arange(
        -(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32, device=device
    )
    kernel_1d = torch.exp(-0.5 * (kernel_1d / sigma) ** 2)
    kernel_1d = kernel_1d / torch.sum(kernel_1d)  # Normalize the kernel
    # Create a 3D Gaussian kernel by computing the outer product of the 1D kernels
    kernel_3d = torch.einsum("i,j,k->ijk", kernel_1d, kernel_1d, kernel_1d)
    kernel_3d = kernel_3d.view(1, 1, kernel_size, kernel_size, kernel_size)
    return kernel_3d
