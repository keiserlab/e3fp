"""Various array operations.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform

QUATERNION_DTYPE = np.float64
X_AXIS, Y_AXIS, Z_AXIS = np.identity(3, dtype=np.float64)
EPS = 1e-12  # epsilon, a number close to 0


# Vector Algebra Methods
def as_unit(v, axis=1):
    """Return array of unit vectors parallel to vectors in `v`.

    Parameters
    ----------
    v : ndarray of float
    axis : int, optional
        Axis along which to normalize length.

    Returns
    -------
    ndarray of float : Unit vector of `v`, i.e. `v` divided by its
                       magnitude along `axis`.
    """
    u = np.array(v, dtype=np.float64, copy=True)
    if u.ndim == 1:
        sqmag = u.dot(u)
        if sqmag >= EPS:
            u /= sqmag ** 0.5
    else:
        if axis == 1:
            sqmag = np.einsum("...ij,...ij->...i", u, u)
        else:
            sqmag = np.einsum("...ij,...ij->...j", u, u)

        sqmag[sqmag < EPS] = 1.0
        u /= np.expand_dims(np.sqrt(sqmag), axis)
    return u


def make_distance_matrix(coords):
    """Build pairwise distance matrix from coordinates.

    Parameters
    ----------
    coords : ndarray of float
        an Mx3 array of cartesian coordinates.

    Returns
    -------
    ndarray of float : square symmetrical distance matrix
    """
    return squareform(pdist(coords))


def make_transform_matrix(center, y=None, z=None):
    """Make 4x4 homogenous transformation matrix.

    Given Nx4 array A where A[:, 4] = 1., the transform matrix M should be
    used with dot(M, A.T).T. Order of operations is 1. translation, 2. align
    `y` x `z` plane to yz-plane 3. align `y` to y-axis.

    Parameters
    ----------
    center : 1x3 array of float
        Coordinate that should be centered after transformation.
    y : None or 1x3 array of float
        Vector that should lie on the y-axis after transformation
    z : None or 1x3 array of float
        Vector that after transformation should lie on yz-plane in direction
        of z-axis.

    Returns
    -------
    4x4 array of float
        4x4 homogenous transformation matrix.
    """
    translate = np.identity(4, dtype=np.float64)
    translate[:3, 3] = -np.asarray(center, dtype=np.float64)
    if y is not None:
        y = np.atleast_2d(y)
        if z is None:
            rotate = np.identity(4, dtype=np.float64)
            rotate[:3, :3] = make_rotation_matrix(y, Y_AXIS)
        else:
            z = np.atleast_2d(z)
            rotate_norm = np.identity(4, dtype=np.float64)
            x_unit = as_unit(np.cross(y, z))
            rotate_norm[:3, :3] = make_rotation_matrix(x_unit, X_AXIS)
            new_y = np.dot(rotate_norm[:3, :3], y.flatten())
            rotate_y = np.identity(4, dtype=np.float64)
            rotate_y[:3, :3] = make_rotation_matrix(new_y.flatten(), Y_AXIS)
            rotate = np.dot(rotate_y, rotate_norm)
        transform = np.dot(rotate, translate)
    else:
        transform = translate
    return transform


def make_rotation_matrix(v0, v1):
    """Create 3x3 matrix of rotation from `v0` onto `v1`.

    Should be used by dot(R, v0.T).T.

    Parameters
    ----------
    v0 : 1x3 array of float
        Initial vector before alignment.
    v1 : 1x3 array of float
        Vector to which to align `v0`.
    """
    v0 = as_unit(v0)
    v1 = as_unit(v1)
    u = np.cross(v0.ravel(), v1.ravel())
    if np.all(u == 0.0):
        return np.identity(3, dtype=np.float64)
    sin_ang = u.dot(u) ** 0.5
    u /= sin_ang
    cos_ang = np.dot(v0, v1.T)
    # fmt: off
    ux = np.array([[   0., -u[2],  u[1]],
                   [ u[2],    0., -u[0]],
                   [-u[1],  u[0],    0.]], dtype=np.float64)
    # fmt: on
    rot = (
        cos_ang * np.identity(3, dtype=np.float64)
        + sin_ang * ux
        + (1 - cos_ang) * np.outer(u, u)
    )
    return rot


def transform_array(transform_matrix, a):
    """Pad an array with 1s, transform, and return with original dimensions.

    Parameters
    ----------
    transform_matrix : 4x4 array of float
        4x4 homogenous transformation matrix
    a : Nx3 array of float
        Array of 3-D coordinates.

    Returns
    -------
    Nx3 array of float : Transformed array
    """
    return unpad_array(np.dot(transform_matrix, pad_array(a).T).T)


def pad_array(a, n=1.0, axis=1):
    """Return `a` with row of `n` appended to `axis`.

    Parameters
    ----------
    a : ndarray
        Array to pad
    n : float or int, optional
        Value to pad `a` with
    axis : int, optional
        Axis of `a` to pad with `n`.

    Returns
    -------
    ndarray
        Padded array.
    """
    if a.ndim == 1:
        pad = np.ones(a.shape[0] + 1, dtype=a.dtype) * n
        pad[: a.shape[0]] = a
    else:
        shape = list(a.shape)
        shape[axis] += 1
        pad = np.ones(shape, dtype=a.dtype)
        pad[: a.shape[0], : a.shape[1]] = a
    return pad


def unpad_array(a, axis=1):
    """Return `a` with row removed along `axis`.

    Parameters
    ----------
    a : ndarray
        Array from which to remove row
    axis : int, optional
        Axis from which to remove row

    Returns
    -------
    ndarray
        Unpadded array.
    """
    if a.ndim == 1:
        return a[:-1]
    else:
        shape = list(a.shape)
        shape[axis] -= 1
        return a[: shape[0], : shape[1]]


def project_to_plane(vec_arr, norm):
    """Project array of vectors to plane with normal `norm`.

    Parameters
    ----------
    vec_arr : Nx3 array
        Array of N 3D vectors.
    norm : 1x3 array
        Normal vector to plane.

    Returns
    -------
    Nx3 array
        Array of vectors projected onto plane.
    """
    unit_norm = as_unit(norm).flatten()
    mag_on_norm = np.dot(vec_arr, unit_norm)
    if vec_arr.ndim == 1:
        vec_on_norm = np.array(unit_norm, copy=True)
        vec_on_norm *= mag_on_norm
    else:
        vec_on_norm = np.tile(unit_norm, (vec_arr.shape[0], 1))
        vec_on_norm *= mag_on_norm[:, None]
    return vec_arr - vec_on_norm


def calculate_angles(vec_arr, ref, ref_norm=None):
    """Calculate angles between vectors in `vec_arr` and `ref` vector.

    If `ref_norm` is not provided, angle ranges between 0 and pi. If it is
    provided, angle ranges between 0 and 2pi. Note that if `ref_norm` is
    orthogonal to `vec_arr` and `ref`, then the angle is rotation around the
    axis, but if a non-orthogonal axis is provided, this may not be the case.

    Parameters
    ----------
    vec_arr : Nx3 array of float
        Array of N 3D vectors.
    ref : 1x3 array of float
        Reference vector
    ref_norm : 1x3 array of float
        Normal vector.

    Returns
    -------
    1-D array
        Array of N angles
    """
    unit_vec_arr = as_unit(vec_arr)
    unit_ref = as_unit(ref).flatten()
    ang = np.arccos(np.clip(np.dot(unit_vec_arr, unit_ref), -1.0, 1.0))
    # handle cases where a vector is the origin
    ang[np.all(unit_vec_arr == np.zeros(3), axis=1)] = 0.0
    if ref_norm is not None:
        sign = np.sign(
            np.dot(ref_norm, np.cross(unit_vec_arr, unit_ref).T)
        ).flatten()
        sign[sign == 0] = 1
        ang = rotate_angles(sign * ang, 2 * np.pi)
    return ang


def rotate_angles(angles, amount):
    """Rotate angles by `amount`, keeping in 0 to 2pi range.

    Parameters
    ----------
    angles : 1-D array of float
        Angles in radians
    amount : float
        Amount to rotate angles by

    Returns
    -------
    1-D array of float : Rotated angles
    """
    return (angles + amount) % (2 * np.pi)


def quaternion_to_transform_matrix(quaternion, translation=np.zeros(3)):
    """Convert quaternion to homogenous 4x4 transform matrix.

    Parameters
    ----------
    quaternion : 4x1 array of float
        Quaternion describing rotation after translation.
    translation : 3x1 array of float, optional
        Translation to be performed before rotation.
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.identity(4, dtype=np.float64)
    q /= n
    q = 2 * np.outer(q, q)
    # fmt: off
    transform_mat = np.array(
        [[1.-q[2, 2]-q[3, 3],    q[1, 2]-q[3, 0],    q[1, 3]+q[2, 0], 0.],
         [   q[1, 2]+q[3, 0], 1.-q[1, 1]-q[3, 3],    q[2, 3]-q[1, 0], 0.],
         [   q[1, 3]-q[2, 0],    q[2, 3]+q[1, 0], 1.-q[1, 1]-q[2, 2], 0.],
         [                0.,                 0.,                 0., 1.]],
        dtype=np.float64
    )
    # fmt: on
    transform_mat[:3, 3] = translation
    return transform_mat


def transform_matrix_to_quaternion(transform_matrix, dtype=QUATERNION_DTYPE):
    """Convert homogenous 4x4 transform matrix to quaternion.

    Parameters
    ----------
    transform_matrix : 4x4 array of float
        Homogenous transformation matrix.
    dtype : numpy dtype, optional
        Datatype for returned quaternion.
    """
    T = np.array(transform_matrix, dtype=np.float64)
    R = T[:3, :3]
    q = np.zeros(4, dtype=dtype)
    q[0] = np.sqrt(1.0 + R.trace()) / 2.0
    q[1] = R[2, 1] - R[1, 2]
    q[2] = R[0, 2] - R[2, 0]
    q[3] = R[1, 0] - R[0, 1]
    q[1:4] /= 4.0 * q[0]
    return q
