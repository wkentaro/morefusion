import math as _math

# General Kinematics #
######################

# matrices and poses


def mat_to_relative_mat(f, cam_inv_extrinsic_mat, cam_extrinsic_mat, batch_size):

    cam_inv_extrinsic_mat_full = f.concatenate((cam_inv_extrinsic_mat, f.reshape(f.tile(f.array([0., 0., 0., 1.]),
                                                                                        [batch_size]),
                                                                                 (batch_size, 1, 4))), -2)
    return f.matmul(cam_extrinsic_mat, cam_inv_extrinsic_mat_full)


def mat_to_relative_pose(f, cam_inv_extrinsic_mat, cam_extrinsic_mat, batch_size):

    relative_mat = mat_to_relative_mat(f, cam_inv_extrinsic_mat, cam_extrinsic_mat, batch_size)

    return mat_to_pose(f, relative_mat)


def mat_to_relative_quaternion_pose(f, cam_inv_extrinsic_mat, cam_extrinsic_mat, batch_size):

    relative_mat = mat_to_relative_mat(f, cam_inv_extrinsic_mat, cam_extrinsic_mat, batch_size)

    return mat_to_quaternion_pose(f, relative_mat)

def mat_to_euler_pose(f, mat):

    translation = mat[:, 3][0:3]
    euler_angles = rot_mat_to_euler(f, mat[0:3, 0:3])

    return f.concatenate((translation, euler_angles), -1)

def mat_to_pose(f, mat):

    translation = mat[:, 3]
    quaternion = rot_mat_to_quaternion(f, mat[:, 0:3])
    rot_vector = quaternion_to_rotation_vector(f, quaternion)

    return f.concatenate((translation, rot_vector), -1)


def increment_quaternion_pose_with_velocity(f, pose, velocity, control_dt):

    current_quaternion = pose[3:7]
    quaternion_vel = velocity[3:7]
    quaternion_transform = scale_quaternion_theta(f, quaternion_vel, control_dt)
    new_quaternion = hamilton_product(f, current_quaternion, quaternion_transform)
    new_pose = f.concatenate((f, pose[0:3] + velocity[0:3] * control_dt, new_quaternion))

    return new_pose


def mat_to_quaternion_pose(f, matrix):

    translation = matrix[:, 3][0:3]
    quaternion = f.array(rot_mat_to_quaternion(f, matrix[0:3, 0:3]))

    return f.concatenate((translation, quaternion))


def quaternion_pose_to_mat(f, pose):

    mat = f.identity(4)
    rot_mat = quaternion_to_rot_mat(f, pose[3:7])
    mat[0:3, 0:3] = rot_mat
    mat[:, 3][0:3] = f.asarray(pose[0:3]).transpose()

    return mat


def euler_pose_to_mat(f, pose):

    mat = f.identity(4)
    rot_mat = euler_to_rot_mat(f, pose[3:6])
    mat[0:3, 0:3] = rot_mat
    mat[:, 3][0:3] = f.asarray(pose[0:3]).transpose()

    return mat


def pose_to_mat(f, pose):

    quaternion = rotation_vector_to_quaternion(f, pose[3:6])
    rot_mat = quaternion_to_rot_mat(f, quaternion)

    return f.concatenate((rot_mat, f.expand_dims(pose[0:3], -1)), -1)


# rotation conversions


def quaternion_from_vector_and_angle(f, vector, theta):

    n = _math.cos(theta / 2)
    e1 = _math.sin(theta / 2) * vector[0]
    e2 = _math.sin(theta / 2) * vector[1]
    e3 = _math.sin(theta / 2) * vector[2]
    q = f.array([e1, e2, e3, n])
    q_len = f.linalg.norm(q)
    if q_len < 0.99 or q_len > 1.01:
        raise Exception('incorrect quaternion length')

    return q


def quaternion_to_axis_angle(f, q):

    vector, angle = vector_and_angle_from_quaternion(f, q)
    theta = f.arccos(vector[2])
    phi = f.arctan2(vector[1], vector[0])

    return [theta, phi, angle]


def axis_angle_to_quaternion(f, angles):

    theta = angles[0]
    phi = angles[1]
    angle = angles[2]
    x = f.sin(theta) * f.cos(phi)
    y = f.sin(theta) * f.sin(phi)
    z = f.cos(theta)
    vector = [x, y, z]

    return quaternion_from_vector_and_angle(f, vector, angle)


def vector_and_angle_from_quaternion(f, q):

    e1 = q[0]
    e2 = q[1]
    e3 = q[2]
    n = q[3]

    theta = 2*_math.acos(max(0, min(n, 1)))
    vector_x = e1/_math.sin(theta/2) if theta != 0 else 0
    vector_y = e2/_math.sin(theta/2) if theta != 0 else 0
    vector_z = e3/_math.sin(theta/2) if theta != 0 else 0

    return f.array([vector_x, vector_y, vector_z]), theta


def quaternion_to_rot_mat(f, q):

    a = q[3]
    b = q[0]
    c = q[1]
    d = q[2]

    return f.array(
        [[_math.pow(a, 2)+_math.pow(b, 2) -
          _math.pow(c, 2)-_math.pow(d, 2),      2*b*c-2*a*d,                        2*b*d+2*a*c],

         [2*b*c+2*a*d,                      _math.pow(a, 2)-_math.pow(b, 2) +
                                            _math.pow(c, 2)-_math.pow(d, 2),        2*c*d-2*a*b],

         [2*b*d-2*a*c,                      2*c*d+2*a*b,                        _math.pow(a,2)-_math.pow(b,2)-
                                                                                _math.pow(c,2)+_math.pow(d,2)]])


def rot_mat_to_quaternion(f, m):
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

    tr = m[0,0] + m[1,1] + m[2,2]

    if tr > 0:
        s = _math.sqrt(tr+1)*2
        qw = 0.25*s
        qx = (m[2,1] - m[1,2])/s
        qy = (m[0,2] - m[2,0])/s
        qz = (m[1,0] - m[0,1])/s
    elif (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
        s = _math.sqrt(1 + m[0,0] - m[1,1] - m[2,2]) * 2
        qw = (m[2,1] - m[1,2]) / s
        qx = 0.25 * s
        qy = (m[0,1] + m[1,0]) / s
        qz = (m[0,2] + m[2,0]) / s
    elif m[1,1] > m[2,2]:
        s = _math.sqrt(1 + m[1,1] - m[0,0] - m[2,2]) * 2
        qw = (m[0,2] - m[2,0]) / s
        qx = (m[0,1] + m[1,0]) / s
        qy = 0.25 * s
        qz = (m[1,2] + m[2,1]) / s
    else:
        s = _math.sqrt(1 + m[2,2] - m[0,0] - m[1,1]) * 2
        qw = (m[1,0] - m[0,1]) / s
        qx = (m[0,2] + m[2,0]) / s
        qy = (m[1,2] + m[2,1]) / s
        qz = 0.25 * s

    return f.array([qx,qy,qz,qw])


def rot_mat_to_euler(f, rot_mat):

    # based on: https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2012/07/euler-angles1.pdf

    alpha = _math.atan2(rot_mat[1, 2], rot_mat[2, 2])
    c2 = _math.sqrt(_math.pow(rot_mat[0, 0], 2) + _math.pow(rot_mat[0, 1], 2))
    beta = _math.atan2(-rot_mat[0, 2], c2)
    s1 = _math.sin(alpha)
    c1 = _math.cos(alpha)
    gamma = _math.atan2(s1 * rot_mat[2, 0] - c1 * rot_mat[1, 0], c1 * rot_mat[1, 1] - s1 * rot_mat[2, 1])

    alpha = -alpha
    beta = -beta
    gamma = -gamma

    return f.array([alpha, beta, gamma])


def euler_to_rot_mat(f, euler_angles):

    alpha = euler_angles[0]
    beta = euler_angles[1]
    gamma = euler_angles[2]

    rot_x = f.array([[1, 0, 0],
                     [0, _math.cos(alpha), -_math.sin(alpha)],
                     [0, _math.sin(alpha), _math.cos(alpha)]])

    rot_y = f.array([[_math.cos(beta), 0, _math.sin(beta)],
                     [0, 1, 0],
                     [-_math.sin(beta), 0, _math.cos(beta)]])

    rot_z = f.array([[_math.cos(gamma), -_math.sin(gamma), 0],
                     [_math.sin(gamma), _math.cos(gamma), 0],
                     [0, 0, 1]])

    return f.matmul(rot_x, f.matmul(rot_y, rot_z))


def quaternion_to_euler(f, q):
    return rot_mat_to_euler(f, quaternion_to_rot_mat(f, q))


def euler_to_quaternion(f, euler_angles):
    return rot_mat_to_quaternion(f, euler_to_rot_mat(f, euler_angles))

def get_random_quaternion(f, max_theta=_math.pi):
    quaternion_vel_vector = f.random.uniform(0, 1, 3)
    vec_len = f.linalg.norm(quaternion_vel_vector)
    if vec_len != 0: quaternion_vel_vector /= vec_len
    theta = f.random.uniform(-max_theta, max_theta)
    return quaternion_from_vector_and_angle(f, quaternion_vel_vector, theta)

def scale_quaternion_theta(f, quaternion, scale):
    vector, angle = vector_and_angle_from_quaternion(f, quaternion)
    angle *= scale
    return quaternion_from_vector_and_angle(f, vector, angle)

def quaternion_to_rotation_vector(f, q):
    vector, angle = vector_and_angle_from_quaternion(f, q)
    return f.asarray(vector) * angle

def rotation_vector_to_quaternion(f, v):
    theta = f.linalg.norm(v)
    vector = v/theta if theta != 0 else v
    return quaternion_from_vector_and_angle(f, vector, theta)


# rotation functions

def hamilton_product(f, q1, q2):
    # https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    term1 = q2[3] * q1[0] + q2[0] * q1[3] + q2[1] * q1[2] - q2[2] * q1[1]
    term2 = q2[3] * q1[1] - q2[0] * q1[2] + q2[1] * q1[3] + q2[2] * q1[0]
    term3 = q2[3] * q1[2] + q2[0] * q1[1] - q2[1] * q1[0] + q2[2] * q1[3]
    term4 = q2[3] * q1[3] - q2[0] * q1[0] - q2[1] * q1[1] - q2[2] * q1[2]
    return f.array([term1, term2, term3, term4])

def inverse_hamilton_product(f, q1, hp):

    a = f.array([[q1[3], q1[2], -q1[1], q1[0]],
                 [-q1[2],  q1[3],  q1[0], q1[1]],
                 [ q1[1], -q1[0],  q1[3], q1[2]],
                 [-q1[0], -q1[1], -q1[2], q1[3]]])
    b = f.array([[hp[0]],
                 [hp[1]],
                 [hp[2]],
                 [hp[3]]])
    a_inv = f.linalg.inv(a)
    # A x = b -> x = A_inv b
    x = f.matmul(a_inv, b)
    q2 = x.reshape(4)

    return q2
