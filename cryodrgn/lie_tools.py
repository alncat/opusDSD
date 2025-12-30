'''
Tools for dealing with SO(3) group and algebra
Adapted from https://github.com/pimdh/lie-vae
All functions are pytorch-ified
'''

import torch
from torch.distributions import Normal
import numpy as np
import healpy as hp

def map_to_lie_algebra(v):
    """Map a point in R^N to the tangent space at the identity, i.e.
    to the Lie Algebra
    Arg:
        v = vector in R^N, (..., 3) in our case
    Return:
        R = v converted to Lie Algebra element, (3,3) in our case"""

    # make sure this is a sample from R^3
    assert v.size()[-1] == 3

    R_x = v.new_tensor([[ 0., 0., 0.],
                        [ 0., 0.,-1.],
                        [ 0., 1., 0.]])

    R_y = v.new_tensor([[ 0., 0., 1.],
                        [ 0., 0., 0.],
                        [-1., 0., 0.]])

    R_z = v.new_tensor([[ 0.,-1., 0.],
                        [ 1., 0., 0.],
                        [ 0., 0., 0.]])

    R = R_x * v[..., 0, None, None] + \
        R_y * v[..., 1, None, None] + \
        R_z * v[..., 2, None, None]
    return R

def expmap(v):
    theta = v.norm(p=2, dim=-1, keepdim=True)
    # normalize K
    K = map_to_lie_algebra(v / theta)

    I = torch.eye(3, device=v.device, dtype=v.dtype)
    R = I + torch.sin(theta)[..., None]*K \
        + (1. - torch.cos(theta))[..., None]*(K@K)
    return R

def s2s1rodrigues(s2_el, s1_el):
    K = map_to_lie_algebra(s2_el)
    cos_theta = s1_el[...,0]
    sin_theta = s1_el[...,1]
    I = torch.eye(3, device=s2_el.device, dtype=s2_el.dtype)
    R = I + sin_theta[..., None, None]*K \
        + (1. - cos_theta)[..., None, None]*(K@K)
    return R

def s2s2_to_SO3(v1, v2=None):
    '''Normalize 2 3-vectors. Project second to orthogonal component.
    Take cross product for third. Stack to form SO matrix.'''
    if v2 is None:
        assert v1.shape[-1] == 6
        v2 = v1[...,3:]
        v1 = v1[...,0:3]
    u1 = v1
    e1 = u1 / u1.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    u2 = v2 - (e1 * v2).sum(-1, keepdim=True) * e1
    e2 = u2 / u2.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    e3 = torch.cross(e1, e2)
    return torch.stack([e1, e2, e3], 1)

def SO3_to_s2s2(r):
    '''Map batch of SO(3) matrices to s2s2 representation as first two
    basis vectors, concatenated as Bx6'''
    return r.view(*r.shape[:-2],9)[...,:6].contiguous()

def rotation_loss(pred_rots, ref_rots):
    #print(pred_rots.shape, ref_rots.shape)
    prod = -(pred_rots*ref_rots).sum(-1).sum(-1)
    return prod

def translation_loss(pred_trans, ref_trans):
    #print(pred_trans.shape, ref_trans.shape)
    return (pred_trans - ref_trans).pow(2).sum(-1)

def SO3_to_quaternions(r):
    """Map batch of SO(3) matrices to quaternions."""
    batch_dims = r.shape[:-2]
    assert list(r.shape[-2:]) == [3, 3], 'Input must be 3x3 matrices'
    r = r.view(-1, 3, 3)
    n = r.shape[0]

    diags = [r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]]
    denom_pre = torch.stack([
        1 + diags[0] - diags[1] - diags[2],
        1 - diags[0] + diags[1] - diags[2],
        1 - diags[0] - diags[1] + diags[2],
        1 + diags[0] + diags[1] + diags[2]
    ], 1)
    denom = 0.5 * torch.sqrt(1E-6 + torch.abs(denom_pre))

    case0 = torch.stack([
        denom[:, 0],
        (r[:, 0, 1] + r[:, 1, 0]) / (4 * denom[:, 0]),
        (r[:, 0, 2] + r[:, 2, 0]) / (4 * denom[:, 0]),
        (r[:, 1, 2] - r[:, 2, 1]) / (4 * denom[:, 0])
    ], 1)
    case1 = torch.stack([
        (r[:, 0, 1] + r[:, 1, 0]) / (4 * denom[:, 1]),
        denom[:, 1],
        (r[:, 1, 2] + r[:, 2, 1]) / (4 * denom[:, 1]),
        (r[:, 2, 0] - r[:, 0, 2]) / (4 * denom[:, 1])
    ], 1)
    case2 = torch.stack([
        (r[:, 0, 2] + r[:, 2, 0]) / (4 * denom[:, 2]),
        (r[:, 1, 2] + r[:, 2, 1]) / (4 * denom[:, 2]),
        denom[:, 2],
        (r[:, 0, 1] - r[:, 1, 0]) / (4 * denom[:, 2])
    ], 1)
    case3 = torch.stack([
        (r[:, 1, 2] - r[:, 2, 1]) / (4 * denom[:, 3]),
        (r[:, 2, 0] - r[:, 0, 2]) / (4 * denom[:, 3]),
        (r[:, 0, 1] - r[:, 1, 0]) / (4 * denom[:, 3]),
        denom[:, 3]
    ], 1)

    cases = torch.stack([case0, case1, case2, case3], 1)

    quaternions = cases[torch.arange(n, dtype=torch.long),
                        torch.argmax(denom.detach(), 1)]
    return quaternions.view(*batch_dims, 4)


def quaternions_to_SO3(q):
    '''Normalizes q and maps to group matrix.'''
    q = q / q.norm(p=2, dim=-1, keepdim=True)
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # can revert the order to x y z
    return torch.stack([
        r*r - i*i - j*j + k*k, 2*(r*i + j*k), 2*(r*j - i*k),
        2*(r*i - j*k), -r*r + i*i - j*j + k*k, 2*(i*j + r*k),
        2*(r*j + i*k), 2*(i*j - r*k), -r*r - i*i + j*j + k*k
        ], -1).view(*q.shape[:-1], 3, 3)

def SO3_to_quaternions_wiki(r):
    """Map batch of SO(3) matrices to quaternions."""
    batch_dims = r.shape[:-2]
    assert list(r.shape[-2:]) == [3, 3], 'Input must be 3x3 matrices'
    r = r.view(-1, 3, 3)
    n = r.shape[0]

    diags = [r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]]
    denom_pre = torch.stack([
        1 + diags[0] + diags[1] + diags[2],
        1 + diags[0] - diags[1] - diags[2],
        1 - diags[0] + diags[1] - diags[2],
        1 - diags[0] - diags[1] + diags[2]
    ], 1)
    denom = 0.5 * torch.sqrt(1e-6 + torch.abs(denom_pre))

    case0 = torch.stack([
        denom[:, 0],
        (r[:, 2, 1] - r[:, 1, 2]) / (4 * denom[:, 0]),
        (r[:, 0, 2] - r[:, 2, 0]) / (4 * denom[:, 0]),
        (r[:, 1, 0] - r[:, 0, 1]) / (4 * denom[:, 0])
    ], 1)
    case1 = torch.stack([
        (r[:, 2, 1] - r[:, 1, 2]) / (4 * denom[:, 1]),
        denom[:, 1],
        (r[:, 0, 1] + r[:, 1, 0]) / (4 * denom[:, 1]),
        (r[:, 0, 2] + r[:, 2, 0]) / (4 * denom[:, 1])
    ], 1)
    case2 = torch.stack([
        (r[:, 0, 2] - r[:, 2, 0]) / (4 * denom[:, 2]),
        (r[:, 0, 1] + r[:, 1, 0]) / (4 * denom[:, 2]),
        denom[:, 2],
        (r[:, 1, 2] + r[:, 2, 1]) / (4 * denom[:, 2])
    ], 1)
    case3 = torch.stack([
        (r[:, 1, 0] - r[:, 0, 1]) / (4 * denom[:, 3]),
        (r[:, 0, 2] + r[:, 2, 0]) / (4 * denom[:, 3]),
        (r[:, 1, 2] + r[:, 2, 1]) / (4 * denom[:, 3]),
        denom[:, 3]
    ], 1)

    cases = torch.stack([case0, case1, case2, case3], 1)

    quaternions = cases[torch.arange(n, dtype=torch.long),
                        torch.argmax(denom.detach(), 1)]
    return quaternions.view(*batch_dims, 4)


def quaternions_to_SO3_wiki(q):
    '''Normalizes q and maps to group matrix.'''
    q = q / q.norm(p=2, dim=-1, keepdim=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    return torch.stack([
        1. - 2*y*y - 2*z*z, 2*(x*y - z*w), 2*(x*z + y*w),
        2*(x*y + z*w), 1. - 2*x*x - 2*z*z, 2*(y*z - x*w),
        2*(x*z - y*w), 2*(y*z + x*w), 1. - 2*x*x - 2*y*y
        ], -1).view(*q.shape[:-1], 3, 3)

def exp_quaternion(q):
    '''exponentiate q'''
    q_norm = q.norm(p=2, dim=-1, keepdim=True)
    sinc_v = torch.special.sinc(q_norm/np.pi)
    q = torch.cat([q_norm.cos(), sinc_v*q], dim=-1)
    return q

def rot_to_axis(R):
    quat = SO3_to_quaternions_wiki(R)
    axis = quat[..., 1:]
    quat_norm = axis.norm(p=2, dim=-1, keepdim=False)
    #angle in degree
    angle = torch.atan2(quat_norm, quat[..., 0])*2/np.pi*180
    return angle, torch.nn.functional.normalize(axis, p=2, dim=-1)

def zrot(x):
    x = x*np.pi/180.
    ca = x.cos()
    sa = x.sin()
    zero = torch.zeros_like(x)
    one = torch.ones_like(x)
    return torch.stack([
        ca, sa, zero,
        -sa, ca, zero,
        zero, zero, one
    ], -1).view(*x.shape, 3, 3)

def yrot(x):
    x = x*np.pi/180.
    ca = x.cos()
    sa = x.sin()
    zero = torch.zeros_like(x)
    one = torch.ones_like(x)
    return torch.stack([
        ca, zero, -sa,
        zero, one, zero,
        sa, zero, ca
    ], -1).view(*x.shape, 3, 3)

def xrot(x):
    x = x*np.pi/180.
    ca = x.cos()
    sa = x.sin()
    zero = torch.zeros_like(x)
    one = torch.ones_like(x)
    return torch.stack([
        one, zero, zero,
        zero, ca, -sa,
        zero, sa, ca
    ], -1).view(*x.shape, 3, 3)

def skew_symmetric(u):
    ux = u[..., 0]
    uy = u[..., 1]
    uz = u[..., 2]
    zero = torch.zeros_like(ux)
    K = torch.stack([
    zero, -uz, uy,
    uz, zero, -ux,
    -uy, ux, zero
    ], -1).view(*u.shape[:-1], 3, 3)
    return K

def axis_rot(x, u):
    x = x*np.pi/180.
    #normalize u
    #u = u/u.norm(p=2, dim=-1, keepdim=True)
    K = skew_symmetric(u)
    K2 = K @ K
    x = x.unsqueeze(-1).unsqueeze(-1)
    R = torch.eye(3, device=u.device) + x.sin()*K + (1 - x.cos())*K2
    #quat = torch.cat([(x.unsqueeze(-1)/2.).cos(), (x.unsqueeze(-1)/2.).sin()*u], dim=-1)
    #R_quat = quaternions_to_SO3_wiki(quat)
    #print( R_quat - R, R @ torch.transpose(R, -1, -2) )
    return R

def exp_se3(u_norm, n, trans):
    #u (n, 3), trans (n, 3)
    #u_norm = u.norm(p=2, dim=-1, keepdim=True)
    #n = u/u_norm
    u_deg = u_norm*180/np.pi
    rot = axis_rot(u_deg.squeeze(-1), n)

    sinc_u = torch.sinc(u_norm.unsqueeze(-1)).detach()
    sinc_u_2 = torch.sinc(u_norm.unsqueeze(-1)*0.5).detach()

    K = skew_symmetric(n)
    #rot_theta = -sinc_u*(K @ K) + sinc_u_2.pow(2)*u_norm.unsqueeze(-1)*0.5*K
    rot_theta = sinc_u_2.pow(2)*u_norm.unsqueeze(-1)*0.5*K
    #print(n.shape, rot_theta.shape, trans.shape)
    rot_center = rot_theta @ trans.unsqueeze(-1)
    #libration = (1 - sinc_u)*(n.unsqueeze(-1) @ n.unsqueeze(-2)) @ trans.unsqueeze(-1) + trans.unsqueeze(-1)*sinc_u
    libration = (1 - sinc_u)* K @ K @ trans.unsqueeze(-1) + trans.unsqueeze(-1)
    return rot, (rot_center + libration).squeeze(-1)

def euler_to_direction(euler):
    alpha = euler[...,0]*np.pi/180
    beta = euler[...,1]*np.pi/180

    ca = torch.cos(alpha)
    cb = torch.cos(beta)
    sa = torch.sin(alpha)
    sb = torch.sin(beta)
    sc = sb * ca
    ss = sb * sa

    vec = torch.stack([sc, ss, cb], dim=-1)
    return vec

def direction_to_euler(v):
    v = v / v.norm(p=2, dim=-1, keepdim=True)
    alpha = torch.atan2(v[..., 1], v[..., 0])/np.pi*180
    beta = torch.acos(v[..., 2])/np.pi*180

    alpha = torch.where(beta.abs() < 0.001, torch.tensor(0.), alpha)
    alpha = torch.where((beta - 180.).abs() < 0.001, torch.tensor(0.), alpha)

    return torch.stack([alpha, beta], dim=-1)

def hopf_to_direction(euler):
    alpha = euler[...,0]*np.pi/180
    beta = euler[...,1]*np.pi/180

    ca = torch.cos(alpha)
    cb = torch.cos(beta)
    sa = torch.sin(alpha)
    sb = torch.sin(beta)
    sc = sb * ca
    ss = sb * sa

    vec = torch.stack([-sc, ss, cb], dim=-1)
    return vec

def direction_to_hopf(v):
    v = v / v.norm(p=2, dim=-1, keepdim=True)
    phi = torch.atan2(v[..., 1], -v[..., 0])/np.pi*180
    theta = torch.acos(v[..., 2])/np.pi*180

    return torch.stack([phi, theta], dim=-1)

def random_direction(n, deg):
    #rad = deg*np.pi/180
    u = torch.rand((n, 2))
    u[..., 0] = u[..., 0]*2*180.
    u[..., 1] = torch.acos(u[..., 1]*2. - 1)/np.pi*deg
    v = euler_to_direction(u)
    return v

def euler_to_SO3(euler):
    #euler = euler*np.pi/180.
    if euler.shape[-1] == 3:
        Ra = zrot(euler[..., 0])
        Rb = yrot(euler[..., 1])
        Ry = zrot(euler[..., 2])
        R = Ry @ Rb @ Ra
    elif euler.shape[-1] == 2:
        Ra = zrot(euler[..., 0])
        Rb = yrot(euler[..., 1])
        R = Rb @ Ra
    else:
        raise Exception("wrong shape {}".format(euler.shape))
    return R

def quat_div(q, r):
    #t = q/r
    t0 = r*q.sum(-1)
    t1 = r[..., 0]*q[..., 1] - r[..., 1]*q[..., 0] - r[..., 2]*q[..., 3] + r[..., 3]*q[..., 2]
    t2 = r[..., 0]*q[..., 2] + r[..., 1]*q[..., 3] - r[..., 2]*q[..., 0] - r[..., 3]*q[..., 1]
    t3 = r[..., 0]*q[..., 3] - r[..., 1]*q[..., 2] + r[..., 2]*q[..., 1] - r[..., 3]*q[..., 0]
    return torch.stack([t0, t1, t2, t3], dim=-1)

def quat_mul(a, b):
    #z = a*b
    z0 = a[..., 0]*b[..., 0] - a[..., 1]*b[..., 1] - a[..., 2]*b[..., 2] - a[..., 3]*b[..., 3]
    z1 = a[..., 0]*b[..., 1] + a[..., 1]*b[..., 0] + a[..., 2]*b[..., 3] - a[..., 3]*b[..., 2]
    z2 = a[..., 0]*b[..., 2] - a[..., 1]*b[..., 3] + a[..., 2]*b[..., 0] + a[..., 3]*b[..., 1]
    z3 = a[..., 0]*b[..., 3] + a[..., 1]*b[..., 2] - a[..., 2]*b[..., 1] + a[..., 3]*b[..., 0]
    return torch.stack([z0, z1, z2, z3], dim=-1)

def euler_to_hopf(euler):
    R = euler_to_SO3(euler)
    return so3_to_hopf(R)

def hopf_to_euler(hopf):
    R = hopf_to_SO3(hopf)
    return so3_to_euler(R)

def quat_to_hopf(v):
    psi = 2*torch.atan2(v[..., 3], v[..., 0])/np.pi*180
    p2 = psi*0.5*np.pi/180
    zero = torch.zeros_like(psi)
    psi_quat = torch.stack([p2.cos(), zero, zero, -p2.sin()], dim=-1)
    v = quat_mul(psi_quat, v)
    ctheta = v[..., 0].pow(2) + v[..., 3].pow(2)
    ctheta = torch.clip(2*ctheta - 1., min=-1., max=1.)
    theta = torch.acos(ctheta)/np.pi*180
    #print(theta[30])
    phi = torch.atan2(v[..., 1], v[..., 2])/np.pi*180
    hopf = torch.stack([phi, theta], dim=-1)
    v = hopf_to_quat(hopf)

    #psi_quat = torch.stack([p2.cos(), zero, zero, p2.sin()], dim=-1)
    #print(quat_mul(psi_quat, v))
    hopf = torch.stack([phi, theta, psi], dim=-1)
    return hopf

def so3_to_hopf(R):
    v = SO3_to_quaternions_wiki(R)
    return quat_to_hopf(v)

def hopf_to_quat(euler):
    # euler should be in degrees
    euler = euler*np.pi/180
    if euler.shape[-1] == 3:
        phi = euler[..., 0]
        theta = euler[..., 1]*0.5
        psi = euler[..., 2]*0.5
        cphi = (phi - psi).cos()
        sphi = (phi - psi).sin()
        ctheta = theta.cos()
        stheta = theta.sin()
        cpsi = psi.cos()
        spsi = psi.sin()
        quat = torch.stack([ctheta*cpsi, stheta*sphi, cphi*stheta, ctheta*spsi], dim=-1)
    elif euler.shape[-1] == 2:
        zero = torch.zeros_like(euler[..., 0])
        phi = euler[..., 0]
        theta = euler[..., 1]*0.5
        cphi = phi.cos()
        sphi = phi.sin()
        ctheta = theta.cos()
        stheta = theta.sin()
        quat = torch.stack([ctheta, stheta*sphi, stheta*cphi, zero], dim=-1)
    else:
        raise Exception("wrong shape {}".format(euler.shape))
    return quat

def hopf_to_SO3(euler):
    euler = euler*np.pi/180
    if euler.shape[-1] == 3:
        phi = euler[..., 0]
        theta = euler[..., 1]*0.5
        psi = euler[..., 2]*0.5
        cphi = (phi - psi).cos()
        sphi = (phi - psi).sin()
        ctheta = theta.cos()
        stheta = theta.sin()
        cpsi = psi.cos()
        spsi = psi.sin()
        quat = torch.stack([ctheta*cpsi, stheta*sphi, cphi*stheta, ctheta*spsi], dim=-1)
        R = quaternions_to_SO3_wiki(quat)
    elif euler.shape[-1] == 2:
        zero = torch.zeros_like(euler[..., 0])
        phi = euler[..., 0]
        theta = euler[..., 1]*0.5
        cphi = phi.cos()
        sphi = phi.sin()
        ctheta = theta.cos()
        stheta = theta.sin()
        quat = torch.stack([ctheta, stheta*sphi, stheta*cphi, zero], dim=-1)
        R = quaternions_to_SO3_wiki(quat)
    elif euler.shape[-1] == 1:
        psi = euler[..., 0]*0.5
        zero = torch.zeros_like(euler[..., 0])
        psi_quat = torch.stack([psi.cos(), zero, zero, psi.sin()], dim=-1)
        R = quaternions_to_SO3_wiki(psi_quat)
    else:
        raise Exception("wrong shape {}".format(euler.shape))
    return R

FLT_EPSILON = np.single(1.19209e-07)

def so3_to_euler(A):
    abs_sb = torch.sqrt(A[..., 0, 2] * A[..., 0, 2] + A[..., 1, 2] * A[..., 1, 2])
    gamma = torch.atan2(A[..., 1, 2], -A[..., 0, 2])
    alpha = torch.atan2(A[..., 2, 1], A[..., 2, 0])
    sign_sb = torch.where(torch.sin(gamma) > 0, torch.sign(A[..., 1, 2]), -torch.sign(A[..., 1, 2]))
    sign_sb = torch.where(torch.abs(torch.sin(gamma)) < FLT_EPSILON, torch.sign(-A[..., 0, 2] / torch.cos(gamma)), sign_sb)
    beta = torch.atan2(sign_sb * abs_sb, A[..., 2, 2])

    alpha = torch.where(abs_sb > 16*FLT_EPSILON, alpha, torch.tensor(np.single(0.)))
    beta_tmp = torch.where(torch.sign(A[..., 2, 2]) > 0., np.single(0.), np.single(np.pi))
    gamma_tmp = torch.where(torch.sign(A[..., 2, 2]) > 0., torch.atan2(-A[..., 1, 0], A[..., 0, 0]), torch.atan2(A[..., 1, 0], -A[..., 0, 0]))
    gamma = torch.where(abs_sb > 16*FLT_EPSILON, gamma, gamma_tmp)
    beta = torch.where(abs_sb > 16*FLT_EPSILON, beta, beta_tmp)

    #if (abs_sb > 16*FLT_EPSILON):
    #    gamma = np.arctan2(A[..., 1, 2], -A[..., 0, 2])
    #    alpha = np.arctan2(A[..., 2, 1], A[..., 2, 0])
    #    if (np.abs(np.sin(gamma)) < FLT_EPSILON):
    #        sign_sb = np.sign(-A[..., 0, 2] / np.cos(gamma))
    #    # if (sin(alpha)<FLT_EPSILON) sign_sb=SGN(-A(0,2)/cos(gamma));
    #    # else sign_sb=(sin(alpha)>0) ? SGN(A(2,1)):-SGN(A(2,1));
    #    else:
    #        sign_sb = np.sign(A[..., 1, 2]) if (np.sin(gamma) > 0) else -np.sign(A[..., 1, 2])
    #    beta  = np.arctan2(sign_sb * abs_sb, A[..., 2, 2])
    #else:
    #    if (np.sign(A[..., 2, 2]) > 0):
    #        # Let's consider the matrix as a rotation around Z
    #        alpha = 0
    #        beta  = 0
    #        gamma = np.arctan2(-A[..., 1, 0], A[..., 0, 0])
    #    else:
    #        alpha = 0
    #        beta  = np.pi
    #        gamma = np.arctan2(A[..., 1, 0], -A[..., 0, 0])

    gamma = torch.rad2deg(gamma)
    beta  = torch.rad2deg(beta)
    alpha = torch.rad2deg(alpha)
    #return alpha, beta, gamma
    return torch.stack((alpha, beta, gamma), dim=-1)

def random_quaternions(n, dtype=torch.float32, device=None):
    u1, u2, u3 = torch.rand(3, n, dtype=dtype, device=device)
    return torch.stack((
        torch.sqrt(1-u1) * torch.sin(2 * np.pi * u2),
        torch.sqrt(1-u1) * torch.cos(2 * np.pi * u2),
        torch.sqrt(u1) * torch.sin(2 * np.pi * u3),
        torch.sqrt(u1) * torch.cos(2 * np.pi * u3),
    ), 1)

def random_biased_quaternions(n, bias=1., device=None):
    u = torch.randn(n, 3, device=device)
    one = torch.randn(n, 1, device=device).sign()*bias
    quat = torch.cat([one, u], dim=1)
    return quat

def random_biased_SO3(n, bias=1., device=None):
    rots = quaternions_to_SO3_wiki(random_biased_quaternions(n, bias, device))
    #print(rots @ rots.transpose(-1, -2))
    return rots

def random_SO3(n, dtype=torch.float32, device=None):
    return quaternions_to_SO3(random_quaternions(n, dtype, device))

def logsumexp(inputs, dim=None, keepdim=False):
    '''Numerically stable logsumexp.
    https://github.com/pytorch/pytorch/issues/2591

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    '''
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def so3_entropy_old(w_eps, std, k=10):
    '''
    w_eps(Tensor of dim 3): sample from so3
    covar(Tensor of dim 3x3): covariance of distribution on so3
    k: 2k+1 samples for truncated summation
    '''
    # entropy of gaussian distribution on so3
    # see appendix C of https://arxiv.org/pdf/1807.04689.pdf
    theta = w_eps.norm(p=2)
    u = w_eps/theta # 3
    angles = 2*np.pi*torch.arange(-k,k+1,dtype=w_eps.dtype,device=w_eps.device) # 2k+1
    theta_hat = theta + angles # 2k+1
    x = u[None,:] * theta_hat[:,None] # 2k+1 , 3
    log_p = Normal(torch.zeros(3,device=w_eps.device),std).log_prob(x) # 2k+1,3
    clamp = 1e-3
    log_vol = torch.log((theta_hat**2).clamp(min=clamp)/(2-2*torch.cos(theta)).clamp(min=clamp)) # 2k+1
    log_p = log_p.sum(-1) + log_vol
    entropy = -logsumexp(log_p)
    return entropy

def so3_entropy(w_eps, std, k=10):
    '''
    w_eps(Tensor of dim Bx3): sample from so3
    std(Tensor of dim Bx3): std of distribution on so3
    k: Use 2k+1 samples for truncated summation
    '''
    # entropy of gaussian distribution on so3
    # see appendix C of https://arxiv.org/pdf/1807.04689.pdf
    theta = w_eps.norm(p=2, dim=-1, keepdim=True) # [B, 1]
    u = w_eps/theta # [B, 3]
    angles = 2*np.pi*torch.arange(-k,k+1,dtype=w_eps.dtype,device=w_eps.device) # 2k+1
    theta_hat = theta[:, None, :] + angles[:, None] # [B, 2k+1, 1]
    x = u[:,None,:] * theta_hat # [B, 2k+1 , 3]
    log_p = Normal(torch.zeros(3,device=w_eps.device),std).log_prob(x.permute([1,0,2])) # [2k+1, B, 3]
    log_p = log_p.permute([1,0,2]) # [B, 2k+1, 3]
    clamp = 1e-3
    log_vol = torch.log((theta_hat**2).clamp(min=clamp)/(2-2*torch.cos(theta_hat)).clamp(min=clamp)) # [B, 2k+1, 1]
    log_p = log_p.sum(-1) + log_vol.sum(-1) #[B, 2k+1]
    entropy = -logsumexp(log_p, -1)
    return entropy


