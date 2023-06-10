from custom_types import *
from functools import reduce
import transforms3d

def get_random_rotation(one_axis=False, max_angle=-1, required_dim=3) -> Tuple[T, float, T]:
    theta = torch.rand(1).item()
    if max_angle > 0:
        if theta > .5:
            theta = 1. - (theta - .5) * max_angle / 180
        else:
            theta = theta * max_angle / 180
    r = torch.zeros((required_dim, required_dim), dtype=torch.float32)
    if one_axis:
        axis = torch.zeros(required_dim, dtype=torch.float32)
        axis[-1] = 1.
    else:
        axis = torch.randn(required_dim, dtype=torch.float32)
        axis = axis / axis.norm(2, 0)
    cos_a = float(np.cos(np.pi * theta))
    sin_a = float(np.sin(np.pi * theta))
    q = torch.tensor([cos_a] + list(sin_a * axis), dtype=torch.float32)
    q = q[:, None] * q[None, :]
    trace = q.trace().item()
    for i in range(required_dim):
        r[i, i] = .5 - trace + q[i, i]
    for i in range(required_dim):
        j = (i + 1) % required_dim
        k = i if i > 0 else -1
        r[i, j] = q[j + 1, i + 1] - q[k, 0]
        r[j, i] = q[j + 1, i + 1] + q[k, 0]
        if one_axis: break
    r *= 2
    return r, theta, axis


def vs_to_affine(vs: T, required_dim=3) -> T:
    if vs.shape[-1] == required_dim:
        return torch.cat([vs, torch.ones(*vs.shape[:-1], 1, device=vs.device, dtype=vs.dtype)], dim=-1)
    return vs


def to_affine(transformation: T, required_dim=3) -> T:
    if transformation.dim() >= 2 and transformation.shape[-1] == required_dim + 1:
        return transformation
    affine = torch.eye(required_dim + 1)
    if transformation.dim() == 2:
        affine[:required_dim, :required_dim] = transformation
    else:
        affine[:required_dim, required_dim] = transformation
    return affine


def from_affine(affine: T, required_dim=3) -> Tuple[T, T]:
    affine = to_affine(affine, required_dim)
    rot = affine[:required_dim, :required_dim]
    translate = affine[:required_dim, required_dim]
    return rot, translate


def combine_affines(*affines: T, required_dim=3) -> T:
    affines = [to_affine(affine, required_dim) for affine in affines]
    return reduce(torch.matmul, affines)


def apply_affine(affine: T, vs: T, required_dim=3) -> T:
    dim = vs.shape[-1]
    should_reduce = dim == required_dim
    affine = to_affine(affine, required_dim)
    if should_reduce:
        vs = vs_to_affine(vs)
    if affine.dim() == 3:
        vs_transformed = torch.einsum('nad,nd->na', [affine, vs])
    else:
        vs_transformed = torch.einsum('ad,nd->na', [affine, vs])
    if should_reduce:
        vs_transformed = vs_transformed[:, :dim]
    return vs_transformed


def vs_to_mat(vs: T) -> T:
    num_mats, num_vs, dim = vs.shape
    vs_affine = vs_to_affine(vs)
    blocks = [torch.zeros(num_mats, num_vs * dim, dim + 1,  device=vs.device, dtype=vs.dtype) for _ in range(dim)]
    arange = torch.arange(num_vs, device=vs.device) * dim
    for i in range(dim):
        blocks[i][:, arange + i] = vs_affine
    return torch.cat(blocks, dim=-1)


def find_affine(vs_source: T, vs_target: T) -> T:
    device = vs_source.device
    should_reduce = vs_source.dim() == 2
    if should_reduce:
        vs_source, vs_target = vs_source.unsqueeze(0), vs_target.unsqueeze(0)
    num_groups, _, dim = vs_source.shape
    assert vs_source.shape[1] > dim and vs_source.shape == vs_target.shape
    vs_source_mat = vs_to_mat(vs_source)
    affine = vs_source_mat[0].pinverse().unsqueeze(0).matmul(vs_target.view(num_groups, -1, 1))
    affine = affine.view(num_groups, dim, dim + 1)
    affine_row = torch.zeros(num_groups, 1, dim + 1, device=device, dtype=vs_source.dtype)
    affine_row[:, 0, -1] = 1
    affine = torch.cat((affine, affine_row), dim=1)
    if should_reduce:
        affine = affine.squeeze(0)
    return affine


def decompose_affines(affines: T):
    should_reduce = affines.dim() == 2
    if should_reduce:
        affines = affines.unsqueeze(0)
    translation = affines[:, :-1, -1]
    affines = affines[:, :-1, :-1]
    scale = affines.norm(2, dim=1)
    affines = affines / scale[:, None, :]
    return translation, affines, scale, None


def decompose_affines_t3(affines: T) -> TS:
    device, dtype = affines.device, affines.dtype
    affines = affines.cpu().detach().numpy()
    transformations = [[] for _ in range(4)]
    for i in range(affines.shape[0]):
        decomposition = transforms3d.affines.decompose44(affines[i])
        for item, lst in zip(decomposition, transformations):
            lst.append(item)
    transformations[1] = [transforms3d.quaternions.mat2quat(r) for r in transformations[1]]
    transformations = map(torch.from_numpy, map(np.stack, transformations))
    translation, quaternions, scale, shear = tuple(map(lambda t: t.to(device, ), transformations))
    return translation, quaternions, scale, shear


def slerp(q_1: T, q_2: T, t: T) -> T:
    dot = (q_1 * q_2).sum(1)
    ma0 = torch.lt(t, 0.01)
    ma1 = torch.gt(t, .99) + torch.gt(dot, .99)
    q_res = torch.zeros_like(q_1)
    q_res[ma0] = q_1[ma0]
    q_res[ma1] = q_2[ma1]
    ma = ~(ma0 + ma1)
    q_1, q_2, t, dot = q_1[ma], q_2[ma], t[ma], dot[ma]
    ma0 = dot < 0
    dot[ma0] = -dot[ma0]
    q_2[ma0] = -q_2[ma0]
    assert (dot >= 0).all()
    omega = torch.acos(dot)
    so = torch.sin(omega)
    q_res[ma] = (torch.sin((- t + 1.0)*omega) / so).unsqueeze(1) * q_1 + (torch.sin(t * omega) / so).unsqueeze(1) * q_2
    return q_res


# def slerp(v0, v1, t_array):
#     """Spherical linear interpolation."""
#     # >>> slerp([1,0,0,0], [0,0,0,1], np.arange(0, 1, 0.001))
#     t_array = np.array(t_array)
#     v0 = np.array(v0)
#     v1 = np.array(v1)
#     dot = np.sum(v0 * v1)
#
#     if dot < 0.0:
#         v1 = -v1
#         dot = -dot
#
#     DOT_THRESHOLD = 0.9995
#     if dot > DOT_THRESHOLD:
#         result = v0[np.newaxis, :] + t_array[:, np.newaxis] * (v1 - v0)[np.newaxis, :]
#         return (result.T / np.linalg.norm(result, axis=1)).T
#
#     theta_0 = np.arccos(dot)
#     sin_theta_0 = np.sin(theta_0)
#
#     theta = theta_0 * t_array
#     sin_theta = np.sin(theta)
#
#     s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
#     s1 = sin_theta / sin_theta_0
#     return (s0[:, np.newaxis] * v0[np.newaxis, :]) + (s1[:, np.newaxis] * v1[np.newaxis, :])

def interpolate_quaternions(quaternions, weights, select: Union[None, T] = None) -> T:
    if select is None:
        quaternions = torch.einsum('wd,nw->nd', [quaternions, weights])
    else:
        quaternions = torch.einsum('nwd,nw->nd', [quaternions[select], weights])
    q = quaternions / quaternions.norm(2, 1)[:, None]
    # q = quaternions[:, 0]
    # cur_w = weights[:, 0]
    # for i in range(weights.shape[1] - 1):
    #     w = cur_w / (cur_w + weights[:, i + 1])
    #     cur_w += weights[:, i + 1]
    #     q = slerp(q, quaternions[:, i + 1], 1- w)
    return q


def quat2rot(q: T) -> T:
    r = torch.zeros(q.shape[0], 3, 3, dtype=q.dtype)
    r[:, 0, 0] = - 2 *( q[:, 2] ** 2 + q[:, 3] ** 2) + 1
    r[:, 1, 1] = - 2 * (q[:, 1] ** 2 + q[:, 3] ** 2) + 1
    r[:, 2, 2] = - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2) + 1
    r[:, 0, 1] = 2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
    r[:, 0, 2] = 2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2])
    r[:, 1, 0] = 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
    r[:, 1, 2] = 2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])
    r[:, 2, 0] = 2 * (q[:, 1] * q[:, 3] - q[:, 2] * q[:, 0])
    r[:, 2, 1] = 2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
    return r


def interpolate(weights: T, affines: T, select: Union[None, T] = None) -> T:
    translation, quaternions, scale, shear = decompose_affines_t3(affines)
    quaternions = interpolate_quaternions(quaternions, weights, select)
    rotation = quat2rot(quaternions)
    if select is None:
        translation = torch.einsum('wd,nw->nd', [translation, weights])
        scale = torch.einsum('wd,nw->nd', [scale, weights])
        shear = torch.einsum('wd,nw->nd', [shear, weights])
    else:
        translation = torch.einsum('nwd,nw->nd', [translation[select], weights])
        scale = torch.einsum('nwd,nw->nd', [scale[select], weights])
        shear = torch.einsum('nwd,nw->nd', [shear[select], weights])
    shear_ = torch.eye(3, dtype=shear.dtype).unsqueeze(0).repeat(shear.shape[0], 1, 1)
    shear_[:, 0, 1:] = shear[:, :2]
    shear_[:, 1, 2] = shear[:, 2]
    affines = torch.einsum('nab, nb, nbc->nac', [rotation, scale, shear_])
    affines = torch.cat((affines, translation[:, :, None]), dim=2)
    pad = torch.zeros(affines.shape[0], 1, 4, dtype=affines.dtype)
    pad[:, :, 3] = 1
    affines = torch.cat((affines, pad), dim=1)
    return affines


def sanity():
    torch.manual_seed(0)
    np.random.seed(0)

    pt_a = torch.rand(3)
    q = torch.rand(1, 4, 4)
    q = q / q.norm(2, 1)[:, None]
    # t = (torch.arange(10, dtype= torch.float32).unsqueeze(0) + 1) / 11
    q_0, q_1, q_2 = q[:, 0], q[:, 1], q[:, 2]
    # q_mid = [q_0] + [slerp(q_0, q_1, t[:, i]) for i in range(len(t))] + [q_1]

    t = torch.tensor(
        [[(i) / 10., (1 - (i) / 10.) / 2, (1 - (i) / 10.) / 2] for i in range(10)]).unsqueeze(0)

    q_mid = [q_0] + [q_1] + [q_2] + [interpolate_quaternions(q[:, :3], t[:, i]) for i in range(t.shape[1])]

    rot = [quat2rot(q)[0] for q in q_mid]
    pts = [torch.einsum('ad,d->a', [r, pt_a]) for r in rot]
    pts = torch.stack(pts, dim=0)
    colors = torch.zeros_like(pts)
    colors[0] = torch.tensor([1, .5, 0])
    colors[1] = torch.tensor([0, 1, 0])
    colors[2] = torch.tensor([0, 0, 1])
    files_utils.export_mesh(pts, 'quat_check', colors=colors)


if __name__ == '__main__':
    from utils import mesh_utils, files_utils
    sanity()
