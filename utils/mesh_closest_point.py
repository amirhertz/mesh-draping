from custom_types import *
from constants import EPSILON
from functools import reduce
import trimesh
from utils import mesh_utils
# import importlib.util
# if importlib.util.find_spec("trimesh") and importlib.util.find_spec("rtree"):
#     import trimesh
#     is_trimesh = True
# else:
#     is_trimesh = False


def vs_cp(vs: T, target_mesh: T_Mesh):
    target_mesh = mesh_utils.triangulate_mesh(target_mesh)[0]
    device = target_mesh[0].device
    mesh_trimesh = trimesh.Trimesh(vertices=target_mesh[0].detach().cpu().numpy(),
                                   faces=target_mesh[1].detach().cpu().numpy())
    cp, distances, _ = mesh_trimesh.nearest.on_surface(vs.detach().cpu().numpy())
    cp, distances = torch.from_numpy(cp).to(device, dtype=vs.dtype), torch.from_numpy(distances).to(device)
    return cp, distances


def mesh_cp(source_mesh: T_Mesh, target_mesh: T_Mesh) -> Tuple[T_Mesh, Optional[T]]:
    if type(target_mesh) is T or target_mesh[1] is None:
        return source_mesh, None
    cp, distances = vs_cp(source_mesh[0], target_mesh)
    return (cp, source_mesh[1]), distances


def mesh_cp_ugly(mesh: T_Mesh, points: T) -> T:

    def multi_and(*masks):
        return reduce(lambda x, y: x * y, masks)

    def expand_as(a, b):
        return a.repeat(b.shape[1]).reshape(-1, b.shape[1])

    vs, faces = mesh
    device = points.device
    triangles = vs[faces]
    distances = torch.zeros(faces.shape[0], points.shape[0], device=device)
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    edge_d = triangles[:, 0, None, :] - points[None, :, :]
    dot_a = expand_as(torch.einsum('fd,fd->f', edge_a, edge_a), distances)
    dot_b = expand_as(torch.einsum('fd,fd->f', edge_b, edge_b), distances)
    dot_d = torch.einsum('fpd,fpd->fp', edge_d, edge_d)
    dot_ab = expand_as(torch.einsum('fd,fd->f', edge_a, edge_b), distances)
    dot_da = torch.einsum('fd,fpd->fp', edge_a, edge_d)
    dot_db = torch.einsum('fd,fpd->fp', edge_b, edge_d)
    det = dot_a * dot_b - dot_ab * dot_ab
    det[det.eq(0)] = EPSILON
    inv_det = det ** -1
    s = dot_ab * dot_db - dot_b * dot_da
    t = dot_ab * dot_da - dot_a * dot_db
    mask_st = (s + t - det).le(0)
    mask_s = s.lt(0)
    mask_t = t.lt(0)
    mask_da = dot_da.lt(0)
    mask_db = dot_db.lt(0)
    mask_daa = (- dot_da - dot_a).ge(0)
    mask_dbb = (- dot_db - dot_b).ge(0)
    val_a = dot_a + 2 * dot_da
    val_b = dot_b + 2 * dot_db
    dot_a_f = dot_a.clone()
    dot_a_f[dot_a_f.eq(0)] = EPSILON
    dot_b_f = dot_b.clone()
    dot_b_f[dot_b_f.eq(0)] = EPSILON
    val_da = - dot_da * dot_da / dot_a_f
    val_db = - dot_db * dot_db / dot_b_f
    ma = multi_and(mask_st, mask_s, mask_t, mask_da, mask_daa)
    distances[ma] = val_a[ma]
    ma = multi_and(mask_st, mask_s, mask_t, mask_da, ~mask_daa)
    distances[ma] = val_da[ma]
    ma = multi_and(mask_st, mask_s, mask_t, ~mask_da, mask_db, mask_dbb)
    distances[ma] = val_b[ma]
    ma = multi_and(mask_st, mask_s, mask_t, ~mask_da, mask_db, ~mask_dbb)
    distances[ma] = val_db[ma]

    ma = multi_and(mask_st, mask_s, ~mask_t, mask_db, mask_dbb)
    distances[ma] = dot_b[ma] + 2 * dot_db[ma]
    ma = multi_and(mask_st, mask_s, ~mask_t, mask_db, ~mask_dbb)
    distances[ma] = val_db[ma]

    ma = multi_and(mask_st, ~mask_s, mask_t, ~mask_da)
    distances[ma] = dot_d[ma]
    ma = multi_and(mask_st, ~mask_s, mask_t, mask_da, mask_daa)
    distances[ma] = val_a[ma]
    ma = multi_and(mask_st, ~mask_s, mask_t, mask_da, ~mask_daa)
    distances[ma] = val_da[ma]
    ma = multi_and(mask_st, ~mask_s, ~mask_t)
    distances[ma] = s[ma] * inv_det[ma] * (dot_a[ma] * s[ma] * inv_det[ma] + dot_ab[ma] * t[ma] * inv_det[ma] + 2 * dot_da[ma])\
                    + t[ma] * inv_det[ma] * (dot_ab[ma] * s[ma] * inv_det[ma] + dot_b[ma] * t[ma] * inv_det[ma] + 2 * dot_db[ma])

    mask_eg = (dot_ab + dot_da - dot_b - dot_db).lt(0)
    mask_eg_b = dot_b + dot_db - dot_ab - dot_da >= dot_a - 2 * dot_ab + dot_b
    ma = multi_and(~mask_st, mask_s, mask_eg, mask_eg_b)
    distances[ma] = val_a[ma]
    ma = multi_and(~mask_st, mask_s, mask_eg, ~mask_eg_b)
    s_tmp = (dot_b[ma] + dot_db[ma] - dot_ab[ma] - dot_da[ma]) / (dot_a[ma] - 2 * dot_ab[ma] + dot_b[ma])
    t_tmp = 1 - s_tmp
    distances[ma] = s_tmp * (dot_a[ma] * s_tmp + dot_ab[ma] * t_tmp + 2 * dot_da[ma]) + \
                    t_tmp * (dot_ab[ma] * s_tmp + dot_b[ma] * t_tmp + 2 * dot_db[ma])

    mask_eg_b = (dot_b + dot_db).le(0)
    mask_eg_c = dot_db.lt(0)
    ma = multi_and(~mask_st, mask_s, ~mask_eg, mask_eg_b)
    distances[ma] = val_b[ma]
    ma = multi_and(~mask_st, mask_s, ~mask_eg, ~mask_eg_b, mask_eg_c)
    distances[ma] = val_db[ma]

    mask_eg = (dot_a + dot_da - dot_ab - dot_db).gt(0)
    mask_eg_b = dot_a + dot_da - dot_ab - dot_db > dot_a - 2 * dot_ab + dot_b
    ma = multi_and(~mask_st, ~mask_s, mask_t, mask_eg, mask_eg_b)
    distances[ma] = val_b[ma]
    ma = multi_and(~mask_st, ~mask_s, mask_t, mask_eg, ~mask_eg_b)
    t_tmp = (dot_a[ma] + dot_da[ma] - dot_ab[ma] - dot_db[ma]) / (dot_a[ma] - 2 * dot_ab[ma] + dot_b[ma])
    s_tmp = 1 - t_tmp
    distances[ma] = s_tmp * (dot_a[ma] * s_tmp + dot_ab[ma] * t_tmp + 2 * dot_da[ma]) + \
                    t_tmp * (dot_ab[ma] * s_tmp + dot_b[ma] * t_tmp + 2 * dot_db[ma])
    mask_eg_b = (dot_a + dot_da).le(0)
    mask_eg_c = dot_da.lt(0)
    ma = multi_and(~mask_st, ~mask_s, mask_t,~mask_eg, mask_eg_b)
    distances[ma] = val_a[ma]
    ma = multi_and(~mask_st, ~mask_s, mask_t, ~mask_eg, ~mask_eg_b, mask_eg_c)
    distances[ma] = val_da[ma]
    numer = dot_b + dot_db - dot_ab - dot_da
    denom = dot_a - 2 * dot_ab + dot_b
    denom[denom == 0] = EPSILON
    ma = multi_and(~mask_st, ~mask_s, ~mask_t, numer <= 0)
    distances[ma] = val_b[ma]
    ma = multi_and(~mask_st, ~mask_s, ~mask_t, numer > 0, numer >= denom)
    distances[ma] = val_a[ma]
    ma = multi_and(~mask_st, ~mask_s, ~mask_t, numer > 0, numer < denom)
    s = (numer / denom)[ma]
    t = 1 - s
    distances[ma] = s * (dot_a[ma] * s + dot_ab[ma] * t + 2 * dot_da[ma]) +\
                    t * (dot_ab[ma] * s + dot_b[ma] * t + 2 * dot_db[ma])
    distances = (distances + dot_d).min(0)[0]
    distances[distances.lt(0)] = 0
    return np.sqrt(distances)

