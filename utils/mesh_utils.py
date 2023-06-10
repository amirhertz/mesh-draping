from custom_types import *
from constants import EPSILON
import igl
from custom_types import T, TS
from functools import reduce
from scipy.spatial import ConvexHull


def get_random_rotation(one_axis=False, max_angle=-1):

    theta = torch.rand(1)
    if max_angle > 0:
        if theta > .5:
            theta = 1 - (theta - .5) * max_angle / 180
        else:
            theta = theta * max_angle / 180
    r = torch.zeros((3, 3))
    if one_axis:
        axis = torch.tensor([0, 0, 1.])
    else:
        axis = torch.randn(3)
        axis = axis / axis.norm(2, 0)
    cos_a = torch.cos(np.pi * theta)
    sin_a = torch.sin(np.pi * theta)
    q = list(cos_a) + list(sin_a * axis)
    q = [[q[i] * q[j] for i in range(j + 1)] for j in range(len(q))]
    r[0, 0] = .5 - q[2][2] - q[3][3]
    r[1, 1] = .5 - q[1][1] - q[3][3]
    r[2, 2] = .5 - q[1][1] - q[2][2]
    r[0, 1] = q[2][1] - q[3][0]
    r[1, 0] = q[2][1] + q[3][0]
    if not one_axis:
        r[0, 2] = q[3][1] + q[2][0]
        r[2, 0] = q[3][1] - q[2][0]
        r[1, 2] = q[3][2] - q[1][0]
        r[2, 1] = q[3][2] + q[1][0]
    r *= 2
    return r


def transform_rotation(points: T, one_axis=False, max_angle=-1):
    r = get_random_rotation(one_axis, max_angle)
    transformed = torch.einsum('nd,rd->nr', points, r)
    return transformed


def split_mesh_side(mesh: T_Mesh, should_stay: float):
    vs, faces = mesh
    face_center = (vs[faces[:, 0]] + vs[faces[:, 1]] + vs[faces[:, 2]]) / 3
    face_areas = compute_face_areas(mesh)[0]
    total_area = face_areas.sum()
    mask = torch.zeros(faces.shape[0], dtype=torch.bool)
    target = total_area * should_stay
    face_center_z = transform_rotation(face_center)[:, 2]
    faces_order = np.argsort(face_center_z)
    for face_idx in faces_order:
        mask[face_idx] = True
        target -= face_areas[face_idx]
        if target <= 0:
            break
    vs_idx = faces[mask].flatten().unique()
    return vs_idx


def scale_all(*values: T):
    # mean_std = [(val.mean(), val.std()) for val in values]
    # values = [val.clamp(scales[0] - scales[1] * 3, scales[0] + scales[1] * 3) for val,scales in zip(values, mean_std)]
    max_val = max([val.max().item() for val in values])
    min_val = min([val.min().item() for val in values])
    scale = max_val - min_val
    values = [(val - min_val) / scale for val in values]
    if len(values) == 1:
        return values[0]
    return values


def get_faces_normals(mesh: Union[T_Mesh, T]) -> T:
    if type(mesh) is not T:
        vs, faces = mesh
        vs_faces = vs[faces]
    else:
        vs_faces = mesh
    if vs_faces.shape[-1] == 2:
        vs_faces = torch.cat(
            (vs_faces, torch.zeros(*vs_faces.shape[:2], 1, dtype=vs_faces.dtype, device=vs_faces.device)), dim=2)
    face_normals = torch.cross(vs_faces[:, 1, :] - vs_faces[:, 0, :], vs_faces[:, 2, :] - vs_faces[:, 1, :])
    return face_normals


def compute_face_areas(mesh: Union[T_Mesh, T]) -> TS:
    face_normals = get_faces_normals(mesh)
    face_areas = torch.norm(face_normals, p=2, dim=1)
    face_areas_ = face_areas.clone()
    face_areas_[torch.eq(face_areas_, 0)] = 1
    face_normals = face_normals / face_areas_[:, None]
    face_areas = 0.5 * face_areas
    return face_areas, face_normals


def check_sign_area(*meshes: T_Mesh) -> bool:
    for mesh in meshes:
        face_normals = get_faces_normals(mesh)
        if not face_normals[:, 2].gt(0).all():
            return False
    return True


def to_numpy(*tensors: T) -> ARRAYS:
    params = [param.detach().cpu().numpy() if type(param) is T else param for param in tensors]
    return params


def create_mapper(mask: T) -> T:
    mapper = torch.zeros(mask.shape[0], dtype=torch.int64, device=mask.device) - 1
    mapper[mask] = torch.arange(mask.sum().item(), device=mask.device)
    return mapper


def mesh_center(mesh: T_Mesh):
    return mesh[0].mean(0)


def to_unit_sphere(mesh: T_Mesh_T, scale=1, in_place: bool = True) -> Tuple[T_Mesh_T, TS]:
    remove_me = 0
    vs, faces = (mesh, remove_me) if type(mesh) is T else mesh
    if not in_place:
        vs = vs.clone()
    max_vals = vs.max(0)[0]
    min_vals = vs.min(0)[0]
    center = (max_vals + min_vals) / 2
    vs -= center[None, :]
    radius = torch.sqrt((vs ** 2).sum(1).max())
    vs = vs * scale / radius
    return vs if faces is remove_me else (vs, faces), (center, scale / radius)


def to_unit_cube(*meshes: T_Mesh_T, scale=1, in_place: bool = True) -> Tuple[Union[T_Mesh_T, Tuple[T_Mesh_T, ...]], TS]:
    remove_me = 0
    meshes = [(mesh, remove_me) if type(mesh) is T else mesh for mesh in meshes]
    vs, faces = meshes[0]
    max_vals = vs.max(0)[0]
    min_vals = vs.min(0)[0]
    max_range = (max_vals - min_vals).max() / 2
    center = (max_vals + min_vals) / 2
    meshes_ = []
    for vs_, faces_ in meshes:
        if not in_place:
            vs_ = vs_.clone()
        vs_ -= center[None, :]
        vs_ /= max_range
        vs_ *= scale
        meshes_.append(vs_ if faces_ is remove_me else (vs_, faces_))
    if len(meshes_) == 1:
        meshes_ = meshes_[0]
    return meshes_, (center, max_range / scale)
# # in place
# def to_unit_edge(*meshes: T_Mesh) -> Tuple[Union[T_Mesh, Tuple[T_Mesh, ...]], Tuple[T, float]]:
#     ref = meshes[0]
#     center = ref[0].mean(0)
#     ratio = edge_lengths(ref).mean().item()
#     for mesh in meshes:
#         vs, _ = mesh
#         vs -= center[None, :].to(vs.device)
#         vs /= ratio
#     if len(meshes) == 1:
#         meshes = meshes[0]
#     return meshes, (center, ratio)


def get_edges_ind(mesh: T_Mesh) -> T:
    vs, faces = mesh
    raw_edges = torch.cat([faces[:, [i, (i + 1) % 3]] for i in range(3)]).sort()
    raw_edges = raw_edges[0].cpu().numpy()
    edges = {(int(edge[0]), int(edge[1])) for edge in raw_edges}
    edges = torch.tensor(list(edges), dtype=torch.int64, device=faces.device)
    return edges


def edge_lengths(mesh: T_Mesh, edges_ind: TN = None) -> T:
    vs, faces = mesh
    if edges_ind is None:
        edges_ind = get_edges_ind(mesh)
    edges = vs[edges_ind]
    return torch.norm(edges[:, 0] - edges[:, 1], 2, dim=1)


# in place
def to_unit_edge(*meshes: T_Mesh) -> Tuple[Union[T_Mesh, Tuple[T_Mesh, ...]], Tuple[T, float]]:
    ref = meshes[0]
    center = ref[0].mean(0)
    ratio = edge_lengths(ref).mean().item()
    for mesh in meshes:
        vs, _ = mesh
        vs -= center[None, :].to(vs.device)
        vs /= ratio
    if len(meshes) == 1:
        meshes = meshes[0]
    return meshes, (center, ratio)


def to(tensors, device: D) -> Union[T_Mesh, TS, T]:
    out = []
    for tensor in tensors:
        if type(tensor) is T:
            out.append(tensor.to(device, ))
        elif type(tensor) is tuple or type(tensors) is List:
            out.append(to(list(tensor), device))
        else:
            out.append(tensor)
    if len(tensors) == 1:
        return out[0]
    else:
        return tuple(out)


def clone(*tensors: Union[T, TS]) -> Union[TS, T_Mesh]:
    out = []
    for t in tensors:
        if type(t) is T:
            out.append(t.clone())
        else:
            out.append(clone(*t))
    return out


def gaussian_curvature(mesh: T_Mesh) -> T:
    device, dtype = mesh[0].device, mesh[0].dtype
    vs, faces = to_numpy(*mesh)
    gc = igl.gaussian_curvature(vs, faces)
    return torch.from_numpy(gc).to(device, dtype=dtype)


def principal_curvature(mesh: T_Mesh) -> TS:
    device, dtype = mesh[0].device, mesh[0].dtype
    vs, faces = to_numpy(*mesh)
    out = igl.principal_curvature(vs, faces)
    min_dir, max_dir, min_val, max_val = [torch.from_numpy(value).to(device, dtype=dtype) for value in out]
    return max_dir, min_dir, max_val, min_val


def get_box(w: float, h: float, d: float) -> T_Mesh:
    vs = [[0, 0, 0], [w, 0, 0], [0, d, 0], [w, d, 0],
          [0, 0, h], [w, 0, h], [0, d, h], [w, d, h]]
    faces = [[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6],
             [0, 1, 5], [0, 5, 4], [2, 6, 7], [3, 2, 7],
             [1, 3, 5], [3, 7, 5], [0, 4, 2], [2, 4, 6]]
    return torch.tensor(vs, dtype=torch.float32), torch.tensor(faces, dtype=torch.int64)


def exact_geodesic(mesh, gamma):
    vs, faces = mesh
    device, dtype = vs.device, vs.dtype
    v_source = torch.arange(vs.shape[0]).unsqueeze(1)
    params = to_numpy(vs, faces,  gamma.unsqueeze(1), v_source)
    geodesic = igl.exact_geodesic(*params)
    return torch.from_numpy(geodesic).to(device, dtype=dtype)


def heat_geodesic(mesh: T_Mesh, gamma: T):
    params = to_numpy(*mesh, gamma)
    geodesic = igl.heat_geodesic(*params[:2], .01, params[2])
    return torch.from_numpy(geodesic).float()


def vs_normals(mesh: T_Mesh) -> T:
    params = to_numpy(*mesh)
    normals = igl.per_vertex_normals(*params)
    return torch.from_numpy(normals).float()


def normalize(t: T):
    t = t / t.norm(2, dim=1)[:, None]
    return t


def interpulate_vs(mesh: T_Mesh, faces_inds: T, weights: T) -> T:
    vs = mesh[0][mesh[1][faces_inds]]
    vs = vs * weights[:, :, None]
    return vs.sum(1)


def sample_uvw(shape, device: D):
    u, v = torch.rand(*shape, device=device), torch.rand(*shape, device=device)
    mask = (u + v).gt(1)
    u[mask], v[mask] = -u[mask] + 1, -v[mask] + 1
    w = -u - v + 1
    uvw = torch.stack([u, v, w], dim=len(shape))
    return uvw


def get_sampled_fe(fe: T, mesh: T_Mesh, face_ids: T, uvw: TN) -> T:
    # to_squeeze =
    if fe.dim() == 1:
        fe = fe.unsqueeze(1)
    if uvw is None:
        fe_iner = fe[face_ids]
    else:
        vs_ids = mesh[1][face_ids]
        fe_unrolled = fe[vs_ids]
        fe_iner = torch.einsum('sad,sa->sd', fe_unrolled, uvw)
    # if to_squeeze:
    #     fe_iner = fe_iner.squeeze_(1)
    return fe_iner


def sample_on_faces(mesh: T_Mesh,  num_samples: int) -> TS:
    vs, faces = mesh
    uvw = sample_uvw([faces.shape[0], num_samples], vs.device)
    samples = torch.einsum('fad,fna->fnd', vs[faces], uvw)
    return samples, uvw


class SampleBy(Enum):
    AREAS = 0
    FACES = 1
    HYB = 2


def sample_on_mesh(mesh: T_Mesh, num_samples: int, face_areas: TN = None,
                   sample_s: SampleBy = SampleBy.AREAS) -> TNS:
    vs, faces = mesh
    if faces is None:  # sample from pc
        uvw = None
        if vs.shape[0] < num_samples:
            chosen_faces_inds = torch.arange(vs.shape[0])
        else:
            chosen_faces_inds = torch.argsort(torch.rand(vs.shape[0]))[:num_samples]
        samples = vs[chosen_faces_inds]
    else:
        weighted_p = []
        if sample_s == SampleBy.AREAS or sample_s == SampleBy.HYB:
            if face_areas is None:
                face_areas, _ = compute_face_areas(mesh)
            face_areas[torch.isnan(face_areas)] = 0
            weighted_p.append(face_areas / face_areas.sum())
        if sample_s == SampleBy.FACES or sample_s == SampleBy.HYB:
            weighted_p.append(torch.ones(mesh[1].shape[0], device=mesh[0].device))
        chosen_faces_inds = [torch.multinomial(weights, num_samples // len(weighted_p), replacement=True) for weights in weighted_p]
        if sample_s == SampleBy.HYB:
            chosen_faces_inds = torch.cat(chosen_faces_inds, dim=0)
        chosen_faces = faces[chosen_faces_inds]
        uvw = sample_uvw([num_samples], vs.device)
        samples = torch.einsum('sf,sfd->sd', uvw, vs[chosen_faces])
    return samples, chosen_faces_inds, uvw


def get_samples(mesh: T_Mesh, num_samples: int, sample_s: SampleBy, *features: T) -> Union[T, TS]:
    samples, face_ids, uvw = sample_on_mesh(mesh, num_samples, sample_s=sample_s)
    if len(features) > 0:
        samples = [samples] + [get_sampled_fe(fe, mesh, face_ids, uvw) for fe in features]
    return samples, face_ids, uvw


def find_barycentric(vs: T, triangles: T) -> T:

    def compute_barycentric(ind):
        triangles[:, ind] = vs
        alpha = compute_face_areas(triangles)[0] / areas
        triangles[:, ind] = recover[:, ind]
        return alpha

    device, dtype = vs.device, vs.dtype
    vs = vs.to(device, dtype=torch.float64)
    triangles = triangles.to(device, dtype=torch.float64)
    areas, _ = compute_face_areas(triangles)
    recover = triangles.clone()
    barycentric = [compute_barycentric(i) for i in range(3)]
    barycentric = torch.stack(barycentric, dim=1)
    # assert barycentric.sum(1).max().item() <= 1 + EPSILON
    return barycentric.to(device, dtype=dtype)


def from_barycentric(mesh: Union[T_Mesh, T], face_ids: T, weights: T) -> T:
    if type(mesh) is not T:
        triangles: T = mesh[0][mesh[1]]
    else:
        triangles: T = mesh
    to_squeeze = weights.dim() == 1
    if to_squeeze:
        weights = weights.unsqueeze(0)
        face_ids = face_ids.unsqueeze(0)
    vs = torch.einsum('nad,na->nd', triangles[face_ids], weights)
    if to_squeeze:
        vs = vs.squeeze(0)
    return vs


def lscm(mesh: T_Mesh, boundary_indices: T, boundary_coordinates: T) -> T:
    vs, faces = mesh
    np_param = tuple(map(lambda x: x.numpy(), [vs, faces, boundary_indices, boundary_coordinates]))
    check, uv = igl.lscm(*np_param)
    return torch.from_numpy(uv).float()


def check_circle_angles(mesh: T_Mesh, center_ind: int, select: T) -> bool:
    vs, _ = mesh
    all_vecs = vs[select] - vs[center_ind][None, :]
    all_vecs = all_vecs / all_vecs.norm(2, 1)[:, None]
    all_vecs = torch.cat([all_vecs, all_vecs[:1]], dim=0)
    all_cos = torch.einsum('nd,nd->n', all_vecs[1:], all_vecs[:-1])
    all_angles = torch.acos_(all_cos)
    all_angles = all_angles.sum()
    return (all_angles - 2 * np.pi).abs() < EPSILON


def vs_over_triangle(vs_mid: T, triangle: T, normals=None) -> T:
    if vs_mid.dim() == 1:
        vs_mid = vs_mid.unsqueeze(0)
        triangle = triangle.unsqueeze(0)
    if normals is None:
        _, normals = compute_face_areas(triangle)
    select = torch.arange(3)
    d_vs = vs_mid[:, None, :] - triangle
    d_f = triangle[:, select] - triangle[:, (select + 1) % 3]
    all_cross = torch.cross(d_vs, d_f, dim=2)
    all_dots = torch.einsum('nd,nad->na', normals, all_cross)
    is_over = all_dots.ge(0).long().sum(1).eq(3)
    return is_over


def f2v(num_faces: int, genus: int = 0) -> int:  # assuming there are not boundaries
    return num_faces // 2 + (1 - genus) * 2


def v2f(num_vs: int, genus: int = 0) -> int:  # assuming there are not boundaries
    return 2 * num_vs - 4 + 4 * genus


def get_dist_mat(a: T, b: T, batch_size: int = 1000, sqrt: bool = False) -> T:
    """
       :param a:
       :param b:
       :param batch_size: Limit batches per distance calculation to avoid out-of-mem
       :return:
       """
    iters = a.shape[0] // batch_size
    dist_list = [((a[i * batch_size: (i + 1) * batch_size, None, :] - b[None, :, :]) ** 2).sum(-1)
                 for i in range(iters + 1)]
    all_dist: T = torch.cat(dist_list, dim=0)
    if sqrt:
        all_dist = all_dist.sqrt_()
    return all_dist


def naive_knn(k: int, dist_mat: T, is_biknn=True):
    """
    :param k:
    :param dist_mat:
    :param is_biknn: When false, calcluates only closest element in a per element of b.
                     When true, calcluates only closest element in a <--> b both ways.
    :param batch_size: Limit batches per distance calculation to avoid out-of-mem
    :return:
    """
    _, close_to_b = dist_mat.topk(k, 0, largest=False)
    if is_biknn:
        _, close_to_a = dist_mat.topk(k, 1, largest=False)
        return close_to_a, close_to_b.t()
    return close_to_b.t()


def simple_chamfer(a: T, b: T, normals_a=None, normals_b=None, dist_mat: Optional[T] = None) -> Union[T, TS]:

    def one_direction(fixed: T, search: T, n_f, n_s, closest_id) -> TS:
        min_dist = (fixed - search[closest_id]).norm(2, 1).mean(0)
        if n_f is not None:
            normals_dist = -torch.einsum('nd,nd->n', n_f, n_s[closest_id]).mean(0)
        else:
            normals_dist = 0
        return min_dist, normals_dist

    if dist_mat is None:
        dist_mat = get_dist_mat(a, b)
    close_to_a, close_to_b = naive_knn(1, dist_mat)
    dist_a, dist_a_n = one_direction(a, b, normals_a, normals_b, close_to_a.flatten())
    dist_b, dist_b_n = one_direction(b, a, normals_b, normals_a, close_to_b.flatten())
    if normals_a is None:
        return dist_a + dist_b
    return dist_a + dist_b, dist_a_n + dist_b_n


def is_quad(mesh: Union[T_Mesh, Tuple[T, List[List[int]]]]) -> bool:
    if type(mesh) is T:
        return False
    if type(mesh[1]) is T:
        return False
    else:
        faces: List[List[int]] = mesh[1]
        for f in faces:
            if len(f) == 4:
                return True
    return False


def align_mesh(mesh: T_Mesh, ref_vs: T) -> T_Mesh:
    vs, faces = mesh
    dist_mat =  get_dist_mat(vs, ref_vs)
    dist, mapping_id = dist_mat.min(1)
    vs_select = dist_mat.min(0)[1]
    if mapping_id.unique().shape[0] != vs.shape[0]:
        print('\n\033[91mWarning, alignment is not bijective\033[0m')
    vs_aligned = vs[vs_select]
    faces_aligned = mapping_id[faces]
    return vs_aligned, faces_aligned


def create_convex_hull(target_mesh: T_Mesh_T):
    if type(target_mesh) is T:
        points = target_mesh
    else:
        points = target_mesh[0]
    device = points.device
    points_np = points.cpu().numpy()
    convex_hull = ConvexHull(points_np)
    faces_raw = torch.from_numpy(convex_hull.simplices).to(dtype=torch.int64)
    vs_inds = faces_raw.flatten().unique(sorted=False)
    mapper = torch.zeros(points.shape[0], dtype=torch.int64)
    mapper[vs_inds] = torch.arange(vs_inds.shape[0])
    faces = mapper[faces_raw].to(device)
    vs = points[vs_inds]
    triangles = vs[faces]
    normals = get_faces_normals(triangles)
    center = vs.mean(0)
    vec_inside = triangles.mean(1) - center[None, :]
    dot = torch.einsum('fd,fd->f', normals, vec_inside)
    should_flip = dot.lt(0)
    to_flip_faces = faces[should_flip]
    to_flip_faces = to_flip_faces[:, torch.tensor([0, 2, 1], dtype=torch.int64)]
    faces[should_flip] = to_flip_faces
    return vs, faces
# def triangulate_mesh(mesh: Union[T_Mesh, Tuple[T, List[List[int]]]]) -> Tuple[T_Mesh, Optional[T]]:
#
#     def check_triangle(triangle: List[int]) -> bool:
#         e_1: T = vs[triangle[1]] - vs[triangle[0]]
#         e_2: T = vs[triangle[2]] - vs[triangle[0]]
#         angle = (e_1 * e_2).sum() / (e_1.norm(2) * e_2.norm(2))
#         return angle.abs().item() < 1 - 1e-6
#
#     def add_triangle(face_: List[int]):
#         triangle = None
#         for i in range(len(face_)):
#             triangle = [face_[i], face_[(i + 1) % len(face_)], face_[(i + 2) % len(face_)]]
#             if check_triangle(triangle):
#                 face_ = [f for j, f in enumerate(face_) if j != (i + 1) % len(face_)]
#                 break
#         assert triangle is not None
#         faces_.append(triangle)
#         face_twin.append(-1)
#         return face_
#
#     if not is_quad(mesh):
#         return mesh, None
#
#     vs, faces = mesh
#     faces_ = []
#     face_twin = []
#     for face in faces:
#         if len(face) == 3:
#             faces_.append(face)
#             face_twin.append(-1)
#         else:
#             while len(face) > 4:
#                 face = add_triangle(face)
#             new_faces = [[face[0], face[1], face[2]], [face[0], face[2], face[3]]]
#             if not check_triangle(new_faces[0]) or not check_triangle(new_faces[1]):
#                 new_faces = [[face[0], face[1], face[3]], [face[1], face[2], face[3]]]
#                 assert check_triangle(new_faces[0]) and check_triangle(new_faces[1])
#             faces_.extend(new_faces)
#             face_twin.extend([len(faces_) - 1, len(faces_) - 2])
#         # else:
#         #     raise ValueError(f'mesh with {len(face)} edges polygons is not supported')
#     faces_ = torch.tensor(faces_, device=vs.device, dtype=torch.int64)
#     face_twin = torch.tensor(face_twin, device=vs.device, dtype=torch.int64)
#     return (vs, faces_), face_twin


def triangulate_mesh(mesh: Union[T_Mesh, Tuple[T, List[List[int]]]]) -> Tuple[T_Mesh, Optional[T]]:

    def get_skinny(faces_) -> T:
        vs_faces = vs[faces_]
        areas = compute_face_areas(vs_faces)[0]
        edges = reduce(
            lambda a, b: a + b,
            map(
                lambda i: ((vs_faces[:, i] - vs_faces[:, (i + 1) % 3]) ** 2).sum(1),
                range(3)
            )
        )
        skinny_value = np.sqrt(48) * areas / edges
        return skinny_value


    if not is_quad(mesh):
        return mesh, None

    vs, faces = mesh
    device = vs.device
    faces_keep = torch.tensor([face for face in faces if len(face) == 3], dtype=torch.int64, device=device)
    faces_quads = torch.tensor([face for face in faces if len(face) != 3], dtype=torch.int64, device=device)
    faces_tris_a, faces_tris_b = faces_quads[:, :3], faces_quads[:, torch.tensor([0, 2, 3], dtype=torch.int64)]
    faces_tris_c, faces_tris_d = faces_quads[:, 1:], faces_quads[:, torch.tensor([0, 1, 3], dtype=torch.int64)]
    skinny = [get_skinny(f) for f in (faces_tris_a, faces_tris_b, faces_tris_c, faces_tris_d)]
    skinny_ab, skinny_cd = torch.stack((skinny[0], skinny[1]), 1), torch.stack((skinny[2], skinny[3]), 1)
    to_flip = skinny_ab.min(1)[0].lt(skinny_cd.min(1)[0])
    faces_tris_a[to_flip], faces_tris_b[to_flip] = faces_tris_c[to_flip], faces_tris_d[to_flip]
    faces_tris = torch.cat((faces_tris_a, faces_tris_b, faces_keep), dim=0)
    face_twin = torch.arange(faces_tris_a.shape[0], device=device)
    face_twin = torch.cat((face_twin + faces_tris_a.shape[0], face_twin,
                           -torch.ones(faces_keep.shape[0], device=device, dtype=torch.int64)))
    return (vs, faces_tris), face_twin


# def fill_holes(path: str):
#     # np_mesh = to_numpy(*mesh)
#     with open(path, 'r') as f:
#         t_mesh = trimesh.exchange.obj.load_obj(f)
#     without_holes = trimesh.repair.fill_holes(t_mesh)
#     return without_holes



# def simple_chamfer(a: T, b: T, normals_a=None, normals_b=None) -> TS:
#     def one_direction(fixed: T, search: T, n_f, n_s) -> TS:
#
#         # closest_id = knn(search, fixed, 1)[1].flatten()
#         closest_id = naive_knn(search, fixed, 1, is_biknn=False, batch_size=1000).flatten()
#         min_dist = (fixed - search[closest_id]).norm(2, 1).mean(0)
#         if n_f is not None:
#             normals_dist = -torch.einsum('nd,nd->n', n_f, n_s[closest_id]).mean(0)
#         else:
#             normals_dist = 0
#         return min_dist, normals_dist
#
#     dist_a, dist_a_n = one_direction(a, b, normals_a, normals_b)
#     dist_b, dist_b_n = one_direction(b, a, normals_b, normals_a)
#     return dist_a + dist_b, dist_a_n + dist_b_n



if __name__ == '__main__':
    from utils import files_utils
    import constants
    mesh_quad = files_utils.load_mesh(f'{constants.RAW_ROOT}face_source_reg_quad')
    mesh_tri = triangulate_mesh(mesh_quad)[0]
    files_utils.export_mesh(mesh_tri, './tri_b')
