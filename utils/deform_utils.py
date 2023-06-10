import igl
import constants
from custom_types import *
from custom_types import DeformType
from utils import files_utils, mesh_utils, transformations_tools
from core import mesh_ds


def only_scale_deform(source: T_Mesh, target: T_Mesh, in_place=True) -> T_Mesh:
    vs_source, vs_target = source[0], target[0]
    if not in_place:
        vs_source = vs_source.clone()
    scale = (vs_target.max(0)[0] - vs_target.min(0)[0]) / (vs_source.max(0)[0] - vs_source.min(0)[0])
    vs_source *= scale[None, :]
    translate = (vs_target.max(0)[0] + vs_target.min(0)[0] - vs_source.max(0)[0] - vs_source.min(0)[0]) / 2
    vs_source += translate[None, :]
    return vs_source, source[1]


def export_with_edges(ds: mesh_ds.MeshDS, edges_ids: Union[T, TS], path: str) -> None:
    if type(edges_ids) is T:
        edges_ids = [edges_ids]
    colors = torch.ones_like(ds.mesh[0]) * .8
    all_edges = []
    for i, ids_ in enumerate(edges_ids):
        edges = ds.edges[ids_]
        all_edges.append(edges)
        vs_ids = edges.flatten().unique()
        if i > len(constants.COLORS) - 1:
            color = torch.rand(3)
        else:
            color = torch.tensor(constants.COLORS[i], dtype=torch.float32) / 255.
        colors[vs_ids] = color
    files_utils.export_mesh(ds.mesh, path, colors=colors, edges=torch.cat(all_edges, dim=0))


def build_adj(ds: mesh_ds.MeshDS, edges_ids: T) -> Tuple[T, T, Dict[tuple, int]]:
    edges = ds.edges[edges_ids]
    vs_ids = edges.flatten().unique().sort()[0]
    mapper = torch.zeros(ds.vs.shape[0], dtype=torch.int64) - 1
    mapper[vs_ids] = torch.arange(vs_ids.shape[0])
    edges_m = mapper[edges]
    adj = torch.zeros(vs_ids.shape[0], vs_ids.shape[0])
    edges_m = edges_m.sort(1)[0]
    adj[edges_m[:, 0], edges_m[:, 1]] = 1
    adj[edges_m[:, 1], edges_m[:, 0]] = 1
    edges_m = edges_m.tolist()
    tuple_to_edge = {tuple(edge): i for i, edge in enumerate(edges_m)}
    return adj, vs_ids, tuple_to_edge


def scale_transform(source: T_Mesh, source_points: T, target_points: T) -> T_Mesh:
    affine: T = transformations_tools.find_affine(source_points, target_points)
    scale = affine.diag()[:-1].mean()
    scale_mat = torch.zeros_like(affine)
    scale_mat[:, -1] = affine[:, -1]
    scale_mat[torch.arange(3), torch.arange(3)] = scale
    source_ = transformations_tools.apply_affine(scale_mat, source[0]), source[1]
    return source_


def rigid_scale_transform(source: T_Mesh, source_points: T, target_points: T) -> T_Mesh:
    center_a, center_b = source_points.mean(0), target_points.mean(0)
    c_points_a, c_points_b = source_points - center_a[None, :], target_points - center_b[None, :]
    scale = c_points_b.norm(2, 1).mean() / c_points_a.norm(2, 1).mean()
    c_points_a = c_points_a * scale
    h: T = torch.einsum('na,nb->ab', c_points_a, c_points_b)
    u, s, v = h.svd()
    rotation = torch.einsum('na,nb->ab', v, u)
    vs_ = (source[0] - center_a[None, :]) * scale
    vs_ = torch.einsum('ad,nd->nd', rotation, vs_) + center_b[None, :]
    return vs_, source[1]


def global_transform(source: T_Mesh, source_points: T, target_points: T) -> T_Mesh:
    affine = transformations_tools.find_affine(source_points, target_points)
    source_ = transformations_tools.apply_affine(affine, source[0]), source[1]
    return source_


def get_boundaries(mesh,  v_inds, dist: Optional[float]) -> TN:
    if dist is not None:
        mesh_ = mesh_utils.to_unit_cube(mesh, in_place=False)[0]
        dist_to_keys = mesh_utils.exact_geodesic(mesh_, v_inds)
        dist_to_keys[v_inds] = dist / 2
        boundaries_mask = dist_to_keys.gt(dist)
        if boundaries_mask.any():
            boundaries_id = torch.where(boundaries_mask)[0]
            return boundaries_id
        else:
            return None


def deform_mesh(mesh: T_Mesh, v_inds: T, v_target: T, dist: Optional[float] = .8) -> T_Mesh:
    device = mesh[0].device
    d_bc = v_target - mesh[0][v_inds]
    boundaries_id = get_boundaries(mesh, v_inds, dist)
    if boundaries_id is not None:
        v_inds = torch.cat((v_inds, boundaries_id), dim=0)
        d_bc = torch.cat((d_bc, torch.zeros(boundaries_id.shape[0], 3, device=d_bc.device)), dim=0)
    vs, faces, v_target, v_inds, d_bc = mesh_utils.to((*mesh, v_target, v_inds, d_bc), CPU)
    d_bc, vs_np = d_bc.numpy(), vs.numpy()
    if not np.isfortran(d_bc) and np.isfortran(vs_np):
        d_bc = np.asfortranarray(d_bc)
    # try:
    d = igl.harmonic_weights(vs_np, faces.numpy(), v_inds.unsqueeze(1).numpy(), d_bc, 2)
    u = vs + torch.from_numpy(d).float()
    # except Exception as err:
    #     print('err')

    #     u = vs
    # if to_plot:
    #     s = np.zeros(len(vs), dtype=np.int)
    #     s[v_inds] = 1
    #     p = subplot(vs.numpy(), faces.numpy(), s, shading={"wireframe": False, "colormap": "tab10"}, s=[1, 2, 0])
    #     subplot(u.numpy(), faces.numpy(), s, shading={"wireframe": False, "colormap": "tab10"}, s=[1, 2, 1], data=p)
    #     p.save('my_harmonic_weights')
    return (u.to(device), faces.to(device, ))


def deform_mesh_arap(mesh: T_Mesh, v_inds: T, v_target: T) -> T_Mesh:
    # points_for_transform = min(len(v_inds), 8)
    # points_for_transform = len(v_target)
    # mesh_a = global_transform(mesh, mesh[0][v_inds[:points_for_transform]], v_target[:points_for_transform])
    # files_utils.export_mesh(mesh_a, '../temp_affine')
    device, dtype = mesh[0].device, mesh[0].dtype
    boundaries_ids = get_boundaries(mesh, v_inds, 0.5)
    if boundaries_ids is not None:
        v_inds = torch.cat([v_inds, boundaries_ids])
        v_target = torch.cat([v_target, mesh[0][boundaries_ids]])
    vs, faces, v_target, v_inds = mesh_utils.to((*mesh, v_target, v_inds), CPU)
    vs_np, faces_np, bc, b = vs.numpy(), faces.numpy(), v_target.numpy(), v_inds.numpy()
    arap = igl.ARAP(vs_np, faces_np, 3, b)
    u = arap.solve(bc, vs_np)
    # if not np.isfortran(d_bc) and np.isfortran(vs_np):
    #     d_bc = np.asfortranarray(d_bc)
    # return mesh_a, mesh_a
    deformed = torch.from_numpy(u).to(device, dtype=dtype), faces.to(device)
    # files_utils.export_mesh(deformed, '../temp_deform')
    return deformed


def deform_mesh_hybrid(mesh: T_Mesh, inds_deform: T, target_deform: T, deform_types : List[DeformType],
                       dist: Optional[float] = .8, apply_affine: bool = True) -> Tuple[T_Mesh, T_Mesh]:

    def split_deforms() -> Tuple[List[T], List[T], List[T], List[T], List[DeformType]]:
        groups_source_ = [[]]
        groups_target_ = [[]]
        groups_source_com_ = [[]]
        groups_target_com_ = [[]]
        groups_type = [deform_types[0]]
        for i, deform_type in enumerate(deform_types):
            if deform_type != groups_type[-1]:
                groups_source_.append([])
                groups_target_.append([])
                groups_source_com_.append(groups_source_com_[-1])
                groups_target_com_.append(groups_target_com_[-1])
                groups_type.append(deform_type)
            groups_source_[-1].append(inds_deform[i])
            groups_target_[-1].append(target_deform[i])
            groups_source_com_[-1].append(inds_deform[i])
            groups_target_com_[-1].append(target_deform[i])
        groups_source_ = [torch.stack(group) for group in groups_source_]
        groups_target_ = [torch.stack(group, 0) for group in groups_target_]
        groups_source_com_ = [torch.stack(group) for group in groups_source_com_]
        groups_target_com_ = [torch.stack(group, 0) for group in groups_target_com_]
        return groups_source_, groups_target_, groups_source_com_, groups_target_com_, groups_type

    deform_types = list(filter(lambda x: x != DeformType.BEAUTY, deform_types))[: inds_deform.shape[0]]
    groups_source, groups_target, groups_source_com, groups_target_com, deform_types = split_deforms()
    if groups_source[0].shape[0] < 5:
        return mesh, mesh
    if apply_affine:
        mesh = mesh_a = global_transform(mesh, mesh[0][groups_source[0]], groups_target[0])
    else:
        mesh_a = mesh
    for i, deform_type  in enumerate(deform_types):
        if deform_type == DeformType.HARMONIC:
            deformed = deform_mesh(mesh, groups_source_com[i], groups_target_com[i], dist)
        else:
            deformed = deform_mesh_arap(mesh, groups_source[i], groups_target[i])
        mesh = deformed
    return mesh_a, mesh

