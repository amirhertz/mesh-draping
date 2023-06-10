import custom_types
from core import drape_online
import constants 
from custom_types import *
import vtk
import multiprocessing as mp
from custom_types import T, TN, T_Mesh
from utils import mesh_utils, deform_utils, files_utils, mesh_closest_point
import ctypes

bg_source_color = (222, 235, 250)
bg_target_color = (219, 233, 189)
bg_mid_color = (255, 218, 184)
default_colors = [(82, 108, 255), (160, 82, 255), (255, 43, 43), (255, 246, 79),
                  (153, 227, 107), (58, 186, 92), (8, 243, 255), (240, 136, 0)]


RGB_COLOR = Tuple[int, int, int]
RGB_FLOAT_COLOR = Tuple[float, float, float]


def rgb_to_float(*colors: RGB_COLOR) -> Union[RGB_FLOAT_COLOR, List[RGB_FLOAT_COLOR]]:
    float_colors = [(c[0] / 255., c[1] / 255., c[2] / 255.) for c in colors]
    if len(float_colors) == 1:
        return float_colors[0]
    return float_colors


def get_new_sphere(radius: float, x, y, z) -> vtk.vtkSphereSource:
    source = vtk.vtkSphereSource()
    source.SetRadius(radius * constants.GLOBAL_SCALE / 25.)
    source.SetCenter(x, y, z)
    source.SetPhiResolution(11)
    source.SetThetaResolution(21)
    return source


def set_default_properties(actor, color):
    properties = actor.GetProperty()
    properties.SetPointSize(10)
    properties.SetDiffuseColor(.6, .6, .6)
    properties.SetDiffuse(.8)
    properties.SetSpecular(.5)
    properties.SetSpecularColor(.2, .2, .2)
    properties.SetSpecularPower(30.0)
    properties.SetColor(*color)
    return actor


def wrap_mesh(source, color):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor = set_default_properties(actor, color)
    return actor, mapper


def point_in_triangle(vs: T, triangles: T):
    try:
        def sign(ind):
            d_vs_ = d_vs[:, (ind + 1) % 3]
            return d_vs_[:, 0] * d_t[ind][:, 1] - d_vs_[:, 1] * d_t[ind][:, 0]

        d_vs = vs[None, None, :] - triangles
        d_t = [triangles[:, i] - triangles[:, (i + 1) % 3] for i in range(3)]
        signs = torch.stack([sign(i) for i in range(3)], 1)
        in_triangle = torch.eq(torch.ge(signs, 0).sum(1), 3) + torch.eq(torch.le(signs, 0).sum(1), 3)
        triangle_ind = torch.where(in_triangle)[0]
        possible_triangles = triangles[triangle_ind]
        expended_vs = vs.unsqueeze(0).expand(possible_triangles.shape[0], *vs.shape)
        triangle_weights = mesh_utils.find_barycentric(expended_vs, possible_triangles)
        # to simplex
        triangle_weights = triangle_weights / triangle_weights.sum(1)[:, None]
        if triangle_weights.shape[0] > 1:
            # find_closest
            projections = torch.einsum('na,nad->nd', triangle_weights, possible_triangles)
            dist = (expended_vs - projections).norm(2, 1)
            closest = dist.argmin()
        else:
            closest = 0
        return triangle_ind[closest], triangle_weights[closest]
    except:
        return None, None


class VtkPointCloud:

    def add_point(self, point: ARRAY, color_scale: float):
        point_id = self.vtkPoints.InsertNextPoint(point[:])
        color = int((point[0] + color_scale) * 122.5 / color_scale)
        self.colors.InsertNextTuple3(color, color, color)
        # self.vtkDepth.InsertNextValue(point[2])
        self.vtkCells.InsertNextCell(1)
        self.vtkCells.InsertCellPoint(point_id)
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        # self.vtkDepth.Modified()

    def clear_points(self):
        # self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        # self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')

    def add_points(self, path: str):
        vs = files_utils.load_mesh(path)
        if type(vs) is tuple:
            vs = vs[0]

        self.colors.SetNumberOfComponents(3)
        vs = mesh_utils.to_unit_cube(vs, scale=constants.GLOBAL_SCALE)[0]
        vs = vs.numpy()
        color_scale = (vs[:, 0].max() - vs[:, 0].min()) * (2 / 3)
        for v in vs:
            self.add_point(v, color_scale)

    def __init__(self, path: str):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        # self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkPolyData = vtk.vtkPolyData()
        self.colors = vtk.vtkUnsignedCharArray()
        self.clear_points()
        self.add_points(path)
        self.vtkPolyData.GetPointData().SetScalars(self.colors)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(-constants.GLOBAL_SCALE, constants.GLOBAL_SCALE)
        mapper.SetScalarVisibility(1)
        self.mapper = mapper
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.GetProperty().SetPointSize(6)
        self.vtkActor.SetMapper(self.mapper)


class History:

    class HistoryItem:

        def __call__(self):
            return self.is_left, (self.address_a, self.address_b), self.deform_type

        def __repr__(self):
            return f'{self.is_left}, {(self.address_a, self.address_b)}, {self.deform_type}'

        def __init__(self, is_left: bool, address_a: str, address_b: str, deform_type: DeformType):
            self.is_left = is_left
            self.address_a = address_a
            self.address_b = address_b
            self.deform_type = deform_type

    def get_deform_types(self) -> List[custom_types.DeformType]:
        return self.flatten([[deform_type] * (1 + int(addresses[1] is not None))
                             for is_left, addresses, deform_type in self if is_left])

    # def split(self, *args: Sized,
    #           callback: Optional[Callable[[List[Any]], Any]] = None) -> Union[List[Tuple[Any, Any]], Tuple[Any, Any]]:
    #     is_rigid_list: List[bool] = self.flatten([[is_rigid] * (1 + int(addresses[1] is not None)) for is_left, addresses, is_rigid in self if is_left])[:len(args[0])]
    #     out = []
    #     for arg in args:
    #         non_rigid = [item for i, item in enumerate(arg) if not is_rigid_list[i]]
    #         rigid = [item for i, item in enumerate(arg) if is_rigid_list[i]]
    #         if callback is not None:
    #             rigid, non_rigid = callback(rigid), callback(non_rigid)
    #         out.append((non_rigid, rigid))
    #     if len(out) == 1:
    #         return out[0]
    #     return out

    @staticmethod
    def flatten(lst):
        flat_list = [item for sublist in lst for item in sublist if item is not None]
        return flat_list

    def get_addresses(self, is_left: bool, is_beauty: bool = False):
        addresses = self.flatten([address for is_left_, address, deform_type in self
                                  if is_left_ == is_left
                                  and (deform_type == DeformType.BEAUTY) == is_beauty])
        return addresses

    def get_beauty_address(self):
        return self.get_addresses(True, True)

    def __len__(self):
        return len(self.history)

    def pop(self):
        return self.history.pop()()

    def add_item(self, is_left, address_a, address_b, deform_type: DeformType):
        self.history.append(self.HistoryItem(is_left, address_a, address_b, deform_type))
        pass

    def __next__(self):
        if self.item >= len(self):
            raise StopIteration
        item = self.history[self.item]
        self.item += 1
        return item()

    def __iter__(self):
        self.item = 0
        return self

    def __init__(self):
        self.history: List[History.HistoryItem] = []
        self.item = 0


class SymmetryControl:

    class SymmetryStatus(Enum):
        ON = 'symmetry on'
        OFF = 'symmetry off'
        AUTO = 'auto symmetry'

    @property
    def is_symmetric(self) -> bool:
        return self.symmetry_status == self.SymmetryStatus.ON or \
               self.symmetry_status == self.SymmetryStatus.AUTO and self.auto_on

    def is_symmetric_val(self, val: T, width: float) -> bool:
        return self.symmetry_status == self.SymmetryStatus.ON or val.item() > width / 25.

    def is_symmetric_source(self, xyz: T) -> bool:
        return self.is_symmetric and self.is_symmetric_val(xyz[self.source_axis].abs(),
                                                           self.sym_source_width[self.source_axis])

    def is_symmetric_target(self, xyz: T) -> bool:
        return self.is_symmetric and self.is_symmetric_val(xyz[self.target_axis].abs(),
                                                           self.sym_target_width[self.target_axis])

    @staticmethod
    def get_symmetry(mesh) -> Tuple[int, Tuple[float, ...], float]:

        def symmetry_score(a: int) -> Tuple[float, float]:
            if type(mesh) is T:
                vs, faces = mesh, None
            else:
                vs, faces = mesh
            vs_sym = vs.clone()
            vs_sym[:, a] = -vs_sym[:, a]
            w = (vs[:, a].max() - vs[:, a].min()) / 2
            if faces is None:
                distances = mesh_utils.get_dist_mat(vs, vs_sym).min(1)[0] / w
            else:
                distances = mesh_closest_point.mesh_cp(mesh, (vs_sym, faces))[1] / w
            nn_max = distances.max().item()
            return nn_max, w

        sym_axis = -1
        val = 1000.
        width = [0] * 3
        for axis in (0, 1, 2):
            val_, width[axis] = symmetry_score(axis)
            if val_ < val:
                sym_axis, val = axis, val_

        return sym_axis, tuple(width), val

    def toggle_symmetry_axis(self, is_left: bool) -> int:
        if is_left:
            self.source_axis = (self.source_axis + 1) % 3
            return self.source_axis
        else:
            self.target_axis = (self.target_axis + 1) % 3
            return self.target_axis

    def toggle_symmetry(self) -> SymmetryStatus:
        if self.symmetry_status == self.SymmetryStatus.AUTO:
            self.symmetry_status = self.SymmetryStatus.ON
        elif self.symmetry_status == self.SymmetryStatus.ON:
            self.symmetry_status = self.SymmetryStatus.OFF
        else:
            self.symmetry_status = self.SymmetryStatus.AUTO
        return self.symmetry_status

    def __init__(self, source_mesh: T_Mesh, target_mesh: T_Mesh):
        self.source_axis, self.sym_source_width, val_source = self.get_symmetry(source_mesh)
        self.target_axis, self.sym_target_width, val_target = self.get_symmetry(target_mesh)
        self.auto_on = val_source < .15 and val_target < .15
        self.symmetry_status = self.SymmetryStatus.AUTO


class MatchStatus:

    def load_points(self) -> Tuple[TN, TN, TN, List[DeformType]]:
        source_points, target_points, beatify_points, deform_types, symmetric = files_utils.load_points(self.save_path, '', self.triangles)
        if len(symmetric) < len(deform_types):
            source_points = None
        if source_points is not None:
            source_points = self.cur_vs[source_points]
            deform_types = [item for i, item in enumerate(deform_types) if symmetric[i]]
            source_points, target_points = source_points[symmetric], target_points[symmetric]
        return source_points, beatify_points, target_points, deform_types

    def start_optimize(self):
        if self.status.value == OptimizingStatus.WAITING.value:
            drape_online.sync_vs(self.shared_vs, self.cur_vs)
            self.status.value = OptimizingStatus.OPTIMIZING.value
        with self.optimize_condition:
            self.optimize_condition.notify_all()

    def stop_optimize(self):
        if self.status.value == OptimizingStatus.OPTIMIZING.value or self.status.value == OptimizingStatus.PAUSE.value:
            self.status.value = OptimizingStatus.WAITING.value
        with self.optimize_condition:
            self.optimize_condition.notify_all()

    def pause_optimize(self):
        if self.status.value == OptimizingStatus.OPTIMIZING.value:
            self.status.value = OptimizingStatus.PAUSE.value
        with self.optimize_condition:
            self.optimize_condition.notify_all()

    def continue_optimize(self):
        if self.status.value == OptimizingStatus.PAUSE.value:
            self.status.value = OptimizingStatus.OPTIMIZING.value
        with self.optimize_condition:
            self.optimize_condition.notify_all()

    @staticmethod
    def to_tensor(lst) -> T:
        if len(lst) == 0:
            item = 1
        else:
            item = lst[0]
            while type(item) is list:
                item = item[0]
        t = torch.tensor(lst, dtype=torch.int64 if type(item) is int else torch.float32)
        return t

    @property
    def left_inds(self):
        return [self.left_mapper[address][2] for address in self.history.get_addresses(True)]

    @property
    def all_left(self):
        addresses = self.history.flatten([address for is_left, address, _ in self.history if is_left])
        return [self.left_mapper[address][2] for address in addresses]


    @property
    def symmetric_list(self):
        addresses = [address for is_left, address, _ in self.history if is_left]
        s_list = []
        for pair in addresses:
            s_list.append(True)
            if None not in pair:
                s_list.append(False)
        return s_list

    def set_deform_type(self, deform_type: DeformType):
        self.deform_type = deform_type

    def get_correspondence(self) -> Tuple[T, T, List[custom_types.DeformType]]:
        source_inds = torch.tensor(self.left_inds[:self.num_points], dtype=torch.int64)
        vs_target = torch.tensor(self.target_points[:self.num_points], dtype=torch.float32)
        deform_types = self.history.get_deform_types()
        # deform_types = list(filter(lambda x: x != DeformType.BEAUTY, self.history.get_deform_types()))[:self.num_points]
        return source_inds, vs_target, deform_types

    @property
    def cur_mesh(self) -> T_Mesh:
        return self.cur_vs, self.source_mesh[1]

    @property
    def base_mesh(self) -> T_Mesh:
        return self.base_vs, self.source_mesh[1]

    @property
    def after_affine(self):
        return self.num_points >= self.min_affine_points

    def get_deform(self):
        if self.after_affine or self.after_opt:
            source_inds, vs_target, deform_types = self.get_correspondence()
            deformed = deform_utils.deform_mesh_hybrid(self.base_mesh, source_inds, vs_target, deform_types, dist=.8,
                                                       apply_affine=not self.after_opt)
            deformed_vs = deformed[1][0]
        else:
            deformed_vs = self.source_mesh[0]
        if (deformed_vs - self.cur_vs).ne(0).any():
            self.cur_vs = deformed_vs
            self.update_ui(self.cur_vs.detach().cpu().numpy())

    def optimizer_callback(self):
        with self.shared_vs:
            vs = drape_online.to_np_array(self.shared_vs)
        self.cur_vs = torch.from_numpy(vs)
        self.update_ui(vs)

    def nearest_point(self, v: T) -> TS:
        dist_s = ((v[None, :] - self.triangles) ** 2).sum(1)
        nearest = dist_s.argmin(0)
        return nearest, torch.zeros(3) - 1

    def to_barycentric(self, vs: T, axis=-1) -> TS:
        if axis > -1:
            vs = vs.clone()
            vs[axis] = -vs[axis]
        if self.pc_mode:
            return self.nearest_point(vs)
        triangle_ind, triangle_weights = point_in_triangle(vs, self.triangles)
        return triangle_ind, triangle_weights

    def save(self):
        if len(self.history) > 0 and self.actor_counter[0] == self.actor_counter[1] - self.actor_counter[2]:
            source_inds, target_pts, deform_types = self.get_correspondence()
            beauty_inds = torch.tensor([self.left_mapper[address][2] for address in self.history.get_beauty_address()], dtype=torch.int64)
            address_right = self.history.get_addresses(False)
            t_a = [int(self.right_mapper[address][2][0]) for address in address_right]
            f_weights = [self.right_mapper[address][2][1].tolist() for address in address_right]
            print('save')
            files_utils.save_pickle({'source_pts': source_inds, 'target_faces': t_a, 'target_barycentric': f_weights,
                                     'deform_types': deform_types, 'symmetric': self.symmetric_list,
                                     'beauty_pts': beauty_inds}, self.save_path)
            files_utils.save_pickle({'source_pts': source_inds, 'target_pts': target_pts, 'beauty_pts': beauty_inds},
                                    constants.POINTS_CACHE)
            with self.fixed_status:
                self.fixed_status.value = OptimizingStatus.Update.value

    @staticmethod
    def remove_color(address, to_update_colors, to_update_mapper, deform_type: DeformType) -> int:
        to_update_mapper[address] = (-1, to_update_mapper[address][1])
        if deform_type != DeformType.BEAUTY:
            to_update_colors.pop()
        return to_update_mapper[address][1]

    def get_color(self, address, to_update_colors, to_update_mapper, not_to_update_colors) -> Tuple[float, ...]:
        if address in to_update_mapper:
            to_update_mapper[address] = (len(to_update_colors), to_update_mapper[address][1])
        else:
            to_update_mapper[address] = (len(to_update_colors), len(to_update_colors))
        if self.deform_type == DeformType.BEAUTY:
            return 1, .78, 1
        if len(to_update_colors) >= len(not_to_update_colors):
            new_color = tuple(torch.rand(3).tolist())
        else:
            new_color = not_to_update_colors[len(to_update_colors)]
        to_update_colors.append(new_color)
        return to_update_colors[-1]

    def get_point_id_left(self, xyz: T, vs: T, axis=-1) -> int:
        if axis > -1:
            xyz = xyz.clone()
            xyz[axis] = -xyz[axis]
        delta = ((vs - xyz[None, :]) ** 2).sum(1)
        point_id = delta.argmin()
        return point_id.item()

    def get_actor_inds(self, is_left: bool, is_beauty: bool) -> List[int]:
        addresses = self.history.get_addresses(is_left, is_beauty)
        mapper = self.left_mapper if is_left else self.right_mapper
        ids = [mapper[address][1] + 1 + int(is_left) for address in addresses]
        return ids

    @property
    def beauty_actors(self) -> List[int]:
        return self.get_actor_inds(True, True)

    @property
    def left_actors(self) -> List[int]:
        return self.get_actor_inds(True, False)

    @property
    def right_actors(self) -> List[int]:
        return self.get_actor_inds(False, False)

    def update(self, address: str, is_left_viewport: bool, xyz) -> Tuple[Tuple[float, ...], bool, any]:
        if is_left_viewport and (address not in self.left_mapper or self.left_mapper[address][0] > -1):
            return (), False, (None,)
        if not is_left_viewport and (address not in self.right_mapper or self.right_mapper[address][0] > -1):
            return (), False, (None,)
        if self.deform_type is DeformType.BEAUTY and not is_left_viewport:
            return (), False, (None,)
        mapper = vtk.vtkPolyDataMapper()
        address = mapper.GetAddressAsString('')
        if self.is_symmetric:
            mapper_b = vtk.vtkPolyDataMapper()
            address_b = mapper_b.GetAddressAsString('')
        else:
            mapper_b = address_b = None
        xyz, xyz_b = torch.tensor(xyz, dtype=self.triangles.dtype), None
        if is_left_viewport:
            color = self.get_color(address, self.left_colors, self.left_mapper, self.right_colors)
            point_id = self.get_point_id_left(xyz, self.cur_vs)
            xyz = self.cur_vs[point_id]
            xyz_org = self.original_vs[point_id]
            self.left_mapper[address] = (self.left_mapper[address][0], self.actor_counter[is_left_viewport], point_id)
            if self.symmetry_control.is_symmetric_source(xyz):
                point_id = self.get_point_id_left(xyz_org, self.original_vs, self.symmetry_control.source_axis)
                xyz_b = self.cur_vs[point_id]
                self.left_mapper[address_b] = (self.left_mapper[address][0],
                                               self.actor_counter[is_left_viewport] + 1, point_id)
            else:
                xyz_b = None
        else:
            if self.symmetry_control.is_symmetric_target(xyz):
                face_id_b, weights_b = self.to_barycentric(xyz, self.symmetry_control.target_axis)
            else:
                if self.symmetry_control.is_symmetric:
                    xyz[self.symmetry_control.target_axis] = 0
                face_id_b = 1
                weights_b = None
            face_id, weights = self.to_barycentric(xyz)
            if face_id is not None and face_id_b is not None:
                color = self.get_color(address, self.right_colors, self.right_mapper, self.left_colors)
                if self.pc_mode:
                    xyz = self.triangles[face_id].tolist()
                else:
                    xyz = mesh_utils.from_barycentric(self.triangles, face_id, weights).tolist()
                self.target_points.append(xyz)
                self.right_mapper[address] = (self.right_mapper[address][0], self.actor_counter[is_left_viewport], (face_id, weights))
                if weights_b is not None:
                    if self.pc_mode:
                        xyz_b = self.triangles[face_id_b].tolist()
                    else:
                        xyz_b = mesh_utils.from_barycentric(self.triangles, face_id_b, weights_b).tolist()
                    self.target_points.append(xyz_b)
                    self.right_mapper[address_b] = (self.right_mapper[address][0],
                                                    self.actor_counter[is_left_viewport] + 1,
                                                    (face_id_b, weights_b))
            else:
                return (), False, None
        new_sphere = get_new_sphere(.2, *xyz)
        mapper.SetInputConnection(new_sphere.GetOutputPort())
        self.actor_counter[is_left_viewport] += 1
        if xyz_b is not None:
            new_sphere = get_new_sphere(.2, *xyz_b)
            mapper_b.SetInputConnection(new_sphere.GetOutputPort())
            self.history.add_item(is_left_viewport, address, address_b, self.deform_type)
            self.actor_counter[is_left_viewport] += 1
        else:
            self.history.add_item(is_left_viewport, address, None, self.deform_type)
            mapper_b = None
        if self.deform_type == DeformType.BEAUTY:
            self.actor_counter[2] += 1 + int(mapper_b is not None)
        self.save()
        return color, True, (mapper, mapper_b)

    def undo(self) -> Tuple[bool, Tuple[Optional[int], Optional[int]]]:
        if len(self.history) == 0:
            return True, (None, None)
        actor_id_b = None
        is_left, (address, address_b), deform_type = self.history.pop()
        if is_left:
            mapper = self.left_mapper
            colors = self.left_colors
        else:
            mapper = self.right_mapper
            colors = self.right_colors
            self.target_points.pop()
        actor_id = self.remove_color(address, colors, mapper, deform_type) + 1 + int(is_left)
        mapper.__delitem__(address)
        if address_b is not None:
            actor_id_b = mapper[address_b][1] + 1 + int(is_left)
            mapper.__delitem__(address_b)
            if not is_left:
                self.target_points.pop()
        self.save()
        self.actor_counter[is_left] -= 1 + int(actor_id_b is not None)
        if deform_type == DeformType.BEAUTY:
            self.actor_counter[2] -= 1 + int(actor_id_b is not None)
        return is_left, (actor_id_b, actor_id)

    def resume_optimization(self):
        self.deform_type = DeformType.HARMONIC
        self.after_opt = True
        self.base_vs = self.cur_vs
        with self.status:
            self.status.value = OptimizingStatus.WAITING.value

    @property
    def num_points(self) -> int:
        return min(self.actor_counter[0], self.actor_counter[1] - self.actor_counter[2])

    @property
    def original_vs(self):
        return self.source_mesh[0]

    @property
    def is_symmetric(self) -> bool:
        return self.symmetry_control.is_symmetric

    def exit(self):
        with self.status:
            self.status.value = OptimizingStatus.Exit.value
        with self.optimize_condition:
            self.optimize_condition.notify_all()
        print('exit optimizer')
        if not constants.DEBUG:
            self.optimizer_process.join()
        print('exit')

    def watch_for_sync(self):
        while self.status.value != OptimizingStatus.Exit.value:
            with self.sync_condition:
                self.sync_condition.wait()
            if self.status.value != OptimizingStatus.Exit.value:
                self.optimizer_callback()

    @property
    def value(self):
        return self.status.value

    def toggle_symmetry(self) -> SymmetryControl.SymmetryStatus:
        return self.symmetry_control.toggle_symmetry()

    def toggle_symmetry_axis(self, is_left: bool):
        return self.symmetry_control.toggle_symmetry_axis(is_left)

    def __init__(self, save_path: str, source_path: str, target_path: str,):
        self.pc_mode = False
        files_utils.delete_single(constants.POINTS_CACHE + '.pkl')
        target_mesh = mesh_utils.to_unit_cube(files_utils.load_mesh(target_path), scale=constants.GLOBAL_SCALE)[0]
        if type(target_mesh) is T:
            self.pc_mode = True
        elif self.pc_mode:
            target_mesh = target_mesh[0]
        elif mesh_utils.is_quad(target_mesh):
            target_mesh = mesh_utils.triangulate_mesh(target_mesh)[0]
        self.source_mesh = mesh_utils.to_unit_cube(files_utils.load_mesh(source_path), scale=constants.GLOBAL_SCALE)[0]
        if mesh_utils.is_quad(self.source_mesh):
            self.source_mesh = mesh_utils.triangulate_mesh(self.source_mesh)[0]
            # files_utils.export_mesh(self.source_mesh, '../raw/dog_triangulate')
        self.symmetry_control = SymmetryControl(self.source_mesh, target_mesh)
        self.cur_vs = self.base_vs = self.source_mesh[0]
        if self.pc_mode:
            self.triangles = target_mesh

        else:
            self.triangles = target_mesh[0][target_mesh[1]]
        self.left_colors: List[Tuple[float, ...]] = []
        self.right_colors: List[Tuple[float, ...]] = []
        self.actor_counter = [0, 0, 0]
        self.left_mapper: Dict[str, Tuple[int, int, int]] = dict()
        self.right_mapper: Dict[str, Tuple[int, int, Tuple[int, T]]] = dict()
        self.history = History()
        self.target_points = []
        self.save_path = save_path
        self.min_affine_points = 5
        self.deform_type = DeformType.HARMONIC
        self.after_opt = False
        self.status = mp.Value('i', OptimizingStatus.WAITING.value)
        self.fixed_status = mp.Value('i', OptimizingStatus.WAITING.value)
        self.optimize_condition = mp.Condition()
        self.sync_condition = mp.Condition()
        self.shared_vs = mp.Array(ctypes.c_float, self.cur_vs.shape[0] * 3)
        self.update_ui: Optional[Callable[[ARRAY], None]] = None
        self.optimizer_process = mp.Process(target=drape_online.draping_opt_main,
                                            args=(CUDA(0), self.source_mesh, target_mesh,
                                                  self.status, self.fixed_status, self.optimize_condition,
                                                  self.sync_condition, self.shared_vs))
        drape_online.sync_vs(self.shared_vs, self.cur_vs)
        self.optimizer_process.start()
