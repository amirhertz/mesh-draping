from __future__ import annotations
import abc
from custom_types import *
from utils import files_utils, mesh_utils, deform_utils
from core.mesh_ds import MeshDS, get_ds
import constants
from functools import reduce


def ring_probs(ds: MeshDS, ring_values: T):
    ring_values = (ring_values * ds.ring_mask.float())
    ring_values = ring_values / ring_values.sum(1)[:, None]
    ring_values.masked_fill_(~ds.ring_mask, 1)
    return ring_values


def ring_entropy_loss(ds: MeshDS, ring_values: T) -> T:
    ring_values = ring_probs(ds, ring_values)
    negative_entropy = (ring_values * torch.log(ring_values)).sum(1)
    return negative_entropy.mean()


class OptimizationParams:

    def fill_args(self, args):
        for arg in args:
            if hasattr(self, arg):
                setattr(self, arg, args[arg])

    def save(self, path: str):
        files_utils.save_pickle({"steps": self.steps,
                                 "curve_iters": self.beautify_iters,
                                 "project_iters": self.project_iters,
                                 "weights_curvature": self.weights_curvature,
                                 "weights_project": self.weights_project
                                 },
                                path)

    def load(self, path: str):
        data = files_utils.load_pickle(path)
        if data is not None:
            self.beautify_iters = data['curve_iters']
            self.project_iters = data['project_iters']
            self.steps = data['steps']
            self.weights_curvature = data['weights_curvature']
            self.weights_project = data['weights_project']
        return self

    @staticmethod
    def filter_dict(weights_dict, level: int) -> Dict[str, float]:
        return {key: value[level] for key, value in weights_dict.items() if value[level] != 0}

    def get_weights(self, level: int):
        return self.filter_dict(self.weights_project, level), self.filter_dict(self.weights_curvature, level)

    @property
    def total_steps(self):
        return sum(self.steps)

    @property
    def name(self) -> str:
        return self.model_type.value
        # return str(time())[:5]
        # return '_'.join([key for key, value in self.weights_info.items() if sum(value) > 0])

    def __init__(self, **kwargs):
        self.lr = 1e-5
        self.beautify_iters = 1
        self.project_iters = 1
        self.steps = (1000, 100)
        self.plot_every: int = 100
        self.k_decay_every = 100
        self.normals_weight = .2
        self.model_type = ModelType.EXPLICIT
        self.weights_curvature = {'face_angles': (1, 0.2), 'flip': (1, .2),  'area_entropy': (0, 0),
                                  'area_dkl': (1., 0.2), 'skinny_reg': (.1, .1), 'areas_reg': (0, 0)}
        self.weights_project = {'chamfer': (1, 1), 'fixed': (100, 10), 'skinny_reg': (.1, .1), 'areas_reg': (0, 0)}
        self.beautify_pts = False
        self.fill_args(kwargs)


class ChamferController:

    def get_sampled_source(self):
        sample_source = mesh_utils.get_sampled_fe(self.source_ds.vs, self.source_ds.mesh, *self.fixed_source)
        sample_source_normals = mesh_utils.get_sampled_fe(self.source_ds.vs_normals, self.source_ds.mesh,
                                                          *self.fixed_source)
        return sample_source, sample_source_normals


    @staticmethod
    def get_closest_id(a: T, b: T, knn: T, arange):
        with torch.no_grad():
            unrolled = b[knn]
            dist_sq = ((a[:, None, :] - unrolled) ** 2).sum(2)
            closet_in_k = dist_sq.argmin(1)
            closet_global = knn[arange, closet_in_k]
        return closet_global

    def get_loss(self):

        def one_direction_loss(a, normals_a, b, normals_b, closest_id):
            dist_loss = (a - b[closest_id]).norm(2, 1).mean(0)
            normals_loss = -torch.einsum('nd,nd->n', normals_a, normals_b[closest_id]).mean(0)
            return dist_loss, normals_loss

        sample_source, sample_source_normals = self.get_sampled_source()
        closest_to_source_id = self.get_closest_id(sample_source, self.fixed_target[0], self.knn[0], self.arange[0])
        closest_to_target = self.get_closest_id(self.fixed_target[0], sample_source, self.knn[1], self.arange[1])
        loss_to_a, loss_normals_a = one_direction_loss(sample_source, sample_source_normals, *self.fixed_target, closest_to_source_id)
        loss_to_b, loss_normals_b = one_direction_loss(*self.fixed_target, sample_source, sample_source_normals, closest_to_target)
        return loss_to_a, loss_normals_a, loss_to_b, loss_normals_b

    def reset_(self):
        with torch.no_grad():
            (samples, _), face_id, uvw = mesh_utils.get_samples(self.source_ds.mesh, self.num_samples,
                                                                mesh_utils.SampleBy.HYB, self.source_ds.vs_normals)
            fixed_target, _, _ = mesh_utils.get_samples(self.target_mesh, self.num_samples,
                                                        mesh_utils.SampleBy.HYB, self.target_normals)
            dist_mat = mesh_utils.get_dist_mat(samples, fixed_target[0])
            knn = mesh_utils.naive_knn(self.k, dist_mat)
            arange = [torch.arange(t.shape[0], device=self.device) for t in (face_id, fixed_target[0])]
        return (face_id, uvw), fixed_target, knn, arange

    def reset(self, num_samples: Optional[int] = None, k: Optional[int] = None):
        self.num_samples = num_samples if num_samples is not None else self.num_samples
        self.k = k if k is not None else self.k
        self.fixed_source, self.fixed_target, self.knn, self.arange = self.reset_()

    def __init__(self, source_ds: MeshDS, target_mesh: T_Mesh, target_normals: T,
                 num_samples: int = 3000, k: int = 50):
        self.device = source_ds.device
        self.target_mesh = target_mesh
        self.target_normals = target_normals
        self.source_ds = source_ds
        self.num_samples = num_samples
        self.k = k
        self.fixed_source, self.fixed_target, self.knn,  self.arange = self.reset_()


class DrapingOpt(abc.ABC):

    def optimize_iter(self, weights_project, weights_beauty, optimizer, model):
        log = {}
        for j in range(self.params.project_iters):
            self.opt_iteration(weights_project, optimizer, log, model)
        for j in range(self.params.beautify_iters):
            self.opt_iteration(weights_beauty, optimizer, log, model)
        model.increase()
        return log

    def between_iterations(self, i: int):
        if (i + 1) % self.params.k_decay_every == 0:
            num_samples = min(10000, self.chamfer_controller.num_samples * 2)
            k = max(10, self.chamfer_controller.k // 2)
            self.chamfer_controller.reset(num_samples=num_samples, k=k)
        return True

    def opt_iteration(self, weights_info, optimizer, log, model):
        optimizer.zero_grad()
        self.source_ds.vs = model(self.fixed_vs)
        loss = torch.zeros(1, device=self.device)
        for loss_name, weight in weights_info.items():
            loss_func = self.loss_mapper[loss_name]
            loss_ = loss_func(self)
            log[loss_name] = loss_
            loss += weight * loss_
        loss.backward()
        optimizer.step()

    def skinny_reg(self) -> T:
        edges = self.source_ds.vs[self.source_ds.faces]
        edges = reduce(
            lambda a, b: a + b,
            map(
                lambda i: ((edges[:, i] - edges[:, (i + 1) % 3]) ** 2).sum(1),
                range(3)
            )
        )
        skinny_value = np.sqrt(48) * self.source_ds.areas / edges
        mask = skinny_value.lt(.1)
        if not mask.any():
            return torch.zeros(1, device=self.device)
        return (1 - skinny_value[mask]).mean()

    # @abc.abstractmethod
    # def optimize(self):
    #     raise NotImplemented

    def local_area_entropy(self) -> T:
        areas = self.source_ds.areas[self.source_ds.v2f]
        return ring_entropy_loss(self.source_ds, areas)

    def local_area_dkl(self) -> T:
        areas_prob = ring_probs(self.source_ds, self.source_ds.areas[self.source_ds.v2f])
        # dkl = (areas_prob * torch.log(areas_prob / self.base_faces_prob)).sum(1)
        dkl = (self.base_faces_prob * torch.log(self.base_faces_prob / areas_prob)).sum(1)
        dkl = self.beautify_weights * dkl
        if torch.isnan(dkl).any():
            print("errror")
        return dkl.mean()

    def ch_iter(self) -> T:
        loss_to_a, loss_normals_a, loss_to_b, loss_normals_b = self.chamfer_controller.get_loss()
        return (loss_normals_a + loss_normals_b) * self.params.normals_weight + loss_to_a + loss_to_b

    def areas_reg(self) -> Union[float, T]:
        areas = self.source_ds.areas
        mask = areas.lt(1e-3)
        if not mask.any().item():
            return 0.
        loss = -areas[mask].sum() / mask.sum().float()
        return loss

    def face_angles_iter(self):
        loss = ((self.source_ds.vs_angles - self.fixed_angles_a) ** 2).sum(1)
        loss = (self.beautify_weights * loss).mean()
        return loss

    def fixed_iter(self):
        if self.source_fixed is None:
            loss = torch.zeros(1, device=self.device)
        else:
            loss = self.mse(self.source_ds.vs[self.source_fixed], self.target_fixed)
        return loss

    def flip_iter(self):
        new_normals = self.source_ds.normals
        loss = 1 - torch.einsum('fd,fd->f', self.base_normals, new_normals)
        loss = loss.mean() * self.iter
        return loss

    def is_nan_model(self, model: nn.Module):
        for param in model.parameters():
            if torch.isnan(param).any() or (param.grad is not None and torch.isnan(param.grad).any()):
                return True
        return False

    def backward_hook(self, grad):
        self.source_ds.invalidate()
        return grad

    def init_net(self):
        return self.mse(self.source_ds.vs, self.fixed_vs * 10)

    def callback_for_init(self, out_dir: str, params: OptimizationParams) -> Callable[[int, T], None]:
        def callback_(i: int, vs: T):
            if i == 0 or (i + 1) % params.plot_every == 0:
                vs = vs.clone().detach()
                files_utils.export_mesh((vs, self.source_mesh[1]), f'{out_dir}/{i:04d}_init',
                                        colors=(71, 237, 190))
        return callback_

    def get_beautify_weights(self) -> Union[T, int]:
        if self.beautify_pts is None or self.beautify_pts.shape[0] == 0:
            return 1
        mesh_ = mesh_utils.to_unit_cube(self.original_mesh, in_place=False)[0]
        dist_to_keys = mesh_utils.exact_geodesic(mesh_, self.beautify_pts)
        weights = (1 - dist_to_keys / dist_to_keys.max()) ** 3
        # dist_to_keys[beautify_inds] = 1
        # weights = 1 / dist_to_keys
        # weights[beautify_inds] = weights.max()
        weights = weights - weights.min()
        weights = weights / weights.max()
        # colors = torch.ones_like(mesh_[0]) * weights[:, None]
        # files_utils.export_mesh(mesh_, "./beautify_colors", colors=colors)
        weights = 1 + weights - weights.mean()
        return weights

    @property
    def source_mesh(self) -> T_Mesh:
        return self.source_ds.mesh

    @property
    def target_mesh(self) -> T_Mesh:
        return self.target_ds.mesh

    def init_const_source(self):
        ds_ = MeshDS(self.original_mesh)
        # ds_ = self.source_ds
        fixed_angles_a = self.source_ds.vs_angles  # self.source_ds.face_angles
        base_faces_probs = ring_probs(ds_, ds_.areas[ds_.v2f])
        return fixed_angles_a, base_faces_probs

    def init_const_target(self) -> TNS:
        return self.target_ds.areas, self.target_ds.vs_normals

    def get_sub_target_mesh(self, target_sub_faces):
        return self.target_mesh[0], self.target_mesh[1][target_sub_faces]

    def get_sub_source_mesh(self, source_sub_faces):
        return self.source_mesh[0], self.source_mesh[1][source_sub_faces]

    loss_mapper: Dict[str, Callable[[DrapingOpt], T]] = {'chamfer': ch_iter, 'fixed': fixed_iter,
                                                        'face_angles': face_angles_iter,
                                                        'flip': flip_iter, 'areas_reg': areas_reg,
                                                        'area_entropy': local_area_entropy,
                                                        'area_dkl': local_area_dkl,
                                                        'init_net': init_net,
                                                        'skinny_reg': skinny_reg,
                                                        }


    def init_optimization(self, ):
        self.iter = 0
        self.fixed_angles_a, self.base_faces_prob = self.init_const_source()
        self.base_normals = self.source_ds.normals
        if self.fixed_vs is None:
            self.fixed_vs = self.source_ds.vs.clone() / constants.GLOBAL_SCALE
        if self.chamfer_controller is None:
            self.chamfer_controller = ChamferController(self.source_ds, self.target_mesh, self.target_normal)

    def __init__(self, device: D, source_mesh: T_Mesh, target_mesh: T_Mesh,
                 parmas: Optional[OptimizationParams] = None):
        self.iter = 0
        self.device = device
        self.mse = nn.MSELoss()
        self.original_mesh = mesh_utils.to(source_mesh, device)
        self.source_ds = MeshDS(self.original_mesh).to(device)
        self.target_ds = get_ds(target_mesh).to(device)
        self.target_areas, self.target_normal = self.init_const_target()
        self.chamfer_controller: Optional[ChamferController] = None
        self.fixed_vs: Optional[T] = None
        self.base_normals: Optional[T] = None
        self.source_fixed: Optional[T] = None
        self.target_fixed: Optional[T] = None
        self.params = parmas
        self.fixed_angles_a: Optional[T] = None
        self.base_faces_prob: Optional[T] = None
        self.beautify_weights: Union[T, int] = 1
        self.beautify_pts: TN = None

