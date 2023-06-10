from __future__ import annotations
from custom_types import *
from utils import files_utils
import constants
from utils import train_utils
import torch.backends.cudnn as cudnn
from models import mlp_models
import multiprocessing as mp
from core import drape_opt
import os
if constants.IS_WINDOWS or 'DISPLAY' in os.environ:
    from pynput.keyboard import Key, Controller
else:
    from utils.mock_keyboard import Key, Controller


def to_np_array(shared_vs) -> ARRAY:
    return np.frombuffer(shared_vs.get_obj(), dtype=np.float32).reshape((-1, 3))


def sync_vs(shared_vs: mp.Array, new_vs: T):
    with shared_vs.get_lock():
        vs_arr = to_np_array(shared_vs)
        vs_arr[:] = new_vs.detach().cpu().numpy()


class DrapingOnline(drape_opt.DrapingOpt):

    def between_iterations(self, i: int) -> bool:
        if (i + 1) % self.params.plot_every == 0:
            # files_utils.export_mesh(self.source_ds.mesh, f'{constants.CHECKPOINTS_ROOT}/demo/target_{i:02d}')
            sync_vs(self.shared_vs, self.source_mesh[0])
            self.keyboard.press(Key.ctrl_l)
            self.keyboard.release(Key.ctrl_l)
        super(DrapingOnline, self).between_iterations(i)
        if self.fixed_status.value == OptimizingStatus.Update.value:
            self.update_fixed_points()
        while self.status.value == OptimizingStatus.PAUSE.value:
            with self.optimize_condition:
                self.optimize_condition.wait()
        if self.status.value == OptimizingStatus.WAITING.value or \
                self.status.value == OptimizingStatus.Exit.value:
            return False
        return True

    def get_model(self):
        model, optimizer = mlp_models.get_model(ModelType.PPE2, 3, 3, [256] * 4, self.device,
                                                2000, None)
        save_path = f'{constants.CACHE_ROOT}/model_ppe2_{constants.GLOBAL_SCALE}.pth'
        if os.path.isfile(save_path):
            model.load_state_dict(torch.load(save_path, map_location=self.device))
        else:
            mlp_models.global_weight_init(model, condition=lambda: self.status.value == OptimizingStatus.OPTIMIZING.value)
            torch.save(model.state_dict(),
                       f'{constants.CACHE_ROOT}/model_{ModelType.PPE2}_{constants.GLOBAL_SCALE}.pth')
        return model, optimizer

    def optimize(self):
        # optimizer = Optimizer(self.model.parameters(), lr=1e-3)
        # model, optimizer = self.get_model()
        iters = 0
        cudnn.benchmark = True
        logger = train_utils.Logger(self.global_opt_step).start(self.params.total_steps, f'draping')
        for level, steps in enumerate(self.params.steps):
            weights_project, weights_beauty = self.params.get_weights(level)
            for i in range(steps):
                self.iter = 1. - float(i) / steps
                log = self.optimize_iter(weights_project, weights_beauty, self.optimizer, self.model)
                logger.stash_iter(log)
                logger.reset_iter()
                if not (self.between_iterations(i)):
                    break
            if self.status.value == OptimizingStatus.WAITING.value or self.status.value == OptimizingStatus.Exit.value:
                files_utils.export_mesh(self.source_ds.mesh, f'{constants.CHECKPOINTS_ROOT}/demo/target_latest')
                break
            iters += steps
        self.global_opt_step += 1
        logger.stop()

    def set_params(self):
        if self.global_opt_step == 0:
            self.params = drape_opt.OptimizationParams(model_type=ModelType.PPE2, steps=(1000, 1000), plot_every=50)
        else:
            self.params = drape_opt.OptimizationParams(model_type=ModelType.PPE2, steps=(0, 500), plot_every=50)

    def update_fixed_points(self):
        fixed_points = files_utils.load_pickle(f'{constants.POINTS_CACHE}')
        if fixed_points is not None:
            source_fixed = fixed_points['source_pts']
            target_fixed = fixed_points['target_pts']
            min_size = min(source_fixed.shape[0], target_fixed.shape[0])
            self.source_fixed = source_fixed[:min_size].to(self.device)
            self.target_fixed = target_fixed[:min_size].to(self.device)
            # self.beautify_weights: T = fixed_points['beauty_pts'].to(self.device)
            # if self.beautify_weights.shape[0] == 0:
            self.beautify_weights = 1
        self.fixed_status.value = OptimizingStatus.WAITING.value

    def init_optimization(self):
        self.update_fixed_points()
        with self.shared_vs:
            arr = to_np_array(self.shared_vs)
            self.source_ds.vs = torch.from_numpy(arr).float().to(self.device)

        super(DrapingOnline, self).init_optimization()
        self.set_params()

    def start_optimize(self):
        self.init_optimization()
        self.optimize()

    def __init__(self, device: D, source_mesh: T_Mesh, target_mesh: T_Mesh, status: mp.Value, fixed_status: mp.Value,
                 optimize_condition: mp.Condition, sync_condition: mp.Condition, shared_vs: mp.Array):
        super(DrapingOnline, self).__init__(device, source_mesh, target_mesh)
        self.global_opt_step = 0
        self.status = status
        self.optimize_condition = optimize_condition
        self.sync_condition = sync_condition
        self.shared_vs = shared_vs
        self.keyboard = Controller()
        self.model, self.optimizer = self.get_model()
        self.fixed_status = fixed_status


def draping_opt_main(device: D, source_mesh: T_Mesh, target_mesh: T_Mesh, status: mp.Value, fixed_status: mp.Value,
                     optimize_condition: mp.Condition, sync_condition: mp.Condition, shared_vs: mp.Array):
    optimizer = DrapingOnline(device, source_mesh, target_mesh, status, fixed_status,
                              optimize_condition, sync_condition, shared_vs)
    while status.value != OptimizingStatus.Exit.value:
        while status.value == OptimizingStatus.WAITING.value:
            with optimize_condition:
                optimize_condition.wait()
        if status.value == OptimizingStatus.OPTIMIZING.value:
            optimizer.start_optimize()
            if status.value != OptimizingStatus.Exit.value:
                with status:
                    status.value = OptimizingStatus.Update.value
                optimizer.keyboard.press(Key.ctrl_l)
                optimizer.keyboard.release(Key.ctrl_l)
    return 0
