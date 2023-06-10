from custom_types import *
import os
from utils import files_utils
import constants as const


class Options:

    @property
    def num_conds(self) -> int:
        return len(self.mesh_name)

    @property
    def name(self) -> str:
        return f'{self.mesh_name}_{self.tag}'

    @property
    def cp_folder(self) -> str:
        return f'{const.PROJECT_ROOT}/checkpoints/{self.name}'

    @property
    def save_path(self) -> str:
        return f'{self.cp_folder}/options.pkl'

    @property
    def already_saved(self) -> bool:
        return os.path.isfile(self.save_path)

    @property
    def in_nf(self):
        return 4

    @property
    def debug(self):
        return 'debug' in self.tag.lower() or const.DEBUG

    @property
    def out_nf(self):
        return 3

    def items(self) -> Iterator[str]:
        return filter(lambda a: not a.startswith('__') and not callable(getattr(self, a)), dir(self))

    def as_dict(self) -> dict:
        return {item: getattr(self, item) for item in self.items()}

    def save(self, force: bool = False):
        if not self.already_saved or False:
            files_utils.save_pickle(self, self.save_path)

    def load(self):
        if self.already_saved:
            loaded = files_utils.load_pickle(self.save_path)
            print(f'loading options from {self.save_path}')
            return backward_compatibility(loaded)
        return backward_compatibility(self)

    def fill_args(self, args):
        for arg in args:
            if hasattr(self, arg):
                setattr(self, arg, args[arg])

    def __init__(self, **kwargs):
        self.tag = 'trial_disc'
        self.mesh_name = 'vase_10'
        # self.template_name = 'sphere'
        self.start_level = 0
        self.num_levels = 3
        self.start_nf, self.min_nf, self.max_nf = 64, 64, 256
        self.num_layers = 12
        self.num_heads = 4
        self.update_axes = False
        self.noise = [0] * self.num_levels
        self.cyclic = True
        self.local_features = [True] * (self.num_levels + 1)
        self.local_updates = [True] * (self.num_levels + 1)
        self.symmetrical_ambiguity = (False, False, False)
        self.reconstruction_weight = 3
        self.penalty_weight = 0.1
        # training params
        self.lr = 1e-3
        self.betas = (.5, .99)
        self.lr_decay = 0.5
        self.lr_decay_every = 1000
        self.export_meshes_every = 400
        self.level_iters = [12000] * self.num_levels
        self.generator_iters = 1
        self.discriminator_iters = 0
        if self.debug:
            self.level_iters = [5] * self.num_levels
        self.fill_args(kwargs)


def backward_compatibility(opt: Options) -> Options:
    to_save = False
    defaults = {'generator_iters': 1, 'discriminator_iters': 0}
    for key, value in defaults.items():
        if not hasattr(opt, key):
            setattr(opt, key, value)
            to_save = True
    if to_save:
        opt.save(True)
    return opt


def copy(opt: Options) -> Options:
    opt_copy = opt.__class__()
    for item in opt_copy.items():
        if hasattr(opt, item):
            try:
                setattr(opt_copy, item, getattr(opt, item))
            except:
                continue
    return opt_copy


