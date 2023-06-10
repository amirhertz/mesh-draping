import abc
# import local__ig__ as si
import constants
from utils import train_utils
from custom_types import *
from torch.nn import Parameter


class SineLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class MlpModel(nn.Module, abc.ABC):
    device = CPU

    @property
    def reset_optimizer(self) -> bool:
        return False

    def reset(self):
        return

    def increase(self, lr_scheduler: Optional[torch.optim.lr_scheduler.ExponentialLR] = None):
        return

    def before_step(self):
        return

class PositionalEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''

    def forward(self, x: T) -> T:
        pe_raw = torch.einsum('nd,f->nfd', x, self.weights).reshape(x.shape[0], -1)
        pe = x, torch.sin(pe_raw), torch.cos(pe_raw)
        pe = torch.cat(pe, dim=-1)
        return pe

    def create_weights(self):
        w = [(2 ** i) * np.pi for i in range(self.num_frequencies)]
        w = torch.tensor(w)
        return w

    def get_out_dim(self, in_fe):
        return in_fe + 2 * in_fe * self.num_frequencies

    def __init__(self, num_frequencies: int = 10):
        super().__init__()
        self.num_frequencies = num_frequencies
        weights = self.create_weights()
        self.register_buffer('weights', weights)


def fixed_uniform_(w: T):
    # m.weight.uniform_(-1 / num_input, 1 / num_input)
    num_output, num_input = w.shape
    # std = 1 / np.sqrt(w.size(1))
    # bound = np.sqrt(c) * std
    bound = (1. / num_input)
    # bound = 1.
    fixed = (torch.rand(num_output, num_input) * 2 - 1) * bound
    # fixed = torch.linspace(- bound, bound, num_output)
    fixed = fixed.gather(0, fixed.abs().argsort(0))
    with torch.no_grad():
        w[:, :] = fixed
        return w
    # fixed = torch.linspace(-1 / num_input, 1 / num_input, w.shape[0])
    # fixed_ne = fixed[: fixed.shape[0] // 2 + fixed.shape[0] % 2].__reversed__()
    # fixed_pos = fixed[fixed.shape[0] // 2:]
    # fixed = torch.stack((fixed_pos, fixed_ne), dim=1).flatten()
    # if w.shape[0] % 2:
    #     fixed = fixed[1:]
    # with torch.no_grad():
    #     w[:, :] = fixed[:, None]
    #     return w


class PositionalEncoding(nn.Module):

    def forward(self, x: T) -> T:
        return torch.cat((x, torch.sin(self.scale * self.linear(x))), dim=-1)

    def __init__(self, in_fe, out_fe):
        super(PositionalEncoding, self).__init__()
        self.linear = nn.Linear(in_fe, out_fe - in_fe, bias=False)
        fixed_uniform_(self.linear.weight)
        self.scale = 30.


class ProgressivePositionalEncoding(PositionalEncoding):

    def reset(self):
        self.cur_block = 1

    def increase(self):
        self.cur_block += 1

    def forward(self, x: T, alpha: float = 0.) -> T:
        alpha = min(1., alpha)
        out = super(ProgressivePositionalEncoding, self).forward(x)
        if self.cur_block < self.num_blocks:
            mask = torch.ones(out.shape[-1], device=out.device)
            mask[self.block_sizes[self.cur_block]: self.block_sizes[self.cur_block + 1]] = alpha
            mask[self.block_sizes[self.cur_block + 1]:] = 0
            out = out * mask[None, :]
        return out

    def __init__(self, in_fe, out_fe, num_blocks: int):
        super(ProgressivePositionalEncoding, self).__init__(in_fe, out_fe)
        self.block_sizes = [0, 2 * in_fe] + [(out_fe - 2 * in_fe) // (num_blocks - 1)] * (num_blocks - 2)
        self.block_sizes.append(out_fe - sum(self.block_sizes))
        self.block_sizes = torch.tensor(self.block_sizes, dtype=torch.int64).cumsum(0)
        self.num_blocks = num_blocks
        self.cur_block = 1

# class PositionalEncoding(nn.Module):
#
#     def forward(self, x: T) -> T:
#         x_ = torch.einsum('mn,bm->bn', self.positional_encoding, x)
#         x = torch.cat((x, torch.cos(x_), torch.sin(x_)), dim=1)
#         return x
#
#     def __init__(self, in_fe, out_fe):
#         super(PositionalEncoding, self).__init__()
#         positional_encoding: T = 2 ** torch.arange(out_fe // 2).float() * np.pi
#         positional_encoding = positional_encoding.unsqueeze(0).expand(in_fe, out_fe // 2)
#         self.register_buffer('positional_encoding', positional_encoding)


def global_weight_init(model: MlpModel, epsilon=constants.EPSILON, max_iters=100000,
                       condition: Optional[Callable[[], bool]] = None):
    optimizer = Optimizer(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)
    logger = train_utils.Logger()
    logger.start(max_iters)
    for i in range(max_iters):
        optimizer.zero_grad()
        x = torch.rand(10000, 3, device=model.device) * 2 - 1
        out = model(x)
        loss = nnf.mse_loss(out, x * constants.GLOBAL_SCALE)
        loss.backward()
        optimizer.step()
        scheduler.step()
        logger.stash_iter('mse', loss)
        logger.reset_iter()
        model.increase()
        if loss.le(epsilon).item():
            break
        if condition is not None and not condition():
            break
    logger.stop(False)
    model.reset()


def init_weights(model: MlpModel, x, target, epsilon=constants.EPSILON, max_iters=5000,
                 callback: Optional[Callable[[int, T], None]] = None,
                 condition: Optional[Callable[[], bool]] = None):
    optimizer = Optimizer(model.parameters(), lr=1e-4)
    logger = train_utils.Logger()
    logger.start(max_iters)
    for i in range(max_iters):
        optimizer.zero_grad()
        out = model(x)
        # loss = ((out - target) ** 2).sum(-1)
        # diff = loss.max()
        # loss = loss.mean()
        loss = nnf.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        logger.stash_iter('mse', loss)
        logger.reset_iter()
        if callback is not None:
            callback(i, out)
        if loss.le(epsilon).item():
            break
        if condition is not None and not condition():
            break
    logger.stop(False)
    model.reset()


class MLP(MlpModel):

    def forward(self, x):
        return self.net(x)

    def __init__(self, ch: Union[List[int], Tuple[int, ...]], act: nn.Module = nn.ReLU):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(ch) - 1):
            layers.append(nn.Linear(ch[i], ch[i + 1]))
            if i < len(ch) - 2:
                layers.append(act(True))
        self.net = nn.Sequential(*layers)


# def get_model(layers, in_fe, out_fe, act=Sine) -> nn.Module:
#     if act is Sine:
#         return SIREN(layers, in_fe, out_fe, 1.0, 30.0,
#                      initializer='siren', c=6)
#     else:
#         return MLP(tuple([in_fe] + list(layers) + [out_fe]), act=act)


class HybridMlp(MlpModel):

    def forward(self, x) -> T:
        return self.mlp(nnf.relu(self.siren_model(x), inplace=True))

    def __init__(self, in_fe: int, out_fe: int, layers_sin, layers_relu):
        super(HybridMlp, self).__init__()
        self.siren_model = Siren(in_fe, layers_relu[0], layers_sin)
        self.mlp = MLP(tuple(list(layers_relu) + [out_fe]))


class ProgressiveMlp(nn.Module):

    def calibrate(self, ds: Iterator, max_iters=5000, epsilon=constants.EPSILON):
        optimizer = Optimizer(self.parameters(), lr=1e-4)
        logger = train_utils.Logger()
        logger.start(max_iters)

        for i in range(max_iters):
            x, target = next(ds)
            siren_target = torch.zeros_like(target)
            optimizer.zero_grad()
            out_a = self.mlp(x)
            out_b = self.hybrid(x)
            loss = nnf.mse_loss(out_a, target) + nnf.mse_loss(out_b, siren_target)
            loss.backward()
            optimizer.step()
            logger.stash_iter('mse', loss)
            logger.reset_iter()
            if loss.le(epsilon).item():
                break
        logger.stop(False)

    def forward(self, x: T) -> T:
        if self.counter > self.num_iters:
            alpha = min(1., (self.counter - self.num_iters) / self.num_iters)
        else:
            alpha = 0
        self.counter += 1
        return self.mlp(x) + alpha * self.hybrid(x)

    def __init__(self, in_fe: int, out_fe: int, layers_sin: List[int], layers_relu: List[int], num_iters: int):
        super(ProgressiveMlp, self).__init__()
        self.num_iters = float(num_iters // 3)
        self.counter = 0
        self.hybrid = HybridMlp(in_fe, out_fe, layers_sin[:1], layers_sin[1:])
        self.mlp = MLP(tuple([in_fe] + list(layers_relu) + [out_fe]))


class PositionalEncodingMlp(nn.Module):

    def forward(self, x: T) -> T:
        return self.mlp(self.pe(x))

    def __init__(self, layers: List[int], in_fe: int, out_fe: int):
        super(PositionalEncodingMlp, self).__init__()
        self.pe = PositionalEncoding(in_fe, layers[0])
        self.mlp = MLP(layers + [out_fe])


def fixed_reverse_(w: T, ignore: int):
    num_output, num_input = w.shape
    select = ignore + ignore % 2 + torch.arange((num_input - ignore) // 2) * 2
    with torch.no_grad():
        w[:, select] = w[:, select + 1]
    return w


class FC(nn.Module):

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x

    def __init__(self, ch: Union[List[int], Tuple[int, ...]], in_fe):
        super(FC, self).__init__()
        layers = []
        for i in range(len(ch) - 1):
            layers.append(nn.Linear(ch[i], ch[i + 1], bias=i > 0))
            if i < len(ch) - 2:
                layers.append(nn.ReLU(True))
        fixed_reverse_(layers[0].weight, in_fe)
        self.net = nn.ModuleList(layers)


class ProgressivePositionalEncodingMLP(MlpModel):

    def reset(self):
        self.iteration = 0
        self.phase = 0
        self.pe.reset()

    def parameters(self, recurse: bool = ...) -> Iterator[Parameter]:
        # return self.mlp.parameters(recurse)
        return super(ProgressivePositionalEncodingMLP, self).parameters(recurse)

    # def calibrate_(self,epsilon=constants.EPSILON, max_iters=5000):
    #     optimizer = Optimizer(super(ProgressivePositionalEncodingMLP, self).parameters(), lr=1e-4)
    #     for i in range(max_iters):
    #         optimizer.zero_grad()
    #         out = self.mlp(self.pe(self.stash[0], 0))
    #         loss = nnf.mse_loss(out, self.stash[1])
    #         loss.backward()
    #         optimizer.step()
    #         if loss.le(epsilon).item():
    #             break
    #     print(f'{i}, {loss.item()}')
    #     self.should_calibrate = False
    #     self.stash = None

    @property
    def reset_optimizer(self) -> bool:
        if self.reset_optimizer_:
            self.reset_optimizer_ = False
            return True
        return False

    def increase(self, lr_scheduler: Optional[torch.optim.lr_scheduler.ExponentialLR] = None):
        self.iteration += 1
        if (self.iteration % self.iters_per_phase) == 0:
            if self.phase % 2 == 1:
                self.pe.increase()
                if self.pe.cur_block < self.pe.num_blocks:
                    self.reset_optimizer_ = True
            elif lr_scheduler is not None:
                lr_scheduler.step()
            self.phase += 1

    def forward(self, x: T) -> T:
        if self.phase % 2 == 1:
            alpha = float(self.iteration % self.iters_per_phase) / self.iters_per_phase
        else:
            alpha = 0
        out = self.mlp(self.pe(x, alpha))
        return out

    def forward_(self, x, alpha: float):
        if 0 < alpha < 1:
            out = (1 - alpha) * (self.mlp(self.pe(x, 0))) + alpha * self.mlp(self.pe(x, 1))
        else:
            out = self.mlp(self.pe(x, alpha))
        return out

    def before_step(self):
        return

    def __init__(self, layers: List[int], in_fe: int, out_fe: int, num_iters: int, num_blocks: int = 6):
        super(ProgressivePositionalEncodingMLP, self).__init__()
        # self.iter_split = [0] + [int((float(i + 1) / num_blocks) * num_iters) for i in range(num_blocks)]
        self.pe = ProgressivePositionalEncoding(in_fe, layers[0], num_blocks)
        self.mlp = FC([layers[0]] + layers + [out_fe], in_fe)
        # list(self.mlp.net.children())[0].register_backward_hook(self.clamp_grad)
        self.iteration = 0
        self.iters_per_phase = num_iters // (3 * num_blocks)
        self.phase = 0
        self.should_calibrate = False
        self.stash: Optional[T] = None
        self.reset_optimizer_ = False


class ProgressiveT2(MlpModel):

    def reset(self):
        self.iteration = 0
        self.phase = 0

    def increase(self, lr_scheduler: Optional[torch.optim.lr_scheduler.ExponentialLR] = None):
        self.iteration += 1
        if self.iteration % self.iters_per_phase == 0:
            self.phase += 1
            if self.phase // 2 + 1 == self.num_blocks + 1:
                print('on full progress')

    def forward_(self, x: T, final_block: int, alpha: float) -> T:
        embeddings = self.pe(x)
        if final_block > self.num_blocks:
            final_block = self.num_blocks
            alpha = 1
        out = self.models[0](embeddings.index_select(-1, self.select[0]))
        if final_block > 2:
            for i in range(1, final_block - 1):
                out += self.models[i](embeddings.index_select(-1, self.select[i]))
        if final_block > 1:
            out += alpha * self.models[final_block - 1](embeddings.index_select(-1, self.select[final_block - 1]))
        return out

    def forward(self, x) -> T:
        if self.phase % 2 == 0:
            alpha = 1
        else:
            alpha = float(self.iteration % self.iters_per_phase) / self.iters_per_phase
        out = self.forward_(x, (self.phase // 2 + 1 + self.phase % 2), alpha)
        return out

    @property
    def num_blocks(self):
        return len(self.models)

    def to(self, *args, **kwargs):
        super(ProgressiveT2, self).to(*args, **kwargs)
        device = args[0]
        for i, item in enumerate(self.select):
            self.select[i] = item.to(device)
        return self

    def decompose(self, x) -> TS:
        embeddings = self.pe(x)
        return [self.models[i](embeddings.index_select(-1, self.select[i])) for i in range(self.num_blocks)]

    def __init__(self, layers: List[int], in_fe: int, out_fe: int, num_iters: int, num_blocks: int = 6):
        super(ProgressiveT2, self).__init__()
        self.pe = PositionalEncoding(in_fe, layers[0])
        block_sizes = [2 * in_fe] + [(layers[0] - 2 * in_fe) // (num_blocks - 1)] * (num_blocks - 2)
        block_sizes.append(layers[0] - sum(block_sizes))
        block_sum = torch.tensor(block_sizes, dtype=torch.int64).cumsum(0).tolist()
        mlps = [MLP([bs] + layers + [out_fe]) for bs in block_sum]
        # block_sum = torch.tensor([0] + block_sizes, dtype=torch.int64).cumsum(0).tolist()
        self.models = nn.ModuleList(mlps)
        # self.select = [torch.arange(start=block_sum[i - 1], end=block_sum[i]) for i in range(1, len(block_sum))]
        self.select = [torch.arange(block_sum[i]) for i in range(len(block_sum))]
        # for i, item in enumerate(self.select):
        #     self.register_buffer(f'select_{i}', item)
        #     self.select[i] = self.__getattr__(f'select_{i}')
        self.iteration = 0
        self.iters_per_phase = num_iters // (2 * num_blocks)
        self.phase = 0


class ProgressiveT3(MlpModel):

    def reset(self):
        self.iteration = 0
        self.phase = 0

    def increase(self, lr_scheduler: Optional[torch.optim.lr_scheduler.ExponentialLR] = None):
        self.iteration += 1
        if self.iteration % self.iters_per_phase == 0:
            self.phase += 1
            if self.phase % 2 == 1 and lr_scheduler is not None:
                lr_scheduler.step()
            if self.phase // 2 + 1 == self.num_blocks + 1:
                print('on full progress')

    def forward_single(self, embeddings, block: int) -> T:
        embeddings = embeddings * self.masks[block][None, :]
        return self.model(embeddings)

    def forward_(self, x: T, final_block: int, alpha: float) -> T:
        embeddings = self.pe(x)
        if final_block > self.num_blocks:
            final_block = self.num_blocks
            alpha = 1
        out = self.forward_single(embeddings, final_block - 1)
        if alpha < 1:
            out_prev = self.forward_single(embeddings, final_block - 2)
            out = out * alpha + out_prev * (1 - alpha)
        return out

    def forward(self, x) -> T:
        if self.phase % 2 == 0:
            alpha = 1
        else:
            alpha = float(self.iteration % self.iters_per_phase) / self.iters_per_phase
        out = self.forward_(x, (self.phase // 2 + 1 + self.phase % 2), alpha)
        return out

    @property
    def num_blocks(self):
        return len(self.masks)

    def to(self, *args, **kwargs):
        super(ProgressiveT3, self).to(*args, **kwargs)
        device = args[0]
        for i, item in enumerate(self.masks):
            self.masks[i] = item.to(device)
        return self

    def __init__(self, layers: List[int], in_fe: int, out_fe: int, num_iters: int, num_blocks: int = 6):
        super(ProgressiveT3, self).__init__()
        self.pe = PositionalEncoding(in_fe, layers[0])
        block_sizes = [2 * in_fe] + [(layers[0] - 2 * in_fe) // (num_blocks - 1)] * (num_blocks - 2)
        block_sizes.append(layers[0] - sum(block_sizes))
        block_sum = torch.tensor(block_sizes, dtype=torch.int64).cumsum(0).tolist()
        self.model = MLP([layers[0]] + layers + [out_fe])
        masks = []
        for i in  range(len(block_sum)):
            mask = torch.zeros(block_sum[-1], dtype=torch.float32)
            mask[:block_sum[i]] = 1
            masks.append(mask)
        self.masks = masks
        self.iteration = 0
        self.iters_per_phase = num_iters // (2 * num_blocks)
        self.phase = 0


class Nerf(MlpModel):

    def forward(self, x) -> T:
        shape = x.shape
        if x.dim() != 2:
            x = x.view(-1, x.shape[-1])
        out = self.mlp(self.pe(x)).view(*shape[:-1], -1)
        return out

    def __init__(self, layers: List[int], in_fe: int, out_fe: int):
        super(Nerf, self).__init__()
        self.pe = PositionalEncodingNeRF()
        self.mlp = MLP([self.pe.get_out_dim(in_fe)] + list(layers) + [out_fe])


class Siren(MlpModel):

    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output

    def __init__(self, in_features, out_features, hidden_features, hidden_layers, outermost_linear: bool = True,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)


class ExplicitModel(MlpModel):

    def forward(self, *args) -> T:
        return self.parameter

    def __init__(self, param: T):
        super(ExplicitModel, self).__init__()
        self.parameter = Parameter(param)


def get_model(model_type: ModelType, in_fe: int, out_ch: int, layers: List[int], device: D,
              num_iters: int, param: TN) -> Tuple[MlpModel, Optimizer]:
    if model_type == ModelType.EXPLICIT:
        model = ExplicitModel(param)
        optimizer = Optimizer(model.parameters(), lr=1e-3)
    elif model_type == ModelType.ReLU:
        model = MLP(tuple([in_fe] + list(layers) + [out_ch]), act=nn.ReLU)
        optimizer = Optimizer(model.parameters(), lr=1e-3)
    elif model_type == ModelType.PPE3:
        model = ProgressiveT3(layers, in_fe, out_ch, num_iters)
        optimizer = Optimizer(model.parameters(), lr=1e-4)
    elif model_type == ModelType.PPE2:
        model = ProgressiveT2(layers, in_fe, out_ch, num_iters)
        optimizer = Optimizer(model.parameters(), lr=1e-3)
    elif model_type == ModelType.PPE:
        model = ProgressivePositionalEncodingMLP(layers, in_fe, out_ch, num_iters)
        optimizer = Optimizer(model.parameters(), lr=1e-4)
    elif model_type == ModelType.SIREN:
        model = Siren(in_fe, out_ch, hidden_features=max(layers), hidden_layers=len(layers), outermost_linear=True)
        optimizer = Optimizer(model.parameters(), lr=1e-5)
    elif model_type == ModelType.PE:
        model = Nerf(layers, in_fe, out_ch)
        optimizer = Optimizer(model.parameters(), lr=1e-5)
    elif model_type == ModelType.HYBRID:
        model = HybridMlp(in_fe, out_ch, layers[:1], layers[1:])
        optimizer = Optimizer(model.parameters(), lr=1e-5)
    else:
        raise ValueError('model type is not exist')
    model.device = device
    return model.to(device), optimizer


if __name__ == '__main__':
    device = CUDA()
    model_type = ModelType.PPE2
    model: ProgressivePositionalEncodingMLP = get_model(model_type, 3, 3, [256] * 4, device, 1000, None)[0]
    global_weight_init(model)
    torch.save(model.state_dict(), f'{constants.CACHE_ROOT}/model_{model_type.value}_{constants.GLOBAL_SCALE}.pth')
    # x = torch.rand(3, 2, device=device)
    #
    # out_a = model(x)
    # model.pe.cur_block = 5
    # out_b = model(x)
    # diff = out_a - out_b
    # print("done")
    # model = ProgressiveT3([256] * 2, 3, 3, 800).to(device)
    # for _ in range(800):
    #     x = torch.rand(1000, 3, device=device)
    #     out = model(x)
    #     model.increase()


