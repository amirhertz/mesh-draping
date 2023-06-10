from matplotlib import pyplot as plt
from PIL import Image
from custom_types import *
from custom_types import T
from utils import files_utils
import imageio


def resize(image_arr: ARRAY, max_min_edge_length: int) -> ARRAY:
    h, w, _ = image_arr.shape
    min_edge_length = min(w, h)
    if min_edge_length <= max_min_edge_length:
        return image_arr
    image = Image.fromarray(image_arr)
    s = max_min_edge_length / float(min_edge_length)
    size = (int(w * s), int(h * s))
    image = image.resize(size, resample=Image.BICUBIC)
    return V(image)


def gifed(folder: str, interval: float, name: str, filter_by: Optional[Callable[[List[str]], bool]] = None,
          loop: int = 0, split: int = 1, reverse: bool = True):
    files = files_utils.collect(folder, '.png')
    if filter_by is not None:
        files = list(filter(filter_by, files))
    files = sorted(files, key=lambda x: x[1])
    # files = sorted(files, key=lambda x: int(x[1].split('_L')[-1]))
    if len(files) > 0:
        images = [[imageio.imread(''.join(file)) for file in files]]
        if split > 1:
            images_ = []
            for i, image in enumerate(images[0]):
                if i % split == 0:
                    images_.append([])
                images_[-1].append(image)
            images = images_
        for i, group in enumerate(images):
            if reverse:
                group_ = group.copy()
                group_.reverse()
                group = group + group_
                interval_ = interval
            else:
                interval_ = [interval] * len(group)
                interval_[0] = 1.5
                interval_[-1] = 1.5
            imageio.mimsave(f'{folder}{name}{str(i) if split > 1 else ""}.gif', group, duration=interval_, loop=loop)


def get_offsets(image_, margin):
    white = np.equal(image_.sum(2), 255 * 3)
    white_rows = np.equal(white.sum(1), image_.shape[1])
    white_cols = np.equal(white.sum(0), image_.shape[0])
    offset_top, offset_bottom = np.where(~white_rows)[0].min() - margin, np.where(~white_rows)[0].max() + 1 + margin
    offset_left, offset_right = np.where(~white_cols)[0].min() - margin, np.where(~white_cols)[0].max() + 1 + margin
    offset_top, offset_left = max(offset_top, 0), max(offset_left, 0)
    offset_bottom, offset_right = min(offset_bottom, image_.shape[0]), min(offset_right, image_.shape[1])
    return offset_top, offset_left, offset_bottom, offset_right


def crop_white(root: str, in_place=True, offset=1, as_first=False, alpha=True):
    paths = files_utils.collect(root, '.jpg', '.png')
    offset_top = offset_left = 1000000
    offset_bottom = offset_right = -23
    if as_first:
        for path in paths:
            image = files_utils.load_image(''.join(path))
            offset_top_, offset_left_, offset_bottom_, offset_right_ = get_offsets(image, offset)
            offset_top, offset_left = min(offset_top, offset_top_), min(offset_left, offset_left_)
            offset_bottom, offset_right = max(offset_bottom, offset_bottom_), max(offset_right, offset_right_)
    for path in paths:
        image = files_utils.load_image(''.join(path))
        if not in_place:
            new_path = list(path)
            new_path[1] = f'{new_path[1]}_cropped'
        else:
            new_path = path
        white = np.equal(image.sum(2), 255 * 3)
        if not as_first:
            offset_top, offset_left, offset_bottom, offset_right = get_offsets(image, offset)
        image = image[offset_top: offset_bottom, offset_left: offset_right]
        if alpha:
            alpha_ = (1 - white[offset_top: offset_bottom, offset_left: offset_right].astype(image.dtype)) * 255
            alpha_ = np.expand_dims(alpha_, 2)
            image = np.concatenate((image, alpha_), axis=2)
        files_utils.save_image(image, ''.join(new_path))


def change_mesh_color(paths, color):
    for path in paths:
        mesh = files_utils.load_mesh(path)
        files_utils.export_mesh(mesh, path, colors=color)


def to_heatmap(vals) -> T:
    if type(vals) is T:
        vals = vals.detach().cpu().numpy()
    vals = (vals * 255).astype(np.uint8)
    colormap = plt.get_cmap('inferno')
    np_heatmap = colormap(vals)[:, :3]
    # np_heatmap = np.ascontiguousarray(cv2.applyColorMap(np_vals, cv2.COLORMAP_HOT)[:, 0, ::-1])
    heatmap = torch.from_numpy(np_heatmap).float()
    return heatmap


def align_meshes(source: T_Mesh, target: T_Mesh) -> T_Mesh:

    def get_range(vs_):
        max_vals = vs.max(0)[0]
        min_vals = vs.min(0)[0]
        max_range = (max_vals - min_vals).max() / 2
        center = (max_vals + min_vals) / 2
        return center, max_range

    vs, faces = source
    center_a, scale_a = get_range(vs)
    vs, faces = target
    vs = vs.clone()
    center_b, scale_b = get_range(vs)
    vs = (vs - center_b[None, :]) * scale_a / scale_b +  center_a[None, :]
    return vs, faces


def less_obj():
    paths = files_utils.collect(r'C:\Users\t-amhert\PycharmProjects\mesh_redress\out_faust\50025_one_leg_loose_01_siren/', '.obj')
    paths = list(filter(lambda x: 'tar' not in x[1], paths))
    paths.sort(key=lambda x: int(x[1]))
    for i, path in enumerate(paths):
        if i % 5 != 0:
            files_utils.delete_single(''.join(path))


if __name__ == '__main__':
    import constants
    # less_obj()
    def is_int(s: str):
        try:
            x = int(s)
        except ValueError:
            return False
        return True
    path = '/home/amir/projects/mesh_redress/renders'
    # mesh = files_utils.load_mesh(path)
    # mesh_b = files_utils.load_mesh(f'{constants.RAW_ROOT}face_source_reg_adapt')
    # mesh = align_meshes(mesh_b, mesh)
    # files_utils.export_mesh(mesh, path)

    # paths = files_utils.collect(f'{constants.OUT_ROOT}face_source_reg_adapt_face_target_glasses_ppe2', '.obj')
    # paths = list(map(lambda x: ''.join(x), filter(lambda x: x[1] == 'project' or is_int(x[1]), paths)))
    # change_mesh_color(paths, (255, 135, 179))
    # gifed(r'C:\Users\t-amhert\PycharmProjects\mesh_redress\plots\3dparam\pe/',
    #       .12, 'opt')
    # gifed(path, .2, 'opt', loop=1, split=61, filter_by=lambda x: 'target' not in x, reverse=False)
    # crop_white(path, True, 10, as_first=True, alpha=False)