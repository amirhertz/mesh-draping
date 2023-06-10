from custom_types import *
import sys
import vtk
import vtk.util.numpy_support as numpy_support
from utils import files_utils, ui_utils, mesh_utils
import constants
from utils.ui_utils import MatchStatus
import colorsys


class ToggleInteractor(vtk.vtkInteractorStyleTrackballCamera):

    def update_default(self):
        click_pos_x = self.GetInteractor().GetEventPosition()[0]
        if click_pos_x < self.renders[0].GetSize()[0]:
            new_render = 0
        elif click_pos_x < self.renders[0].GetSize()[0] + self.renders[1].GetSize()[0]:
            new_render = 1
        else:
            new_render = 2
        if new_render != self.cur_render:
            self.SetDefaultRenderer(self.renders[new_render])
            self.cur_render = new_render

    def left_button_press_event(self, obj, event):
        self.update_default()
        self.OnLeftButtonDown()
        return

    def right_button_press_event(self, obj, event):
        self.update_default()
        for ren in self.renders:
            actors = ren.GetActors()
            for i in range(2, actors.GetNumberOfItems()):
                actor = actors.GetItemAsObject(i)
                actor.GetProperty().SetOpacity(float(self.status))
        self.status = not self.status
        self.OnRightButtonDown()
        return

    def update_camera(self):
        camera = self.renders[self.cur_render].GetActiveCamera()
        camera_position = camera.GetPosition()
        camera_focal_point = camera.GetFocalPoint()
        clipping_range = camera.GetClippingRange()
        distance = camera.GetDistance()
        focal_distance = camera.GetFocalDistance()
        view_up = camera.GetViewUp()
        roll = camera.GetRoll()
        for i in range(len(self.renders)):
            if i != self.cur_render:
                camera = self.renders[i].GetActiveCamera()
                camera.SetPosition(*camera_position)
                camera.SetFocalPoint(*camera_focal_point)
                camera.SetClippingRange(*clipping_range)
                camera.SetDistance(distance)
                camera.SetFocalDistance(focal_distance)
                camera.SetViewUp(*view_up)
                camera.SetRoll(roll)

    def on_mouse_move_event(self, obj, event):
        self.update_camera()
        self.OnMouseMove()
        return

    def on_left_release(self, obj, event):
        self.update_camera()
        self.OnLeftButtonUp()
        return

    def zoom_in(self, obj, event):
        self.update_camera()
        self.OnMouseWheelForward()
        return

    def zoom_out(self, obj, event):
        self.update_camera()
        self.OnMouseWheelBackward()
        return

    def __init__(self, *renders):
        self.AddObserver("RightButtonPressEvent", self.right_button_press_event)
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.AddObserver("LeftButtonReleaseEvent", self.on_left_release)
        self.AddObserver("MouseMoveEvent", self.on_mouse_move_event)
        self.AddObserver("MouseWheelForwardEvent", self.zoom_in)
        self.AddObserver("MouseWheelBackwardEvent", self.zoom_out)
        self.cur_render = 0
        self.status = True
        self.renders = renders


# vtk.vtkBorderWidget
class CustomTextWidget(vtk.vtkTextWidget):

    title = "Draping"
    stop_text = "Stop"

    def set_to_start(self):
        self.EnabledOff()
        self.GetRepresentation().SetRenderer(self.ren[0])
        self.SetCurrentRenderer(self.ren[0])
        self.GetTextActor().SetInput(self.title)
        h, s, v = colorsys.rgb_to_hsv(*ui_utils.bg_target_color)
        # self.GetTextActor().GetTextProperty().SetColor(*ui_utils.rgb_to_float(colorsys.hsv_to_rgb(h, .5, 255)))
        self.GetTextActor().GetTextProperty().SetColor(*ui_utils.rgb_to_float(colorsys.hsv_to_rgb(h, .5, 200)))
        self.GetRepresentation().GetPositionCoordinate().SetValue(.85, 0.95)
        self.GetRepresentation().GetPosition2Coordinate().SetValue(0.14, 0.03)
        self.EnabledOn()
        self.GetInteractor().Render()

    def set_to_stop(self):
        self.EnabledOff()
        self.GetRepresentation().SetRenderer(self.ren[1])
        self.SetCurrentRenderer(self.ren[1])
        self.GetTextActor().SetInput(self.stop_text)
        h, s, v = colorsys.rgb_to_hsv(*ui_utils.bg_source_color)
        # self.GetTextActor().GetTextProperty().SetColor(*ui_utils.rgb_to_float(colorsys.hsv_to_rgb(h, .9, 255)))
        self.GetTextActor().GetTextProperty().SetColor(*ui_utils.rgb_to_float(colorsys.hsv_to_rgb(h, .5, 200)))
        self.GetRepresentation().GetPositionCoordinate().SetValue(0.01, 0.95)
        self.GetRepresentation().GetPosition2Coordinate().SetValue(0.08, 0.03)
        self.EnabledOn()
        self.GetInteractor().Render()

    def left_button_press_event(self, obj, event):
        # if self.on:
        #     self.set_to_start()
        # else:
        #     self.set_to_stop()
        # self.on = not self.on
        if self.status.value == OptimizingStatus.OPTIMIZING.value:
            self.set_to_start()
            self.status.stop_optimize()
            # self.status.after_affine and
        elif self.status.value == OptimizingStatus.WAITING.value: # and self.status.after_affine:
            self.status.start_optimize()
            self.set_to_stop()

    def __init__(self, status: ui_utils.MatchStatus,
                 interactor: vtk.vtkRenderWindowInteractor, ren: Tuple[vtk.vtkRenderer, vtk.vtkRenderer]):
        super(CustomTextWidget, self).__init__()
        # self.interactor = interactor
        self.ren = ren
        self.SetInteractor(interactor)
        text_actor = vtk.vtkTextActor()
        text_representation = vtk.vtkTextRepresentation()
        self.SetRepresentation(text_representation)

        self.SetTextActor(text_actor)
        self.set_to_start()
        # self.Set
        self.SelectableOn()
        self.SetResizable(0)
        self.EnabledOn()
        self.On()
        self.AddObserver(vtk.vtkCommand.EndInteractionEvent, self.left_button_press_event)
        self.on = False
        self.status = status
        self.ren = ren


class MouseInteractorHighLightActor(vtk.vtkInteractorStyleTrackballCamera):

    class MouseStatus:

        def update(self, pos: Tuple[int, int], is_left: bool) -> bool:
            is_changed = is_left != self.is_left or (pos[0] - self.last_pos[0]) ** 2 > 4 and (pos[1] - self.last_pos[1]) ** 2 > 4
            self.is_left = is_left
            self.last_pos = pos
            return is_changed

        def __init__(self):
            self.last_pos = (0, 0)
            self.is_left = True

    @staticmethod
    def sync_cameras(source_render, *target_renders):
        camera = source_render.GetActiveCamera()
        camera_position = camera.GetPosition()
        camera_focal_point = camera.GetFocalPoint()
        clipping_range = camera.GetClippingRange()
        distance = camera.GetDistance()
        focal_distance = camera.GetFocalDistance()
        view_up = camera.GetViewUp()
        roll = camera.GetRoll()
        for i in range(len(target_renders)):
            camera = target_renders[i].GetActiveCamera()
            camera.SetPosition(*camera_position)
            camera.SetFocalPoint(*camera_focal_point)
            camera.SetClippingRange(*clipping_range)
            camera.SetDistance(distance)
            camera.SetFocalDistance(focal_distance)
            camera.SetViewUp(*view_up)
            camera.SetRoll(roll)

    def update_camera(self):
        if self.view_status == 'sync':
            if self.is_left_viewport:
                self.sync_cameras(self.left_ren, self.right_ren)
            else:
                self.sync_cameras(self.right_ren, self.left_ren)

    def on_mouse_move_event(self, obj, event):
        self.update_camera()
        self.OnMouseMove()
        return

    def on_left_release(self, obj, event):
        self.update_camera()
        self.OnLeftButtonUp()
        return

    def zoom_in(self, obj, event):
        self.update_camera()
        self.OnMouseWheelForward()
        return

    def zoom_out(self, obj, event):
        self.update_camera()
        self.OnMouseWheelBackward()
        return

    def replace_vs(self, new_vs: ARRAY):
        new_vs_vtk = numpy_support.numpy_to_vtk(new_vs)
        actors = self.left_ren.GetActors()
        for i in range(2):
            vs_vtk = actors.GetItemAsObject(i).GetMapper().GetInput().GetPoints()
            vs_vtk.SetData(new_vs_vtk)
        for i in range(2, actors.GetNumberOfItems()):
            vs_vtk = self.left_ren.GetActors().GetItemAsObject(i).GetMapper().GetInput().GetPoints()
            old_vs: ARRAY = numpy_support.vtk_to_numpy(vs_vtk.GetData())
            update_vs = old_vs - old_vs.mean(0)[None, :] + new_vs[self.status.all_left[i - 2]][None, :]
            vs_vtk.SetData(numpy_support.numpy_to_vtk(update_vs))

    def change_status(self):
        self.status.get_deform()
        if self.status.num_points >= self.status.min_affine_points:
            self.view_status = 'sync'
            self.sync_cameras(self.right_ren, self.left_ren)
        else:
            self.view_status = 'init'

    @staticmethod
    def replace_sphere(mapper, radius: float):
        x, y, z = mapper.GetInput().GetCenter()
        source = ui_utils.get_new_sphere(radius, x, y, z)
        mapper.SetInputConnection(source.GetOutputPort())

    def reset_camera(self):
        self.left_ren.GetActiveCamera().SetPosition(*self.default_camera_left[0])
        self.left_ren.GetActiveCamera().SetFocalPoint(*self.default_camera_left[1])
        self.right_ren.GetActiveCamera().SetPosition(*self.default_camera_right[0])
        self.right_ren.GetActiveCamera().SetFocalPoint(*self.default_camera_right[1])

    def change_opacity(self, sphere_ids: List[int], is_left: bool, on: bool):
        actors = self.left_ren.GetActors() if is_left else self.right_ren.GetActors()
        for sphere_ids in sphere_ids:
            properties = actors.GetItemAsObject(sphere_ids).GetProperty()
            properties.SetOpacity(float(on))

    def toggle_poitns(self):
        to_toggle = beauty_on = False
        if self.status.deform_type == DeformType.BEAUTY and self.point_view != DeformType.BEAUTY:
            to_toggle = beauty_on = True
        elif self.status.deform_type != DeformType.BEAUTY and self.point_view == DeformType.BEAUTY:
            to_toggle, beauty_on= True, False
        if to_toggle:
            self.change_opacity( self.status.beauty_actors, True, beauty_on)
            self.change_opacity(self.status.left_actors, True, not beauty_on)
            self.change_opacity(self.status.right_actors, False, not beauty_on)
            self.GetInteractor().Render()
        self.point_view = self.status.deform_type

    def on_key_press(self, obj, event):
        key: str = self.GetInteractor().GetKeySym().lower()
        if key == 'return' or key == 'kp_enter':
            self.change_status()
        elif key == 'r' and self.status.after_affine:
            self.status.set_deform_type(DeformType.RIGID)
        elif key == 'b':
            self.status.set_deform_type(DeformType.BEAUTY)
        elif key == 'd':
            self.status.set_deform_type(DeformType.HARMONIC)
            print(self.status.deform_type.value)
        elif key == 'space' and self.status.after_affine:
            self.status.set_deform_type(DeformType.HARMONIC if self.status.deform_type is not DeformType.HARMONIC
                                        else DeformType.RIGID)
        elif key == 'control_l':
            self.status.optimizer_callback()
            if self.status.value == OptimizingStatus.Update.value:
                self.button.set_to_start()
                self.status.resume_optimization()
            self.GetInteractor().Render()
        # elif key == 'shift_l':
        #     print(self.status.toggle_symmetry().value)
        # elif key == 'control_l':
        #     print(f'left symmetry axis is {self.status.toggle_symmetry_axis(True)}')
        # elif key == 'control_r':
        #     print(f'right symmetry axis is {self.status.toggle_symmetry_axis(False)}')
        if key in ['r', 'b', 'd', 'space']:
            self.toggle_poitns()
            print(self.status.deform_type.value)
        # else:
        #     print(key)
        return
    
    def update_default(self):
        clickPos = self.GetInteractor().GetEventPosition()
        picker = vtk.vtkPropPicker()
        is_left_picker = clickPos[0] < self.GetDefaultRenderer().GetSize()[0]
        if is_left_picker != self.is_left_viewport:
            if is_left_picker:
                self.SetDefaultRenderer(self.left_ren)
            else:
                self.SetDefaultRenderer(self.right_ren)
            self.is_left_viewport = is_left_picker
        picker.Pick(clickPos[0], clickPos[1], 0, self.GetDefaultRenderer())
        return picker, self.mouse_status.update((clickPos[0], clickPos[1]), is_left_picker)

    def right_button_release_event(self, obj, event):
        _, is_changed = self.update_default()
        if not is_changed: # and self.status.value != OptimizingStatus.OPTIMIZING.value:
            is_left, actor_numbers = self.status.undo()
            for actor_number in actor_numbers:
                if actor_number is not None:
                    if is_left:
                        self.left_ren.RemoveActor(self.left_ren.GetActors().GetItemAsObject(actor_number))
                        # actor = self.left_ren.GetActors().GetItemAsObject(actor_number)
                        # self.replace_sphere(actor.GetMapper(), .2)
                        # ui_utils.set_default_properties(actor, (1., 1., 1.))
                    else:
                        self.right_ren.RemoveActor(self.right_ren.GetActors().GetItemAsObject(actor_number))
        self.update_camera()
        self.OnLeftButtonUp()
        return

    def add_point(self, object_id, xyz):
        color, should_update, mappers_ = self.status.update(object_id, self.is_left_viewport, xyz)
        if should_update:
            for mapper_ in mappers_:
                if mapper_ is not None:
                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper_)
                    if self.is_left_viewport:
                        self.left_ren.AddActor(actor)
                    else:
                        self.right_ren.AddActor(actor)
                    ui_utils.set_default_properties(actor, color)

    def left_button_release_event(self, obj, event):
        picker, is_changed = self.update_default()
        if not is_changed:  #and self.status.value != OptimizingStatus.OPTIMIZING.value:
            # get the new
            self.NewPickedActor = picker.GetActor()
            if self.NewPickedActor:
                mapper = self.NewPickedActor.GetMapper()
                object_id = mapper.GetAddressAsString('')
                xyz = picker.GetPickPosition()
                self.add_point(object_id, xyz)
        self.update_camera()
        self.OnLeftButtonUp()
        return

    def left_button_press_event(self, obj, event):
        _ = self.update_default()
        return self.OnLeftButtonDown()

    def right_button_press_event(self, obj, event):
        _ = self.update_default()
        return self.OnRightButtonDown()

    @staticmethod
    def get_default(ren):
        position = ren.GetActiveCamera().GetPosition()
        focal_point = ren.GetActiveCamera().GetFocalPoint()
        return position, focal_point

    def set_defaults(self):
        self.default_camera_left = self.get_default(self.left_ren)
        self.default_camera_right = self.get_default(self.right_ren)

    def add_buttons(self):
        self.button = CustomTextWidget(self.status, self.GetInteractor(), (self.left_ren, self.right_ren))

    def load_points(self):
        try:
            source_points, beatify_points, target_points, deform_types = self.status.load_points()
        except BaseException:
            return
        if source_points is not None:
            is_left_ = self.is_left_viewport
            object_id = list(self.status.left_mapper.keys())[0], list(self.status.right_mapper.keys())[0]
            for i, (point_a, point_b, deform_type) in enumerate(zip(source_points.tolist(),
                                                                    target_points.tolist(), deform_types)):
                self.status.set_deform_type(deform_type)
                self.is_left_viewport = True
                self.add_point(object_id[0], point_a)
                self.is_left_viewport = False
                self.add_point(object_id[1], point_b)
            self.is_left_viewport = is_left_
            self.GetInteractor().Render()

    def __init__(self, status: MatchStatus, left_ren, right_ren):
        super(MouseInteractorHighLightActor, self).__init__()
        self.mouse_status = self.MouseStatus()
        self.AddObserver(vtk.vtkCommand.KeyPressEvent, self.on_key_press)
        self.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self.left_button_press_event)
        self.AddObserver(vtk.vtkCommand.LeftButtonReleaseEvent, self.left_button_release_event)
        self.AddObserver(vtk.vtkCommand.RightButtonPressEvent, self.right_button_press_event)
        self.AddObserver(vtk.vtkCommand.RightButtonReleaseEvent, self.right_button_release_event)
        self.AddObserver(vtk.vtkCommand.MouseMoveEvent, self.on_mouse_move_event)
        self.AddObserver(vtk.vtkCommand.MouseWheelForwardEvent, self.zoom_in)
        self.AddObserver(vtk.vtkCommand.MouseWheelBackwardEvent, self.zoom_out)
        self.status = status
        self.status.update_ui = self.replace_vs
        self.LastPickedActor = None
        self.LastPickedProperty = vtk.vtkProperty()
        self.left_ren = left_ren
        self.right_ren = right_ren
        self.default_camera_right = self.default_camera_left = None
        self.is_left_viewport = True
        self.view_status = 'init'
        self.point_view = DeformType.HARMONIC
        self.button: Optional[CustomTextWidget] = None


def access_each_vertex(mesh, render, sphere_dict_source):
    vs = mesh.GetPoints()
    num_vertices = vs.GetNumberOfPoints()
    vs = vs.GetData()

    for i in range(num_vertices):
        x = vs.GetComponent(i, 0)
        y = vs.GetComponent(i, 1)
        z = vs.GetComponent(i, 2)
        source = ui_utils.get_new_sphere(.2, x, y, z)
        actor, mapper = ui_utils.wrap_mesh(source, (1., 1., 1.))
        sphere_dict_source[mapper.GetAddressAsString('')] = (-1, i)
        render.AddActor(actor)


def add_points(render, xyz: T, colors):
    for coords, color in zip(xyz.tolist(), colors):
        source = ui_utils.get_new_sphere(.2, *coords)
        actor, _ = ui_utils.wrap_mesh(source, color)
        actor.GetProperty().SetOpacity(0)
        render.AddActor(actor)


def add_point_cloud(path, render):
    pc = ui_utils.VtkPointCloud(path)
    render.AddActor(pc.vtkActor)
    return pc, pc.mapper


def add_mesh(path, render, add_mesh=True, add_edges=True):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(path)
    reader.Update()
    mesh = reader.GetOutput()
    vs_vtk = mesh.GetPoints()
    old_vs = numpy_support.vtk_to_numpy(vs_vtk.GetData())
    new_vs = mesh_utils.to_unit_cube(torch.from_numpy(old_vs), scale=constants.GLOBAL_SCALE)[0]
    vs_vtk.SetData(numpy_support.numpy_to_vtk(new_vs.numpy()))
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    if add_mesh:
        actor_mesh = vtk.vtkActor()
        actor_mesh.SetMapper(mapper)
        render.AddActor(actor_mesh)
    if add_edges:
        actor_edge = vtk.vtkActor()
        actor_edge.SetMapper(mapper)
        actor_edge.GetProperty().SetRepresentationToWireframe()
        actor_edge.GetProperty().SetLineWidth(.5)
        actor_edge.GetProperty().SetDiffuseColor(.3, .3, .3)
        render.AddActor(actor_edge)
    return mesh, mapper


def main_ui(source_path, target_path, load: bool = True):
    source_path, target_path = files_utils.add_suffix(source_path, '.obj'), files_utils.add_suffix(target_path, '.obj')
    root, source_name, _ = files_utils.split_path(source_path)
    target_name = files_utils.split_path(target_path)[1]
    status = MatchStatus(f'{root}/{source_name}_{target_name}', source_path, target_path)
    ren_left = vtk.vtkRenderer()
    ren_left.SetViewport(0.0, 0.0, 0.5, 1.0)
    ren_right = vtk.vtkRenderer()
    ren_right.SetViewport(0.5, 0.0, 1.0, 1.0)
    ren_left.SetBackground(*ui_utils.rgb_to_float(ui_utils.bg_source_color))
    ren_right.SetBackground(*ui_utils.rgb_to_float(ui_utils.bg_target_color))
    ren_window = vtk.vtkRenderWindow()
    ren_window.AddRenderer(ren_right)
    ren_window.AddRenderer(ren_left)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_window)
    style = MouseInteractorHighLightActor(status, ren_left, ren_right)
    style.SetDefaultRenderer(ren_left)
    iren.SetInteractorStyle(style)
    if status.pc_mode:
        _, mapper_ = add_point_cloud(target_path, ren_right)
    else:
        _, mapper_ = add_mesh(target_path, ren_right, add_edges=False)
    status.right_mapper[mapper_.GetAddressAsString('')] = (-1, 0)
    # access_each_vertex(mesh_, ren_right, status.right_mapper)
    _, mapper_ = add_mesh(source_path, ren_left)
    status.left_mapper[mapper_.GetAddressAsString('')] = (-1, 0)
    style.add_buttons()
    if load:
        style.load_points()
    # access_each_vertex(mesh_, ren_left,  status.left_mapper)
    iren.Initialize()
    ren_window.Render()
    style.set_defaults()
    iren.Start()
    del iren
    del ren_window
    status.exit()


PC_MODE = False


def main():
    source_path = f'{constants.RAW_ROOT}{sys.argv[1]}'
    target_path = f'{constants.RAW_ROOT}{sys.argv[2]}'
    main_ui(source_path, target_path, False)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
