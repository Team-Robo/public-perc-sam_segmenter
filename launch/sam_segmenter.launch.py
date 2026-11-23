from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml


def launch_segmenter(context):
    get = lambda name: LaunchConfiguration(name).perform(context)
    overrides = {'target_object': get('target_object'), 'use_sim_time': get('use_sim_time') == 'true'}
    for name, converter in dict(model_family=str, model_path=str, num_threads=int,
                                conf_threshold=float, use_gpu=lambda value: value == 'true',
                                always_hot=lambda value: value == 'true',
                                single_frame_mode=lambda value: value == 'true').items():
        if get(name):
            overrides[name] = converter(get(name))
    if get('model_family') == 'sam2' and not get('model_path'):
        overrides['model_path'] = 'models/sam2.1_hiera_tiny_q4_0.ggml'
    if get('box_prompt'):
        box = yaml.safe_load(get('box_prompt'))
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError('box_prompt must be [x0, y0, x1, y1]')
        overrides['box_prompt'] = [float(coordinate) for coordinate in box]
    for name in ('use_gpu', 'use_sim_time', 'single_frame_mode', 'always_hot'):
        if get(name) and get(name) not in ('true', 'false'):
            raise ValueError(name + ' must be true or false')
    return [Node(package='sam_segmenter', executable='sam_segmenter_node', name='sam_segmenter_node',
                 parameters=[get('params_file'), overrides], output='screen')]


def generate_launch_description():
    params = str(Path(get_package_share_directory('sam_segmenter')) / 'config/params.yaml')
    arguments = dict(params_file=params, target_object='cup', use_sim_time='false',
                     model_family='', model_path='', use_gpu='', always_hot='', num_threads='',
                     box_prompt='', conf_threshold='', single_frame_mode='')
    return LaunchDescription([DeclareLaunchArgument(name, default_value=value)
                              for name, value in arguments.items()] + [OpaqueFunction(function=launch_segmenter)])
