"""RGB-only text SAM node with fresh triggers and a bounded latest-frame slot."""
import gc
import json
import threading
import time
import signal
from pathlib import Path

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from sensor_msgs.msg import Image
from core_r1_interfaces.msg import InstanceSegmentation, SegmentedInstance
import uuid
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from .targets import parse_targets
from .worker_client import WorkerClient


def stamp_ns(header):
    return header.stamp.sec * 1000000000 + header.stamp.nanosec


def rgb_array(message, max_pixels):
    if message.encoding not in ('rgb8', 'bgr8'):
        raise ValueError('SAM3 requires rgb8 or bgr8 input')
    if not 0 < message.height * message.width <= max_pixels:
        raise ValueError('Camera image exceeds max_image_pixels or is empty')
    if message.step < message.width * 3 or len(message.data) != message.height * message.step:
        raise ValueError('Camera image has an invalid step or data size')
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    image = rows[:, :message.width * 3].reshape(message.height, message.width, 3)
    return np.ascontiguousarray(image[:, :, ::-1] if message.encoding == 'bgr8' else image)


def resolve_checkpoint(path):
    if not path:
        raise ValueError('Set sam3_checkpoint_path to your locally provisioned checkpoint')
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    from ament_index_python.packages import get_package_share_directory
    choices = [Path(get_package_share_directory('sam_segmenter')) / candidate,
               Path(__file__).resolve().parents[1] / candidate, candidate]
    for choice in choices:
        if choice.is_file():
            return str(choice.resolve())
    raise ValueError('SAM3 checkpoint not found: ' + path)


class Sam3Node(Node):
    def __init__(self, *, worker=None, **kwargs):
        super().__init__('sam_segmenter_node', **kwargs)
        self.cv = threading.Condition(threading.RLock())
        self.latest = None
        self.last_header = None
        self.last_height = 0
        self.last_width = 0
        self.last_stamp = -1
        self.last_clock = self.get_clock().now().nanoseconds
        self.generation = 0
        self.epoch = uuid.uuid4().hex
        self.acquisition_id = 0
        self.request = None
        self.processing = False
        self.stopping = False
        self.last_error = ''
        self.last_result = 'waiting_for_input'
        self.declare_parameter('target_object', '')
        self.targets = parse_targets(self.get_parameter('target_object').value)
        defaults = dict(preload_models=True, release_cuda_cache=False, sam_mode='sam3', single_frame_mode=False,
                        image_topic='/camera/camera/color/image_raw', mask_topic='/perc/grasp/mask',
                        instances_topic='/perc/grasp/instances', source_id='', arm_id='',
                        detection_topic='/perc/grasp/detections', inference_header_topic='/perc/grasp/segmentation_input',
                        sam3_python='', sam3_checkpoint_path='',
                        sam3_device='cuda', sam3_confidence_threshold=0.5, sam3_bpe_path='',
                        startup_timeout=120.0, acquisition_timeout=25.0, inference_timeout=10.0,
                        max_image_pixels=4194304)
        for name, value in defaults.items():
            self.declare_parameter(name, value, ParameterDescriptor(read_only=True))
        get = lambda name: self.get_parameter(name).value
        if get('sam_mode') != 'sam3':
            raise ValueError('The text SAM node requires sam_mode:=sam3')
        self.single = get('single_frame_mode')
        self.acquisition_timeout = float(get('acquisition_timeout'))
        self.inference_timeout = float(get('inference_timeout'))
        self.max_pixels = get('max_image_pixels')
        for name in ('startup_timeout', 'acquisition_timeout', 'inference_timeout'):
            value = float(get(name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(name + ' must be finite and positive')
        if not 0 < self.max_pixels <= 4194304:
            raise ValueError('max_image_pixels must be between 1 and 4194304')
        if worker is None:
            python = get('sam3_python')
            if python and not Path(python).expanduser().is_absolute():
                python = resolve_checkpoint(python)
            bpe = resolve_checkpoint(get('sam3_bpe_path')) if get('sam3_bpe_path') else ''
            worker = WorkerClient(resolve_checkpoint(get('sam3_checkpoint_path')), python,
                get('sam3_device'), get('sam3_confidence_threshold'), get('startup_timeout'), bpe, preload=get('preload_models'),
                release_cuda_cache=get('release_cuda_cache'))
        self.worker = worker
        self.instances_pub = self.create_publisher(InstanceSegmentation, get("instances_topic"), 1)
        self.mask_pub = self.create_publisher(Image, get('mask_topic'), 1)
        self.detection_pub = self.create_publisher(Detection2DArray, get('detection_topic'), 1)
        self.inference_header_pub = self.create_publisher(Header, get('inference_header_topic'), 1)
        self.status_pub = self.create_publisher(String, '/sam_segmenter/status', 1)
        group = ReentrantCallbackGroup()
        self.subscription = self.create_subscription(Image, get('image_topic'), self.on_image,
                                                      qos_profile_sensor_data, callback_group=group)
        if self.single:
            self.service = self.create_service(Trigger, '/sam_segmenter/trigger', self.trigger,
                                               callback_group=group)
        self.add_on_set_parameters_callback(self.on_parameters)
        self.timer = self.create_timer(0.5, self.status, callback_group=group)
        self.thread = threading.Thread(target=self.run, daemon=True, name='sam3-inference')
        self.thread.start()
        self.get_logger().info('Text SAM ready; targets=' + ','.join(self.targets))

    def check_clock(self):
        now = self.get_clock().now().nanoseconds
        if now < self.last_clock:
            self.epoch = uuid.uuid4().hex
            self.acquisition_id = 0
            self.generation += 1
            self.latest = None
            self.last_stamp = -1
            if self.request:
                self.request.update(done=True, success=False, message='ROS clock reset during trigger')
            if self.last_header is not None:
                self.detection_pub.publish(Detection2DArray(header=self.last_header))
            if self.last_header is not None and self.last_width and self.last_height:
                self.publish(self.last_header, self.last_height, self.last_width, np.zeros((0,self.last_height,self.last_width),np.uint8),[],invalidated=True)
            self.cv.notify_all()
        self.last_clock = now
        return now

    def on_image(self, message):
        with self.cv:
            self.check_clock()
            stamp = stamp_ns(message.header)
            if stamp <= self.last_stamp:
                return
            if not 0 < message.height * message.width <= self.max_pixels or len(message.data) > self.max_pixels * 4:
                self.last_error = 'Camera image is empty or exceeds its size bound'
                return
            self.last_stamp = stamp
            self.last_header = message.header
            self.last_height = message.height
            self.last_width = message.width
            if self.single:
                if not self.request or self.request['claimed'] or self.request['done']:
                    return
                if stamp <= self.request['boundary']:
                    return
            self.latest = message
            self.cv.notify_all()

    def on_parameters(self, parameters):
        try:
            value = next((p.value for p in reversed(parameters) if p.name == 'target_object'), None)
            if value is None:
                return SetParametersResult(successful=True)
            targets = parse_targets(value)
            with self.cv:
                if self.request or (self.single and self.processing):
                    raise ValueError('Cannot change target_object during an active trigger')
                self.targets = targets
                self.generation += 1
                self.latest = None
                if self.last_header is not None and self.last_width and self.last_height:
                    self.publish(self.last_header, self.last_height, self.last_width, np.zeros((0, self.last_height, self.last_width), np.uint8), [],invalidated=True)
                self.cv.notify_all()
            return SetParametersResult(successful=True)
        except (TypeError, ValueError) as exc:
            return SetParametersResult(successful=False, reason=str(exc))

    def trigger(self, request, response):
        del request
        try:
            if hasattr(self.worker, 'ensure_started'):
                self.worker.ensure_started()
        except Exception as exc:
            response.success, response.message = False, str(exc)
            return response
        with self.cv:
            now = self.check_clock()
            if self.request or self.processing:
                response.success, response.message = False, 'SAM segmentation is busy'
                return response
            if not self.worker.ready:
                response.success, response.message = False, 'SAM3 worker unavailable; restart the node'
                return response
            self.latest = None
            job = dict(boundary=max(now, self.last_stamp), claimed=False, done=False,
                       acquisition_deadline=time.monotonic() + self.acquisition_timeout,
                       deadline=None, success=False, message='')
            self.request = job
            self.cv.notify_all()
            while not job['done'] and not self.stopping:
                deadline = job['deadline'] if job['claimed'] else job['acquisition_deadline']
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.generation += 1
                    self.latest = None
                    job['message'] = ('Inference timeout' if job['claimed'] else 'Timeout waiting for fresh RGB')
                    job['done'] = True
                    break
                self.cv.wait(remaining)
            self.request = None
            self.latest = None
            response.success = job['success']
            response.message = job['message'] or 'SAM node stopped'
            if self.get_parameter('release_cuda_cache').value:
                gc.collect()
            return response

    def publish(self, header, height, width, masks, detections, *, invalidated=False):
        if masks.dtype != np.uint8 or masks.shape != (len(detections), height, width) or len(masks) > 32 or masks.nbytes > 64*1024*1024:
            raise ValueError('Invalid SAM per-instance mask packet')
        if not np.isin(masks, [0,255]).all():
            raise ValueError('SAM per-instance masks must be binary')
        self.acquisition_id += 1
        packet = InstanceSegmentation(header=header,
            source_id=self.get_parameter('source_id').value or self.get_parameter('image_topic').value,
            arm_id=self.get_parameter('arm_id').value, perception_epoch=self.epoch,
            perception_generation=self.generation, acquisition_id=self.acquisition_id,invalidated=invalidated)
        for index, (mask, item) in enumerate(zip(masks, detections)):
            image = Image(header=header,height=height,width=width,
                          encoding='mono8',step=width,data=mask.tobytes())
            packet.instances.append(SegmentedInstance(instance_id=f'{self.epoch}:{self.acquisition_id}:{index}',
                track_id=-1,class_label=item['label'],confidence=float(item['score']),mask=image))
        mask = np.bitwise_or.reduce(masks,axis=0) if len(masks) else np.zeros((height,width),np.uint8)
        boxes = Detection2DArray(header=header)
        for item in detections:
            x0, y0, x1, y1 = item['box']
            if not np.isfinite([x0, y0, x1, y1, item['score']]).all():
                raise ValueError('Non-finite SAM detection')
            detection = Detection2D(header=header)
            detection.bbox.center.position.x = (x0 + x1) / 2
            detection.bbox.center.position.y = (y0 + y1) / 2
            detection.bbox.size_x = float(max(0., x1 - x0))
            detection.bbox.size_y = float(max(0., y1 - y0))
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = item['label']
            hypothesis.hypothesis.score = float(item['score'])
            detection.results.append(hypothesis)
            boxes.detections.append(detection)
        output = Image(header=header, height=height, width=width,
                       encoding='mono8', is_bigendian=0, step=width, data=mask.tobytes())
        self.instances_pub.publish(packet)
        self.mask_pub.publish(output)
        self.detection_pub.publish(boxes)

    def run(self):
        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.stopping or self.latest is not None)
                if self.stopping:
                    return
                message, self.latest = self.latest, None
                generation, targets, job = self.generation, self.targets, self.request
                if self.single:
                    if not job or job['done'] or time.monotonic() >= job['acquisition_deadline']:
                        continue
                    job['claimed'] = True
                    job['deadline'] = time.monotonic() + self.inference_timeout
                    self.cv.notify_all()
                self.processing = True
            try:
                self.inference_header_pub.publish(message.header)
                image = rgb_array(message, self.max_pixels)
                if hasattr(self.worker, 'ensure_started'):
                    self.worker.ensure_started()
                started = time.monotonic()
                mask, detections = self.worker.segment(image, targets, self.inference_timeout)
                with self.cv:
                    self.check_clock()
                    if self.stopping or not rclpy.ok() or generation != self.generation or (job and job['done']):
                        continue
                    if time.monotonic() - started >= self.inference_timeout:
                        raise TimeoutError('Inference timeout')
                    self.publish(message.header, message.height, message.width, mask, detections)
                    found = bool(np.any(mask))
                    self.last_error = ''
                    self.last_result = 'targets_found' if found else 'no_target'
                    if job:
                        job.update(done=True, success=True,
                                   message='Successfully segmented frame' if found else 'No requested target found')
            except Exception as exc:
                with self.cv:
                    self.last_error = str(exc)
                    self.last_result = 'error'
                    if job and not job['done']:
                        job.update(done=True, success=False, message=str(exc))
                if not self.stopping and rclpy.ok():
                    self.get_logger().error('SAM segmentation failed: ' + str(exc))
            finally:
                if 'image' in locals():
                    del image
                if 'mask' in locals():
                    del mask
                if 'detections' in locals():
                    del detections
                if 'message' in locals():
                    del message
                if self.get_parameter('release_cuda_cache').value:
                    gc.collect()
                with self.cv:
                    self.processing = False
                    self.cv.notify_all()

    def status(self):
        with self.cv:
            self.check_clock()
            if self.single and not self.processing and self.request is None:
                self.latest = None
                if self.get_parameter('release_cuda_cache').value:
                    gc.collect()
            status = dict(model_ready=self.worker.ready, input_ready=self.last_stamp >= 0,
                          busy=self.processing or self.request is not None,
                          targets=self.targets, result=self.last_result, error=self.last_error,
                          memory=getattr(self.worker, 'memory_stats', {}))
        self.status_pub.publish(String(data=json.dumps(status)))

    def destroy_node(self):
        with self.cv:
            self.stopping = True
            self.cv.notify_all()
        self.worker.close()
        self.thread.join(timeout=2)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        node = Sam3Node()
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Launch may send another SIGINT after the first one shuts down ROS.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if node:
            node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
