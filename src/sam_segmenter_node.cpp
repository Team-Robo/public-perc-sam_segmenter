#include "sam3.h"
#include <grasp_pipeline/model_lifecycle.hpp>
#include <grasp_pipeline/cuda_transport.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <cv_bridge/cv_bridge.h>
#include <core_r1_interfaces/msg/instance_segmentation.hpp>
#include <core_r1_interfaces/srv/detect_all.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <algorithm>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <thread>
#include <unistd.h>
#include <cctype>
#include <cmath>

namespace sam_segmenter {
using Image = sensor_msgs::msg::Image;
using Trigger = std_srvs::srv::Trigger;
using DetectAll = core_r1_interfaces::srv::DetectAll;
using Packet = core_r1_interfaces::msg::InstanceSegmentation;
using Clock = std::chrono::steady_clock;

static float sam_box_iou(const sam3_box &a, const sam3_box &b) {
  const float x0 = std::max(a.x0, b.x0);
  const float y0 = std::max(a.y0, b.y0);
  const float x1 = std::min(a.x1, b.x1);
  const float y1 = std::min(a.y1, b.y1);
  const float inter = std::max(0.0f, x1 - x0) * std::max(0.0f, y1 - y0);
  const float area_a = std::max(0.0f, a.x1 - a.x0) * std::max(0.0f, a.y1 - a.y0);
  const float area_b = std::max(0.0f, b.x1 - b.x0) * std::max(0.0f, b.y1 - b.y0);
  const float uni = area_a + area_b - inter;
  return uni > 0.0f ? (inter / uni) : 0.0f;
}
class SamSegmenterNode : public rclcpp::Node {
 public:
  explicit SamSegmenterNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
      : Node("sam_segmenter_node", options) {
    if (declare_parameter("backend_engine", "sam3_cpp") != "sam3_cpp")
      throw std::invalid_argument("backend_engine must be sam3_cpp");
    family_ = declare_parameter("model_family", "sam3");
    if (family_ != "sam3" && family_ != "sam2") throw std::invalid_argument("model_family must be sam3 or sam2");
    targets_ = parse_targets(declare_parameter("target_object", ""));
    box_ = declare_parameter("box_prompt", std::vector<double>{});
    if (family_ == "sam2" && (box_.size() != 4 || !std::all_of(box_.begin(), box_.end(), [](double v) { return std::isfinite(v); }) ||
        box_[0] < 0 || box_[1] < 0 || box_[2] <= box_[0] || box_[3] <= box_[1]))
      throw std::invalid_argument("sam2 requires box_prompt=[x0,y0,x1,y1]");
    auto path = std::filesystem::path(declare_parameter("model_path", "models/sam3-q4_0.ggml"));
    if (path.is_relative()) {
      const auto share = std::filesystem::path(ament_index_cpp::get_package_share_directory("sam_segmenter"));
      path = share / path;
    }
    if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("Missing GGML model: " + path.string() + "; run provision_models.sh");
    params_.model_path = path.string();
    params_.use_gpu = declare_parameter("use_gpu", true);
    params_.n_threads = declare_parameter("num_threads", 6);
    params_.encode_img_size = declare_parameter("encode_img_size", 0);
    declare_parameter("log_stage_timings", false);
    single_ = declare_parameter("single_frame_mode", true);
    always_hot_ = declare_parameter("always_hot", true);
    threshold_ = declare_parameter("conf_threshold", 0.5);
    acquisition_timeout_ = declare_parameter("acquisition_timeout", 25.0);
    inference_timeout_ = declare_parameter("inference_timeout", 10.0);
    max_pixels_ = declare_parameter("max_image_pixels", 4194304);
    if (params_.n_threads < 1 || !std::isfinite(threshold_) || threshold_ < 0 || threshold_ > 1 ||
        !std::isfinite(acquisition_timeout_) || acquisition_timeout_ <= 0 || !std::isfinite(inference_timeout_) || inference_timeout_ <= 0 || max_pixels_ < 1)
      throw std::invalid_argument("Invalid SAM inference parameters");
    const auto topic = declare_parameter("image_topic", "/camera/camera/color/image_raw");
    source_ = declare_parameter("source_id", ""); if (source_.empty()) source_ = topic;
    arm_ = declare_parameter("arm_id", "");
    epoch_ = std::to_string(getpid()) + ":" + std::to_string(Clock::now().time_since_epoch().count());
    model_id_ = "sam:" + epoch_;
    group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    rclcpp::SubscriptionOptions subscription_options; subscription_options.callback_group = group_;
    auto image_qos = rclcpp::SensorDataQoS();
    if (declare_parameter("reliable_input", false)) image_qos.reliable();
    images_ = create_subscription<Image>(topic, image_qos,
      [this](Image::ConstSharedPtr image) { receive(std::move(image)); }, subscription_options);
    masks_ = create_publisher<Image>(declare_parameter("mask_topic", "/perc/grasp/mask"), 1);
    selected_masks_ = create_publisher<Image>(declare_parameter("selected_mask_topic", "/perc/grasp/selected_mask"), 1);
    publish_selected_ = declare_parameter("publish_selected_mask", true);
    instances_ = create_publisher<Packet>(declare_parameter("instances_topic", "/perc/grasp/instances"), 1);
    detections_ = create_publisher<vision_msgs::msg::Detection2DArray>(declare_parameter("detection_topic", "/perc/grasp/detections"), 1);
    all_instances_ = create_publisher<Packet>(declare_parameter("all_instances_topic", "/perc/sam/all_instances"), 1);
    all_detections_ = create_publisher<vision_msgs::msg::Detection2DArray>(declare_parameter("all_detections_topic", "/perc/sam/all_detections"), 1);
    inference_headers_ = create_publisher<std_msgs::msg::Header>(
      declare_parameter("inference_header_topic", "/perc/grasp/segmentation_input"), 1);
    const auto acquisition_rgb_topic = declare_parameter<std::string>("acquisition_rgb_topic", "");
    if (!acquisition_rgb_topic.empty())
      acquisition_rgb_ = create_publisher<Image>(acquisition_rgb_topic, rclcpp::QoS(1).reliable());
    const auto trigger_topic = declare_parameter<std::string>("trigger_service", "/sam_segmenter/trigger");
    trigger_ = create_service<Trigger>(trigger_topic, [this](Trigger::Request::SharedPtr, Trigger::Response::SharedPtr response) {
      trigger(*response);
    }, rmw_qos_profile_services_default, group_);
    const auto detect_all_topic = declare_parameter<std::string>("detect_all_service", "/sam_segmenter/detect_all");
    detect_all_srv_ = create_service<DetectAll>(detect_all_topic, [this](DetectAll::Request::SharedPtr request, DetectAll::Response::SharedPtr response) {
      handle_detect_all(*request, *response);
    }, rmw_qos_profile_services_default, group_);
    grasp_pipeline::ModelLifecycleManager::Model spec;
    spec.device_bytes = params_.use_gpu ? (family_ == "sam2" ? 100000000ULL : 800000000ULL) : 0;
    spec.always_hot = always_hot_;
    spec.release_to = (single_ && !always_hot_) ? grasp_pipeline::ModelLifecycleManager::State::COLD
                                                : grasp_pipeline::ModelLifecycleManager::State::HOT;
    spec.warm = [this] { mapped_ = std::make_unique<grasp_pipeline::MappedModelFile>(params_.model_path); };
    spec.cold = [this] {
      params_.model_data = nullptr;
      params_.model_size = 0;
      mapped_.reset();
    };
    spec.load = [this] {
      const auto started=Clock::now();
      params_.model_data = mapped_->data(); params_.model_size = mapped_->size();
      model_ = sam3_load_model(params_);
      if (!model_ || (family_ == "sam3" && sam3_is_visual_only(*model_)) ||
          (family_ == "sam2" && sam3_get_model_type(*model_) != SAM3_MODEL_SAM2))
        throw std::runtime_error("GGML model does not match model_family");
      RCLCPP_INFO(get_logger(),"SAM WARM to HOT: %.3f ms",std::chrono::duration<double,std::milli>(Clock::now()-started).count());
    };
    spec.unload = [this] {
      const auto status=params_.use_gpu?cudaDeviceSynchronize():cudaSuccess;
      model_.reset();
      RCLCPP_INFO(get_logger(), "SAM HOT to %s: unloaded from GPU", (single_ && !always_hot_) ? "COLD" : "WARM");
      if(status!=cudaSuccess)RCLCPP_ERROR(get_logger(),"SAM release after CUDA error: %s",cudaGetErrorString(status));
    };
    grasp_pipeline::ModelLifecycleManager::shared().register_model(model_id_, std::move(spec));
    registered_ = true;
    parameter_callback_ = add_on_set_parameters_callback([this](const std::vector<rclcpp::Parameter> &values) {
      rcl_interfaces::msg::SetParametersResult result; result.successful = true;
      try {
        for (const auto &value : values) if (value.get_name() == "target_object") {
          const auto targets = parse_targets(value.as_string());
          std::lock_guard<std::mutex> lock(mutex_);
          if (busy_ || requested_) throw std::invalid_argument("Cannot change target during a trigger");
          targets_ = targets; ++generation_; latest_.reset();
          if (last_width_ && last_height_) {
            Image dummy; dummy.header = last_header_; dummy.width = last_width_; dummy.height = last_height_;
            publish(dummy, {}, {}, true);
          }
        }
      } catch (const std::exception &error) { result.successful = false; result.reason = error.what(); }
      return result;
    });
    model_status_ = create_service<Trigger>("~/model_status", [this](Trigger::Request::SharedPtr, Trigger::Response::SharedPtr response) {
      const auto stats=grasp_pipeline::ModelLifecycleManager::shared().statistics(model_id_);
      std::lock_guard<std::mutex> lock(mutex_);
      response->success=stats.state==grasp_pipeline::ModelLifecycleManager::State::HOT;
      response->message="loaded="+std::to_string(response->success)+", warmed="+std::to_string(response->success && warmed_load_==stats.loads)+
        ", busy="+std::to_string(busy_)+", loads="+std::to_string(stats.loads)+", unloads="+std::to_string(stats.unloads)+", error="+error_;
    },rmw_qos_profile_services_default,group_);
    worker_ = std::thread([this] { run(); });
    RCLCPP_INFO(get_logger(), "sam3_cpp ready (%s mode, %s); %s",
                single_ ? "trigger" : "continuous",
                always_hot_ ? "preloaded HOT" : "on-demand",
                always_hot_ ? "weights loaded and resident in GPU memory" :
                (single_ ? "loads and unloads on each trigger" : "weights reused until another model phase needs the GPU"));
  }
  ~SamSegmenterNode() override {
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; cv_.notify_all(); }
    if (worker_.joinable()) worker_.join();
    if (registered_) grasp_pipeline::ModelLifecycleManager::shared().unregister_model(model_id_);
  }
 private:
  std::vector<std::string> parse_targets(const std::string &value) const {
    std::vector<std::string> targets;
    if (value.empty()) {
      if (family_ == "sam3") {
        targets.push_back("object");
      }
      return targets;
    }
    std::istringstream input(value); std::string item;
    while (std::getline(input, item, ',')) {
      const auto start = item.find_first_not_of(" \t\r\n"), end = item.find_last_not_of(" \t\r\n");
      if (start == std::string::npos) continue;
      item = item.substr(start, end-start+1);
      if (item.size() > 256 || std::all_of(item.begin(), item.end(), [](unsigned char c) { return std::isdigit(c); }))
        throw std::invalid_argument("target_object requires names or phrases");
      if (std::find(targets.begin(), targets.end(), item) == targets.end()) targets.push_back(item);
    }
    if (family_ == "sam3" && targets.empty()) targets.push_back("object");
    if (family_ == "sam3" && targets.size() > 32) throw std::invalid_argument("Supply 1-32 target_object prompts");
    return targets;
  }
  void receive(Image::ConstSharedPtr image) {
    if (!image->width || !image->height || size_t(image->width)*image->height > size_t(max_pixels_)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    const int64_t clock = now().nanoseconds();
    if (last_clock_ && clock < last_clock_) {
      ++generation_; latest_.reset();
      epoch_=std::to_string(getpid())+":"+std::to_string(Clock::now().time_since_epoch().count());
      acquisition_=0;
      if (requested_) { done_=true;success_=false;error_="ROS clock reset during acquisition";cv_.notify_all(); }
      if (detect_all_requested_) { detect_all_done_=true;detect_all_success_=false;detect_all_error_="ROS clock reset";cv_.notify_all(); }
    }
    last_clock_ = clock;
    const int64_t stamp = rclcpp::Time(image->header.stamp).nanoseconds();
    last_stamp_ = stamp;
    last_header_ = image->header;
    last_width_ = image->width;
    last_height_ = image->height;
    if (single_) {
      const bool active_request = (requested_ && !done_) || (detect_all_requested_ && !detect_all_done_);
      if (!active_request || busy_ || stamp <= boundary_) return;
    }
    latest_ = std::move(image); cv_.notify_all();
  }
  void trigger(Trigger::Response &response) {
    const auto started = Clock::now();
    std::unique_lock<std::mutex> lock(mutex_);
    if (busy_ || requested_ || detect_all_requested_) { response.success = false; response.message = "SAM segmentation is busy"; return; }
    requested_ = true; done_ = false; success_ = false; error_.clear(); latest_.reset();
    boundary_ = std::max(now().nanoseconds(), last_stamp_);
    deadline_ = started + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(acquisition_timeout_));
    cv_.notify_all();
    while (!done_ && !stopping_) {
      if (cv_.wait_until(lock, deadline_) == std::cv_status::timeout && Clock::now() >= deadline_) {
        ++generation_; latest_.reset(); error_ = busy_ ? "Inference timeout" : "Timeout waiting for fresh RGB"; break;
      }
    }
    response.success = done_ && success_;
    response.message = error_.empty() ? "Successfully segmented frame" : error_;
    requested_ = false;
    latest_.reset();
    RCLCPP_INFO(get_logger(), "Trigger to mask: %.3f ms", std::chrono::duration<double,std::milli>(Clock::now()-started).count());
  }
  void handle_detect_all(const DetectAll::Request &request, DetectAll::Response &response) {
    const auto started = Clock::now();
    std::unique_lock<std::mutex> lock(mutex_);
    if (busy_ || requested_ || detect_all_requested_) {
      response.success = false; response.message = "SAM segmentation is busy"; return;
    }
    detect_all_requested_ = true; detect_all_done_ = false; detect_all_success_ = false;
    detect_all_error_.clear(); detect_all_packet_.reset();
    detect_all_vocab_ = request.prompt_vocabulary;
    detect_all_threshold_ = request.confidence_threshold;
    latest_.reset();
    boundary_ = std::max(now().nanoseconds(), last_stamp_);
    deadline_ = started + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(acquisition_timeout_));
    cv_.notify_all();
    while (!detect_all_done_ && !stopping_) {
      if (cv_.wait_until(lock, deadline_) == std::cv_status::timeout && Clock::now() >= deadline_) {
        ++generation_; latest_.reset();
        detect_all_error_ = busy_ ? "Inference timeout" : "Timeout waiting for fresh RGB";
        break;
      }
    }
    response.success = detect_all_done_ && detect_all_success_;
    response.message = detect_all_error_.empty() ? "Successfully detected all objects" : detect_all_error_;
    if (detect_all_packet_) {
      response.instances = *detect_all_packet_;
    }
    detect_all_requested_ = false;
    detect_all_packet_.reset();
    latest_.reset();
    RCLCPP_INFO(get_logger(), "DetectAll to output: %.3f ms", std::chrono::duration<double,std::milli>(Clock::now()-started).count());
  }
  std::pair<std::unique_ptr<Packet>, vision_msgs::msg::Detection2DArray>
  build_packet_and_boxes(const Image &image, std::vector<sam3_detection> &results,
                         const std::vector<std::string> &labels, bool invalidated = false) {
    auto packet = std::make_unique<Packet>(); packet->header = image.header;
    packet->source_id = source_; packet->arm_id = arm_; packet->perception_epoch = epoch_;
    packet->perception_generation = generation_; packet->acquisition_id = ++acquisition_; packet->invalidated = invalidated;
    vision_msgs::msg::Detection2DArray boxes; boxes.header = image.header;
    const size_t image_pixels = size_t(image.width) * image.height;
    if (results.size() > 64 || results.size() * image_pixels > 128 * 1024 * 1024ULL) {
      throw std::runtime_error("SAM instance packet exceeds interface bounds");
    }
    for (size_t i = 0; i < results.size(); ++i) {
      auto &result = results[i];
      if (result.mask.width != int(image.width) || result.mask.height != int(image.height) ||
          result.mask.data.size() != image_pixels || !std::isfinite(result.score)) {
        continue;
      }
      // Compute tight 2D bounding box from segmentation mask to ensure exact alignment
      int min_x = image.width, min_y = image.height;
      int max_x = 0, max_y = 0;
      bool has_mask = false;
      const uint8_t *mdata = result.mask.data.data();
      for (int y = 0; y < int(image.height); ++y) {
        for (int x = 0; x < int(image.width); ++x) {
          if (mdata[y * image.width + x] > 0) {
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
            has_mask = true;
          }
        }
      }
      if (has_mask) {
        result.box.x0 = min_x;
        result.box.y0 = min_y;
        result.box.x1 = max_x;
        result.box.y1 = max_y;
      }
      core_r1_interfaces::msg::SegmentedInstance instance;
      instance.instance_id = epoch_ + ":" + std::to_string(acquisition_) + ":" + std::to_string(i);
      instance.track_id = -1; instance.class_label = labels.at(i); instance.confidence = result.score;
      instance.mask.header = image.header; instance.mask.height = image.height; instance.mask.width = image.width;
      instance.mask.encoding = "mono8"; instance.mask.step = image.width; instance.mask.data = std::move(result.mask.data);
      packet->instances.push_back(std::move(instance));
      vision_msgs::msg::Detection2D box; box.header = image.header;
      box.bbox.center.position.x = (result.box.x0 + result.box.x1) / 2.0;
      box.bbox.center.position.y = (result.box.y0 + result.box.y1) / 2.0;
      box.bbox.size_x = result.box.x1 - result.box.x0; box.bbox.size_y = result.box.y1 - result.box.y0;
      vision_msgs::msg::ObjectHypothesisWithPose hypothesis; hypothesis.hypothesis.class_id = labels.at(i); hypothesis.hypothesis.score = result.score;
      box.results.push_back(hypothesis); boxes.detections.push_back(std::move(box));
    }
    return {std::move(packet), std::move(boxes)};
  }
  void publish(const Image &image, std::vector<sam3_detection> results,
               const std::vector<std::string> &labels, bool invalidated = false) {
    auto mask = std::make_unique<Image>(); mask->header = image.header; mask->height = image.height; mask->width = image.width;
    mask->encoding = "mono8"; mask->step = image.width; mask->data.resize(size_t(image.width)*image.height);
    std::unique_ptr<Image> selected;
    if (publish_selected_) {
      selected=std::make_unique<Image>();selected->header=image.header;
      selected->height=image.height;selected->width=image.width;selected->encoding="mono8";selected->step=image.width;
      const auto best=std::max_element(results.begin(),results.end(),[](const auto &a,const auto &b){return a.score<b.score;});
      if(best!=results.end())selected->data=best->mask.data;
      else selected->data.resize(mask->data.size());
    }
    for (const auto &res : results) {
      if (res.mask.data.size() == mask->data.size()) {
        for (size_t p = 0; p < mask->data.size(); ++p) mask->data[p] |= res.mask.data[p];
      }
    }
    auto [packet, boxes] = build_packet_and_boxes(image, results, labels, invalidated);
    grasp_pipeline::publish_owned(*instances_,std::move(packet),get_node_options().use_intra_process_comms());
    if (selected) {
      grasp_pipeline::publish_owned(*selected_masks_,std::move(selected),get_node_options().use_intra_process_comms());
    }
    grasp_pipeline::publish_owned(*masks_,std::move(mask),get_node_options().use_intra_process_comms());
    detections_->publish(boxes);
  }
  void run() {
    for (;;) {
      Image::ConstSharedPtr image; std::vector<std::string> targets; uint64_t generation;
      bool is_detect_all = false;
      std::string detect_all_vocab;
      float detect_all_thresh = 0.0f;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return stopping_ || latest_; });
        if (stopping_) return;
        image = std::move(latest_); targets = targets_; generation = generation_; busy_ = true;
        is_detect_all = detect_all_requested_;
        detect_all_vocab = detect_all_vocab_;
        detect_all_thresh = detect_all_threshold_;
        if (requested_ || detect_all_requested_) {
          deadline_ = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(inference_timeout_));
        }
        cv_.notify_all();
      }
      try {
        // Geometry retains this exact acquisition while segmentation is running.
        inference_headers_->publish(image->header);
        if (acquisition_rgb_) acquisition_rgb_->publish(*image);
        const auto stage_started=Clock::now();
        auto lease = grasp_pipeline::ModelLifecycleManager::shared().lease(model_id_);
        const auto leased_at=Clock::now();
        // Image workspaces are frame-local so consecutive frames cannot overlap
        // encoder allocations. The leased weights and backend remain reusable.
        auto state = sam3_create_state(*model_, params_);
        if (!state) throw std::runtime_error("Cannot allocate SAM inference state");
        const auto state_at=Clock::now();
        const auto rgb = cv_bridge::toCvShare(image, "rgb8");
        sam3_image input; input.width = rgb->image.cols; input.height = rgb->image.rows;
        if (!rgb->image.isContinuous()) throw std::runtime_error("Non-contiguous RGB image");
        input.borrowed_pixels = rgb->image.data;
        std::unique_ptr<grasp_pipeline::RegisteredHostBuffer> registered;
        if(params_.use_gpu) registered=std::make_unique<grasp_pipeline::RegisteredHostBuffer>(input.borrowed_pixels,rgb->image.total()*3);
        const auto prepared_at=Clock::now();
        if (!sam3_encode_image(*state, *model_, input)) throw std::runtime_error("SAM image encoding failed");
        const auto encoded_at=Clock::now();
        std::vector<sam3_detection> results; std::vector<std::string> labels;
        if (is_detect_all) {
          std::vector<std::string> vocab;
          if (!detect_all_vocab.empty()) {
            std::istringstream ss(detect_all_vocab);
            std::string item;
            while (std::getline(ss, item, ',')) {
              const auto start = item.find_first_not_of(" \t\r\n"), end = item.find_last_not_of(" \t\r\n");
              if (start != std::string::npos) vocab.push_back(item.substr(start, end-start+1));
            }
          }
          if (vocab.empty()) {
            vocab = {"object", "bottle", "cup", "can", "box", "bowl", "container", "tool", "fruit", "device", "part"};
          }
          const float eff_thresh = (detect_all_thresh > 0.0f && detect_all_thresh <= 1.0f) ? detect_all_thresh : static_cast<float>(threshold_);
          for (const auto &concept : vocab) {
            sam3_pcs_params prompt; prompt.text_prompt = concept; prompt.score_threshold = eff_thresh;
            auto output = sam3_segment_pcs(*state, *model_, prompt);
            for (auto &det : output.detections) {
              bool duplicate = false;
              for (const auto &existing : results) {
                if (sam_box_iou(det.box, existing.box) > 0.65f) { duplicate = true; break; }
              }
              if (!duplicate) {
                results.push_back(std::move(det));
                labels.push_back(concept);
              }
            }
          }
        } else if (family_ == "sam3") {
          for (const auto &target : targets) {
            sam3_pcs_params prompt; prompt.text_prompt = target; prompt.score_threshold = threshold_;
            auto output = sam3_segment_pcs(*state, *model_, prompt);
            for (auto &detection : output.detections) { results.push_back(std::move(detection)); labels.push_back(target); }
          }
        } else {
          if (box_[2] > input.width || box_[3] > input.height) throw std::invalid_argument("box_prompt exceeds image dimensions");
          sam3_pvs_params prompt; prompt.use_box = true;
          prompt.box = {float(box_[0]),float(box_[1]),float(box_[2]),float(box_[3])};
          auto output = sam3_segment_pvs(*state, *model_, prompt);
          for (auto &detection : output.detections) { if (detection.score >= threshold_) { results.push_back(std::move(detection)); labels.push_back(targets.empty() ? "box" : targets.front()); } }
        }
        if (get_parameter("log_stage_timings").as_bool()) {
          const auto ms=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};
          RCLCPP_INFO(get_logger(),"SAM stages ms: lease=%.3f state=%.3f preprocessing=%.3f encode=%.3f prompts=%.3f",
            ms(stage_started,leased_at),ms(leased_at,state_at),ms(state_at,prepared_at),ms(prepared_at,encoded_at),ms(encoded_at,Clock::now()));
        }
        // Free this frame's workspace before releasing the phase lease.
        registered.reset();
        state.reset();
        const auto completed_load=grasp_pipeline::ModelLifecycleManager::shared().statistics(model_id_).loads;
        lease.reset();
        std::lock_guard<std::mutex> lock(mutex_);
        warmed_load_=completed_load;
        if (generation == generation_ && !stopping_) {
          if (is_detect_all) {
            const bool empty = results.empty();
            auto [packet, boxes] = build_packet_and_boxes(*image, results, labels);
            grasp_pipeline::publish_owned(*all_instances_, std::make_unique<Packet>(*packet), get_node_options().use_intra_process_comms());
            all_detections_->publish(boxes);
            detect_all_packet_ = std::move(packet);
            detect_all_done_ = true;
            detect_all_success_ = true;
            if (empty) detect_all_error_ = "No objects detected";
          } else {
            const bool empty=results.empty();
            publish(*image, std::move(results), labels); done_ = true; success_ = true;
            if (empty) error_ = "No requested target found";
          }
        }
        image.reset();
      } catch (const std::exception &error) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (is_detect_all) {
          detect_all_error_ = error.what();
          detect_all_done_ = true;
          detect_all_success_ = false;
        } else {
          error_ = error.what();
          done_ = true;
          success_ = false;
        }
        RCLCPP_ERROR(get_logger(), "%s", error.what());
      }
      { std::lock_guard<std::mutex> lock(mutex_); busy_ = false; cv_.notify_all(); }
    }
  }
  sam3_params params_;
  rclcpp::Service<Trigger>::SharedPtr model_status_;
  uint64_t warmed_load_=0;
  std::shared_ptr<sam3_model> model_;
  std::unique_ptr<grasp_pipeline::MappedModelFile> mapped_;
  std::string family_, source_, arm_, epoch_, model_id_, error_;
  std::vector<std::string> targets_;
  std::vector<double> box_;
  double threshold_, acquisition_timeout_, inference_timeout_;
  int max_pixels_;
  bool single_, publish_selected_, always_hot_{true}, stopping_ = false, busy_ = false, requested_ = false, done_ = false, success_ = false, registered_ = false;
  uint64_t generation_ = 1, acquisition_ = 0;
  int64_t last_clock_ = 0, last_stamp_ = 0, boundary_ = 0;
  Clock::time_point deadline_;
  std::mutex mutex_; std::condition_variable cv_; std::thread worker_;
  Image::ConstSharedPtr latest_;
  std_msgs::msg::Header last_header_;
  uint32_t last_width_{0}, last_height_{0};
  rclcpp::CallbackGroup::SharedPtr group_;
  rclcpp::Subscription<Image>::SharedPtr images_;
  rclcpp::Publisher<Image>::SharedPtr masks_, selected_masks_;
  rclcpp::Publisher<Packet>::SharedPtr instances_;
  rclcpp::Publisher<Packet>::SharedPtr all_instances_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detections_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr all_detections_;
  rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr inference_headers_;
  rclcpp::Publisher<Image>::SharedPtr acquisition_rgb_;
  rclcpp::Service<Trigger>::SharedPtr trigger_;
  rclcpp::Service<DetectAll>::SharedPtr detect_all_srv_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_;
  bool detect_all_requested_ = false, detect_all_done_ = false, detect_all_success_ = false;
  std::string detect_all_vocab_, detect_all_error_;
  float detect_all_threshold_{0.0f};
  std::unique_ptr<Packet> detect_all_packet_;
};
std::shared_ptr<rclcpp::Node> make_node(const rclcpp::NodeOptions &options) {
  return std::make_shared<SamSegmenterNode>(options);
}
}  // namespace sam_segmenter
RCLCPP_COMPONENTS_REGISTER_NODE(sam_segmenter::SamSegmenterNode)
