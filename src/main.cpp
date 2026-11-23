#include <rclcpp/rclcpp.hpp>
namespace sam_segmenter { std::shared_ptr<rclcpp::Node> make_node(const rclcpp::NodeOptions &); }

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto executor = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
  auto node = sam_segmenter::make_node(rclcpp::NodeOptions());
  executor->add_node(node);
  executor->spin(); rclcpp::shutdown();
}
