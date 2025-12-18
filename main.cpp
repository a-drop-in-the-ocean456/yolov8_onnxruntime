#include <iostream>
#include <opencv2/opencv.hpp>
#include "yolov8_onnx.h"
#include "yolov8_utils.h"

// 主函数：执行YOLOv8目标检测
int main() {
    // 设置输入图片路径和模型路径
    std::string img_path = "../2.jpg";  // 输入图片路径
    std::string model_path = "../yolov8n.onnx";  // YOLOv8模型路径

    // 加载输入图片
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        std::cerr << "无法加载图像" << std::endl;
        return -1;
    }

    // 创建YOLOv8检测任务对象
    Yolov8Onnx task;
    // 加载模型（CPU模式，无GPU加速，无预热）
    if (task.ReadModel(model_path, false, 0, false)) {
        std::cout << "模型加载成功" << std::endl;
    } else {
        return -1;
    }

    // 执行目标检测
    std::vector<OutputParams> result;
    if (task.OnnxDetect(img, result)) {
        std::cout << "检测到 " << result.size() << " 个对象" << std::endl;
        if (!result.empty()) {
            // 生成固定颜色列表用于绘制不同类别的边界框
            std::vector<cv::Scalar> color = {
                cv::Scalar(0, 0, 255),   // 红
                cv::Scalar(0, 255, 0),   // 绿
                cv::Scalar(255, 0, 0),   // 蓝
                cv::Scalar(0, 255, 255), // 黄
                cv::Scalar(255, 0, 255), // 紫
                cv::Scalar(255, 255, 0), // 青
                cv::Scalar(128, 0, 128), // 深紫
                cv::Scalar(0, 128, 128), // 深青
                cv::Scalar(128, 128, 0), // 橄榄
                cv::Scalar(255, 165, 0), // 橙
            };
            // 重复颜色列表以达到80个颜色（对应COCO数据集的80个类别）
            while (color.size() < 80) {
                color.insert(color.end(), color.begin(), color.end());
            }
            color.resize(80);  // 确保颜色列表大小为80
            // 在图片上绘制检测结果
            DrawPred(img, result, task._className, color);
        }
        // 保存结果图片
        cv::imwrite("../result.jpg", img);
        std::cout << "结果保存到 ../result.jpg" << std::endl;
    } else {
        std::cout << "检测失败" << std::endl;
    }

    return 0;
}
