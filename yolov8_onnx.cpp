#include "yolov8_onnx.h"
using namespace Ort;


// bool Yolov8Onnx::ReadModel 是 Yolov8Onnx 类的成员函数，
// 用于加载和初始化YOLOv8 ONNX模型，包括设置会话选项、创建ONNX Runtime会话、初始化输入输出节点等。
bool Yolov8Onnx::ReadModel(const std::string& modelPath, bool isCuda, int cudaID, bool warmUp) {
	// 设置批处理大小
	if (_batchSize < 1) _batchSize = 1;
	// try 是 C++ 中的异常处理语句，用于定义一个代码块，
	// 在该块中如果发生异常，可以被 catch 语句捕获和处理。
	try
	{
		// 检查模型文件是否存在
		// 虽然 yolov8_onnx.cpp 直接只包含了 #include "yolov8_onnx.h"，
		// 但 yolov8_onnx.h 中包含了 #include "yolov8_utils.h"，
		// 因此 CheckModelPath 等函数的声明被间接包含进来。
		// 编译器通过这种头文件链式包含机制知道函数声明，链接器在链接时找到 yolov8_utils.cpp 中的实现。
		if (!CheckModelPath(modelPath))
			return false;
		// 获取可用的ONNX Runtime提供商
		std::vector<std::string> available_providers = GetAvailableProviders();
		auto cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider");


		std::cout << "************* Infer model on CPU! *************" << std::endl;

		// 设置图优化级别
		_OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

// 创建ONNX Runtime会话
// 这段代码根据操作系统处理字符串编码差异：
// Windows (_WIN32): Windows API使用宽字符（UTF-16），所以需要将std::string转换为std::wstring
// Linux/Unix: 使用UTF-8字符串，直接使用std::string
// 这是因为ONNX Runtime的C++ API在不同平台对路径字符串的编码要求不同。Windows需要宽字符，Linux使用普通字符串。
#ifdef _WIN32
		std::wstring model_path(modelPath.begin(), modelPath.end());
		_OrtSession = new Ort::Session(_OrtEnv, model_path.c_str(), _OrtSessionOptions);
#else
		// 这是创建 ONNX Runtime 会话对象的代码：
		// new Ort::Session(...) 在堆上创建 Ort::Session 对象
		// 参数：
		// _OrtEnv: ONNX Runtime 环境对象
		// modelPath.c_str(): 模型文件路径的C字符串
		// _OrtSessionOptions: 会话选项（包含优化设置、执行提供商等）
		// 会话对象负责加载ONNX模型、准备推理环境，并提供Run方法执行推理
		_OrtSession = new Ort::Session(_OrtEnv, modelPath.c_str(), _OrtSessionOptions);
#endif

		Ort::AllocatorWithDefaultOptions allocator;
		// 获取输入节点的数量
		_inputNodesNum = _OrtSession->GetInputCount();
		// 根据ONNX Runtime版本获取输入节点名称
#if ORT_API_VERSION < ORT_OLD_VISON
		// 旧版本API：直接获取输入名称
		_inputName = _OrtSession->GetInputName(0, allocator);
		_inputNodeNames.push_back(_inputName);
#else
		// 新版本API：使用AllocatedStringPtr，需要move语义
		_inputName = std::move(_OrtSession->GetInputNameAllocated(0, allocator));
		_inputNodeNames.push_back(_inputName.get());
#endif
		// 可选：打印输入节点名称用于调试
		//std::cout << _inputNodeNames[0] << std::endl;
		// 获取输入节点的类型信息
		Ort::TypeInfo inputTypeInfo = _OrtSession->GetInputTypeInfo(0);
		// 从类型信息中提取tensor信息
		auto input_tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();
		// 获取输入tensor的数据类型
		_inputNodeDataType = input_tensor_info.GetElementType();
		// 获取输入tensor的形状
		_inputTensorShape = input_tensor_info.GetShape();

		// 如果batch维度是动态的（-1），设置为批处理大小
		if (_inputTensorShape[0] == -1)
		{
			_isDynamicShape = true;
			_inputTensorShape[0] = _batchSize;
		}
		// 如果高度或宽度是动态的，设置为网络的输入尺寸
		if (_inputTensorShape[2] == -1 || _inputTensorShape[3] == -1) {
			_isDynamicShape = true;
			_inputTensorShape[2] = _netHeight;
			_inputTensorShape[3] = _netWidth;
		}
		// 初始化输出节点
		_outputNodesNum = _OrtSession->GetOutputCount();
#if ORT_API_VERSION < ORT_OLD_VISON
		_output_name0 = _OrtSession->GetOutputName(0, allocator);
		_outputNodeNames.push_back(_output_name0);
#else
		_output_name0 = std::move(_OrtSession->GetOutputNameAllocated(0, allocator));
		_outputNodeNames.push_back(_output_name0.get());
#endif
		Ort::TypeInfo type_info_output0(nullptr);
		type_info_output0 = _OrtSession->GetOutputTypeInfo(0);  //output0

		auto tensor_info_output0 = type_info_output0.GetTensorTypeAndShapeInfo();
		_outputNodeDataType = tensor_info_output0.GetElementType();
		_outputTensorShape = tensor_info_output0.GetShape();

		// GPU预热（warm up）：如果使用GPU且启用预热，则运行几次推理来初始化GPU
		if (isCuda && warmUp) {
			std::cout << "Start warming up" << std::endl;
			// 计算输入tensor的总长度
			size_t input_tensor_length = VectorProduct(_inputTensorShape);
			// 分配临时内存用于预热数据
			float* temp = new float[input_tensor_length];
			std::vector<Ort::Value> input_tensors;
			std::vector<Ort::Value> output_tensors;
			// 创建输入tensor（使用临时数据）
			input_tensors.push_back(Ort::Value::CreateTensor<float>(
				_OrtMemoryInfo, temp, input_tensor_length, _inputTensorShape.data(),
				_inputTensorShape.size()));
			// 运行3次推理进行预热
			for (int i = 0; i < 3; ++i) {
				output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
					_inputNodeNames.data(),
					input_tensors.data(),
					_inputNodeNames.size(),
					_outputNodeNames.data(),
					_outputNodeNames.size());
			}
			// 释放临时内存
			delete[]temp;
		}
	}
	catch (const std::exception&) {
		return false;
	}
	return true;

}

int Yolov8Onnx::Preprocessing(const std::vector<cv::Mat>& srcImgs, std::vector<cv::Mat>& outSrcImgs, std::vector<cv::Vec4d>& params) {
	outSrcImgs.clear();
	cv::Size input_size = cv::Size(_netWidth, _netHeight);
	for (int i = 0; i < srcImgs.size(); ++i) {
		cv::Mat temp_img = srcImgs[i];
		cv::Vec4d temp_param = {1,1,0,0};
		if (temp_img.size() != input_size) {
			cv::Mat borderImg;
			LetterBox(temp_img, borderImg, temp_param, input_size, false, false, true, 32);
			//std::cout << borderImg.size() << std::endl;
			outSrcImgs.push_back(borderImg);
			params.push_back(temp_param);
		}
		else {
			outSrcImgs.push_back(temp_img);
			params.push_back(temp_param);
		}
	}

	int lack_num =  _batchSize- srcImgs.size();
	if (lack_num > 0) {
		for (int i = 0; i < lack_num; ++i) {
			cv::Mat temp_img = cv::Mat::zeros(input_size, CV_8UC3);
			cv::Vec4d temp_param = { 1,1,0,0 };
			outSrcImgs.push_back(temp_img);
			params.push_back(temp_param);
		}
	}
	return 0;

}
bool Yolov8Onnx::OnnxDetect(cv::Mat& srcImg, std::vector<OutputParams>& output) {
	std::vector<cv::Mat> input_data = { srcImg };
	std::vector<std::vector<OutputParams>> tenp_output;
	if (OnnxBatchDetect(input_data, tenp_output)) {
		output = tenp_output[0];
		return true;
	}
	else return false;
}
bool Yolov8Onnx::OnnxBatchDetect(std::vector<cv::Mat>& srcImgs, std::vector<std::vector<OutputParams>>& output) {
	std::vector<cv::Vec4d> params;
	std::vector<cv::Mat> input_images;
	cv::Size input_size(_netWidth, _netHeight);
	// 预处理输入图片
	Preprocessing(srcImgs, input_images, params);
	// 将图片转换为blob格式
	cv::Mat blob = cv::dnn::blobFromImages(input_images, 1 / 255.0, input_size, cv::Scalar(0, 0, 0), true, false);

	int64_t input_tensor_length = VectorProduct(_inputTensorShape);
	std::vector<Ort::Value> input_tensors;
	std::vector<Ort::Value> output_tensors;
	// 创建输入tensor
	input_tensors.push_back(Ort::Value::CreateTensor<float>(_OrtMemoryInfo, (float*)blob.data, input_tensor_length, _inputTensorShape.data(), _inputTensorShape.size()));

	// 运行ONNX推理
	output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
		_inputNodeNames.data(),
		input_tensors.data(),
		_inputNodeNames.size(),
		_outputNodeNames.data(),
		_outputNodeNames.size()
	);
	// 获取输出tensor的浮点数据指针
	float* all_data = output_tensors[0].GetTensorMutableData<float>();
	// 更新输出tensor形状
	_outputTensorShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
	// 计算每行的宽度（特征数量）
	int net_width = _outputTensorShape[1];
	// 计算类别得分数组长度（总特征数减去4个框参数）
	int socre_array_length = net_width - 4;
	// 计算单个图片的输出长度
	int64_t one_output_length = VectorProduct(_outputTensorShape) / _outputTensorShape[0];
	// 遍历每张输入图片
	for (int img_index = 0; img_index < srcImgs.size(); ++img_index) {
		// 将输出数据重塑为矩阵并转置，从[bs,116,8400]变为[bs,8400,116]
		cv::Mat output0 = cv::Mat(cv::Size((int)_outputTensorShape[2], (int)_outputTensorShape[1]), CV_32F, all_data).t();
		// 移动到下一张图片的数据
		all_data += one_output_length;
		// 获取当前图片数据的指针
		float* pdata = (float*)output0.data;
		// 获取行数（检测框数量）
		int rows = output0.rows;
		// 初始化存储类别ID、置信度和边界框的向量
		std::vector<int> class_ids;
		std::vector<float> confidences;
		std::vector<cv::Rect> boxes;
		// 遍历每个检测框
		for (int r = 0; r < rows; ++r) {
			// 提取类别得分（从第5个元素开始）
			cv::Mat scores(1, socre_array_length, CV_32F, pdata + 4);
			cv::Point classIdPoint;
			double max_class_socre;
			// 找到最大类别得分及其索引
			minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
			max_class_socre = (float)max_class_socre;
			// 如果超过置信度阈值，保存检测结果
			if (max_class_socre >= _classThreshold) {
				// 计算边界框坐标（x,y,w,h）
				float x = (pdata[0] - params[img_index][2]) / params[img_index][0];  // 中心x坐标
				float y = (pdata[1] - params[img_index][3]) / params[img_index][1];  // 中心y坐标
				float w = pdata[2] / params[img_index][0];  // 宽度
				float h = pdata[3] / params[img_index][1];  // 高度
				// 计算左上角坐标
				int left = MAX(int(x - 0.5 * w + 0.5), 0);
				int top = MAX(int(y - 0.5 * h + 0.5), 0);
				// 保存类别ID、置信度和边界框
				class_ids.push_back(classIdPoint.x);
				confidences.push_back(max_class_socre);
				boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
			}
			// 移动到下一个检测框的数据
			pdata += net_width;
		}

		// 执行非极大值抑制（NMS）
		std::vector<int> nms_result;
		cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result);
		// 创建图像边界矩形
		std::vector<std::vector<float>> temp_mask_proposals;
		cv::Rect holeImgRect(0, 0, srcImgs[img_index].cols, srcImgs[img_index].rows);
		std::vector<OutputParams> temp_output;
		// 收集NMS后的检测结果
		for (int i = 0; i < nms_result.size(); ++i) {
			int idx = nms_result[i];
			OutputParams result;
			result.id = class_ids[idx];
			result.confidence = confidences[idx];
			result.box = boxes[idx] & holeImgRect;  // 确保边界框在图像范围内
			if (result.box.area() < 1)
				continue;
			temp_output.push_back(result);
		}
		// 将结果添加到输出向量
		output.push_back(temp_output);
	}

	// 如果有输出结果，返回true
	if (output.size())
		return true;
	else
		return false;

}