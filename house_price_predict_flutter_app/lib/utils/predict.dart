import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:logger/logger.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:path_provider/path_provider.dart';

final log = Logger();

Future<double?> predictHousePrice(List<double> inputFeatures) async {
  OrtSession? session;
  OrtValue? inputTensor;
  OrtValue? outputTensor;

  try {
    // Load the model file from assets into a temporary directory.
    final modelPath = await _loadModelFile('assets/house_price_model.onnx');

    // Create an ONNX runtime session.
    // Note: The first argument must be a File, the second an OrtSessionOptions instance.
    session = OrtSession.fromFile(File(modelPath), OrtSessionOptions());

    // Convert the input features (List<double>) into a Float32List.
    final floatInput = Float32List.fromList(inputFeatures);

    // Create the input tensor.
    // Here we assume that the package exposes a method called fromBytes.
    // (If not, check the documentation for the correct tensor creation method.)
    inputTensor = OrtValue.fromBytes(
      floatInput.buffer.asUint8List(),
      shape: [1, inputFeatures.length],
      type: OrtTensorElementDataType.float,
    );

    // Run inference.
    final results = session.run(OrtRunOptions(), {'input': inputTensor});

    // Extract the output tensor.
    // We assume the model returns one output tensor.
    outputTensor = results.isNotEmpty ? results.first : null;

    // Return the predicted price.
    // We assume outputTensor.value is a List whose first element is a double.
    if (outputTensor != null && outputTensor.value is List) {
      final List<dynamic> outData = outputTensor.value as List<dynamic>;
      return outData.isNotEmpty ? outData.first as double : null;
    }
    return null;
  } catch (e) {
    log.e('Error running ONNX model: $e');
    return null;
  } finally {
    // Release resources.
    inputTensor?.release();
    outputTensor?.release();
    session?.release();
  }
}

Future<String> _loadModelFile(String assetPath) async {
  final modelData = await rootBundle.load(assetPath);
  final directory = await getTemporaryDirectory();
  final modelFile = File('${directory.path}/house_price_model.onnx');
  await modelFile.writeAsBytes(modelData.buffer.asUint8List());
  return modelFile.path;
}
