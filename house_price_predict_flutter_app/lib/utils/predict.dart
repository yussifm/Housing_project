import 'dart:io';

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
    // Load model file into a temporary directory
    final modelPath = await _loadModelFile('assets/house_price_model.onnx');

    // Create an ONNX runtime session
    session = OrtSession.fromFile(Env(), File(modelPath));

    // Prepare input tensor
    inputTensor = OrtValue.createTensor(inputFeatures);

    // Run inference
    final results = session.run({'input': inputTensor});

    // Extract output tensor (assuming first output is the predicted value)
    outputTensor = results['output'];

    // Return predicted price (assuming shape [1])
    return outputTensor?.getTensorData<double>()?.first;
  } catch (e) {
    log.e('Error running ONNX model: $e');
    return null;
  } finally {
    // Ensure all resources are released
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
