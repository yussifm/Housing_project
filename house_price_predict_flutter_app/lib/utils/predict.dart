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

    // Create session options
    final sessionOptions = OrtSessionOptions()
      ..setIntraOpNumThreads(1)
      ..setOptimizationLevel(1);

    // Create an ONNX runtime session.
    session = await OrtSession.fromFile(modelPath, options: sessionOptions);

    // Convert inputFeatures (List<double>) to a Float32List.
    final floatInput = Float32List.fromList(inputFeatures);

    // Create the input tensor.
    // Use OrtValue.tensorFrom methods
    inputTensor = OrtValue.tensorFrom(
      value: floatInput,
      shape: [1, inputFeatures.length],
    );

    // Run inference.
    final results = await session.run({
      'input': inputTensor,
    });

    // Extract the output tensor.
    // We assume the model returns one output, so we take the first element.
    if (results.isNotEmpty) {
      final outputTensor = results.first;
      
      // Safely extract the value
      if (outputTensor.value is List) {
        final List<dynamic> outData = outputTensor.value as List<dynamic>;
        return outData.isNotEmpty ? outData.first as double : null;
      }
    }
    
    return null;
  } catch (e) {
    log.e('Error running ONNX model: $e');
    return null;
  } finally {
    // Always release resources.
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
