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
  OrtEnv.instance.init();
  final options = OrtSessionOptions();
  final model = await rootBundle.load("assets/house_price_model.onnx");

  
  

  try {
    session = OrtSession.fromBuffer(model.buffer.asUint8List(), options);

    // Convert the input features (List<double>) into a Float32List.
    final floatInput = Float32List.fromList(inputFeatures);

    // Create the input tensor.
    // Here we assume that the package exposes a method called fromBytes.
    // (If not, check the documentation for the correct tensor creation method.)
    final shape = [1, 2, 3];
    final inputOrt = OrtValueTensor.createTensorWithDataList(floatInput, shape);
    final inputs = {'input': inputOrt};
    final runOptions = OrtRunOptions();
    final outputs = await session.runAsync(runOptions, inputs);
    inputOrt.release();
    runOptions.release();
    outputs?.forEach((element) {
      element?.release();
    });

   

    // Extract the output tensor.
    // We assume the model returns one output tensor.
    outputTensor = outputs!.isNotEmpty ? outputs.first : null;

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
    OrtEnv.instance.release();
    outputTensor?.release();
    session?.release();
  }
}

