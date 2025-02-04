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
    final inputNames = session.inputNames;
    final outputNames = session.outputNames;

    print('Input Names: $inputNames');
    print('Output Names: $outputNames');

    // Convert the input features to Float32List
    final floatInput = Float32List.fromList(inputFeatures);

    // Create the input tensor with the correct shape and name
    final shape = [1, 4]; // 1 sample, 4 features
    final inputOrt = OrtValueTensor.createTensorWithDataList(floatInput, shape);
    // Get the input and output names from the session

    final inputs = {'float_input': inputOrt};

    final runOptions = OrtRunOptions();
    final outputs = await session.runAsync(runOptions, inputs);

    inputOrt.release();
    runOptions.release();

    // Extract the output tensor
    if (outputs != null && outputs.isNotEmpty) {
      final outputTensor = outputs.first;

      // Handle nested list output
      dynamic outputValue = outputTensor?.value;

      // Drill down to the actual numeric value
      while (outputValue is List && outputValue.isNotEmpty) {
        outputValue = outputValue.first;
      }

      // Convert to double
      if (outputValue is double) {
        return outputValue;
      } else if (outputValue is num) {
        return outputValue.toDouble();
      }
    }

    return null;
  } catch (e) {
    log.e('Error running ONNX model: $e');
    return null;
  } finally {
    // Release resources
    OrtEnv.instance.release();
    session?.release();
  }
}
