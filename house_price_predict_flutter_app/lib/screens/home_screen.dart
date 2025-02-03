import 'package:flutter/material.dart';
import '../utils/predict.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _formKey = GlobalKey<FormState>();

  final TextEditingController _sqftController = TextEditingController();
  final TextEditingController _bathController = TextEditingController();
  final TextEditingController _balconyController = TextEditingController();
  final TextEditingController _sizeController = TextEditingController();

  double? _predictedPrice;
  bool _isLoading = false;

  Future<void> _predictPrice() async {
    if (_formKey.currentState?.validate() ?? false) {
      setState(() {
        _isLoading = true;
      });

      // Parse user inputs
      final double totalSqft = double.parse(_sqftController.text);
      final double bath = double.parse(_bathController.text);
      final double balcony = double.parse(_balconyController.text);
      final double size = double.parse(_sizeController.text);

      final inputFeatures = [totalSqft, bath, balcony, size];

      // Run prediction using the ONNX model
      final predicted = await predictHousePrice(inputFeatures);

      setState(() {
        _predictedPrice = predicted;
        _isLoading = false;
      });
    }
  }

  @override
  void dispose() {
    _sqftController.dispose();
    _bathController.dispose();
    _balconyController.dispose();
    _sizeController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('House Price Prediction'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: Column(
            children: [
              TextFormField(
                controller: _sqftController,
                decoration: const InputDecoration(
                  labelText: 'Total Sqft',
                  hintText: 'Enter total square footage',
                ),
                keyboardType: TextInputType.number,
                validator: (value) => value == null || value.isEmpty
                    ? 'Please enter total sqft'
                    : null,
              ),
              TextFormField(
                controller: _bathController,
                decoration: const InputDecoration(
                  labelText: 'Bath',
                  hintText: 'Enter number of bathrooms',
                ),
                keyboardType: TextInputType.number,
                validator: (value) => value == null || value.isEmpty
                    ? 'Please enter number of bathrooms'
                    : null,
              ),
              TextFormField(
                controller: _balconyController,
                decoration: const InputDecoration(
                  labelText: 'Balcony',
                  hintText: 'Enter number of balconies',
                ),
                keyboardType: TextInputType.number,
                validator: (value) => value == null || value.isEmpty
                    ? 'Please enter number of balconies'
                    : null,
              ),
              TextFormField(
                controller: _sizeController,
                decoration: const InputDecoration(
                  labelText: 'Size (Bedrooms)',
                  hintText: 'Enter number of bedrooms',
                ),
                keyboardType: TextInputType.number,
                validator: (value) =>
                    value == null || value.isEmpty ? 'Please enter size' : null,
              ),
              const SizedBox(height: 20),
              _isLoading
                  ? const CircularProgressIndicator()
                  : ElevatedButton(
                      onPressed: _predictPrice,
                      child: const Text('Predict Price'),
                    ),
              const SizedBox(height: 20),
              if (_predictedPrice != null)
                Text(
                  'Predicted Price: ${_predictedPrice!.toStringAsFixed(2)} Lakhs',
                  style: const TextStyle(
                      fontSize: 20, fontWeight: FontWeight.bold),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
