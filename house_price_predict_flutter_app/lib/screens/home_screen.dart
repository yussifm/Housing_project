import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../utils/predict.dart'; // Import your prediction utility

class HousePricePredictionScreen extends StatefulWidget {
  const HousePricePredictionScreen({Key? key}) : super(key: key);

  @override
  _HousePricePredictionScreenState createState() =>
      _HousePricePredictionScreenState();
}

class _HousePricePredictionScreenState
    extends State<HousePricePredictionScreen> {
  // Controllers for input fields
  final TextEditingController _sqftController = TextEditingController();
  final TextEditingController _bathController = TextEditingController();
  final TextEditingController _balconyController = TextEditingController();
  final TextEditingController _bhkController = TextEditingController();

  // Prediction result
  double? _predictedPrice;
  bool _isLoading = false;

  // Input validation
  final _formKey = GlobalKey<FormState>();

  // Predict house price
  Future<void> _predictPrice() async {
    if (_formKey.currentState!.validate()) {
      setState(() {
        _isLoading = true;
        _predictedPrice = null;
      });

      try {
        // Convert input to doubles
        final inputFeatures = [
          double.parse(_sqftController.text),
          double.parse(_bathController.text),
          double.parse(_balconyController.text),
          double.parse(_bhkController.text)
        ];

        // Call prediction function
        final result = await predictHousePrice(inputFeatures);

        setState(() {
          _predictedPrice = result;
          _isLoading = false;
        });
      } catch (e) {
        setState(() {
          _isLoading = false;
        });
        // Show error dialog
        _showErrorDialog(e.toString());
      }
    }
  }

  // Error dialog
  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Error'),
        content: Text(message),
        actions: <Widget>[
          TextButton(
            child: const Text('Okay'),
            onPressed: () {
              Navigator.of(ctx).pop();
            },
          )
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('House Price Predictor'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Input Card
              Card(
                elevation: 4,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      // SQFT Input
                      _buildInputField(
                        controller: _sqftController,
                        label: 'Square Feet',
                        hint: 'Enter total area',
                        icon: Icons.square_foot,
                      ),
                      const SizedBox(height: 16),

                      // Bathrooms Input
                      _buildInputField(
                        controller: _bathController,
                        label: 'Bathrooms',
                        hint: 'Number of bathrooms',
                        icon: Icons.bathroom,
                      ),
                      const SizedBox(height: 16),

                      // Balcony Input
                      _buildInputField(
                        controller: _balconyController,
                        label: 'Balconies',
                        hint: 'Number of balconies',
                        icon: Icons.balcony,
                      ),
                      const SizedBox(height: 16),

                      // BHK (Bedrooms) Input
                      _buildInputField(
                        controller: _bhkController,
                        label: 'BHK',
                        hint: 'Number of bedrooms',
                        icon: Icons.king_bed,
                      ),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 24),

              // Predict Button
              ElevatedButton(
                onPressed: _predictPrice,
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: _isLoading 
                  ? const CircularProgressIndicator()
                    : const Text(
                        'Predict Price',
                        style: TextStyle(fontSize: 18),
                      ),
              ),

              const SizedBox(height: 24),

              // Result Display
              if (_predictedPrice != null)
                Card(
                  color: Colors.green.shade50,
                  elevation: 4,
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      children: [
                        const Text(
                          'Predicted House Price',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 12),
                        Text(
                          'Ghc ${_predictedPrice!.toStringAsFixed(2)}',
                          style: TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.w600,
                            color: Colors.green.shade800,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  // Reusable input field builder
  Widget _buildInputField({
    required TextEditingController controller,
    required String label,
    required String hint,
    required IconData icon,
  }) {
    return TextFormField(
      controller: controller,
      decoration: InputDecoration(
        prefixIcon: Icon(icon),
        labelText: label,
        hintText: hint,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
        ),
      ),
      keyboardType: TextInputType.number,
      inputFormatters: [
        FilteringTextInputFormatter.allow(RegExp(r'^\d*\.?\d{0,2}')),
      ],
      validator: (value) {
        if (value == null || value.isEmpty) {
          return 'Please enter a value';
        }
        // Validate numeric input
        if (double.tryParse(value) == null) {
          return 'Please enter a valid number';
        }
        return null;
      },
    );
  }

  // Clean up controllers
  @override
  void dispose() {
    _sqftController.dispose();
    _bathController.dispose();
    _balconyController.dispose();
    _bhkController.dispose();
    super.dispose();
  }
}
