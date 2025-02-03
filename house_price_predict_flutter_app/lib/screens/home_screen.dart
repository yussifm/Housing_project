import 'package:flutter/material.dart';
import 'package:intl/intl.dart'; // Import for number formatting
import '../utils/predict.dart'; // Import your prediction utility

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

      try {
        final double? totalSqft = double.tryParse(_sqftController.text);
        final double? bath = double.tryParse(_bathController.text);
        final double? balcony = double.tryParse(_balconyController.text);
        final double? size = double.tryParse(_sizeController.text);

        if (totalSqft == null ||
            bath == null ||
            balcony == null ||
            size == null) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Please enter valid numbers')),
          );
          setState(() {
            _isLoading = false;
          });
          return; // Stop prediction if input is invalid
        }

        final inputFeatures = [totalSqft, bath, balcony, size];
        final predicted = await predictHousePrice(inputFeatures);

        setState(() {
          _predictedPrice = predicted;
          _isLoading = false;
        });
      } catch (e) {
        setState(() {
          _isLoading = false;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e')), // User-friendly message
        );
        print('Error during prediction: $e'); // Log the error for debugging
      }
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
                  'Predicted Price: ${NumberFormat('#,##,##0', 'en_IN').format(_predictedPrice)} Lakhs', // Formatted price
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
