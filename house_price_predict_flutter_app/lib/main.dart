import 'package:flutter/material.dart';

import 'screens/home_screen.dart';

void main() {
  runApp(const HousePriceApp());
}

class HousePriceApp extends StatelessWidget {
  const HousePriceApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'House Price Prediction',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const HousePricePredictionScreen(),
    );
  }
}
