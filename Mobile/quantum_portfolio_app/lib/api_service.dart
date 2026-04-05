import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  // localhost on iOS simulator maps to the Mac host
  static const String baseUrl = 'http://localhost:5001';

  static Future<bool> healthCheck() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/api/health'))
          .timeout(const Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  static Future<List<Map<String, dynamic>>> getAssets() async {
    final response = await http.get(Uri.parse('$baseUrl/api/assets'));
    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      return List<Map<String, dynamic>>.from(data['assets']);
    }
    throw Exception('Failed to load assets');
  }

  static Future<Map<String, dynamic>> runSimulation({
    required double lambda,
    required double budget,
    required double riskAversion,
    required int numAssets,
    required int numLayers,
    required int maxTerms,
    required double absCutoff,
    required String noiseModel,
    required double noiseProb,
    required int gammaSteps,
    required int betaSteps,
    required int shots,
  }) async {
    final body = json.encode({
      'lambda': lambda,
      'budget': budget,
      'risk_aversion': riskAversion,
      'num_assets': numAssets,
      'num_layers': numLayers,
      'max_terms': maxTerms,
      'abs_cutoff': absCutoff,
      'noise_model': noiseModel,
      'noise_prob': noiseProb,
      'gamma_steps': gammaSteps,
      'beta_steps': betaSteps,
      'shots': shots,
    });

    final response = await http
        .post(
          Uri.parse('$baseUrl/api/run'),
          headers: {'Content-Type': 'application/json'},
          body: body,
        )
        .timeout(const Duration(minutes: 5));

    if (response.statusCode == 200) {
      return json.decode(response.body);
    }
    final err = json.decode(response.body);
    throw Exception(err['error'] ?? 'Simulation failed');
  }
}
