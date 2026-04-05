import 'package:flutter/material.dart';
import 'api_service.dart';
import 'results_screen.dart';

class ParameterScreen extends StatefulWidget {
  const ParameterScreen({super.key});

  @override
  State<ParameterScreen> createState() => _ParameterScreenState();
}

class _ParameterScreenState extends State<ParameterScreen> {
  // QUBO parameters
  double _lambda = 0.1;
  double _budget = 4.0;
  double _riskAversion = 0.7;
  int _numAssets = 8;

  // QAOA parameters
  int _numLayers = 10;
  int _maxTerms = 100;
  final double _absCutoff = 1e-6;

  // Simulation parameters
  String _noiseModel = 'none';
  double _noiseProb = 0.0;
  int _gammaSteps = 7;
  int _betaSteps = 7;
  int _shots = 1000;

  bool _isRunning = false;
  bool _backendOnline = false;
  bool _checkingBackend = true;

  @override
  void initState() {
    super.initState();
    _checkBackend();
  }

  Future<void> _checkBackend() async {
    setState(() => _checkingBackend = true);
    final online = await ApiService.healthCheck();
    setState(() {
      _backendOnline = online;
      _checkingBackend = false;
    });
  }

  Future<void> _runSimulation() async {
    setState(() => _isRunning = true);
    try {
      final result = await ApiService.runSimulation(
        lambda: _lambda,
        budget: _budget,
        riskAversion: _riskAversion,
        numAssets: _numAssets,
        numLayers: _numLayers,
        maxTerms: _maxTerms,
        absCutoff: _absCutoff,
        noiseModel: _noiseModel,
        noiseProb: _noiseProb,
        gammaSteps: _gammaSteps,
        betaSteps: _betaSteps,
        shots: _shots,
      );

      if (!mounted) return;

      if (result['success'] == true) {
        Navigator.of(context).push(
          MaterialPageRoute(
            builder: (_) => ResultsScreen(data: result),
          ),
        );
      } else {
        _showError(result['error'] ?? 'Unknown error');
      }
    } catch (e) {
      if (mounted) _showError(e.toString());
    } finally {
      if (mounted) setState(() => _isRunning = false);
    }
  }

  void _showError(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg), backgroundColor: Colors.red),
    );
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('QAOA Portfolio Optimizer'),
        actions: [
          IconButton(
            icon: Icon(
              _checkingBackend
                  ? Icons.sync
                  : _backendOnline
                      ? Icons.cloud_done
                      : Icons.cloud_off,
              color: _backendOnline ? Colors.green : Colors.red,
            ),
            onPressed: _checkBackend,
            tooltip: _backendOnline ? 'Backend online' : 'Backend offline',
          ),
        ],
      ),
      body: _isRunning ? _buildRunning() : _buildForm(cs),
    );
  }

  Widget _buildRunning() {
    return const Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: 80,
            height: 80,
            child: CircularProgressIndicator(strokeWidth: 6),
          ),
          SizedBox(height: 24),
          Text(
            'Running QAOA Simulation...',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.w500),
          ),
          SizedBox(height: 8),
          Text(
            'This may take a minute for larger parameter spaces.',
            style: TextStyle(fontSize: 14, color: Colors.grey),
          ),
        ],
      ),
    );
  }

  Widget _buildForm(ColorScheme cs) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        // QUBO Parameters
        _sectionHeader('Portfolio Parameters', Icons.account_balance),
        _sliderTile(
          'Risk Aversion (q)',
          _riskAversion,
          0.0,
          2.0,
          (v) => setState(() => _riskAversion = double.parse(v.toStringAsFixed(2))),
          decimals: 2,
        ),
        _sliderTile(
          'Budget (B)',
          _budget,
          1.0,
          20.0,
          (v) => setState(() => _budget = v.roundToDouble()),
          decimals: 0,
        ),
        _sliderTile(
          'Lambda',
          _lambda,
          0.01,
          2.0,
          (v) => setState(() => _lambda = double.parse(v.toStringAsFixed(2))),
          decimals: 2,
        ),
        _sliderTile(
          'Num Assets',
          _numAssets.toDouble(),
          2,
          16,
          (v) => setState(() => _numAssets = v.round()),
          decimals: 0,
        ),
        const SizedBox(height: 16),

        // QAOA Parameters
        _sectionHeader('QAOA Parameters', Icons.settings),
        _sliderTile(
          'Layers (p)',
          _numLayers.toDouble(),
          1,
          30,
          (v) => setState(() => _numLayers = v.round()),
          decimals: 0,
        ),
        _sliderTile(
          'Max Pauli Terms',
          _maxTerms.toDouble(),
          10,
          500,
          (v) => setState(() => _maxTerms = v.round()),
          decimals: 0,
        ),
        _sliderTile(
          'Gamma Grid Steps',
          _gammaSteps.toDouble(),
          3,
          15,
          (v) => setState(() => _gammaSteps = v.round()),
          decimals: 0,
        ),
        _sliderTile(
          'Beta Grid Steps',
          _betaSteps.toDouble(),
          3,
          15,
          (v) => setState(() => _betaSteps = v.round()),
          decimals: 0,
        ),
        const SizedBox(height: 16),

        // Simulation Parameters
        _sectionHeader('Simulation', Icons.science),
        _sliderTile(
          'Shots',
          _shots.toDouble(),
          100,
          10000,
          (v) => setState(() => _shots = v.round()),
          decimals: 0,
        ),
        ListTile(
          title: const Text('Noise Model'),
          trailing: SegmentedButton<String>(
            segments: const [
              ButtonSegment(value: 'none', label: Text('None')),
              ButtonSegment(value: 'bit_flip', label: Text('Bit Flip')),
            ],
            selected: {_noiseModel},
            onSelectionChanged: (v) => setState(() => _noiseModel = v.first),
          ),
        ),
        if (_noiseModel == 'bit_flip')
          _sliderTile(
            'Noise Probability',
            _noiseProb,
            0.0,
            0.2,
            (v) => setState(() => _noiseProb = double.parse(v.toStringAsFixed(3))),
            decimals: 3,
          ),
        const SizedBox(height: 24),

        // Run button
        FilledButton.icon(
          onPressed: _backendOnline ? _runSimulation : null,
          icon: const Icon(Icons.play_arrow),
          label: Padding(
            padding: const EdgeInsets.symmetric(vertical: 14),
            child: Text(
              _backendOnline ? 'Run Simulation' : 'Backend Offline',
              style: const TextStyle(fontSize: 18),
            ),
          ),
        ),
        if (!_backendOnline && !_checkingBackend)
          Padding(
            padding: const EdgeInsets.only(top: 8),
            child: Text(
              'Start the backend: python backend/server.py',
              textAlign: TextAlign.center,
              style: TextStyle(color: cs.error, fontSize: 13),
            ),
          ),
        const SizedBox(height: 32),
      ],
    );
  }

  Widget _sectionHeader(String title, IconData icon) {
    return Padding(
      padding: const EdgeInsets.only(top: 8, bottom: 4),
      child: Row(
        children: [
          Icon(icon, size: 20),
          const SizedBox(width: 8),
          Text(
            title,
            style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }

  Widget _sliderTile(
    String label,
    double value,
    double min,
    double max,
    ValueChanged<double> onChanged, {
    int decimals = 1,
  }) {
    return ListTile(
      title: Text(label),
      subtitle: Slider(
        value: value.clamp(min, max),
        min: min,
        max: max,
        divisions: decimals == 0 ? (max - min).round() : null,
        label: value.toStringAsFixed(decimals),
        onChanged: onChanged,
      ),
      trailing: SizedBox(
        width: 60,
        child: Text(
          value.toStringAsFixed(decimals),
          textAlign: TextAlign.right,
          style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 15),
        ),
      ),
    );
  }
}
