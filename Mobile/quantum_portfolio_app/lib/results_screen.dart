import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';

class ResultsScreen extends StatelessWidget {
  final Map<String, dynamic> data;

  const ResultsScreen({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    final params = data['parameters'] as Map<String, dynamic>;
    final results = List<Map<String, dynamic>>.from(data['results']);
    final stats = data['stats'] as Map<String, dynamic>;
    final best = results.isNotEmpty ? results.first : null;

    return Scaffold(
      appBar: AppBar(title: const Text('Simulation Results')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Tuned parameters card
          _buildTunedParamsCard(params),
          const SizedBox(height: 16),

          // Best portfolio card
          if (best != null) _buildBestPortfolioCard(best, params),
          const SizedBox(height: 16),

          // Probability distribution chart
          if (results.isNotEmpty) _buildProbChart(context, results),
          const SizedBox(height: 16),

          // Objective comparison chart
          if (results.isNotEmpty) _buildObjectiveChart(context, results, stats),
          const SizedBox(height: 16),

          // Full results table
          _buildResultsTable(results, stats),
          const SizedBox(height: 16),

          // Stats card
          _buildStatsCard(stats),
          const SizedBox(height: 32),
        ],
      ),
    );
  }

  Widget _buildTunedParamsCard(Map<String, dynamic> params) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.tune, size: 20),
                SizedBox(width: 8),
                Text('Tuned Parameters',
                    style:
                        TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
              ],
            ),
            const Divider(),
            Wrap(
              spacing: 24,
              runSpacing: 8,
              children: [
                _chip('Gamma', (params['best_gamma'] as num).toStringAsFixed(4)),
                _chip('Beta', (params['best_beta'] as num).toStringAsFixed(4)),
                _chip('Max Prob',
                    (params['best_max_prob'] as num).toStringAsFixed(4)),
                _chip('Layers', '${params['num_layers']}'),
                _chip('Noise', '${params['noise_model']}'),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBestPortfolioCard(
      Map<String, dynamic> best, Map<String, dynamic> params) {
    final selectedAssets = List<String>.from(best['selected_assets']);
    final budgetTarget = params['budget'] as num;
    final budgetMet = selectedAssets.length == budgetTarget.round();

    return Card(
      color: budgetMet ? Colors.green.shade50 : Colors.orange.shade50,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.star,
                    color: budgetMet ? Colors.green : Colors.orange, size: 20),
                const SizedBox(width: 8),
                const Text('Best Portfolio',
                    style:
                        TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
              ],
            ),
            const Divider(),
            Text('Bitstring: |${best['bitstring']}|',
                style: const TextStyle(
                    fontFamily: 'monospace', fontWeight: FontWeight.w600)),
            const SizedBox(height: 8),
            Text('Assets: ${selectedAssets.join(", ")}'),
            const SizedBox(height: 4),
            Text(
                'Count: ${selectedAssets.length} / ${budgetTarget.round()} (budget)'),
            const SizedBox(height: 8),
            Wrap(
              spacing: 16,
              runSpacing: 8,
              children: [
                _chip('Return',
                    (best['expected_return'] as num).toStringAsFixed(4)),
                _chip(
                    'Risk', (best['portfolio_risk'] as num).toStringAsFixed(4)),
                _chip('Variance',
                    (best['portfolio_variance'] as num).toStringAsFixed(4)),
                _chip(
                    'Objective', (best['objective'] as num).toStringAsFixed(4)),
                _chip('Prob',
                    (best['probability'] as num).toStringAsFixed(4)),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildProbChart(
      BuildContext context, List<Map<String, dynamic>> results) {
    final top = results.take(10).toList();

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Probability Distribution (Top 10)',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            const SizedBox(height: 16),
            SizedBox(
              height: 220,
              child: BarChart(
                BarChartData(
                  alignment: BarChartAlignment.spaceAround,
                  maxY: (top.first['probability'] as num).toDouble() * 1.2,
                  barTouchData: BarTouchData(
                    touchTooltipData: BarTouchTooltipData(
                      getTooltipItem: (group, groupIndex, rod, rodIndex) {
                        return BarTooltipItem(
                          '${top[groupIndex]['bitstring']}\np=${(top[groupIndex]['probability'] as num).toStringAsFixed(4)}',
                          const TextStyle(
                              color: Colors.white, fontSize: 11),
                        );
                      },
                    ),
                  ),
                  titlesData: FlTitlesData(
                    show: true,
                    bottomTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        getTitlesWidget: (value, meta) {
                          final idx = value.toInt();
                          if (idx < top.length) {
                            final bs = top[idx]['bitstring'] as String;
                            return Padding(
                              padding: const EdgeInsets.only(top: 4),
                              child: Text(
                                bs.length > 6
                                    ? '${bs.substring(0, 3)}...'
                                    : bs,
                                style: const TextStyle(fontSize: 9),
                              ),
                            );
                          }
                          return const SizedBox.shrink();
                        },
                        reservedSize: 28,
                      ),
                    ),
                    leftTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        reservedSize: 40,
                        getTitlesWidget: (value, meta) {
                          return Text(value.toStringAsFixed(3),
                              style: const TextStyle(fontSize: 10));
                        },
                      ),
                    ),
                    topTitles: const AxisTitles(
                        sideTitles: SideTitles(showTitles: false)),
                    rightTitles: const AxisTitles(
                        sideTitles: SideTitles(showTitles: false)),
                  ),
                  borderData: FlBorderData(show: false),
                  barGroups: List.generate(top.length, (i) {
                    return BarChartGroupData(
                      x: i,
                      barRods: [
                        BarChartRodData(
                          toY: (top[i]['probability'] as num).toDouble(),
                          color: Theme.of(context).colorScheme.primary,
                          width: 16,
                          borderRadius: const BorderRadius.vertical(
                              top: Radius.circular(4)),
                        ),
                      ],
                    );
                  }),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildObjectiveChart(BuildContext context,
      List<Map<String, dynamic>> results, Map<String, dynamic> stats) {
    final top = results.take(10).toList();
    final avgObj = (stats['avg_objective'] as num).toDouble();

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Objective vs Average (Top 10)',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            const SizedBox(height: 16),
            SizedBox(
              height: 220,
              child: BarChart(
                BarChartData(
                  alignment: BarChartAlignment.spaceAround,
                  barTouchData: BarTouchData(
                    touchTooltipData: BarTouchTooltipData(
                      getTooltipItem: (group, groupIndex, rod, rodIndex) {
                        final obj = (top[groupIndex]['objective'] as num);
                        return BarTooltipItem(
                          'obj=${obj.toStringAsFixed(4)}\navg=${avgObj.toStringAsFixed(4)}',
                          const TextStyle(
                              color: Colors.white, fontSize: 11),
                        );
                      },
                    ),
                  ),
                  titlesData: FlTitlesData(
                    show: true,
                    bottomTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        getTitlesWidget: (value, meta) {
                          final idx = value.toInt();
                          if (idx < top.length) {
                            return Padding(
                              padding: const EdgeInsets.only(top: 4),
                              child: Text('#${idx + 1}',
                                  style: const TextStyle(fontSize: 10)),
                            );
                          }
                          return const SizedBox.shrink();
                        },
                        reservedSize: 24,
                      ),
                    ),
                    leftTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        reservedSize: 48,
                        getTitlesWidget: (value, meta) {
                          return Text(value.toStringAsFixed(2),
                              style: const TextStyle(fontSize: 10));
                        },
                      ),
                    ),
                    topTitles: const AxisTitles(
                        sideTitles: SideTitles(showTitles: false)),
                    rightTitles: const AxisTitles(
                        sideTitles: SideTitles(showTitles: false)),
                  ),
                  borderData: FlBorderData(show: false),
                  barGroups: List.generate(top.length, (i) {
                    final obj = (top[i]['objective'] as num).toDouble();
                    final diff = obj - avgObj;
                    return BarChartGroupData(
                      x: i,
                      barRods: [
                        BarChartRodData(
                          toY: obj,
                          color: diff < 0 ? Colors.green : Colors.red.shade300,
                          width: 16,
                          borderRadius: const BorderRadius.vertical(
                              top: Radius.circular(4)),
                        ),
                      ],
                    );
                  }),
                  extraLinesData: ExtraLinesData(
                    horizontalLines: [
                      HorizontalLine(
                        y: avgObj,
                        color: Colors.orange,
                        strokeWidth: 2,
                        dashArray: [6, 4],
                        label: HorizontalLineLabel(
                          show: true,
                          alignment: Alignment.topRight,
                          style: const TextStyle(
                              fontSize: 10, color: Colors.orange),
                          labelResolver: (_) => 'avg',
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultsTable(
      List<Map<String, dynamic>> results, Map<String, dynamic> stats) {
    final avgObj = (stats['avg_objective'] as num).toDouble();

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('All Results',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: DataTable(
                columnSpacing: 16,
                columns: const [
                  DataColumn(label: Text('#')),
                  DataColumn(label: Text('Bitstring')),
                  DataColumn(label: Text('Assets')),
                  DataColumn(label: Text('Prob')),
                  DataColumn(label: Text('Objective')),
                  DataColumn(label: Text('vs Avg')),
                ],
                rows: List.generate(results.length, (i) {
                  final r = results[i];
                  final obj = (r['objective'] as num).toDouble();
                  final diff = obj - avgObj;
                  return DataRow(cells: [
                    DataCell(Text('${i + 1}')),
                    DataCell(Text(r['bitstring'],
                        style: const TextStyle(fontFamily: 'monospace'))),
                    DataCell(Text('${r['num_selected']}')),
                    DataCell(Text(
                        (r['probability'] as num).toStringAsFixed(4))),
                    DataCell(Text(obj.toStringAsFixed(4))),
                    DataCell(Text(
                      '${diff >= 0 ? "+" : ""}${diff.toStringAsFixed(4)}',
                      style: TextStyle(
                          color: diff < 0 ? Colors.green : Colors.red),
                    )),
                  ]);
                }),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatsCard(Map<String, dynamic> stats) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.analytics, size: 20),
                SizedBox(width: 8),
                Text('Global Statistics',
                    style:
                        TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
              ],
            ),
            const Divider(),
            _statRow('Total bitstrings', '${stats['total_bitstrings']}'),
            _statRow('Qubits', '${stats['n_qubits']}'),
            _statRow('Avg objective',
                (stats['avg_objective'] as num).toStringAsFixed(6)),
            _statRow('Min objective',
                (stats['min_objective'] as num).toStringAsFixed(6)),
            _statRow('Min bitstring', '|${stats['min_bitstring']}|'),
          ],
        ),
      ),
    );
  }

  Widget _chip(String label, String value) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label,
            style: const TextStyle(fontSize: 11, color: Colors.grey)),
        Text(value,
            style:
                const TextStyle(fontSize: 14, fontWeight: FontWeight.w600)),
      ],
    );
  }

  Widget _statRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label),
          Text(value,
              style: const TextStyle(
                  fontWeight: FontWeight.w600, fontFamily: 'monospace')),
        ],
      ),
    );
  }
}
