import 'package:flutter_test/flutter_test.dart';
import 'package:quantum_portfolio_app/main.dart';

void main() {
  testWidgets('App launches', (WidgetTester tester) async {
    await tester.pumpWidget(const QuantumPortfolioApp());
    expect(find.text('QAOA Portfolio Optimizer'), findsOneWidget);
  });
}
