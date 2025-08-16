# -*- coding: utf-8 -*-
from flask import Flask, render_template_string, request, jsonify
from datetime import datetime, timedelta
import json
import random

app = Flask(__name__)

# Sample dataset generator
def generate_sample_data(days=90):
    base_date = datetime.today()
    dates = [(base_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)][::-1]
    
    data = {
        "dates": dates,
        "daily_revenue": [random.randint(8000, 12000) for _ in range(days)],
        "dso": random.randint(5, 30),
        "dpo": random.randint(15, 60),
        "expenses": [
            {"name": "Hotel Payments", "amount": 0.65, "frequency": "daily", "type": "percent"},
            {"name": "Marketing", "amount": 18000, "frequency": "monthly", "type": "fixed"},
            {"name": "Salaries", "amount": 45000, "frequency": "monthly", "type": "fixed"},
            {"name": "Platform Fees", "amount": 0.08, "frequency": "daily", "type": "percent"},
            {"name": "Payment Processing", "amount": 0.025, "frequency": "daily", "type": "percent"}
        ],
        "cash_reserve": 150000
    }
    return data

# Cash flow calculation engine
def calculate_cashflow(params):
    dso = params['dso']
    dpo = params['dpo']
    expenses = params['expenses']
    starting_cash = params.get('cash_reserve', 0)
    
    # Generate daily revenue data if not provided
    if 'daily_revenue' in params and len(params['daily_revenue']) > 0:
        daily_revenue = params['daily_revenue']
    else:
        daily_revenue = [random.randint(8000, 12000) for _ in range(90)]
    
    periods = len(daily_revenue)
    cash_in = [0] * periods
    cash_out = [0] * periods
    net_cashflow = [0] * periods
    cumulative_cash = [starting_cash] * periods
    
    # Calculate cash in with DSO delay
    for i in range(periods):
        if i >= dso:
            cash_in[i] = daily_revenue[i - dso]
    
    # Calculate cash out with expenses and DPO
    for i in range(periods):
        # Supplier payments with DPO delay (70% of revenue)
        supplier_cost = daily_revenue[i] * 0.70
        if i >= dpo:
            cash_out[i] += supplier_cost
        
        # Other expenses
        for expense in expenses:
            try:
                amount = float(expense['amount'])
                if expense['frequency'] == 'daily':
                    if expense['type'] == 'percent':
                        cash_out[i] += daily_revenue[i] * amount
                    else:
                        cash_out[i] += amount
                elif expense['frequency'] == 'monthly' and i % 30 == 0:
                    cash_out[i] += amount
            except ValueError:
                continue
        
        # Net cash flow calculations
        net_cashflow[i] = cash_in[i] - cash_out[i]
        if i > 0:
            cumulative_cash[i] = cumulative_cash[i-1] + net_cashflow[i]
    
    return {
        "cash_in": cash_in,
        "cash_out": cash_out,
        "net_cashflow": net_cashflow,
        "cumulative_cash": cumulative_cash,
        "daily_revenue": daily_revenue
    }

# HTML template with proper encoding
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>OTA Cash Flow Model</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/luxon@3.3.0/build/global/luxon.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1.3.1"></script>
    <style>
        :root {
            --primary: #4361ee;
            --secondary: #3f37c9;
            --success: #4cc9f0;
            --danger: #f72585;
            --light: #f8f9fa;
            --dark: #212529;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f7fb;
            color: #333;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .dashboard {
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        .card {
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        .card-header {
            font-weight: 600;
            font-size: 1.2rem;
            margin-bottom: 15px;
            color: var(--primary);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
        }
        input, select {
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 1rem;
        }
        button {
            background: var(--primary);
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 600;
            width: 100%;
            margin-top: 10px;
        }
        button:hover {
            background: var(--secondary);
        }
        .chart-container {
            height: 400px;
            position: relative;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .metric-card {
            background: white;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            margin: 10px 0;
        }
        .metric-title {
            color: #666;
            font-size: 0.9rem;
        }
        .cash-positive { color: #2ecc71; }
        .cash-negative { color: #e74c3c; }
        .expense-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        .tabs {
            display: flex;
            margin-bottom: 15px;
        }
        .tab {
            padding: 10px 15px;
            cursor: pointer;
            background: #eee;
            margin-right: 5px;
            border-radius: 4px 4px 0 0;
        }
        .tab.active {
            background: var(--primary);
            color: white;
        }
        .remove-btn {
            background: #e74c3c;
            width: 30px;
            padding: 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>OTA Cash Flow Simulator</h1>
        <p>Model your cash flow based on DSO, DPO, and operational expenses</p>
        
        <div class="dashboard">
            <div class="control-panel">
                <div class="card">
                    <div class="card-header">Parameters</div>
                    <div class="form-group">
                        <label for="dso">DSO (Days Sales Outstanding)</label>
                        <input type="number" id="dso" value="30">
                    </div>
                    <div class="form-group">
                        <label for="dpo">DPO (Days Payable Outstanding)</label>
                        <input type="number" id="dpo" value="45">
                    </div>
                    <div class="form-group">
                        <label for="cash-reserve">Starting Cash Reserve ($)</label>
                        <input type="number" id="cash-reserve" value="150000">
                    </div>
                    <button id="run-simulation">Run Simulation</button>
                    <button id="sample-data" style="background: #6c757d;">Load Sample Data</button>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <span>Expenses</span>
                        <button style="width: auto; padding: 2px 8px;" id="add-expense">+ Add Expense</button>
                    </div>
                    <div id="expenses-container">
                        <!-- Expenses will be added here dynamically -->
                    </div>
                </div>
            </div>
            
            <div class="results-panel">
                <div class="tabs">
                    <div class="tab active" data-tab="cash-flow">Cash Flow</div>
                    <div class="tab" data-tab="metrics">Key Metrics</div>
                </div>
                
                <div class="card">
                    <div class="card-header">Cash Flow Projection (90 Days)</div>
                    <div class="chart-container">
                        <canvas id="cashflow-chart"></canvas>
                    </div>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-title">Cash Runway</div>
                        <div class="metric-value" id="cash-runway">0 days</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Min Cash Balance</div>
                        <div class="metric-value" id="min-cash">$0</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Avg Daily Burn</div>
                        <div class="metric-value" id="daily-burn">$0</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Cash Efficiency</div>
                        <div class="metric-value" id="cash-efficiency">0.0</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // DOM elements
        const dsoInput = document.getElementById('dso');
        const dpoInput = document.getElementById('dpo');
        const cashReserveInput = document.getElementById('cash-reserve');
        const runBtn = document.getElementById('run-simulation');
        const sampleBtn = document.getElementById('sample-data');
        const addExpenseBtn = document.getElementById('add-expense');
        const expensesContainer = document.getElementById('expenses-container');
        const tabs = document.querySelectorAll('.tab');
        
        // Chart initialization
        const ctx = document.getElementById('cashflow-chart').getContext('2d');
        let cashflowChart = new Chart(ctx, {
            type: 'line',
            data: { datasets: [] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: { unit: 'day' }
                    },
                    y: {
                        beginAtZero: true,
                        ticks: { 
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        }
                    }
                },
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': $' + context.parsed.y.toLocaleString();
                            }
                        }
                    }
                }
            }
        });
        
        // Sample data loader
        sampleBtn.addEventListener('click', function() {
            fetch('/sample-data')
                .then(response => response.json())
                .then(sampleData => {
                    loadData(sampleData);
                    runSimulation();
                });
        });
        
        // Run simulation
        runBtn.addEventListener('click', runSimulation);
        
        // Tab switching
        tabs.forEach(tab => {
            tab.addEventListener('click', function() {
                tabs.forEach(t => t.classList.remove('active'));
                this.classList.add('active');
            });
        });
        
        // Add expense row
        addExpenseBtn.addEventListener('click', function() {
            const expenseRow = document.createElement('div');
            expenseRow.className = 'expense-item';
            expenseRow.innerHTML = `
                <div style="flex: 2;">
                    <input type="text" placeholder="Expense name" class="expense-name">
                </div>
                <div style="flex: 1;">
                    <input type="number" placeholder="Amount" class="expense-amount">
                </div>
                <div style="flex: 1;">
                    <select class="expense-frequency">
                        <option value="daily">Daily</option>
                        <option value="monthly">Monthly</option>
                    </select>
                </div>
                <div style="flex: 1;">
                    <select class="expense-type">
                        <option value="fixed">Fixed</option>
                        <option value="percent">% of Revenue</option>
                    </select>
                </div>
                <button class="remove-expense remove-btn">X</button>
            `;
            expensesContainer.appendChild(expenseRow);
            
            // Add remove functionality
            expenseRow.querySelector('.remove-expense').addEventListener('click', function() {
                expenseRow.remove();
            });
        });
        
        // Load data into form
        function loadData(data) {
            dsoInput.value = data.dso;
            dpoInput.value = data.dpo;
            cashReserveInput.value = data.cash_reserve;
            
            // Clear existing expenses
            expensesContainer.innerHTML = '';
            
            // Load expenses
            data.expenses.forEach(expense => {
                const expenseRow = document.createElement('div');
                expenseRow.className = 'expense-item';
                expenseRow.innerHTML = `
                    <div style="flex: 2;">
                        <input type="text" value="${expense.name}" class="expense-name">
                    </div>
                    <div style="flex: 1;">
                        <input type="number" value="${expense.amount}" class="expense-amount">
                    </div>
                    <div style="flex: 1;">
                        <select class="expense-frequency">
                            <option value="daily" ${expense.frequency === 'daily' ? 'selected' : ''}>Daily</option>
                            <option value="monthly" ${expense.frequency === 'monthly' ? 'selected' : ''}>Monthly</option>
                        </select>
                    </div>
                    <div style="flex: 1;">
                        <select class="expense-type">
                            <option value="fixed" ${expense.type === 'fixed' ? 'selected' : ''}>Fixed</option>
                            <option value="percent" ${expense.type === 'percent' ? 'selected' : ''}>% of Revenue</option>
                        </select>
                    </div>
                    <button class="remove-expense remove-btn">X</button>
                `;
                expensesContainer.appendChild(expenseRow);
                
                // Add remove functionality
                expenseRow.querySelector('.remove-expense').addEventListener('click', function() {
                    expenseRow.remove();
                });
            });
        }
        
        // Run simulation function
        async function runSimulation() {
            // Gather data from form
            const expenses = [];
            document.querySelectorAll('.expense-item').forEach(row => {
                const name = row.querySelector('.expense-name').value;
                const amount = row.querySelector('.expense-amount').value;
                const frequency = row.querySelector('.expense-frequency').value;
                const type = row.querySelector('.expense-type').value;
                
                if (name && amount) {
                    expenses.push({
                        name: name,
                        amount: amount,
                        frequency: frequency,
                        type: type
                    });
                }
            });
            
            const payload = {
                dso: parseInt(dsoInput.value) || 30,
                dpo: parseInt(dpoInput.value) || 45,
                cash_reserve: parseFloat(cashReserveInput.value) || 150000,
                expenses: expenses
            };
            
            // Send to backend
            try {
                const response = await fetch('/calculate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                const results = await response.json();
                visualizeResults(results);
                calculateMetrics(results);
            } catch (error) {
                console.error('Error:', error);
                alert('Error in calculation: ' + error.message);
            }
        }
        
        // Visualize results
        function visualizeResults(data) {
            // Generate dates for the past 90 days
            const dates = [];
            const today = new Date();
            for (let i = 0; i < 90; i++) {
                const date = new Date(today);
                date.setDate(date.getDate() - 89 + i);
                dates.push(date);
            }
            
            // Update chart data
            cashflowChart.data = {
                labels: dates,
                datasets: [
                    {
                        label: 'Cash In',
                        data: data.cash_in,
                        borderColor: '#2ecc71',
                        backgroundColor: 'rgba(46, 204, 113, 0.1)',
                        tension: 0.3
                    },
                    {
                        label: 'Cash Out',
                        data: data.cash_out,
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        tension: 0.3
                    },
                    {
                        label: 'Cash Balance',
                        data: data.cumulative_cash,
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.3
                    }
                ]
            };
            
            cashflowChart.update();
        }
        
        // Calculate key metrics
        function calculateMetrics(data) {
            const cumulativeCash = data.cumulative_cash;
            const minCash = Math.min(...cumulativeCash);
            const lastCash = cumulativeCash[cumulativeCash.length - 1];
            
            // Calculate average daily burn
            const totalOutflow = data.cash_out.reduce((a, b) => a + b, 0);
            const totalInflow = data.cash_in.reduce((a, b) => a + b, 0);
            const netCashflow = totalInflow - totalOutflow;
            const avgDailyBurn = netCashflow < 0 ? (totalOutflow - totalInflow) / cumulativeCash.length : 0;
            
            // Calculate cash runway
            let cashRunway = 0;
            if (avgDailyBurn > 0) {
                cashRunway = Math.floor(lastCash / avgDailyBurn);
            }
            
            // Calculate cash efficiency
            const cashEfficiency = totalInflow / totalOutflow;
            
            // Update UI
            document.getElementById('min-cash').textContent = '$' + Math.round(minCash).toLocaleString();
            document.getElementById('min-cash').className = 'metric-value ' + (minCash < 0 ? 'cash-negative' : 'cash-positive');
            document.getElementById('daily-burn').textContent = '$' + Math.round(avgDailyBurn).toLocaleString();
            document.getElementById('cash-runway').textContent = cashRunway + ' days';
            document.getElementById('cash-efficiency').textContent = cashEfficiency.toFixed(2);
        }
        
        // Initialize with sample data
        document.addEventListener('DOMContentLoaded', function() {
            // Load initial sample data
            fetch('/sample-data')
                .then(response => response.json())
                .then(sampleData => {
                    loadData(sampleData);
                    runSimulation();
                });
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    sample_data = generate_sample_data()
    return render_template_string(HTML_TEMPLATE, sample_data=json.dumps(sample_data))

@app.route('/sample-data')
def sample_data():
    return jsonify(generate_sample_data())

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    results = calculate_cashflow(data)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)