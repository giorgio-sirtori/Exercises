import csv
from collections import defaultdict

# File path for your input CSV
input_file = 'input.csv'
output_file = 'aggregated_items.csv'

# Dictionary to accumulate totals
item_totals = defaultdict(float)

with open(input_file, mode='r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) != 2:
            continue  # Skip malformed rows

        items_str, amount_str = row
        items = [item.strip() for item in items_str.split(',') if item.strip()]
        try:
            amount = float(amount_str.strip())
        except ValueError:
            continue  # Skip rows with invalid amount

        if not items:
            continue  # Skip if no valid items

        amount_per_item = amount / len(items)

        for item in items:
            item_totals[item] += amount_per_item

# Write output to CSV
with open(output_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Item", "Total Amount"])
    for item, total in sorted(item_totals.items()):
        writer.writerow([item, round(total, 2)])

print(f"✅ Done! Output saved to '{output_file}'")
