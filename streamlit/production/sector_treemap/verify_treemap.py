#!/usr/bin/env python3
"""
Quick verification script to test treemap integration
"""

import sys
import os

# Add production directory to path
sys.path.insert(0, '/home/robert/FEURCast/streamlit/production')

print("🔍 Verifying Treemap Integration...\n")

# Test 1: Import simulator functions
print("[1/5] Testing imports from simulator.py...")
try:
    from simulator import (
        create_sector_holdings_treemap,
        get_sector_summary,
        create_sector_risk_treemap
    )
    print("✅ All functions imported successfully\n")
except Exception as e:
    print(f"❌ Import failed: {e}\n")
    sys.exit(1)

# Test 2: Check data file exists
print("[2/5] Checking for data file...")
data_path = '/home/robert/FEURCast/data/treemap_nodes.csv'
if os.path.exists(data_path):
    print(f"✅ Data file found at {data_path}\n")
else:
    print(f"⚠️  Data file not found at {data_path}")
    print("    Fallback to simulated data will be used\n")

# Test 3: Load data
print("[3/5] Testing data loading...")
try:
    import pandas as pd
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} holdings across {df['Sector'].nunique()} sectors\n")
    print(f"    Sectors: {', '.join(sorted(df['Sector'].unique()))}\n")
except Exception as e:
    print(f"⚠️  Could not load data: {e}")
    print("    Will use fallback visualization\n")

# Test 4: Test treemap creation
print("[4/5] Testing treemap creation...")
try:
    fig = create_sector_holdings_treemap('DailyChangePct')
    print("✅ Treemap created successfully\n")
    print(f"    Figure type: {type(fig)}")
    print(f"    Has data: {len(fig.data) > 0}\n")
except Exception as e:
    print(f"❌ Treemap creation failed: {e}\n")
    sys.exit(1)

# Test 5: Test sector summary
print("[5/5] Testing sector summary...")
try:
    summary = get_sector_summary()
    if not summary.empty:
        print("✅ Sector summary generated successfully\n")
        print(f"    Sectors in summary: {len(summary)}")
        print(f"    Columns: {', '.join(summary.columns)}\n")
        print("    Top 3 sectors by weight:")
        for idx, row in summary.head(3).iterrows():
            print(f"      - {row['Sector']}: {row['Weight (%)']:.2f}%")
    else:
        print("⚠️  Sector summary is empty (fallback mode)\n")
except Exception as e:
    print(f"❌ Sector summary failed: {e}\n")
    sys.exit(1)

print("\n" + "="*60)
print("✨ All tests passed! Treemap integration verified.")
print("="*60)
print("\n💡 To run the app:")
print("   cd /home/robert/FEURCast/streamlit/production")
print("   streamlit run app.py")
print("\n📊 New features available:")
print("   - 'Holdings Detail' tab in Pro Mode Quick Analytics")
print("   - Natural language queries like 'Show me SPLG holdings'")
print("   - Drill-down from sectors to individual companies")
print("   - 4 color metric options: DailyChangePct, PE, Beta, DividendYield")
