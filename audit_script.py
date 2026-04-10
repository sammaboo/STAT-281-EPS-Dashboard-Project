import pandas as pd
import numpy as np

df = pd.read_csv(r'c:\Users\sam\Documents\ibes_compustat_merged.csv')
df['fpedats'] = pd.to_datetime(df['fpedats'])
df['statpers'] = pd.to_datetime(df['statpers'])

print('='*80)
print('AUDIT 1: What return columns exist and their basic stats')
print('='*80)
ret_cols = [c for c in df.columns if 'ret' in c.lower() or 'price' in c.lower()]
print(f'Return/price columns: {ret_cols}')
for col in ret_cols:
    print(f'\n{col}:')
    print(f'  non-null: {df[col].notna().sum()}/{len(df)}')
    print(f'  min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}')
    print(f'  sample: {df[col].dropna().head(5).tolist()}')

print()
print('='*80)
print('AUDIT 2: NVDA - Check quarterly_ret values vs what we expect')
print('='*80)
nvda = df[df['ticker']=='NVDA'].copy()
nvda_q = nvda.sort_values('statpers').groupby('fpedats').agg({
    'quarterly_ret': 'first',
    'quarter_end_price': 'first',
    'annual_ret': 'first',
    'year_end_price': 'first',
    'fyear': 'first',
    'fpe_year': 'first',
    'fpe_quarter': 'first',
    'actual': 'first'
}).reset_index().sort_values('fpedats')
print('NVDA unique quarters:')
print(nvda_q[['fpedats','fyear','fpe_year','fpe_quarter','quarter_end_price','quarterly_ret','year_end_price','annual_ret']].to_string())

print()
print('='*80)
print('AUDIT 3: Verify quarterly_ret calculation manually')
print('='*80)
nvda_q_sorted = nvda_q.sort_values('fpedats')
for i in range(1, min(8, len(nvda_q_sorted))):
    row = nvda_q_sorted.iloc[i]
    prev_row = nvda_q_sorted.iloc[i-1]
    expected = (abs(row['quarter_end_price']) - abs(prev_row['quarter_end_price'])) / abs(prev_row['quarter_end_price']) * 100
    stored = row['quarterly_ret']
    match = abs(expected - stored) < 0.5 if pd.notna(stored) and pd.notna(expected) else 'N/A'
    fpe_date = row['fpedats'].strftime('%Y-%m-%d')
    fpe_q = int(row['fpe_quarter'])
    qep = row['quarter_end_price']
    prev_qep = prev_row['quarter_end_price']
    print(f'{fpe_date} fpe_q={fpe_q}: price={qep:.2f}, prev_price={prev_qep:.2f}, expected_ret={expected:.2f}%, stored_ret={stored:.2f}%, match={match}')

print()
print('='*80)
print('AUDIT 4: Check if quarter_end_price aligns with fpedats quarter')
print('='*80)
print('NVDA fpedats vs fpe_year/fpe_quarter:')
for _, row in nvda_q_sorted.head(12).iterrows():
    fpe = row['fpedats']
    fpe_yr = int(row['fpe_year'])
    fpe_q = int(row['fpe_quarter'])
    fy = int(row['fyear'])
    print(f'  fpedats={fpe.strftime("%Y-%m-%d")} -> fpe_year={fpe_yr}, fpe_quarter={fpe_q}, fyear={fy}')
    cal_quarter = fpe.quarter
    cal_year = fpe.year
    print(f'    calendar: year={cal_year}, quarter={cal_quarter}')
    if cal_quarter != fpe_q or cal_year != fpe_yr:
        print(f'    *** MISMATCH! ***')
