"""
Static Site Generator for EPS Dashboard
========================================
Generates a static version of the dashboard in the docs/ folder for GitHub Pages.

Usage:
    python build.py

Workflow:
    1. Update your CSV data locally
    2. Run: python build.py
    3. git add . && git commit -m "Update data" && git push
    4. GitHub Pages serves the docs/ folder automatically
"""
import os
import sys
import json
import shutil
import re
import time

# Ensure we can import from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import (
    app, load_data, get_ticker_list, get_year_range, get_summary_stats,
    create_eps_history_chart, create_revision_trail_chart,
    create_dispersion_chart, create_eps_predictability_chart,
    create_company_comparison_chart, create_prediction_chart,
    create_surprise_prediction_chart, create_eps_surprise_returns_chart,
    create_eps_vs_returns_trend_chart, create_revenue_chart,
    create_eps_estimates_chart, create_profitability_chart,
    create_ebitda_margin_chart, create_analyst_coverage_chart,
    create_estimate_accuracy_chart, create_pead_chart,
    predict_eps
)

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs')
PREDICTION_METHODS = ['linear', 'exponential', 'moving_avg', 'analyst', 'holt_winters', 'sarima']
REVISION_QUARTER_OPTIONS = [4, 6, 8, 12]


def write_json(path, data):
    """Write compact JSON file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, separators=(',', ':'))


def export_ticker_data(df, ticker_symbols):
    """Export raw quarterly EPS data per ticker for client-side prediction engine"""
    import pandas as pd
    d = os.path.join(DOCS_DIR, 'api', 'ticker_data')
    os.makedirs(d, exist_ok=True)
    for ticker in ticker_symbols:
        td = df[df['ticker'] == ticker].copy()
        td = td.dropna(subset=['actual'])
        td['fpedats'] = pd.to_datetime(td['fpedats'])
        td['statpers'] = pd.to_datetime(td['statpers'])
        if 'fpi' in td.columns:
            td = td[td['fpi'].isin(['6', '7', '8', '9', 6, 7, 8, 9])]
        td = td[td['statpers'] <= td['fpedats']]
        td = td.drop_duplicates(subset=['ticker', 'fpedats', 'statpers'])
        eps = td.sort_values('statpers').groupby('fpedats').agg({
            'actual': 'first', 'meanest': 'last'
        }).reset_index().sort_values('fpedats')
        quarters = []
        for _, row in eps.iterrows():
            quarters.append({
                'd': row['fpedats'].strftime('%Y-%m-%d'),
                'a': round(float(row['actual']), 4),
                'e': round(float(row['meanest']), 4) if pd.notna(row['meanest']) else None
            })
        write_json(os.path.join(d, f'{ticker}.json'), {'quarters': quarters})


def generate_api_files(df, ticker_symbols):
    """Pre-generate all API JSON responses as static files"""
    api_dir = os.path.join(DOCS_DIR, 'api')
    total = len(ticker_symbols)
    
    # --- Single-ticker charts ---
    print(f"  eps_history ({total} tickers)...")
    d = os.path.join(api_dir, 'eps_history')
    for t in ticker_symbols:
        chart = create_eps_history_chart(df, t)
        write_json(os.path.join(d, f'{t}.json'), {'chart': chart})
    
    print(f"  revision_trail ({total} tickers x {len(REVISION_QUARTER_OPTIONS)} quarter options)...")
    d = os.path.join(api_dir, 'revision_trail')
    for t in ticker_symbols:
        for nq in REVISION_QUARTER_OPTIONS:
            chart, metrics = create_revision_trail_chart(df, t, nq)
            write_json(os.path.join(d, f'{t}_{nq}.json'), {'chart': chart, 'metrics': metrics})
    
    print(f"  dispersion ({total} tickers)...")
    d = os.path.join(api_dir, 'dispersion')
    for t in ticker_symbols:
        chart = create_dispersion_chart(df, t)
        write_json(os.path.join(d, f'{t}.json'), {'chart': chart})
    
    print(f"  predictability ({total} tickers + ALL)...")
    d = os.path.join(api_dir, 'predictability')
    # "All Companies" (no ticker filter)
    chart = create_eps_predictability_chart(df, None)
    write_json(os.path.join(d, 'ALL.json'), {'chart': chart})
    for t in ticker_symbols:
        chart = create_eps_predictability_chart(df, t)
        write_json(os.path.join(d, f'{t}.json'), {'chart': chart})
    
    print(f"  eps_returns_trend ({total} tickers)...")
    d = os.path.join(api_dir, 'eps_returns_trend')
    for t in ticker_symbols:
        chart = create_eps_vs_returns_trend_chart(df, t)
        write_json(os.path.join(d, f'{t}.json'), {'chart': chart})
    
    print(f"  surprise_analysis ({total} tickers)...")
    d = os.path.join(api_dir, 'surprise_analysis')
    for t in ticker_symbols:
        chart = create_surprise_prediction_chart(df, t)
        write_json(os.path.join(d, f'{t}.json'), {'chart': chart})
    
    # --- Global charts (no ticker param) ---
    print("  eps_surprise_returns (1 file)...")
    d = os.path.join(api_dir, 'eps_surprise_returns')
    chart = create_eps_surprise_returns_chart(df)
    write_json(os.path.join(d, 'all.json'), {'chart': chart})
    
    # --- Prediction charts (ticker x method) ---
    print(f"  prediction & prediction_data ({total} tickers x {len(PREDICTION_METHODS)} methods)...")
    pred_dir = os.path.join(api_dir, 'prediction')
    data_dir = os.path.join(api_dir, 'prediction_data')
    for i, t in enumerate(ticker_symbols):
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{total}...")
        for method in PREDICTION_METHODS:
            chart = create_prediction_chart(df, t, method)
            write_json(os.path.join(pred_dir, f'{t}_{method}.json'), {'chart': chart})
            pred_data, _ = predict_eps(df, t, method)
            write_json(os.path.join(data_dir, f'{t}_{method}.json'), pred_data)
    
    # --- Comparison charts (ticker1 x ticker2) ---
    pairs = total * total
    print(f"  comparison ({pairs} pairs)...")
    d = os.path.join(api_dir, 'comparison')
    count = 0
    for t1 in ticker_symbols:
        for t2 in ticker_symbols:
            chart = create_company_comparison_chart(df, t1, t2)
            write_json(os.path.join(d, f'{t1}_{t2}.json'), {'chart': chart})
            count += 1
        if count % 200 == 0:
            print(f"    {count}/{pairs}...")

    # --- Year-filtered charts (for Start Year / End Year dropdowns) ---
    min_year, max_year = get_year_range(df)
    year_pairs = [(ys, ye) for ys in range(min_year, max_year + 1)
                  for ye in range(ys, max_year + 1)]
    n_yp = len(year_pairs)

    print(f"  predictability + year filters ({total + 1} x {n_yp} year pairs)...")
    d = os.path.join(api_dir, 'predictability')
    for t in ticker_symbols + [None]:
        key = t or 'ALL'
        for ys, ye in year_pairs:
            chart = create_eps_predictability_chart(df, t, ys, ye)
            write_json(os.path.join(d, f'{key}_{ys}_{ye}.json'), {'chart': chart})

    print(f"  eps_surprise_returns + year filters ({n_yp} year pairs)...")
    d = os.path.join(api_dir, 'eps_surprise_returns')
    for ys, ye in year_pairs:
        chart = create_eps_surprise_returns_chart(df, ys, ye)
        write_json(os.path.join(d, f'{ys}_{ye}.json'), {'chart': chart})

    print(f"  eps_returns_trend + year filters ({total} x {n_yp} year pairs)...")
    d = os.path.join(api_dir, 'eps_returns_trend')
    for t in ticker_symbols:
        for ys, ye in year_pairs:
            chart = create_eps_vs_returns_trend_chart(df, t, ys, ye)
            write_json(os.path.join(d, f'{t}_{ys}_{ye}.json'), {'chart': chart})

    # Note: comparison year filtering is handled client-side via Plotly xaxis range

    # --- Raw ticker data for client-side prediction engine ---
    print(f"  ticker_data ({total} tickers)...")
    export_ticker_data(df, ticker_symbols)


STATIC_FETCH_SCRIPT = """<script src="static/prediction_engine.js"></script>
<script>
var _sfCache = {};
function _applyYearRange(data, ys, ye) {
    if (data.chart) {
        var c = JSON.parse(data.chart);
        var xr = [ys + '-01-01', ye + '-12-31'];
        if (c.layout.xaxis) c.layout.xaxis.range = xr;
        if (c.layout.xaxis2) c.layout.xaxis2.range = xr;
        data.chart = JSON.stringify(c);
    }
    return new Response(JSON.stringify(data), {headers: {'Content-Type': 'application/json'}});
}
var _clientMethods = {linear:1, exponential:1, moving_avg:1, analyst:1};
function staticFetch(url) {
    var parts = url.split('?');
    var path = parts[0].replace('/api/chart/', '').replace('/api/', '');
    var params = new URLSearchParams(parts[1] || '');
    var sd = params.get('start_date'), ed = params.get('end_date');
    var _h = {'Content-Type': 'application/json'};

    // Client-side prediction: when training window slider sets a date range
    if (sd && ed && window._predEngine) {
        var _tk = params.get('ticker') || 'JNJ';
        var _mt = params.get('method') || 'linear';
        if (path === 'prediction' && _clientMethods[_mt]) {
            return window._predEngine.load(_tk).then(function(d) {
                var p = window._predEngine.compute(d.quarters, _mt, 4, sd, ed);
                if (!p) return new Response('{"chart":null}', {headers: _h});
                var c = window._predEngine.buildPredChart(d.quarters, p, _tk, _mt, true);
                return new Response(JSON.stringify({chart: JSON.stringify(c)}), {headers: _h});
            });
        }
        if (path === 'surprise_analysis') {
            return window._predEngine.load(_tk).then(function(d) {
                var filt = d.quarters.filter(function(q) { return q.d >= sd && q.d <= ed; });
                if (filt.length < 2) return new Response('{"chart":null}', {headers: _h});
                var c = window._predEngine.buildSurpChart(d.quarters, filt, _tk, true);
                return new Response(JSON.stringify({chart: JSON.stringify(c)}), {headers: _h});
            });
        }
        if (path === 'prediction_data' && _clientMethods[_mt]) {
            return window._predEngine.load(_tk).then(function(d) {
                var p = window._predEngine.compute(d.quarters, _mt, 4, sd, ed);
                return new Response(JSON.stringify(p || {}), {headers: _h});
            });
        }
    }

    // Standard file path resolution
    var ys = params.get('year_start');
    var ye = params.get('year_end');
    var yearSuffix = (ys && ye) ? '_' + ys + '_' + ye : '';
    var isComparison = false;
    var filePath = 'api/';

    if (path === 'eps_history') filePath += 'eps_history/' + (params.get('ticker') || 'JNJ') + '.json';
    else if (path === 'revision_trail') filePath += 'revision_trail/' + (params.get('ticker') || 'JNJ') + '_' + (params.get('num_quarters') || '6') + '.json';
    else if (path === 'dispersion') filePath += 'dispersion/' + (params.get('ticker') || 'JNJ') + '.json';
    else if (path === 'comparison') { isComparison = true; filePath += 'comparison/' + (params.get('ticker1') || 'JNJ') + '_' + (params.get('ticker2') || 'MSFT') + '.json'; }
    else if (path === 'predictability') { var t = params.get('ticker'); filePath += 'predictability/' + (t || 'ALL') + yearSuffix + '.json'; }
    else if (path === 'prediction') filePath += 'prediction/' + (params.get('ticker') || 'JNJ') + '_' + (params.get('method') || 'linear') + '.json';
    else if (path === 'surprise_analysis') filePath += 'surprise_analysis/' + (params.get('ticker') || 'JNJ') + '.json';
    else if (path === 'prediction_data') filePath += 'prediction_data/' + (params.get('ticker') || 'JNJ') + '_' + (params.get('method') || 'linear') + '.json';
    else if (path === 'eps_surprise_returns') filePath += 'eps_surprise_returns/' + (ys && ye ? ys + '_' + ye : 'all') + '.json';
    else if (path === 'eps_returns_trend') filePath += 'eps_returns_trend/' + (params.get('ticker') || 'JNJ') + yearSuffix + '.json';
    else filePath += path + '.json';

    if (_sfCache[filePath]) {
        if (isComparison && ys && ye) return _sfCache[filePath].clone().json().then(function(d) { return _applyYearRange(d, ys, ye); });
        return Promise.resolve(_sfCache[filePath].clone());
    }
    return fetch(filePath).then(function(r) {
        _sfCache[filePath] = r.clone();
        if (isComparison && ys && ye) return r.json().then(function(d) { return _applyYearRange(d, ys, ye); });
        return r;
    }).catch(function() {
        var fallback = filePath.replace(/_[0-9]{4}_[0-9]{4}[.]json$/, '.json');
        if (fallback !== filePath) return fetch(fallback);
        return new Response('{}', {headers: {'Content-Type': 'application/json'}});
    });
}
</script>"""


def rewrite_html_for_static(html):
    """Transform Flask-rendered HTML into a static-site-compatible version"""
    # Fix static file paths (Flask generates /static/...)
    html = html.replace('/static/', 'static/')
    
    # Fix navigation links
    html = html.replace('href="/"', 'href="index.html"')
    html = html.replace("href='/'", "href='index.html'")
    html = html.replace('href="/data"', 'href="data.html"')
    html = html.replace('href="/refresh"', 'href="#"')
    
    # Remove the Refresh button from the stats bar
    html = re.sub(r'<a href="#"[^>]*class="btn[^"]*"[^>]*>Refresh</a>', '', html)
    
    # Remove noindex meta tag so the site is indexable
    html = html.replace('<meta name="robots" content="noindex">', '')
    
    # Inject the static fetch adapter before </head>
    html = html.replace('</head>', STATIC_FETCH_SCRIPT + '\n</head>')
    
    # Replace all fetch('/api/...) calls with staticFetch
    html = html.replace("fetch('/api/", "staticFetch('/api/")
    
    # Also catch fetch(url) where url is a variable holding an /api/ path
    # These appear in updatePredictability and similar functions
    html = re.sub(r'\bfetch\(url\)', 'staticFetch(url)', html)
    
    return html


def build():
    start = time.time()
    
    print("=" * 60)
    print("EPS Dashboard - Static Site Generator")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading data...")
    df = load_data()
    if df is None:
        print("ERROR: Could not load CSV data. Check DATA_FILE path in app.py")
        sys.exit(1)
    
    tickers_list = get_ticker_list(df)
    ticker_symbols = [t['ticker'] for t in tickers_list]
    print(f"  Found {len(ticker_symbols)} tickers, {len(df):,} records")
    
    # Clean output directory
    print("\n[2/5] Preparing output directory...")
    if os.path.exists(DOCS_DIR):
        shutil.rmtree(DOCS_DIR)
    os.makedirs(DOCS_DIR)
    
    # Copy static assets
    static_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    shutil.copytree(static_src, os.path.join(DOCS_DIR, 'static'))
    print(f"  Copied static/ to docs/static/")
    
    # Generate all API JSON files
    print("\n[3/5] Generating API JSON files (this may take a few minutes)...")
    generate_api_files(df, ticker_symbols)
    
    # Render HTML pages using Flask test client
    print("\n[4/5] Rendering HTML pages...")
    with app.test_client() as client:
        # Main dashboard
        response = client.get('/')
        html = response.data.decode('utf-8')
        html = rewrite_html_for_static(html)
        with open(os.path.join(DOCS_DIR, 'index.html'), 'w', encoding='utf-8') as f:
            f.write(html)
        print("  index.html ✓")
        
        # Data table page
        response = client.get('/data')
        html = response.data.decode('utf-8')
        html = rewrite_html_for_static(html)
        with open(os.path.join(DOCS_DIR, 'data.html'), 'w', encoding='utf-8') as f:
            f.write(html)
        print("  data.html ✓")
    
    # Create .nojekyll file (tells GitHub Pages to serve files as-is)
    with open(os.path.join(DOCS_DIR, '.nojekyll'), 'w') as f:
        pass
    
    # Summary
    elapsed = time.time() - start
    file_count = sum(len(files) for _, _, files in os.walk(DOCS_DIR))
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, f))
        for dirpath, _, filenames in os.walk(DOCS_DIR)
        for f in filenames
    )
    
    print(f"\n[5/5] Done!")
    print(f"  Output: {DOCS_DIR}")
    print(f"  Files:  {file_count:,}")
    print(f"  Size:   {total_size / 1024 / 1024:.1f} MB")
    print(f"  Time:   {elapsed:.0f}s")
    print(f"\nNext steps:")
    print(f"  git add docs/")
    print(f"  git commit -m 'Build static site'")
    print(f"  git push")


if __name__ == '__main__':
    build()
