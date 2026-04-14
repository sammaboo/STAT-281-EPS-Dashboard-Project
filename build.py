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
    chart, stats = create_eps_predictability_chart(df, None)
    write_json(os.path.join(d, 'ALL.json'), {'chart': chart, 'stats': stats})
    for t in ticker_symbols:
        chart, stats = create_eps_predictability_chart(df, t)
        write_json(os.path.join(d, f'{t}.json'), {'chart': chart, 'stats': stats})
    
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
    
    # --- EPS Surprise Returns (per ticker) ---
    print(f"  eps_surprise_returns ({total} tickers + ALL)...")
    d = os.path.join(api_dir, 'eps_surprise_returns')
    chart = create_eps_surprise_returns_chart(df, 'ALL')
    write_json(os.path.join(d, 'ALL.json'), {'chart': chart})
    for t in ticker_symbols:
        chart = create_eps_surprise_returns_chart(df, t)
        write_json(os.path.join(d, f'{t}.json'), {'chart': chart})
    
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
    
    # --- Comparison charts (one per ticker) ---
    print(f"  comparison ({len(ticker_symbols)} tickers)...")
    d = os.path.join(api_dir, 'comparison')
    for t in ticker_symbols:
        chart = create_company_comparison_chart(df, t)
        write_json(os.path.join(d, f'{t}.json'), {'chart': chart})

    # --- Year-filtered charts removed ---
    # Predictability, eps_surprise_returns, and eps_returns_trend year filtering
    # is now handled client-side via Plotly xaxis range (same as comparison charts)

    # --- Raw ticker data for client-side prediction engine ---
    print(f"  ticker_data ({total} tickers)...")
    export_ticker_data(df, ticker_symbols)


STATIC_FETCH_SCRIPT = """<script src="static/prediction_engine.js?v=""" + str(int(time.time())) + """"></script>
<script>
var _sfCache = {};
var _h = {'Content-Type': 'application/json'};
function _resp(data) { return new Response(JSON.stringify(data), {headers: _h}); }
function _decodeF8(arr) {
    if (Array.isArray(arr)) return arr;
    if (arr && arr.bdata) { var b = atob(arr.bdata), ab = new ArrayBuffer(b.length), v = new Uint8Array(ab); for (var i=0;i<b.length;i++) v[i]=b.charCodeAt(i); return Array.from(new Float64Array(ab)); }
    return arr || [];
}
function _decodeAny(arr) {
    if (Array.isArray(arr)) return arr;
    if (arr && arr.bdata) {
        var b = atob(arr.bdata), ab = new ArrayBuffer(b.length), v = new Uint8Array(ab);
        for (var i=0;i<b.length;i++) v[i]=b.charCodeAt(i);
        if (arr.dtype === 'f8') return Array.from(new Float64Array(ab));
        if (arr.dtype === 'i4') return Array.from(new Int32Array(ab));
        if (arr.dtype === 'i2') return Array.from(new Int16Array(ab));
        return Array.from(new Float64Array(ab));
    }
    return arr || [];
}
function _applyYearRange(data, ys, ye) {
    if (data.chart) {
        var c = JSON.parse(data.chart);
        var xr = [ys + '-01-01', ye + '-12-31'];
        if (c.layout.xaxis) c.layout.xaxis.range = xr;
        if (c.layout.xaxis2) c.layout.xaxis2.range = xr;
        data.chart = JSON.stringify(c);
    }
    return _resp(data);
}
function _filterReturnsTrend(data, ys, ye) {
    if (!data.chart) return _resp(data);
    var c = JSON.parse(data.chart), ysN = parseInt(ys), yeN = parseInt(ye);
    for (var i = 0; i < c.data.length; i++) {
        var tr = c.data[i], xs = tr.x;
        if (!Array.isArray(xs) || xs.length === 0 || typeof xs[0] !== 'string') continue;
        var nx = [], ny = [];
        var ys2 = _decodeF8(tr.y);
        for (var j = 0; j < xs.length; j++) {
            var yr = parseInt(xs[j].substring(0, 4));
            if (yr >= ysN && yr <= yeN) { nx.push(xs[j]); ny.push(ys2[j]); }
        }
        tr.x = nx; tr.y = ny;
    }
    data.chart = JSON.stringify(c);
    return _resp(data);
}
function _filterSurpriseReturns(data, ys, ye) {
    if (!data.chart) return _resp(data);
    var c = JSON.parse(data.chart), ysN = parseInt(ys), yeN = parseInt(ye);
    var fx = [], fy = [];
    for (var i = 0; i < c.data.length; i++) {
        var tr = c.data[i];
        if (tr.mode !== 'markers') continue;
        var xs = _decodeF8(tr.x), ya = _decodeF8(tr.y);
        var cd = tr.customdata;
        if (cd && cd.bdata) { cd = _decodeAny(cd); cd = cd.map(function(v){return [v];}); }
        else if (Array.isArray(cd) && cd.length > 0 && !Array.isArray(cd[0])) { cd = cd.map(function(v){return [v];}); }
        var nx = [], ny = [], ncd = [];
        for (var j = 0; j < xs.length; j++) {
            var yr = (cd && cd[j]) ? (Array.isArray(cd[j]) ? cd[j][0] : cd[j]) : null;
            if (yr !== null && yr >= ysN && yr <= yeN) { nx.push(xs[j]); ny.push(ya[j]); ncd.push(cd[j]); fx.push(xs[j]); fy.push(ya[j]); }
        }
        tr.x = nx; tr.y = ny; if (cd) tr.customdata = ncd;
    }
    if (fx.length >= 2) {
        var n = fx.length, sx = 0, sy = 0, sxy = 0, sxx = 0;
        for (var k = 0; k < n; k++) { sx += fx[k]; sy += fy[k]; sxy += fx[k]*fy[k]; sxx += fx[k]*fx[k]; }
        var sl = (n*sxy - sx*sy) / (n*sxx - sx*sx), ic = (sy - sl*sx) / n;
        var xMin = Math.min.apply(null, fx), xMax = Math.max.apply(null, fx);
        for (var i2 = 0; i2 < c.data.length; i2++) {
            if (c.data[i2].mode === 'lines') { c.data[i2].x = [xMin, xMax]; c.data[i2].y = [sl*xMin+ic, sl*xMax+ic]; break; }
        }
    }
    data.chart = JSON.stringify(c);
    return _resp(data);
}
function _filterComparison(data, ys, ye, nq) {
    if (!data.chart) return _resp(data);
    var c = JSON.parse(data.chart);
    var ysN = ys ? parseInt(ys) : null, yeN = ye ? parseInt(ye) : null;
    // Collect all unique quarter labels from data
    var allQLabels = {};
    for (var i = 0; i < c.data.length; i++) {
        var tr = c.data[i];
        if (tr.mode !== 'markers' || !tr.customdata) continue;
        var cd = tr.customdata;
        for (var j = 0; j < cd.length; j++) { allQLabels[cd[j][0]] = true; }
    }
    // If num_quarters filter, keep only the N most recent quarter labels
    var allowedQ = null;
    if (nq) {
        var qArr = Object.keys(allQLabels).sort();
        if (nq < qArr.length) { allowedQ = {}; for (var qi = qArr.length - nq; qi < qArr.length; qi++) allowedQ[qArr[qi]] = true; }
    }
    var allX = [], allY = [];
    for (var i = 0; i < c.data.length; i++) {
        var tr = c.data[i];
        if (tr.mode !== 'markers' || !tr.customdata) continue;
        var xs = _decodeF8(tr.x), ya = _decodeF8(tr.y), cd = tr.customdata;
        var nx = [], ny = [], ncd = [];
        for (var j = 0; j < xs.length; j++) {
            var ql = cd[j][0]; var yr = parseInt(ql.substring(0, 4));
            var keep = true;
            if (ysN !== null && yr < ysN) keep = false;
            if (yeN !== null && yr > yeN) keep = false;
            if (allowedQ && !allowedQ[ql]) keep = false;
            if (keep) { nx.push(xs[j]); ny.push(ya[j]); ncd.push(cd[j]); allX.push(xs[j]); allY.push(ya[j]); }
        }
        tr.x = nx; tr.y = ny; tr.customdata = ncd;
    }
    if (allX.length > 0) {
        var bins = {};
        for (var k = 0; k < allX.length; k++) { var mb = Math.round(allX[k]); if (!bins[mb]) bins[mb] = []; bins[mb].push(allY[k]); }
        var months = Object.keys(bins).map(Number).sort(function(a,b){return b-a;});
        var avgD = months.filter(function(m){return bins[m].length >= 2;}).map(function(m) {
            var v = bins[m].slice().sort(function(a,b){return a-b;}), s = v.reduce(function(a,b){return a+b;},0);
            return {m:m, mean:s/v.length, p25:v[Math.floor(v.length*0.25)], p75:v[Math.min(Math.floor(v.length*0.75),v.length-1)]};
        });
        for (var i2 = 0; i2 < c.data.length; i2++) {
            if (c.data[i2].fill === 'toself' && avgD.length > 0) {
                c.data[i2].x = avgD.map(function(d){return d.m;}).concat(avgD.slice().reverse().map(function(d){return d.m;}));
                c.data[i2].y = avgD.map(function(d){return d.p75;}).concat(avgD.slice().reverse().map(function(d){return d.p25;}));
            }
            if (c.data[i2].name === 'Avg Error' && avgD.length > 0) {
                c.data[i2].x = avgD.map(function(d){return d.m;}); c.data[i2].y = avgD.map(function(d){return d.mean;});
            }
        }
    }
    data.chart = JSON.stringify(c);
    return _resp(data);
}
function _filterPredictability(data, ys, ye) {
    if (!data.chart) return _resp(data);
    var c = JSON.parse(data.chart);
    var ysN = parseInt(ys), yeN = parseInt(ye);
    // Filter scatter trace (first trace with mode=markers)
    var scatterIdx = -1, fx = [], fy = [];
    for (var i = 0; i < c.data.length; i++) {
        if (c.data[i].mode === 'markers') { scatterIdx = i; break; }
    }
    if (scatterIdx >= 0) {
        var tr = c.data[scatterIdx];
        var xs = _decodeF8(tr.x), ysArr = _decodeF8(tr.y);
        var cd = tr.customdata;
        if (cd && cd.bdata) { cd = _decodeAny(cd); cd = cd.map(function(v){return [v];}); }
        else if (Array.isArray(cd) && cd.length > 0 && !Array.isArray(cd[0])) { cd = cd.map(function(v){return [v];}); }
        var nx = [], ny = [], ncd = [];
        for (var j = 0; j < xs.length; j++) {
            var yr = (cd && cd[j]) ? (Array.isArray(cd[j]) ? cd[j][0] : cd[j]) : null;
            if (yr !== null && yr >= ysN && yr <= yeN) { nx.push(xs[j]); ny.push(ysArr[j]); ncd.push(cd ? cd[j] : null); fx.push(xs[j]); fy.push(ysArr[j]); }
        }
        tr.x = nx; tr.y = ny; if (cd) tr.customdata = ncd;
    }
    // Recompute OLS trendline (second trace, mode=lines)
    if (fx.length >= 2) {
        var n = fx.length, sx = 0, sy = 0, sxy = 0, sxx = 0;
        for (var k = 0; k < n; k++) { sx += fx[k]; sy += fy[k]; sxy += fx[k]*fy[k]; sxx += fx[k]*fx[k]; }
        var slope = (n*sxy - sx*sy) / (n*sxx - sx*sx);
        var intercept = (sy - slope*sx) / n;
        var xMin = Math.min.apply(null, fx), xMax = Math.max.apply(null, fx);
        for (var i2 = 0; i2 < c.data.length; i2++) {
            if (c.data[i2].mode === 'lines') { c.data[i2].x = [xMin, xMax]; c.data[i2].y = [slope*xMin+intercept, slope*xMax+intercept]; break; }
        }
        // Recompute stats
        var meanY = sy/n, ssRes = 0, ssTot = 0;
        for (var m = 0; m < n; m++) { var pred = slope*fx[m]+intercept; ssRes += (fy[m]-pred)*(fy[m]-pred); ssTot += (fy[m]-meanY)*(fy[m]-meanY); }
        var r2 = ssTot > 0 ? Math.round((1 - ssRes/ssTot)*1000)/1000 : 0;
        var stdY = Math.sqrt(ssTot/n);
        var tickers = {};
        if (scatterIdx >= 0 && c.data[scatterIdx].customdata) {
            // count unique tickers from hovertemplate or just use 1 if single-ticker
        }
        data.stats = { r_squared: r2, slope: Math.round(slope*1000)/1000, mean_eps: Math.round(meanY*100)/100, std_eps: Math.round(stdY*100)/100, n_obs: n, n_companies: data.stats ? data.stats.n_companies : 1 };
    } else {
        data.stats = null;
    }
    data.chart = JSON.stringify(c);
    return _resp(data);
}
var _clientMethods = {linear:1, exponential:1, moving_avg:1, analyst:1, holt_winters:1, sarima:1};
function staticFetch(url) {
    var parts = url.split('?');
    var path = parts[0].replace('/api/chart/', '').replace('/api/', '');
    var params = new URLSearchParams(parts[1] || '');
    var sd = params.get('start_date'), ed = params.get('end_date');
    var _h = {'Content-Type': 'application/json'};

    // Client-side prediction: when training window slider sets a date range OR surprise method changed
    var _sm = params.get('surprise_method') || 'seasonal_avg';
    var _needClientCompute = (sd && ed) || (_sm !== 'seasonal_avg');
    if (_needClientCompute && window._predEngine) {
        var _tk = params.get('ticker') || 'JNJ';
        var _mt = params.get('method') || 'linear';
        var _isSub = !!(sd && ed);
        if (path === 'prediction' && _clientMethods[_mt]) {
            return window._predEngine.load(_tk).then(function(d) {
                var p = _isSub ? window._predEngine.compute(d.quarters, _mt, 4, sd, ed) : window._predEngine.compute(d.quarters, _mt, 4);
                if (!p) return new Response('{"chart":null}', {headers: _h});
                var c = window._predEngine.buildPredChart(d.quarters, p, _tk, _mt, _isSub, _sm);
                return new Response(JSON.stringify({chart: JSON.stringify(c)}), {headers: _h});
            });
        }
        if (path === 'surprise_analysis') {
            return window._predEngine.load(_tk).then(function(d) {
                var filt = _isSub ? d.quarters.filter(function(q) { return q.d >= sd && q.d <= ed; }) : d.quarters;
                if (filt.length < 2) return new Response('{"chart":null}', {headers: _h});
                var c = window._predEngine.buildSurpChart(d.quarters, filt, _tk, _isSub);
                return new Response(JSON.stringify({chart: JSON.stringify(c)}), {headers: _h});
            });
        }
        if (path === 'prediction_data' && _clientMethods[_mt]) {
            return window._predEngine.load(_tk).then(function(d) {
                var p = _isSub ? window._predEngine.compute(d.quarters, _mt, 4, sd, ed) : window._predEngine.compute(d.quarters, _mt, 4);
                return new Response(JSON.stringify(p || {}), {headers: _h});
            });
        }
    }

    // Standard file path resolution
    var ys = params.get('year_start');
    var ye = params.get('year_end');
    var yearSuffix = (ys && ye) ? '_' + ys + '_' + ye : '';
    var isPredictability = false;
    var _yearFilter = null;
    var filePath = 'api/';

    if (path === 'eps_history') { _yearFilter = '_applyYearRange'; filePath += 'eps_history/' + (params.get('ticker') || 'JNJ') + '.json'; }
    else if (path === 'revision_trail') { _yearFilter = '_applyYearRange'; filePath += 'revision_trail/' + (params.get('ticker') || 'JNJ') + '_' + (params.get('num_quarters') || '6') + '.json'; }
    else if (path === 'dispersion') { _yearFilter = '_applyYearRange'; filePath += 'dispersion/' + (params.get('ticker') || 'JNJ') + '.json'; }
    else if (path === 'comparison') { filePath += 'comparison/' + (params.get('ticker') || 'JNJ') + '.json'; }
    else if (path === 'predictability') { isPredictability = true; var t = params.get('ticker'); filePath += 'predictability/' + (t || 'ALL') + '.json'; }
    else if (path === 'prediction') filePath += 'prediction/' + (params.get('ticker') || 'JNJ') + '_' + (params.get('method') || 'linear') + '.json';
    else if (path === 'surprise_analysis') filePath += 'surprise_analysis/' + (params.get('ticker') || 'JNJ') + '.json';
    else if (path === 'prediction_data') filePath += 'prediction_data/' + (params.get('ticker') || 'JNJ') + '_' + (params.get('method') || 'linear') + '.json';
    else if (path === 'eps_surprise_returns') { _yearFilter = '_filterSurpriseReturns'; filePath += 'eps_surprise_returns/' + (params.get('ticker') || 'JNJ') + '.json'; }
    else if (path === 'eps_returns_trend') { _yearFilter = '_filterReturnsTrend'; filePath += 'eps_returns_trend/' + (params.get('ticker') || 'JNJ') + '.json'; }
    else filePath += path + '.json';

    var _filters = {_applyYearRange: _applyYearRange, _filterComparison: _filterComparison, _filterSurpriseReturns: _filterSurpriseReturns, _filterReturnsTrend: _filterReturnsTrend};
    var _isComparison = (path === 'comparison');
    var _compNQ = _isComparison ? params.get('num_quarters') : null;
    var _needsCompFilter = _isComparison && ((ys && ye) || _compNQ);

    if (_sfCache[filePath]) {
        if (isPredictability && ys && ye) return _sfCache[filePath].clone().json().then(function(d) { return _filterPredictability(JSON.parse(JSON.stringify(d)), ys, ye); });
        if (_needsCompFilter) return _sfCache[filePath].clone().json().then(function(d) { return _filterComparison(JSON.parse(JSON.stringify(d)), ys, ye, _compNQ ? parseInt(_compNQ) : null); });
        if (_yearFilter && ys && ye) return _sfCache[filePath].clone().json().then(function(d) { return _filters[_yearFilter](JSON.parse(JSON.stringify(d)), ys, ye); });
        return Promise.resolve(_sfCache[filePath].clone());
    }
    return fetch(filePath).then(function(r) {
        _sfCache[filePath] = r.clone();
        if (isPredictability && ys && ye) return r.json().then(function(d) { return _filterPredictability(JSON.parse(JSON.stringify(d)), ys, ye); });
        if (_needsCompFilter) return r.json().then(function(d) { return _filterComparison(JSON.parse(JSON.stringify(d)), ys, ye, _compNQ ? parseInt(_compNQ) : null); });
        if (_yearFilter && ys && ye) return r.json().then(function(d) { return _filters[_yearFilter](JSON.parse(JSON.stringify(d)), ys, ye); });
        return r;
    }).catch(function() {
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
