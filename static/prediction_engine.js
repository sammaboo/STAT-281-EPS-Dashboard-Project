/**
 * Client-side EPS Prediction Engine for Static Site
 * Enables the training window slider on GitHub Pages
 * by computing predictions directly in the browser.
 *
 * Supported methods: linear, exponential, moving_avg, analyst
 * (holt_winters and sarima fall back to pre-generated all-time data)
 */
(function() {
    'use strict';

    var BG = '#161b22', GRID = '#30363d',
        TEXT = '#c9d1d9', MUTED = '#8b949e',
        GREEN = '#3fb950', RED = '#f85149', BLUE = '#58a6ff', ORANGE = '#f0883e';

    var _cache = {};

    function loadData(ticker) {
        if (_cache[ticker]) return Promise.resolve(_cache[ticker]);
        return fetch('api/ticker_data/' + ticker + '.json')
            .then(function(r) { return r.json(); })
            .then(function(d) { _cache[ticker] = d; return d; });
    }

    // ---- Math helpers ----
    function sum(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s; }
    function avg(a) { return a.length ? sum(a) / a.length : 0; }
    function stdev(a) {
        if (a.length < 2) return 0;
        var m = avg(a), s = 0;
        for (var i = 0; i < a.length; i++) s += (a[i] - m) * (a[i] - m);
        return Math.sqrt(s / (a.length - 1));
    }
    function linreg(x, y) {
        var n = x.length;
        if (n < 2) return { s: 0, i: y[0] || 0, r2: 0 };
        var sx = sum(x), sy = sum(y), sxy = 0, sx2 = 0;
        for (var j = 0; j < n; j++) { sxy += x[j] * y[j]; sx2 += x[j] * x[j]; }
        var d = n * sx2 - sx * sx;
        if (!d) return { s: 0, i: sy / n, r2: 0 };
        var slope = (n * sxy - sx * sy) / d, inter = (sy - slope * sx) / n;
        var ym = sy / n, tot = 0, res = 0;
        for (var j = 0; j < n; j++) { tot += (y[j] - ym) * (y[j] - ym); var p = inter + slope * x[j]; res += (y[j] - p) * (y[j] - p); }
        return { s: slope, i: inter, r2: tot ? 1 - res / tot : 0 };
    }
    function rolling(arr, w) {
        var r = [];
        for (var i = 0; i < arr.length; i++) {
            var s = Math.max(0, i - w + 1);
            r.push(avg(arr.slice(s, i + 1)));
        }
        return r;
    }

    // ---- Future date generation ----
    function futureDates(last, n) {
        var dates = [], d = new Date(last + 'T00:00:00');
        for (var i = 0; i < n; i++) {
            d = new Date(d); d.setMonth(d.getMonth() + 3);
            var m = d.getMonth();
            if (m < 3) { d.setMonth(2); d.setDate(31); }
            else if (m < 6) { d.setMonth(5); d.setDate(30); }
            else if (m < 9) { d.setMonth(8); d.setDate(30); }
            else { d.setMonth(11); d.setDate(31); }
            dates.push(d.toISOString().slice(0, 10));
        }
        return dates;
    }

    // ---- Prediction methods ----
    function predLinear(eps, per) {
        var x = []; for (var i = 0; i < eps.length; i++) x.push(i);
        var r = linreg(x, eps), pred = [];
        for (var i = 0; i < per; i++) pred.push(r.i + r.s * (eps.length + i));
        return pred;
    }
    function predExp(eps, per) {
        var a = 0.3, sm = [eps[0]];
        for (var i = 1; i < eps.length; i++) sm.push(a * eps[i] + (1 - a) * sm[sm.length - 1]);
        var w = Math.min(4, eps.length), diffs = [];
        for (var i = eps.length - w; i < eps.length - 1; i++) if (i >= 0) diffs.push(eps[i + 1] - eps[i]);
        var trend = diffs.length ? avg(diffs) : 0, pred = [];
        for (var i = 0; i < per; i++) pred.push(sm[sm.length - 1] + trend * (i + 1));
        return pred;
    }
    function predMA(eps, per) {
        var w = Math.min(4, eps.length), ma = avg(eps.slice(-w)), diffs = [];
        for (var i = eps.length - w; i < eps.length - 1; i++) if (i >= 0) diffs.push(eps[i + 1] - eps[i]);
        var trend = diffs.length ? avg(diffs) : 0, pred = [];
        for (var i = 0; i < per; i++) pred.push(ma + trend * (i + 1));
        return pred;
    }
    function predAnalyst(eps, est, per) {
        var surps = [];
        for (var i = 0; i < eps.length; i++) if (est[i] && est[i] !== 0) surps.push(eps[i] - est[i]);
        var avgS = surps.length ? avg(surps) : 0, last = est[est.length - 1] || eps[eps.length - 1], pred = [];
        for (var i = 0; i < per; i++) pred.push(last + avgS);
        return pred;
    }

    // ---- Holt-Winters (Triple Exponential Smoothing) ----
    function hwFit(eps, sp, alpha, beta, gamma) {
        var n = eps.length;
        // Initialize level, trend, seasonal
        var l = avg(eps.slice(0, sp));
        var b = 0;
        if (sp < n) b = (avg(eps.slice(sp, Math.min(2 * sp, n))) - avg(eps.slice(0, sp))) / sp;
        var s = [];
        for (var i = 0; i < sp; i++) s.push(eps[i] - l);
        // Run through data
        var levels = [l], trends = [b], seasons = s.slice();
        var sse = 0;
        for (var t = sp; t < n; t++) {
            var si = seasons[t % sp];
            var lNew = alpha * (eps[t] - si) + (1 - alpha) * (l + b);
            var bNew = beta * (lNew - l) + (1 - beta) * b;
            var sNew = gamma * (eps[t] - lNew) + (1 - gamma) * si;
            var fitted = l + b + si;
            sse += (eps[t] - fitted) * (eps[t] - fitted);
            l = lNew; b = bNew;
            seasons[t % sp] = sNew;
            levels.push(l); trends.push(b);
        }
        return { l: l, b: b, s: seasons, sse: sse };
    }
    function predHoltWinters(eps, per) {
        var n = eps.length;
        var sp = Math.min(4, Math.floor(n / 2));
        // If not enough data for seasonality, use double exponential (trend only)
        if (n < 2 * sp || sp < 2) {
            // Double exponential smoothing (Holt's method)
            var alpha = 0.3, beta = 0.1;
            var l = eps[0], b = n > 1 ? eps[1] - eps[0] : 0;
            for (var i = 1; i < n; i++) {
                var lNew = alpha * eps[i] + (1 - alpha) * (l + b);
                var bNew = beta * (lNew - l) + (1 - beta) * b;
                l = lNew; b = bNew;
            }
            var pred = [];
            for (var i = 0; i < per; i++) pred.push(l + b * (i + 1));
            return pred;
        }
        // Grid search for best alpha, beta, gamma
        var best = null, bestSSE = Infinity;
        var vals = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9];
        for (var ai = 0; ai < vals.length; ai++) {
            for (var bi = 0; bi < vals.length; bi++) {
                for (var gi = 0; gi < vals.length; gi++) {
                    var res = hwFit(eps, sp, vals[ai], vals[bi], vals[gi]);
                    if (res.sse < bestSSE) { bestSSE = res.sse; best = res; }
                }
            }
        }
        var pred = [];
        for (var i = 0; i < per; i++) {
            pred.push(best.l + best.b * (i + 1) + best.s[(n + i) % sp]);
        }
        return pred;
    }

    // ---- SARIMA approximation ----
    // Simplified: difference(d=1), seasonal difference if enough data, then AR(1) forecast
    function predSarima(eps, per) {
        var n = eps.length;
        if (n < 3) {
            var rt = n > 1 ? eps[n - 1] - eps[n - 2] : 0;
            var p = []; for (var i = 0; i < per; i++) p.push(eps[n - 1] + rt * (i + 1)); return p;
        }
        var sp = 4; // seasonal period (quarterly)
        var useSeasonal = n >= 8;

        // First difference
        var diff = [];
        for (var i = 1; i < n; i++) diff.push(eps[i] - eps[i - 1]);

        // Seasonal difference of the first-differenced series
        var sdiff = diff;
        if (useSeasonal && diff.length > sp) {
            sdiff = [];
            for (var i = sp; i < diff.length; i++) sdiff.push(diff[i] - diff[i - sp]);
        }

        // AR(1) coefficient from autocorrelation of sdiff
        var phi = 0;
        if (sdiff.length > 1) {
            var m = avg(sdiff), c0 = 0, c1 = 0;
            for (var i = 0; i < sdiff.length; i++) c0 += (sdiff[i] - m) * (sdiff[i] - m);
            for (var i = 1; i < sdiff.length; i++) c1 += (sdiff[i] - m) * (sdiff[i - 1] - m);
            phi = c0 > 0 ? c1 / c0 : 0;
            phi = Math.max(-0.95, Math.min(0.95, phi)); // clamp for stability
        }

        // Forecast by reversing differences
        var pred = [];
        var lastDiff = diff[diff.length - 1];
        var lastVal = eps[n - 1];
        for (var i = 0; i < per; i++) {
            // AR(1) step on differenced series
            var nextDiff = phi * lastDiff + avg(diff) * (1 - phi);
            // Add seasonal component if available
            if (useSeasonal && n > sp) {
                var sIdx = n - sp + i;
                if (sIdx >= 0 && sIdx < n - 1) {
                    nextDiff += (eps[sIdx + 1] - eps[sIdx]) - avg(diff);
                }
            }
            lastVal = lastVal + nextDiff;
            lastDiff = nextDiff;
            pred.push(lastVal);
        }
        return pred;
    }

    // ---- Backtest confidence ----
    function backtest(eps, est, method) {
        var n = eps.length, ho = Math.min(4, Math.floor(n / 2));
        if (n <= ho + 2) return { conf: 0.5, mape: null };
        var tr = eps.slice(0, n - ho), te = est.slice(0, n - ho), act = eps.slice(n - ho), bt;
        if (method === 'linear') bt = predLinear(tr, ho);
        else if (method === 'exponential') bt = predExp(tr, ho);
        else if (method === 'moving_avg') bt = predMA(tr, ho);
        else if (method === 'analyst') bt = predAnalyst(tr, te, ho);
        else if (method === 'holt_winters') bt = predHoltWinters(tr, ho);
        else if (method === 'sarima') bt = predSarima(tr, ho);
        else bt = Array(ho).fill(tr[tr.length - 1]);
        var errs = [];
        for (var i = 0; i < ho; i++) if (act[i] !== 0) errs.push(Math.abs(bt[i] - act[i]) / Math.abs(act[i]));
        if (!errs.length) return { conf: 0.5, mape: null };
        var ae = avg(errs);
        return { conf: Math.max(0, Math.min(1, 1 - ae)), mape: Math.round(ae * 1000) / 10 };
    }

    // ---- Compute full prediction ----
    function compute(quarters, method, periods, startDate, endDate) {
        var filt = quarters;
        if (startDate && endDate) filt = quarters.filter(function(q) { return q.d >= startDate && q.d <= endDate; });
        if (filt.length < 2) return null;
        var dates = filt.map(function(q) { return q.d; }),
            eps = filt.map(function(q) { return q.a; }),
            est = filt.map(function(q) { return q.e; });
        var surps = [];
        for (var i = 0; i < eps.length; i++) {
            var e = est[i]; surps.push(e && Math.abs(e) > 0 ? ((eps[i] - e) / Math.abs(e)) * 100 : 0);
        }
        var pred;
        if (method === 'linear') pred = predLinear(eps, periods);
        else if (method === 'exponential') pred = predExp(eps, periods);
        else if (method === 'moving_avg') pred = predMA(eps, periods);
        else if (method === 'analyst') pred = predAnalyst(eps, est, periods);
        else if (method === 'holt_winters') pred = predHoltWinters(eps, periods);
        else if (method === 'sarima') pred = predSarima(eps, periods);
        else return null;
        var fd = futureDates(dates[dates.length - 1], periods);
        var bt = backtest(eps, est, method);
        return {
            historical_dates: dates, historical_eps: eps, historical_estimates: est,
            historical_surprises: surps, future_dates: fd,
            predicted_eps: pred.map(function(v) { return Math.round(v * 100) / 100; }),
            avg_surprise_pct: Math.round(avg(surps) * 100) / 100,
            std_surprise_pct: Math.round(stdev(surps) * 100) / 100,
            confidence: Math.round(bt.conf * 100) / 100,
            backtest_mape: bt.mape, method: method
        };
    }

    // ---- Chart axis helper ----
    function ax(extra) {
        var b = { gridcolor: GRID, linecolor: GRID, tickfont: { color: MUTED }, zeroline: false };
        for (var k in extra) b[k] = extra[k]; return b;
    }

    // ---- Build prediction chart (2-row subplot) ----
    function buildPredChart(allQ, pred, ticker, method, isSub, surpriseMethod) {
        if (!surpriseMethod) surpriseMethod = 'seasonal_avg';
        var traces = [];

        // Faded full history background (when sub-range)
        if (isSub) {
            var ad = allQ.map(function(q) { return q.d; }),
                ae = allQ.map(function(q) { return q.a; }),
                aest = allQ.map(function(q) { return q.e; });
            traces.push({ x: ad, y: ae, name: 'Full History (Actual)', line: { color: GREEN, width: 1, dash: 'dot' }, mode: 'lines+markers', marker: { size: 5, opacity: 0.3 }, opacity: 0.35 });
            traces.push({ x: ad, y: aest, name: 'Full History (Estimate)', line: { color: BLUE, width: 1, dash: 'dot' }, mode: 'lines+markers', marker: { size: 4, opacity: 0.3 }, opacity: 0.35 });
        }

        // Historical actual + estimate
        traces.push({ x: pred.historical_dates, y: pred.historical_eps, name: 'Actual EPS' + (isSub ? ' (Model Window)' : ''), line: { color: GREEN, width: 2 }, mode: 'lines+markers', marker: { size: 8 } });
        traces.push({ x: pred.historical_dates, y: pred.historical_estimates, name: 'Analyst Estimate' + (isSub ? ' (Model Window)' : ''), line: { color: BLUE, width: 2, dash: 'dot' }, mode: 'lines+markers', marker: { size: 6 } });

        // Predicted EPS
        traces.push({ x: pred.future_dates, y: pred.predicted_eps, name: 'Predicted EPS', line: { color: ORANGE, width: 3 }, mode: 'lines+markers', marker: { size: 10, symbol: 'diamond' } });

        // Confidence band
        var sd = stdev(pred.historical_eps), c = pred.confidence;
        var up = pred.predicted_eps.map(function(e) { return e + sd * (1 - c); }),
            lo = pred.predicted_eps.map(function(e) { return e - sd * (1 - c); });
        traces.push({ x: pred.future_dates.concat(pred.future_dates.slice().reverse()), y: up.concat(lo.slice().reverse()), fill: 'toself', fillcolor: 'rgba(240,136,62,0.2)', line: { color: 'rgba(0,0,0,0)' }, name: 'Confidence Band', showlegend: true });

        // Bottom chart: surprise bars
        if (isSub) {
            var as = allQ.map(function(q) { return q.e && Math.abs(q.e) > 0 ? ((q.a - q.e) / Math.abs(q.e)) * 100 : 0; });
            var ac = as.map(function(s) { return s >= 0 ? 'rgba(63,185,80,0.3)' : 'rgba(248,81,73,0.3)'; });
            traces.push({ x: allQ.map(function(q) { return q.d; }), y: as, type: 'bar', marker: { color: ac }, showlegend: false, opacity: 0.3, xaxis: 'x2', yaxis: 'y2' });
        }
        var cs = pred.historical_surprises.map(function(s) { return s >= 0 ? GREEN : RED; });
        traces.push({ x: pred.historical_dates, y: pred.historical_surprises, type: 'bar', name: 'Surprise %', marker: { color: cs }, showlegend: false, xaxis: 'x2', yaxis: 'y2' });

        // Predicted surprise bars
        var predSurp;
        if (surpriseMethod === 'overall_avg') {
            var totalS = 0;
            for (var oi = 0; oi < pred.historical_surprises.length; oi++) totalS += pred.historical_surprises[oi];
            var avgS = pred.historical_surprises.length ? totalS / pred.historical_surprises.length : 0;
            predSurp = pred.future_dates.map(function() { return avgS; });
        } else if (surpriseMethod === 'last_year') {
            var lastByQ = {};
            for (var li = 0; li < pred.historical_dates.length; li++) {
                var lm = new Date(pred.historical_dates[li] + 'T00:00:00').getMonth();
                var lq = lm < 3 ? 1 : lm < 6 ? 2 : lm < 9 ? 3 : 4;
                lastByQ[lq] = pred.historical_surprises[li];
            }
            predSurp = pred.future_dates.map(function(fd) {
                var fm = new Date(fd + 'T00:00:00').getMonth();
                var fq = fm < 3 ? 1 : fm < 6 ? 2 : fm < 9 ? 3 : 4;
                return lastByQ[fq] !== undefined ? lastByQ[fq] : pred.avg_surprise_pct;
            });
        } else if (surpriseMethod === 'trend') {
            var n = pred.historical_surprises.length;
            if (n >= 2) {
                var sx = 0, sy = 0, sxy = 0, sxx = 0;
                for (var ti = 0; ti < n; ti++) { sx += ti; sy += pred.historical_surprises[ti]; sxy += ti * pred.historical_surprises[ti]; sxx += ti * ti; }
                var slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
                var intercept = (sy - slope * sx) / n;
                predSurp = pred.future_dates.map(function(fd, fi) { return slope * (n + fi) + intercept; });
            } else {
                predSurp = pred.future_dates.map(function() { return pred.avg_surprise_pct; });
            }
        } else { // seasonal_avg (default)
            var qAvg = {}, qCnt = {};
            for (var si = 0; si < pred.historical_dates.length; si++) {
                var qn = new Date(pred.historical_dates[si] + 'T00:00:00').getMonth();
                var qk = qn < 3 ? 1 : qn < 6 ? 2 : qn < 9 ? 3 : 4;
                if (!qAvg[qk]) { qAvg[qk] = 0; qCnt[qk] = 0; }
                qAvg[qk] += pred.historical_surprises[si]; qCnt[qk]++;
            }
            for (var qk2 in qAvg) qAvg[qk2] /= qCnt[qk2];
            predSurp = pred.future_dates.map(function(fd) {
                var fq = new Date(fd + 'T00:00:00').getMonth();
                var fqn = fq < 3 ? 1 : fq < 6 ? 2 : fq < 9 ? 3 : 4;
                return qAvg[fqn] !== undefined ? qAvg[fqn] : pred.avg_surprise_pct;
            });
        }
        traces.push({ x: pred.future_dates, y: predSurp, type: 'bar', name: 'Predicted Surprise', marker: { color: predSurp.map(function() { return 'rgba(240,136,62,0.7)'; }), line: { color: ORANGE, width: 1.5 } }, opacity: 0.6, showlegend: true, xaxis: 'x2', yaxis: 'y2' });

        var mt = method.charAt(0).toUpperCase() + method.slice(1).replace('_', ' ');
        var layout = {
            paper_bgcolor: BG, plot_bgcolor: BG, font: { color: TEXT, family: 'Inter, sans-serif' },
            height: 550, showlegend: true, margin: { t: 40, b: 40, l: 60, r: 20 },
            annotations: [
                { text: ticker + ' EPS Forecast (' + mt + ' Method)', x: 0.5, y: 1.02, xref: 'paper', yref: 'paper', showarrow: false, font: { color: TEXT, size: 13 } },
                { text: ticker + ' Surprise Pattern (Historical + Predicted)', x: 0.5, y: 0.37, xref: 'paper', yref: 'paper', showarrow: false, font: { color: TEXT, size: 12 } }
            ],
            xaxis: ax({ domain: [0, 1], anchor: 'y' }),
            yaxis: ax({ domain: [0.45, 1], anchor: 'x', title: { text: 'EPS ($)', font: { color: MUTED } } }),
            xaxis2: ax({ domain: [0, 1], anchor: 'y2', matches: 'x' }),
            yaxis2: ax({ domain: [0, 0.33], anchor: 'x2', title: { text: 'Surprise (%)', font: { color: MUTED } } }),
            shapes: [
                { type: 'line', x0: 0, x1: 1, xref: 'paper', y0: pred.avg_surprise_pct, y1: pred.avg_surprise_pct, yref: 'y2', line: { color: MUTED, dash: 'dash', width: 1 } },
                { type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 0, y1: 0, yref: 'y2', line: { color: GRID, width: 1 } }
            ]
        };
        if (isSub && pred.historical_dates.length) {
            layout.shapes.push({
                type: 'rect', x0: pred.historical_dates[0], x1: pred.historical_dates[pred.historical_dates.length - 1],
                y0: 0, y1: 1, yref: 'paper', xref: 'x',
                fillcolor: 'rgba(88,166,255,0.06)', line: { color: 'rgba(88,166,255,0.25)', width: 1, dash: 'dot' }
            });
            layout.annotations.push({ x: pred.historical_dates[0], y: 1, yref: 'paper', xref: 'x', text: 'Training Window', showarrow: false, font: { color: BLUE, size: 10 }, xanchor: 'left', yanchor: 'top' });
        }
        return { data: traces, layout: layout };
    }

    // ---- Build surprise analysis chart (1-row, 2-col subplot) ----
    function buildSurpChart(allQ, filtQ, ticker, isSub) {
        var filt = filtQ.map(function(q) {
            var s = q.e && Math.abs(q.e) > 0 ? ((q.a - q.e) / Math.abs(q.e)) * 100 : 0;
            var m = new Date(q.d + 'T00:00:00').getMonth();
            return { d: q.d, s: s, q: m < 3 ? 1 : m < 6 ? 2 : m < 9 ? 3 : 4 };
        });

        // Quarter averages
        var qd = { 1: [], 2: [], 3: [], 4: [] };
        filt.forEach(function(f) { qd[f.q].push(f.s); });
        var ql = [], qa = [], qs = [], qc = [];
        for (var q = 1; q <= 4; q++) {
            if (qd[q].length) { ql.push('Q' + q); qa.push(avg(qd[q])); qs.push(stdev(qd[q])); qc.push(avg(qd[q]) >= 0 ? GREEN : RED); }
        }

        var traces = [];
        traces.push({ x: ql, y: qa, type: 'bar', error_y: { type: 'data', array: qs, visible: true }, marker: { color: qc }, name: 'Avg Surprise', xaxis: 'x', yaxis: 'y' });

        // Right panel: faded full history when sub-range
        if (isSub) {
            var as = allQ.map(function(q) { return q.e && Math.abs(q.e) > 0 ? ((q.a - q.e) / Math.abs(q.e)) * 100 : 0; });
            var ad = allQ.map(function(q) { return q.d; });
            traces.push({ x: ad, y: as, name: 'Full History', mode: 'markers', marker: { size: 6, color: BLUE, opacity: 0.25 }, showlegend: false, xaxis: 'x2', yaxis: 'y2' });
            traces.push({ x: ad, y: rolling(as, 4), name: 'Full Rolling Avg', line: { color: ORANGE, width: 1, dash: 'dot' }, opacity: 0.3, showlegend: false, xaxis: 'x2', yaxis: 'y2' });
        }

        var sd = filt.map(function(f) { return f.d; }), ss = filt.map(function(f) { return f.s; });
        traces.push({ x: sd, y: ss, name: 'Actual Surprise', mode: 'markers', marker: { size: 8, color: BLUE }, xaxis: 'x2', yaxis: 'y2' });
        traces.push({ x: sd, y: rolling(ss, 4), name: '4-Quarter Rolling Avg', line: { color: ORANGE, width: 2 }, xaxis: 'x2', yaxis: 'y2' });

        var layout = {
            paper_bgcolor: BG, plot_bgcolor: BG, font: { color: TEXT, family: 'Inter, sans-serif' },
            height: 350, showlegend: true, margin: { t: 40, b: 40, l: 60, r: 20 },
            annotations: [
                { text: 'Surprise by Quarter (Seasonality)', x: 0.22, y: 1.08, xref: 'paper', yref: 'paper', showarrow: false, font: { color: TEXT, size: 12 } },
                { text: 'Surprise Trend Over Time', x: 0.78, y: 1.08, xref: 'paper', yref: 'paper', showarrow: false, font: { color: TEXT, size: 12 } }
            ],
            xaxis: ax({ domain: [0, 0.44], anchor: 'y' }),
            yaxis: ax({ domain: [0, 1], anchor: 'x', title: { text: 'Surprise (%)', font: { color: MUTED } } }),
            xaxis2: ax({ domain: [0.56, 1], anchor: 'y2' }),
            yaxis2: ax({ domain: [0, 1], anchor: 'x2', title: { text: 'Surprise (%)', font: { color: MUTED } } }),
            shapes: [
                { type: 'line', x0: 0, x1: 0.44, xref: 'paper', y0: 0, y1: 0, yref: 'y', line: { color: GRID, width: 1 } },
                { type: 'line', x0: 0.56, x1: 1, xref: 'paper', y0: 0, y1: 0, yref: 'y2', line: { color: GRID, width: 1 } }
            ]
        };
        return { data: traces, layout: layout };
    }

    // Expose API for staticFetch integration
    window._predEngine = {
        load: loadData,
        compute: compute,
        buildPredChart: buildPredChart,
        buildSurpChart: buildSurpChart
    };
})();
