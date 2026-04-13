"""
WRDS Financial Data Dashboard
Flask web application for visualizing IBES/Compustat data with data refresh capability
"""
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import subprocess
import sys
import os
from datetime import datetime
import threading

app = Flask(__name__)
app.jinja_env.globals['now'] = datetime.now

# Path to data file and pull script
DATA_FILE = r'c:\Users\sam\Documents\STAT_281\ibes_compustat_merged.csv'
PULL_SCRIPT = r'c:\Users\sam\Documents\wrds_pull_stat_281.py'

# Global status for refresh operation
refresh_status = {'running': False, 'message': '', 'success': None}

# Dark theme layout for Plotly charts
DARK_LAYOUT = {
    'paper_bgcolor': '#161b22',
    'plot_bgcolor': '#161b22',
    'font': {'color': '#c9d1d9', 'family': 'Inter, sans-serif'},
    'title_font': {'color': '#c9d1d9', 'size': 14},
    'xaxis': {
        'gridcolor': '#30363d',
        'linecolor': '#30363d',
        'tickfont': {'color': '#8b949e'},
        'title': {'font': {'color': '#8b949e'}}
    },
    'yaxis': {
        'gridcolor': '#30363d',
        'linecolor': '#30363d',
        'tickfont': {'color': '#8b949e'},
        'title': {'font': {'color': '#8b949e'}}
    },
    'legend': {'font': {'color': '#c9d1d9'}},
    'coloraxis': {'colorbar': {'tickfont': {'color': '#8b949e'}}}
}

def load_data():
    """Load the merged IBES/Compustat data"""
    try:
        df = pd.read_csv(DATA_FILE)
        if 'datadate' in df.columns:
            df['datadate'] = pd.to_datetime(df['datadate'])
        df['statpers'] = pd.to_datetime(df['statpers'])
        df['fpedats'] = pd.to_datetime(df['fpedats'])
        return df
    except FileNotFoundError:
        return None

def get_summary_stats(df):
    """Calculate summary statistics"""
    if df is None:
        return {}
    
    unique_tickers = df['ticker'].nunique()
    date_range = f"{df['datadate'].min().strftime('%Y-%m-%d')} to {df['datadate'].max().strftime('%Y-%m-%d')}"
    total_records = len(df)
    
    return {
        'unique_tickers': unique_tickers,
        'date_range': date_range,
        'total_records': total_records
    }

def create_eps_history_chart(df, featured_ticker='JNJ'):
    """Create historical EPS estimates vs actuals chart for a single company"""
    if df is None:
        return None
    
    ticker = featured_ticker
    if not ticker or ticker not in df['ticker'].values:
        ticker = df.groupby('ticker')['numest'].max().idxmax()
    
    # Get company name if available
    company_name = ticker
    if 'company_name' in df.columns:
        names = df.loc[df['ticker'] == ticker, 'company_name'].dropna()
        if len(names) > 0:
            company_name = f"{ticker} - {names.iloc[0]}"
    
    eps_data = df[df['ticker'] == ticker].copy()
    eps_data = eps_data.dropna(subset=['meanest', 'actual'])
    eps_data['fpedats'] = pd.to_datetime(eps_data['fpedats'])
    eps_data['statpers'] = pd.to_datetime(eps_data['statpers'])
    
    if 'fpi' in eps_data.columns:
        eps_data = eps_data[eps_data['fpi'].isin(['6', '7', '8', '9', 6, 7, 8, 9])]
    
    eps_data = eps_data.drop_duplicates(subset=['ticker', 'fpedats', 'statpers'])
    
    eps_agg = eps_data.sort_values('statpers').groupby(['ticker', 'fpedats']).agg({
        'meanest': 'last',
        'actual': 'first',
        'numest': 'first'
    }).reset_index()
    eps_agg = eps_agg.sort_values('fpedats')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=eps_agg['fpedats'],
        y=eps_agg['meanest'],
        name='Consensus Estimate',
        line=dict(color='#58a6ff', width=2.5),
        mode='lines+markers',
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=eps_agg['fpedats'],
        y=eps_agg['actual'],
        name='Actual EPS',
        line=dict(color='#3fb950', width=2.5),
        mode='lines+markers',
        marker=dict(size=6)
    ))
    
    # Shade beat/miss regions
    for _, row in eps_agg.iterrows():
        if pd.notna(row['actual']) and pd.notna(row['meanest']):
            color = 'rgba(63,185,80,0.15)' if row['actual'] >= row['meanest'] else 'rgba(248,81,73,0.15)'
            fig.add_shape(
                type='rect',
                x0=row['fpedats'] - pd.Timedelta(days=30),
                x1=row['fpedats'] + pd.Timedelta(days=30),
                y0=min(row['actual'], row['meanest']),
                y1=max(row['actual'], row['meanest']),
                fillcolor=color,
                line=dict(width=0),
                layer='below'
            )
    
    fig.update_layout(
        title=f'Historical EPS: Analyst Estimates vs Actuals — {company_name}',
        xaxis_title='Fiscal Period End',
        yaxis_title='EPS ($)',
        height=450,
        **DARK_LAYOUT
    )
    fig.update_xaxes(gridcolor='#30363d', linecolor='#30363d', tickfont=dict(color='#8b949e'))
    fig.update_yaxes(gridcolor='#30363d', linecolor='#30363d', tickfont=dict(color='#8b949e'))
    
    return fig.to_json()

def create_revision_trail_chart(df, ticker='JNJ', num_quarters=6):
    """Create 2-panel revision trail: beat/miss colored trails with avg path + accuracy funnel"""
    import numpy as np
    if df is None:
        return None, {}
    
    if ticker not in df['ticker'].values:
        ticker = df['ticker'].value_counts().index[0]
    
    ticker_data = df[df['ticker'] == ticker].copy()
    ticker_data['fpedats'] = pd.to_datetime(ticker_data['fpedats'])
    ticker_data['statpers'] = pd.to_datetime(ticker_data['statpers'])
    ticker_data = ticker_data.dropna(subset=['meanest'])
    
    if 'fpi' in ticker_data.columns:
        ticker_data = ticker_data[ticker_data['fpi'].isin(['6', '7', '8', '9', 6, 7, 8, 9])]
    
    ticker_data = ticker_data.drop_duplicates(subset=['ticker', 'fpedats', 'statpers'])
    ticker_data = ticker_data.sort_values(['fpedats', 'statpers'])
    
    ticker_data['months_before'] = ((ticker_data['fpedats'] - ticker_data['statpers']).dt.days / 30.44).round(1)
    ticker_data = ticker_data[ticker_data['months_before'] >= 0]
    
    all_quarters = ticker_data.dropna(subset=['actual'])['fpedats'].unique()
    all_quarters = sorted(all_quarters)
    display_quarters = all_quarters[-num_quarters:]
    
    company_name = ticker
    if 'company_name' in df.columns:
        names = df.loc[df['ticker'] == ticker, 'company_name'].dropna()
        if len(names) > 0:
            company_name = f"{ticker} - {names.iloc[0]}"
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.14,
        subplot_titles=['', '']
    )
    
    # =====================================================
    # ROW 1: Clean Estimate Trail — beat/miss coloring + average path
    # =====================================================
    BEAT_COLOR = '#3fb950'
    MISS_COLOR = '#f85149'
    AVG_COLOR = '#e3b341'
    
    # Pre-classify quarters and collect data for average path
    avg_path_data = {}  # month_bucket -> list of estimates
    beat_legend_added = False
    miss_legend_added = False
    
    n_display = len(display_quarters)
    
    for i, fpe in enumerate(display_quarters):
        q_data = ticker_data[ticker_data['fpedats'] == fpe].sort_values('months_before')
        actual = q_data['actual'].dropna().iloc[0] if q_data['actual'].notna().any() else None
        if len(q_data) == 0:
            continue
        
        fpe_label = pd.Timestamp(fpe).strftime('%Y-Q') + str((pd.Timestamp(fpe).month - 1) // 3 + 1)
        last_est = q_data.sort_values('months_before').iloc[0]['meanest']
        is_beat = actual is not None and actual > last_est
        color = BEAT_COLOR if is_beat else MISS_COLOR
        
        # Collect for average path
        for _, row in q_data.iterrows():
            mb = int(round(row['months_before']))
            avg_path_data.setdefault(mb, []).append(row['meanest'])
        
        # Gradient opacity: most recent quarters are more visible
        base_opacity = 0.25 + 0.45 * (i / max(n_display - 1, 1))
        line_width = 1.0 + 0.8 * (i / max(n_display - 1, 1))
        
        # Thin background line (individual quarter)
        show_legend = False
        legend_name = ''
        if is_beat and not beat_legend_added:
            show_legend = True
            legend_name = 'Beat (actual > estimate)'
            beat_legend_added = True
        elif not is_beat and not miss_legend_added:
            show_legend = True
            legend_name = 'Miss (actual ≤ estimate)'
            miss_legend_added = True
        
        fig.add_trace(go.Scatter(
            x=q_data['months_before'], y=q_data['meanest'],
            name=legend_name,
            legendgroup='beat' if is_beat else 'miss',
            showlegend=show_legend,
            line=dict(color=color, width=line_width),
            opacity=base_opacity,
            mode='lines',
            hovertemplate=f'{fpe_label}<br>%{{x:.0f}} mo before<br>Consensus: $%{{y:.2f}}<br>Actual: {"${:.2f}".format(actual) if actual else "N/A"}<extra></extra>'
        ), row=1, col=1)
        
        # Endpoint star at x=0 for the actual value
        if actual is not None:
            fig.add_trace(go.Scatter(
                x=[0], y=[actual],
                mode='markers',
                marker=dict(size=9, color=color, symbol='star',
                            line=dict(color='#0d1117', width=0.5)),
                showlegend=False,
                hovertemplate=f'{fpe_label}<br>★ Actual: ${actual:.2f}<extra></extra>'
            ), row=1, col=1)
        
        # Small label at the start of the line (earliest estimate)
        if len(q_data) > 0:
            start_row = q_data.iloc[-1]
            fig.add_annotation(
                x=start_row['months_before'], y=start_row['meanest'],
                text=fpe_label, showarrow=False,
                font=dict(size=8, color=color),
                opacity=base_opacity + 0.15,
                xanchor='left', yanchor='middle',
                row=1, col=1
            )
    
    # Vertical line at x=0 marking earnings date
    fig.add_vline(x=0, line_dash='solid', line_color='#484f58', line_width=1, row=1, col=1)
    fig.add_annotation(
        x=0, y=1, yref='y domain', xref='x',
        text='Earnings', showarrow=False,
        font=dict(size=9, color='#484f58'),
        xanchor='right', yanchor='top',
        row=1, col=1
    )
    
    # Bold average convergence path
    if avg_path_data:
        avg_months = sorted(avg_path_data.keys())
        avg_values = [np.mean(avg_path_data[m]) for m in avg_months]
        
        # Glow effect — wider semi-transparent line underneath
        fig.add_trace(go.Scatter(
            x=avg_months, y=avg_values,
            mode='lines', line=dict(color=AVG_COLOR, width=10),
            opacity=0.12, showlegend=False, hoverinfo='skip'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=avg_months, y=avg_values,
            name='Avg Consensus Path',
            line=dict(color=AVG_COLOR, width=3),
            mode='lines+markers', marker=dict(size=5, color=AVG_COLOR),
            hovertemplate='Avg across %{customdata} quarters<br>%{x} mo before<br>Consensus: $%{y:.2f}<extra></extra>',
            customdata=[len(avg_path_data[m]) for m in avg_months]
        ), row=1, col=1)
    
    # =====================================================
    # ROW 2: Accuracy Funnel + convergence annotation
    # =====================================================
    horizon_errors = {}
    for fpe in all_quarters:
        q_data = ticker_data[ticker_data['fpedats'] == fpe]
        actual = q_data['actual'].dropna().iloc[0] if q_data['actual'].notna().any() else None
        if actual is None or actual == 0:
            continue
        for _, row in q_data.iterrows():
            mb = int(round(row['months_before']))
            pct_err = abs((row['meanest'] - actual) / abs(actual) * 100)
            horizon_errors.setdefault(mb, []).append(pct_err)
    
    if horizon_errors:
        months_sorted = sorted(horizon_errors.keys())
        medians = [np.median(horizon_errors[m]) for m in months_sorted]
        p25 = [np.percentile(horizon_errors[m], 25) for m in months_sorted]
        p75 = [np.percentile(horizon_errors[m], 75) for m in months_sorted]
        p10 = [np.percentile(horizon_errors[m], 10) if len(horizon_errors[m]) >= 5 else p25[j] for j, m in enumerate(months_sorted)]
        p90 = [np.percentile(horizon_errors[m], 90) if len(horizon_errors[m]) >= 5 else p75[j] for j, m in enumerate(months_sorted)]
        counts = [len(horizon_errors[m]) for m in months_sorted]
        
        # P10-P90 outer band
        fig.add_trace(go.Scatter(
            x=months_sorted, y=p90,
            mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=months_sorted, y=p10,
            mode='lines', line=dict(width=0), fill='tonexty',
            fillcolor='rgba(88,166,255,0.06)', showlegend=False, hoverinfo='skip'
        ), row=2, col=1)
        
        # P25-P75 inner band
        fig.add_trace(go.Scatter(
            x=months_sorted, y=p75,
            mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=months_sorted, y=p25,
            mode='lines', line=dict(width=0), fill='tonexty',
            fillcolor='rgba(88,166,255,0.15)',
            name='P25–P75 Range', showlegend=True, hoverinfo='skip'
        ), row=2, col=1)
        
        # Median line with glow
        fig.add_trace(go.Scatter(
            x=months_sorted, y=medians,
            mode='lines', line=dict(color='#58a6ff', width=8),
            opacity=0.12, showlegend=False, hoverinfo='skip'
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=months_sorted, y=medians,
            name='Median |Error|',
            line=dict(color='#58a6ff', width=3),
            mode='lines+markers', marker=dict(size=5, symbol='circle'),
            hovertemplate='%{x} mo before<br>Median error: %{y:.1f}%<br>n=%{customdata}<extra></extra>',
            customdata=counts
        ), row=2, col=1)
        
        # 5% threshold line
        fig.add_hline(y=5, line_dash='dot', line_color='#3fb950', line_width=1,
                      annotation_text='5% threshold', annotation_font_color='#3fb950',
                      annotation_font_size=10, row=2, col=1)
    
    # =====================================================
    # Compute metrics
    # =====================================================
    metrics = {}
    walk_changes = []
    final_errors = []
    convergence_errors = []
    
    for fpe in all_quarters:
        q_data = ticker_data[ticker_data['fpedats'] == fpe].sort_values('months_before')
        actual = q_data['actual'].dropna().iloc[0] if q_data['actual'].notna().any() else None
        if actual is None or actual == 0:
            continue
        
        estimates_sorted = q_data.sort_values('months_before')
        if len(estimates_sorted) >= 2:
            earliest = estimates_sorted.iloc[-1]['meanest']
            latest = estimates_sorted.iloc[0]['meanest']
            walk_changes.append((latest - earliest) / abs(actual) * 100)
        
        last_est = estimates_sorted.iloc[0]['meanest']
        final_errors.append(abs((last_est - actual) / abs(actual) * 100))
        
        for _, row in estimates_sorted.iterrows():
            convergence_errors.append((row['months_before'], abs((row['meanest'] - actual) / abs(actual) * 100)))
    
    metrics['walk_down_pct'] = round(np.mean(walk_changes), 1) if walk_changes else None
    metrics['median_final_accuracy'] = round(np.median(final_errors), 1) if final_errors else None
    
    # Beat rate & surprise stats
    beat_count = 0
    miss_count = 0
    surprises = []
    for fpe in all_quarters:
        q_data = ticker_data[ticker_data['fpedats'] == fpe].sort_values('months_before')
        actual = q_data['actual'].dropna().iloc[0] if q_data['actual'].notna().any() else None
        if actual is None or actual == 0:
            continue
        last_est = q_data.iloc[0]['meanest']
        surprise_pct = (actual - last_est) / abs(actual) * 100
        surprises.append(surprise_pct)
        if actual > last_est:
            beat_count += 1
        else:
            miss_count += 1
    
    total_qtr = beat_count + miss_count
    metrics['beat_rate'] = round(beat_count / total_qtr * 100, 0) if total_qtr > 0 else None
    metrics['avg_surprise'] = round(np.mean(surprises), 1) if surprises else None
    metrics['quarters_analyzed'] = total_qtr
    
    # Convergence month
    if convergence_errors:
        conv_df = pd.DataFrame(convergence_errors, columns=['mb', 'err'])
        conv_df['mb_int'] = conv_df['mb'].astype(int)
        med_by_month = conv_df.groupby('mb_int')['err'].median().sort_index()
        below_5 = med_by_month[med_by_month < 5]
        metrics['convergence_month'] = int(below_5.index.max()) if len(below_5) > 0 else None
    else:
        metrics['convergence_month'] = None
    
    # Add convergence annotation to funnel
    if metrics['convergence_month'] is not None:
        fig.add_vline(x=metrics['convergence_month'], line_dash='dash', line_color='#bc8cff',
                      line_width=1, row=2, col=1)
        fig.add_annotation(
            x=metrics['convergence_month'], y=0,
            text=f"Converges <5% at {metrics['convergence_month']} mo",
            showarrow=True, arrowhead=0, arrowcolor='#bc8cff', arrowwidth=1,
            font=dict(size=10, color='#bc8cff'),
            yref='y2', yanchor='bottom', xanchor='left',
            ay=-25,
            row=2, col=1
        )
    
    # =====================================================
    # Layout
    # =====================================================
    fig.update_layout(
        height=750,
        **DARK_LAYOUT
    )
    fig.update_layout(
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font=dict(size=10, color='#c9d1d9'))
    )
    
    # Custom subplot titles as annotations for better styling
    fig.add_annotation(
        text=f'<b>Consensus Estimate Trail</b> — {company_name}',
        x=0.5, y=1.08, xref='paper', yref='paper',
        showarrow=False, font=dict(size=14, color='#c9d1d9')
    )
    fig.add_annotation(
        text=f'<b>Forecast Accuracy Funnel</b> — {len(all_quarters)} quarters analyzed',
        x=0.5, y=0.34, xref='paper', yref='paper',
        showarrow=False, font=dict(size=13, color='#c9d1d9')
    )
    
    for i in range(1, 3):
        fig.update_xaxes(gridcolor='#30363d', linecolor='#30363d', tickfont=dict(color='#8b949e'), row=i, col=1)
        fig.update_yaxes(gridcolor='#30363d', linecolor='#30363d', tickfont=dict(color='#8b949e'), row=i, col=1)
    
    fig.update_xaxes(autorange='reversed', title_text='Months Before Fiscal Period End  →',
                     title_font=dict(size=11, color='#8b949e'), dtick=1, row=1, col=1)
    fig.update_yaxes(title_text='EPS ($)', title_font=dict(size=11, color='#8b949e'), row=1, col=1)
    
    fig.update_xaxes(autorange='reversed', title_text='Months Before Fiscal Period End  →',
                     title_font=dict(size=11, color='#8b949e'), dtick=1, row=2, col=1)
    fig.update_yaxes(title_text='|% Error|', title_font=dict(size=11, color='#8b949e'),
                     rangemode='tozero', row=2, col=1)
    
    # Clean up the empty subplot_titles annotations
    fig.layout.annotations = [a for a in fig.layout.annotations if a.text != '']
    
    return fig.to_json(), metrics

def create_pead_chart(df):
    """Create Post-Earnings Announcement Drift chart"""
    if df is None:
        return None
    
    # Need quarterly returns data
    if 'quarterly_ret' not in df.columns:
        return None
    
    eps_data = df.copy()
    eps_data['fpedats'] = pd.to_datetime(eps_data['fpedats'])
    eps_data['statpers'] = pd.to_datetime(eps_data['statpers'])
    eps_data = eps_data.dropna(subset=['meanest', 'actual'])
    
    if 'fpi' in eps_data.columns:
        eps_data = eps_data[eps_data['fpi'].isin(['6', '7', '8', '9', 6, 7, 8, 9])]
    
    # Get last estimate before earnings for each quarter
    eps_data = eps_data[eps_data['statpers'] <= eps_data['fpedats']]
    eps_data = eps_data.drop_duplicates(subset=['ticker', 'fpedats', 'statpers'])
    eps_agg = eps_data.sort_values('statpers').groupby(['ticker', 'fpedats']).agg({
        'meanest': 'last',
        'actual': 'first',
        'quarterly_ret': 'first'
    }).reset_index()
    eps_agg = eps_agg.sort_values(['ticker', 'fpedats'])
    
    # Calculate surprise
    eps_agg['surprise_pct'] = ((eps_agg['actual'] - eps_agg['meanest']) / eps_agg['meanest'].abs()) * 100
    
    # Next-quarter return (post-announcement drift)
    eps_agg['next_q_ret'] = eps_agg.groupby('ticker')['quarterly_ret'].shift(-1)
    eps_agg = eps_agg.dropna(subset=['surprise_pct', 'next_q_ret'])
    
    # Bin into surprise quintiles
    eps_agg['surprise_bin'] = pd.qcut(eps_agg['surprise_pct'], 5, labels=['Q1\n(Big Miss)', 'Q2\n(Miss)', 'Q3\n(In-Line)', 'Q4\n(Beat)', 'Q5\n(Big Beat)'], duplicates='drop')
    
    if eps_agg['surprise_bin'].isna().all():
        return None
    
    bin_stats = eps_agg.groupby('surprise_bin', observed=True).agg(
        mean_drift=('next_q_ret', 'mean'),
        count=('next_q_ret', 'count')
    ).reset_index()
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        'Post-Earnings Drift by Surprise Quintile',
        'Earnings Surprise vs Next-Quarter Return'
    ], horizontal_spacing=0.12)
    
    # Left: Bar chart of average next-quarter return by surprise quintile
    bar_colors = ['#f85149', '#f0883e', '#8b949e', '#58a6ff', '#3fb950']
    fig.add_trace(go.Bar(
        x=bin_stats['surprise_bin'],
        y=bin_stats['mean_drift'] * 100,
        marker_color=bar_colors[:len(bin_stats)],
        text=[f'{v*100:.1f}%<br>n={n}' for v, n in zip(bin_stats['mean_drift'], bin_stats['count'])],
        textposition='outside',
        textfont=dict(color='#c9d1d9', size=10),
        showlegend=False,
        hovertemplate='%{x}<br>Avg Next-Q Return: %{y:.1f}%<extra></extra>'
    ), row=1, col=1)
    
    # Right: Scatter of surprise vs next-quarter return
    fig.add_trace(go.Scatter(
        x=eps_agg['surprise_pct'],
        y=eps_agg['next_q_ret'] * 100,
        mode='markers',
        marker=dict(
            size=6,
            color=eps_agg['surprise_pct'],
            colorscale=[[0, '#f85149'], [0.5, '#8b949e'], [1, '#3fb950']],
            opacity=0.6
        ),
        showlegend=False,
        hovertemplate='Surprise: %{x:.1f}%<br>Next-Q Return: %{y:.1f}%<extra></extra>'
    ), row=1, col=2)
    
    # Add OLS trendline
    from scipy import stats as sp_stats
    mask = eps_agg['surprise_pct'].notna() & eps_agg['next_q_ret'].notna()
    if mask.sum() > 2:
        slope, intercept, r, p, se = sp_stats.linregress(eps_agg.loc[mask, 'surprise_pct'], eps_agg.loc[mask, 'next_q_ret'] * 100)
        x_line = [eps_agg['surprise_pct'].min(), eps_agg['surprise_pct'].max()]
        y_line = [intercept + slope * x for x in x_line]
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode='lines',
            line=dict(color='#f0883e', width=2, dash='dash'),
            name=f'OLS (R²={r**2:.3f}, p={p:.3f})',
            showlegend=True
        ), row=1, col=2)
    
    fig.add_hline(y=0, line_color='#30363d', row=1, col=1)
    fig.add_hline(y=0, line_color='#30363d', row=1, col=2)
    
    fig.update_layout(
        title='Post-Earnings Announcement Drift (PEAD)<br><sub>Do stocks continue drifting after earnings surprises? Next-quarter returns by surprise magnitude.</sub>',
        height=420,
        **DARK_LAYOUT
    )
    fig.update_xaxes(gridcolor='#30363d', linecolor='#30363d', tickfont=dict(color='#8b949e'))
    fig.update_yaxes(gridcolor='#30363d', linecolor='#30363d', tickfont=dict(color='#8b949e'))
    fig.update_yaxes(title_text='Avg Next-Quarter Return (%)', row=1, col=1)
    fig.update_xaxes(title_text='Surprise Quintile', row=1, col=1)
    fig.update_yaxes(title_text='Next-Quarter Return (%)', row=1, col=2)
    fig.update_xaxes(title_text='EPS Surprise (%)', row=1, col=2)
    
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color='#c9d1d9', size=12)
    
    return fig.to_json()

def create_dispersion_chart(df, ticker='JNJ'):
    """Create analyst estimate dispersion chart showing consensus uncertainty over time"""
    if df is None:
        return None
    
    if ticker not in df['ticker'].values:
        ticker = df['ticker'].value_counts().index[0]
    
    ticker_data = df[df['ticker'] == ticker].copy()
    ticker_data['fpedats'] = pd.to_datetime(ticker_data['fpedats'])
    ticker_data['statpers'] = pd.to_datetime(ticker_data['statpers'])
    ticker_data = ticker_data.dropna(subset=['meanest'])
    
    if 'fpi' in ticker_data.columns:
        ticker_data = ticker_data[ticker_data['fpi'].isin(['6', '7', '8', '9', 6, 7, 8, 9])]
    
    # Get last observation per fiscal quarter
    ticker_data = ticker_data[ticker_data['statpers'] <= ticker_data['fpedats']]
    ticker_data = ticker_data.drop_duplicates(subset=['ticker', 'fpedats', 'statpers'])
    last_est = ticker_data.sort_values('statpers').groupby('fpedats').last().reset_index()
    last_est = last_est.sort_values('fpedats')
    
    # Use stdev if available, else mean-median gap as proxy
    has_stdev = 'stdev' in last_est.columns and last_est['stdev'].notna().sum() > 0
    
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=[
                            f'{ticker} Estimate Dispersion Over Time',
                            f'{ticker} Analyst Coverage & Agreement'
                        ],
                        vertical_spacing=0.15,
                        row_heights=[0.6, 0.4])
    
    # Top chart: meanest with uncertainty band
    fig.add_trace(go.Scatter(
        x=last_est['fpedats'], y=last_est['meanest'],
        name='Mean Estimate',
        line=dict(color='#58a6ff', width=2.5),
        mode='lines+markers',
        marker=dict(size=6)
    ), row=1, col=1)
    
    if has_stdev:
        upper = last_est['meanest'] + last_est['stdev']
        lower = last_est['meanest'] - last_est['stdev']
        disp_label = 'Std Dev'
    else:
        # Use mean-median gap * scaling as proxy
        gap = (last_est['meanest'] - last_est['medest']).abs()
        upper = last_est['meanest'] + gap * 3
        lower = last_est['meanest'] - gap * 3
        disp_label = 'Dispersion (Mean-Median gap × 3)'
    
    fig.add_trace(go.Scatter(
        x=pd.concat([last_est['fpedats'], last_est['fpedats'][::-1]]),
        y=pd.concat([upper, lower[::-1]]),
        fill='toself',
        fillcolor='rgba(88, 166, 255, 0.15)',
        line=dict(color='rgba(0,0,0,0)'),
        name=f'±1 {disp_label}',
        showlegend=True
    ), row=1, col=1)
    
    if last_est['medest'].notna().any():
        fig.add_trace(go.Scatter(
            x=last_est['fpedats'], y=last_est['medest'],
            name='Median Estimate',
            line=dict(color='#bc8cff', width=1.5, dash='dot'),
            mode='lines',
        ), row=1, col=1)
    
    if last_est['actual'].notna().any():
        fig.add_trace(go.Scatter(
            x=last_est['fpedats'], y=last_est['actual'],
            name='Actual EPS',
            line=dict(color='#3fb950', width=2),
            mode='lines+markers',
            marker=dict(size=6, symbol='diamond')
        ), row=1, col=1)
    
    # Bottom chart: analyst count + dispersion magnitude
    fig.add_trace(go.Bar(
        x=last_est['fpedats'],
        y=last_est['numest'],
        name='# Analysts',
        marker_color='rgba(88, 166, 255, 0.6)',
        showlegend=True
    ), row=2, col=1)
    
    # Overlay dispersion metric as line on secondary
    if has_stdev:
        disp_values = last_est['stdev']
        disp_name = 'Std Dev ($)'
    else:
        disp_values = (last_est['meanest'] - last_est['medest']).abs()
        disp_name = '|Mean - Median| ($)'
    
    fig.add_trace(go.Scatter(
        x=last_est['fpedats'], y=disp_values,
        name=disp_name,
        line=dict(color='#f0883e', width=2),
        mode='lines+markers',
        marker=dict(size=5),
        yaxis='y4'
    ), row=2, col=1)
    
    company_name = ticker
    if 'company_name' in df.columns:
        names = df.loc[df['ticker'] == ticker, 'company_name'].dropna()
        if len(names) > 0:
            company_name = f"{ticker} - {names.iloc[0]}"
    
    fig.update_layout(
        title=f'Analyst Estimate Dispersion — {company_name}<br><sub>Wide bands = high uncertainty. Narrow bands = strong consensus.</sub>',
        height=550,
        yaxis4=dict(
            overlaying='y3', side='right',
            gridcolor='rgba(0,0,0,0)',
            tickfont=dict(color='#f0883e'),
            title=dict(text=disp_name, font=dict(color='#f0883e'))
        ),
        **DARK_LAYOUT
    )
    fig.update_xaxes(gridcolor='#30363d', linecolor='#30363d', tickfont=dict(color='#8b949e'))
    fig.update_yaxes(gridcolor='#30363d', linecolor='#30363d', tickfont=dict(color='#8b949e'))
    fig.update_yaxes(title_text='EPS ($)', row=1, col=1)
    fig.update_yaxes(title_text='# Analysts', row=2, col=1)
    
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color='#c9d1d9', size=12)
    
    return fig.to_json()

def create_revenue_chart(df):
    """Create revenue by company chart"""
    if df is None:
        return None
    
    # Get latest year data per ticker
    latest = df.sort_values('fyear').groupby('ticker').last().reset_index()
    latest = latest.nlargest(20, 'sale')
    
    fig = px.bar(latest, x='ticker', y='sale', 
                 title='Revenue by Company (Latest Year, Top 20)',
                 labels={'sale': 'Revenue ($M)', 'ticker': 'Ticker'},
                 color='sale',
                 color_continuous_scale=[[0, '#1f6feb'], [1, '#58a6ff']])
    fig.update_layout(showlegend=False, height=400, **DARK_LAYOUT)
    return fig.to_json()

def create_eps_estimates_chart(df):
    """Create EPS estimates vs actual chart"""
    if df is None:
        return None
    
    # Filter to quarterly data only (exclude annual fpi=1)
    df_quarterly = df.copy()
    if 'fpi' in df_quarterly.columns:
        df_quarterly = df_quarterly[df_quarterly['fpi'].isin(['6', '7', '8', '9', 6, 7, 8, 9])]
    
    # Aggregate by ticker - get most recent estimate period
    latest_estimates = df_quarterly.sort_values('statpers').groupby('ticker').last().reset_index()
    latest_estimates = latest_estimates.dropna(subset=['meanest', 'actual'])
    latest_estimates = latest_estimates.nlargest(15, 'numest')
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Mean Estimate', x=latest_estimates['ticker'], 
                         y=latest_estimates['meanest'], marker_color='#58a6ff'))
    fig.add_trace(go.Bar(name='Actual', x=latest_estimates['ticker'], 
                         y=latest_estimates['actual'], marker_color='#3fb950'))
    
    fig.update_layout(
        title='EPS: Analyst Estimates vs Actual (Latest Period)',
        xaxis_title='Ticker',
        yaxis_title='EPS ($)',
        barmode='group',
        height=400,
        **DARK_LAYOUT
    )
    return fig.to_json()

def create_profitability_chart(df):
    """Create net income trends chart"""
    if df is None:
        return None
    
    # Get top 10 companies by revenue
    top_tickers = df.groupby('ticker')['sale'].max().nlargest(10).index.tolist()
    top_df = df[df['ticker'].isin(top_tickers)]
    
    # Aggregate by ticker and year
    yearly = top_df.groupby(['ticker', 'fyear']).agg({'ni': 'first', 'sale': 'first'}).reset_index()
    
    fig = px.line(yearly, x='fyear', y='ni', color='ticker',
                  title='Net Income Trends (Top 10 Companies by Revenue)',
                  labels={'fyear': 'Fiscal Year', 'ni': 'Net Income ($M)', 'ticker': 'Company'},
                  markers=True)
    fig.update_layout(height=400, **DARK_LAYOUT)
    return fig.to_json()

def create_ebitda_margin_chart(df):
    """Create Net Income Margin chart (quarterly data doesn't have EBITDA)"""
    if df is None:
        return None
    
    # Calculate Net Income margin (since quarterly data doesn't have EBITDA)
    latest = df.sort_values('fpedats').groupby('ticker').last().reset_index()
    latest = latest[latest['sale'] > 0]
    latest['ni_margin'] = (latest['ni'] / latest['sale']) * 100
    latest = latest.dropna(subset=['ni_margin'])
    latest = latest.nlargest(20, 'sale')
    latest = latest.sort_values('ni_margin', ascending=True)
    
    fig = px.bar(latest, x='ni_margin', y='ticker', orientation='h',
                 title='Net Income Margin % (Top 20 by Revenue)',
                 labels={'ni_margin': 'Net Income Margin (%)', 'ticker': 'Ticker'},
                 color='ni_margin',
                 color_continuous_scale=[[0, '#f85149'], [0.5, '#8b949e'], [1, '#3fb950']])
    fig.update_layout(height=500, showlegend=False, **DARK_LAYOUT)
    return fig.to_json()

def create_analyst_coverage_chart(df):
    """Create analyst coverage chart"""
    if df is None:
        return None
    
    # Get latest analyst count per ticker
    latest = df.sort_values('statpers').groupby('ticker').last().reset_index()
    latest = latest.dropna(subset=['numest'])
    latest = latest.nlargest(20, 'numest')
    
    fig = px.bar(latest, x='ticker', y='numest',
                 title='Analyst Coverage (Number of Estimates)',
                 labels={'numest': 'Number of Analysts', 'ticker': 'Ticker'},
                 color='numest',
                 color_continuous_scale=[[0, '#1f6feb'], [1, '#a5d6ff']])
    fig.update_layout(height=400, showlegend=False, **DARK_LAYOUT)
    return fig.to_json()

def create_estimate_accuracy_chart(df):
    """Create estimate accuracy chart"""
    if df is None:
        return None
    
    # Calculate estimate error
    df_calc = df.dropna(subset=['meanest', 'actual']).copy()
    df_calc['estimate_error'] = ((df_calc['meanest'] - df_calc['actual']) / df_calc['actual'].abs()) * 100
    
    # Aggregate by ticker
    accuracy = df_calc.groupby('ticker')['estimate_error'].mean().reset_index()
    accuracy = accuracy.sort_values('estimate_error')
    
    fig = px.bar(accuracy, x='ticker', y='estimate_error',
                 title='Average EPS Estimate Error by Company (%)',
                 labels={'estimate_error': 'Mean Error (%)', 'ticker': 'Ticker'},
                 color='estimate_error',
                 color_continuous_scale=[[0, '#3fb950'], [0.5, '#8b949e'], [1, '#f85149']])
    fig.update_layout(height=400, showlegend=False, **DARK_LAYOUT)
    return fig.to_json()

def create_eps_predictability_chart(df, ticker=None, year_start=None, year_end=None):
    """Create EPS Predictability scatter - current EPS vs lagged EPS with trendline"""
    if df is None:
        return None
    
    # Get unique EPS observations per ticker and fiscal period (fpedats)
    eps_data = df.dropna(subset=['actual']).copy()
    eps_data['fpedats'] = pd.to_datetime(eps_data['fpedats'])
    eps_data['statpers'] = pd.to_datetime(eps_data['statpers'])
    
    # Filter to quarterly data only (fpi 6=Q1, 7=Q2, 8=Q3, 9=Q4) - exclude annual (fpi=1)
    if 'fpi' in eps_data.columns:
        eps_data = eps_data[eps_data['fpi'].isin(['6', '7', '8', '9', 6, 7, 8, 9])]
    
    # First deduplicate to unique IBES observations (remove Compustat duplicates)
    eps_data = eps_data.drop_duplicates(subset=['ticker', 'fpedats', 'statpers'])
    
    # Sort by statpers before aggregating to get correct last estimate
    eps_agg = eps_data.sort_values('statpers').groupby(['ticker', 'fpedats']).agg({'actual': 'first', 'meanest': 'last'}).reset_index()
    eps_agg = eps_agg.sort_values(['ticker', 'fpedats'])
    
    # Apply year filter before creating lag
    if year_start:
        eps_agg = eps_agg[eps_agg['fpedats'].dt.year >= year_start]
    if year_end:
        eps_agg = eps_agg[eps_agg['fpedats'].dt.year <= year_end]
    
    # Create lagged EPS (previous period's actual)
    eps_agg['eps_lag'] = eps_agg.groupby('ticker')['actual'].shift(1)
    eps_agg = eps_agg.dropna(subset=['eps_lag'])
    
    if ticker:
        eps_agg = eps_agg[eps_agg['ticker'] == ticker]
        title = f'EPS Predictability: {ticker}'
    else:
        title = 'EPS Predictability (All Companies)'
    
    if len(eps_agg) == 0:
        return None
    
    # Single trendline for all data (don't color by ticker when showing all)
    fig = px.scatter(eps_agg, x='eps_lag', y='actual', 
                     hover_data=['ticker'] if not ticker else None,
                     trendline='ols',
                     title=title,
                     labels={'eps_lag': 'Previous Period EPS', 'actual': 'Current EPS', 'ticker': 'Company'})
    
    # Use consistent styling
    fig.update_traces(marker=dict(size=8, opacity=0.7, color='#58a6ff'), selector=dict(mode='markers'))
    fig.update_layout(height=400, **DARK_LAYOUT)
    
    # Style the trendline
    for trace in fig.data:
        if trace.mode == 'lines':
            trace.line.color = '#f0883e'
            trace.line.width = 2
    
    return fig.to_json()

def create_revenue_time_chart(df, ticker1='JNJ', ticker2='MSFT', year_start=None, year_end=None):
    """Create Revenue over Time chart for two companies"""
    if df is None:
        return None
    
    fig = go.Figure()
    colors = ['#58a6ff', '#f0883e']  # Blue for company 1, orange for company 2
    
    for i, ticker in enumerate([ticker1, ticker2]):
        # Filter for the selected ticker
        company_data = df[df['ticker'] == ticker].copy()
        
        # Aggregate by fiscal year
        yearly = company_data.groupby('fyear').agg({'sale': 'first'}).reset_index()
        yearly = yearly.sort_values('fyear')
        
        # Apply year filter
        if year_start:
            yearly = yearly[yearly['fyear'] >= year_start]
        if year_end:
            yearly = yearly[yearly['fyear'] <= year_end]
        
        if len(yearly) == 0:
            continue
        
        fig.add_trace(go.Scatter(
            x=yearly['fyear'],
            y=yearly['sale'],
            mode='lines+markers',
            line=dict(color=colors[i], width=3),
            marker=dict(size=8, color=colors[i]),
            name=ticker
        ))
    
    fig.update_layout(
        title=f'Revenue over Time: {ticker1} vs {ticker2}',
        xaxis_title='Fiscal Year',
        yaxis_title='Revenue ($M)',
        height=350,
        showlegend=True,
        **DARK_LAYOUT
    )
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.update_xaxes(dtick=1, tickformat='d')  # Force integer years only
    return fig.to_json()

def create_company_comparison_chart(df, ticker1='JNJ', ticker2='MSFT', year_start=None, year_end=None):
    """Create side-by-side comparison showing estimate convergence to actuals over time"""
    if df is None:
        return None
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=[ticker1, ticker2],
                        horizontal_spacing=0.1)
    
    for i, ticker in enumerate([ticker1, ticker2], 1):
        # Get EPS data for ticker
        ticker_data = df[df['ticker'] == ticker].copy()
        ticker_data = ticker_data.dropna(subset=['meanest', 'actual'])
        ticker_data['fpedats'] = pd.to_datetime(ticker_data['fpedats'])
        ticker_data['statpers'] = pd.to_datetime(ticker_data['statpers'])
        
        # Filter to quarterly data only (fpi 6=Q1, 7=Q2, 8=Q3, 9=Q4) - exclude annual (fpi=1)
        if 'fpi' in ticker_data.columns:
            ticker_data = ticker_data[ticker_data['fpi'].isin(['6', '7', '8', '9', 6, 7, 8, 9])]
        
        # Deduplicate to unique IBES observations (remove Compustat duplicates)
        ticker_data = ticker_data.drop_duplicates(subset=['ticker', 'fpedats', 'statpers'])
        ticker_data = ticker_data.sort_values('statpers')
        
        # Apply year filter
        if year_start:
            ticker_data = ticker_data[ticker_data['fpedats'].dt.year >= year_start]
        if year_end:
            ticker_data = ticker_data[ticker_data['fpedats'].dt.year <= year_end]
        
        if len(ticker_data) == 0:
            continue
        
        # Get unique fiscal periods and their actuals
        actuals = ticker_data.groupby('fpedats').agg({'actual': 'first'}).reset_index()
        actuals = actuals.sort_values('fpedats')
        
        # Plot all estimate points over time (x = statpers, y = meanest)
        # Color by fiscal period to show convergence for each quarter
        fig.add_trace(go.Scatter(
            x=ticker_data['statpers'],
            y=ticker_data['meanest'],
            name='Estimates',
            mode='markers',
            marker=dict(size=6, color='#f0883e', opacity=0.6),
            hovertemplate='%{x}<br>Estimate: $%{y:.2f}<extra></extra>',
            showlegend=(i == 1)
        ), row=1, col=i)
        
        # Plot actuals as horizontal step lines (flat lines for each quarter)
        # Create step-like data for actuals
        for idx, row in actuals.iterrows():
            fpedats = row['fpedats']
            actual = row['actual']
            
            # Find all estimate dates for this fiscal period
            period_data = ticker_data[ticker_data['fpedats'] == fpedats]
            if len(period_data) == 0:
                continue
            
            min_date = period_data['statpers'].min()
            max_date = period_data['statpers'].max()
            
            # Draw horizontal line for actual
            fig.add_trace(go.Scatter(
                x=[min_date, max_date],
                y=[actual, actual],
                name='Actual' if (i == 1 and idx == 0) else None,
                mode='lines',
                line=dict(color='#3fb950', width=3),
                hovertemplate=f'Actual: ${actual:.2f}<extra></extra>',
                showlegend=(i == 1 and idx == 0)
            ), row=1, col=i)
    
    fig.update_layout(
        title='Estimate Convergence: How Forecasts Approach Actuals Over Time',
        height=400,
        **DARK_LAYOUT
    )
    fig.update_xaxes(gridcolor='#30363d', linecolor='#30363d', tickfont=dict(color='#8b949e', size=9), title_text='Estimate Date')
    fig.update_yaxes(gridcolor='#30363d', linecolor='#30363d', tickfont=dict(color='#8b949e', size=9), title_text='EPS ($)')
    
    # Update subplot title colors
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color='#c9d1d9', size=12)
    
    return fig.to_json()

def create_eps_surprise_returns_chart(df, year_start=None, year_end=None):
    """Create scatter plot of EPS Surprise vs Quarterly Stock Returns"""
    if df is None:
        return None
    
    # Check if returns data exists
    if 'quarterly_ret' not in df.columns or 'eps_surprise' not in df.columns:
        return None
    
    # Use quarterly data: pair each quarter's surprise with that quarter's return
    plot_data = df.dropna(subset=['quarterly_ret', 'eps_surprise']).copy()
    plot_data['fpedats'] = pd.to_datetime(plot_data['fpedats'])
    
    # Deduplicate to one row per ticker per fiscal quarter (fpedats)
    plot_data = plot_data.sort_values('statpers').groupby(['ticker', 'fpedats']).agg({
        'eps_surprise': 'first',
        'quarterly_ret': 'first',
        'fyear': 'first'
    }).reset_index()
    
    # Apply year filter
    if year_start:
        plot_data = plot_data[plot_data['fyear'] >= year_start]
    if year_end:
        plot_data = plot_data[plot_data['fyear'] <= year_end]
    
    if len(plot_data) == 0:
        return None
    
    from scipy import stats as sp_stats
    
    fig = go.Figure()
    
    # Color each ticker differently
    tickers = plot_data['ticker'].unique()
    palette = px.colors.qualitative.Plotly
    for i, t in enumerate(tickers):
        td = plot_data[plot_data['ticker'] == t]
        fig.add_trace(go.Scatter(
            x=td['eps_surprise'], y=td['quarterly_ret'],
            mode='markers',
            name=t,
            marker=dict(size=10, opacity=0.7, color=palette[i % len(palette)]),
            hovertemplate=f'{t}<br>Surprise: %{{x:.1f}}%<br>Return: %{{y:.1f}}%<extra></extra>'
        ))
    
    # Single OLS trendline for ALL data
    mask = plot_data['eps_surprise'].notna() & plot_data['quarterly_ret'].notna()
    if mask.sum() > 2:
        slope, intercept, r, p, se = sp_stats.linregress(plot_data.loc[mask, 'eps_surprise'], plot_data.loc[mask, 'quarterly_ret'])
        x_range = [plot_data['eps_surprise'].min(), plot_data['eps_surprise'].max()]
        y_range = [intercept + slope * x for x in x_range]
        fig.add_trace(go.Scatter(
            x=x_range, y=y_range,
            mode='lines',
            line=dict(color='#f0883e', width=3, dash='dash'),
            name=f'OLS (R²={r**2:.3f}, p={p:.3f})',
            showlegend=True
        ))
    
    fig.update_layout(
        title='EPS Surprise vs Quarterly Stock Return',
        xaxis_title='EPS Surprise (%)',
        yaxis_title='Quarterly Return (%)',
        height=400, **DARK_LAYOUT
    )
    
    # Add reference lines at 0
    fig.add_hline(y=0, line_dash="dash", line_color="#8b949e", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="#8b949e", opacity=0.5)
    
    return fig.to_json()

def create_returns_comparison_chart(df, ticker1='JNJ', ticker2='MSFT', year_start=None, year_end=None):
    """Create annual returns comparison for two companies"""
    if df is None:
        return None
    
    if 'annual_ret' not in df.columns or 'fpe_year' not in df.columns:
        return None
    
    fig = go.Figure()
    colors = ['#58a6ff', '#f0883e']
    
    for i, ticker in enumerate([ticker1, ticker2]):
        # Get returns data for ticker
        ticker_data = df[df['ticker'] == ticker].copy()
        ticker_data = ticker_data.dropna(subset=['annual_ret'])
        
        # Aggregate by calendar year (fpe_year) since annual_ret is calendar-year based
        yearly = ticker_data.groupby('fpe_year').agg({'annual_ret': 'mean'}).reset_index()
        yearly = yearly.sort_values('fpe_year')
        
        # Apply year filter
        if year_start:
            yearly = yearly[yearly['fpe_year'] >= year_start]
        if year_end:
            yearly = yearly[yearly['fpe_year'] <= year_end]
        
        if len(yearly) == 0:
            continue
        
        fig.add_trace(go.Bar(
            x=yearly['fpe_year'],
            y=yearly['annual_ret'],
            name=ticker,
            marker_color=colors[i],
            opacity=0.8
        ))
    
    fig.update_layout(
        title=f'Annual Stock Returns: {ticker1} vs {ticker2}',
        xaxis_title='Calendar Year',
        yaxis_title='Annual Return (%)',
        height=350,
        barmode='group',
        **DARK_LAYOUT
    )
    fig.update_xaxes(dtick=1, tickformat='d')
    fig.add_hline(y=0, line_dash="dash", line_color="#8b949e", opacity=0.5)
    
    return fig.to_json()

def create_eps_vs_returns_trend_chart(df, ticker='JNJ', year_start=None, year_end=None):
    """Create dual-axis chart showing quarterly EPS and Returns over time for a company"""
    if df is None:
        return None
    
    if 'quarterly_ret' not in df.columns:
        return None
    
    ticker_data = df[df['ticker'] == ticker].copy()
    ticker_data = ticker_data.dropna(subset=['actual', 'quarterly_ret'])
    ticker_data['fpedats'] = pd.to_datetime(ticker_data['fpedats'])
    
    # Deduplicate to one row per fiscal quarter
    quarterly = ticker_data.sort_values('statpers').groupby('fpedats').agg({
        'actual': 'first',
        'meanest': 'last',
        'quarterly_ret': 'first',
        'fyear': 'first'
    }).reset_index()
    quarterly = quarterly.sort_values('fpedats')
    
    # Apply year filter
    if year_start:
        quarterly = quarterly[quarterly['fyear'] >= year_start]
    if year_end:
        quarterly = quarterly[quarterly['fyear'] <= year_end]
    
    if len(quarterly) == 0:
        return None
    
    # Format quarter labels for x-axis
    quarterly['quarter_label'] = quarterly['fpedats'].dt.to_period('Q').astype(str)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # EPS Actual on primary y-axis
    fig.add_trace(go.Scatter(
        x=quarterly['quarter_label'],
        y=quarterly['actual'],
        name='Actual EPS',
        line=dict(color='#3fb950', width=3),
        mode='lines+markers',
        marker=dict(size=8)
    ), secondary_y=False)
    
    # EPS Estimate on primary y-axis
    fig.add_trace(go.Scatter(
        x=quarterly['quarter_label'],
        y=quarterly['meanest'],
        name='Estimated EPS',
        line=dict(color='#58a6ff', width=2, dash='dot'),
        mode='lines+markers',
        marker=dict(size=6)
    ), secondary_y=False)
    
    # Stock Return on secondary y-axis
    fig.add_trace(go.Bar(
        x=quarterly['quarter_label'],
        y=quarterly['quarterly_ret'],
        name='Quarterly Return',
        marker_color='#f0883e',
        opacity=0.5
    ), secondary_y=True)
    
    fig.update_layout(
        title=f'{ticker}: Quarterly EPS vs Stock Returns',
        height=400,
        **DARK_LAYOUT
    )
    fig.update_xaxes(title_text='Quarter', tickangle=-45, tickfont=dict(size=9))
    fig.update_yaxes(title_text='EPS ($)', secondary_y=False, gridcolor='#30363d')
    fig.update_yaxes(title_text='Quarterly Return (%)', secondary_y=True, gridcolor='#30363d')
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    
    return fig.to_json()

def get_ticker_list(df):
    """Get list of unique tickers with company names for dropdowns"""
    if df is None:
        return []
    
    # Check if company_name column exists
    if 'company_name' in df.columns:
        # Get unique ticker-company pairs
        ticker_names = df.dropna(subset=['ticker']).drop_duplicates(subset=['ticker'])[['ticker', 'company_name']]
        ticker_names = ticker_names.sort_values('ticker')
        return ticker_names.to_dict('records')  # List of {'ticker': 'AAPL', 'company_name': 'Apple Inc'}
    else:
        # Fallback to just tickers if no company names
        tickers = sorted(df['ticker'].unique().tolist())
        return [{'ticker': t, 'company_name': t} for t in tickers]

def get_year_range(df):
    """Get min and max years from the data"""
    if df is None:
        return 2020, 2025
    min_year = max(int(df['fyear'].min()), 2020)
    max_year = int(df['fyear'].max())
    return min_year, max_year

# =============================================================================
# Prediction Functions
# =============================================================================

def predict_eps(df, ticker, method='linear', periods=4, timeframe='all', start_date=None, end_date=None):
    """
    Predict future EPS based on historical data
    
    Methods:
    - linear: Linear regression trend
    - exponential: Exponential smoothing
    - moving_avg: Moving average projection
    - analyst: Use current analyst estimates (meanest)
    
    timeframe: 'all', '3m', '6m', '1y', '2y'
    start_date/end_date: arbitrary date range (overrides timeframe if both provided)
    """
    import numpy as np
    from scipy import stats
    
    if df is None:
        return None, None
    
    # Get historical data for ticker
    ticker_data = df[df['ticker'] == ticker].copy()
    ticker_data = ticker_data.dropna(subset=['actual'])
    ticker_data['fpedats'] = pd.to_datetime(ticker_data['fpedats'])
    ticker_data['statpers'] = pd.to_datetime(ticker_data['statpers'])
    
    # Filter to quarterly data only (exclude annual fpi=1)
    if 'fpi' in ticker_data.columns:
        ticker_data = ticker_data[ticker_data['fpi'].isin(['6', '7', '8', '9', 6, 7, 8, 9])]
    
    # Only use estimates made before or on the announcement date
    ticker_data = ticker_data[ticker_data['statpers'] <= ticker_data['fpedats']]
    
    # Deduplicate to unique quarters
    ticker_data = ticker_data.drop_duplicates(subset=['ticker', 'fpedats', 'statpers'])
    eps_history = ticker_data.sort_values('statpers').groupby('fpedats').agg({
        'actual': 'first',
        'meanest': 'last'
    }).reset_index()
    eps_history = eps_history.sort_values('fpedats')
    
    # Apply timeframe filter
    if start_date and end_date and len(eps_history) > 0:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        eps_history = eps_history[(eps_history['fpedats'] >= start_dt) & (eps_history['fpedats'] <= end_dt)]
    elif timeframe != 'all' and len(eps_history) > 0:
        cutoff_map = {'3m': 3, '6m': 6, '1y': 12, '2y': 24}
        months_back = cutoff_map.get(timeframe, None)
        if months_back:
            cutoff_date = eps_history['fpedats'].max() - pd.DateOffset(months=months_back)
            eps_history = eps_history[eps_history['fpedats'] >= cutoff_date]
    
    if len(eps_history) < 2:
        return None, None
    
    # Calculate historical surprises
    eps_history['surprise'] = eps_history['actual'] - eps_history['meanest']
    eps_history['surprise_pct'] = (eps_history['surprise'] / eps_history['meanest'].abs()) * 100
    
    # Prepare for prediction
    historical_eps = eps_history['actual'].values
    historical_dates = eps_history['fpedats'].values
    n = len(historical_eps)
    x = np.arange(n)
    
    # Generate future dates (next 4 quarters)
    last_date = pd.to_datetime(historical_dates[-1])
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=3), periods=periods, freq='QE')
    
    if method == 'linear':
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, historical_eps)
        future_x = np.arange(n, n + periods)
        predicted_eps = intercept + slope * future_x
        confidence = r_value ** 2  # R-squared as confidence
        
    elif method == 'exponential':
        # Simple exponential smoothing
        alpha = 0.3  # Smoothing factor
        smoothed = [historical_eps[0]]
        for i in range(1, n):
            smoothed.append(alpha * historical_eps[i] + (1 - alpha) * smoothed[-1])
        
        # Project forward using trend from last few periods
        recent_trend = np.mean(np.diff(historical_eps[-4:])) if len(historical_eps) >= 4 else 0
        predicted_eps = [smoothed[-1] + recent_trend * (i + 1) for i in range(periods)]
        
    elif method == 'moving_avg':
        # Moving average with trend
        window = min(4, len(historical_eps))
        ma = np.mean(historical_eps[-window:])
        trend = np.mean(np.diff(historical_eps[-window:])) if window > 1 else 0
        predicted_eps = [ma + trend * (i + 1) for i in range(periods)]
        
    elif method == 'analyst':
        # Use analyst estimates if available for future periods
        # For now, project based on latest estimate with historical surprise pattern
        latest_estimate = eps_history['meanest'].iloc[-1]
        avg_surprise = eps_history['surprise'].mean()
        predicted_eps = [latest_estimate * (1 + avg_surprise / 100) for _ in range(periods)]
    
    elif method == 'holt_winters':
        # Holt-Winters Exponential Smoothing (triple exponential smoothing)
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        try:
            seasonal_periods = min(4, n // 2)
            if n >= 2 * seasonal_periods and seasonal_periods >= 2:
                model = ExponentialSmoothing(
                    historical_eps,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=seasonal_periods
                ).fit(optimized=True)
            else:
                # Not enough data for seasonality, fall back to additive trend only
                model = ExponentialSmoothing(
                    historical_eps,
                    trend='add',
                    seasonal=None
                ).fit(optimized=True)
            predicted_eps = model.forecast(periods).tolist()
        except Exception:
            # Fallback: simple trend projection
            recent_trend = np.mean(np.diff(historical_eps[-4:])) if n >= 4 else 0
            predicted_eps = [historical_eps[-1] + recent_trend * (i + 1) for i in range(periods)]
    
    elif method == 'sarima':
        # SARIMA (Seasonal ARIMA)
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                seasonal_order = (1, 0, 0, 4) if n >= 8 else (0, 0, 0, 0)
                model = SARIMAX(
                    historical_eps,
                    order=(1, 1, 1),
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)
                predicted_eps = model.forecast(periods).tolist()
        except Exception:
            # Fallback: simple trend projection
            recent_trend = np.mean(np.diff(historical_eps[-4:])) if n >= 4 else 0
            predicted_eps = [historical_eps[-1] + recent_trend * (i + 1) for i in range(periods)]
    
    else:
        return None, None
    
    # --- Backtest-based confidence ---
    # Leave-last-4-out: train on history[:-4], predict 4 ahead, compare to actual
    backtest_errors = []
    holdout = min(4, n // 2)  # use up to 4 quarters for backtest
    if n > holdout + 2:  # need at least 3 training points
        train_eps = historical_eps[:-holdout]
        actual_holdout = historical_eps[-holdout:]
        train_n = len(train_eps)
        train_x = np.arange(train_n)
        
        if method == 'linear':
            s, i_c, _, _, _ = stats.linregress(train_x, train_eps)
            bt_pred = [i_c + s * (train_n + j) for j in range(holdout)]
        elif method == 'exponential':
            sm = [train_eps[0]]
            for k in range(1, train_n):
                sm.append(0.3 * train_eps[k] + 0.7 * sm[-1])
            tr = np.mean(np.diff(train_eps[-4:])) if train_n >= 4 else 0
            bt_pred = [sm[-1] + tr * (j + 1) for j in range(holdout)]
        elif method == 'moving_avg':
            w = min(4, train_n)
            m = np.mean(train_eps[-w:])
            t = np.mean(np.diff(train_eps[-w:])) if w > 1 else 0
            bt_pred = [m + t * (j + 1) for j in range(holdout)]
        elif method == 'analyst':
            bt_pred = [train_eps[-1]] * holdout  # naive: last actual as forecast
        elif method == 'holt_winters':
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            try:
                sp = min(4, train_n // 2)
                if train_n >= 2 * sp and sp >= 2:
                    hw_model = ExponentialSmoothing(train_eps, trend='add', seasonal='add', seasonal_periods=sp).fit(optimized=True)
                else:
                    hw_model = ExponentialSmoothing(train_eps, trend='add', seasonal=None).fit(optimized=True)
                bt_pred = hw_model.forecast(holdout).tolist()
            except Exception:
                bt_pred = [train_eps[-1]] * holdout
        elif method == 'sarima':
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            import warnings
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    so = (1, 0, 0, 4) if train_n >= 8 else (0, 0, 0, 0)
                    sarima_model = SARIMAX(train_eps, order=(1, 1, 1), seasonal_order=so, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                    bt_pred = sarima_model.forecast(holdout).tolist()
            except Exception:
                bt_pred = [train_eps[-1]] * holdout
        
        for j in range(holdout):
            if actual_holdout[j] != 0:
                backtest_errors.append(abs(bt_pred[j] - actual_holdout[j]) / abs(actual_holdout[j]))
    
    if backtest_errors:
        avg_error = np.mean(backtest_errors)
        # Convert error to confidence: 0% error -> 1.0, 100% error -> 0.0
        confidence = max(0.0, min(1.0, 1.0 - avg_error))
        backtest_mape = round(avg_error * 100, 1)
    else:
        # Fallback if not enough data for backtest
        confidence = r_value ** 2 if method == 'linear' else 0.5
        backtest_mape = None
    
    # Predict surprises based on historical pattern
    avg_surprise_pct = eps_history['surprise_pct'].mean()
    std_surprise_pct = eps_history['surprise_pct'].std()
    
    return {
        'historical_dates': [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10] for d in historical_dates],
        'historical_eps': historical_eps.tolist(),
        'historical_estimates': eps_history['meanest'].tolist(),
        'historical_surprises': eps_history['surprise_pct'].tolist(),
        'future_dates': [d.strftime('%Y-%m-%d') for d in future_dates],
        'predicted_eps': [round(e, 2) for e in predicted_eps],
        'avg_surprise_pct': round(avg_surprise_pct, 2),
        'std_surprise_pct': round(std_surprise_pct, 2),
        'confidence': round(confidence, 2),
        'backtest_mape': backtest_mape,
        'method': method
    }, eps_history

def create_prediction_chart(df, ticker='JNJ', method='linear', timeframe='all', start_date=None, end_date=None, surprise_method='seasonal_avg'):
    """Create EPS prediction chart with historical data and forecast"""
    import numpy as np
    
    prediction_data, eps_history = predict_eps(df, ticker, method, periods=4, timeframe=timeframe, start_date=start_date, end_date=end_date)
    
    if prediction_data is None:
        return None
    
    # Auto-show faded full history when a sub-range is selected
    is_subrange = bool(start_date and end_date) or (timeframe != 'all')
    all_history_data = None
    if is_subrange:
        all_history_data, _ = predict_eps(df, ticker, method, periods=4, timeframe='all')
    
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=[f'{ticker} EPS Forecast ({method.title()} Method)', 
                                       f'{ticker} Surprise Pattern (Historical + Predicted)'],
                        vertical_spacing=0.15,
                        row_heights=[0.6, 0.4],
                        shared_xaxes=True)
    
    # --- Top chart: EPS forecast ---
    # Show all historical data as faded background if toggle is on
    if all_history_data:
        fig.add_trace(go.Scatter(
            x=all_history_data['historical_dates'],
            y=all_history_data['historical_eps'],
            name='Full History (Actual)',
            line=dict(color='#3fb950', width=1, dash='dot'),
            mode='lines+markers',
            marker=dict(size=5, opacity=0.3),
            opacity=0.35
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=all_history_data['historical_dates'],
            y=all_history_data['historical_estimates'],
            name='Full History (Estimate)',
            line=dict(color='#58a6ff', width=1, dash='dot'),
            mode='lines+markers',
            marker=dict(size=4, opacity=0.3),
            opacity=0.35
        ), row=1, col=1)
    
    # Historical actual EPS (timeframe-filtered, used for prediction)
    fig.add_trace(go.Scatter(
        x=prediction_data['historical_dates'],
        y=prediction_data['historical_eps'],
        name='Actual EPS' + (' (Model Window)' if all_history_data else ''),
        line=dict(color='#3fb950', width=2),
        mode='lines+markers',
        marker=dict(size=8)
    ), row=1, col=1)
    
    # Historical estimates
    fig.add_trace(go.Scatter(
        x=prediction_data['historical_dates'],
        y=prediction_data['historical_estimates'],
        name='Analyst Estimate' + (' (Model Window)' if all_history_data else ''),
        line=dict(color='#58a6ff', width=2, dash='dot'),
        mode='lines+markers',
        marker=dict(size=6)
    ), row=1, col=1)
    
    # Predicted EPS
    fig.add_trace(go.Scatter(
        x=prediction_data['future_dates'],
        y=prediction_data['predicted_eps'],
        name='Predicted EPS',
        line=dict(color='#f0883e', width=3),
        mode='lines+markers',
        marker=dict(size=10, symbol='diamond')
    ), row=1, col=1)
    
    # Confidence band for predictions
    confidence = prediction_data['confidence']
    std_eps = np.std(prediction_data['historical_eps'])
    upper_band = [e + std_eps * (1 - confidence) for e in prediction_data['predicted_eps']]
    lower_band = [e - std_eps * (1 - confidence) for e in prediction_data['predicted_eps']]
    
    fig.add_trace(go.Scatter(
        x=prediction_data['future_dates'] + prediction_data['future_dates'][::-1],
        y=upper_band + lower_band[::-1],
        fill='toself',
        fillcolor='rgba(240, 136, 62, 0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Confidence Band',
        showlegend=True
    ), row=1, col=1)
    
    # Training window overlay when sub-range is active
    if is_subrange and prediction_data['historical_dates']:
        fig.add_vrect(
            x0=prediction_data['historical_dates'][0],
            x1=prediction_data['historical_dates'][-1],
            fillcolor='rgba(88, 166, 255, 0.06)',
            line=dict(color='rgba(88, 166, 255, 0.25)', width=1, dash='dot'),
            annotation_text='Training Window',
            annotation_position='top left',
            annotation=dict(font=dict(color='#58a6ff', size=10)),
            row=1, col=1
        )
    
    # --- Bottom chart: Surprise pattern ---
    # Show all historical surprises as faded background when sub-range is active
    if all_history_data:
        all_colors = ['rgba(63,185,80,0.3)' if s >= 0 else 'rgba(248,81,73,0.3)' for s in all_history_data['historical_surprises']]
        fig.add_trace(go.Bar(
            x=all_history_data['historical_dates'],
            y=all_history_data['historical_surprises'],
            name='Full History Surprise',
            marker_color=all_colors,
            showlegend=False,
            opacity=0.3
        ), row=2, col=1)
    
    colors = ['#3fb950' if s >= 0 else '#f85149' for s in prediction_data['historical_surprises']]
    fig.add_trace(go.Bar(
        x=prediction_data['historical_dates'],
        y=prediction_data['historical_surprises'],
        name='Surprise %',
        marker_color=colors,
        showlegend=False
    ), row=2, col=1)
    
    # --- Predicted surprise bars for future quarters ---
    hist_dates_dt = pd.to_datetime(prediction_data['historical_dates'])
    hist_surprises = prediction_data['historical_surprises']
    future_dates_dt = pd.to_datetime(prediction_data['future_dates'])
    
    if surprise_method == 'overall_avg':
        avg_s = np.mean(hist_surprises) if len(hist_surprises) > 0 else 0
        predicted_surprises = [avg_s] * len(future_dates_dt)
    elif surprise_method == 'last_year':
        # Use the most recent surprise for each calendar quarter
        last_by_q = {}
        for dt, sv in zip(hist_dates_dt, hist_surprises):
            last_by_q[dt.quarter] = sv
        predicted_surprises = [last_by_q.get(fd.quarter, prediction_data['avg_surprise_pct']) for fd in future_dates_dt]
    elif surprise_method == 'trend':
        # Linear trend extrapolation of surprise values
        if len(hist_surprises) >= 2:
            x_vals = np.arange(len(hist_surprises))
            coeffs = np.polyfit(x_vals, hist_surprises, 1)
            predicted_surprises = [float(coeffs[0] * (len(hist_surprises) + i) + coeffs[1]) for i in range(len(future_dates_dt))]
        else:
            predicted_surprises = [prediction_data['avg_surprise_pct']] * len(future_dates_dt)
    else:  # seasonal_avg (default)
        hist_quarters = hist_dates_dt.quarter
        q_avgs = {}
        for q_num, s_val in zip(hist_quarters, hist_surprises):
            q_avgs.setdefault(q_num, []).append(s_val)
        q_avgs = {k: np.mean(v) for k, v in q_avgs.items()}
        predicted_surprises = [q_avgs.get(fd.quarter, prediction_data['avg_surprise_pct']) for fd in future_dates_dt]
    
    pred_surp_colors = ['rgba(240,136,62,0.7)' for _ in predicted_surprises]
    fig.add_trace(go.Bar(
        x=prediction_data['future_dates'],
        y=predicted_surprises,
        name='Predicted Surprise',
        marker_color=pred_surp_colors,
        marker_line=dict(color='#f0883e', width=1.5),
        showlegend=True,
        opacity=0.6
    ), row=2, col=1)
    
    # Add average surprise line
    avg_surprise = prediction_data['avg_surprise_pct']
    fig.add_hline(y=avg_surprise, line_dash="dash", line_color="#8b949e", 
                  annotation_text=f"Avg: {avg_surprise:.1f}%", row=2, col=1)
    fig.add_hline(y=0, line_color="#30363d", row=2, col=1)
    
    fig.update_layout(
        height=550,
        **DARK_LAYOUT
    )
    # Override DARK_LAYOUT legend with horizontal positioning
    fig.update_layout(
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(color='#c9d1d9')
        )
    )
    fig.update_xaxes(gridcolor='#30363d', linecolor='#30363d', tickfont=dict(color='#8b949e'))
    fig.update_yaxes(gridcolor='#30363d', linecolor='#30363d', tickfont=dict(color='#8b949e'))
    fig.update_yaxes(title_text='EPS ($)', row=1, col=1)
    fig.update_yaxes(title_text='Surprise (%)', row=2, col=1)
    
    # Update subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color='#c9d1d9', size=12)
    
    return fig.to_json()

def create_surprise_prediction_chart(df, ticker='JNJ', timeframe='all', start_date=None, end_date=None):
    """Create chart predicting next quarter's surprise based on patterns"""
    import numpy as np
    
    if df is None:
        return None
    
    # Get historical data
    ticker_data = df[df['ticker'] == ticker].copy()
    ticker_data = ticker_data.dropna(subset=['actual', 'meanest'])
    ticker_data['fpedats'] = pd.to_datetime(ticker_data['fpedats'])
    ticker_data['statpers'] = pd.to_datetime(ticker_data['statpers'])
    
    # Filter to quarterly data only (exclude annual fpi=1)
    if 'fpi' in ticker_data.columns:
        ticker_data = ticker_data[ticker_data['fpi'].isin(['6', '7', '8', '9', 6, 7, 8, 9])]
    
    # Only use estimates made before or on the announcement date
    ticker_data = ticker_data[ticker_data['statpers'] <= ticker_data['fpedats']]
    
    # Deduplicate
    ticker_data = ticker_data.drop_duplicates(subset=['ticker', 'fpedats', 'statpers'])
    eps_all = ticker_data.sort_values('statpers').groupby('fpedats').agg({
        'actual': 'first',
        'meanest': 'last'
    }).reset_index()
    eps_all = eps_all.sort_values('fpedats')
    
    # Keep full history for display if needed
    eps_history = eps_all.copy()
    is_subrange = bool(start_date and end_date) or (timeframe != 'all')
    
    # Apply timeframe filter
    if start_date and end_date and len(eps_history) > 0:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        eps_history = eps_history[(eps_history['fpedats'] >= start_dt) & (eps_history['fpedats'] <= end_dt)]
    elif timeframe != 'all' and len(eps_history) > 0:
        cutoff_map = {'3m': 3, '6m': 6, '1y': 12, '2y': 24}
        months_back = cutoff_map.get(timeframe, None)
        if months_back:
            cutoff_date = eps_history['fpedats'].max() - pd.DateOffset(months=months_back)
            eps_history = eps_history[eps_history['fpedats'] >= cutoff_date]
    
    if len(eps_history) < 2:
        return None
    
    # Calculate surprises for filtered data
    eps_history['surprise_pct'] = ((eps_history['actual'] - eps_history['meanest']) / 
                                    eps_history['meanest'].abs()) * 100
    eps_history['quarter'] = pd.to_datetime(eps_history['fpedats']).dt.quarter
    
    # Also calculate for full history if sub-range is active
    if is_subrange:
        eps_all['surprise_pct'] = ((eps_all['actual'] - eps_all['meanest']) / 
                                    eps_all['meanest'].abs()) * 100
        eps_all['quarter'] = pd.to_datetime(eps_all['fpedats']).dt.quarter
    
    # Analyze by quarter (seasonality)
    quarterly_avg = eps_history.groupby('quarter')['surprise_pct'].agg(['mean', 'std']).reset_index()
    quarterly_avg.columns = ['quarter', 'avg_surprise', 'std_surprise']
    
    # Create visualization
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Surprise by Quarter (Seasonality)', 'Surprise Trend Over Time'],
                        horizontal_spacing=0.12)
    
    # Left: Quarterly pattern
    q_labels = ['Q1', 'Q2', 'Q3', 'Q4']
    colors = ['#3fb950' if s >= 0 else '#f85149' for s in quarterly_avg['avg_surprise']]
    
    fig.add_trace(go.Bar(
        x=[q_labels[int(q)-1] for q in quarterly_avg['quarter']],
        y=quarterly_avg['avg_surprise'],
        error_y=dict(type='data', array=quarterly_avg['std_surprise'], visible=True),
        marker_color=colors,
        name='Avg Surprise'
    ), row=1, col=1)
    
    # Right: Trend over time with rolling average
    # Show full history as faded dots when sub-range is active
    if is_subrange:
        all_rolling = eps_all['surprise_pct'].rolling(window=4, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=eps_all['fpedats'],
            y=eps_all['surprise_pct'],
            name='Full History',
            mode='markers',
            marker=dict(size=6, color='#58a6ff', opacity=0.25),
            showlegend=False
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=eps_all['fpedats'],
            y=all_rolling,
            name='Full Rolling Avg',
            line=dict(color='#f0883e', width=1, dash='dot'),
            opacity=0.3,
            showlegend=False
        ), row=1, col=2)
    
    rolling_avg = eps_history['surprise_pct'].rolling(window=4, min_periods=1).mean()
    
    fig.add_trace(go.Scatter(
        x=eps_history['fpedats'],
        y=eps_history['surprise_pct'],
        name='Actual Surprise',
        mode='markers',
        marker=dict(size=8, color='#58a6ff')
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=eps_history['fpedats'],
        y=rolling_avg,
        name='4-Quarter Rolling Avg',
        line=dict(color='#f0883e', width=2)
    ), row=1, col=2)
    
    fig.add_hline(y=0, line_color="#30363d", row=1, col=1)
    fig.add_hline(y=0, line_color="#30363d", row=1, col=2)
    
    fig.update_layout(
        title=f'{ticker}: Earnings Surprise Analysis',
        height=350,
        **DARK_LAYOUT
    )
    fig.update_xaxes(gridcolor='#30363d', linecolor='#30363d', tickfont=dict(color='#8b949e'))
    fig.update_yaxes(gridcolor='#30363d', linecolor='#30363d', tickfont=dict(color='#8b949e'), 
                     title_text='Surprise (%)')
    
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color='#c9d1d9', size=12)
    
    return fig.to_json()

@app.route('/')
def index():
    """Main dashboard page"""
    df = load_data()
    stats = get_summary_stats(df)
    tickers = get_ticker_list(df)
    min_year, max_year = get_year_range(df)
    
    # Get list of ticker symbols for default selection
    ticker_symbols = [t['ticker'] for t in tickers] if tickers else []
    
    # Default selections for interactive charts
    default_ticker1 = 'JNJ' if 'JNJ' in ticker_symbols else (ticker_symbols[0] if ticker_symbols else None)
    default_ticker2 = 'MSFT' if 'MSFT' in ticker_symbols else (ticker_symbols[1] if len(ticker_symbols) > 1 else default_ticker1)
    
    charts = {
        'eps_history': create_eps_history_chart(df, default_ticker1),
        'revenue': create_revenue_chart(df),
        'eps_estimates': create_eps_estimates_chart(df),
        'profitability': create_profitability_chart(df),
        'ebitda_margin': create_ebitda_margin_chart(df),
        'analyst_coverage': create_analyst_coverage_chart(df),
        'estimate_accuracy': create_estimate_accuracy_chart(df),
        # Interactive charts
        'eps_predictability': create_eps_predictability_chart(df),
        'revenue_time': create_revenue_time_chart(df, default_ticker1, default_ticker2),
        'company_comparison': create_company_comparison_chart(df, default_ticker1, default_ticker2),
        # Returns charts (requires CRSP data)
        'eps_surprise_returns': create_eps_surprise_returns_chart(df),
        'returns_comparison': create_returns_comparison_chart(df, default_ticker1, default_ticker2),
        'eps_returns_trend': create_eps_vs_returns_trend_chart(df, default_ticker1),
        # New analysis charts
        'revision_trail': create_revision_trail_chart(df, default_ticker1)[0],
        'pead': create_pead_chart(df),
        'dispersion': create_dispersion_chart(df, default_ticker1)
    }
    
    data_exists = df is not None
    last_modified = None
    if data_exists and os.path.exists(DATA_FILE):
        last_modified = datetime.fromtimestamp(os.path.getmtime(DATA_FILE)).strftime('%Y-%m-%d %H:%M:%S')
    
    return render_template('index.html', 
                         stats=stats, 
                         charts=charts, 
                         data_exists=data_exists,
                         last_modified=last_modified,
                         refresh_status=refresh_status,
                         tickers=tickers,
                         default_ticker1=default_ticker1,
                         default_ticker2=default_ticker2,
                         min_year=min_year,
                         max_year=max_year)

@app.route('/api/chart/eps_history')
def api_eps_history():
    """API to get historical EPS chart for a featured ticker"""
    ticker = request.args.get('ticker', 'JNJ')
    df = load_data()
    chart = create_eps_history_chart(df, ticker)
    return jsonify({'chart': chart})

@app.route('/api/chart/revision_trail')
def api_revision_trail():
    """API to get estimate revision trail chart"""
    ticker = request.args.get('ticker', 'JNJ')
    num_quarters = request.args.get('num_quarters', 6, type=int)
    num_quarters = max(2, min(20, num_quarters))
    df = load_data()
    chart, metrics = create_revision_trail_chart(df, ticker, num_quarters)
    return jsonify({'chart': chart, 'metrics': metrics})

@app.route('/api/chart/pead')
def api_pead():
    """API to get post-earnings announcement drift chart"""
    df = load_data()
    chart = create_pead_chart(df)
    return jsonify({'chart': chart})

@app.route('/api/chart/dispersion')
def api_dispersion():
    """API to get estimate dispersion chart"""
    ticker = request.args.get('ticker', 'JNJ')
    df = load_data()
    chart = create_dispersion_chart(df, ticker)
    return jsonify({'chart': chart})

@app.route('/api/chart/revenue_time')
def api_revenue_time():
    """API to get revenue time chart for selected tickers"""
    ticker1 = request.args.get('ticker1', 'JNJ')
    ticker2 = request.args.get('ticker2', 'MSFT')
    year_start = request.args.get('year_start', type=int)
    year_end = request.args.get('year_end', type=int)
    df = load_data()
    chart = create_revenue_time_chart(df, ticker1, ticker2, year_start, year_end)
    return jsonify({'chart': chart})

@app.route('/api/chart/predictability')
def api_predictability():
    """API to get EPS predictability chart for a specific ticker"""
    ticker = request.args.get('ticker', None)
    year_start = request.args.get('year_start', type=int)
    year_end = request.args.get('year_end', type=int)
    df = load_data()
    chart = create_eps_predictability_chart(df, ticker if ticker else None, year_start, year_end)
    return jsonify({'chart': chart})

@app.route('/api/chart/comparison')
def api_comparison():
    """API to get company comparison chart"""
    ticker1 = request.args.get('ticker1', 'JNJ')
    ticker2 = request.args.get('ticker2', 'MSFT')
    year_start = request.args.get('year_start', type=int)
    year_end = request.args.get('year_end', type=int)
    df = load_data()
    chart = create_company_comparison_chart(df, ticker1, ticker2, year_start, year_end)
    return jsonify({'chart': chart})

@app.route('/api/chart/eps_surprise_returns')
def api_eps_surprise_returns():
    """API to get EPS surprise vs returns scatter"""
    year_start = request.args.get('year_start', type=int)
    year_end = request.args.get('year_end', type=int)
    df = load_data()
    chart = create_eps_surprise_returns_chart(df, year_start, year_end)
    return jsonify({'chart': chart})

@app.route('/api/chart/returns_comparison')
def api_returns_comparison():
    """API to get returns comparison chart"""
    ticker1 = request.args.get('ticker1', 'JNJ')
    ticker2 = request.args.get('ticker2', 'MSFT')
    year_start = request.args.get('year_start', type=int)
    year_end = request.args.get('year_end', type=int)
    df = load_data()
    chart = create_returns_comparison_chart(df, ticker1, ticker2, year_start, year_end)
    return jsonify({'chart': chart})

@app.route('/api/chart/eps_returns_trend')
def api_eps_returns_trend():
    """API to get EPS vs returns trend for a ticker"""
    ticker = request.args.get('ticker', 'JNJ')
    year_start = request.args.get('year_start', type=int)
    year_end = request.args.get('year_end', type=int)
    df = load_data()
    chart = create_eps_vs_returns_trend_chart(df, ticker, year_start, year_end)
    return jsonify({'chart': chart})

@app.route('/api/chart/prediction')
def api_prediction():
    """API to get EPS prediction chart"""
    ticker = request.args.get('ticker', 'JNJ')
    method = request.args.get('method', 'linear')
    timeframe = request.args.get('timeframe', 'all')
    start_date = request.args.get('start_date', None)
    end_date = request.args.get('end_date', None)
    surprise_method = request.args.get('surprise_method', 'seasonal_avg')
    df = load_data()
    chart = create_prediction_chart(df, ticker, method, timeframe, start_date=start_date, end_date=end_date, surprise_method=surprise_method)
    return jsonify({'chart': chart})

@app.route('/api/chart/surprise_analysis')
def api_surprise_analysis():
    """API to get surprise analysis chart"""
    ticker = request.args.get('ticker', 'JNJ')
    timeframe = request.args.get('timeframe', 'all')
    start_date = request.args.get('start_date', None)
    end_date = request.args.get('end_date', None)
    df = load_data()
    chart = create_surprise_prediction_chart(df, ticker, timeframe, start_date=start_date, end_date=end_date)
    return jsonify({'chart': chart})

@app.route('/api/prediction_data')
def api_prediction_data():
    """API to get raw prediction data"""
    ticker = request.args.get('ticker', 'JNJ')
    method = request.args.get('method', 'linear')
    timeframe = request.args.get('timeframe', 'all')
    start_date = request.args.get('start_date', None)
    end_date = request.args.get('end_date', None)
    df = load_data()
    prediction_data, _ = predict_eps(df, ticker, method, timeframe=timeframe, start_date=start_date, end_date=end_date)
    return jsonify(prediction_data)

@app.route('/refresh', methods=['GET', 'POST'])
def refresh_page():
    """Page for refreshing WRDS data"""
    return render_template('refresh.html', refresh_status=refresh_status)

@app.route('/api/refresh', methods=['POST'])
def refresh_data():
    """API endpoint to refresh data from WRDS"""
    global refresh_status
    
    if refresh_status['running']:
        return jsonify({'error': 'Refresh already in progress'}), 400
    
    username = request.form.get('username')
    password = request.form.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    def run_pull():
        global refresh_status
        refresh_status = {'running': True, 'message': 'Connecting to WRDS...', 'success': None}
        
        try:
            # Create a modified version of the script that uses provided credentials
            script_content = f'''
import pandas as pd
import wrds
from datetime import datetime

# Using provided credentials
username = "{username}"
password = "{password}"

db = wrds.Connection(wrds_username=username, wrds_password=password)

# Top 50 S&P 500 companies by market cap (tickers)
top_50_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'BRK.B', 'TSLA', 'UNH', 'XOM',
    'JPM', 'JNJ', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'LLY',
    'PEP', 'KO', 'COST', 'AVGO', 'WMT', 'MCD', 'CSCO', 'TMO', 'ABT', 'CRM',
    'ACN', 'DHR', 'ADBE', 'NKE', 'CMCSA', 'VZ', 'NEE', 'INTC', 'TXN', 'PM',
    'WFC', 'BMY', 'UPS', 'RTX', 'QCOM', 'HON', 'T', 'ORCL', 'IBM', 'LOW'
]

# Pull IBES analyst estimates data - QUARTERLY ONLY (exclude annual fpi='1')
print("Pulling IBES quarterly data...")
ibes_query = """
    SELECT ticker, cusip, statpers, fpedats, meanest, medest, stdev, actual, numest, fpi
    FROM ibes.statsum_epsus
    WHERE ticker IN %(tickers)s
    AND statpers >= '2020-01-01'
    AND fpi IN ('6', '7', '8', '9')
"""
ibes_data = db.raw_sql(ibes_query, params={{'tickers': tuple(top_50_tickers)}})
print(f"IBES data shape: {{ibes_data.shape}}")
print(f"FPI breakdown (6=Q1, 7=Q2, 8=Q3, 9=Q4): {{ibes_data['fpi'].value_counts().to_dict()}}")

# Deduplicate IBES - keep one row per ticker/fpedats/statpers (different fpi have same actual)
ibes_data = ibes_data.sort_values(['ticker', 'fpedats', 'statpers', 'fpi'])
ibes_data = ibes_data.drop_duplicates(subset=['ticker', 'fpedats', 'statpers'], keep='last')
print(f"IBES after dedup: {{ibes_data.shape}}")

# Pull Compustat QUARTERLY data
print("Pulling Compustat quarterly data...")
comp_query = """
    SELECT gvkey, tic, cusip, datadate, fyearq as fyear, fqtr, atq as at, saleq as sale, niq as ni
    FROM comp.fundq
    WHERE tic IN %(tickers)s
    AND datadate >= '2020-01-01'
    AND indfmt = 'INDL'
    AND datafmt = 'STD'
    AND popsrc = 'D'
    AND consol = 'C'
"""
comp_data = db.raw_sql(comp_query, params={{'tickers': tuple(top_50_tickers)}})
print(f"Compustat quarterly data shape: {{comp_data.shape}}")

# Get IBES to CRSP linking table
print("Pulling linking tables...")
ibes_crsp_query = """
    SELECT ticker, permno, sdate, edate
    FROM wrdsapps.ibcrsphist
    WHERE ticker IN %(tickers)s
"""
ibes_crsp_link = db.raw_sql(ibes_crsp_query, params={{'tickers': tuple(top_50_tickers)}})

# Get CRSP to Compustat linking table
ccm_query = """
    SELECT lpermno as permno, gvkey, linkdt, linkenddt
    FROM crsp.ccmxpf_linktable
    WHERE lpermno IN %(permnos)s
    AND linktype IN ('LC', 'LU')
    AND linkprim IN ('P', 'C')
"""
permnos = tuple(ibes_crsp_link['permno'].unique().tolist())
ccm_link = db.raw_sql(ccm_query, params={{'permnos': permnos}})

# Merge data
print("Merging datasets...")
ibes_linked = ibes_data.merge(ibes_crsp_link, on='ticker', how='inner')
ibes_linked['statpers'] = pd.to_datetime(ibes_linked['statpers'])
ibes_linked['sdate'] = pd.to_datetime(ibes_linked['sdate'])
ibes_linked['edate'] = pd.to_datetime(ibes_linked['edate'])
ibes_linked = ibes_linked[
    (ibes_linked['statpers'] >= ibes_linked['sdate']) & 
    (ibes_linked['statpers'] <= ibes_linked['edate'])
]

ibes_linked = ibes_linked.merge(ccm_link, on='permno', how='inner')
ibes_linked['linkdt'] = pd.to_datetime(ibes_linked['linkdt'])
ibes_linked['linkenddt'] = pd.to_datetime(ibes_linked['linkenddt'], errors='coerce')
ibes_linked['linkenddt'] = ibes_linked['linkenddt'].fillna(pd.Timestamp.today())
ibes_linked = ibes_linked[
    (ibes_linked['statpers'] >= ibes_linked['linkdt']) & 
    (ibes_linked['statpers'] <= ibes_linked['linkenddt'])
]

# Pull CRSP monthly stock returns + delisting returns
print("Pulling CRSP monthly returns...")
permnos_list = ibes_linked['permno'].unique().tolist()
crsp_query = """
    SELECT permno, date, ret, prc, shrout, vol
    FROM crsp.msf
    WHERE permno IN %(permnos)s
    AND date >= '2020-01-01'
"""
crsp_data = db.raw_sql(crsp_query, params={{'permnos': tuple(permnos_list)}})

delist_query = """
    SELECT permno, dlstdt as date, dlret
    FROM crsp.msedelist
    WHERE permno IN %(permnos)s
    AND dlstdt >= '2020-01-01'
"""
delist_data = db.raw_sql(delist_query, params={{'permnos': tuple(permnos_list)}})
print(f"CRSP data shape: {{crsp_data.shape}}, Delisting records: {{len(delist_data)}}")

crsp_data['date'] = pd.to_datetime(crsp_data['date'])

# Filter out missing/invalid returns and prices
crsp_data = crsp_data[crsp_data['ret'].notna()]
crsp_data = crsp_data[(crsp_data['ret'] > -1) & (crsp_data['ret'] < 10)]
crsp_data = crsp_data[crsp_data['prc'].notna() & (crsp_data['prc'].abs() > 0)]
print(f"CRSP data after filtering: {{len(crsp_data)}} records")

# Incorporate delisting returns
delist_data['date'] = pd.to_datetime(delist_data['date'])
delist_data['dlret'] = pd.to_numeric(delist_data['dlret'], errors='coerce')
delist_data['date'] = delist_data['date'] + pd.offsets.MonthEnd(0)
crsp_data = crsp_data.merge(delist_data, on=['permno', 'date'], how='left')
crsp_data['ret_adj'] = crsp_data['ret']
has_dlret = crsp_data['dlret'].notna()
crsp_data.loc[has_dlret, 'ret_adj'] = (
    (1 + crsp_data.loc[has_dlret, 'ret']) * (1 + crsp_data.loc[has_dlret, 'dlret']) - 1
)
crsp_data['ret_plus1'] = 1 + crsp_data['ret_adj']
crsp_data['prc_abs'] = crsp_data['prc'].abs()

# Build FPEDATS-based quarter windows
ibes_linked['fpedats'] = pd.to_datetime(ibes_linked['fpedats'])
quarter_windows = ibes_linked[['permno', 'fpedats']].drop_duplicates().sort_values(['permno', 'fpedats'])
quarter_windows['prev_fpedats'] = quarter_windows.groupby('permno')['fpedats'].shift(1)
quarter_windows['qtr_start'] = quarter_windows['prev_fpedats'] + pd.Timedelta(days=1)
quarter_windows.loc[quarter_windows['prev_fpedats'].isna(), 'qtr_start'] = (
    quarter_windows.loc[quarter_windows['prev_fpedats'].isna(), 'fpedats'] - pd.Timedelta(days=90)
)
quarter_windows['qtr_end'] = quarter_windows['fpedats']
quarter_windows = quarter_windows.drop(columns=['prev_fpedats'])

# Compound CRSP monthly returns within each FPEDATS window
print("Computing FPEDATS-aligned quarterly returns...")
results = []
for permno, grp in quarter_windows.groupby('permno'):
    crsp_perm = crsp_data[crsp_data['permno'] == permno].sort_values('date')
    if len(crsp_perm) == 0:
        for _, row in grp.iterrows():
            results.append({{'permno': permno, 'fpedats': row['fpedats'],
                'quarterly_ret': None, 'quarter_end_price': None,
                'ret_flag': 'no_crsp_data', 'n_months': 0}})
        continue
    for _, row in grp.iterrows():
        qstart, qend = row['qtr_start'], row['qtr_end']
        mask = (crsp_perm['date'] > qstart) & (crsp_perm['date'] <= qend)
        window_data = crsp_perm[mask]
        if len(window_data) == 0:
            results.append({{'permno': permno, 'fpedats': row['fpedats'],
                'quarterly_ret': None, 'quarter_end_price': None,
                'ret_flag': 'no_months_in_window', 'n_months': 0}})
        else:
            compound = window_data['ret_plus1'].prod()
            ret_qtr = (compound - 1) * 100
            price = window_data['prc_abs'].iloc[-1]
            n = len(window_data)
            flag = 'ok' if n >= 2 else 'short_window'
            results.append({{'permno': permno, 'fpedats': row['fpedats'],
                'quarterly_ret': round(ret_qtr, 4),
                'quarter_end_price': price,
                'ret_flag': flag, 'n_months': n}})
quarterly_returns = pd.DataFrame(results)
print(f"Computed {{len(quarterly_returns)}} quarterly returns")

# Calendar-year annual returns (for comparison chart)
crsp_data['year'] = crsp_data['date'].dt.year
crsp_annual = crsp_data.sort_values('date').groupby(['permno', 'year']).agg({{
    'ret_plus1': 'prod',
    'prc_abs': 'last',
    'vol': 'sum'
}}).reset_index()
crsp_annual.columns = ['permno', 'year', 'annual_ret_factor', 'year_end_price', 'annual_volume']
crsp_annual['annual_ret'] = (crsp_annual['annual_ret_factor'] - 1) * 100
crsp_annual = crsp_annual.drop(columns=['annual_ret_factor'])

# Merge IBES with Compustat quarterly
print("Merging IBES with Compustat quarterly...")
ibes_linked['fpedats'] = pd.to_datetime(ibes_linked['fpedats'])
comp_data['datadate'] = pd.to_datetime(comp_data['datadate'])

# Deduplicate Compustat by gvkey+datadate (in case of duplicates)
comp_data = comp_data.drop_duplicates(subset=['gvkey', 'datadate'], keep='last')
print(f"Compustat after dedup: {{len(comp_data)}} records")

# Match on gvkey and fiscal quarter end date
merged_data = ibes_linked.merge(
    comp_data, 
    left_on=['gvkey', 'fpedats'], 
    right_on=['gvkey', 'datadate'], 
    how='left', 
    suffixes=('_ibes', '_comp')
)
print(f"After Compustat merge: {{len(merged_data)}} records")

# Add calendar info
merged_data['fpe_year'] = merged_data['fpedats'].dt.year
merged_data['fpe_quarter'] = merged_data['fpedats'].dt.quarter

# Merge FPEDATS-aligned quarterly returns
merged_data = merged_data.merge(quarterly_returns, 
    on=['permno', 'fpedats'], 
    how='left')

# Merge CRSP annual returns (calendar-year for comparison chart)
merged_data = merged_data.merge(crsp_annual, 
    left_on=['permno', 'fpe_year'], 
    right_on=['permno', 'year'], 
    how='left',
    suffixes=('', '_ann'))

# Calculate EPS surprise (actual - estimate) / |estimate|
merged_data['eps_surprise'] = (merged_data['actual'] - merged_data['meanest']) / merged_data['meanest'].abs() * 100
merged_data['eps_surprise'] = merged_data['eps_surprise'].clip(-100, 100)  # Cap extreme values

# Clean up duplicate columns
cols_to_drop = [c for c in merged_data.columns if c.endswith('_qtr') or c.endswith('_ann')]
merged_data = merged_data.drop(columns=cols_to_drop, errors='ignore')

# Summary stats
print(f"\\nFinal dataset summary:")
print(f"Total records: {{len(merged_data)}}")
print(f"Unique tickers: {{merged_data['ticker'].nunique()}}")
print(f"FPI breakdown (6=Q1, 7=Q2, 8=Q3, 9=Q4):")
print(merged_data['fpi'].value_counts().sort_index())
print(f"\\nUnique fiscal periods per ticker:")
print(merged_data.groupby('ticker')['fpedats'].nunique().describe())

# Show sample for verification
print(f"\\nSample AAPL data:")
aapl_sample = merged_data[merged_data['ticker']=='AAPL'].sort_values('fpedats').drop_duplicates('fpedats')[['ticker','fpedats','fpi','meanest','actual','quarterly_ret']].head(10)
print(aapl_sample.to_string())

merged_data.to_csv(r'{DATA_FILE}', index=False)
print(f"\\nSaved {{len(merged_data)}} records with quarterly EPS data")
'''
            
            # Run the script
            result = subprocess.run(
                [sys.executable, '-c', script_content],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                refresh_status = {'running': False, 'message': 'Data refresh completed successfully!', 'success': True}
            else:
                refresh_status = {'running': False, 'message': f'Error: {result.stderr}', 'success': False}
                
        except subprocess.TimeoutExpired:
            refresh_status = {'running': False, 'message': 'Timeout: Data pull took too long', 'success': False}
        except Exception as e:
            refresh_status = {'running': False, 'message': f'Error: {str(e)}', 'success': False}
    
    # Run in background thread
    thread = threading.Thread(target=run_pull)
    thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/api/refresh/status')
def get_refresh_status():
    """Get current refresh status"""
    return jsonify(refresh_status)

@app.route('/data')
def data_table():
    """View raw data as table"""
    df = load_data()
    if df is None:
        return render_template('data.html', data=None)
    
    # Get unique ticker/quarter combinations for cleaner view
    agg_dict = {
        'sale': 'first',
        'ni': 'first',
        'at': 'first',
        'meanest': 'last',
        'actual': 'last',
        'numest': 'last'
    }
    # Include fpi if available (fiscal period indicator)
    group_cols = ['ticker', 'fpedats']
    if 'fpi' in df.columns:
        group_cols.append('fpi')
    summary = df.groupby(group_cols).agg(agg_dict).reset_index()
    
    return render_template('data.html', 
                         data=summary.to_html(classes='table table-striped table-hover', index=False))

if __name__ == '__main__':
    print("Starting WRDS Dashboard...")
    print(f"Data file: {DATA_FILE}")
    app.run(debug=True, port=5000)
