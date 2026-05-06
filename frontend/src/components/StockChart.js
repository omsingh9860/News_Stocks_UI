import React, { useEffect, useRef } from 'react';
import { X, ExternalLink } from 'lucide-react';

// Map our internal symbols to TradingView symbols
const toTradingViewSymbol = (symbol, exchange) => {
  const US_SYMBOLS = {
    SPX: 'SP:SPX',
    DJI: 'DJ:DJI',
    IXIC: 'NASDAQ:IXIC',
    AAPL: 'NASDAQ:AAPL',
    MSFT: 'NASDAQ:MSFT',
    GOOGL: 'NASDAQ:GOOGL',
    AMZN: 'NASDAQ:AMZN',
    TSLA: 'NASDAQ:TSLA',
    NVDA: 'NASDAQ:NVDA',
    META: 'NASDAQ:META',
  };
  const NSE_INDICES = {
    NIFTY50: 'NSE:NIFTY',
    BANKNIFTY: 'NSE:BANKNIFTY',
    SENSEX: 'BSE:SENSEX',
  };
  if (US_SYMBOLS[symbol]) return US_SYMBOLS[symbol];
  if (NSE_INDICES[symbol]) return NSE_INDICES[symbol];
  // Default NSE stock
  return `NSE:${symbol}`;
};

const StockChart = ({ stock, onClose }) => {
  const containerRef = useRef(null);
  const widgetRef = useRef(null);

  useEffect(() => {
    if (!stock || !containerRef.current) return;

    const tvSymbol = toTradingViewSymbol(stock.symbol, stock.exchange);
    const container = containerRef.current;

    // Remove previous widget if any
    container.innerHTML = '';

    const script = document.createElement('script');
    script.src = 'https://s3.tradingview.com/tv.js';
    script.async = true;
    script.onload = () => {
      if (window.TradingView && container) {
        widgetRef.current = new window.TradingView.widget({
          autosize: true,
          symbol: tvSymbol,
          interval: 'D',
          timezone: 'Asia/Kolkata',
          theme: 'dark',
          style: '1', // Candlestick
          locale: 'en',
          toolbar_bg: '#111111',
          enable_publishing: false,
          allow_symbol_change: true,
          container_id: container.id,
          hide_side_toolbar: false,
          studies: ['RSI@tv-basicstudies'],
        });
      }
    };

    container.appendChild(script);

    return () => {
      container.innerHTML = '';
    };
  }, [stock]);

  if (!stock) return null;

  const tvSymbol = toTradingViewSymbol(stock.symbol, stock.exchange);
  const tvLink = `https://www.tradingview.com/chart/?symbol=${encodeURIComponent(tvSymbol)}`;

  return (
    <div className="stock-chart-overlay" onClick={onClose}>
      <div className="stock-chart-modal" onClick={e => e.stopPropagation()}>
        <div className="stock-chart-header">
          <div className="stock-chart-title">
            <span className="chart-symbol">{stock.symbol}</span>
            <span className="chart-name">{stock.name}</span>
          </div>
          <div className="stock-chart-actions">
            <a
              href={tvLink}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-secondary btn-sm"
              title="Open full chart on TradingView"
            >
              <ExternalLink size={14} />
              Full Chart
            </a>
            <button className="btn btn-secondary btn-sm" onClick={onClose} title="Close chart">
              <X size={14} />
            </button>
          </div>
        </div>
        <div
          id="tradingview-widget-container"
          ref={containerRef}
          className="tradingview-container"
        />
      </div>
    </div>
  );
};

export default StockChart;
