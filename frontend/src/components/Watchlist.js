import React, { useState, useMemo } from 'react';
import { Plus, X, Star, Search, ChevronUp } from 'lucide-react';

// Available stocks: NSE blue-chips + select US indices/stocks
export const AVAILABLE_STOCKS = [
  // NSE Stocks
  { symbol: 'RELIANCE', name: 'Reliance Industries', exchange: 'NSE' },
  { symbol: 'TCS', name: 'Tata Consultancy Services', exchange: 'NSE' },
  { symbol: 'INFY', name: 'Infosys', exchange: 'NSE' },
  { symbol: 'HDFCBANK', name: 'HDFC Bank', exchange: 'NSE' },
  { symbol: 'WIPRO', name: 'Wipro', exchange: 'NSE' },
  { symbol: 'BHARTIARTL', name: 'Bharti Airtel', exchange: 'NSE' },
  { symbol: 'ITC', name: 'ITC', exchange: 'NSE' },
  { symbol: 'SBIN', name: 'State Bank of India', exchange: 'NSE' },
  { symbol: 'LT', name: 'Larsen & Toubro', exchange: 'NSE' },
  { symbol: 'HCLTECH', name: 'HCL Technologies', exchange: 'NSE' },
  { symbol: 'AXISBANK', name: 'Axis Bank', exchange: 'NSE' },
  { symbol: 'MARUTI', name: 'Maruti Suzuki', exchange: 'NSE' },
  { symbol: 'BAJFINANCE', name: 'Bajaj Finance', exchange: 'NSE' },
  { symbol: 'ASIANPAINT', name: 'Asian Paints', exchange: 'NSE' },
  { symbol: 'HINDUNILVR', name: 'Hindustan Unilever', exchange: 'NSE' },
  { symbol: 'TITAN', name: 'Titan Company', exchange: 'NSE' },
  { symbol: 'NESTLEIND', name: 'Nestle India', exchange: 'NSE' },
  { symbol: 'ADANIENT', name: 'Adani Enterprises', exchange: 'NSE' },
  { symbol: 'TATAMOTORS', name: 'Tata Motors', exchange: 'NSE' },
  { symbol: 'NTPC', name: 'NTPC', exchange: 'NSE' },
  { symbol: 'SUNPHARMA', name: 'Sun Pharmaceutical', exchange: 'NSE' },
  { symbol: 'DRREDDY', name: "Dr. Reddy's Laboratories", exchange: 'NSE' },
  { symbol: 'TECHM', name: 'Tech Mahindra', exchange: 'NSE' },
  { symbol: 'ULTRACEMCO', name: 'UltraTech Cement', exchange: 'NSE' },
  { symbol: 'CIPLA', name: 'Cipla', exchange: 'NSE' },
  { symbol: 'JSWSTEEL', name: 'JSW Steel', exchange: 'NSE' },
  { symbol: 'TATASTEEL', name: 'Tata Steel', exchange: 'NSE' },
  { symbol: 'COALINDIA', name: 'Coal India', exchange: 'NSE' },
  { symbol: 'POWERGRID', name: 'Power Grid Corporation', exchange: 'NSE' },
  { symbol: 'NIFTY50', name: 'Nifty 50 Index', exchange: 'NSE' },
  { symbol: 'BANKNIFTY', name: 'Bank Nifty Index', exchange: 'NSE' },
  { symbol: 'SENSEX', name: 'BSE Sensex Index', exchange: 'BSE' },
  // US Indices & Stocks
  { symbol: 'SPX', name: 'S&P 500 Index', exchange: 'US' },
  { symbol: 'DJI', name: 'Dow Jones Industrial Average', exchange: 'US' },
  { symbol: 'IXIC', name: 'NASDAQ Composite', exchange: 'US' },
  { symbol: 'AAPL', name: 'Apple Inc.', exchange: 'NASDAQ' },
  { symbol: 'MSFT', name: 'Microsoft Corporation', exchange: 'NASDAQ' },
  { symbol: 'GOOGL', name: 'Alphabet Inc.', exchange: 'NASDAQ' },
  { symbol: 'AMZN', name: 'Amazon.com Inc.', exchange: 'NASDAQ' },
  { symbol: 'TSLA', name: 'Tesla Inc.', exchange: 'NASDAQ' },
  { symbol: 'NVDA', name: 'NVIDIA Corporation', exchange: 'NASDAQ' },
  { symbol: 'META', name: 'Meta Platforms Inc.', exchange: 'NASDAQ' },
];

const EXCHANGE_COLORS = {
  NSE: 'var(--accent-blue)',
  BSE: 'var(--accent-purple)',
  NASDAQ: 'var(--accent-green)',
  US: 'var(--accent-orange)',
};

const Watchlist = ({ watchlist, onAdd, onRemove, isInWatchlist, onSelectStock }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [showAddPanel, setShowAddPanel] = useState(false);
  const [filterExchange, setFilterExchange] = useState('all');

  const filteredAvailable = useMemo(() => {
    return AVAILABLE_STOCKS.filter(stock => {
      const matchesSearch =
        stock.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
        stock.name.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesExchange = filterExchange === 'all' || stock.exchange === filterExchange;
      return matchesSearch && matchesExchange;
    });
  }, [searchQuery, filterExchange]);

  return (
    <section className="watchlist-section">
      <div className="section-header">
        <h2 className="section-title">
          <Star size={20} />
          My Watchlist
          {watchlist.length > 0 && (
            <span className="watchlist-count">{watchlist.length}</span>
          )}
        </h2>
        <button
          className={`btn btn-secondary watchlist-toggle-btn ${showAddPanel ? 'active' : ''}`}
          onClick={() => setShowAddPanel(v => !v)}
        >
          {showAddPanel ? <ChevronUp size={16} /> : <Plus size={16} />}
          {showAddPanel ? 'Close' : 'Add Stocks'}
        </button>
      </div>

      {/* Add Stock Panel */}
      {showAddPanel && (
        <div className="add-stock-panel">
          <div className="add-stock-controls">
            <div className="search-container">
              <Search size={16} />
              <input
                type="text"
                className="search-input"
                placeholder="Search stocks by name or symbol..."
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
              />
            </div>
            <select
              className="filter-select"
              value={filterExchange}
              onChange={e => setFilterExchange(e.target.value)}
            >
              <option value="all">All Exchanges</option>
              <option value="NSE">NSE</option>
              <option value="BSE">BSE</option>
              <option value="NASDAQ">NASDAQ</option>
              <option value="US">US Indices</option>
            </select>
          </div>
          <div className="available-stocks-grid">
            {filteredAvailable.map(stock => {
              const inList = isInWatchlist(stock.symbol);
              return (
                <div key={stock.symbol} className="available-stock-item">
                  <div className="available-stock-info">
                    <span
                      className="exchange-badge"
                      style={{ backgroundColor: EXCHANGE_COLORS[stock.exchange] + '25', color: EXCHANGE_COLORS[stock.exchange] }}
                    >
                      {stock.exchange}
                    </span>
                    <span className="available-stock-symbol">{stock.symbol}</span>
                    <span className="available-stock-name">{stock.name}</span>
                  </div>
                  <button
                    className={`btn btn-sm ${inList ? 'btn-remove' : 'btn-add'}`}
                    onClick={() => inList ? onRemove(stock.symbol) : onAdd(stock)}
                    title={inList ? 'Remove from watchlist' : 'Add to watchlist'}
                  >
                    {inList ? <X size={14} /> : <Plus size={14} />}
                    {inList ? 'Remove' : 'Add'}
                  </button>
                </div>
              );
            })}
            {filteredAvailable.length === 0 && (
              <p className="empty-search-msg">No stocks found matching your search.</p>
            )}
          </div>
        </div>
      )}

      {/* Current Watchlist */}
      {watchlist.length === 0 ? (
        <div className="watchlist-empty">
          <Star size={40} />
          <p>Your watchlist is empty.</p>
          <p className="watchlist-hint">Add NSE or US stocks above to track them here and get personalized news.</p>
        </div>
      ) : (
        <div className="watchlist-grid">
          {watchlist.map(stock => (
            <div key={stock.symbol} className="watchlist-card" onClick={() => onSelectStock && onSelectStock(stock)}>
              <div className="watchlist-card-header">
                <span
                  className="exchange-badge"
                  style={{ backgroundColor: EXCHANGE_COLORS[stock.exchange] + '25', color: EXCHANGE_COLORS[stock.exchange] }}
                >
                  {stock.exchange}
                </span>
                <button
                  className="watchlist-remove-btn"
                  onClick={e => { e.stopPropagation(); onRemove(stock.symbol); }}
                  title="Remove from watchlist"
                >
                  <X size={14} />
                </button>
              </div>
              <div className="watchlist-card-body">
                <span className="watchlist-symbol">{stock.symbol}</span>
                <span className="watchlist-name">{stock.name}</span>
              </div>
              {onSelectStock && (
                <div className="watchlist-card-footer">
                  <span className="watchlist-chart-hint">Click to view chart</span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </section>
  );
};

export default Watchlist;
