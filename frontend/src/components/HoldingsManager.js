import React, { useState } from 'react';
import { Plus, Edit2, Trash2, Save, X, Briefcase, DollarSign, AlertCircle } from 'lucide-react';
import { formatPrice } from '../utils/helpers';

const EMPTY_FORM = {
  symbol: '',
  qty: '',
  buyPrice: '',
  buyDate: '',
  notes: '',
  manualCurrentPrice: '',
};

const FormField = ({ label, children, hint }) => (
  <div className="hm-field">
    <label className="hm-label">{label}</label>
    {children}
    {hint && <span className="hm-hint">{hint}</span>}
  </div>
);

const HoldingsManager = ({ holdings, onAdd, onUpdate, onRemove, onUpdatePrice }) => {
  const [showForm, setShowForm] = useState(false);
  const [editingSymbol, setEditingSymbol] = useState(null);
  const [form, setForm] = useState(EMPTY_FORM);
  const [priceInputs, setPriceInputs] = useState({});
  const [errors, setErrors] = useState({});

  const resetForm = () => {
    setForm(EMPTY_FORM);
    setErrors({});
    setEditingSymbol(null);
    setShowForm(false);
  };

  const validate = () => {
    const errs = {};
    if (!form.symbol.trim()) errs.symbol = 'Symbol is required';
    if (!form.qty || isNaN(form.qty) || Number(form.qty) <= 0) errs.qty = 'Enter a valid quantity > 0';
    if (!form.buyPrice || isNaN(form.buyPrice) || Number(form.buyPrice) <= 0) errs.buyPrice = 'Enter a valid buy price > 0';
    if (form.manualCurrentPrice && (isNaN(form.manualCurrentPrice) || Number(form.manualCurrentPrice) < 0)) {
      errs.manualCurrentPrice = 'Enter a valid price';
    }
    return errs;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const errs = validate();
    if (Object.keys(errs).length) { setErrors(errs); return; }

    const holding = {
      symbol: form.symbol.trim().toUpperCase(),
      qty: parseFloat(form.qty),
      buyPrice: parseFloat(form.buyPrice),
      buyDate: form.buyDate || null,
      notes: form.notes.trim() || null,
      manualCurrentPrice: form.manualCurrentPrice ? parseFloat(form.manualCurrentPrice) : null,
    };

    if (editingSymbol) {
      onUpdate(editingSymbol, holding);
    } else {
      onAdd(holding);
    }
    resetForm();
  };

  const handleEdit = (h) => {
    setForm({
      symbol: h.symbol,
      qty: String(h.qty),
      buyPrice: String(h.buyPrice),
      buyDate: h.buyDate || '',
      notes: h.notes || '',
      manualCurrentPrice: h.manualCurrentPrice != null ? String(h.manualCurrentPrice) : '',
    });
    setEditingSymbol(h.symbol);
    setShowForm(true);
  };

  const handlePriceInputChange = (symbol, value) => {
    setPriceInputs(prev => ({ ...prev, [symbol]: value }));
  };

  const handlePriceSave = (symbol) => {
    const val = priceInputs[symbol];
    if (val && !isNaN(val) && parseFloat(val) > 0) {
      onUpdatePrice(symbol, val);
      setPriceInputs(prev => ({ ...prev, [symbol]: '' }));
    }
  };

  const fieldChange = (field) => (e) => {
    setForm(prev => ({ ...prev, [field]: e.target.value }));
    if (errors[field]) setErrors(prev => { const n = { ...prev }; delete n[field]; return n; });
  };

  return (
    <div className="holdings-manager">
      <div className="holdings-header">
        <h2 className="section-title">
          <Briefcase size={20} /> My Holdings
        </h2>
        {!showForm && (
          <button className="btn btn-primary btn-sm" onClick={() => setShowForm(true)}>
            <Plus size={16} /> Add Holding
          </button>
        )}
      </div>

      {/* Add / Edit Form */}
      {showForm && (
        <div className="hm-form-card">
          <div className="hm-form-header">
            <span>{editingSymbol ? `Edit ${editingSymbol}` : 'Add New Holding'}</span>
            <button className="icon-btn" onClick={resetForm} title="Cancel">
              <X size={18} />
            </button>
          </div>
          <form className="hm-form" onSubmit={handleSubmit} noValidate>
            <div className="hm-form-grid">
              <FormField label="Symbol *">
                <input
                  className={`hm-input${errors.symbol ? ' hm-input-error' : ''}`}
                  placeholder="e.g. RELIANCE"
                  value={form.symbol}
                  onChange={fieldChange('symbol')}
                  disabled={!!editingSymbol}
                />
                {errors.symbol && <span className="hm-error">{errors.symbol}</span>}
              </FormField>

              <FormField label="Quantity *">
                <input
                  className={`hm-input${errors.qty ? ' hm-input-error' : ''}`}
                  type="number"
                  min="0"
                  step="any"
                  placeholder="e.g. 10"
                  value={form.qty}
                  onChange={fieldChange('qty')}
                />
                {errors.qty && <span className="hm-error">{errors.qty}</span>}
              </FormField>

              <FormField label="Avg Buy Price (₹) *">
                <input
                  className={`hm-input${errors.buyPrice ? ' hm-input-error' : ''}`}
                  type="number"
                  min="0"
                  step="any"
                  placeholder="e.g. 2450"
                  value={form.buyPrice}
                  onChange={fieldChange('buyPrice')}
                />
                {errors.buyPrice && <span className="hm-error">{errors.buyPrice}</span>}
              </FormField>

              <FormField label="Buy Date" hint="Optional">
                <input
                  className="hm-input"
                  type="date"
                  value={form.buyDate}
                  onChange={fieldChange('buyDate')}
                />
              </FormField>

              <FormField label="Manual Current Price (₹)" hint="Optional — used when live price is unavailable">
                <input
                  className={`hm-input${errors.manualCurrentPrice ? ' hm-input-error' : ''}`}
                  type="number"
                  min="0"
                  step="any"
                  placeholder="e.g. 2680"
                  value={form.manualCurrentPrice}
                  onChange={fieldChange('manualCurrentPrice')}
                />
                {errors.manualCurrentPrice && <span className="hm-error">{errors.manualCurrentPrice}</span>}
              </FormField>

              <FormField label="Notes" hint="Optional">
                <input
                  className="hm-input"
                  placeholder="e.g. Long-term hold"
                  value={form.notes}
                  onChange={fieldChange('notes')}
                />
              </FormField>
            </div>

            <div className="hm-form-actions">
              <button type="submit" className="btn btn-primary">
                <Save size={16} /> {editingSymbol ? 'Save Changes' : 'Add Holding'}
              </button>
              <button type="button" className="btn btn-secondary" onClick={resetForm}>
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Holdings List */}
      {holdings.length === 0 && !showForm ? (
        <div className="empty-state">
          <Briefcase size={48} />
          <p>No holdings added yet.</p>
          <p className="watchlist-hint">Click "Add Holding" to track your portfolio.</p>
        </div>
      ) : (
        holdings.length > 0 && (
          <div className="hm-list">
            {holdings.map(h => {
              const invested = h.qty * h.buyPrice;
              const currentPrice = h.manualCurrentPrice;
              const hasPrice = currentPrice != null;
              const currentValue = hasPrice ? h.qty * currentPrice : null;
              const pnl = hasPrice ? currentValue - invested : null;
              const pnlPct = hasPrice ? ((pnl / invested) * 100).toFixed(2) : null;
              const priceVal = priceInputs[h.symbol] !== undefined ? priceInputs[h.symbol] : '';

              return (
                <div key={h.symbol} className="hm-row">
                  <div className="hm-row-main">
                    <div className="hm-symbol">
                      <strong>{h.symbol}</strong>
                      {h.buyDate && <span className="hm-date">{h.buyDate}</span>}
                    </div>
                    <div className="hm-metrics">
                      <span className="hm-metric">
                        <span className="hm-metric-label">Qty</span>
                        <span className="hm-metric-value">{h.qty}</span>
                      </span>
                      <span className="hm-metric">
                        <span className="hm-metric-label">Buy</span>
                        <span className="hm-metric-value">₹{formatPrice(h.buyPrice)}</span>
                      </span>
                      <span className="hm-metric">
                        <span className="hm-metric-label">Invested</span>
                        <span className="hm-metric-value">₹{formatPrice(invested)}</span>
                      </span>
                      {hasPrice ? (
                        <>
                          <span className="hm-metric">
                            <span className="hm-metric-label">Current</span>
                            <span className="hm-metric-value">₹{formatPrice(currentPrice)}</span>
                          </span>
                          <span className="hm-metric">
                            <span className="hm-metric-label">Value</span>
                            <span className="hm-metric-value">₹{formatPrice(currentValue)}</span>
                          </span>
                          <span className="hm-metric">
                            <span className="hm-metric-label">P&L</span>
                            <span className="hm-metric-value" style={{ color: pnl >= 0 ? '#10b981' : '#ef4444' }}>
                              {pnl >= 0 ? '+' : ''}₹{formatPrice(pnl)} ({pnlPct}%)
                            </span>
                          </span>
                        </>
                      ) : (
                        <span className="hm-metric hm-no-price">
                          <AlertCircle size={13} />
                          <span>No current price</span>
                        </span>
                      )}
                    </div>
                    {h.notes && <div className="hm-notes">📝 {h.notes}</div>}
                  </div>

                  {/* Manual price override row */}
                  {!hasPrice && (
                    <div className="hm-price-row">
                      <DollarSign size={14} />
                      <input
                        className="hm-input hm-price-input"
                        type="number"
                        min="0"
                        step="any"
                        placeholder="Enter current price to calc P&L"
                        value={priceVal}
                        onChange={e => handlePriceInputChange(h.symbol, e.target.value)}
                        onKeyDown={e => e.key === 'Enter' && handlePriceSave(h.symbol)}
                      />
                      <button
                        className="btn btn-primary btn-sm"
                        onClick={() => handlePriceSave(h.symbol)}
                        disabled={!priceVal}
                      >
                        Set
                      </button>
                    </div>
                  )}

                  <div className="hm-row-actions">
                    <button className="icon-btn" title="Edit" onClick={() => handleEdit(h)}>
                      <Edit2 size={15} />
                    </button>
                    <button className="icon-btn icon-btn-danger" title="Remove" onClick={() => onRemove(h.symbol)}>
                      <Trash2 size={15} />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )
      )}
    </div>
  );
};

export default HoldingsManager;
