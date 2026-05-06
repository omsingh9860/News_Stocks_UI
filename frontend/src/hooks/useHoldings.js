import { useState, useEffect, useCallback } from 'react';

const HOLDINGS_KEY = 'portfolio_holdings_v1';

export const useHoldings = () => {
  const [holdings, setHoldings] = useState(() => {
    try {
      const saved = localStorage.getItem(HOLDINGS_KEY);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(HOLDINGS_KEY, JSON.stringify(holdings));
    } catch (e) {
      console.warn('Failed to save holdings to localStorage', e);
    }
  }, [holdings]);

  const addHolding = useCallback((holding) => {
    setHoldings(prev => {
      const existing = prev.findIndex(h => h.symbol.toUpperCase() === holding.symbol.toUpperCase());
      if (existing >= 0) {
        const updated = [...prev];
        updated[existing] = { ...holding, symbol: holding.symbol.toUpperCase(), updatedAt: new Date().toISOString() };
        return updated;
      }
      return [...prev, { ...holding, symbol: holding.symbol.toUpperCase(), addedAt: new Date().toISOString() }];
    });
  }, []);

  const updateHolding = useCallback((symbol, updates) => {
    setHoldings(prev =>
      prev.map(h =>
        h.symbol === symbol.toUpperCase()
          ? { ...h, ...updates, symbol: symbol.toUpperCase(), updatedAt: new Date().toISOString() }
          : h
      )
    );
  }, []);

  const removeHolding = useCallback((symbol) => {
    setHoldings(prev => prev.filter(h => h.symbol !== symbol.toUpperCase()));
  }, []);

  const updateCurrentPrice = useCallback((symbol, price) => {
    setHoldings(prev =>
      prev.map(h =>
        h.symbol === symbol.toUpperCase()
          ? { ...h, manualCurrentPrice: parseFloat(price) || null, priceUpdatedAt: new Date().toISOString() }
          : h
      )
    );
  }, []);

  return { holdings, addHolding, updateHolding, removeHolding, updateCurrentPrice };
};
