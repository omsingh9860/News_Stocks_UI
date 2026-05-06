import React from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

/**
 * Formats a numeric price value using the en-IN locale.
 * @param {number} price
 * @returns {string}
 */
export const formatPrice = (price) =>
  new Intl.NumberFormat('en-IN', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(price);

/**
 * Formats a Date object as a time string (HH:MM:SS) in the en-IN locale.
 * @param {Date} date
 * @returns {string}
 */
export const formatTime = (date) =>
  new Intl.DateTimeFormat('en-IN', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).format(date);

/**
 * Formats a Date object as a locale date string in en-IN.
 * @param {Date} date
 * @returns {string}
 */
export const formatDate = (date) =>
  new Intl.DateTimeFormat('en-IN', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  }).format(date);

/**
 * Returns a hex colour string corresponding to a sentiment label.
 * @param {'positive'|'negative'|'neutral'} sentiment
 * @returns {string}
 */
export const getSentimentColor = (sentiment) => {
  switch (sentiment) {
    case 'positive': return '#10b981';
    case 'negative': return '#ef4444';
    default: return '#6b7280';
  }
};

/**
 * Returns a Lucide icon element for the given sentiment.
 * @param {'positive'|'negative'|'neutral'} sentiment
 * @param {number} [size=16]
 * @returns {JSX.Element}
 */
export const getSentimentIcon = (sentiment, size = 16) => {
  switch (sentiment) {
    case 'positive': return <TrendingUp size={size} />;
    case 'negative': return <TrendingDown size={size} />;
    default: return <Minus size={size} />;
  }
};

/**
 * Returns a hex colour string for a trading signal label.
 * @param {string} signal - e.g. 'BUY', 'SELL', 'EDUCATIONAL'
 * @returns {string}
 */
export const getSignalColor = (signal) => {
  switch (signal?.toUpperCase()) {
    case 'BUY': return '#10b981';
    case 'SELL': return '#ef4444';
    default: return '#3b82f6';
  }
};
