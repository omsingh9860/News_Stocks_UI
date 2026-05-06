import React, { useState } from 'react';
import { MessageSquare, Send, X, CheckCircle } from 'lucide-react';

const FEEDBACK_KEY = 'user_feedback_v1';

const FeedbackForm = ({ onClose }) => {
  const [form, setForm] = useState({ category: 'general', subject: '', message: '', rating: 0 });
  const [submitted, setSubmitted] = useState(false);
  const [hoveredRating, setHoveredRating] = useState(0);

  const handleChange = (e) => {
    setForm(prev => ({ ...prev, [e.target.name]: e.target.value }));
  };

  const handleRating = (val) => {
    setForm(prev => ({ ...prev, rating: val }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!form.message.trim()) return;

    const entry = {
      ...form,
      submittedAt: new Date().toISOString(),
      id: Date.now(),
    };

    // Save to localStorage
    try {
      const existing = JSON.parse(localStorage.getItem(FEEDBACK_KEY) || '[]');
      localStorage.setItem(FEEDBACK_KEY, JSON.stringify([entry, ...existing].slice(0, 50)));
    } catch (e) {
      console.warn('Could not save feedback to localStorage', e);
    }

    // Log to console (Phase 1 — no backend storage yet)
    console.log('[Feedback Submitted]', entry);

    setSubmitted(true);
  };

  return (
    <div className="feedback-overlay" onClick={onClose}>
      <div className="feedback-modal" onClick={e => e.stopPropagation()}>
        <div className="feedback-header">
          <h3 className="feedback-title">
            <MessageSquare size={20} />
            Share Your Feedback
          </h3>
          <button className="btn btn-secondary btn-sm" onClick={onClose}>
            <X size={14} />
          </button>
        </div>

        {submitted ? (
          <div className="feedback-success">
            <CheckCircle size={48} />
            <h4>Thank you for your feedback!</h4>
            <p>Your response helps us improve. We'll incorporate your suggestions in future updates.</p>
            <button className="btn btn-primary" onClick={onClose}>
              Close
            </button>
          </div>
        ) : (
          <form className="feedback-form" onSubmit={handleSubmit}>
            {/* Star Rating */}
            <div className="feedback-field">
              <label className="feedback-label">Overall Experience</label>
              <div className="star-rating">
                {[1, 2, 3, 4, 5].map(star => (
                  <button
                    key={star}
                    type="button"
                    className={`star-btn ${star <= (hoveredRating || form.rating) ? 'active' : ''}`}
                    onClick={() => handleRating(star)}
                    onMouseEnter={() => setHoveredRating(star)}
                    onMouseLeave={() => setHoveredRating(0)}
                  >
                    ★
                  </button>
                ))}
                {form.rating > 0 && (
                  <span className="star-label">
                    {['', 'Poor', 'Fair', 'Good', 'Very Good', 'Excellent'][form.rating]}
                  </span>
                )}
              </div>
            </div>

            {/* Category */}
            <div className="feedback-field">
              <label className="feedback-label">Category</label>
              <select
                name="category"
                className="filter-select feedback-select"
                value={form.category}
                onChange={handleChange}
              >
                <option value="general">General Feedback</option>
                <option value="bug">Bug Report</option>
                <option value="feature">Feature Request</option>
                <option value="ui">UI / Design</option>
                <option value="data">Data Accuracy</option>
              </select>
            </div>

            {/* Subject */}
            <div className="feedback-field">
              <label className="feedback-label">Subject (optional)</label>
              <input
                type="text"
                name="subject"
                className="search-input feedback-input"
                placeholder="Brief subject..."
                value={form.subject}
                onChange={handleChange}
                maxLength={100}
              />
            </div>

            {/* Message */}
            <div className="feedback-field">
              <label className="feedback-label">Message *</label>
              <textarea
                name="message"
                className="feedback-textarea"
                placeholder="Tell us what you think, what's missing, or any issues you've encountered..."
                value={form.message}
                onChange={handleChange}
                rows={4}
                required
                maxLength={1000}
              />
              <span className="feedback-char-count">{form.message.length}/1000</span>
            </div>

            <button
              type="submit"
              className="btn btn-primary feedback-submit"
              disabled={!form.message.trim()}
            >
              <Send size={16} />
              Submit Feedback
            </button>
          </form>
        )}
      </div>
    </div>
  );
};

export default FeedbackForm;
