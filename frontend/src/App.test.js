import { render, screen } from '@testing-library/react';
import App from './App';
import ThemeProvider from './context/ThemeContext';

test('renders the market insights dashboard header', () => {
  render(
    <ThemeProvider>
      <App />
    </ThemeProvider>
  );
  // The h1 heading should always be present
  const heading = screen.getByRole('heading', { level: 1 });
  expect(heading).toBeInTheDocument();
});
