/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/ui/**/*.{js,jsx,ts,tsx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Menlo', 'monospace'],
      },
      colors: {
        terminal: {
          bg:     '#0d1117',
          panel:  '#161b22',
          border: '#30363d',
          dim:    '#8b949e',
          text:   '#e6edf3',
          green:  '#3fb950',
          yellow: '#d29922',
          red:    '#f85149',
          blue:   '#58a6ff',
          purple: '#bc8cff',
          cyan:   '#39d353',
        }
      },
      animation: {
        'blink': 'blink 1s step-end infinite',
        'pulse-slow': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        blink: { '0%, 100%': { opacity: 1 }, '50%': { opacity: 0 } }
      }
    }
  },
  plugins: []
}
