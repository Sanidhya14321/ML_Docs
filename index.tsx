
import React from 'react';
import ReactDOM from 'react-dom/client';
import './src/styles/globals.css';
import App from './App';
import { ThemeProvider } from './contexts/ThemeContext';
import { CourseProvider } from './contexts/CourseContext';

window.addEventListener('error', (e) => {
  const errDiv = document.createElement('div');
  errDiv.style.position = 'fixed';
  errDiv.style.top = '0';
  errDiv.style.left = '0';
  errDiv.style.zIndex = '9999';
  errDiv.style.background = 'red';
  errDiv.style.color = 'white';
  errDiv.style.padding = '20px';
  errDiv.style.whiteSpace = 'pre-wrap';
  errDiv.innerText = `Error: ${e.message}\n\nStack: ${e.error?.stack}`;
  document.body.appendChild(errDiv);
});

window.addEventListener('unhandledrejection', (e) => {
  const errDiv = document.createElement('div');
  errDiv.style.position = 'fixed';
  errDiv.style.top = '0';
  errDiv.style.left = '0';
  errDiv.style.zIndex = '9999';
  errDiv.style.background = 'orange';
  errDiv.style.color = 'white';
  errDiv.style.padding = '20px';
  errDiv.style.whiteSpace = 'pre-wrap';
  errDiv.innerText = `Unhandled Rejection: ${e.reason?.message}\n\nStack: ${e.reason?.stack}`;
  document.body.appendChild(errDiv);
});

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error("Could not find root element to mount to");
}

const root = ReactDOM.createRoot(rootElement);
root.render(
  <React.StrictMode>
    <ThemeProvider>
      <CourseProvider>
        <App />
      </CourseProvider>
    </ThemeProvider>
  </React.StrictMode>
);
