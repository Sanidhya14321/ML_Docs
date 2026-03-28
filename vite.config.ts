import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import fs from 'fs';

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    {
      name: 'error-logger',
      configureServer(server) {
        server.middlewares.use('/api/log_error', (req, res) => {
          let body = '';
          req.on('data', chunk => body += chunk);
          req.on('end', () => {
            fs.appendFileSync('error.log', body + '\n');
            res.end('ok');
          });
        });
      }
    }
  ],
  build: {
    outDir: 'dist',
  }
});