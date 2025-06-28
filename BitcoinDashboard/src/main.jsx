import React from 'react'
import ReactDOM from 'react-dom/client'
import { AppProvider } from './context/AppContext' // Import provider
import App from './App'
import 'antd/dist/reset.css'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AppProvider> {/* Wrap App with provider */}
      <App />
    </AppProvider>
  </React.StrictMode>,
)