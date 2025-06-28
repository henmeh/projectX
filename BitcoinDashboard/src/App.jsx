import React from 'react';
import { Layout, ConfigProvider, theme } from 'antd';
import { useAppContext } from './context/AppContext';
import Topbar from './components/Topbar/Topbar';
import Sidebar from './components/Sidebar/Sidebar';
import Dashboard from './components/Dashboard/Dashboard';
import LoginModal from './components/Auth/LoginModal';
//import ErrorBoundary from './components/ErrorBoundary';
import './App.css';

const { darkAlgorithm } = theme;

function App() {
  const { darkMode, isMobile, mobileSidebarVisible, setMobileSidebarVisible } = useAppContext();
  
  return (
      <ConfigProvider
        theme={{
          algorithm: darkMode ? darkAlgorithm : theme.defaultAlgorithm,
          token: {
            colorPrimary: '#1890ff',
            borderRadius: 4,
            colorBgContainer: darkMode ? '#1f1f1f' : '#ffffff',
          },
        }}
      >
        <Layout className="app-layout">
          {!isMobile && <Sidebar />}
          {isMobile && mobileSidebarVisible && (
            <div 
              className="sidebar-backdrop"
              onClick={() => setMobileSidebarVisible(false)}
            />
          )}
          {isMobile && <Sidebar />}
          
          <Layout>
            <Topbar />
            <Dashboard />
          </Layout>
          <LoginModal />
        </Layout>
      </ConfigProvider>
  );
}

export default App;