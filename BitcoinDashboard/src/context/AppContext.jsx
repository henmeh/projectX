import React, { createContext, useState, useContext, useEffect } from 'react';

const AppContext = createContext();

export const AppProvider = ({ children }) => {
  const [darkMode, setDarkMode] = useState(true);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [loginVisible, setLoginVisible] = useState(false);
  const [activeView, setActiveView] = useState('whale-transactions');
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  const [mobileSidebarVisible, setMobileSidebarVisible] = useState(false);
  
  // Handle screen resize
  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      
      // Auto-collapse sidebar on mobile
      if (mobile) {
        setSidebarCollapsed(true);
      } else {
        setMobileSidebarVisible(false);
      }
    };
    
    window.addEventListener('resize', handleResize);
    handleResize(); // Initial check
    
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  const toggleDarkMode = () => setDarkMode(!darkMode);
  const toggleSidebar = () => {
    if (isMobile) {
      setMobileSidebarVisible(!mobileSidebarVisible);
    } else {
      setSidebarCollapsed(!sidebarCollapsed);
    }
  };
  
  const changeView = (view) => {
    setActiveView(view);
    if (isMobile) {
      setMobileSidebarVisible(false);
    }
  };
  
  return (
    <AppContext.Provider value={{
      darkMode,
      toggleDarkMode,
      sidebarCollapsed,
      toggleSidebar,
      isLoggedIn,
      setIsLoggedIn,
      loginVisible,
      setLoginVisible,
      activeView,
      changeView,
      isMobile,
      mobileSidebarVisible,
      setMobileSidebarVisible
    }}>
      {children}
    </AppContext.Provider>
  );
};

export const useAppContext = () => useContext(AppContext);