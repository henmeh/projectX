import React, { createContext, useState, useContext } from 'react';

const AppContext = createContext();

export const AppProvider = ({ children }) => {
  const [darkMode, setDarkMode] = useState(true);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [loginVisible, setLoginVisible] = useState(false);
  const [activeView, setActiveView] = useState('whale-transactions');
  
  const toggleDarkMode = () => setDarkMode(!darkMode);
  const toggleSidebar = () => setSidebarCollapsed(!sidebarCollapsed);
  const changeView = (view) => setActiveView(view);
  
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
      changeView
    }}>
      {children}
    </AppContext.Provider>
  );
};

export const useAppContext = () => useContext(AppContext);