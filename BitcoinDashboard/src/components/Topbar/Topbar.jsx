import React, { useState } from 'react';
import { Layout, Button, Avatar, Dropdown, Menu } from 'antd';
import { 
  MenuFoldOutlined, 
  MenuUnfoldOutlined, 
  UserOutlined, 
  LoginOutlined,
  LogoutOutlined
} from '@ant-design/icons';
import { useAppContext } from '../../context/AppContext';
import './Topbar.css';

const { Header } = Layout;

const Topbar = () => {
  const { 
    darkMode, 
    toggleSidebar, 
    sidebarCollapsed, 
    isLoggedIn, 
    setLoginVisible 
  } = useAppContext();
  
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  
  const handleLogin = () => {
    setLoginVisible(true);
    setUserMenuOpen(false);
  };
  
  const handleLogout = () => {
    setUserMenuOpen(false);
  };
  
  const menuItems = [
    {
      key: 'login',
      icon: <LoginOutlined />,
      label: 'Login',
      onClick: handleLogin,
      style: { display: isLoggedIn ? 'none' : 'block' }
    },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: 'Logout',
      onClick: handleLogout,
      style: { display: isLoggedIn ? 'block' : 'none' }
    }
  ];

  return (
    <Header className={`topbar ${darkMode ? 'dark' : 'light'}`}>
      <div className="topbar-left">
        <Button
          type="text"
          icon={sidebarCollapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
          onClick={toggleSidebar}
          className="menu-toggle"
        />
        <div className="app-name">Bitcoin Analytics Dashboard</div>
      </div>
      
      <div className="topbar-right">
        <Dropdown 
          menu={{ items: menuItems }}
          open={userMenuOpen}
          onOpenChange={setUserMenuOpen}
          trigger={['click']}
        >
          <Button type="text" className="user-menu">
            <Avatar icon={<UserOutlined />} />
          </Button>
        </Dropdown>
      </div>
    </Header>
  );
};

export default Topbar;