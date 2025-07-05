import React from 'react';
import { Layout, Menu, Button } from 'antd';
import { 
  LineChartOutlined, 
  BarChartOutlined, 
  DollarOutlined, 
  RocketOutlined, 
  HeatMapOutlined,
  CloseOutlined
} from '@ant-design/icons';
import { useAppContext } from '../../context/AppContext';
import './Sidebar.css';

const { Sider } = Layout;

const Sidebar = () => {
  const { 
    darkMode, 
    sidebarCollapsed, 
    activeView, 
    changeView,
    isMobile,
    mobileSidebarVisible,
    setMobileSidebarVisible
  } = useAppContext();
  
  const menuItems = [
    { 
      key: 'whale-transactions', 
      icon: <DollarOutlined />, 
      label: 'Whale Transactions' 
    },
    { 
      key: 'fee-histogram', 
      icon: <BarChartOutlined />, 
      label: 'Fee Histogram' 
    },
    //{ 
    //  key: 'fee-cards', 
    //  icon: <DollarOutlined />, 
    //  label: 'Current Fees' 
    //},
    { 
      key: 'fee-predictions', 
      icon: <RocketOutlined />, 
      label: 'Fee Statistics' 
    },
    { 
      key: 'fee-pressure', 
      icon: <HeatMapOutlined />, 
      label: 'Fee Pressure Map' 
    },
  ];
  
  const handleMenuClick = (e) => {
    changeView(e.key);
  };
  
  // Mobile sidebar styles
  const mobileSidebarStyle = {
    position: 'fixed',
    top: 0,
    left: 0,
    height: '100vh',
    zIndex: 1000,
    transform: mobileSidebarVisible ? 'translateX(0)' : 'translateX(-100%)',
    transition: 'transform 0.3s ease-in-out',
    boxShadow: '2px 0 8px rgba(0, 0, 0, 0.15)'
  };
  
  return (
    <Sider 
      collapsible
      collapsed={sidebarCollapsed}
      className={`sidebar ${darkMode ? 'dark' : 'light'}`}
      trigger={null}
      style={isMobile ? mobileSidebarStyle : {}}
      width={isMobile ? '80%' : undefined}
    >
      {isMobile && (
        <div className="mobile-sidebar-header">
          <Button 
            type="text" 
            icon={<CloseOutlined />}
            onClick={() => setMobileSidebarVisible(false)}
            className="close-sidebar-btn"
          />
        </div>
      )}
      
      <div className="logo">B</div>
      <Menu
        mode="inline"
        selectedKeys={[activeView]}
        onClick={handleMenuClick}
        items={menuItems}
      />
    </Sider>
  );
};

export default Sidebar;