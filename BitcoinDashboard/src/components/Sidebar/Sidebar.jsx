import React from 'react';
import { Layout, Menu } from 'antd';
import { 
  LineChartOutlined, 
  BarChartOutlined, 
  DollarOutlined, 
  RocketOutlined, 
  HeatMapOutlined 
} from '@ant-design/icons';
import { useAppContext } from '../../context/AppContext';
import './Sidebar.css';

const { Sider } = Layout;

const Sidebar = () => {
  const { darkMode, sidebarCollapsed, activeView, changeView } = useAppContext();
  
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
    { 
      key: 'fee-cards', 
      icon: <DollarOutlined />, 
      label: 'Current Fees' 
    },
    { 
      key: 'fee-predictions', 
      icon: <LineChartOutlined />, 
      label: 'Fee Predictions' 
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
  
  return (
    <Sider 
      collapsible
      collapsed={sidebarCollapsed}
      className={`sidebar ${darkMode ? 'dark' : 'light'}`}
      trigger={null}
    >
      <div className="logo">B</div>
      <Menu
        theme={darkMode ? "dark" : "light"}
        mode="inline"
        selectedKeys={[activeView]}
        onClick={handleMenuClick}
        items={menuItems}
      />
    </Sider>
  );
};

export default Sidebar;