import React from 'react';
import { Layout } from 'antd';
import WhaleTransactions from './WhaleTransactions/WhaleTransactions';
import { useAppContext } from '../../context/AppContext';
import './Dashboard.css';
import MempoolInsights from './MempoolInsights/MempoolInsights';
import FeeStatistics from './FeeStatistics/FeeStatistics';


const { Content } = Layout;

const Dashboard = () => {
  const { activeView } = useAppContext();
  
  const renderActiveView = () => {
    switch(activeView) {
      case 'whale-transactions':
        return <WhaleTransactions />;
      case 'fee-histogram':
        return <MempoolInsights />;
      case 'fee-statistics':
        return <FeeStatistics />;
      default:
        return <WhaleTransactions />;
    }
  };
  
  return (
    <Content className="dashboard-content">
      {renderActiveView()}
    </Content>
  );
};

export default Dashboard;