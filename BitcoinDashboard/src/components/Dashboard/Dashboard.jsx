import React from 'react';
import { Layout } from 'antd';
import WhaleTransactions from './WhaleTransactions';
import FeeHistogram from './FeeHistogram';
import FeeCards from './FeeCards';
import FeePredictions from './FeePredictions';
import FeePressureMap from './FeePressureMap';
import { useAppContext } from '../../context/AppContext';

const { Content } = Layout;

const Dashboard = () => {
  const { activeView } = useAppContext();
  
  const renderActiveView = () => {
    switch(activeView) {
      case 'whale-transactions':
        return <WhaleTransactions />;
      case 'fee-histogram':
        return <FeeHistogram />;
      case 'fee-cards':
        return <FeeCards />;
      case 'fee-predictions':
        return <FeePredictions />;
      case 'fee-pressure':
        return <FeePressureMap />;
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