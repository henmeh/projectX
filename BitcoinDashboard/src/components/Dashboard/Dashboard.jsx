import { Layout } from 'antd';
import { useAppContext } from '../../context/AppContext';
import './Dashboard.css';
import FeeStatistics from './FeeStatistics/FeeStatistics';
import Mempool from './Mempool./Mempool';


const { Content } = Layout;

const Dashboard = () => {
  const { activeView } = useAppContext();
  
  const renderActiveView = () => {
    switch(activeView) {
      case 'mempool':
        return <Mempool />;
      case 'fee-statistics':
        return <FeeStatistics />;
      default:
        return <Mempool />;
    }
  };
  
  return (
    <Content className="dashboard-content">
      {renderActiveView()}
    </Content>
  );
};

export default Dashboard;