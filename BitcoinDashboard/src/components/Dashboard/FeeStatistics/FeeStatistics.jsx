import { Card, Tabs } from 'antd';
import { PieChartOutlined, ClockCircleOutlined } from '@ant-design/icons';
import FeePredictions from "../FeePredictions/FeePredictions";
import HistoricalFeeHeatmap from "../HistoricalFeeHeatmap/HistoricalFeeHeatmap";

const FeeStatistics = () => {
  const items = [
    {
      key: '1',
      label: <><PieChartOutlined /> Current Market Fees / Fee Predictions</>,
      children: <FeePredictions />,
    },
    {
      key: '2',
      label: <><ClockCircleOutlined /> Historical Patterns</>,
      children: <HistoricalFeeHeatmap />,
    },
  ];

  return (
    <Card className="dashboard-card">
      <Tabs defaultActiveKey="1" items={items} />
    </Card>
  );
};

export default FeeStatistics;