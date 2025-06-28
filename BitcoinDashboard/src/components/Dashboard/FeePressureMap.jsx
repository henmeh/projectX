import React, { useState, useEffect } from 'react';
import { Card, Typography, Row, Col, Segmented } from 'antd';
import { fetchMempoolCongestion } from '../../services/api';
import { Heatmap } from '@ant-design/plots';

const { Title } = Typography;

const FeePressureMap = () => {
  const [congestion, setCongestion] = useState(null);
  const [heatmapData, setHeatmapData] = useState([]);
  const [timeRange, setTimeRange] = useState('1h');
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      const data = await fetchMempoolCongestion();
      setCongestion(data);
      
      // Generate mock heatmap data (in a real app, this would come from API)
      const mockData = [];
      for (let i = 0; i < 100; i++) {
        mockData.push({
          x: Math.floor(Math.random() * 24),
          y: Math.floor(Math.random() * 10),
          value: Math.random() * 100,
          id: `point-${i}`
        });
      }
      setHeatmapData(mockData);
      setLoading(false);
    };
    
    loadData();
    
    // Refresh every 30 seconds
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, [timeRange]);
  
  const config = {
    data: heatmapData,
    xField: 'x',
    yField: 'y',
    colorField: 'value',
    legend: {
      position: 'right',
    },
    color: ['#174c83', '#7eb6d4', '#efefeb', '#efa759', '#ff4d4f'],
    tooltip: {
      formatter: (datum) => {
        return { name: 'Fee Pressure', value: datum.value.toFixed(2) };
      },
    },
    xAxis: {
      title: {
        text: 'Time',
      },
    },
    yAxis: {
      title: {
        text: 'Fee Level',
      },
    },
  };
  
  return (
    <Card 
      title={<Title level={4} style={{ margin: 0 }}>Fee Pressure Map</Title>}
      className="dashboard-card"
      loading={loading}
      extra={
        <Segmented
          options={['1h', '6h', '24h']}
          value={timeRange}
          onChange={setTimeRange}
        />
      }
    >
      {congestion && (
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col span={24}>
            <div className={`congestion-indicator ${congestion.congestion_status.toLowerCase()}`}>
              <div className="congestion-label">
                Mempool Status: <strong>{congestion.congestion_status}</strong>
              </div>
              <div className="congestion-stats">
                Total vSize: {(congestion.total_vsize / 1000000).toFixed(2)} MB
              </div>
            </div>
          </Col>
        </Row>
      )}
      
      <Heatmap {...config} />
    </Card>
  );
};

export default FeePressureMap;