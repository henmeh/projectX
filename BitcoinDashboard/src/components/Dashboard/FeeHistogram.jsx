import React, { useState, useEffect } from 'react';
import { Card, Typography } from 'antd';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { fetchFeeHistogram } from '../../services/api';

const { Title } = Typography;

const FeeHistogram = () => {
  const [histogramData, setHistogramData] = useState([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      const data = await fetchFeeHistogram();
      if (data && data.histogram) {
        // Transform data for visualization with unique keys
        const transformed = data.histogram
          .filter(item => item[1] > 0)
          .map(([fee, count], index) => ({ 
            fee, 
            count,
            id: `${fee}-${index}`  // Add unique key
          }));
        setHistogramData(transformed);
      }
      setLoading(false);
    };
    
    loadData();
    
    // Refresh every 30 seconds
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, []);
  
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="custom-tooltip">
          <p className="label">{`${payload[0].payload.fee} sat/vB`}</p>
          <p className="intro">{`Transactions: ${payload[0].value.toLocaleString()}`}</p>
        </div>
      );
    }
    return null;
  };
  
  return (
    <Card 
      title={<Title level={4} style={{ margin: 0 }}>Fee Distribution Histogram</Title>}
      className="dashboard-card"
      loading={loading}
    >
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={histogramData}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="fee" 
            label={{ value: 'Fee (sat/vB)', position: 'insideBottom', offset: -5 }} 
          />
          <YAxis 
            label={{ value: 'Transactions', angle: -90, position: 'insideLeft' }} 
          />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="count" fill="#1890ff" name="Transaction Count" />
        </BarChart>
      </ResponsiveContainer>
    </Card>
  );
};

export default FeeHistogram;