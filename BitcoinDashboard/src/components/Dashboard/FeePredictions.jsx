import React, { useState, useEffect } from 'react';
import { Card, Typography, Row, Col, Statistic } from 'antd';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { fetchFeeEstimation, fetchFeePrediction } from '../../services/api';

const { Title } = Typography;

const FeePredictions = () => {
  const [currentFees, setCurrentFees] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [historicalData, setHistoricalData] = useState([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      
      // Fetch current fees
      const fees = await fetchFeeEstimation();
      setCurrentFees(fees);
      
      // Fetch prediction
      const pred = await fetchFeePrediction();
      setPrediction(pred);
      
      // Update historical data
      if (fees && pred) {
        const newData = [...historicalData];
        if (newData.length >= 20) newData.shift();
        
        newData.push({
          timestamp: new Date().toLocaleTimeString(),
          currentLow: fees.low_fee,
          currentMedium: fees.medium_fee,
          currentHigh: fees.fast_fee,
          predictedLow: pred.low_fee,
          predictedMedium: pred.medium_fee,
          predictedHigh: pred.fast_fee
        });
        
        setHistoricalData(newData);
      }
      
      setLoading(false);
    };
    
    loadData();
    
    // Refresh every minute
    const interval = setInterval(loadData, 60000);
    return () => clearInterval(interval);
  }, []);
  
  const getPredictionAccuracy = (current, predicted) => {
    if (!current || !predicted) return null;
    
    const accuracy = 100 - Math.abs((predicted - current) / current) * 100;
    return Math.max(0, Math.min(100, accuracy)).toFixed(1);
  };
  
  const renderTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="custom-tooltip">
          <p className="label">{label}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }}>
              {entry.name}: {entry.value.toFixed(2)} sat/vB
            </p>
          ))}
        </div>
      );
    }
    return null;
  };
  
  return (
    <Card 
      title={<Title level={4} style={{ margin: 0 }}>Fee Predictions</Title>}
      className="dashboard-card"
      loading={loading}
    >
      {prediction && currentFees && (
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col span={8}>
            <Card className="prediction-card">
              <Statistic
                title="Low Fee Prediction"
                value={prediction.low_fee}
                suffix="sat/vB"
                precision={1}
              />
              <div className="accuracy">
                Accuracy: {getPredictionAccuracy(currentFees.low_fee, prediction.low_fee)}%
              </div>
            </Card>
          </Col>
          <Col span={8}>
            <Card className="prediction-card">
              <Statistic
                title="Medium Fee Prediction"
                value={prediction.medium_fee}
                suffix="sat/vB"
                precision={1}
              />
              <div className="accuracy">
                Accuracy: {getPredictionAccuracy(currentFees.medium_fee, prediction.medium_fee)}%
              </div>
            </Card>
          </Col>
          <Col span={8}>
            <Card className="prediction-card">
              <Statistic
                title="High Fee Prediction"
                value={prediction.fast_fee}
                suffix="sat/vB"
                precision={1}
              />
              <div className="accuracy">
                Accuracy: {getPredictionAccuracy(currentFees.fast_fee, prediction.fast_fee)}%
              </div>
            </Card>
          </Col>
        </Row>
      )}
      
      <ResponsiveContainer width="100%" height={300}>
        <LineChart
          data={historicalData}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timestamp" />
          <YAxis label={{ value: 'Fee (sat/vB)', angle: -90, position: 'insideLeft' }} />
          <Tooltip content={renderTooltip} />
          <Line 
            type="monotone" 
            dataKey="currentLow" 
            name="Current Low" 
            stroke="#52c41a" 
            dot={false} 
          />
          <Line 
            type="monotone" 
            dataKey="predictedLow" 
            name="Predicted Low" 
            stroke="#52c41a" 
            strokeDasharray="3 3" 
            dot={false} 
          />
          <Line 
            type="monotone" 
            dataKey="currentMedium" 
            name="Current Medium" 
            stroke="#1890ff" 
            dot={false} 
          />
          <Line 
            type="monotone" 
            dataKey="predictedMedium" 
            name="Predicted Medium" 
            stroke="#1890ff" 
            strokeDasharray="3 3" 
            dot={false} 
          />
          <Line 
            type="monotone" 
            dataKey="currentHigh" 
            name="Current High" 
            stroke="#ff4d4f" 
            dot={false} 
          />
          <Line 
            type="monotone" 
            dataKey="predictedHigh" 
            name="Predicted High" 
            stroke="#ff4d4f" 
            strokeDasharray="3 3" 
            dot={false} 
          />
        </LineChart>
      </ResponsiveContainer>
    </Card>
  );
};

export default FeePredictions;