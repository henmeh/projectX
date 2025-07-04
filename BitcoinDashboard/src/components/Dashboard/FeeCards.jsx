import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Typography } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined } from '@ant-design/icons';
import { fetchFeeEstimation } from '../../services/api';

const { Title } = Typography;

const FeeCards = () => {
  const [fees, setFees] = useState(null);
  const [previousFees, setPreviousFees] = useState(null);
  
  useEffect(() => {
    const loadData = async () => {
      const data = await fetchFeeEstimation();
      if (data) {
        setPreviousFees(fees);
        setFees(data);
      }
    };
    
    loadData();
    
    // Refresh every 30 seconds
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, []);
  
  const getTrend = (current, previous, feeType) => {
    if (!previous || !current) return null;
    
    const currentValue = current[feeType];
    const previousValue = previous[feeType];
    
    if (currentValue > previousValue) {
      return <ArrowUpOutlined style={{ color: '#ff4d4f' }} />;
    } else if (currentValue < previousValue) {
      return <ArrowDownOutlined style={{ color: '#52c41a' }} />;
    }
    return null;
  };
  
  return (
    <Row gutter={16}>
      <Col span={8}>
        <Card className="fee-card low-fee">
          <Statistic
            title="Low Fee"
            value={fees?.low_fee || 'N/A'}
            suffix="sat/vB"
            precision={1}
          />
          <div className="fee-trend">
            {getTrend(fees, previousFees, 'low_fee')}
          </div>
        </Card>
      </Col>
      <Col span={8}>
        <Card className="fee-card medium-fee">
          <Statistic
            title="Medium Fee"
            value={fees?.medium_fee || 'N/A'}
            suffix="sat/vB"
            precision={1}
          />
          <div className="fee-trend">
            {getTrend(fees, previousFees, 'medium_fee')}
          </div>
        </Card>
      </Col>
      <Col span={8}>
        <Card className="fee-card high-fee">
          <Statistic
            title="High Fee"
            value={fees?.fast_fee || 'N/A'}
            suffix="sat/vB"
            precision={1}
          />
          <div className="fee-trend">
            {getTrend(fees, previousFees, 'fast_fee')}
          </div>
        </Card>
      </Col>
    </Row>
  );
};

export default FeeCards;