import React, { useState, useEffect } from 'react';
import { Card, Typography, Row, Col, Statistic, Tag, Progress, Skeleton, Alert, Tabs, Radio } from 'antd';
import { fetchMempoolCongestion, fetchFeeHistogram } from '../../../services/api';
import "./CurrentMempoolVisualizer.css"
import "../Dashboard.css";
import DataCard from '../../DataCard/DataCard.jsx';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

const CurrentMempoolVisualizer = () => {
  const [congestionStatus, setCongestionStatus] = useState(null);
  const [mempoolBlocks, setMempoolBlocks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const BLOCK_VSIZE_LIMIT = 1000000; // 1 MB in vBytes

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      setError(null);
      try {
        const [congestionData, histogramData] = await Promise.all([
          fetchMempoolCongestion(),
          fetchFeeHistogram()
        ]);
        setCongestionStatus(congestionData);

        if (!histogramData?.histogram || !Array.isArray(histogramData.histogram)) {
          console.warn("Fee histogram data is missing or malformed, setting empty array.");
          setMempoolBlocks([]);
          return;
        }

        const sortedFeeLevels = histogramData.histogram
          .map(([fee, v_size]) => ({ fee: parseFloat(fee), v_size: parseInt(v_size) }))
          .sort((a, b) => b.fee - a.fee);

        const blocks = [];
        let currentBlock = { v_size: 0, fees: [] };

        for (const level of sortedFeeLevels) {
          if (currentBlock.v_size + level.v_size > BLOCK_VSIZE_LIMIT && currentBlock.v_size > 0) {
            blocks.push(currentBlock);
            currentBlock = { v_size: 0, fees: [] };
          }
          currentBlock.v_size += level.v_size;
          currentBlock.fees.push(level.fee);
        }
        if (currentBlock.v_size > 0) {
          blocks.push(currentBlock);
        }
        setMempoolBlocks(blocks);

      } catch (err) {
        console.error("Error loading current mempool data:", err);
        setError("Could not load current mempool data. Please try again later.");
        setCongestionStatus(null);
        setMempoolBlocks([]);
      } finally {
        setLoading(false);
      }
    };

    loadData();
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status) => {
    if (!status) return 'default';
    const lowerStatus = status.toLowerCase();
    if (lowerStatus === 'high') return 'error';
    if (lowerStatus === 'medium') return 'warning';
    return 'success';
  };

  const getBlockFeeRange = (fees) => {
    if (!fees || fees.length === 0) return 'N/A';
    const min = Math.min(...fees);
    const max = Math.max(...fees);
    return min === max ? `${min.toFixed(0)} sat/vB` : `${min.toFixed(0)} - ${max.toFixed(0)} sat/vB`;
  };

  return (
    <Skeleton loading={loading} active paragraph={{ rows: 8 }}>
      {error && <Alert message="Error" description={error} type="error" showIcon style={{ marginBottom: 16 }} />}
      {congestionStatus && (
        <Row gutter={[24, 24]} style={{ marginTop: 24, marginBottom: 24 }} align="stretch">
          <Col xs={24} lg={12}>
            <DataCard className="data-card" title="Mempool Status" data={<Tag color={getStatusColor(congestionStatus.congestion_status)}>{congestionStatus.congestion_status || 'Unknown'}</Tag>} />
          </Col>
          <Col xs={24} lg={12}>
            <DataCard className="data-card" title="Total vSize" data={`${(congestionStatus.total_vsize / 1000000).toFixed(2)} MB`} />
          </Col>
        </Row>
      )}
      <Card
        title={<Title level={4} style={{ margin: 0 }}>Mempool Blocks</Title>}
        className="dashboard-card"
      >
        <Text type="secondary">A real-time view of data waiting in the mempool, organized into 1MB blocks.</Text>
        <div className="mempool-blocks-container">
          {mempoolBlocks.map((block, index) => {
            const percentage = Math.min((block.v_size / BLOCK_VSIZE_LIMIT) * 100, 100);
            return (
              <div key={index} className="mempool-block">
               <div className="block-header">
                  <Text strong>{index === 0 ? 'Next Block' : `Block #${index + 1}`}</Text>
                  <Text type="secondary">{getBlockFeeRange(block.fees)}</Text>
                </div>
                <Progress percent={percentage} strokeColor={percentage > 95 ? '#ff4d4f' : percentage > 70 ? '#faad14' : '#52c41a'} format={() => `${(block.v_size / 1000000).toFixed(2)} MB`} />
              </div>
            );
          })}
          {!mempoolBlocks.length && !loading && !error && <Text type="secondary" style={{ display: 'block', textAlign: 'center', marginTop: 32 }}>Mempool is currently empty.</Text>}
        </div>
      </Card>
    </Skeleton>
  );
};

export default CurrentMempoolVisualizer
