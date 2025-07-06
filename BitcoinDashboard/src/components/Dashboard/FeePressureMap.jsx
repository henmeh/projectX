import React, { useState, useEffect, useMemo } from 'react';
import { Card, Typography, Row, Col, Statistic, Tag, Progress, Skeleton, Alert, Tabs, Segmented } from 'antd';
import { ClockCircleOutlined, PieChartOutlined } from '@ant-design/icons';
import { Heatmap } from '@ant-design/plots';
import { ResponsiveHeatMap } from '@nivo/heatmap'


// Import API functions
import { fetchMempoolCongestion, fetchFeeHistogram, fetchHistoricalFeeHeatmap } from '../../services/api';

const { Title, Text } = Typography;
const { TabPane } = Tabs;


// This component is unchanged, as requested.
const CurrentMempoolVisualizer = () => {
  const [congestionStatus, setCongestionStatus] = useState(null);
  const [mempoolBlocks, setMempoolBlocks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const BLOCK_VSIZE_LIMIT = 1000000;

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
        if (!histogramData?.histogram) throw new Error("Fee histogram data missing.");
        const sortedFeeLevels = histogramData.histogram.map(([fee, v_size]) => ({ fee: parseFloat(fee), v_size })).sort((a, b) => b.fee - a.fee);
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
        if (currentBlock.v_size > 0) blocks.push(currentBlock);
        setMempoolBlocks(blocks);
      } catch (err) {
        console.log(err)
        setError("Could not load current mempool data.");
      } finally {
        setLoading(false);
      }
    };
    loadData();
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status) => (status?.toLowerCase() === 'high' ? 'error' : status?.toLowerCase() === 'medium' ? 'warning' : 'success');
  const getBlockFeeRange = (fees) => {
    if (!fees.length) return 'N/A';
    const min = Math.min(...fees);
    const max = Math.max(...fees);
    return min === max ? `${min.toFixed(0)} sat/vB` : `${min.toFixed(0)} - ${max.toFixed(0)} sat/vB`;
  };

  return (
    <Skeleton loading={loading} active paragraph={{ rows: 8 }}>
      {error && <Alert message="Error" description={error} type="error" showIcon style={{ marginBottom: 16 }} />}
      {congestionStatus && (
        <Row gutter={16} style={{ marginBottom: 24 }} align="middle">
          <Col xs={24} sm={8}>
            <Statistic title="Mempool Status" value=" " prefix={<Tag color={getStatusColor(congestionStatus.congestion_status)}>{congestionStatus.congestion_status || 'Unknown'}</Tag>} />
          </Col>
          <Col xs={24} sm={8}>
            <Statistic title="Total vSize" value={(congestionStatus.total_vsize / 1000000).toFixed(2)} suffix="MB" />
          </Col>
        </Row>
      )}
      <Title level={5}>Mempool Blocks</Title>
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
    </Skeleton>
  );
};


// --- ✨ NEW AND IMPROVED HistoricalFeeHeatmap component ---
// --- CORRECTED HistoricalFeeHeatmap component ---
const HistoricalFeeHeatmap = () => {
  const [heatmapData, setHeatmapData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeRange, setTimeRange] = useState('Last 7 Days');

  useEffect(() => {
    const loadHistoricalData = async () => {
      setLoading(true);
      setError(null);
      try {
        const daysToFetch = timeRange === 'Last 30 Days' ? 30 : 7;
        const rawData = await fetchHistoricalFeeHeatmap(daysToFetch);

        if (!Array.isArray(rawData) || rawData.length === 0) {
          throw new Error("No historical data returned from API.");
        }
        setHeatmapData(rawData);

      } catch (err) {
        console.error("Failed to load historical fee heatmap:", err);
        setError("Could not load historical fee data. Please try again later.");
        setHeatmapData([]); // Clear old data on error
      } finally {
        setLoading(false);
      }
    };
    loadHistoricalData();
  }, [timeRange]);

  const heatmapStats = useMemo(() => {
    if (!heatmapData || heatmapData.length === 0) return null;
    const fees = heatmapData.filter(d => d.avg_fee > 0).map(d => d.avg_fee);
    if (fees.length === 0) return null;
    return {
      minFee: Math.min(...fees),
      maxFee: Math.max(...fees),
      avgFee: fees.reduce((sum, fee) => sum + fee, 0) / fees.length,
    };
  }, [heatmapData]);

const config = {
    data: heatmapData,
    xField: 'hour',    
    yField: 'day',     
    colorField: 'avg_fee',
    legend: {},
    mark: 'cell',
    axis: {
      x:{
        title: 'Hour of the day',
      },
      y:{
        title: 'Day',
      }
    },
  };

  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24, flexWrap: 'wrap', gap: '16px'}}>
        <div>
          <Title level={5} style={{ margin: 0 }}>Network Fee Hotspots </Title>
          <Text type="secondary">Find the best time to send by seeing when fees are typically high or low.</Text>
        </div>
        <Segmented options={['Last 7 Days', 'Last 30 Days']} value={timeRange} onChange={setTimeRange} />
      </div>

      {loading && <Skeleton active paragraph={{ rows: 10 }} />}
      
      {!loading && error && <Alert message="Error Loading Data" description={error} type="error" showIcon />}
      
      {!loading && !error && heatmapData.length > 0 && (
        <>
          {heatmapStats && (
            <Row gutter={16} style={{ marginBottom: 24 }}>
              <Col xs={24} sm={8}><Card size="small"><Statistic title="Lowest Fee" value={heatmapStats.minFee.toFixed(1)} suffix="sat/vB" valueStyle={{ color: '#38A169' }} /></Card></Col>
              <Col xs={24} sm={8}><Card size="small"><Statistic title="Average Fee" value={heatmapStats.avgFee.toFixed(1)} suffix="sat/vB" /></Card></Col>
              <Col xs={24} sm={8}><Card size="small"><Statistic title="Peak Fee" value={heatmapStats.maxFee.toFixed(1)} suffix="sat/vB" valueStyle={{ color: '#E53E3E' }} /></Card></Col>
            </Row>
          )}

          <div style={{ height: 350, position: 'relative' , background: 'white'}}>
            <Heatmap {...config} />
          </div>

          <div style={{ textAlign: 'center', marginTop: 16 }}>
            <Text type="secondary">
              Hover over a block to see the average fee for that hour.
            </Text>
          </div>
        </>
      )}
      
       {!loading && !error && heatmapData.length === 0 && (
        <div style={{ textAlign: 'center', padding: '48px 0' }}>
            <PieChartOutlined style={{ fontSize: 48, color: '#bfbfbf' }} />
            <Title level={5} style={{ marginTop: 16 }}>No Historical Data Available</Title>
            <Text type="secondary">There is no fee data to display for the selected time period.</Text>
        </div>
       )}
    </>
  );
};

// Main Component: The Combined View with the corrected Tabs API
const CombinedFeeView = () => {
  const items = [
    {
      key: '1',
      label: <><PieChartOutlined /> Current Mempool</>,
      children: <CurrentMempoolVisualizer />,
    },
    {
      key: '2',
      label: <><ClockCircleOutlined /> Historical Patterns</>,
      children: <HistoricalFeeHeatmap />,
    },
  ];

  return (
    <Card>
      <Tabs defaultActiveKey="1" items={items} />
    </Card>
  );
};

export default CombinedFeeView;