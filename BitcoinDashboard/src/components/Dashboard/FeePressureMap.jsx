import React, { useState, useEffect, useMemo } from 'react';
import { Card, Typography, Row, Col, Statistic, Tag, Progress, Skeleton, Alert, Tabs, Segmented } from 'antd';
import { ClockCircleOutlined, PieChartOutlined } from '@ant-design/icons';
import { Heatmap } from '@ant-design/plots';

// Import API functions
import { fetchMempoolCongestion, fetchFeeHistogram, fetchHistoricalFeeHeatmap } from '../../services/api';

// IMPORT THE EXTERNAL DATA PREPARATION FUNCTION. NO JAVASCRIPT CODE BLOCKS HERE.
import { prepareHeatmapData } from '../../utils/dataPreparation'; // <<<--- ADJUST THIS PATH TO MATCH YOUR PROJECT STRUCTURE

const { Title, Text } = Typography;
const { TabPane } = Tabs;


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

        const sortedFeeLevels = histogramData.histogram
          .map(([fee, v_size]) => ({ fee: parseFloat(fee), v_size }))
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


const HistoricalFeeHeatmap = () => {
  const [heatmapData, setHeatmapData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeRange, setTimeRange] = useState('Last 7 Days');

  useEffect(() => {
    const loadHistoricalData = async () => {
      setLoading(true);
      setError(null);
      let daysToFetch = 7;
      if (timeRange === 'Last 30 Days') {
        daysToFetch = 30;
      }

      try {
        const rawData = await fetchHistoricalFeeHeatmap(daysToFetch);

        if (Array.isArray(rawData) && rawData.length > 0) {
          const processedData = prepareHeatmapData(rawData);
          setHeatmapData(processedData);
        } else {
          setHeatmapData([]);
          setError("No historical fee data available.");
        }
      } catch (err) {
        console.error("Failed to load historical fee heatmap:", err);
        setError("Could not load historical fee data.");
      } finally {
        setLoading(false);
      }
    };

    loadHistoricalData();
  }, [timeRange]);

  // Calculate overall statistics for the heatmap
  const heatmapStats = useMemo(() => {
    if (heatmapData.length === 0) return null;
    
    const nonZeroData = heatmapData.filter(d => d.avg_fee > 0);
    if (nonZeroData.length === 0) return null;
    
    const fees = nonZeroData.map(d => d.avg_fee);
    const minFee = Math.min(...fees);
    const maxFee = Math.max(...fees);
    const totalFees = fees.reduce((sum, fee) => sum + fee, 0);
    const avgFee = totalFees / fees.length;
    
    return { minFee, maxFee, avgFee };
  }, [heatmapData]);

  // Enhanced heatmap configuration
  const config = {
    data: heatmapData,
    xField: 'hour',
    yField: 'day',
    colorField: 'avg_fee',
    color: ({ avg_fee }) => {
      if (avg_fee === 0) return '#f0f0f0'; // Light grey for zero/no data
      if (avg_fee < 10) return '#52c41a'; // Low fee (Green)
      if (avg_fee < 30) return '#faad14'; // Medium fee (Yellow/Orange)
      return '#ff4d4f'; // High fee (Red)
    },
    meta: {
      day: {
        type: 'cat',
        values: ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
        alias: 'Day'
      },
      hour: {
        type: 'cat',
        alias: 'Hour (UTC)'
      },
      avg_fee: {
        alias: 'Avg Fee',
        formatter: (val) => val === 0 ? 'No data' : `${val.toFixed(1)} sat/vB`,
      },
    },
    tooltip: {
      title: (title, datum) => {
        const day = datum?.day || '';
        const hour = datum?.hour !== undefined ? String(datum.hour).padStart(2, '0') : '';
        return `${day}, ${hour}:00 UTC`;
      },
      formatter: (datum) => {
        const fee = datum.avg_fee;
        return {
          name: 'Average Fee',
          value: fee === 0 ? 'No data available' : `${fee.toFixed(1)} sat/vB`
        };
      },
      customContent: (title, items) => {
        if (!items || items.length === 0) return null;
        
        const item = items[0];
        const fee = item.value;
        let status = '';
        let tip = '';
        
        if (fee === 0) {
          status = 'No data';
          tip = 'No transaction data available for this time period';
        } else if (fee < 10) {
          status = 'Low fee period';
          tip = 'Best time for cost-effective transactions';
        } else if (fee < 30) {
          status = 'Medium fee period';
          tip = 'Average network fee conditions';
        } else {
          status = 'High fee period';
          tip = 'Consider waiting for lower fee opportunities';
        }
        
        return (
          <div style={{ 
            background: '#fff', 
            padding: '12px',
            border: '1px solid #ddd',
            borderRadius: '4px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
          }}>
            <div style={{ fontWeight: 600, marginBottom: 8 }}>{title}</div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
              <span>Fee Rate:</span>
              <span style={{ fontWeight: 500 }}>{fee}</span>
            </div>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between',
              padding: '4px 8px',
              background: fee < 10 ? '#f6ffed' : fee < 30 ? '#fffbe6' : '#fff2f0',
              borderRadius: '4px',
              marginBottom: 8
            }}>
              <span>Status:</span>
              <span style={{ fontWeight: 500, color: fee < 10 ? '#52c41a' : fee < 30 ? '#faad14' : '#ff4d4f' }}>
                {status}
              </span>
            </div>
            <div style={{ fontSize: 12, color: '#666' }}>{tip}</div>
          </div>
        );
      }
    },
    xAxis: {
      title: { text: 'Hour of Day (UTC)', style: { fontSize: 12 } },
      label: {
        autoHide: false,
        autoRotate: false,
        formatter: (val) => String(val).padStart(2, '0'),
        style: { fontSize: 10 }
      },
    },
    yAxis: {
      title: { text: 'Day of Week', style: { fontSize: 12 } },
      label: { style: { fontSize: 10 } },
    },
    label: {
      offset: -2,
      style: {
        fill: '#fff',
        fontSize: 9,
        shadowBlur: 2,
        shadowColor: 'rgba(0, 0, 0, 0.45)',
      },
      formatter: (datum) => datum.avg_fee > 0 ? datum.avg_fee.toFixed(0) : '',
    },
    interactions: [{ type: 'element-active' }],
  };

  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div>
          <Title level={5} style={{ margin: 0 }}>Network Fee Hotspots</Title>
          <Text type="secondary">
            Visualize when fees are typically lowest for cost-effective transactions
          </Text>
        </div>
        <Segmented 
          options={['Last 7 Days', 'Last 30 Days']} 
          value={timeRange} 
          onChange={setTimeRange} 
        />
      </div>
      
      {heatmapStats && (
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={8}>
            <Card size="small" hoverable>
              <Statistic
                title="Lowest Observed Fee"
                value={heatmapStats.minFee.toFixed(1)}
                suffix="sat/vB"
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small" hoverable>
              <Statistic
                title="Average Fee"
                value={heatmapStats.avgFee.toFixed(1)}
                suffix="sat/vB"
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small" hoverable>
              <Statistic
                title="Peak Fee"
                value={heatmapStats.maxFee.toFixed(1)}
                suffix="sat/vB"
                valueStyle={{ color: '#ff4d4f' }}
              />
            </Card>
          </Col>
        </Row>
      )}
      
      <Skeleton loading={loading} active paragraph={{ rows: 8 }}>
        {error && <Alert message="Error" description={error} type="error" showIcon style={{ marginBottom: 16 }} />}
        
        {heatmapData.length > 0 ? (
          <div style={{ position: 'relative' }}>
            <div style={{ height: 350 }}>
              <Heatmap {...config} />
            </div>
            
            {/* Custom Legend */}
            <div style={{ 
              display: 'flex', 
              justifyContent: 'center', 
              marginTop: 16,
              flexWrap: 'wrap',
              gap: 16
            }}>
              {[
                { color: '#f0f0f0', label: 'No data', description: 'No transactions recorded' },
                { color: '#52c41a', label: 'Low fee (<10 sat/vB)', description: 'Ideal for transactions' },
                { color: '#faad14', label: 'Medium fee (10-30 sat/vB)', description: 'Average network conditions' },
                { color: '#ff4d4f', label: 'High fee (≥30 sat/vB)', description: 'Consider waiting if possible' }
              ].map((item, index) => (
                <div key={index} style={{ display: 'flex', alignItems: 'center' }}>
                  <div style={{
                    width: 16,
                    height: 16,
                    backgroundColor: item.color,
                    marginRight: 8,
                    border: '1px solid #ddd'
                  }} />
                  <div>
                    <Text strong style={{ fontSize: 12 }}>{item.label}</Text>
                    <Text type="secondary" style={{ display: 'block', fontSize: 11 }}>{item.description}</Text>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          !loading && !error && (
            <div style={{ 
              display: 'flex', 
              flexDirection: 'column', 
              alignItems: 'center', 
              justifyContent: 'center', 
              height: 350,
              background: '#fafafa',
              borderRadius: 4,
              border: '1px dashed #ddd',
              marginTop: 16
            }}>
              <PieChartOutlined style={{ fontSize: 48, color: '#bfbfbf', marginBottom: 16 }} />
              <Text type="secondary">No historical data available for the selected period</Text>
              <Text type="secondary" style={{ marginTop: 8 }}>Try selecting a different time range</Text>
            </div>
          )
        )}
      </Skeleton>
      
      {heatmapData.length > 0 && (
        <div style={{ marginTop: 16, padding: 12, background: '#f6ffed', borderRadius: 4 }}>
          <Text>
            <strong>Tip:</strong> For standard transactions, aim for the green periods where fees are typically below 10 sat/vB. 
            High-value transactions may prioritize security over fee savings.
          </Text>
        </div>
      )}
    </>
  );
};

const CombinedFeeView = () => (
  <Card>
    <Tabs defaultActiveKey="1">
      <TabPane tab={<><PieChartOutlined />Current Mempool</>} key="1">
        <CurrentMempoolVisualizer />
      </TabPane>
      <TabPane tab={<><ClockCircleOutlined />Historical Patterns</>} key="2">
        <HistoricalFeeHeatmap />
      </TabPane>
    </Tabs>
  </Card>
);

export default CombinedFeeView;