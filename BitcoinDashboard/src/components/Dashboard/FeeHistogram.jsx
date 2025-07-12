import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Card, Typography, Row, Col, Statistic, Alert, Skeleton } from 'antd';
import { BarChart, Bar, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { AreaChartOutlined, BarChartOutlined, ClockCircleOutlined } from '@ant-design/icons';
import "./Dashboard.css"

// Importing from your services/api.js file
import { fetchFeeEstimation, fetchFeeHistogram, fetchMempoolInsights } from '../../services/api';

const { Title, Text } = Typography;

// Helper to format timestamps
const formatFullTimestamp = (isoString) => {
  if (!isoString) return 'N/A';
  try {
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric', month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
    }).format(new Date(isoString));
  } catch (e) {
    console.log(e)
    return 'Invalid Date';
  }
};

const MempoolOverview = () => {
  // --- STATE MANAGEMENT ---
  const [feeHistogramData, setFeeHistogramData] = useState([]);
  const [mempoolInsightsData, setMempoolInsightsData] = useState([]);
  //const [feeEstimation, setFeeEstimation] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  // --- DATA FETCHING ---
  const loadAllData = useCallback(async () => {
    // On subsequent refreshes, don't show the main loader, just update data
    if (!loading) setLoading(true); 
    setError(null);

    try {
      // ✨ NEW: Fetch all data in parallel for faster loading
      const [histogramRes, insightsRes, _] = await Promise.all([
        fetchFeeHistogram(),
        fetchMempoolInsights(),
        fetchFeeEstimation()
      ]);

      // Process Fee Histogram
      if (histogramRes?.histogram) {
        const transformed = histogramRes.histogram
          .filter(item => item[1] > 0)
          .map(([fee, count]) => ({ fee: parseFloat(fee), count }));
        setFeeHistogramData(transformed);
      }

      // Process Mempool Insights
      if (insightsRes?.length > 0) {
        const transformed = insightsRes.map(item => ({
          ...item,
          total_vsize_bytes: parseFloat(item.total_vsize_bytes),
          avg_fee_per_vbyte: parseFloat(item.avg_fee_per_vbyte),
        }));
        setMempoolInsightsData(transformed);
        setLastUpdated(transformed[0].generated_at);
      }

      // Process Fee Estimation
      //setFeeEstimation(estimationRes);

    } catch (err) {
      console.error("Failed to load dashboard data:", err);
      setError("Could not fetch fresh data. Displaying last known data if available.");
    } finally {
      setLoading(false);
    }
  }, []); // The callback itself doesn't have dependencies

  useEffect(() => {
    loadAllData();
    const interval = setInterval(loadAllData, 30000);
    return () => clearInterval(interval);
  }, [loadAllData]);

  // --- MEMOIZED CALCULATIONS ---
  const summaryStats = useMemo(() => {
    const totalVsize = mempoolInsightsData.reduce((sum, item) => sum + item.total_vsize_bytes, 0);
    const totalTransactions = mempoolInsightsData.reduce((sum, item) => sum + item.transaction_count, 0);
    return { totalVsize, totalTransactions };
  }, [mempoolInsightsData]);

  const totalHistogramTransactions = useMemo(() => {
    return feeHistogramData.reduce((sum, item) => sum + item.count, 0);
  }, [feeHistogramData]);

  // --- CUSTOM CHART TOOLTIPS ---
  const FeeHistogramTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const percentage = totalHistogramTransactions > 0 ? (data.count / totalHistogramTransactions * 100).toFixed(2) : 0;
      return (
        <div>
          <Text strong>{`${data.fee} sat/vB`}</Text><br />
          <Text>Transactions: {data.count.toLocaleString()}</Text><br />
          <Text type="secondary">{percentage}% of mempool Txs</Text>
        </div>
      );
    }
    return null;
  };

  const InsightsTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const vsizeData = payload.find(p => p.dataKey === 'total_vsize_bytes');
      const feeData = payload.find(p => p.dataKey === 'avg_fee_per_vbyte');

      return (
        <div>
          <Text strong>Value Sent: {label}</Text><br />
          {vsizeData && <Text style={{ color: vsizeData.color }}>Total VSize: {(vsizeData.value / 1_000_000).toFixed(2)} MB</Text>}<br />
          {feeData && <Text style={{ color: feeData.color }}>Avg. Fee: {feeData.value.toFixed(2)} sat/vB</Text>}
        </div>
      );
    }
    return null;
  };

  // --- RENDER LOGIC ---
  return (
    <Card 
      title={<Title level={4} style={{ margin: 0 }}>Mempool Insights</Title>}
      className="dashboard-card"
      loading={loading}
    >
      {lastUpdated && <Text type="secondary"><ClockCircleOutlined style={{ marginRight: 8 }} />Last Updated: {formatFullTimestamp(lastUpdated)}</Text>}
      
      {error && <Alert message="Network Error" description={error} type="warning" showIcon closable style={{ marginTop: 16 }} />}

      <Row gutter={[24, 24]} style={{ marginTop: 24, marginBottom: 24 }} align="stretch">
        <Col xs={24} lg={12}>
          <Card title="Total Mempool Size" className="data-card">
            <Statistic value={(summaryStats.totalVsize / 1000000).toFixed(2)} suffix="MB" />
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="Transactions in Mempool" className="data-card">
            <Statistic value={summaryStats.totalTransactions.toLocaleString()} />
          </Card>
        </Col>
        {/*
        <Col xs={24} lg={8} style={{ display: 'flex' }}>
        <Card title="Recommended Fees (sat/vB)" className="data-card">
          <Statistic
            value={feeEstimation ? `${feeEstimation.fast_fee.toFixed(0)}` : '--'}
            suffix={feeEstimation ? ` (Fast) / ${feeEstimation.medium_fee.toFixed(0)} (Med) / ${feeEstimation.low_fee.toFixed(0)} (Low)`  : ''}
          />
        </Card>
        </Col>
        */}
      </Row>

      <Row gutter={[24, 24]}>
        <Col xs={24} lg={12}>
          <Card title={<><BarChartOutlined /> Mempool Fee Rate Distribution</>}>
            <Skeleton loading={loading} active paragraph={{ rows: 6 }}>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={feeHistogramData} margin={{ top: 5, right: 20, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="fee" label={{ value: 'Fee (sat/vB)', position: 'insideBottom', offset: -10 }} minTickGap={25} />
                  <YAxis label={{ value: 'Transactions', angle: -90, position: 'insideLeft' }} tickFormatter={(val) => `${(val / 1000).toFixed(0)}k`} />
                  <Tooltip content={<FeeHistogramTooltip />} />
                  <Bar dataKey="count" fill="#1890ff" name="Transaction Count" />
                </BarChart>
              </ResponsiveContainer>
            </Skeleton>
          </Card>
        </Col>

        {/* ✨ NICER DISPLAY: Combined Chart for Insights */}
        <Col xs={24} lg={12}>
          <Card title={<><AreaChartOutlined /> Mempool Insights by Value Sent</>}>
            <Skeleton loading={loading} active paragraph={{ rows: 6 }}>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={mempoolInsightsData} margin={{ top: 5, right: 5, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="amount_range" angle={-45} textAnchor="end" height={70} interval={0} />
                  
                  {/* Define two separate Y-Axes */}
                  <YAxis yAxisId="left" orientation="left" stroke="#8884d8" label={{ value: 'Total VSize (MB)', angle: -90, position: 'insideLeft' }} tickFormatter={(val) => (val / 1_000_000).toFixed(1)} />
                  <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" label={{ value: 'Avg Fee (sat/vB)', angle: 90, position: 'insideRight' }} />
                  
                  <Tooltip content={<InsightsTooltip />} />
                  <Legend verticalAlign="top" height={36} />
                  
                  {/* Assign each data key to a Y-Axis */}
                  <Bar yAxisId="left" dataKey="total_vsize_bytes" name="Total VSize" fill="#8884d8" />
                  <Line yAxisId="right" type="monotone" dataKey="avg_fee_per_vbyte" name="Average Fee" stroke="#82ca9d" strokeWidth={2} dot={false} />
                </BarChart>
              </ResponsiveContainer>
            </Skeleton>
          </Card>
        </Col>
      </Row>
    </Card>
  );
};

export default MempoolOverview;