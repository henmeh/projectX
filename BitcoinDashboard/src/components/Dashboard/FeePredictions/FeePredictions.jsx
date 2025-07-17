import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Card, Typography, Row, Col, Statistic, Radio, Select, Space, Tabs } from 'antd';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { fetchFeeEstimation, fetchFeePrediction } from '../../../services/api';
import moment from 'moment'; // Make sure to install moment: npm install moment or yarn add moment
import '../Dashboard.css'
import './FeePredictions.css'
import DataCard from '../../DataCard/DataCard.jsx';


const { Title, Text } = Typography;
const { Option } = Select;

// Helper to format timestamps for display (e.g., "Jul 4th, 11:30")
const formatTimestamp = (timestamp) => {
  // Assuming timestamps from DB are ISO strings (e.g., "2025-07-04T11:00:00+02:00")
  return moment(timestamp).format('MMM Do, HH:mm');
};

const FeePredictions = () => {
  const [currentFees, setCurrentFees] = useState(null); // Holds current market fee estimation
  const [selectedModelTable, setSelectedModelTable] = useState('fee_predictions_prophet'); // 'fee_predictions_prophet' or 'fee_predictions_random_forest'
  const [futurePredictionsBatch, setFuturePredictionsBatch] = useState([]); // Array of all future predictions from the latest batch
  const [latestGeneratedAt, setLatestGeneratedAt] = useState(null); // The common 'generated_at' timestamp for the fetched batch
  const [availableLookbackIntervals, setAvailableLookbackIntervals] = useState([]); // e.g., ['hourly', 'daily', 'weekly'] for the selected model
  const [selectedLookbackInterval, setSelectedLookbackInterval] = useState(null); // The currently selected interval for display
  const [loading, setLoading] = useState(true);

  // Function to fetch all data for current fees and predictions
  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      // 1. Fetch current market fees
      const fees = await fetchFeeEstimation();
      setCurrentFees(fees);

      // 2. Fetch the latest batch of predictions for the currently selected model table
      const predictions = await fetchFeePrediction(selectedModelTable);
      setFuturePredictionsBatch(predictions);

      if (predictions.length > 0) {
        // Extract unique 'model_name' values (which correspond to lookback intervals like 'hourly', 'daily')
        const uniqueIntervals = [...new Set(predictions.map(p => p.model_name))].sort();
        setAvailableLookbackIntervals(uniqueIntervals);

        // If no lookback interval is selected, or the previously selected one is not available
        // default to 'hourly' if present, otherwise to the first available interval.
        if (!selectedLookbackInterval || !uniqueIntervals.includes(selectedLookbackInterval)) {
          setSelectedLookbackInterval(uniqueIntervals.includes('hourly') ? 'hourly' : uniqueIntervals[0]);
        }
        // Set the 'generated_at' timestamp for the entire batch (all predictions in this batch share it)
        setLatestGeneratedAt(predictions[0].generated_at);
      } else {
        // Reset if no predictions are found
        setAvailableLookbackIntervals([]);
        setSelectedLookbackInterval(null);
        setLatestGeneratedAt(null);
      }

    } catch (error) {
      console.error("Failed to load fee data:", error);
      // Ensure all states are reset to null/empty on error to avoid stale data
      setCurrentFees(null);
      setFuturePredictionsBatch([]);
      setAvailableLookbackIntervals([]);
      setSelectedLookbackInterval(null);
      setLatestGeneratedAt(null);
    } finally {
      setLoading(false);
    }
  }, [selectedModelTable, selectedLookbackInterval]); // Depend on model and lookback selection to re-fetch

  // Effect hook to run data loading on component mount and on interval
  useEffect(() => {
    loadData(); // Initial load
    // Refresh every hour to get the very latest batch (if new batches are stored in DB)
    const interval = setInterval(loadData, 60000*60);
    return () => clearInterval(interval); // Cleanup on unmount
  }, [loadData]); // `loadData` is a dependency, ensuring it's always the latest version

  // Filter and format the predictions batch for the chart based on selected lookback interval
  const chartData = useMemo(() => {
    if (!futurePredictionsBatch || !selectedLookbackInterval) return [];
    return futurePredictionsBatch
      .filter(p => p.model_name === selectedLookbackInterval)
      .map(p => ({
        timestamp: moment(p.prediction_time).format('HH:mm'), // Formatted time for X-axis labels
        fullTimestamp: p.prediction_time, // Keep original for detailed tooltip
        predictedLow: parseFloat(p.low_fee),
        predictedMedium: parseFloat(p.medium_fee),
        predictedHigh: parseFloat(p.fast_fee),
      }))
      .sort((a, b) => moment(a.fullTimestamp).valueOf() - moment(b.fullTimestamp).valueOf()); // Ensure data is sorted by time
  }, [futurePredictionsBatch, selectedLookbackInterval]);

  // Extract the immediate next prediction for the Statistic cards
  // This typically takes the first prediction point from the filtered chartData
  const immediateNextPrediction = useMemo(() => {
      if (!chartData || chartData.length === 0) return null;
      return chartData[0]; // The first entry is the earliest future prediction
  }, [chartData]);


  // Custom tooltip rendering for the chart
  const renderTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      // Use the fullTimestamp stored in the payload for more accurate time display
      const fullTimestamp = payload[0].payload.fullTimestamp;
      return (
        <div className="custom-tooltip" style={{ background: '#fff', padding: '10px', border: '1px solid #ccc', borderRadius: '4px', boxShadow: '0 0 10px rgba(0,0,0,0.1)' }}>
          <p className="label" style={{ fontWeight: 'bold' }}>Prediction Time: {moment(fullTimestamp).format('MMM Do, HH:mm')}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color, margin: '0' }}>
              {entry.name}: {entry.value.toFixed(2)} sat/vB
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  // Options for the model selection Radio.Group
  const modelOptions = [
    { label: 'Prophet Model', value: 'fee_predictions_prophet' },
    { label: 'Random Forest Model', value: 'fee_predictions_random_forest' },
  ];

  return (
    <Card
      title={<Title level={4} style={{ margin: 0 }}>Fee Statistics</Title>}
      className="dashboard-card"
      loading={loading}
    >
      {/* Display Current Actual Fees */}
      {currentFees && (
        <>
          <Title level={5} style={{ marginBottom: 16 }}>Current Market Fees</Title>
          <Row gutter={16} style={{ marginBottom: 24 }}>
            <Col span={8}>
              <DataCard className="fee-card low-fee" title="Current Low Fee" data={`${currentFees.low_fee} sat/vB`} />
            </Col>
            <Col span={8}>
              <DataCard className="fee-card medium-fee" title="Current Medium Fee" data={`${currentFees.medium_fee} sat/vB`} />
            </Col>
            <Col span={8}>
              <DataCard className="fee-card high-fee" title="Current High Fee" data={`${currentFees.fast_fee} sat/vB`} />
            </Col>
          </Row>
        </>
      )}

      {/* Prediction Setup Section */}
      <Space style={{ marginBottom: 24, marginTop: 24 }}> {/* Added marginBottom here for spacing */}
          {/* Radio Group for Model Selection (Prophet vs. Random Forest) */}
          <Radio.Group
            options={modelOptions}
            onChange={(e) => setSelectedModelTable(e.target.value)}
            value={selectedModelTable}
            optionType="button"
            buttonStyle="solid"
          />
          {/* Select dropdown for Lookback Interval (e.g., 'hourly', 'daily', 'weekly') */}
          {availableLookbackIntervals.length > 0 && ( // Only show if there are intervals
            <Select
              value={selectedLookbackInterval}
              onChange={(value) => setSelectedLookbackInterval(value)}
              style={{ width: 150 }}
              placeholder="Select Interval"
            >
              {availableLookbackIntervals.map(interval => (
                <Option key={interval} value={interval}>{interval.replace(/_/g, ' ')}</Option>
              ))}
            </Select>
          )}
      </Space>

      {/* Display the immediate next predicted fee (e.g., the closest future point) */}
      {immediateNextPrediction && (
        <>
          <Title level={5} style={{ marginBottom: 16 }}>
            Next Predicted Fees (for {moment(immediateNextPrediction.fullTimestamp).format('HH:mm')})
          </Title>
          <Row gutter={16} style={{ marginBottom: 24 }}>
            <Col span={8}>
              <DataCard className="fee-card low-fee" title="Predicted Low Fee" data={`${immediateNextPrediction.predictedLow.toFixed(2)} sat/vB`} />
            </Col> 
            <Col span={8}>
              <DataCard className="fee-card medium-fee" title="Predicted Medium Fee" data={`${immediateNextPrediction.predictedMedium.toFixed(2)} sat/vB`} />
            </Col>
            <Col span={8}>
              <DataCard className="fee-card high-fee" title="High Fee" data={`${immediateNextPrediction.predictedHigh.toFixed(2)} sat/vB`} />
            </Col>
          </Row>
        </>
      )}

      <Card className='data-card'>
      {/* Display the timestamp when this forecast batch was generated */}
      {latestGeneratedAt && (
        <Typography.Text strong style={{ marginBottom: 16, display: 'block' }}>
          Forecast Generated At: {formatTimestamp(latestGeneratedAt)}
        </Typography.Text>
      )}

      {/* Chart for Future Predictions */}
      {chartData.length > 0 ? (
        <ResponsiveContainer width="100%" height={300}>
          <LineChart
            data={chartData}
            margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="timestamp"
              label={{ value: 'Prediction Time', position: 'insideBottom', offset: -10 }}
              interval="preserveStartEnd" // Helps prevent too many labels on dense data
            />
            <YAxis label={{ value: 'Fee (sat/vB)', angle: -90, position: 'insideLeft' }} />
            <Tooltip content={renderTooltip} />
            {/* Lines for predicted low, medium, and high fees */}
            <Line
              type="monotone"
              dataKey="predictedLow"
              name={`Predicted Low (${selectedLookbackInterval})`}
              stroke="#52c41a"
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="predictedMedium"
              name={`Predicted Medium (${selectedLookbackInterval})`}
              stroke="#1890ff"
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="predictedHigh"
              name={`Predicted High (${selectedLookbackInterval})`}
              stroke="#ff4d4f"
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <Text disabled>No future predictions available for the selected model and interval.</Text>
      )}
      </Card>
    </Card>
  );
};

export default FeePredictions;