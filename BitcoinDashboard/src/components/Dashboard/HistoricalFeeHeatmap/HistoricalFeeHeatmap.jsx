import React, { useState, useEffect, useMemo } from 'react';
import { Card, Typography, Row, Col, Statistic, Tag, Progress, Skeleton, Alert, Tabs, Radio } from 'antd';
import { ClockCircleOutlined, PieChartOutlined, InfoCircleOutlined } from '@ant-design/icons';
import { fetchHistoricalFeeHeatmap, fetchFeePattern } from '../../../services/api';
import { Heatmap } from '@ant-design/plots'; // Ensure Heatmap is imported
import "../Dashboard.css";
import "../FeePredictions/FeePredictions.css";
import DataCard from '../../DataCard/DataCard.jsx';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

const mapCategoryKey = (categoryString) => {
  if (!categoryString) return 'unknown';
  const lowerCaseCategory = categoryString.toLowerCase();
  if (lowerCaseCategory.includes('low')) return 'low';
  if (lowerCaseCategory.includes('medium')) return 'medium';
  if (lowerCaseCategory.includes('high')) return 'high';
  return 'unknown';
};

const HistoricalFeeHeatmap = () => {
  const [heatmapData, setHeatmapData] = useState([]);
  const [categorizedFeePatterns, setCategorizedFeePatterns] = useState(null); 
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeRange, setTimeRange] = useState('Last 7 Days');

  // Mapping for day_of_week_num to full day names (assuming 0=Sunday, 1=Monday, ..., 6=Saturday from PostgreSQL EXTRACT(DOW))
  const dayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

  useEffect(() => {
    const loadHistoricalData = async () => {
      setLoading(true);
      setError(null);
      try {
        const daysToFetch = timeRange === 'Last 30 Days' ? 30 : 7;
        
        const [rawHeatmapData, rawFeePatternData] = await Promise.all([
          fetchHistoricalFeeHeatmap(daysToFetch),
          fetchFeePattern() // FIXED: No daysToFetch parameter here
        ]);

        // --- Process Heatmap Data ---
        if (!Array.isArray(rawHeatmapData) || rawHeatmapData.length === 0) {
          console.warn("No historical heatmap data returned from API.");
          setHeatmapData([]);
        } else {
          setHeatmapData(rawHeatmapData);
        }

        // --- Process Fee Pattern Data for the Text Summary ---
        if (!Array.isArray(rawFeePatternData) || rawFeePatternData.length === 0) {
             console.warn("No fee pattern data returned from API.");
             setCategorizedFeePatterns(null); // Clear previous data if no new data
        } else {
            // Initialize structure to hold categorized patterns
            // Use 'unknown' as a fallback category to capture any unexpected data
            const patterns = {
                high: { hours: {}, totalFee: 0, count: 0 },
                medium: { hours: {}, totalFee: 0, count: 0 },
                low: { hours: {}, totalFee: 0, count: 0 },
                unknown: { hours: {}, totalFee: 0, count: 0 } // Add an unknown category for robustness
            };

            rawFeePatternData.forEach(d => {
                const category = mapCategoryKey(d.fee_category); // Use the helper to map category key
                const dayNum = parseInt(d.day_of_week_num, 10); // Ensure base 10 for parseInt
                const hour = parseInt(d.start_hour, 10); // Store hour as integer for reliable sorting
                const avgFee = parseFloat(d.avg_fee_for_category);

                // Only process if category is one of the expected types and avgFee is a valid number
                if (patterns[category] && !isNaN(dayNum) && !isNaN(hour) && !isNaN(avgFee)) {
                    if (!patterns[category].hours[dayNum]) {
                        patterns[category].hours[dayNum] = new Set(); // Use Set to avoid duplicate hours
                    }
                    patterns[category].hours[dayNum].add(hour); // Store hour as integer
                    patterns[category].totalFee += avgFee;
                    patterns[category].count++;
                } else {
                    console.warn(`Skipping malformed fee pattern data entry:`, d);
                }
            });

            // Format the aggregated data into the desired text summary structure
            const formattedOutput = {};
            // Iterate over categories in a specific, prioritized order for display
            ['high', 'medium', 'low'].forEach(category => {
                const data = patterns[category];
                if (data.count > 0) { // Only include categories that actually have data
                    const avgCategoryFee = (data.totalFee / data.count).toFixed(2);
                    const daySummaries = [];

                    // Sort days numerically to ensure consistent order (Sunday, Monday, etc.)
                    const sortedDayNums = Object.keys(data.hours).map(Number).sort((a, b) => a - b);

                    sortedDayNums.forEach(dayNum => {
                        const dayName = dayNames[dayNum];
                        // Convert Set to Array and sort hours numerically (as they are stored as integers)
                        const sortedHours = Array.from(data.hours[dayNum]).sort((a, b) => a - b);
                        
                        // Consolidate consecutive hours into ranges (e.g., "00, 01, 02" -> "00:00-02:00")
                        let hourRanges = [];
                        if (sortedHours.length > 0) {
                            let startRange = sortedHours[0];
                            let endRange = startRange;
                            for (let i = 1; i < sortedHours.length; i++) {
                                const currentHour = sortedHours[i];
                                if (currentHour === endRange + 1) {
                                    endRange = currentHour;
                                } else {
                                    // Push the completed range and start a new one
                                    hourRanges.push(startRange === endRange ? 
                                        `${String(startRange).padStart(2, '0')}:00` : 
                                        `${String(startRange).padStart(2, '0')}:00-${String(endRange).padStart(2, '0')}:00`);
                                    startRange = currentHour;
                                    endRange = currentHour;
                                }
                            }
                            // Push the last range after the loop
                            hourRanges.push(startRange === endRange ? 
                                `${String(startRange).padStart(2, '0')}:00` : 
                                `${String(startRange).padStart(2, '0')}:00-${String(endRange).padStart(2, '0')}:00`);
                        }
                        daySummaries.push(`${dayName}: ${hourRanges.join(', ')} UTC`);
                    });
                    
                    formattedOutput[category] = {
                        avgFee: avgCategoryFee,
                        times: daySummaries
                    };
                }
            });
            setCategorizedFeePatterns(formattedOutput);
        }

      } catch (err) {
        console.error("Failed to load historical fee data:", err);
        setError("Could not load historical fee data. Please try again later.");
        setHeatmapData([]); 
        setCategorizedFeePatterns(null); 
      } finally {
        setLoading(false);
      }
    };
    loadHistoricalData();
  }, [timeRange]); // Dependency on timeRange ensures data re-fetches when range changes

  /**
   * Memoized calculation of heatmap statistics (min, max, avg fees).
   * Re-calculates only when heatmapData changes.
   */
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

  /**
   * Configuration for the Heatmap chart.
   */
  const heatmapConfig = {
    data: heatmapData,
    xField: 'hour',    
    yField: 'day',     
    colorField: 'avg_fee',
    mark: 'cell',
    legend: {},
    axis: {
      x: {
        title: "Hour of the day",
        titleFill: "#e6e6e6",
        labelFontSize: 12,
        labelFill: "#e6e6e6"
      },
      y: {
        title: "Day",
        titleFill: "#e6e6e6",
        labelFontSize: 12,
        labelFill: "#e6e6e6"
      }
    },
    tooltip: {
      items: [
        { channel: 'x', title: 'Hour' },
        { channel: 'y', title: 'Day' },
        { channel: 'color', title: 'Avg. Fee (sat/vB)', valueFormatter: (d) => d.toFixed(1) },
      ],
    },
    interactions: [{ type: 'brush' }], // Allows for brushing/selection on the heatmap
  };

  /**
   * Helper function to get text color based on fee category.
   * @param {string} category - The fee category ('low', 'medium', 'high').
   * @returns {string} CSS color string.
   */

  return (
    <Card
    className='dashboard-card'
    title={<Title level={4} style={{ margin: 0 }}>Network Fee Hotspots</Title>}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <Text type="secondary">Find the best time to send by seeing when fees are typically high or low.</Text>
        <Radio.Group
          options={['Last 7 Days', 'Last 30 Days']}
          value={timeRange}
          onChange={(e) => setTimeRange(e.target.value)}
          optionType="button"
          buttonStyle="solid"
        />
      </div>

      {loading && <Skeleton active paragraph={{ rows: 10 }} />}
      
      {!loading && error && <Alert message="Error" description={error} type="error" showIcon />}
      
      {!loading && !error && heatmapData.length > 0 ? ( // Only render if heatmap data exists
        <>
          {heatmapStats && (
            <Row gutter={16} style={{ marginBottom: 24 }}>
              <Col xs={24} sm={8}><DataCard title="Lowest Fee" data={`${heatmapStats.minFee.toFixed(1)} sat/vB`}/></Col>
              <Col xs={24} sm={8}><DataCard title="Average Fee" data={`${heatmapStats.avgFee.toFixed(1)} sat/vB`}/></Col>
              <Col xs={24} sm={8}><DataCard title="Highest Fee" data={`${heatmapStats.maxFee.toFixed(1)} sat/vB`}/></Col>
              </Row>
          )}
          <Card 
            className="data-card"
            title={<Title level={4} style={{ margin: 0 }}> Fee Hotspots </Title>}
          >
            <div style={{ height: 350, position: 'relative'}}>
              <Heatmap {...heatmapConfig} /> 
            </div>

            <div style={{ textAlign: 'center', marginTop: 16 }}>
              <Text type="secondary">
                Hover over a block to see the average fee for that hour.
              </Text>
            </div>
          </Card>

          <Card 
            className="dashboard-card"
            title={<Title level={4}> Weekly Fee Summary </Title>}
            style={{ marginTop: 24 }}
          >            
            <Row gutter={16} style={{ marginBottom: 24 }}>
              {categorizedFeePatterns && Object.keys(categorizedFeePatterns).length > 0 ? (
                <>
                    {['low', 'medium', 'high'].map(category => {
                        const data = categorizedFeePatterns[category];
                        if (data && data.times.length > 0) { // Ensure there are times to display for the category
                            return (
                                <Col xs={24} sm={8}>
                                  <DataCard className={`fee-card ${category}-fee`} key={category} title={`${category.toUpperCase()} Fee Times`} data={<ul style={{ listStyleType: 'disc', paddingLeft: 20, marginTop: 5 }}>
                                        <Text strong>Average Fee: {data.avgFee} sat/vB</Text>
                                        {data.times.map((timeEntry, index) => (
                                            <li key={index} style={{ marginBottom: 4, marginTop: 4 }}>
                                                <Text>{timeEntry}</Text>
                                            </li>
                                        ))}
                                    </ul>}/>
                                </Col>
                            );
                        }
                        return null; 
                    })}
                </>
            ) : (
                <div style={{ textAlign: 'center', padding: '48px 0' }}>
                    <InfoCircleOutlined style={{ fontSize: 48, color: '#bfbfbf' }} />
                    <Title level={5} style={{ marginTop: 16 }}>No Fee Pattern Summary Available</Title>
                    <Text type="secondary">Could not generate a typical weekly fee summary for the selected period. This might be due to a lack of data or an issue with the backend API.</Text>
                </div>
            )}
            </Row>
            
          </Card>
        </>
      ) : ( // Fallback for when no historical data is available at all for heatmap or patterns
        <div style={{ textAlign: 'center', padding: '48px 0' }}>
            <PieChartOutlined style={{ fontSize: 48, color: '#bfbfbf' }} />
            <Title level={5} style={{ marginTop: 16 }}>No Historical Data Available</Title>
            <Text type="secondary">There is no fee data to display for the selected time period.</Text>
        </div>
       )}
    </Card>
  );
};

export default HistoricalFeeHeatmap;