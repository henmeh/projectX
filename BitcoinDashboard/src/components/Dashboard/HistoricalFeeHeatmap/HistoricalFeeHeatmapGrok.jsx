import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import html2canvas from 'html2canvas'; // Install via npm i html2canvas
import { Card, Typography, Row, Col, Radio, Button, Input, Alert, Skeleton, Modal } from 'antd';
import { InfoCircleOutlined, QuestionCircleOutlined } from '@ant-design/icons';
import { fetchHistoricalFeeHeatmap, fetchFeePattern } from '../../../services/api.js';
import "../Dashboard.css";
import "../FeePredictions/FeePredictions.css";
import DataCard from '../../DataCard/DataCard.jsx';

const { Title, Text } = Typography;

// --- Helper Functions ---
const mapCategoryKey = (categoryString) => {
  if (!categoryString) return 'unknown';
  const lowerCaseCategory = categoryString.toLowerCase();
  if (lowerCaseCategory.includes('low')) return 'low';
  if (lowerCaseCategory.includes('medium')) return 'medium';
  if (lowerCaseCategory.includes('high')) return 'high';
  return 'unknown';
};

// --- Color Scale ---
// Updated with more colors for a smoother gradient and better visual appeal
const getFeeColor = (fee, minFee, maxFee) => {
  if (fee <= 0) return '#31313d'; // Neutral for empty slots
  const ratio = (fee - minFee) / (maxFee - minFee);
  if (ratio < 0.1) return '#003366'; // Very dark blue (extremely low)
  if (ratio < 0.2) return '#00a9ff'; // Deep blue (very low)
  if (ratio < 0.3) return '#66c2ff'; // Light blue
  if (ratio < 0.4) return '#00e1a4'; // Green (low)
  if (ratio < 0.5) return '#99ffcc'; // Light green
  if (ratio < 0.6) return '#fff200'; // Yellow (medium)
  if (ratio < 0.7) return '#ffd700'; // Gold yellow
  if (ratio < 0.8) return '#ff8c00'; // Orange (high)
  if (ratio < 0.9) return '#ff4500'; // Red-orange
  return '#ff005d'; // Red (very high)
};

// --- Main Component ---
const FeeHotspots = () => {
  const [heatmapData, setHeatmapData] = useState([]);
  const [categorizedFeePatterns, setCategorizedFeePatterns] = useState(null); 
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeRange, setTimeRange] = useState('Last 7 Days');
  const [displayTimezone, setDisplayTimezone] = useState('UTC');
  const [feeType, setFeeType] = useState('fast');
  const [hoveredCell, setHoveredCell] = useState(null);
  const [tooltipPosition, setTooltipPosition] = useState({ left: 0, top: 0 });
  const [activeCategory, setActiveCategory] = useState(null);
  const [streak, setStreak] = useState(0);
  const [txSize, setTxSize] = useState(0);
  const [claimedSlot, setClaimedSlot] = useState(null);
  const [showOptimizerHelp, setShowOptimizerHelp] = useState(false);
  const heatmapRef = useRef(null);

  const dayNames = useMemo(() => ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'], []); // Short names to match API and picture
  const fullDayNames = useMemo(() => ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], []);
  const hours = useMemo(() => Array.from({ length: 24 }, (_, i) => i), []);

  // Gamification: Streak tracking
  useEffect(() => {
    const today = new Date().toDateString();
    const lastVisit = localStorage.getItem('lastVisit');
    let currentStreak = parseInt(localStorage.getItem('streak') || '0', 10);
    if (lastVisit === today) {
      setStreak(currentStreak);
    } else {
      currentStreak = lastVisit ? currentStreak + 1 : 1;
      localStorage.setItem('streak', currentStreak);
      localStorage.setItem('lastVisit', today);
      setStreak(currentStreak);
    }
  }, []);

  // Fetch and process data
  useEffect(() => {
    const loadHistoricalData = async () => {
      setLoading(true);
      setError(null);
      try {
        const daysToFetch = timeRange === 'Last 30 Days' ? 30 : 7;
        const rawHeatmapData = await fetchHistoricalFeeHeatmap(daysToFetch, feeType);
        
        if (!Array.isArray(rawHeatmapData) || rawHeatmapData.length === 0) {
          throw new Error("No historical heatmap data returned from API.");
        }
        // Map short day to full if needed, but keep short for display
        const processedData = rawHeatmapData.map(d => ({...d, fullDay: fullDayNames[dayNames.indexOf(d.day)]}));
        setHeatmapData(processedData);

        const rawFeePatternData = await fetchFeePattern();
        if (Array.isArray(rawFeePatternData) && rawFeePatternData.length > 0) {
          const patterns = {
            high: { ranges: {} },
            medium: { ranges: {} },
            low: { ranges: {} },
            unknown: { ranges: {} }
          };

          rawFeePatternData.forEach(d => {
            const category = mapCategoryKey(d.fee_category);
            const dayNum = parseInt(d.day_of_week_num, 10);
            const startHour = parseInt(d.start_hour, 10);
            const endHour = parseInt(d.end_hour, 10);
            const avgFee = parseFloat(d.avg_fee_for_category);

            if (patterns[category] && !isNaN(dayNum) && !isNaN(startHour) && !isNaN(endHour) && !isNaN(avgFee)) {
              if (!patterns[category].ranges[dayNum]) {
                patterns[category].ranges[dayNum] = [];
              }
              patterns[category].ranges[dayNum].push({ start: startHour, end: endHour, avgFee });
            }
          });

          const formattedOutput = {};
          ['high', 'medium', 'low'].forEach(category => {
            const data = patterns[category];
            const daySummaries = [];
            let totalFee = 0;
            let count = 0;

            const sortedDayNums = Object.keys(data.ranges).map(Number).sort((a, b) => a - b);

            sortedDayNums.forEach(dayNum => {
              const dayName = fullDayNames[dayNum];
              const dayRanges = data.ranges[dayNum].sort((a, b) => a.start - b.start);
              const hourRanges = dayRanges.map(range => 
                `${String(range.start).padStart(2, '0')}:00-${String(range.end).padStart(2, '0')}:00`
              );
              daySummaries.push(`${dayName}: ${hourRanges.join(', ')} UTC`);
              dayRanges.forEach(range => {
                totalFee += range.avgFee;
                count++;
              });
            });
            
            formattedOutput[category] = {
              avgFee: count > 0 ? (totalFee / count).toFixed(2) : '0.00',
              times: daySummaries
            };
          });
          setCategorizedFeePatterns(formattedOutput);
        }

      } catch (err) {
        console.error("Failed to load historical fee data:", err);
        setError(err.message);
        setHeatmapData([]);
        setCategorizedFeePatterns(null);
      } finally {
        setLoading(false);
      }
    };
    loadHistoricalData();
  }, [timeRange, feeType]);

  const adjustedFeePatterns = useMemo(() => {
    if (!categorizedFeePatterns || displayTimezone === 'UTC') return categorizedFeePatterns;

    const offset = new Date().getTimezoneOffset() / 60;
    const adjusted = {};

    Object.keys(categorizedFeePatterns).forEach(category => {
      const original = categorizedFeePatterns[category];
      const daySummaries = [];

      original.times.forEach(timeStr => {
        const [dayName, rangesStr] = timeStr.split(': ');
        const dayIndex = fullDayNames.indexOf(dayName);
        const ranges = rangesStr.split(', ').map(rangeStr => {
          const [startStr, endStr] = rangeStr.split('-');
          let start = parseInt(startStr.split(':')[0]);
          let end = parseInt(endStr.split(':')[0]);

          start = (start - offset + 24) % 24;
          end = (end - offset + 24) % 24;

          if (end <= start) end += 24;  // Handle wrap-around

          return `${String(start).padStart(2, '0')}:00-${String(end % 24).padStart(2, '0')}:00`;
        });

        daySummaries.push(`${dayName}: ${ranges.join(', ')} Local`);
      });

      adjusted[category] = {
        ...original,
        times: daySummaries
      };
    });

    return adjusted;
  }, [categorizedFeePatterns, displayTimezone, fullDayNames]);

  // Memoized calculations for performance
  const { dataByDayHour, stats, bestTime, maxStddev } = useMemo(() => {
    if (!heatmapData || heatmapData.length === 0) {
      return { dataByDayHour: new Map(), stats: null, bestTime: null, maxStddev: 0 };
    }

    const dataMap = new Map();
    let minFee = Infinity;
    let maxFee = -Infinity;
    let totalFee = 0;
    let count = 0;
    let maxStd = 0;
    let bestTimeCandidate = { fee: Infinity, day: '', fullDay: '', hour: 0 };

    heatmapData.forEach(d => {
      const key = `${d.day}-${d.hour}`;
      dataMap.set(key, d);
      if (d.median_fee > 0) {
        totalFee += d.median_fee;
        count++;
        if (d.median_fee < minFee) minFee = d.median_fee;
        if (d.median_fee > maxFee) maxFee = d.median_fee;

        if (d.median_fee < bestTimeCandidate.fee) {
            bestTimeCandidate = { fee: d.median_fee, day: d.day, fullDay: d.fullDay || d.day, hour: d.hour };
        }
        if (d.fee_stddev > maxStd) maxStd = d.fee_stddev;
      }
    });
    
    const stats = {
      minFee: minFee === Infinity ? 0 : minFee,
      maxFee: maxFee === -Infinity ? 0 : maxFee,
      avgFee: count > 0 ? totalFee / count : 0,
    };

    return { dataByDayHour: dataMap, stats, bestTime: bestTimeCandidate, maxStddev: maxStd };
  }, [heatmapData]);

  // --- Timezone Adjusted Data ---
  const adjustedData = useMemo(() => {
    const offset = displayTimezone === 'Local' ? new Date().getTimezoneOffset() / 60 : 0;
    const adjustedMap = new Map();

    for (const [key, value] of dataByDayHour.entries()) {
      let { day, hour, fullDay } = value;
      let dayIndex = dayNames.indexOf(day);

      let adjustedHour = hour - offset;
      let adjustedDayIndex = dayIndex;

      if (adjustedHour < 0) {
        adjustedHour += 24;
        adjustedDayIndex = (dayIndex - 1 + 7) % 7;
      } else if (adjustedHour >= 24) {
        adjustedHour -= 24;
        adjustedDayIndex = (dayIndex + 1) % 7;
      }
      
      const newKey = `${dayNames[adjustedDayIndex]}-${adjustedHour}`;
      adjustedMap.set(newKey, { ...value, adjustedDay: dayNames[adjustedDayIndex], adjustedFullDay: fullDayNames[adjustedDayIndex], adjustedHour });
    }
    return adjustedMap;
  }, [dataByDayHour, displayTimezone, dayNames, fullDayNames]);

  // Savings calc for gamification
  const savings = useMemo(() => {
    if (txSize <= 0 || !stats.avgFee || !bestTime.fee) return 0;
    const lowFeeCost = bestTime.fee * txSize;
    const avgFeeCost = stats.avgFee * txSize;
    return ((avgFeeCost - lowFeeCost) / avgFeeCost * 100).toFixed(1);
  }, [txSize, stats, bestTime]);

  // Share on X
  const shareOnX = useCallback(() => {
    if (heatmapRef.current) {
      html2canvas(heatmapRef.current).then(canvas => {
        canvas.toBlob(blob => {
          const a = document.createElement('a');
          a.download = 'fee-hotspots.png';
          a.href = URL.createObjectURL(blob);
          a.click();
        });
      });
    }
    const text = `Check out this Bitcoin Fee Hotspots heatmap! Cheapest time: ${bestTime.fullDay}, ${String(bestTime.hour).padStart(2, '0')}:00 UTC at ${bestTime.fee.toFixed(1)} sat/vB. #Bitcoin #CryptoFees [your-site-url]`;
    window.open(`https://x.com/intent/tweet?text=${encodeURIComponent(text)}`, '_blank');
  }, [bestTime]);

  // --- Rendering ---
  const renderHeatmap = () => {
    const cellWidth = 38;
    const cellHeight = 38;
    const cornerRadius = 8;

    return (
      <div>
      <svg
        ref={heatmapRef}
        className="w-full h-auto"
        viewBox={`0 0 ${cellWidth * 25} ${cellHeight * 8}`}
        onMouseLeave={() => setHoveredCell(null)}
      >
        {/* Y-Axis Labels (Days) */}
        {dayNames.map((day, i) => (
          <text key={day} x="0" y={cellHeight * (i + 2) - cellHeight/2} style={{fill: "white"}}>
            {day}
          </text>
        ))}
        {/* X-Axis Labels (Hours) */}
        {hours.map(hour => (
          //(hour % 2 === 0) &&
          <text key={`label-${hour}`} x={cellWidth * (hour + 1.5)} y={cellHeight - 10} textAnchor="middle" style={{fill: "white"}}>
              {String(hour).padStart(2, '0')}
          </text>
        ))}

        {/* Heatmap Cells */}
        {dayNames.map((day, dayIndex) =>
          hours.map(hour => {
            const cellData = adjustedData.get(`${day}-${hour}`);
            const fee = cellData ? cellData.median_fee : -1;
            const stddev = cellData ? cellData.fee_stddev : 0;
            const color = getFeeColor(fee, stats.minFee, stats.maxFee);
            const opacity = maxStddev > 0 ? Math.max(0.5, 1 - (stddev / maxStddev) * 0.5) : 1;
            
            const isHovered = hoveredCell && hoveredCell.day === day && hoveredCell.hour === hour;
            const isDimmed = activeCategory && activeCategory !== getFeeCategory(fee, stats.minFee, stats.maxFee);
            const isClaimed = claimedSlot && claimedSlot.day === day && claimedSlot.hour === hour;

            return (
              <rect
                key={`${day}-${hour}`}
                x={cellWidth * (hour + 1)}
                y={cellHeight * (dayIndex + 1)}
                width={cellWidth - 2}
                height={cellHeight - 2}
                rx={cornerRadius}
                ry={cornerRadius}
                fill={color}
                opacity={isDimmed ? opacity * 0.2 : opacity}
                className={`transition-all duration-200 ease-in-out ${isClaimed ? 'stroke-yellow-400 stroke-2' : ''}`}
                style={{ transformOrigin: 'center center' }}
                transform={isHovered ? 'scale(1.01)' : 'scale(1)'}
                onMouseEnter={(e) => {
                  setHoveredCell({ day, fullDay: cellData?.adjustedFullDay || day, hour, fee: cellData?.median_fee, stddev: cellData?.fee_stddev, originalFullDay: cellData?.fullDay || cellData?.day, originalHour: cellData?.hour });
                  setTooltipPosition({ left: e.clientX + 10, top: e.clientY + 10 });
                }}
                onClick={() => {
                  const cat = getFeeCategory(fee, stats.minFee, stats.maxFee);
                  if (cat === 'low') {
                    setClaimedSlot({ day, hour });
                    alert('Claimed low-fee slot! Build your streak for rewards.');
                  }
                }}
              />
            );
          })
        )}
      </svg>
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={4}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }} onMouseEnter={() => setActiveCategory('low')} onMouseLeave={() => setActiveCategory(null)}>
                <div style={{ width: 16, height: 16, borderRadius: '50%', backgroundColor: '#003366' }}></div>
                <Text>Very Low Fees</Text>
              </div>
        </Col>
              <Col xs={24} sm={4}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }} onMouseEnter={() => setActiveCategory('low')} onMouseLeave={() => setActiveCategory(null)}>
                <div style={{ width: 16, height: 16, borderRadius: '50%', backgroundColor: '#00e1a4' }}></div>
                <Text>Low Fees</Text>
              </div>
                         </Col>
              <Col xs={24} sm={4}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }} onMouseEnter={() => setActiveCategory('medium')} onMouseLeave={() => setActiveCategory(null)}>
                <div style={{ width: 16, height: 16, borderRadius: '50%', backgroundColor: '#fff200' }}></div>
                <Text>Medium Fees</Text>
              </div>
              </Col>
              <Col xs={24} sm={4}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }} onMouseEnter={() => setActiveCategory('high')} onMouseLeave={() => setActiveCategory(null)}>
                <div style={{ width: 16, height: 16, borderRadius: '50%', backgroundColor: '#ff005d' }}></div>
                <Text>High Fees</Text>
              </div>
              </Col>
      </Row> 
      </div>
    );
  };
  
  const getFeeCategory = (fee, min, max) => {
    if (fee < 0) return 'unknown';
    const ratio = (fee - min) / (max - min);
    if (ratio < 0.33) return 'low';
    if (ratio < 0.66) return 'medium';
    return 'high';
  };

  const renderTooltip = () => {
    if (!hoveredCell) return null;
    
    const feeCategory = getFeeCategory(hoveredCell.fee, stats.minFee, stats.maxFee);
    const categoryColors = {
      low: 'text-green-400',
      medium: 'text-yellow-400',
      high: 'text-red-400'
    };

    return (
      <div 
        style={{ position: 'fixed', left: tooltipPosition.left, top: tooltipPosition.top, zIndex: 10, backgroundColor: "grey", opacity: 0.9, padding: 8, borderRadius: 4 }}
      >
        <div className="font-bold text-lg mb-2">{hoveredCell.fullDay}, {String(hoveredCell.hour).padStart(2, '0')}:00 <span className="text-sm font-light text-gray-400">{displayTimezone}</span></div>
        {hoveredCell.fee > 0 ? (
            <>
                <div className="text-3xl font-light mb-2">
                    {hoveredCell.fee.toFixed(1)} ± {hoveredCell.stddev.toFixed(1)} <span className="text-xl">sat/vB</span>
                </div>
                <div className={`text-sm font-bold uppercase tracking-widest ${categoryColors[feeCategory]}`}>{feeCategory} Fees</div>
                <div className="text-xs text-gray-400 mt-2">
                    (Original: {hoveredCell.originalFullDay}, {String(hoveredCell.originalHour).padStart(2, '0')}:00 UTC)
                </div>
            </>
        ) : (
            <div className="text-gray-400">No data for this period</div>
        )}
      </div>
    );
  };

  if (loading) {
    return <Skeleton active paragraph={{ rows: 10 }} />;
  }
  
  if (error) {
     return <Alert message="Error" description={error} type="error" showIcon />;
  }

  return (
    <Card className="dashboard-card" title={<div><Title level={4} style={{ margin: 0 }}>Network Fee Hotspots</Title> <Text type="secondary">Find the best time to send by seeing when fees are typically high or low.</Text></div>}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <Radio.Group options={['fast', 'medium', 'low']} value={feeType} onChange={(e) => setFeeType(e.target.value)} optionType='button' buttonStyle='solid'/>
        <Radio.Group options={['UTC', 'Local']} value={displayTimezone} onChange={(e) => setDisplayTimezone(e.target.value)} optionType='button' buttonStyle='solid'/>
        <Radio.Group options={['Last 7 Days', 'Last 30 Days']} value={timeRange} onChange={(e) => setTimeRange(e.target.value)} optionType="button" buttonStyle="solid"/>
        <Button onClick={shareOnX}>Share on X</Button>
      </div>

      {/* Main Content */}
      <div className="relative">
        {/* Heatmap */}
        <div>
          {renderHeatmap()}
          {renderTooltip()}
        </div>
      </div>

      <Row gutter={16} style={{ marginBottom: 24, marginTop: 24 }}>
        <Col xs={24} sm={8}><DataCard title="Lowest Fee" data={`${stats.minFee.toFixed(1)} sat/vB`}/></Col>
        <Col xs={24} sm={8}><DataCard title="Average Fee" data={`${stats.avgFee.toFixed(1)} sat/vB`}/></Col>
        <Col xs={24} sm={8}><DataCard title="Highest Fee" data={`${stats.maxFee.toFixed(1)} sat/vB`}/></Col>
      </Row>

      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={8}>
          <DataCard title="Smart Suggestion" data={
            bestTime && (
              <Text>
                The cheapest time is typically around <Text strong>{bestTime.fullDay}, {String(bestTime.hour).padStart(2,'0')}:00 UTC</Text> with fees near <Text strong>{bestTime.fee.toFixed(1)} sat/vB</Text>.
              </Text>
            )
          }/>
        </Col>
        <Col xs={24} sm={8}>
          <DataCard title="Savings Optimizer" data={
            <>
              <QuestionCircleOutlined 
                style={{ cursor: 'pointer', float: 'right', color: '#1890ff' }} 
                onClick={() => setShowOptimizerHelp(true)} 
              />
              <Text>Streak: {streak} days 🔥</Text>
              <Input
                type="number"
                placeholder="Tx size (vB)"
                value={txSize}
                onChange={(e) => setTxSize(e.target.value ? parseFloat(e.target.value) : 0)}
                style={{ marginTop: 8, marginBottom: 8 }}
              />
              <Text>Savings Potential: {savings}%</Text>
              <Text block style={{ marginTop: 8 }}>Click low-fee cells to claim!</Text>
            </>
          }/>
        </Col>
      </Row>

      <Card 
        className="dashboard-card"
        title={<Title level={4}> Weekly Fee Summary </Title>}
        style={{ marginTop: 24 }}
      >            
        <Row gutter={16} style={{ marginBottom: 24 }}>
          {adjustedFeePatterns && Object.keys(adjustedFeePatterns).length > 0 ? (
            <>
                {['low', 'medium', 'high'].map(category => {
                    const data = adjustedFeePatterns[category];
                    if (data && data.times.length > 0) { // Ensure there are times to display for the category
                        return (
                            <Col xs={24} sm={8} key={category}>
                              <DataCard className={`fee-card ${category}-fee`} title={`${category.toUpperCase()} Fee Times`} data={<ul style={{ listStyleType: 'disc', paddingLeft: 20, marginTop: 5 }}>
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
      <Modal
        title="Savings Optimizer Guide"
        open={showOptimizerHelp}
        onCancel={() => setShowOptimizerHelp(false)}
        footer={null}
      >
        <p>Here, you can optimize your Bitcoin transactions and have some fun!</p>
        <ul>
          <li><strong>Streak</strong>: Visit daily to build your streak and unlock future rewards.</li>
          <li><strong>Tx Size Input</strong>: Enter your transaction size in vB to see potential savings by timing your transaction during low-fee periods.</li>
          <li><strong>Claim Slots</strong>: Click green (low-fee) cells on the heatmap to claim them. Build a collection and maintain your streak!</li>
        </ul>
        <p>Keep coming back to maximize savings and climb the leaderboard (coming soon).</p>
      </Modal>
    </Card>
  );
};

export default FeeHotspots;