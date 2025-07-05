// src/utils/dataPreparation.js

/**
 * Prepares raw historical fee data into a complete grid format suitable for a heatmap.
 * Fills in missing data points with a default average fee of 0.
 *
 * @param {Array<Object>} rawData - The array of objects received from the API.
 * @returns {Array<Object>} The processed array for the heatmap.
 */
export const prepareHeatmapData = (rawData) => {
  const dayOrder = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

  // Create a quick lookup map for the fetched data
  const fetchedDataMap = new Map();
  rawData.forEach(item => {
    const dayStr = dayOrder[item.day_of_week_num];
    const hour = item.hour_of_day;
    fetchedDataMap.set(`${dayStr}-${hour}`, parseFloat(item.avg_fee));
  });

  // Generate a complete grid for all 7 days and 24 hours
  const completeDataGrid = [];
  for (let d = 0; d < 7; d++) {
    const currentDay = dayOrder[d];
    for (let h = 0; h < 24; h++) {
      const avgFee = fetchedDataMap.get(`${currentDay}-${h}`);
      completeDataGrid.push({
        day: currentDay,
        hour: h,
        avg_fee: avgFee !== undefined ? avgFee : 0, // Fill missing data with 0
      });
    }
  }

  // Sort the data for consistent rendering (day then hour)
  completeDataGrid.sort((a, b) => {
    const dayIndexA = dayOrder.indexOf(a.day);
    const dayIndexB = dayOrder.indexOf(b.day);
    if (dayIndexA !== dayIndexB) {
      return dayIndexA - dayIndexB;
    }
    return a.hour - b.hour;
  });

  return completeDataGrid;
};