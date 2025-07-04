import axios from 'axios';

// Use Vite's proxy in development
const API_BASE_URL = import.meta.env.PROD 
  ? 'https://your-production-api.com' 
  : '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
});

export const fetchWhaleTransactions = async (minBtc = 10.0) => {
  try {
    const response = await api.get('/whale-transactions', {
      params: { min_btc: minBtc }
    });
    return response.data.whale_transactions || [];
  } catch (error) {
    console.error('Error fetching whale transactions:', error);
    return [];
  }
};

export const fetchFeeEstimation = async () => {
  try {
    const response = await api.get('/fee-estimation');
    return response.data || null;
  } catch (error) {
    console.error('Error fetching fee estimation:', error);
    return null;
  }
};

export const fetchFeeHistogram = async () => {
  try {
    const response = await api.get('/fee-histogram');
    return response.data || null;
  } catch (error) {
    console.error('Error fetching fee histogram:', error);
    return null;
  }
};

export const fetchFeePrediction = async (tableName) => {
  try {
    const response = await api.get(`/fee-prediction/${tableName}`);
    return response.data || [];
  } catch (error) {
    console.error(`Error fetching fee predictions for ${tableName}:`, error);
    return [];
  }
};

export const fetchMempoolCongestion = async () => {
  try {
    const response = await api.get('/mempool-congestion');
    return response.data || null;
  } catch (error) {
    console.error('Error fetching mempool congestion:', error);
    return null;
  }
};