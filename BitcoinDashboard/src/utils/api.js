import axios from "axios";

const API_BASE_URL = "http://0.0.0.0:8000";

export const fetchWhaleTransactions = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/whale-transactions`);
    return response.data;
  } catch (error) {
    console.error("Error fetching whale transactions:", error);
    return [];
  }
};

export async function fetchFeeHistogram() {
    try {
        const response = await axios.get(`${API_BASE_URL}/fee-histogram`);
        return response.data;
      } catch (error) {
        console.error("Error fetching whale transactions:", error);
        return [];
      }
};

export async function fetchMempoolCongestion() {
    try {
        const response = await axios.get(`${API_BASE_URL}/mempool-congestion`);
        return response.data;
      } catch (error) {
        console.error("Error fetching whale transactions:", error);
        return [];
      }
}
