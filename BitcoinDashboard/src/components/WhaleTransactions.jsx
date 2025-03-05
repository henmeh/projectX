import { useEffect, useState } from "react";
import { Table } from "antd";
import { fetchWhaleTransactions } from "../utils/api";

export default function WhaleTransactions() {
  const [transactions, setTransactions] = useState([]);

  useEffect(() => {
    // Function to load whale transactions
    async function loadTransactions() {
      try {
        const data = await fetchWhaleTransactions();
        if (data && data.whale_transactions) {
          const sortedTransactions = data.whale_transactions
            .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
            .slice(0, 500);
          setTransactions(sortedTransactions);
        }
      } catch (error) {
        console.error("Error fetching transactions:", error);
      }
    }

    // Load transactions initially
    loadTransactions();

    // Set interval to fetch data every 60 seconds
    const intervalId = setInterval(() => {
      loadTransactions();
    }, 60000);  // 60,000ms = 1 minute

    // Cleanup function to clear the interval when the component unmounts
    return () => clearInterval(intervalId);
  }, []); // Empty dependency array ensures this only runs once on mount

  const columns = [
    {
      title: "Timestamp",
      dataIndex: "timestamp",
      key: "timestamp",
      width: 200,
    },
    {
      title: "Transaction ID",
      dataIndex: "txid",
      key: "txid",
      width: 150,
      render: (text) => (
        <a href={`https://mempool.space/tx/${text}`} target="_blank" rel="noopener noreferrer">
          {text}
        </a>
      ),
    },
    {
      title: "Total Sent (BTC)",
      dataIndex: "total_sent",
      key: "total_sent",
      width: 150,
      render: (text) => Number(text).toFixed(8),
    },
  ];

  return (
    <div>
      <h2>Whale Transactions</h2>
      <Table
        dataSource={transactions}
        columns={columns}
        rowKey="txid"
        pagination={true}
        scroll={{ x: 700 }}
      />
    </div>
  );
}
