import { useEffect, useState } from "react";
import { Table, Tooltip } from "antd";
import { fetchWhaleTransactions } from "../utils/api";

export default function WhaleTransactions() {
  const [transactions, setTransactions] = useState([]);

  useEffect(() => {
    async function loadTransactions() {
      try {
        const data = await fetchWhaleTransactions();
        if (data && data.whale_transactions) {
          const sortedTransactions = data.whale_transactions
            .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
            .slice(0, 50);
          setTransactions(sortedTransactions);
        }
      } catch (error) {
        console.error("Error fetching transactions:", error);
      }
    }

    loadTransactions();

    // Set interval to fetch data every 60 seconds
    const interval = setInterval(loadTransactions, 60000);

    return () => clearInterval(interval);
  }, []);

  // Format txid (show first 5 + last 5 chars)
  const formatTxid = (txid) => {
    if (!txid) return "";
    return `${txid.slice(0, 5)}...${txid.slice(-5)}`;
  };

  // Format addresses (show first 5 + last 5 chars)
  const formatAddress = (address) => {
    if (!address) return "";
    return `${address.slice(0, 8)}...${address.slice(-8)}`;
  };

  // Parse JSON string and format each address
  const renderAddresses = (addresses) => {
    if (!addresses) return "N/A";

    try {
      const parsedAddresses = JSON.parse(addresses); // Convert string to array

      if (!Array.isArray(parsedAddresses) || parsedAddresses.length === 0) {
        return "N/A";
      }

      return (
        <Tooltip title={parsedAddresses.join(", ")}>
          {parsedAddresses.map(formatAddress).slice(0,3).join(", ")} + {parsedAddresses.length - 3} more
        </Tooltip>
      )
    } catch (error) {
      console.error("Error parsing addresses:", error);
      return "Invalid data";
    }
  };

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
        <a
          href={`https://mempool.space/tx/${text}`}
          target="_blank"
          rel="noopener noreferrer"
        >
          {formatTxid(text)}
        </a>
      ),
    },
    {
      title: "Input Addresses",
      dataIndex: "tx_in_addr",
      key: "input_addresses",
      width: 300,
      render: renderAddresses, // Properly formats all addresses
    },
    {
        title: "Output Addresses",
        dataIndex: "tx_out_addr",
        key: "input_addresses",
        width: 300,
        render: renderAddresses, // Properly formats all addresses
    },
    {
      title: "Total Sent (BTC)",
      dataIndex: "total_sent",
      key: "total_sent",
      width: 150,
      render: (text) => Number(text).toFixed(3),
    },
    {
        title: "Total Fee (sats)",
        dataIndex: "fee_paid",
        key: "fee_paid",
        width: 150,
        render: (text) => Number(text).toFixed(3),
    },
    {
    title: "rel. Fee (sats/vbyte)",
    dataIndex: "fee_per_vbyte",
    key: "fee_per_vbyte",
    width: 150,
    render: (text) => Number(text).toFixed(3),
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
