import React, { useState, useEffect } from 'react';
import { Table, Card, InputNumber, Typography } from 'antd';
import { fetchWhaleTransactions } from '../../services/api';
import './Dashboard.css';
import './WhaleTransactions.css'


const { Title } = Typography;

const WhaleTransactions = () => {
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [minBtc, setMinBtc] = useState(10.0);
  
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      const data = await fetchWhaleTransactions(minBtc);
      setTransactions(data);
      setLoading(false);
    };
    
    loadData();
  }, [minBtc]);
  
  const columns = [
    {
      title: 'Timestamp',
      dataIndex: 'timestamp',
      key: 'timestamp',
      sorter: (a, b) => new Date(a.timestamp) - new Date(b.timestamp),
    },
    {
      title: 'Transaction ID',
      dataIndex: 'txid',
      key: 'txid',
      render: (text) => <span className="txid">{text.substring(0, 16)}...</span>,
    },
    {
      title: 'Amount (BTC)',
      dataIndex: 'total_sent',
      key: 'total_sent',
      sorter: (a, b) => a.total_sent - b.total_sent,
      render: (value) => value.toFixed(2),
    },
    {
      title: 'Fee (sat/vB)',
      dataIndex: 'fee_per_vbyte',
      key: 'fee_per_vbyte',
      sorter: (a, b) => a.fee_per_vbyte - b.fee_per_vbyte,
    },
  ];
  
  return (
    <Card 
      className="dashboard-card"
      title={<Title level={4} style={{ margin: 0 }}>Whale Transactions</Title>}
      extra={
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <span style={{ marginRight: 8 }}>Min BTC:</span>
          <InputNumber 
            min={0} 
            max={21000000} 
            value={minBtc}
            onChange={setMinBtc}
          />
        </div>
      }
    >
      <Table 
        dataSource={transactions} 
        columns={columns} 
        loading={loading}
        pagination={{ pageSize: 10 }}
        rowKey={record => record.id || record.txid}
        scroll={{ x: true }}
      />
    </Card>
  );
};

export default WhaleTransactions;