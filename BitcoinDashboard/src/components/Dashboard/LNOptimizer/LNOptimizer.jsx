import React, { useState } from 'react';
import { App, Card, Form, InputNumber, Button, Table, message } from 'antd';
import axios from 'axios';  // npm i axios

const LNOptimizer = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const { message: msg } = App.useApp();  // Use dynamic message from App context

  const onFinish = async (values) => {
    setLoading(true);
    try {
      const url = `http://127.0.0.1:8000/optimize-ln-channel/${values.channel_size_vb}_${values.duration_days}`;
      const response = await axios.get(url);
      console.log(response)
      setResult(response.data);
    } catch (err) {
      msg.error('Failed to optimize: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const columns = [
    { title: 'Action', dataIndex: 'action' },
    { title: 'Optimal Time', dataIndex: 'time' },
    { title: 'Predicted Fee (sat/vB)', dataIndex: 'predicted_fee' },
    { title: 'Cost (sat)', dataIndex: 'cost' },
    { title: 'Savings (%)', dataIndex: 'savings_pct' },
  ];

  const data = result ? [
    { action: 'Open Channel', ...result.optimal_open },
    { action: 'Close Channel', ...result.optimal_close },
  ] : [];

  return (
    <App>  {/* Wrap with App for context */}
      <Card title="Lightning Channel Optimizer">
        <Form form={form} onFinish={onFinish} layout="vertical" initialValues={{ channel_size_vb: 140, duration_days: 30 }}>
          <Form.Item name="channel_size_vb" label="Channel Tx Size (vB)">
            <InputNumber min={100} max={1000} />
          </Form.Item>
          <Form.Item name="duration_days" label="Channel Duration (days)">
            <InputNumber min={1} max={365} />
          </Form.Item>
          <Button type="primary" htmlType="submit" loading={loading}>Optimize</Button>
        </Form>
        {result && (
          <>
            <Table columns={columns} dataSource={data} pagination={false} style={{ marginTop: 16 }} />
            <p>Total Savings: {result.total_savings} sat (Current Fee: {result.current_fee} sat/vB)</p>
          </>
        )}
      </Card>
    </App>
  );
};

export default LNOptimizer;