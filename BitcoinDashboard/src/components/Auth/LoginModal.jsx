import React, { useState } from 'react';
import { Modal, Form, Input, Button, message } from 'antd';
import { useAppContext } from '../../context/AppContext';

const LoginModal = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const { loginVisible, setLoginVisible, setIsLoggedIn } = useAppContext();
  
  const handleLogin = async (values) => {
    setLoading(true);
    try {
      // Log the values for debugging
      console.log('Login attempt with:', values);
      
      // Simulate API call
      await new Promise((resolve, reject) => {
        setTimeout(() => {
          // Mock authentication logic
          if (values.username === 'admin' && values.password === 'password') {
            resolve();
          } else {
            reject(new Error('Invalid credentials'));
          }
        }, 1000);
      });
      
      setIsLoggedIn(true);
      setLoginVisible(false);
      message.success('Login successful!');
    } catch (error) {
      message.error(`Login failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  const handleCancel = () => {
    form.resetFields();
    setLoginVisible(false);
  };
  
  return (
    <Modal
      title="Login"
      open={loginVisible}
      onCancel={handleCancel}
      footer={null}
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleLogin}
      >
        <Form.Item
          name="username"
          label="Username"
          rules={[{ required: true, message: 'Please input your username!' }]}
        >
          <Input />
        </Form.Item>
        
        <Form.Item
          name="password"
          label="Password"
          rules={[{ required: true, message: 'Please input your password!' }]}
        >
          <Input.Password />
        </Form.Item>
        
        <Form.Item>
          <Button 
            type="primary" 
            htmlType="submit" 
            loading={loading}
            block
          >
            Log in
          </Button>
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default LoginModal;