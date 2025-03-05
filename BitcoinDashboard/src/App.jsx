import { Layout, Row, Col, Card } from "antd";
import WhaleTransactions from "./components/WhaleTransactions";
import FeeHistogram from "./components/FeeHistogram";

const { Header, Content, Footer } = Layout;

function App() {
  return (
    <Layout>
      <Header style={{ color: "white", fontSize: "20px" }}>Bitcoin Analytics</Header>
      <Content style={{ padding: "20px" }}>
        <Row gutter={[16, 16]} justify="space-around">
          <Col xs={24} sm={12} md={12} lg={12}>
            <Card title="Whale Transactions" bordered={false}>
              <WhaleTransactions />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={12} lg={10}>
            <Card title="Fee Histogram" bordered={false}>
              <FeeHistogram />
            </Card>
          </Col>
        </Row>
      </Content>
      <Footer style={{ textAlign: "center" }}>©2025 Bitcoin Dashboard</Footer>
    </Layout>
  );
}

export default App;
