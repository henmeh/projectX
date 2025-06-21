import { useState } from "react";
import { Layout, Menu, Button, Drawer } from "antd";
import { MenuOutlined } from "@ant-design/icons";
import WhaleTransactions from "./components/WhaleTransactions";
import FeeHistogram from "./components/FeeHistogram";

const { Header, Sider, Content, Footer } = Layout;

export default function App() {
  const [collapsed, setCollapsed] = useState(false); // Track sidebar state
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [selectedMenu, setSelectedMenu] = useState("whale-transactions"); // Default view

  const menuItems = [
    { key: "whale-transactions", label: "Whale Transactions" },
    { key: "fee-histogram", label: "Fee Statistics" },
  ];

  return (
    <Layout>
      {/* Desktop Sidebar */}
      <Sider
        breakpoint="lg"
        collapsedWidth="0"
        onCollapse={(collapsed) => setCollapsed(collapsed)}
        style={{ minHeight: "100vh" }}
      >
        <div style={{ color: "white", textAlign: "center", padding: "10px", fontSize: "18px" }}>
          Bitcoin Analytics
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[selectedMenu]}
          onClick={(e) => setSelectedMenu(e.key)}
          items={menuItems}
        />
      </Sider>

      <Layout>
        {/* Mobile Header - Show menu button only if sidebar is fully collapsed */}
        {collapsed && (
          <Header style={{ background: "#001529", padding: "0 16px", display: "flex", alignItems: "left" }}>
            collapsed ? <Button type="text" icon={<MenuOutlined />} onClick={() => setDrawerVisible(true)} style={{ color: "white" }} />
            <span style={{ color: "white", marginLeft: "16px", fontSize: "20px" }}>Bitcoin Analytics</span>
          </Header>
        )}

        {!collapsed && (
          <Header style={{ background: "#001529", padding: "0 16px", display: "flex", alignItems: "center" }}>
            <span style={{ color: "white", marginLeft: "16px", fontSize: "20px" }}>Bitcoin Analytics</span>
          </Header>
        )}

        {/* Mobile Drawer Menu */}
        <Drawer title="Menu" placement="left" onClose={() => setDrawerVisible(false)} open={drawerVisible}>
          <Menu
            mode="vertical"
            selectedKeys={[selectedMenu]}
            onClick={(e) => {
              setSelectedMenu(e.key);
              setDrawerVisible(false); // Close drawer after selection
            }}
            items={menuItems}
          />
        </Drawer>

        {/* Main Content - Show only the selected section */}
        <Content style={{ padding: "20px" }}>
          {selectedMenu === "whale-transactions" && <WhaleTransactions />}
          {selectedMenu === "fee-histogram" && <FeeHistogram />}
        </Content>

        <Footer style={{ textAlign: "center" }}>©2025 Bitcoin Dashboard</Footer>
      </Layout>
    </Layout>
  );
}
