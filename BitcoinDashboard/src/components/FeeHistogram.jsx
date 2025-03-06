import { useEffect, useState } from "react";
import { Bar } from "@ant-design/plots";
import { fetchFeeHistogram, fetchMempoolCongestion } from "../utils/api";
import { Card, Spin, Typography } from "antd";

const { Text } = Typography;

export default function FeeHistogram() {
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [congestion, setCongestion] = useState(null);

    useEffect(() => {
        async function loadData() {
            try {
                // Fetch fee histogram
                const response = await fetchFeeHistogram();
                if (response && response.histogram) {
                    const formattedData = response.histogram.map(([feeRate, vsize]) => ({
                        feeRate,
                        vsize
                    }));
                    setData(formattedData);
                }

                // Fetch mempool congestion data
                const congestionResponse = await fetchMempoolCongestion();
                if (congestionResponse) {
                    setCongestion(congestionResponse);
                }
            } catch (error) {
                console.error("Error fetching data:", error);
            }
            setLoading(false);
        }

        loadData();

        // Set interval to fetch data every 60 seconds
        const interval = setInterval(loadData, 60000);
    
        return () => clearInterval(interval);
    }, []);

    const config = {
        data,
        xField: "vsize",
        yField: "feeRate",
        seriesField: "feeRate",
        colorField: "feeRate",
        xAxis: {
            label: { autoRotate: true },
            title: { text: "Virtual Size (vsize)", style: { fontSize: 14 } },
        },
        yAxis: {
            title: { text: "Fee Rate (sat/vB)", style: { fontSize: 14 } },
        },
        tooltip: { showMarkers: false },
        meta: {
            feeRate: { alias: "Fee Rate (sat/vB)" },
            vsize: { alias: "Virtual Size (vB)" },
        },
    };

    return (
        <div style={{ display: "flex", flexWrap: "wrap", gap: "20px" }}>
            {/* Fee Histogram Card */}
            <Card
                title="Fee Histogram"
                style={{ flex: 1, minWidth: "300px" }}
            >
                {loading ? (
                    <Spin size="large" />
                ) : (
                    <Bar {...config} />
                )}
            </Card>

            {/* Mempool Congestion Card */}
            <Card
                title="Mempool Congestion"
                style={{ flex: 1, minWidth: "300px" }}
            >
                {loading ? (
                    <Spin size="large" />
                ) : (
                    congestion && (
                        <div>
                            <p>
                                <Text strong style={{ fontSize: "16px" }}>
                                    Mempool Congestion: {congestion.congestion_status}
                                </Text>
                            </p>
                            <p>
                                <Text strong style={{ fontSize: "16px" }}>
                                    Total Size: {congestion.total_vsize} vB
                                </Text>
                            </p>
                        </div>
                    )
                )}
            </Card>
        </div>
    );
}
