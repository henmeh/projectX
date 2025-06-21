import { useEffect, useState } from "react";
import { Bar } from "@ant-design/plots";
import { fetchFeeEstimation, fetchFeeHistogram, fetchMempoolCongestion } from "../utils/api";
import { Card, Spin, Typography } from "antd";

const { Text } = Typography;

export default function FeeHistogram() {
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [congestion, setCongestion] = useState(null);
    const [feeEstimation, setFeeEstimation] = useState(null);

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

                // Fetch fee estimation data
                const feeEstimationResponse = await fetchFeeEstimation();
                if (feeEstimationResponse) {
                    setFeeEstimation(feeEstimationResponse);
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
        <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
            {/* Fee Estimation Cards - Placed in a row */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: "20px", justifyContent: "center" }}>
                <Card style={{ flex: 1, minWidth: "250px", textAlign: "center" }}>
                    <Text strong style={{ fontSize: "12px" }}>
                        Fast Transaction: {feeEstimation?.fast_fee} sats/vbyte
                    </Text>
                </Card>

                <Card style={{ flex: 1, minWidth: "250px", textAlign: "center" }}>
                    <Text strong style={{ fontSize: "12px" }}>
                        Medium Transaction: {feeEstimation?.medium_fee} sats/vbyte
                    </Text>
                </Card>

                <Card style={{ flex: 1, minWidth: "250px", textAlign: "center" }}>
                    <Text strong style={{ fontSize: "12px" }}>
                        Slow Transaction: {feeEstimation?.low_fee} sats/vbyte
                    </Text>
                </Card>  
            </div>

            {/* Fee Histogram & Mempool Congestion Cards - Placed below */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: "20px" }}>
                <Card title="Fee Histogram" style={{ flex: 1, minWidth: "300px" }}>
                    {loading ? <Spin size="large" /> : <Bar {...config} />}
                </Card>

                <Card title="Mempool Congestion" style={{ flex: 1, minWidth: "300px" }}>
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
        </div>
    );
}
