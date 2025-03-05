import { useEffect, useState } from "react";
import { Bar } from "@ant-design/plots";
import { fetchFeeHistogram } from "../utils/api";
import { Card, Spin } from "antd";

export default function FeeHistogram() {
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        async function loadData() {
            try {
                const response = await fetchFeeHistogram();
                if (response && response.histogram) {
                    const formattedData = response.histogram.map(([feeRate, vsize]) => ({
                        feeRate,
                        vsize
                    }));
                    setData(formattedData);
                }
            } catch (error) {
                console.error("Error fetching fee histogram:", error);
            }
            setLoading(false);
        }
        loadData();
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
        <Card title="Fee Histogram" style={{ marginBottom: 20 }}>
            {loading ? <Spin size="large" /> : <Bar {...config} />}
        </Card>
    );
}
