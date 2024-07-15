import { Card, Col, Row, Statistic } from "antd"
import { CustomActiveShapePieChart } from "../charts/CustomActiveShapPieChart"
import { ArrowDownOutlined, ArrowUpOutlined } from "@ant-design/icons";

export const SystemMetrics = ({ run }) => {

    const diskUsageData = [
        { name: 'Usage', value: parseFloat(run.data.metrics?.filter(m => m.key === "system/disk_usage_percentage")?.[0].value) },
        { name: 'Remain', value: 100 - parseFloat(run.data.metrics?.filter(m => m.key === "system/disk_usage_percentage")?.[0].value) },
    ];

    const cpuUsageData = [
        { name: 'Usage', value: parseFloat(run.data.metrics?.filter(m => m.key === "system/cpu_utilization_percentage")?.[0].value) },
        { name: 'Remain', value: 100 - parseFloat(run.data.metrics?.filter(m => m.key === "system/cpu_utilization_percentage")?.[0].value) },
    ];

    const memoryUsageData = [
        { name: 'Usage', value: parseFloat(run.data.metrics?.filter(m => m.key === "system/system_memory_usage_percentage")?.[0].value) },
        { name: 'Remain', value: 100 - parseFloat(run.data.metrics?.filter(m => m.key === "system/system_memory_usage_percentage")?.[0].value) },
    ];

    const disk_usage_megabytes = run.data.metrics?.filter(m => m.key === "system/disk_usage_megabytes")?.[0].value
    const system_memory_usage_megabytes = run.data.metrics?.filter(m => m.key === "system/system_memory_usage_megabytes")?.[0].value
    const network_transmit_megabytes = run.data.metrics?.filter(m => m.key === "system/network_transmit_megabytes")?.[0].value
    return (
        <>
            <Row gutter={8}>
                <Col span={8} style={{ minHeight: 500, minWidth: 400 }}>
                    <CustomActiveShapePieChart
                        data={diskUsageData}
                        width={400}
                        height={400}
                        innerRadius={80}
                        outerRadius={100}
                        fill1={"#8884d8"}
                        fill2={"green"}
                        title={"Disk"}
                    />
                </Col>
                <Col span={8} style={{ minHeight: 500, minWidth: 400 }}>
                    <CustomActiveShapePieChart
                        data={cpuUsageData}
                        width={400}
                        height={400}
                        innerRadius={80}
                        outerRadius={100}
                        fill1={"#8884d8"}
                        fill2={"blue"}
                        title={"CPU"}
                    />
                </Col>
                <Col span={8} style={{ minHeight: 500, minWidth: 400 }}>
                    <CustomActiveShapePieChart
                        data={memoryUsageData}
                        width={400}
                        height={400}
                        innerRadius={80}
                        outerRadius={100}
                        fill1={"#8884d8"}
                        fill2={"yellow"}
                        title={"Memory"}
                    />
                </Col>
            </Row>
            <Row gutter={8}>
                <Col span={8}>
                    <Card bordered={false}>
                        <Statistic
                            style={{ textAlign: "center" }}
                            title="Disk usage megabytes"
                            value={parseFloat(disk_usage_megabytes)}
                            precision={2}
                            valueStyle={{ color: '#3f8600' }}
                            prefix={<ArrowUpOutlined />}
                            suffix=""
                        />
                    </Card>
                </Col>
                <Col span={8}>
                    <Card bordered={false}>
                        <Statistic
                            style={{ textAlign: "center" }}
                            title="System memory usage megabytes"
                            value={parseFloat(system_memory_usage_megabytes)}
                            precision={2}
                            valueStyle={{ color: 'blue' }}
                            prefix={<ArrowUpOutlined />}
                            suffix=""
                        />
                    </Card>
                </Col>
                <Col span={8}>
                    <Card bordered={false}>
                        <Statistic
                            style={{ textAlign: "center" }}
                            title="Network transmit megabytes"
                            value={parseFloat(network_transmit_megabytes)}
                            precision={2}
                            valueStyle={{ color: 'orange' }}
                            prefix={<ArrowUpOutlined />}
                            suffix=""
                        />
                    </Card>
                </Col>
            </Row>
        </>
    )
}