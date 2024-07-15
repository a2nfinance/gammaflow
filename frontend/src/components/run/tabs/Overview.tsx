import { Card, Descriptions, Divider, Table, Tag } from "antd";

export const Overview = ({ run }) => {
    const paramsColumns = [
        {
            title: 'Parameter',
            dataIndex: 'key',
            key: 'key',
        },
        {
            title: "Value",
            dataIndex: "value",
            key: "value",
        },

    ]

    const metricsColumns = [
        {
            title: 'Metric',
            dataIndex: 'key',
            key: 'key',
        },
        {
            title: "Value",
            dataIndex: "value",
            key: "value"
        },

    ]

    return (
        <>
            <Descriptions key={"run-details"} title={"Details"} column={3}>
                <Descriptions.Item label="Created at">
                    {new Date(parseInt(run.info.start_time)).toLocaleString()}
                </Descriptions.Item>
                <Descriptions.Item label="Created by">
                    {run.info.user_id}
                </Descriptions.Item>
                <Descriptions.Item label="Experiment ID">
                    <Tag onClick={() => window.open(`/experiments/runs/${run.info.experiment_id}`, "_blank")}>{run.info.experiment_id}</Tag>
                </Descriptions.Item>
                <Descriptions.Item label="Status">
                    {run.info.status}
                </Descriptions.Item>
                <Descriptions.Item label="Run ID">
                    {run.info.run_uuid}
                </Descriptions.Item>
                <Descriptions.Item label="Duration">
                    {((parseInt(run.info.end_time) - parseInt(run.info.start_time)) / 1000).toFixed(1)}s
                </Descriptions.Item>
            </Descriptions>
            <Divider />
            <Descriptions key={"parameters_metrics"} layout="vertical" column={3}>
                <Descriptions.Item label={`Parameters (${run.data.params?.length ?? 0})`}>
                    <Card>
                        <Table
                            style={{ minWidth: 300 }}
                            size="large"
                            pagination={false}
                            columns={paramsColumns}
                            dataSource={run.data.params}
                        />
                    </Card>

                </Descriptions.Item>
                <Descriptions.Item label={`Model metrics (${run.data.metrics?.filter(m => !m.key.includes("system/")).length ?? 0})`}>
                    <Card>
                        <Table
                            style={{ minWidth: 300 }}
                            pagination={false}
                            columns={metricsColumns}
                            dataSource={run.data.metrics?.filter(m => !m.key.includes("system/"))}
                        />
                    </Card>
                </Descriptions.Item>
                <Descriptions.Item label={`System metrics (${run.data.metrics?.filter(m => m.key.includes("system/")).length ?? 0})`}>
                    <Card>
                        <Table
                            style={{ minWidth: 300 }}
                            pagination={false}
                            columns={metricsColumns}
                            dataSource={run.data.metrics?.filter(m => m.key.includes("system/"))}
                        />
                    </Card>
                </Descriptions.Item>

            </Descriptions>

        </>
    )
}