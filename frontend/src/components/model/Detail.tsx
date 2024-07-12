import { useAppSelector } from "@/controller/hooks"
import { headStyle } from "@/theme/layout";
import { Button, Card, Descriptions, Space, Table } from "antd";
import { useRouter } from "next/router";

export const Detail = () => {
    const router = useRouter();
    const { model, modelVersions } = useAppSelector(state => state.model);
    const columns = [
        {
            title: "Version",
            dataIndex: "version",
            key: "verion"
        },
        {
            title: "Current stage",
            dataIndex: "current_stage",
            key: "current_stage"
        },
        {
            title: "Creation time",
            dataIndex: "creation_timestamp",
            key: "creation_timestamp",
            render: (_, record) => (
                new Date(parseInt(record.creation_timestamp)).toLocaleString()
            )
        },
        {
            title: "Updated time",
            dataIndex: "last_updated_timestamp",
            key: "last_updated_timestamp",
            render: (_, record) => (
                record.last_updated_timestamp ? new Date(parseInt(record.last_updated_timestamp)).toLocaleString() : "N/A"
            )
        },
        {
            title: "Status",
            dataIndex: "status",
            key: "status"
        },
        {
            title: "Run",
            dataIndex: "run_id",
            key: "run_id",
            render: (_, record) => (
                <Button type="link" onClick={() => router.push(`/run/${record.run_id}`)}>Link</Button>
            )
        },
        {
            title: "Actions",
            dataIndex: "action",
            key: "action",
            render: (_, record, index) => (
                <Space>
                    <Button type="primary" onClick={() => { }}>Generate DockerFile</Button>
                    <Button type="primary" onClick={() => { }}>Download</Button>
                    {/* <Button type="primary" onClick={() =>{}}>Download</Button> */}
                </Space>
            )
        },

    ]

    return (
        <>
            <Descriptions>
                <Descriptions.Item label={"Name"}>{model.name}</Descriptions.Item>
                <Descriptions.Item label={"Description"}>{model.description}</Descriptions.Item>
            </Descriptions>
            <Card title="All registered version" headStyle={headStyle}>
                <Table
                    columns={columns}
                    dataSource={modelVersions}
                />

            </Card>
        </>

    )
}