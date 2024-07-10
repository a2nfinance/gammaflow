import { useAppSelector } from "@/controller/hooks";
import { useExperiments } from "@/hooks/useExperiments"
import { headStyle } from "@/theme/layout";
import { Button, Card, Space, Table, Tag } from "antd";
import { useRouter } from "next/router";
import { useEffect } from "react"

export const Detail = () => {
    const { runs } = useAppSelector(state => state.experiment);
    const { searchRunByExperimentId } = useExperiments();
    const router = useRouter();

    useEffect(() => {
        if (router.query?.id) {
            console.log(router?.query?.id)
            searchRunByExperimentId(router.query?.id?.toString());
        }
    }, [router.query?.id]);

    const columns = [
        {
            title: 'Name',
            dataIndex: 'run_name',
            key: 'run_name',
        },
        {
            title: "status",
            dataIndex: "status",
            key: "status"
        },
        {
            title: "Start time",
            dataIndex: "start_time",
            key: "start_time",
            render: (_, record) => (
                new Date(parseInt(record.start_time)).toLocaleString()
            )
        },
        {
            title: "End time",
            dataIndex: "end_time",
            key: "end_time",
            render: (_, record) => (
                new Date(parseInt(record.end_time) * 1000).toLocaleString()
            )
        },
        {
            title: "Stage",
            dataIndex: "lifecycle_stage",
            key: "lifecycle_stage",
            render: (_, record) => (
                <Tag color={record.status ? "green" : "red"}>{record.status ? "active" : "deleted"} </Tag>
            )
        },
        {
            title: "Actions",
            dataIndex: "action",
            key: "action",
            render: (_, record, index) => (
                <Space>
                    <Button type="default">Delete</Button>
                    <Button type="primary" onClick={() => router.push(`/run/${record.run_id}`)}>Details</Button>
                </Space>
            )
        },

    ]
    return (
        <Card title="Runs" headStyle={headStyle} extra={
            <Space>
                  <Button type='primary' size="large">New run</Button>
            </Space>
           
          }>
            <Table
                columns={columns}
                dataSource={runs?.map(r => r.info)}
            />

        </Card>

    )
}