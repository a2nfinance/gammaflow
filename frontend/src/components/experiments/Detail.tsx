import { useAppSelector } from "@/controller/hooks";
import { useExperiments } from "@/hooks/useExperiments"
import { headStyle } from "@/theme/layout";
import { Button, Card, Modal, Space, Table, Tag } from "antd";
import { useRouter } from "next/router";
import { useEffect, useState } from "react"
import { NewRunForm } from "../run/Form";

export const Detail = () => {
    const { runs } = useAppSelector(state => state.experiment);
    const { searchRunByExperimentId } = useExperiments();
    const router = useRouter();
    const [isModalOpen, setIsModalOpen] = useState(false);
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
                record.end_time ? new Date(parseInt(record.end_time)).toLocaleString() : "N/A"
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



    const showModal = () => {
        setIsModalOpen(true);
    };

    const handleOk = () => {
        setIsModalOpen(false);
    };

    const handleCancel = () => {
        setIsModalOpen(false);
    };
    return (
        <Card title="Runs" headStyle={headStyle} extra={
            <Space>
                <Button type='primary' size="large" onClick={showModal}>New run</Button>
            </Space>

        }>
            <Table
                columns={columns}
                dataSource={runs?.map(r => r.info)}
            />
            <Modal style={{minWidth: 900}} title="New run" open={isModalOpen} footer={[]}   onOk={handleOk} onCancel={handleCancel}>
                <NewRunForm />
            </Modal>
        </Card>

    )
}