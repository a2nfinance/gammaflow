import { useAppSelector } from "@/controller/hooks";
import { useExperiments } from "@/hooks/useExperiments"
import { headStyle } from "@/theme/layout";
import { Button, Card, Col, Form, Input, Modal, Row, Select, Space, Table, Tag } from "antd";
import { useRouter } from "next/router";
import { useEffect, useState } from "react"
import { NewRunForm } from "../run/Form";
import { SearchOutlined } from "@ant-design/icons";
import { MdHelp } from "react-icons/md";

export const Detail = () => {
    const { runs } = useAppSelector(state => state.experiment);
    const { searchRunByExperimentIDAction } = useAppSelector(state => state.process);
    const { searchRunByExperimentId } = useExperiments();
    const router = useRouter();
    const [isModalOpen, setIsModalOpen] = useState(false);
    useEffect(() => {
        if (router.query?.id) {
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
            title: "Start time",
            dataIndex: "start_time",
            key: "start_time",
            render: (_, record) => (
                new Date(parseInt(record.start_time)).toLocaleString()
            )
        },
        {
            title: "Duration",
            dataIndex: "duration",
            key: "duration",
            render: (_, record) => (
                record.end_time ? (((parseInt(record.end_time) - parseInt(record.start_time)) / 1000).toFixed(2) + "s") : "N/A"
            )
        },
        {
            title: "State",
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

    const onFinishSearchForm = (values: FormData) => {
        // Filter feature at here
    }
    return (
        <Card title="Expirement runs" headStyle={headStyle} extra={
            <Space>
                <Button type='primary' size="large" onClick={showModal}>New run</Button>
            </Space>

        }>
            <Form
            onFinish={onFinishSearchForm}
            initialValues={{
                "time_created": 0,
                "state": 1,
                "sort_by": 1

            }}>
                <Row gutter={12}>
                    <Col span={10}>
                        <Form.Item>
                            <Input size="large" addonBefore={<SearchOutlined />} addonAfter={<MdHelp />} placeholder="metrics.rmse < 1 and params.model = 'tree'" />
                        </Form.Item>
                    </Col>
                    <Col span={4}>
                        <Form.Item name={"time_created"}>
                            <Select size="large" options={[
                                { label: "Time created", value: 0 },
                                { label: "Last hour", value: 1 },
                                { label: "Last 24 hours", value: 2 },
                                { label: "Last 7 days", value: 3 },
                                { label: "Last 30 days", value: 4 },
                                { label: "Last year", value: 5 },
                            ]} />
                        </Form.Item>
                    </Col>
                    <Col span={4}>
                        <Form.Item name={"state"}>
                            <Select size="large" options={[
                                { label: "State: active", value: 1 },
                                { label: "State: delete", value: 2 },
                            ]} />
                        </Form.Item>
                    </Col>
                    <Col span={4}>
                        <Form.Item name={"sort_by"}>
                            <Select size="large" options={[
                                { label: "Sort: Created", value: 1 },
                                { label: "Sort: Run name", value: 2 },
                                { label: "Sort: User", value: 3 }
                            ]} />
                        </Form.Item>
                    </Col>
                    <Col span={2}>
                        <Button type="primary" htmlType="submit" size="large" icon={<SearchOutlined />}></Button>
                    </Col>
                </Row>
            </Form>
            <Table
                loading={searchRunByExperimentIDAction}
                columns={columns}
                dataSource={runs?.map(r => r.info)}
            />
            <Modal style={{ minWidth: 1024 }} title="New experiment run" open={isModalOpen} footer={[]} onOk={handleOk} onCancel={handleCancel}>
                <NewRunForm />
            </Modal>
        </Card>

    )
}