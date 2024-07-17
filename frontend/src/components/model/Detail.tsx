import { useAppDispatch, useAppSelector } from "@/controller/hooks";
import { setSelectedVersion } from "@/controller/model/modelSlice";
import { useModels } from "@/hooks/useModels";
import { useWebsocket } from "@/hooks/useWebsocket";
import { headStyle } from "@/theme/layout";
import { generatedZipCommand } from "@/utils/command-template";
import { timeAgo } from "@/utils/timeUtils";
import { LinkOutlined } from "@ant-design/icons";
import { Alert, Button, Card, Col, Descriptions, Divider, Form, Input, Modal, Row, Space, Table } from "antd";
import { useRouter } from "next/router";
import { useCallback, useState } from "react";

export const Detail = () => {
    const dispatch = useAppDispatch();
    const { downloadDockerFile } = useModels();
    const router = useRouter();
    const { model, modelVersions, selectedVersion } = useAppSelector(state => state.model);
    const { sendCommandToTrackingServer } = useWebsocket();
    const { downloadDockerFilesAction, buildAndPushDockerFileActions, getModelVersionsByNameAction, generateDockerFilesAction } = useAppSelector(state => state.process);
    const columns = [
        {
            title: "Version",
            dataIndex: "version",
            key: "verion"
        },
        // {
        //     title: "Current stage",
        //     dataIndex: "current_stage",
        //     key: "current_stage"
        // },
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
                record.last_updated_timestamp ? <span title={`${new Date(parseInt(record.last_updated_timestamp)).toLocaleString()}`}>{timeAgo(record.last_updated_timestamp)}</span> : "N/A"
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
            render: (_, record, index) => (
                <Button icon={<LinkOutlined />} key={`link-${index}`} type="primary" onClick={() => router.push(`/run/${record.run_id}`)}></Button>
            )
        },
        {
            title: "Docker actions",
            dataIndex: "action",
            key: "action",
            render: (_, record, index) => (
                <Space key={`actions-${index}`}>
                    <Button type="primary" onClick={() => {
                        dispatch(setSelectedVersion(record.version));
                        showModal();
                    }}>Generate | Download</Button>
                    <Button type="primary" onClick={() => {
                        dispatch(setSelectedVersion(record.version));
                        showBuildModal();
                    }}>Build | Push</Button>
                </Space>
            )
        },

    ]
    const [isModalOpen, setIsModalOpen] = useState(false);

    const showModal = () => {
        setIsModalOpen(true);
    };

    const handleOk = () => {
        setIsModalOpen(false);
    };

    const handleCancel = () => {
        setIsModalOpen(false);
    };

    const handleZip = useCallback(() => {
        sendCommandToTrackingServer(
            generatedZipCommand(model, selectedVersion),
            "runlog"
        )
    }, [selectedVersion])
    const handleDownload = useCallback(() => {
        downloadDockerFile(model.name, selectedVersion);
    }, [selectedVersion])



    const [isBuildModalOpen, setIsBuildModalOpen] = useState(false);

    const showBuildModal = () => {
        setIsBuildModalOpen(true);
    };

    const handleOkBuildModal = () => {
        setIsBuildModalOpen(false);
    };

    const handleCancelBuildModal = () => {
        setIsBuildModalOpen(false);
    };
    return (
        <>
            <Descriptions>
                <Descriptions.Item label={"Name"}>{model.name}</Descriptions.Item>
                <Descriptions.Item label={"Description"}>{model.description}</Descriptions.Item>
            </Descriptions>
            <Card title="All registered version" headStyle={headStyle}>
                <Table
                    loading={getModelVersionsByNameAction}
                    columns={columns}
                    dataSource={modelVersions}
                />

            </Card>

            <Modal title="Do you want to generate a Docker image for this AI model?" open={isModalOpen} footer={[]} onOk={handleOk} onCancel={handleCancel}>
                <textarea placeholder="When you see --end-process--, that means the generation process has ended." id={`runlog`}
                    style={{ marginTop: 10, borderRadius: 10, height: "95%", width: "100%", backgroundColor: "#333", color: "whitesmoke", padding: "10px" }} />
                <Divider />
                <Row gutter={12}>
                    <Col span="12">
                        <Button block size="large" type="primary" loading={generateDockerFilesAction} onClick={() => handleZip()}>Yes</Button>
                    </Col>
                    <Col span="12">
                        <Button block size="large" type="primary" loading={downloadDockerFilesAction} onClick={() => handleDownload()}>
                            {downloadDockerFilesAction ? "Downloading..." : "Download docker files"}
                        </Button>
                    </Col>
                </Row>


            </Modal>

            <Modal title="Build and push Docker images to Docker Hub" open={isBuildModalOpen} footer={[]} onOk={handleOkBuildModal} onCancel={handleCancelBuildModal}>
                <Card>
                    <Alert showIcon={true} type="info" message="Please be patient, as building Docker images and publishing them to Docker Hub can take several minutes. The completion time depends on the size of your Docker images." />
                    <Divider />
                    <Form onFinish={() => { }} layout="vertical">
                        <Row gutter={12}>
                            <Col span={12}>
                                <Form.Item label={"Username"} name={"username"} rules={[{ required: true, message: "Missing username" }]}>
                                    <Input size="large" type="text" />
                                </Form.Item>
                            </Col>
                            <Col span={12}>
                                <Form.Item label={"Password"} name={"password"} rules={[{ required: true, message: "Missing password" }]}>
                                    <Input size="large" type="password" />
                                </Form.Item>
                            </Col>
                        </Row>
                        <Row gutter={12}>
                            <Col span={12}>
                                <Form.Item label={"Repository"} name={"repository"} rules={[{ required: true, message: "Missing repository" }]}>
                                    <Input size="large" type="text" />
                                </Form.Item>
                            </Col>
                            <Col span={12}>
                                <Form.Item label={"Version"} name={"version"} rules={[{ required: true, message: "Missing version" }]}>
                                    <Input size="large" type="text" />
                                </Form.Item>
                            </Col>
                        </Row>

                        <Button block size="large" type="primary" htmlType="submit" loading={buildAndPushDockerFileActions} onClick={() => handleDownload()}>
                            {buildAndPushDockerFileActions ? "Processing..." : "Build and Push"}
                        </Button>

                    </Form>

                </Card>



            </Modal>
        </>

    )
}