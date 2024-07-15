import { useAppSelector } from "@/controller/hooks";
import { useModels } from "@/hooks/useModels";
import { useConnectWallet } from "@web3-onboard/react";
import { Button, Card, Table } from "antd";
import { useRouter } from "next/router";
import { useEffect } from "react";

export const DeployedVersion = () => {
    const [{ wallet }] = useConnectWallet();
    const router = useRouter();
    const { deployedVersions } = useAppSelector(state => state.model);
    const { getModelVersionsByAddress } = useModels();
    useEffect(() => {
        if (wallet?.accounts[0].address) {
            getModelVersionsByAddress()
        }

    }, [wallet?.accounts[0].address])


    const columns = [
        {
            title: "Version",
            dataIndex: "version",
            key: "verion"
        },
        {
            title: "Model name",
            dataIndex: "name",
            key: "name",
            render: (_, record, index) => (
                <Button key={`model-link-${index}`} type="dashed" onClick={() => router.push(`/models/detail/${record.name}`)}>{record.name}</Button>
            )
        },
        {
            title: "Docker image",
            dataIndex: "docker_image",
            key: "docker_image",
            render: (_, record) => (
                <span>{JSON.parse(record.tags.filter(t => t.key === "deployment_info")[0].value).docker_image}</span>
            )
        },
        {
            title: "Inference endpoint",
            dataIndex: "inference_endpoint",
            key: "inference_endpoint",
            render: (_, record, index) => {
                let endpoint = JSON.parse(record.tags.filter(t => t.key === "deployment_info")[0].value).inference_endpoint;
                return <Button key={`inference-link-${index}`} type="dashed" onClick={() => window.open(endpoint, "_blank")}>
                    {endpoint}
                </Button>
            }

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
                <Button key={`link-${index}`} type="dashed" onClick={() => router.push(`/run/${record.run_id}`)}>Link</Button>
            )
        }

    ]
    return (
        <Card title="All deployed versions">
            <Table
                columns={columns}
                dataSource={deployedVersions}
            />

        </Card>
    )
}