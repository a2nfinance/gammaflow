import { useAppSelector } from "@/controller/hooks";
import { useModels } from "@/hooks/useModels";
import { headStyle } from "@/theme/layout";
import { timeAgo } from "@/utils/timeUtils";
import { useConnectWallet, useWallets } from "@web3-onboard/react";
import { Button, Card, Space, Table } from "antd";
import { useRouter } from "next/router";
import { useEffect, useState } from "react";

export const List = () => {
    const [{ wallet }] = useConnectWallet();
    const { modelList } = useAppSelector(state => state.model);
    const { searchRegisteredModels } = useModels();
    const router = useRouter();
    useEffect(() => {
        if (wallet?.accounts?.length) {
            searchRegisteredModels();
        }
    }, [wallet?.accounts?.length]);

    const columns = [
        {
            title: 'Name',
            dataIndex: 'name',
            key: 'name',
        },
        {
            title: "Latest version",
            dataIndex: "version",
            key: "verion",
            render: (_, record) => (
                <span>{record.latest_versions.length}</span>
            )
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
                record.last_updated_timestamp ? <span title={`${new Date(parseInt(record.last_updated_timestamp)).toLocaleString()}`}>{timeAgo(record.last_updated_timestamp)}</span> : "N/A"
            )
        },
        {
            title: "Actions",
            dataIndex: "action",
            key: "action",
            render: (_, record, index) => (
                <Space>
                    <Button type="primary" onClick={() => router.push(`/models/detail/${record.name}`)}>Details</Button>
                </Space>
            )
        },

    ]

    return (
        <Card title="My registered models" headStyle={headStyle}>
            <Table
                columns={columns}
                dataSource={modelList}
            />

        </Card>

    )
}