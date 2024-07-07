import {
    AppstoreOutlined,
    GithubOutlined,
    MenuFoldOutlined,
    MenuUnfoldOutlined
} from '@ant-design/icons';
import { Button, Form, Image, Layout, Menu, Space, theme } from 'antd';
import { useRouter } from 'next/router';
import React, { useState } from "react";
import { FaSuperscript } from "react-icons/fa";
import { SlOrganization } from "react-icons/sl";
import { AiOutlineExperiment, AiTwotoneExperiment } from "react-icons/ai";
import { LuBrainCircuit } from "react-icons/lu";
import { CiPlay1 } from "react-icons/ci";
import { GrDeploy } from "react-icons/gr";
import { ConnectWallet } from './common/ConnectWallet';
// import { ConnectWallet } from './common/ConnectWallet';
const { Header, Sider, Content, Footer } = Layout;

interface Props {
    children: React.ReactNode | React.ReactNode[];
}

export const LayoutProvider = (props: Props) => {
    const [collapsed, setCollapsed] = useState(false);
    const router = useRouter();
    const {
        token: { colorBgContainer },
    } = theme.useToken();

    return (
        <Layout style={{ minHeight: '100vh' }}>
            <Sider width={250} onCollapse={() => setCollapsed(!collapsed)} collapsed={collapsed} style={{ background: colorBgContainer }}>
                <div style={{ height: 50, margin: 16 }}>
                    {
                        !collapsed ? <Image src={"/logo.png"} alt="DeTrain" preview={false} width={150} /> : <Image src={"/icon.png"} alt="DeTrain" preview={false} width={50} height={50} />
                    }
                </div>

                <Menu
                    style={{ fontWeight: 600 }}
                    inlineIndent={10}
                    mode="inline"
                    defaultSelectedKeys={['1']}
                    items={[
                        {
                            key: '2',
                            icon: <AiTwotoneExperiment />,
                            label: "My experiments",
                            onClick: () => router.push("/experiments/list")
                        },
                        {
                            key: '3',
                            icon: <AiOutlineExperiment />,
                            label: "New experiment",
                            onClick: () => router.push("/experiments/create")
                        },
                        { type: "divider" },
                        {
                            key: '4',
                            icon: <LuBrainCircuit />,
                            label: "My models",
                            onClick: () => router.push("/models/list")
                        },
                        {
                            key: '6',
                            icon: <GrDeploy />,
                            label: "My Deployments",
                            onClick: () => router.push("/deployments")
                        },
                        {
                            key: '5',
                            icon: <CiPlay1 />,
                            label: "Playground",
                            onClick: () => router.push("/playground")
                        },
                        { type: "divider" },
                        {
                            key: "7",
                            type: "group",
                            label: !collapsed ? 'GammaFlow v1.0.0' : "",
                            children: [
                                {
                                    key: '7.1',
                                    icon: <FaSuperscript />,
                                    label: 'Twitter',
                                    onClick: () => window.open("https://twitter.com/GammaFlowA2N", "_blank")
                                },
                                {
                                    key: '7.2',
                                    icon: <GithubOutlined />,
                                    label: 'Github',
                                    onClick: () => window.open("https://github.com/a2nfinance/gammaflow", "_blank")
                                },
                            ]
                        },

                    ]}
                />
            </Sider>
            <Layout>

                <Header //@ts-ignore
                    style={{ padding: 0, backgroundColor: colorBgContainer }}>
                    <Space align="center" style={{ display: "flex", justifyContent: "space-between" }}>
                        <Button
                            type="text"
                            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
                            onClick={() => setCollapsed(!collapsed)}
                            style={{
                                fontSize: '16px',
                                width: 64,
                                height: 64,
                            }}
                        />
                        <Form layout="inline">
                            <Form.Item>

                                <ConnectWallet />
                            </Form.Item>
                        </Form>
                    </Space>
                </Header>
                <Content
                    style={{
                        margin: '24px 16px 0 16px',
                        padding: 16,
                        boxSizing: "border-box",
                        // background: colorBgContainer,
                        maxWidth: 1440,
                        marginRight: "auto",
                        marginLeft: "auto"
                    }}
                >
                    {props.children}
                </Content>
                <Footer style={{ textAlign: 'center', maxHeight: 50 }}>GammaFlow ©2024 Created by A2N Finance</Footer>
            </Layout>

        </Layout>
    )

}
