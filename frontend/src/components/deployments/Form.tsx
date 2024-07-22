import { useAppSelector } from "@/controller/hooks";
import { useModels } from "@/hooks/useModels";
import { headStyle } from "@/theme/layout";
import { useConnectWallet } from "@web3-onboard/react";
import { AutoComplete, Button, Card, Col, Form, Input, Row } from "antd";
import { DefaultOptionType } from "antd/es/select";
import { useEffect, useState } from "react";
import { LuBrainCircuit } from "react-icons/lu";
import { GoVersions } from "react-icons/go";
import { FaDocker } from "react-icons/fa";
import { TbApi } from "react-icons/tb";
export const DeploymentForm = () => {
    const [{ wallet }] = useConnectWallet();
    const { modelList } = useAppSelector(state => state.model);
    const {createDeploymentInfoAction} = useAppSelector(state => state.process);
    const { searchRegisteredModels, setModelVersionDeploymentInfo } = useModels();
    const [options, setOptions] = useState<DefaultOptionType[]>([]);
    
    useEffect(() => {
        if (wallet?.accounts?.length) {
            searchRegisteredModels();
        }
    }, [wallet?.accounts?.length]);

    const onFinish = (values: FormData) => {
        setModelVersionDeploymentInfo(values);
    };

    const handleSearch = (value: string) => {
        console.log(modelList);
        let models = modelList.filter(m => m.name.includes(value)).map((m) => ({
            label: `${m.name} | Lastest version: ${m.latest_versions[0].version}`,
            value: `${m.name}`,
        }));
        setOptions(models);
    };
    return (
        <Card headStyle={headStyle} title={"Update deployment information"}>
            <Form layout="vertical" onFinish={onFinish}>

                <Row gutter={12}>
                    <Col span={12}>
                        <Form.Item name={"model_name"} label={"Select AI model"} rules={[{ required: true, message: 'Missing selected AI model'}]}>
                            <AutoComplete
                                suffixIcon={<LuBrainCircuit />}
                                size="large"
                                onSearch={handleSearch}
                                placeholder="Enter model name"
                                options={options}
                            />
                        </Form.Item>
                    </Col>
                    <Col span={12}>
                        <Form.Item name={"version"} label={"Version"} rules={[{ required: true, message: 'Missing version'}]}>
                            <Input addonBefore={<GoVersions />} size="large" />
                        </Form.Item>
                    </Col>
                </Row>
                <Row gutter={12}>
                    <Col span={12}>
                        <Form.Item name={"docker_image"} label={"Docker image"} rules={[{ required: true, message: 'Missing docker image' }]}>
                            <Input addonBefore={<FaDocker/>} size="large" />
                        </Form.Item>
                    </Col>
                    <Col span={12}>
                        <Form.Item name={"inference_endpoint"} label={"Inference endpoint"} rules={[{ required: true, message: 'Incorrect inference endpoint', type: "url" }]}>
                            <Input addonBefore={<TbApi/>} size="large" />
                        </Form.Item>
                    </Col>
                </Row>
                <Button type="primary" htmlType="submit" loading={createDeploymentInfoAction} block>Submit</Button>
            </Form>
        </Card>
    )

}