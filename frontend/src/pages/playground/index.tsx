import { useAppSelector } from "@/controller/hooks";
import { useAIServices } from "@/hooks/useAIServices";
import { useModels } from "@/hooks/useModels";
import { headStyle } from "@/theme/layout";
import { MinusCircleOutlined, PlusOutlined, UploadOutlined } from "@ant-design/icons";
import { useConnectWallet } from "@web3-onboard/react";
import { AutoComplete, Button, Card, Col, Collapse, Divider, Form, Input, Radio, Row, Upload } from "antd";
import { DefaultOptionType } from "antd/es/select";
import { useEffect, useState } from "react";

export default function Index() {
    const [{ wallet }] = useConnectWallet();
    const { callInferenceServicesAction } = useAppSelector(state => state.process)
    const { deployedVersions } = useAppSelector(state => state.model)
    const { callAIEnpoint } = useAIServices()
    const { getModelVersionsByAddress } = useModels();
    const [inputType, setInputType] = useState("1");
    const [form] = Form.useForm();
    const [options, setOptions] = useState<DefaultOptionType[]>([]);

    const onFinish = (values: FormData) => {
        const firstBodyStructure = values["sequence_services"][0].body;
        if (values["input_type"] === "2") {
            // Get the first file in the input
            const file = values["input"][0].originFileObj;


            if (file) {
                const reader = new FileReader();
                reader.onload = function (event) {
                    //@ts-ignore
                    let base64String: any = reader.result;
                    let inputJson = JSON.parse(firstBodyStructure.replace(":inputdata", base64String.substr(base64String.indexOf(',') + 1)));
                    callAIEnpoint(inputJson, values);
                };

                reader.readAsDataURL(file); // Read file as an ArrayBuffer
            }
        }

        if (values["input_type"] === "1") {
            const input = values["input"];
            let inputJson = JSON.parse(firstBodyStructure.replace(":inputdata", input));
            callAIEnpoint(inputJson, values);
        }

        if (values["input_type"] === "3") {
            const input = values["input"];
            let inputJson = JSON.parse(firstBodyStructure.replace(":inputdata", input));
            callAIEnpoint(inputJson, values);
        }
    }

    const getFile = (e) => {
        console.log('Upload event:', e);

        if (Array.isArray(e)) {
            return e;
        }
        return e && e.fileList;
    };

    useEffect(() => {
        if (wallet?.accounts?.length) {
            getModelVersionsByAddress()
        }

    }, [wallet?.accounts?.length])

    const handleSearch = (value: string) => {
        let models = deployedVersions.filter(m => m.name.includes(value)).map((m) => {
            let tagValue = m.tags.filter(t => t.key === "deployment_info")[0].value;
            let endpoint = JSON.parse(tagValue).inference_endpoint;
            return {
                label: `${m.name} v${m.version}`,
                value: `${endpoint}`,
            }
        });
        setOptions(models);
    };
    return (
        <div style={{ maxWidth: 1440, minWidth: 1024, margin: "auto" }}>
            <Card headStyle={headStyle} title={"Inference services playground"}>
                <Row gutter={12}>
                    <Col span={12}>
                        <Card>
                            <Form
                                form={form}
                                layout="vertical"
                                onFinish={onFinish}
                                initialValues={{
                                    "input_type": "1",
                                    "output_type": "1",
                                    "sequence_services": [
                                        {"endpoint": "", "body": ""}
                                    ]
                                }}>


                                <Form.Item label="Input type" name={"input_type"} >
                                    <Radio.Group onChange={(e) => setInputType(e.target.value)}>
                                        <Radio value={"1"}>Text</Radio>
                                        <Radio value={"2"}>File</Radio>
                                        <Radio value={"3"}>JSON</Radio>
                                    </Radio.Group>
                                </Form.Item>
                                <Form.Item label="Output type" name={"output_type"}>
                                    <Radio.Group>
                                        <Radio value={"1"}>Text</Radio>
                                        <Radio value={"2"}>Audio</Radio>
                                        <Radio value={"3"}>Video</Radio>
                                    </Radio.Group>
                                </Form.Item>
                                <Divider />
                                {
                                    (inputType === "1" || inputType === "3") ?
                                        <Form.Item label="Input data" name={"input"}>
                                            <Input size="large" />
                                        </Form.Item>
                                        :
                                        <Form.Item label="Input data" name={"input"} getValueFromEvent={getFile}>
                                            <Upload>
                                                <Button icon={<UploadOutlined />}>Click to Upload</Button>
                                            </Upload>
                                        </Form.Item>
                                }
                                <Divider />
                                <Collapse items={[
                                    {
                                        key: "1",
                                        label: "Sequence inference services",
                                        children:
                                            <Form.List name="sequence_services">
                                                {(fields, { add, remove }) => (
                                                    <>
                                                        {fields.map(({ key, name, ...restField }, index) => (
                                                            <Card key={`card-${key}`} title={`Service (${index})`} style={{ marginBottom: 8 }}>
                                                                <Row key={key} style={{ display: 'flex' }} gutter={12}>

                                                                    <Col span={22}>
                                                                        <Form.Item
                                                                            {...restField}
                                                                            name={[name, 'endpoint']}
                                                                            rules={[{ required: true, message: 'Missing endpoint' }]}>
                                                                            <AutoComplete
                                                                                size="large"
                                                                                onSearch={handleSearch}
                                                                                placeholder="Enter model name to search endpoint"
                                                                                options={options}
                                                                            />
                                                                        </Form.Item>

                                                                    </Col>
                                                                    <Col span={2}>
                                                                        {fields.length > 1 ? (
                                                                            <MinusCircleOutlined
                                                                                className="dynamic-delete-button"
                                                                                onClick={() => remove(name)}
                                                                            />
                                                                        ) : null}
                                                                    </Col>

                                                                </Row>
                                                                <Form.Item
                                                                    {...restField}
                                                                    name={[name, 'body']}
                                                                    rules={[{ required: true, message: 'Missing body structure data' }]}
                                                                >
                                                                    <Input type={"text"} size='large' placeholder='E.g. {"inputs": [{"audio": [":inputdata"]}]}' />
                                                                </Form.Item>
                                                            </Card>
                                                        ))}

                                                        <Form.Item>
                                                            <Button type="dashed" onClick={() => add()} block icon={<PlusOutlined />}>
                                                                Add inference service
                                                            </Button>
                                                        </Form.Item>
                                                    </>
                                                )}
                                            </Form.List>
                                    }
                                ]} />

                                <Divider />
                                <Button htmlType="submit" loading={callInferenceServicesAction} type="primary" block size="large">Submit</Button>
                            </Form>
                        </Card>
                    </Col>
                    <Col span={12} >
                        <Card style={{ height: "100%" }}>
                            <label>Output</label>
                            <div id={`playground-output`} style={{ marginTop: 10, borderRadius: 10, height: "100%", width: "100%", backgroundColor: "#333", color: "whitesmoke", padding: "10px" }} />
                        </Card>


                    </Col>
                </Row>
            </Card>
        </div>
    )
}