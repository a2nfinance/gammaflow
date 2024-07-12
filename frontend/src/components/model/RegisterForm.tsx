import { useAppSelector } from "@/controller/hooks";
import { useModels } from "@/hooks/useModels";
import { Button, Card, Form, Input, Radio, Select } from "antd";
import { useEffect, useState } from "react";

export const RegisterForm = () => {
    const { createModelVersionAction, createRegisteredModelAction } = useAppSelector(state => state.process);
    const { modelList } = useAppSelector(state => state.model);
    const { createRegisteredModel, searchRegisteredModels, } = useModels();
    const [newOrUpdate, setNewOrUpdate] = useState("1");
    const onFinish = (values: FormData) => {
        createRegisteredModel(values);
    }

    useEffect(() => {
        searchRegisteredModels();
    }, [])
    return (
        <Card>
            <Form layout='vertical' onFinish={onFinish} initialValues={{
                "new_or_update": "1"
            }}>
                <Form.Item name={"new_or_update"}>
                    <Radio.Group onChange={(e) => setNewOrUpdate(newOrUpdate === "1" ? "0" : "1")}>
                        <Radio value={"1"}>New nodel</Radio>
                        <Radio value={"0"}>Update registered model</Radio>
                    </Radio.Group>
                </Form.Item>
                {
                    newOrUpdate === "1" && <>
                        <Form.Item label={"Model name"} name={"name"} rules={[{ required: true, message: 'Missing model name' }]}>
                            <Input size={"large"} placeholder='E.g. llma_2' />
                        </Form.Item>
                        <Form.Item label={"Description"} name={"description"}>
                            <Input size={"large"} placeholder='E.g. chat completion' />
                        </Form.Item>
                    </>
                }

                {
                    newOrUpdate === "0" && <Form.Item label="Select a registered model" name={"registered_model_name"} rules={[{ required: true, message: 'Missing selected model' }]}>
                        <Select size={"large"} options={modelList.map(m => ({ label: m.name, value: m.name }))} />
                    </Form.Item>
                }

                <Button block htmlType='submit' type='primary' size='large' loading={createRegisteredModelAction}>Submit</Button>
            </Form>

        </Card>
    )
}