import { useAppSelector } from "@/controller/hooks";
import { useExperiments } from "@/hooks/useExperiments";
import { headStyle } from "@/theme/layout";
import { Button, Card, Col, Form, Input, Radio, Row } from "antd";
export const Settings = () => {
    const {createExperimentAction} = useAppSelector(state => state.process);
    const { createExperiment } = useExperiments();

    const handleSubmitForm = (values: FormData) => {
        //validate here
        console.log(values);
        createExperiment(values);
    }
    return (

        <Form layout="vertical" onFinish={handleSubmitForm} >
            <Card title="Create experiment" headStyle={headStyle}>
                <Row gutter={12}>
                    <Col span={12}>
                        <Form.Item name="name" label="Experiment name">
                            <Input size='large' placeholder="Experiment name" />
                        </Form.Item>

                        <Form.Item name={"github_repo"} label="Github repository" rules={[{ message: 'Incorrect contact github repo' }]}>
                            <Input type="text" placeholder="Github repository address" size="large" />
                        </Form.Item>
                        <Form.Item name={"is_private_repo"} label="Is private repository">
                            <Radio.Group>
                                <Radio value={1}>Yes</Radio>
                                <Radio value={0}>No</Radio>
                            </Radio.Group>
                        </Form.Item>

                    </Col>
                    <Col span={12}>
                        <Form.Item name={"node_address"} label="Theta node address">
                            <Input type="text" placeholder="IP/Domain address" size="large" />
                        </Form.Item>

                        <Form.Item name={"training_script_path"} label="Training script path">
                            <Input type="text" size='large' placeholder="E.g. scripts/main.py" />
                        </Form.Item>

                    </Col>
                </Row>


                <Button type="primary" loading={createExperimentAction} size="large" block htmlType="submit">Submit</Button>
            </Card>
        </Form>



    )
}