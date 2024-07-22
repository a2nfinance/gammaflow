import { useAppSelector } from "@/controller/hooks";
import { useExperiments } from "@/hooks/useExperiments";
import { headStyle } from "@/theme/layout";
import { Button, Card, Col, Form, Input, Radio, Row } from "antd";
import { AiOutlineExperiment, AiTwotoneExperiment } from "react-icons/ai";
import { PiComputerTower } from "react-icons/pi";
import { IoIosCode } from "react-icons/io";
import { FaGithub } from "react-icons/fa";
export const Settings = () => {
    const {createExperimentAction} = useAppSelector(state => state.process);
    const { createExperiment } = useExperiments();

    const handleSubmitForm = (values: FormData) => {
        //validate here
        console.log(values);
        createExperiment(values);
    }
    return (

        <Form layout="vertical" onFinish={handleSubmitForm} initialValues={{
            'name': "Experiment 01",
            "is_private_repo": 0
        }} >
            <Card title="Create experiment" headStyle={headStyle}>
                <Row gutter={12}>
                    <Col span={12}>
                        <Form.Item name="name" label="Experiment name" rules={[{ required: true, message: 'Missing experiment name'}]} >
                            <Input addonBefore={<AiOutlineExperiment />} size='large' placeholder="Experiment name" />
                        </Form.Item>

                        <Form.Item name={"github_repo"} label="Github repository" rules={[{ required: true, message: 'Incorrect contact github repo', type: "url" }]}>
                            <Input type="text" addonBefore={<FaGithub />} placeholder="Github repository" size="large" />
                        </Form.Item>
                        <Form.Item name={"is_private_repo"} label="Is private repository" rules={[{ required: true, message: 'Missing github repository type'}]}>
                            <Radio.Group>
                                <Radio value={1}>Yes</Radio>
                                <Radio value={0}>No</Radio>
                            </Radio.Group>
                        </Form.Item>

                    </Col>
                    <Col span={12}>
                        <Form.Item name={"node_address"} label="Theta node address" rules={[{ required: true, message: 'Incorrect node address', type: "url" }]}>
                            <Input type="text" addonBefore={<PiComputerTower />} placeholder="IP/Domain address with protocol (https, http)" size="large" />
                        </Form.Item>

                        <Form.Item name={"training_script_path"} label="Training script path">
                            <Input type="text" addonBefore={<IoIosCode />} size='large' placeholder="E.g. scripts/main.py" />
                        </Form.Item>

                    </Col>
                </Row>


                <Button type="primary" loading={createExperimentAction} size="large" block htmlType="submit">Submit</Button>
            </Card>
        </Form>



    )
}