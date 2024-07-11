import { Button, Col, Divider, Form, Input, Row, Select } from "antd"

export const NewRunForm = () => {
    const onFinish = (values: FormData) => {

    }
    return (

        <Row gutter={12}>
            <Col span={12}>
                <Form layout="vertical" onFinish={onFinish}>
                    <Form.Item label="Name" name={"run_name"} rules={[{ required: true, message: 'Missing run name' }]}>
                        <Input type="text" size="large" />
                    </Form.Item>
                    <Row gutter={12}>
                        <Col span={12}>
                            <Form.Item label="Github username" name={"github_username"}>
                                <Input type="text" size="large" />
                            </Form.Item>
                        </Col>
                        <Col span={12}>

                            <Form.Item label="Github password" name={"github_password"}>
                                <Input type="password" size="large" />
                            </Form.Item>
                        </Col>
                    </Row>
                    <Divider />
                    <Form.Item help="All system dependencies will be installed before training scripts run" label="System dependencies" name={"system_dependencies"}>
                        <Input type="text" size="large" placeholder="E.g. ffmpeg,mlflow" />
                    </Form.Item>
                    <Form.Item label="Python dependencies" help="All python dependencies will be installed using pip before training scripts run" name={"system_dependencies"}>
                        <Input type="text" size="large" placeholder="E.g. scipy,numpy,sklearn" />
                    </Form.Item>
                    <Form.Item label="Install dependencies using requirements.txt">
                        <Select size="large" options={[
                            {
                                label: "Yes",
                                value: 1,
                            },
                            {
                                label: "No",
                                value: 0,
                            },

                        ]} />
                    </Form.Item>
                    <Button block size="large" type="primary">Start new run on the remote node</Button>
                </Form>
            </Col>
            <Col span={12}>
                <label htmlFor={"runlog"}>Logs</label>
                <textarea placeholder="Remote server - training process logs" id={`runlog`} style={{ marginTop: 10,borderRadius: 10, height: "95%", width: "100%", backgroundColor: "#333", color: "whitesmoke", padding: "10px" }} />
            </Col>
        </Row>



    )
}