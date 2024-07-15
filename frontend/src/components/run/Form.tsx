import { useAppSelector } from "@/controller/hooks";
import { useWebsocket } from "@/hooks/useWebsocket";
import { cloneGitCommand, installDependenciesCommand, pullGitCommand, runScriptCommand } from "@/utils/command-template";
import { Button, Col, Collapse, Divider, Form, Input, Radio, Row, Select } from "antd";

export const NewRunForm = () => {
    const { sendCommand } = useWebsocket();
    const { currentExperiment } = useAppSelector(state => state.experiment);
    const onFinish = (values: FormData) => {
      
        let installCommand = installDependenciesCommand(values);
        let shellCommand = installCommand; 
        if (values["clone_or_pull"] === "1") {
            let githubCommand =  cloneGitCommand(values["github_repo"], values["is_private_repo"] === "1", values["username"], values["password"]);
            let runTrainingScriptCommand = runScriptCommand(values["github_repo"], true, values["training_script_path"]);
            shellCommand += githubCommand + runTrainingScriptCommand;
        } else {
            let githubCommand = pullGitCommand(values["github_repo"], values["is_private_repo"] === "1", values["username"], values["password"]);
            let runTrainingScriptCommand = runScriptCommand(values["github_repo"], false, values["training_script_path"]);
            shellCommand += githubCommand + runTrainingScriptCommand;
        }
        if (shellCommand) {
            console.log(shellCommand);
            sendCommand("localhost", shellCommand, "runlog", 5000);
        }
       
    }
    return (

        <Row gutter={12}>
            <Col span={12}>
                <Form layout="vertical" onFinish={onFinish} initialValues={
                    {
                        "github_repo": currentExperiment?.tags.filter(tag => tag.key === "github_repo")[0]?.value,
                        "is_private_repo": currentExperiment?.tags.filter(tag => tag.key === "is_private_repo")[0]?.value,
                        "training_script_path": currentExperiment?.tags.filter(tag => tag.key === "training_script_path")[0]?.value,
                        "clone_or_pull": "0"
                    }
                }>
                    <Form.Item label="Name" name={"run_name"} rules={[{ required: true, message: 'Missing run name' }]}>
                        <Input type="text" size="large" />
                    </Form.Item>

                    <Collapse
                        items={[
                            {
                                key: "1",
                                label: "Github settings",
                                children: <>

                                    <Form.Item label="Repository" name={"github_repo"}>
                                        <Input type="text" size="large" />
                                    </Form.Item>
                                    <Form.Item label="Traning script path" name={"training_script_path"}>
                                        <Input type="text" size="large" />
                                    </Form.Item>
                                    <Row gutter={12}>

                                        <Col span={12}>
                                            <Form.Item label="Username" name={"github_username"}>
                                                <Input type="text" size="large" />
                                            </Form.Item>
                                        </Col>
                                        <Col span={12}>

                                            <Form.Item label="Password" name={"github_password"}>
                                                <Input type="password" size="large" />
                                            </Form.Item>
                                        </Col>
                                    </Row>
                                    <Row gutter={12}>
                                        <Col span={12}>
                                            <Form.Item name={"is_private_repo"} label="Is private repository">
                                                <Radio.Group>
                                                    <Radio value={"1"}>Yes</Radio>
                                                    <Radio value={"0"}>No</Radio>
                                                </Radio.Group>
                                            </Form.Item>
                                        </Col>
                                        <Col span={12}>
                                            <Form.Item name={"clone_or_pull"} label="Clone or Pull">
                                                <Radio.Group>
                                                    <Radio value={"1"}>Clone</Radio>
                                                    <Radio value={"0"}>Pull</Radio>
                                                </Radio.Group>
                                            </Form.Item>
                                        </Col>
                                    </Row>


                                </>
                            }
                        ]}
                        onChange={() => { }} />

                    <Divider />
                    <Form.Item help="All system dependencies will be installed before training scripts run" label="System dependencies" name={"system_dependencies"}>
                        <Input type="text" size="large" placeholder="E.g. ffmpeg,mlflow" />
                    </Form.Item>
                    <Form.Item label="Python dependencies" help="All python dependencies will be installed using pip before training scripts run" name={"system_dependencies"}>
                        <Input type="text" size="large" placeholder="E.g. scipy,numpy,sklearn" />
                    </Form.Item>
                    <Form.Item label="Install dependencies using requirements.txt" name={"use_requirements"}>
                        <Select size="large" options={[
                            {
                                label: "Yes",
                                value: "1",
                            },
                            {
                                label: "No",
                                value: "0",
                            },

                        ]} />
                    </Form.Item>
                    <Button block size="large" type="primary" htmlType="submit">Start new run on the remote node</Button>
                </Form>
            </Col>
            <Col span={12}>
                <label htmlFor={"runlog"}>Logs</label>
                <textarea readOnly placeholder="Remote server - training process logs" id={`runlog`} style={{ marginTop: 10, borderRadius: 10, height: "95%", width: "100%", backgroundColor: "#333", color: "whitesmoke", padding: "10px" }} />
            </Col>
        </Row>



    )
}