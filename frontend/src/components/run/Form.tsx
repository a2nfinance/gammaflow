import { useAppSelector } from "@/controller/hooks";
import { useWebsocket } from "@/hooks/useWebsocket";
import { cloneGitCommand, installDependenciesCommand, pullGitCommand, runScriptCommand } from "@/utils/command-template";
import { Button, Card, Col, Collapse, Divider, Form, Input, Radio, Row, Select } from "antd";
import { FaGithub } from "react-icons/fa";
import { IoIosCode } from "react-icons/io";
import { PiComputerTower } from "react-icons/pi";
import { MdOutlineLibraryAdd } from "react-icons/md";

export const NewRunForm = () => {
    const { startTrainingAction } = useAppSelector(state => state.process);
    const { sendCommand } = useWebsocket();
    const { currentExperiment } = useAppSelector(state => state.experiment);
    const onFinish = (values: FormData) => {
        let shellCommand = "";
        let installCommand = installDependenciesCommand(values);

        if (values["clone_or_pull"] === "1") {
            let githubCommand = cloneGitCommand(values["github_repo"], values["is_private_repo"] === "1", values["username"], values["password"]);
            let runTrainingScriptCommand = runScriptCommand(values["github_repo"], true, values["training_script_path"]);
            shellCommand += githubCommand + installCommand + runTrainingScriptCommand;
        } else {
            let githubCommand = pullGitCommand(values["github_repo"], values["is_private_repo"] === "1", values["username"], values["password"]);

            let runTrainingScriptCommand = runScriptCommand(values["github_repo"], false, values["training_script_path"]);
            shellCommand += githubCommand + installCommand + runTrainingScriptCommand;
        }
        if (shellCommand) {
            sendCommand(values["remote_address"], shellCommand, "runlog", values["port"]);
        }

    }
    return (
        <Card>
            <Row gutter={12}>
                <Col span={12}>
                    <Form layout="vertical" onFinish={onFinish} initialValues={
                        {
                            "github_repo": currentExperiment?.tags.filter(tag => tag.key === "github_repo")[0]?.value,
                            "is_private_repo": currentExperiment?.tags.filter(tag => tag.key === "is_private_repo")[0]?.value,
                            "training_script_path": currentExperiment?.tags.filter(tag => tag.key === "training_script_path")[0]?.value,
                            "clone_or_pull": "0",
                            "remote_address": currentExperiment?.tags.filter(t => t.key === "node_address")[0]?.value,
                            "port": 80,
                            "use_requirements": "0"
                        }
                    }>
                        <Row gutter={8}>
                            <Col span={18}>
                                <Form.Item label="Theta node address" name={"remote_address"}>
                                    <Input addonBefore={<PiComputerTower />} type="text" size="large" />
                                </Form.Item>
                            </Col>
                            <Col span={6}>
                                <Form.Item label="Port" name={"port"}>
                                    <Input type="number" size="large" />
                                </Form.Item>
                            </Col>

                        </Row>

                        <Collapse
                            items={[
                                {
                                    key: "1",
                                    label: "Github settings",
                                    children: <>

                                        <Form.Item label="Repository" name={"github_repo"}>
                                            <Input addonBefore={<FaGithub />} type="text" size="large" />
                                        </Form.Item>
                                        <Form.Item label="Traning script path" name={"training_script_path"}>
                                            <Input addonBefore={<IoIosCode />} type="text" size="large" />
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
                            <Input addonBefore={<MdOutlineLibraryAdd />} type="text" size="large" placeholder="E.g. ffmpeg,mlflow" />
                        </Form.Item>
                        <Form.Item label="Python dependencies" help="All python dependencies will be installed using pip before training scripts run" name={"system_dependencies"}>
                            <Input addonBefore={<MdOutlineLibraryAdd />} type="text" size="large" placeholder="E.g. scipy,numpy,sklearn" />
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
                        <Button block size="large" type="primary" loading={startTrainingAction} htmlType="submit">Start a run on the remote node</Button>
                    </Form>
                </Col>
                <Col span={12}>
                    <label htmlFor={"runlog"}>Logs</label>
                    <textarea readOnly placeholder="Remote server - training process logs" id={`runlog`} style={{ marginTop: 10, borderRadius: 10, height: "95%", width: "100%", backgroundColor: "#333", color: "whitesmoke", padding: "10px" }} />
                </Col>
            </Row>

        </Card>

    )
}