import { useExperiments } from "@/hooks/useExperiments";
import { headStyle } from "@/theme/layout";
import { Button, Card, Divider, Form, Input } from "antd";
import { MdPassword } from "react-icons/md";
export const Settings = () => {
    const {createExperiment} = useExperiments();

    const handleSubmitForm = (values: FormData) => {
        console.log(values);
        createExperiment();
    }
    return (

        <Form layout="vertical" onFinish={handleSubmitForm} >
            <Card title="Create experiment"  headStyle={headStyle}>
                <Form.Item name={"node_address"} label="Node address">
                    <Input type="text" placeholder="Node Address" size="large" />
                </Form.Item>

                <Divider />
                <Form.Item name={"github_repo"} label="Github repository" rules={[{ message: 'Incorrect contact github repo' }]}>
                    <Input type="text" placeholder="Github repository" size="large" />
                </Form.Item>
                <Form.Item name={"user_name"} label="Username">
                    <Input type="text" placeholder="Username" size="large" />
                </Form.Item>
                <Form.Item name="password" label="Password">
                    <Input type="password" addonBefore={<MdPassword />} size='large' />
                </Form.Item>
                <Divider />
                <Form.Item name={"Training script file"} label="Training sript">
                    <Input type="text" size='large' />
                </Form.Item>
                <Form.Item name={"Saved models folder"} label="Model folder">
                    <Input type="text" size='large' />
                </Form.Item>
                <Form.Item name="Logs folder" label="Logs">
                    <Input type="text" size='large' />
                </Form.Item>
                <Button type="primary" size="large" block htmlType="submit">Submit</Button>
            </Card>
        </Form>



    )
}