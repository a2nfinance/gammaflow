import { Button, Card, Form, Input, Modal, Tabs } from 'antd';
import type { TabsProps } from 'antd';
import { Overview } from './tabs/Overview';
import { ModelMetrics } from './tabs/ModelMetrics';
import { SystemMetrics } from './tabs/SystemMetrics';
import { Artifacts } from './tabs/Artifacts';
import { useAppSelector } from '@/controller/hooks';
import { useState } from 'react';
import { useModels } from '@/hooks/useModels';
export const Detail = () => {
    const { run } = useAppSelector(state => state.run);
    const {createModelVersionAction, createRegisteredModelAction} = useAppSelector(state => state.process);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const {createRegisteredModel} = useModels();
    const items: TabsProps['items'] = [
        {
            key: '1',
            label: 'Overview',
            children: <Overview />,
        },
        {
            key: '2',
            label: 'Model metrics',
            children: <ModelMetrics />,
        },
        {
            key: '3',
            label: 'System metrics',
            children: <SystemMetrics />,
        },
        {
            key: '4',
            label: 'Artifacts',
            children: <Artifacts />,
        },
    ];

    const showModal = () => {
        setIsModalOpen(true);
    };

    const handleOk = () => {
        setIsModalOpen(false);
    };

    const handleCancel = () => {
        setIsModalOpen(false);
    };
    const onFinish = (values: FormData) => {
        createRegisteredModel(values);
    }
    return (
        <Card title={run?.info?.run_name} extra={[
            <Button size='large' type='primary' onClick={() => showModal()}>Register model</Button>
        ]}>
            <Tabs defaultActiveKey="1" items={items} onChange={() => { }} />
            <Modal title="Register model" open={isModalOpen} footer={[]} onOk={handleOk} onCancel={handleCancel}>
                <Form layout='vertical' onFinish={onFinish}>
                    <Form.Item label={"Model name"} name={"name"} rules={[{ required: true, message: 'Missing model name' }]}>
                        <Input size={"large"} placeholder='E.g. llma_2' />
                    </Form.Item>
                    <Form.Item label={"Description"} name={"description"}>
                        <Input size={"large"} placeholder='E.g. chat completion' />
                    </Form.Item>
                    <Button block htmlType='submit' type='primary' size='large' loading={createRegisteredModelAction}>Submit</Button>
                </Form>
            </Modal>
        </Card>
    )

}