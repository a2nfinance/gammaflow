import { useAppSelector } from '@/controller/hooks';
import type { TabsProps } from 'antd';
import { Button, Card, Modal, Tabs } from 'antd';
import { useState } from 'react';
import { RegisterForm } from '../model/RegisterForm';
import { Artifacts } from './tabs/Artifacts';
import { ModelMetrics } from './tabs/ModelMetrics';
import { Overview } from './tabs/Overview';
import { SystemMetrics } from './tabs/SystemMetrics';
export const Detail = () => {
    const { run } = useAppSelector(state => state.run);
    const [isModalOpen, setIsModalOpen] = useState(false);

    const items: TabsProps['items'] = [
        {
            key: '1',
            label: 'Overview',
            children: <Overview run={run} />,
        },
        // {
        //     key: '2',
        //     label: 'Model metrics',
        //     children: <ModelMetrics run={run} />,
        // },
        {
            key: '3',
            label: 'System metrics',
            children: <SystemMetrics run={run} />,
        },
        {
            key: '4',
            label: 'Artifacts',
            children: <Artifacts  />,
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
   
    return (
        <Card title={run?.info?.run_name} extra={[
            <Button key={`card-regis-model`} size='large' type='primary' onClick={() => showModal()}>Register model</Button>
        ]}>
            <Tabs defaultActiveKey="1" items={items} onChange={() => { }} />
            <Modal title="Register model" open={isModalOpen} footer={[]} onOk={handleOk} onCancel={handleCancel}>
                <RegisterForm />
            </Modal>
        </Card>
    )

}