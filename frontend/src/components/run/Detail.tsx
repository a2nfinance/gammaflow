import { Card, Tabs } from 'antd';
import type { TabsProps } from 'antd';
import { Overview } from './tabs/Overview';
import { ModelMetrics } from './tabs/ModelMetrics';
import { SystemMetrics } from './tabs/SystemMetrics';
import { Artifacts } from './tabs/Artifacts';
import { useAppSelector } from '@/controller/hooks';
export const Detail = () => {
    const { run } = useAppSelector(state => state.run);
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
    return (
        <Card title={run?.info?.run_name}>
            <Tabs defaultActiveKey="1" items={items} onChange={() => { }} />
        </Card>
    )
      
}