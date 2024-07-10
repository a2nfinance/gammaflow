import { headStyle } from '@/theme/layout';
import { Button, Card, Descriptions, Space } from 'antd';
import { useRouter } from 'next/router';

export const Item = ({ index, experiment }: {index: number, experiment: any}) => {
  const router = useRouter();
  return (
    <Card key={`experiment-${index}`} title={experiment.name}  headStyle={headStyle} style={{ margin: 5 }} extra={
      <Space>
            <Button type='primary' onClick={() => router.push(`/experiments/runs/${experiment.experiment_id}`)}>All runs</Button>
      </Space>
     
    }>
      <Descriptions column={2} layout="vertical">
                <Descriptions.Item label="Stage">
                    {experiment.lifecycle_stage}
                </Descriptions.Item>
                <Descriptions.Item label="Github repo">
                    <Button type='link' onClick={() => window.open(`${experiment.tags[0].value}`, "_blank")}>link</Button>
                </Descriptions.Item>
                <Descriptions.Item label="Created at">
                    {new Date(experiment.creation_time).toLocaleString()}
                </Descriptions.Item>
                <Descriptions.Item label="Last updated at">
                    {new Date(experiment.last_update_time).toLocaleString()}
                </Descriptions.Item>
              
                <Descriptions.Item label="Node address">
                <Button type='link' onClick={() => window.open(`${experiment.tags[2].value}`, "_blank")}>link</Button>
                </Descriptions.Item>
                <Descriptions.Item label="Training script path">
                    {experiment.tags[3].value}
                </Descriptions.Item>
            </Descriptions>
    </Card>
  );
}