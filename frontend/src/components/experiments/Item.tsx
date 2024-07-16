import { headStyle } from '@/theme/layout';
import { timeAgo } from '@/utils/timeUtils';
import { Button, Card, Descriptions, Space } from 'antd';
import { useRouter } from 'next/router';
import { FaLink } from "react-icons/fa";
export const Item = ({ index, experiment }: {index: number, experiment: any}) => {
  const router = useRouter();
  return (
    <Card key={`experiment-${index}`} title={experiment.name}  headStyle={headStyle} style={{ margin: 5 }} extra={
      <Space>
            <Button type='primary' onClick={() => router.push(`/experiments/runs/${experiment.experiment_id}`)}>Experiment runs</Button>
      </Space>
     
    }>
      <Descriptions column={2} layout="vertical">
                <Descriptions.Item label="State">
                    {experiment.lifecycle_stage}
                </Descriptions.Item>
                <Descriptions.Item label="Github repo">
                    <Button icon={<FaLink />} type='dashed' onClick={() => window.open(`${experiment.tags[2].value}`, "_blank")}></Button>
                </Descriptions.Item>
                <Descriptions.Item label="Created at">
                    {new Date(experiment.creation_time).toLocaleString()}
                </Descriptions.Item>
                <Descriptions.Item label="Last updated at">
                    {experiment.last_update_time ? <span title={`${new Date(parseInt(experiment.last_update_time)).toLocaleString()}`}>{timeAgo(experiment.last_update_time)}</span> : "N/A"}
                </Descriptions.Item>
              
                <Descriptions.Item label="Node address">
                <Button icon={<FaLink />} type='dashed' onClick={() => window.open(`${experiment.tags[1].value}`, "_blank")}></Button>
                </Descriptions.Item>
                <Descriptions.Item label="Training script path">
                    {experiment.tags[0].value}
                </Descriptions.Item>
            </Descriptions>
    </Card>
  );
}