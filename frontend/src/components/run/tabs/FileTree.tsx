import React from 'react';
import { DownOutlined } from '@ant-design/icons';
import { Tree } from 'antd';
import type { TreeDataNode, TreeProps } from 'antd';
import { useAppSelector } from '@/controller/hooks';



export const FileTree: React.FC = () => {
    const {rootFolder} = useAppSelector(state => state.run);
    const treeData: TreeDataNode[] = [
        {
            title: `${rootFolder}`,
            key: '0-0',
            children: [
                {
                    title: 'metadata',
                    key: '0-0-0',
                    children: [
                        {
                            title: 'MLModel',
                            key: '0-0-0-0',
                        },
                        {
                            title: 'conda.yaml',
                            key: '0-0-0-1',
                        },
                        {
                            title: 'python_env.yaml',
                            key: '0-0-0-2',
                        },
                        {
                            title: 'requirements.txt',
                            key: '0-0-0-3',
                        },
                    ],
                },
                {
                    title: 'MLModel',
                    key: '0-0-1',
                },
                {
                    title: 'conda.yaml',
                    key: '0-0-2',
                },
                {
                    title: 'input_example.json',
                    key: '0-0-4',
                },
                {
                    title: 'model.pkl',
                    key: '0-0-5',
                },
                {
                    title: 'python_env.yaml',
                    key: '0-0-6',
                },
                {
                    title: 'requirements.txt',
                    key: '0-0-7',
                },
            ],
        },
    ];
    const onSelect: TreeProps['onSelect'] = (selectedKeys, info) => {
        console.log('selected', selectedKeys, info);
    };

    return (
        <Tree
            showLine
            switcherIcon={<DownOutlined />}
            defaultExpandedKeys={['0-0-0']}
            onSelect={onSelect}
            treeData={treeData}
        />
    );
};