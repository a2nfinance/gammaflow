import React from 'react';
import { DownOutlined } from '@ant-design/icons';
import { Tree } from 'antd';
import type { TreeDataNode, TreeProps } from 'antd';
import { useAppDispatch, useAppSelector } from '@/controller/hooks';
import { setFileContent } from '@/controller/run/runSlice';
import { actionNames, updateActionStatus } from '@/controller/process/processSlice';



export const FileTree: React.FC = () => {
    const dispatch = useAppDispatch();
    const { tree, run } = useAppSelector(state => state.run);
    const treeData: TreeDataNode[] = [
        tree
    ];
    const onSelect: TreeProps['onSelect'] = (selectedKeys, info) => {
        // @ts-ignore
        if (!info.node.isDir) {
            dispatch(updateActionStatus({actionName: actionNames.getFileContentAction, value: true}))
            // @ts-ignore
            fetch(`${process.env.NEXT_PUBLIC_MLFLOW_TRACKING_SERVER}/get-artifact?path=${info.node.path}&run_uuid=${run.info.run_uuid}`)
                .then(r => r.text())
                .then(t => {
                    dispatch(setFileContent(t))
                    dispatch(updateActionStatus({actionName: actionNames.getFileContentAction, value: false}))
                })
        }

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