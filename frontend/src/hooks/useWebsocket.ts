import { useAppDispatch } from "@/controller/hooks";
import { actionNames, updateActionStatus } from "@/controller/process/processSlice";
import { w3cwebsocket as W3CWebSocket } from "websocket";


export const useWebsocket = () => {
    const dispatch = useAppDispatch();

    const sendCommand = async (remoteHostIP: string, command: string, outputElementId: string, openPort: number) => {
        try {
            let client = new W3CWebSocket(`${remoteHostIP}:${openPort}`);
            client.onopen = () => {
                dispatch(updateActionStatus({ actionName: actionNames.startTrainingAction, value: true }));
                client.send(command);
                console.log('WebSocket Client Connected');
            };

            let element = document.getElementById(outputElementId);
            client.onmessage = (message) => {
                try {
                    new Blob([message.data]).text().then(value => {
                        element?.append(value);
                        if (value.includes("end-process")) {
                            dispatch(updateActionStatus({ actionName: actionNames.startTrainingAction, value: false }));
                        }

                    });
                } catch (e: any) {
                    console.log(e.message)
                    dispatch(updateActionStatus({ actionName: actionNames.startTrainingAction, value: false }));
                }
            };

            return () => {
                console.log("Close");
                client.close();
            };
        } catch (e) {
            console.log(e);
        }


    }

    const sendCommandToTrackingServer = async (command: string, outputElementId: string) => {
        try {
            const client = new W3CWebSocket(`${process.env.NEXT_PUBLIC_SERVER_COMMANDER}`);

            client.onopen = () => {
                dispatch(updateActionStatus({ actionName: actionNames.generateDockerFilesAction, value: true }));
                client.send(command);
                console.log('WebSocket Client Connected');
            };

            let element = document.getElementById(outputElementId);
            client.onmessage = (message) => {
                try {
                    new Blob([message.data]).text().then(value => {
                        element?.append(value);
                        if (value.includes("end-process")) {
                            dispatch(updateActionStatus({ actionName: actionNames.generateDockerFilesAction, value: false }));
                        }

                    });

                } catch (e: any) {
                    console.log(e.message);
                    dispatch(updateActionStatus({ actionName: actionNames.generateDockerFilesAction, value: false }));
                }
            };

            return () => {
                client.close();
            };
        } catch (e) {
            console.log(e);
        }
    }

    const buildAndPushDockerImageOnServer = async (command: string) => {
        try {
            const client = new W3CWebSocket(`${process.env.NEXT_PUBLIC_SERVER_COMMANDER}`);

            client.onopen = () => {
                dispatch(updateActionStatus({ actionName: actionNames.buildAndPushDockerFileActions, value: true }));
                client.send(command);
                console.log('WebSocket Client Connected');
            };

            client.onmessage = (message) => {
                try {
                    new Blob([message.data]).text().then(value => {
                        if (value.includes("end-process")) {
                            dispatch(updateActionStatus({ actionName: actionNames.buildAndPushDockerFileActions, value: false }));
                        }

                    });

                } catch (e: any) {
                    console.log(e.message);
                    dispatch(updateActionStatus({ actionName: actionNames.buildAndPushDockerFileActions, value: false }));
                }
            };

            return () => {
                client.close();
            };
        } catch (e) {
            console.log(e);
        }
    }


    return { sendCommand, sendCommandToTrackingServer, buildAndPushDockerImageOnServer };
}