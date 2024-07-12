import { useAppDispatch, useAppSelector } from "@/controller/hooks";
import { actionNames, updateActionStatus } from "@/controller/process/processSlice";
import { w3cwebsocket as W3CWebSocket } from "websocket";


export const useWebsocket = () => {
    const dispatch = useAppDispatch();

    const sendCommand = async (remoteHostIP: string, command: string, outputElementId: string,  openPort: number) => {
        try {
            const client = new W3CWebSocket(`ws://${remoteHostIP}:${openPort}`);
           
            client.onopen = () => {
                client.send(command);
                console.log('WebSocket Client Connected');
            };

            let element = document.getElementById(outputElementId);
            client.onmessage = (message) => {
                try {
                    // check if value === "--end-process--" then stop loading status.
                    // Dis patch here
                    new Blob([message.data]).text().then( value => element?.append(value));
                } catch (e: any) {
                    console.log(e.message)
                }
            };
           
            return () => {
                client.close();
            };
        } catch (e) {
            console.log(e);
        }


    }


    return { sendCommand };
}