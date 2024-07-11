import { CREATE_MODEL_VERSION_ENDPOINT, CREATE_REGISTERED_MODEL_ENDPOINT, SEARCH_MODEL_ENDPOINT } from "@/configs";
import { createRegisteredModelMessage } from "@/configs/messages";
import { useAppDispatch, useAppSelector } from "@/controller/hooks";
import { setModelList } from "@/controller/model/modelSlice";
import { actionNames, updateActionStatus } from "@/controller/process/processSlice";
import { MESSAGE_TYPE, openNotification } from "@/utils/noti";
import { useConnectWallet } from "@web3-onboard/react";
export const useModels = () => {
    const dispatch = useAppDispatch();
    const [{ wallet }] = useConnectWallet();
    const { run } = useAppSelector(state => state.run)
    const searchRegisteredModels = async () => {
        try {
            if (!wallet?.accounts?.length) return;
            let req = await fetch(`${SEARCH_MODEL_ENDPOINT}`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    max_results: 1000,
                    filter: `tags.wallet_address = '${wallet.accounts[0].address}'`,
                })
            })

            let res = await req.json();
            dispatch(setModelList(res));
        } catch (e) {
            console.log(e);
        }

    }

    const createRegisteredModel = async (values: FormData) => {
        try {
            if (!wallet?.accounts?.length) return;
            dispatch(updateActionStatus({ actionName: actionNames.createRegisteredModelAction, value: true }))
            let req = await fetch(`${CREATE_REGISTERED_MODEL_ENDPOINT}`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    name: `${values["name"]}`,
                    // run_link: ``,
                    description: `${values["description"]}`,
                    tags: [
                        { key: "wallet_address", value: `${wallet.accounts[0].address}` },
                    ]
                })
            })

            let res = await req.json();
            updateModelVersion(res.registered_model.name);
            let messages = createRegisteredModelMessage(res.registered_model.name);
            openNotification(messages.title, messages.success, MESSAGE_TYPE.SUCCESS);
        } catch (e) {
            console.log(e);
            let messages = createRegisteredModelMessage();
            openNotification(messages.title, messages.fail, MESSAGE_TYPE.ERROR);
        }
        dispatch(updateActionStatus({ actionName: actionNames.createRegisteredModelAction, value: false }))
    }

    const updateModelVersion = async (registeredModelName: string) => {
        // run_id: `${run.info.run_id}`,
        //             source: `${run.info.artifact_uri}`,
        let req = await fetch(`${CREATE_MODEL_VERSION_ENDPOINT}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                name: registeredModelName,
                run_id: `${run.info.run_id}`,
                source: `${run.info.artifact_uri}`,
                // run_link: ``,
                // description: `${values["description"]}`,
                // tags: [
                //     { key: "wallet_address", value: `${wallet.accounts[0].address}` },
                // ]
            })
        })
        await req.json();
    }

    // Update registered model version here

    return { searchRegisteredModels, createRegisteredModel }
}