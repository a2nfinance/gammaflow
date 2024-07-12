import { CREATE_EXPERIMENT_ENDPOINT, GET_EXPERIMENT_ENDPOINT, SEARCH_EXPERIMENT_ENDPOINT, SEARCH_RUNS } from "@/configs";
import { createExperimentMessage } from "@/configs/messages";
import { setCurrentExperiment, setList, setRuns } from "@/controller/experiment/experimentSlice";
import { useAppDispatch } from "@/controller/hooks";
import { actionNames, updateActionStatus } from "@/controller/process/processSlice";
import { MESSAGE_TYPE, openNotification } from "@/utils/noti";
import { useConnectWallet } from "@web3-onboard/react";

export const useExperiments = () => {
    const dispatch = useAppDispatch();
    const [{ wallet }] = useConnectWallet();
    const createExperiment = async (values: FormData) => {
        if (!wallet?.accounts?.length) return;
        try {
            dispatch(updateActionStatus({ actionName: actionNames.createExperimentAction, value: true }))

            let createReq = await fetch(`${CREATE_EXPERIMENT_ENDPOINT}`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    name: `${values["name"]}`,
                    tags: [
                        { key: "wallet_address", value: `${wallet.accounts[0].address}` },
                        { key: "node_address", value: `${values["node_address"]}` },
                        { key: "training_script_path", value: `${values["training_script_path"]}` },
                        { key: "github_repo", value: `${values["github_repo"]}` },
                        { key: "is_private_repo", value: `${values["is_private_repo"]}` },
                    ]
                })
            })
            let res = await createReq.json();
            let messages = createExperimentMessage(res.experiment_id);
            openNotification(messages.title, messages.success, MESSAGE_TYPE.SUCCESS);
        } catch (e) {
            console.log(e);
            let messages = createExperimentMessage();
            openNotification(messages.title, messages.fail, MESSAGE_TYPE.ERROR);
        }

        dispatch(updateActionStatus({ actionName: actionNames.createExperimentAction, value: false }))

    };

    const getExperimentsByCreator = async () => {
        try {
            if (!wallet?.accounts?.length) return;
            let createReq = await fetch(`${SEARCH_EXPERIMENT_ENDPOINT}`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    max_results: 1000,
                    filter: `tags.wallet_address = '${wallet.accounts[0].address}'`,
                })
            })
            let res = await createReq.json();
            dispatch(setList(res.experiments));
        } catch (e) {
            console.log(e);
        }


    }
    const getExperimentById = async (id: string) => {
        try {

            let req = await fetch(`${GET_EXPERIMENT_ENDPOINT}?experiment_id=${id}`, {
                method: "GET"
            })
            let res = await req.json();
            console.log("Current experiment:", res.experiment);
            dispatch(setCurrentExperiment(res.experiment))
        } catch (e) {
            console.log(e);
        }
    };

    const searchRunByExperimentId = async (id: string) => {
        try {
            let req = await fetch(`${SEARCH_RUNS}`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    experiment_ids: [id],
                })
            })
            let res = await req.json();
            console.log(res)
            dispatch(setRuns(res.runs));
        } catch (e) {
            console.log(e);
        }

    }
    return { getExperimentsByCreator, createExperiment, getExperimentById, searchRunByExperimentId };
};