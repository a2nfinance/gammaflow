import { CREATE_MODEL_VERSION_ENDPOINT, CREATE_REGISTERED_MODEL_ENDPOINT, DOWNLOADER_ENDPOINT, GET_DOWNLOAD_URI_FOR_MODEL_VERSION_ARTIFACTS_ENDPOINT, GET_REGISTERED_MODEL_ENDPOINT, SEARCH_MODEL_ENDPOINT, SEARCH_MODEL_VERSIONS_ENDPOINT } from "@/configs";
import { createRegisteredModelMessage, updateRegisteredModelMessage } from "@/configs/messages";
import { useAppDispatch, useAppSelector } from "@/controller/hooks";
import { setModel, setModelList, setModelVersions } from "@/controller/model/modelSlice";
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
            let req = await fetch(`${SEARCH_MODEL_ENDPOINT}?max_results=1000&filter=tags.wallet_address LIKE '${wallet.accounts[0].address}'`, {
                method: "GET"
            })

            let res = await req.json();
            console.log(res);
            dispatch(setModelList(res.registered_models));
        } catch (e) {
            console.log(e);
        }

    }

    const getRegisteredModelsByName = async (name: string) => {
        try {
            if (!wallet?.accounts?.length) return;
            let req = await fetch(`${GET_REGISTERED_MODEL_ENDPOINT}?name=${name}`, {
                method: "GET"
            })

            let res = await req.json();
            dispatch(setModel(res.registered_model));
        } catch (e) {
            console.log(e);
        }
    }

    const getModelVersionsByName = async (name: string) => {
        try {
            if (!wallet?.accounts?.length) return;
            let req = await fetch(`${SEARCH_MODEL_VERSIONS_ENDPOINT}?filter=name LIKE '${name}'`, {
                method: "GET"
            })

            let res = await req.json();
            dispatch(setModelVersions(res.model_versions));
        } catch (e) {
            console.log(e);
        }
    }

    const createRegisteredModel = async (values: FormData) => {
        try {
            if (!wallet?.accounts?.length) return;
            dispatch(updateActionStatus({ actionName: actionNames.createRegisteredModelAction, value: true }))
            if (values["new_or_update"] === "1") {
                let req = await fetch(`${CREATE_REGISTERED_MODEL_ENDPOINT}`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        name: `${values["name"]}`,
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
            } else {
                updateModelVersion(values["registered_model_name"]);
                let messages = updateRegisteredModelMessage(values["registered_model_name"]);
                openNotification(messages.title, messages.success, MESSAGE_TYPE.SUCCESS);
            }

        } catch (e) {
            console.log(e);
            let messages = createRegisteredModelMessage();
            openNotification(messages.title, messages.fail, MESSAGE_TYPE.ERROR);
        }
        dispatch(updateActionStatus({ actionName: actionNames.createRegisteredModelAction, value: false }))
    }

    const updateModelVersion = async (registeredModelName: string) => {
        let loggedModelString = run.data.tags.filter(tag => tag.key === "mlflow.log-model.history")[0].value;
        let loggedModelJson = JSON.parse(loggedModelString);
        let correctRun = loggedModelJson.filter(model => model.run_id === run.info.run_id)[0];
        let req = await fetch(`${CREATE_MODEL_VERSION_ENDPOINT}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                name: registeredModelName,
                run_id: `${run.info.run_id}`,
                source: `${run.info.artifact_uri}/${correctRun.artifact_path}`,
                // run_link: ``,
                // description: `${values["description"]}`,
                // tags: [
                //     { key: "wallet_address", value: `${wallet.accounts[0].address}` },
                // ]
            })
        })
        await req.json();
    }

    const getDownloadURIForModelVersionArtifacts = async (modelName: string, version: string) => {
        try {
            let req = await fetch(`${GET_DOWNLOAD_URI_FOR_MODEL_VERSION_ARTIFACTS_ENDPOINT}?name=${modelName}&version=${version}`, {
                method: "GET"
            })

            let res = await req.json();
            console.log(res.artifact_uri);
        } catch (e) {
            console.log(e);
        }
    }


    const downloadDockerFile = async (modelName: string, version: string) => {
        try {
            dispatch(updateActionStatus({actionName: actionNames.downloadDockerFilesAction, value: true}));
            let outputDirectory = modelName.replaceAll(" ", "_");
            let response = await fetch(`${DOWNLOADER_ENDPOINT}`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    path: `../${outputDirectory}_v${version}`,
                    need_zip: true,
                    output_file: `../${outputDirectory}_v${version}.zip`
                })
            })

            let blob = await response.blob();
            let href = window.URL.createObjectURL(blob);
            const a = Object.assign(document.createElement("a"), {
                href,
                style: "display:none",
                download: `${outputDirectory}_v${version}.zip`,
            });
            document.body.appendChild(a);
            a.click();
            URL.revokeObjectURL(href);
            a.remove();
        } catch (e) {
            console.log(e);
        }
        dispatch(updateActionStatus({actionName: actionNames.downloadDockerFilesAction, value: false}));
    }


    // Update registered model version here

    return { searchRegisteredModels, getModelVersionsByName, getRegisteredModelsByName, createRegisteredModel, getDownloadURIForModelVersionArtifacts, downloadDockerFile }
}