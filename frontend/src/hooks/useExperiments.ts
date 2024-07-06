import { CREATE_EXPERIMENT_ENDPOINT, GET_EXPERIMENT_ENDPOINT } from "@/configs";

export const useExperiments = () => {
    const createExperiment = async () => {
        try {

            let createReq = await fetch(`${CREATE_EXPERIMENT_ENDPOINT}`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    name: "Test",
                    artifact_location: "artifacts",
                    tags: [
                        { key: "training_script", value: "main.py" }
                    ]
                })
            })
            let res = await createReq.json();
            console.log(res);
        } catch (e) {
            console.log(e);
        }




    };

    const getExperimentById = async (id: string) => {
        try {

            let req = await fetch(`${GET_EXPERIMENT_ENDPOINT}?experiment_id=${id}`, {
                method: "GET"
            })
            let res = await req.json();
            console.log(res);
        } catch (e) {
            console.log(e);
        }
    };
    return { createExperiment, getExperimentById };
};