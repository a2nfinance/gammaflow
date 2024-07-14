import { useAppDispatch, useAppSelector } from "@/controller/hooks";
import { actionNames, updateActionStatus } from "@/controller/process/processSlice";

export const useAIServices = () => {
    const dispatch = useAppDispatch();
    const callAIEnpoint = async (input: any, values: FormData) => {
        try {
            dispatch(updateActionStatus({ actionName: actionNames.callInferenceServicesAction, value: true }));
            let req = await fetch(`/api/ai-service`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    input: input,
                    sequence_services: values["sequence_services"]
                })
            })
            let res = await req.json();
            if (values["output_type"] === "1") {
                let output = document.getElementById("playground-output");
                if (output) {
                    output.innerText = res;
                }

            }
        } catch (e) {
            console.log(e);
        }
        dispatch(updateActionStatus({ actionName: actionNames.callInferenceServicesAction, value: false }))
    };


    return { callAIEnpoint };
};