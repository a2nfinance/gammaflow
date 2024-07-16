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
                    sequence_services: values["sequence_services"],
                    output_type: values["output_type"]
                })
            })
           
            
            if (values["output_type"] === "1") {
                let res = await req.json();
                let output = document.getElementById("playground-output");
                if (output) {
                    output.innerText = res;
                }

            } else if (values["output_type"] === "3") {
                let res = await req.arrayBuffer();
                const blob = new Blob([res], {type: "video/mp4"});
                const video = document.createElement('video'); 
                video.src = URL.createObjectURL(blob);
                video.load();
                video.autoplay = true;
                video.loop = true;
                video.setAttribute("width", "100%");
                video.canPlayType("video/mp4");
                video.onloadeddata = function() {
                    video.play();
                }
                let output = document.getElementById("playground-output");
               
                if (output) {
                    output.innerHTML = "";
                    output.appendChild(video);

                }
            }
        } catch (e) {
            console.log(e);
        }
        dispatch(updateActionStatus({ actionName: actionNames.callInferenceServicesAction, value: false }))
    };


    return { callAIEnpoint };
};