import { GET_ARTIFACTS_LIST } from "@/configs";
import { useAppDispatch } from "@/controller/hooks";
import { setRun } from "@/controller/run/runSlice";

export const useArtifact = () => {
    const dispatch = useAppDispatch();
    const listArtifacts = async (runId: string) => {
        try {
            let req = await fetch(`${GET_ARTIFACTS_LIST}?run_id=${runId}`, {
                method: "GET"
            })

            let res = await req.json();
            dispatch(setRun(res.run));
        } catch (e) {
            console.log(e);
        }

    }

    return { listArtifacts }
}