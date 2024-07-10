import { GET_ARTIFACTS_LIST, GET_RUN } from "@/configs";
import { useAppDispatch } from "@/controller/hooks";
import { setRootFolder, setRun } from "@/controller/run/runSlice";

export const useRuns = () => {
    const dispatch = useAppDispatch();
    const getRun = async (id: string) => {
        try {
            let req = await fetch(`${GET_RUN}?run_id=${id}`, {
                method: "GET"
            })

            let res = await req.json();
            console.log("Run", res);
            dispatch(setRun(res));
        } catch (e) {
            console.log(e);
        }

    }

    const getArtifactsList = async (run_id: string) => {
        try {
            let req = await fetch(`${GET_ARTIFACTS_LIST}?run_id=${run_id}`, {
                method: "GET"
            })

            let res = await req.json();
            let parentFolder = res.files[0].path;
            // let getChildsReq = await fetch(`${GET_ARTIFACTS_LIST}?run_id=${run_id}&path=${parentFolder}/metadata`, {
            //     method: "GET"
            // })
            // let childElms = await getChildsReq.json();
            // console.log(childElms);
            dispatch(setRootFolder(parentFolder));
        } catch (e) {
            console.log(e);
        }

    }

    return { getRun, getArtifactsList }
}