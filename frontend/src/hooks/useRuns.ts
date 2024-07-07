import { GET_RUN } from "@/configs"

export const useExperiments = () => {

    const getRun = async (id: string) => {
        let req = await fetch(`${GET_RUN}?run_id=${id}`, {
            method: "GET"
        })

        let res = await req.json();
        console.log("Run");
        return res;
    }

    return { getRun }
}