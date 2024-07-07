import { useExperiments } from "@/hooks/useExperiments"
import { useRouter } from "next/router";
import { useEffect } from "react"

export const Detail = () => {
    const {getExperimentById} = useExperiments();
    const router = useRouter();

    useEffect(() => {
        if (router.query?.id) {
            console.log(router.query?.id);
        }
    }, []);
  
    return (
        <>Detail</>
    )
}