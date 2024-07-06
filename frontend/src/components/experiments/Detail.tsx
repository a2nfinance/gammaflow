import { useExperiments } from "@/hooks/useExperiments"
import { useEffect } from "react"

export const Detail = () => {
    const {getExperimentById} = useExperiments();
    useEffect(() => {
        getExperimentById("677616616261565730")
    }, [])
    return (
        <>Detail</>
    )
}