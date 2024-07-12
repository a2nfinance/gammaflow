import { Detail } from "@/components/experiments/Detail";
import { useExperiments } from "@/hooks/useExperiments";
import { useRouter } from "next/router";
import { useEffect } from "react";

export default function ID() {
    const { searchRunByExperimentId, getExperimentById } = useExperiments();
    const router = useRouter();

    useEffect(() => {
        if (router.query?.id) {
            searchRunByExperimentId(router.query?.id?.toString());
            getExperimentById(router.query?.id?.toString())
        }
    }, [router.query?.id]);
    return (
        <div style={{ maxWidth: 1440, minWidth: 1024, margin: "auto" }}>
            <Detail />
        </div>
    )
}