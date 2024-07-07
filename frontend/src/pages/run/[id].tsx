import { Detail } from "@/components/run/Detail";
import { useRuns } from "@/hooks/useRuns";
import { useRouter } from "next/router";
import { useEffect } from "react";

export default function RunId() {
    const { getRun } = useRuns();
    const router = useRouter();

    useEffect(() => {
        if (router.query?.id) {
            getRun(router.query?.id?.toString());
        }
    }, [router.query?.id]);
    return (
        <div style={{ maxWidth: "1440", minWidth: 900, margin: "auto" }}>
            <Detail />
        </div>
    )
}