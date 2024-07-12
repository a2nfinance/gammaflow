import { Detail } from "@/components/model/Detail";
import { useModels } from "@/hooks/useModels";
import { useRouter } from "next/router";
import { useEffect } from "react";

export default function ModelDetail() {
    const { getRegisteredModelsByName, getModelVersionsByName } = useModels();
    const router = useRouter();

    useEffect(() => {
        if (router.query?.name) {
            getRegisteredModelsByName(router.query?.name?.toString());
            getModelVersionsByName(router.query?.name?.toString())
        }
    }, [router.query?.name]);
    return (
        <div style={{ maxWidth: 1440, minWidth: 1024, margin: "auto" }}>
            <Detail />
        </div>
    )
}