import { useAppSelector } from "@/controller/hooks"
import { useRuns } from "@/hooks/useRuns"
import { Button, Col, Row } from "antd"
import { useRouter } from "next/router"
import { useEffect } from "react"
import { FileTree } from "./FileTree"
export const Artifacts = () => {
    const router = useRouter()
    const {getArtifactsList} = useRuns();
    useEffect(() => {
        if (router?.query?.id) {
            getArtifactsList(router?.query?.id.toString());
        }
    }, [router?.query?.id])
    return (
        <Row gutter={6}>
            <Col span={6}>
            
            <FileTree />
            
            </Col>
            <Col span={18}>
                <Button size="large" type="primary">Deploy</Button>
            </Col>
        </Row>
    )
}