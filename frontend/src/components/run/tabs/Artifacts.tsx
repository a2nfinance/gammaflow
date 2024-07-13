import { useRuns } from "@/hooks/useRuns"
import { Button, Col, Input, Row } from "antd"
import { useRouter } from "next/router"
import { useEffect } from "react"
import { FileTree } from "./FileTree"
import { useAppSelector } from "@/controller/hooks"
export const Artifacts = () => {
    const {fileContent} = useAppSelector(state => state.run)
    const router = useRouter()
    const {getArtifactsList, getFolderTree} = useRuns();
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
                <Input.TextArea style={{height: "100%"}} contentEditable={false} value={fileContent} />
            </Col>
        </Row>
    )
}