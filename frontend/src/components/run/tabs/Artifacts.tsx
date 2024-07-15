import { useAppSelector } from "@/controller/hooks"
import { useRuns } from "@/hooks/useRuns"
import { Col, Input, Row, Spin } from "antd"
import { useRouter } from "next/router"
import { useEffect } from "react"
import { FileTree } from "./FileTree"
export const Artifacts = () => {
    const { fileContent } = useAppSelector(state => state.run)
    const { getFileContentAction } = useAppSelector(state => state.process);
    const router = useRouter()
    const { getArtifactsList } = useRuns();
    useEffect(() => {
        if (router?.query?.id) {
            getArtifactsList(router?.query?.id.toString());
        }
    }, [router?.query?.id])
    return (
        <Spin spinning={getFileContentAction}>
            <Row gutter={6}>

                <Col span={6}>

                    <FileTree />

                </Col>
                <Col span={18}>

                    <Input.TextArea readOnly style={{ height: "100%", minHeight: "300px" }} value={fileContent} />
                </Col>

            </Row>
        </Spin>
    )
}