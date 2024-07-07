import { useExperiments } from "@/hooks/useExperiments"
import { List } from "antd"
import { useEffect } from "react"
import { Item } from "./Item"
import { useAppSelector } from "@/controller/hooks"
import { useConnectWallet } from "@web3-onboard/react"

export const ExperimentList = () => {
    const [{wallet}] = useConnectWallet();
    const {getExperimentsByCreator} = useExperiments();
    const {experiments} = useAppSelector(state => state.experiment);
    useEffect(() => {
        getExperimentsByCreator();
    }, [wallet?.accounts?.length])
    return (
        <List
        grid={{
            gutter: 12,
            column: 3
        }}
        size="large"
        pagination={{
            onChange: (page) => {
                console.log(page);
            },
            pageSize: 6,
            align: "center",
        }}
        dataSource={experiments}
        renderItem={(item, index) => (
            <Item index={index} experiment={item} />
        )}
    />
    )
}