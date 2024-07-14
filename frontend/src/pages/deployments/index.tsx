import { DeployedVersion } from "@/components/deployments/DeployedVersion";
import { DeploymentForm } from "@/components/deployments/Form";
import { Divider } from "antd";

export default function Index() {
    return (
        <div style={{ maxWidth: 1440, minWidth: 1024, margin: "auto" }}>
            <DeploymentForm />
            <Divider />
            <DeployedVersion />
        </div>
    )
}