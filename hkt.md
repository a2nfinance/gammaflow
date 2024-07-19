## Inspiration
Two months ago, our team decided to join this hackathon and researched Theta Network community tools and dApps. We found Theta Edge Cloud to be an excellent platform for AI products, but we identified several issues:

- There are no tools specifically for AI model training on Theta cloud nodes.
- There is no dApp that combines Theta network wallet addresses with AI model versioning and experiment management.
- There is a lack of platforms supporting the entire workflow from AI model experimentation to production deployment on the Theta Edge Cloud.

To address these issues, we developed the GammaFlow project.

## Demo Information
- [GammaFlow dApp](https://gammaflow.a2n.finance)
- [Github](https://github.com/a2nfinance/gammaflow)
- [Video demo]()

## What it does


| Features | Descriptions |
| -------- | ------- |
| Experiment management | Store experiment data based on developers' Theta account address, Theta Edge Cloud node address, and general information. |
| AI model versioning | Each time an experiment runs, developers can create a new version or update an existing one after reviewing logs, metrics, and artifacts. |
| Centralized logs for running scripts | When running a script, the tracking server stores parameters, metrics, and other logs in the database. Developers can use this data for comparison and evaluation. |
| Build Docker images from AI model versions for inference services | Developers can generate, download, build, and push Docker images to their Docker Hub without needing a local environment installation. |
| Model deployment management | Manage models, versions, and deployment information more easily. |
| Sequential inference services playground for testing | The GammaFlow playground allows developers to test one or more inference services with custom input data structures. |
| Remote shell command tool integrated with Theta Cloud nodes | Allows developers to execute commands on remote machines from the web UI. This feature is used for running scripts directly. |



## How we built it
### System Design

| ![System Design](https://gammaflow.a2n.finance/docs/SystemDesign.jpg) | 
|:--:| 
|GammaFlow is designed with three main components: the tracking server, the dApp, and Theta cloud nodes.|
|- The tracking server runs on GCP, integrated with MLFlow and Docker API.|
|- The GammaFlow dApp functions as a control panel, connecting and managing all components.|
|- Theta cloud nodes are used for training, testing, and deploying AI models as inference services.|
|- All components in GammaFlow communicate via WebSocket and REST API.|

### Workflow

| ![](https://gammaflow.a2n.finance/docs/workflow.jpg)  |
|:--:|
|This is a basic workflow for using GammaFlow with Theta cloud nodes to train and test AI models. For detailed instructions, please refer to our video demo.|

## Challenges we ran into

**Model Tracking When Executing Scripts on Theta Nodes:** When running training or testing scripts on remote machines, developers need to track metrics, logs, artifacts, and versions for comparison and evaluation. While there are some good solutions available, no existing product distinguishes tracking data for Web3 users. To address this, we implemented a solution using wallet addresses in backend data integrated with MLFlow.

**Lack of API for Creating Templates and Deployment:** In the initial phase of development, we aimed to create a complete workflow allowing users to create Theta Edge Cloud templates and deployments based on Docker images of AI model versions via API and signed transactions. This would significantly improve UX. However, since Theta Cloud does not currently support such an API, we had to break tasks into smaller steps. Users can generate, download, build, and push Docker images, but must create templates and deployments manually.

**Text-to-Video Model Training:** Besides GammaFlow, we developed a custom text-to-video AI model to showcase how GammaFlow works with Theta nodes. However, many open-source implementations based on published papers lack clear and workable instructions, often hiding crucial data or code blocks. Our team spent over a month full-time customizing and making the text-to-video model functional. Although it is not production-ready and requires significant improvements, its functionality is demonstrated in our video demo.

**Single Port Mapping Limitation:** Theta cloud nodes currently support only one mapped port, posing a challenge for deploying multiple services on a testing node, such as remote training and inference services after training scripts are executed. While it is possible with a single port, it requires more complex configuration. We hope Theta Edge Cloud will support multiple port mappings in the near future.

## Accomplishments We're Proud Of

GammaFlow is one of the first projects in the Theta ecosystem that enables engineers to conduct AI experiments for training and testing directly on Theta nodes.

## What We Learned

During the hackathon, we understood the importance of MLOps and its role in AI model development. We also learned how AI can integrate with decentralized infrastructure and cater to Web3 users. Our knowledge improved with each challenge we solved. We extend special thanks to the engineers on the hackathon support channel—without their guidance, we could not have completed GammaFlow.

## What's Next for GammaFlow

AI in the context of DePIN is a vast area to explore. GammaFlow is in its initial phase of development, and many features related to security, training pipelines, testing, and deployment need to be improved. We hope GammaFlow will contribute to the growth of the Theta ecosystem in the future.

