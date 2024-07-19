## Introduction
GammaFlow: A tool for AI model training, testing, and deployment on Theta Edge Cloud nodes.
## Demo information
- [GammaFlow dApp](https://gammaflow.a2n.finance)
- [Video demo]()
- [Theta node conenctor docker image](https://hub.docker.com/r/a2nfinance/theta-node-connector)
- [Audio to text docker image](https://hub.docker.com/r/a2nfinance/audio_to_text)
- [Text to video docker image](https://hub.docker.com/r/a2nfinance/text_to_video)

For more detailed information on product features, you can refer to [our project description on DevPost.](https://devpost.com/software/gammaflow)
## System design
| ![System Design](/frontend/public/docs/SystemDesign.jpg) | 
|:--:| 
|GammaFlow is designed with three main components: the tracking server, the dApp, and Theta cloud nodes.|
|- The tracking server runs on GCP, integrated with MLFlow and Docker API.|
|- The GammaFlow dApp functions as a control panel, connecting and managing all components.|
|- Theta cloud nodes are used for training, testing, and deploying AI models as inference services.|
|- All components in GammaFlow communicate via WebSocket and REST API.|
## Technology
| Component | Techstack |
| -------- | ------- |
|Frontend (GammaFlow dApp)|NextJS, Web3-onboard, Ant Design, Websocket client, Redux.|
|Node_connector|Docker, Python, Websocket|
|Tracking_server_connector|Flask, Rest API, Websocket|
|MLFlow_cors (enable cors for tracking server)|MLFlow, Python|
|Text_to_video|Flask, Torch, FFMPEG, Scikit-learn, Skvideo, ImageIO|
## Installation

### Theta nodes for remote training & experimenting.

### Frontend

### Text to video testing

