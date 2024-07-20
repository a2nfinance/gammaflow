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
![System Design](/frontend/public/docs/SystemDesign.jpg)

GammaFlow is designed with three main components: the tracking server, the dApp, and Theta cloud nodes.
- The tracking server runs on GCP, integrated with MLFlow and Docker API.
- The GammaFlow dApp functions as a control panel, connecting and managing all components.
- Theta cloud nodes are used for training, testing, and deploying AI models as inference services.
- All components in GammaFlow communicate via WebSocket and REST API.
## Technology

| Component | Techstack |
| -------- | ------- |
|Frontend (GammaFlow dApp)|NextJS, Web3-onboard, Ant Design, Websocket client, Redux.|
|Node_connector|Docker, Python, Websocket|
|Tracking_server_connector|Flask, Rest API, Websocket|
|MLFlow_cors (enable cors for tracking server)|MLFlow, Python|
|Text_to_video|Flask, Torch, FFMPEG, Scikit-learn, Skvideo, ImageIO|

## Installation
#### GammaFlow dApp

You need to setup the .env file first.

| Environment variable | Required
| -------- | ------- |
|NEXT_PUBLIC_MLFLOW_TRACKING_SERVER|✅ |
|NEXT_PUBLIC_THETA_RPC|✅|
|NEXT_PUBLIC_WALLET_CONNECT_PROJECT_ID|✅|
|NEXT_PUBLIC_SUPPORT_EMAIL|✅|
|NEXT_PUBLIC_APP_URL|✅|
|NEXT_PUBLIC_SERVER_DOWNLOADER|✅|
|NEXT_PUBLIC_SERVER_COMMANDER|✅|

Commands:

- ```cd frontend```
- ```npm i```
- ```npm run dev``` for developer mode
- ```npm run build; npm run start``` for production mode

#### Theta nodes for remote training & experimenting.
Use this docker image to create template and deployment:

[Theta node conenctor docker image](https://hub.docker.com/r/a2nfinance/theta-node-connector)

## Testing

You can test GammaFlow by using the dApp and executing your own Python scripts. If you want to use some available scripts, you can see [GammaFlow's sample scripts](https://github.com/a2nfinance/gammaflow_training_script) for quick testing.


