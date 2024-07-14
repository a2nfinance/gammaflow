import connect from '@/database/connect';
import { NextApiRequest, NextApiResponse } from 'next';
import Pipeline from "@/database/models/pipeline";

const handler = async (req: NextApiRequest, res: NextApiResponse) => {
    if (req.method === 'POST') {
        const {
            input,
            sequence_services
        } = req.body;
        try {
            let firstInput = input;
            let output = null;
            for (let i = 0; i < sequence_services.length; i++) {
                let bodyStructure = sequence_services[i].body;
                if (i > 0) firstInput = JSON.parse(bodyStructure.replace(":inputdata", output));;
                let serviceEndpoint = sequence_services[i].endpoint;
                let req = await fetch(`${serviceEndpoint}`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify(firstInput)
                })
                let res = await req.json();
                output = res.predictions;
            }
            return res.status(200).send(output);
        } catch (error) {
            console.log(error)
            return res.status(500).send(error.message);
        }
    } else {
        res.status(422).send('req_method_not_supported');
    }
};

export default connect(handler);

export const config = {
    api: {
        bodyParser: {
            sizeLimit: '50mb',
        },
    },
}