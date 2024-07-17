import { NextApiRequest, NextApiResponse } from 'next';

const handler = async (req: NextApiRequest, res: NextApiResponse) => {
    if (req.method === 'POST') {
        const {
            input,
            sequence_services,
            output_type
        } = req.body;
        try {
            let firstInput = input;
            let output:any = null;
            for (let i = 0; i < sequence_services.length; i++) {
                let bodyStructure = sequence_services[i].body;
                if (i > 0) firstInput = JSON.parse(bodyStructure.replace(":inputdata", output));;
                let serviceEndpoint = sequence_services[i].endpoint;
                let request = await fetch(`${serviceEndpoint}`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify(firstInput)
                })
                
                if (output_type === "3") {
                    output = await request.blob();
                    res.setHeader('Content-Type', output.type);
                    res.setHeader('Content-Length', output.size);
                    let arrayBuffer = await output.arrayBuffer();
                    const buffer = Buffer.from(arrayBuffer);
                    return res.status(200).send(buffer);
                } else {
                    let res = await request.json();
                    output = res.predictions;
                    return res.status(200).send(output);
                }
               
            }
           
          
        } catch (error) {
            console.log(error)
            return res.status(500).send(error.message);
        }
    } else {
        res.status(422).send('req_method_not_supported');
    }
};

export default handler;

export const config = {
    api: {
        bodyParser: {
            sizeLimit: '50mb',
        },
    },
}