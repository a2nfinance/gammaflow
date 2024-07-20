import { EXPORT_COMMAND, TRACKING_SERVER_MLFLOW_PATH } from "@/configs";

export const  getToFolder = (gitRepo: string) => {
    let parts: string[] = gitRepo.split("/");
    let toFolder = parts[parts.length - 1].replace(".git", "");
    return toFolder;
}
export const cloneGitCommand = (
    gitRepo: string,
    isPrivate: boolean,
    userName?: string,
    password?: string
) => {
    let toFolder = getToFolder(gitRepo);
    if (!isPrivate) {
        return `git clone ${gitRepo} ${toFolder};cd ${toFolder};`
    } else {
        let firstIndex = gitRepo.indexOf("github");
        let part1 = gitRepo.slice(0, firstIndex);
        let part2 = gitRepo.slice(firstIndex, gitRepo.length);
        return `git clone ${part1}${userName}:${password}@${part2} ${toFolder};cd ${toFolder};`
    }
}

export const pullGitCommand = (
    gitRepo: string,
    isPrivate: boolean,
    userName?: string,
    password?: string
) => {
    let toFolder = getToFolder(gitRepo);
    if (!isPrivate) {
        return `cd ${toFolder}; git pull;`
    } else {
        let firstIndex = gitRepo.indexOf("github");
        let part1 = gitRepo.slice(0, firstIndex);
        let part2 = gitRepo.slice(firstIndex, gitRepo.length);
        return `cd ${toFolder}; git pull ${part1}${userName}:${password}@${part2};`
    }

}

export const runScriptCommand = (gitRepo: string, isClone: boolean, scriptPath: string) => {
    return `python ${scriptPath}`;
}

export const installDependenciesCommand = (values: FormData) => {
    let systemDependencies = values["system_dependencies"];
    let pythonDependencies = values["system_dependencies"];
    let useRequirements = values["use_requirements"];
   
    let installCommand = "";
    if (systemDependencies) {
        installCommand += "apt-get install " + systemDependencies.split(",").join(" ") + ";";
    }
    if (pythonDependencies) {
        installCommand += "pip install " + pythonDependencies.split(",").join(" ") + ";";
    }
    if (useRequirements === "1") {
        installCommand += `pip install -r requirements.txt;`;
    }
    return installCommand;
}

export const generatedZipCommand = (model: any, version: string) => {
    let outputDirectory = model.name.replaceAll(" ", "_");
    let generatedCommand = `${EXPORT_COMMAND};${TRACKING_SERVER_MLFLOW_PATH} models generate-dockerfile -m models:/"${model.name}"/${version} --output-directory ${outputDirectory}_v${version} --enable-mlserver`;
    return generatedCommand;
}

export const buildAndPushImageCommand = (model: any, version: string, values: FormData) => {
    let outputDirectory = model.name.replaceAll(" ", "_");
    let cdCommand = `cd ${outputDirectory}_v${version};`;
    let loginCommand = `docker login -u "${values["username"]}" -p "${values["password"]}"" docker.io;`;
    let buildCommand = `docker build --tag '${values["username"]}/${values["repository"]}:${values["version"]}' . --network=host;`;
    let pushCommand = `docker push '${values["username"]}/${values["repository"]}:${values["version"]}';`;
    let finalCommands = cdCommand + loginCommand + buildCommand + pushCommand;
    return finalCommands;
} 