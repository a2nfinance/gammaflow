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
        return `git clone ${gitRepo} ${toFolder};`
    } else {
        let firstIndex = gitRepo.indexOf("github");
        let part1 = gitRepo.slice(0, firstIndex);
        let part2 = gitRepo.slice(firstIndex, gitRepo.length);
        return `git clone ${part1}${userName}:${password}@${part2} ${toFolder};`
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
    if (isClone) {
        let toFolder = getToFolder(gitRepo);
        return `cd ${toFolder};python ${scriptPath}`;
    } else {
        return `python ${scriptPath}`;
    }
}

export const installDependenciesCommand = (values: FormData) => {
    let githubRepo = values["github_repo"];
    let toFolder = getToFolder(githubRepo);
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
        installCommand += `pip install -r ${toFolder}/requirements.txt;`;
    }
    return installCommand;
}
