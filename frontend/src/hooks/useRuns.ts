import { GET_ARTIFACTS_LIST, GET_RUN } from "@/configs";
import { useAppDispatch } from "@/controller/hooks";
import { setRun, setTree } from "@/controller/run/runSlice";


export const useRuns = () => {
    const dispatch = useAppDispatch();
    const getRun = async (id: string) => {
        try {
            let req = await fetch(`${GET_RUN}?run_id=${id}`, {
                method: "GET"
            })

            let res = await req.json();
            dispatch(setRun(res.run));
        } catch (e) {
            console.log(e);
        }

    }

    const getArtifactsList = async (run_id: string) => {
        try {
            let tree: any = {};
            tree = await getFolderTree(run_id, "", tree);
            console.log(tree);
            dispatch(setTree(tree));
        } catch (e) {
            console.log(e);
        }

    }

    const getFolderTree = async (runId: string, dir = "", tree: any = {}, key = "0-0") => {
        try {
            let req = await fetch(`${GET_ARTIFACTS_LIST}?run_id=${runId}&path=${dir}`, {
                method: "GET"
            })

            let res = await req.json();
            let items = res.files;
            for (let i = 0; i < items.length; i++) {
                let item = items[i];
                const itemPath: string = item.path;
                const pathName = itemPath.slice(itemPath.lastIndexOf("/") + 1, itemPath.length);
                const isDir = item.is_dir;
                // let currentDepth = tree.length;
                if (isDir) {
                    let isExistChildrens = tree.children ? true : false;
                    isExistChildrens ? tree.children.push({ title: pathName, isDir: true, key: `${key}-${i}`, children: [] }) : (tree = { title: pathName, isDir: true, key: key, children: [] });
                    await getFolderTree(runId, itemPath, isExistChildrens ? tree.children[tree.children.length - 1] : tree, isExistChildrens ? `${key}-${i}` : key);
                } else {
                    tree.children.push({
                        title: pathName,
                        key: `${key}-${i}`,
                        path: itemPath,
                        isDir: false
                    })
                }
            }
            return tree;
        } catch (e) {
            console.log(e);
        }
    }

    return { getRun, getArtifactsList, getFolderTree }
}