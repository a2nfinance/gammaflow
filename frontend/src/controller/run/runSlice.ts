import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export type RunDetailState = {
    run: any,
    tree: any,
    fileContent: string,
}


const initialRunDetailState: RunDetailState = {
    run: {
        info: {},
        inputs: {},
        data: {
            params: [],
            metrics: [],
            tags: []
        }
    },
    tree: {},
    fileContent: ""
}

export const runSlice = createSlice({
    name: 'run',
    initialState: initialRunDetailState,
    reducers: {
        setRun:  (state: RunDetailState, action: PayloadAction<any>) => {
            state.run = action.payload
        },
        setTree: (state: RunDetailState, action: PayloadAction<any>) => {
            state.tree = action.payload
        },
        setFileContent: (state: RunDetailState, action: PayloadAction<any>) => {
            state.fileContent = action.payload
        },
    }
})
export const { setRun, setTree, setFileContent } = runSlice.actions;
export default runSlice.reducer;