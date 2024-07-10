import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export type RunDetailState = {
    run: any,
    rootFolder: string
}


const initialRunDetailState: RunDetailState = {
    run: {},
    rootFolder: ""
}

export const runSlice = createSlice({
    name: 'run',
    initialState: initialRunDetailState,
    reducers: {
        setRun:  (state: RunDetailState, action: PayloadAction<any>) => {
            state.run = action.payload
        },
        setRootFolder: (state: RunDetailState, action: PayloadAction<any>) => {
            state.rootFolder = action.payload
        },
    }
})
export const { setRun, setRootFolder } = runSlice.actions;
export default runSlice.reducer;