import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export type RunDetailState = {
    run: any,
}


const initialRunDetailState: RunDetailState = {
    run: {}
}

export const runSlice = createSlice({
    name: 'run',
    initialState: initialRunDetailState,
    reducers: {
        setRun:  (state: RunDetailState, action: PayloadAction<any>) => {
            state.run = action.payload
        },
    }
})
export const { setRun } = runSlice.actions;
export default runSlice.reducer;