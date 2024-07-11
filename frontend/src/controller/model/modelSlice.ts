import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export type ModelState = {
    model: any,
    modelList: any[]
}


const initialModelState: ModelState = {
    model: {},
    modelList: []
}

export const modelSlice = createSlice({
    name: 'model',
    initialState: initialModelState,
    reducers: {
        setRun:  (state: ModelState, action: PayloadAction<any>) => {
            state.model = action.payload
        },
        setModelList: (state: ModelState, action: PayloadAction<any[]>) => {
            state.modelList = action.payload
        },
    }
})
export const { setRun, setModelList } = modelSlice.actions;
export default modelSlice.reducer;