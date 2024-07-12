import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export type ModelState = {
    model: any,
    modelList: any[],
    modelVersions: any[],
}


const initialModelState: ModelState = {
    model: {},
    modelVersions: [],
    modelList: []
}

export const modelSlice = createSlice({
    name: 'model',
    initialState: initialModelState,
    reducers: {
        setModel:  (state: ModelState, action: PayloadAction<any>) => {
            state.model = action.payload
        },
        setModelList: (state: ModelState, action: PayloadAction<any[]>) => {
            state.modelList = action.payload
        },
        setModelVersions: (state: ModelState, action: PayloadAction<any[]>) => {
            state.modelVersions = action.payload
        },
    }
})
export const { setModel, setModelList, setModelVersions } = modelSlice.actions;
export default modelSlice.reducer;