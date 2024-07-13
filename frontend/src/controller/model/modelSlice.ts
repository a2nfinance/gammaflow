import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export type ModelState = {
    model: any,
    modelList: any[],
    modelVersions: any[],
    selectedVersion: string,
}


const initialModelState: ModelState = {
    model: {},
    modelVersions: [],
    modelList: [],
    selectedVersion: "1"
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
        setSelectedVersion: (state: ModelState, action: PayloadAction<string>) => {
            state.selectedVersion = action.payload
        },
    }
})
export const { setModel, setModelList, setModelVersions,setSelectedVersion } = modelSlice.actions;
export default modelSlice.reducer;