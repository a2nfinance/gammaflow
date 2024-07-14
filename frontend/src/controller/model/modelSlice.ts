import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export type ModelState = {
    model: any,
    modelList: any[],
    modelVersions: any[],
    selectedVersion: string,
    deployedVersions: any
}


const initialModelState: ModelState = {
    model: {},
    modelVersions: [],
    modelList: [],
    selectedVersion: "1",
    deployedVersions: []
}

export const modelSlice = createSlice({
    name: 'model',
    initialState: initialModelState,
    reducers: {
        setModel: (state: ModelState, action: PayloadAction<any>) => {
            state.model = action.payload
        },
        setModelList: (state: ModelState, action: PayloadAction<any[]>) => {
            state.modelList = action.payload
        },
        setModelVersions: (state: ModelState, action: PayloadAction<any[]>) => {
            state.modelVersions = action.payload
        },
        setDeployedVersions: (state: ModelState, action: PayloadAction<any[]>) => {
            state.deployedVersions = action.payload
        },
        setSelectedVersion: (state: ModelState, action: PayloadAction<string>) => {
            state.selectedVersion = action.payload
        },
    }
})
export const { setModel, setModelList, setModelVersions, setSelectedVersion, setDeployedVersions } = modelSlice.actions;
export default modelSlice.reducer;